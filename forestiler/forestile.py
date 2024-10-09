from argparse import ArgumentParser
from pathlib import Path
import multiprocessing as mp
import re
from warnings import warn

from tqdm import tqdm
from shapely import STRtree, box
import numpy as np
import rasterio as rio
import geopandas as gpd
import torch
from torch.nn import functional as f
from psutil import virtual_memory

from forestiler.chipIO import raster_worker, vector_chips

SIZEOF_DOUBLE = 8

def main():
    parser = ArgumentParser(
        description="forestile creates image tiles from large input rasters according to "
                    "a classified mask vector file.",
        epilog="Copyright: Florian Katerndahl <florian@katerndahl.com>",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bar")
    parser.add_argument(
        "--kernel-size",
        type=int,
        required=False,
        default=256,
        help="Kernel size in pixels.",
    )
    parser.add_argument(
        "--stride", type=int, required=False, default=256, help="Stride of kernel."
    )
    parser.add_argument(
        "--vector-mask",
        type=Path,
        required=True,
        help="Path to vector file. Always reads first layer, if driver supports "
             "multi-layer files (e.g. Geopackages).",
    )
    parser.add_argument(
        "--class-field",
        type=str,
        required=False,
        help="Attribute field containing class values.",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Generate image chips for all unique values in class field.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=False,
        default=[],
        help="List of classes to build image tiles for.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        required=False,
        default="*.tif",
        help="Optional glob pattern to filter files in input directory. "
             "May not exist prior to program invocation.",
    )
    parser.add_argument(
        "--cubed",
        action="store_true",
        help="Set this flag if input is a single granule of a datacube",
    )
    parser.add_argument(
        "--geo-tiff",
        action="store_true",
        help="Store image chips as GeoTiffs instead of PNGs.",
    )
    parser.add_argument(
        "--footprint-only",
        action="store_true",
        help="Only output the vector database with granule footprints. Do not save raster tiles."
    )
    parser.add_argument(
        "input", type=Path, help="Directory containing raster files to tile."
    )
    parser.add_argument(
        "out", type=Path, help="Directory where output files should be stored."
    )
    args = parser.parse_args()

    args.out.mkdir(exist_ok=True)

    output_queue = mp.Queue(maxsize=4096)

    rworker = []
    for _ in range(int(mp.cpu_count() * 0.8)):
        r = mp.Process(target=raster_worker, args=(output_queue,), daemon=True)
        rworker.append(r)
        r.start()

    # order does not seem to make a difference (bounding boxes as STRtree or geometries as STRtree)
    mask_field = args.class_field
    mask_vector = gpd.read_file(args.vector_mask)

    assert all(mask_vector.geom_type == "Polygon"), \
        "Mask vector is only allowed to contain `Polygon` geometries"

    if not args.all_classes:
        mask_vector = mask_vector.loc[mask_vector[mask_field].isin(args.classes)]
    mask_tree = STRtree(mask_vector.geometry)

    PADDING = 0  # padding is disabled because it messes with the coordinates, may be fixed later
    FILE_SUFFIX = ".tif" if args.geo_tiff else ".png"
    single_use = not args.cubed
    coordinate_kernels_missing = True
    vector_footprint_written = False

    for raster_file in tqdm(
        args.input.rglob(args.input_glob),
        "Scene(s)",
        unit="file",
        bar_format="{desc}: {n_fmt} [{elapsed} elapsed, {rate_fmt}{postfix}]",
        disable=args.quiet,
    ):
        with rio.open(raster_file) as raster:
            raster_values = raster.read()
            bands, rows, cols = raster_values.shape
            x_orig, y_orig = raster.transform * (0, 0)
            psx, psy = raster.res
            psy *= -1
            raster_crs = raster.crs.to_epsg()

        expected_memory_usage: int = (((rows - args.kernel_size + 1) / args.stride) * ((cols - args.kernel_size + 1) / args.stride) * (args.kernel_size ** 2 * bands * SIZEOF_DOUBLE))

        assert virtual_memory().available > expected_memory_usage, \
            "System does not provide sufficient available memory for creation of tiles. Either decrease input size (e.g. by using gdal_retile) or choose a larger kernel and/or stride."
        
        assert expected_memory_usage > 0, \
            "Kernel can't be bigger than input image"

        assert raster_crs == mask_vector.crs.to_epsg(), \
            "Raster and vector files must be in the same coordinate system"

        raster_tensor = torch.from_numpy(raster_values).double()
        del raster_values
        raster_tensor = raster_tensor[None, ...]
        raster_kernels = f.unfold(
            raster_tensor,
            kernel_size=args.kernel_size,
            padding=PADDING,
            stride=args.stride,
        ).permute(0, 2, 1)

        if single_use or coordinate_kernels_missing:
            # raw_cell_indices = torch.arange(rows)
            x = (
                torch.arange(cols)[None, None, None, :]
                .repeat_interleave(rows, dim=2)
                .double()
            )
            y = (
                torch.arange(rows)[None, None, :, None]
                .repeat_interleave(cols, dim=3)
                .double()
            )

            center_coordinates = torch.empty((1, 2, rows, cols), dtype=torch.double)
            center_coordinates[:, 0, :, :] = (y * psy + y_orig) + (psy / 2.0)
            center_coordinates[:, 1, :, :] = (x * psx + x_orig) + (psx / 2.0)

            # trades code simplcity for efficiency
            bbox_kernels = f.unfold(
                center_coordinates,
                kernel_size=args.kernel_size,
                padding=PADDING,
                stride=args.stride,
            ).permute(0, 2, 1)

            assert (
                raster_kernels.shape[1] == bbox_kernels.shape[1]
            ), "Differing number of kernels for raster dataset and coordinates"

            _, number_of_kernels, values_per_kernel_stack = bbox_kernels.shape
            # dimension at index 1: x coordinates, and then y coordinates 
            #  (bbox_kernels_reshaped[:, 0, :] gives all center y coordinates)
            bbox_kernels_reshaped = bbox_kernels.reshape(
                (number_of_kernels, 2, args.kernel_size**2)
            )

            # The block below assumes north-up images, right?
            coordinate_minima = bbox_kernels_reshaped.min(dim=2).values
            coordinate_maxima = bbox_kernels_reshaped.max(dim=2).values
            coordinate_minima[..., 0] -= abs(psy) / 2.0
            coordinate_maxima[..., 0] += abs(psy) / 2.0
            coordinate_minima[..., 1] -= psx / 2.0
            coordinate_maxima[..., 1] += psx / 2.0

            coordinates = torch.hstack(
                (coordinate_minima[..., (1, 0)], coordinate_maxima[..., (1, 0)])
            )

            bboxes_list = []
            for i in coordinates:
                bboxes_list.append(box(*i.numpy()))
            bboxes = np.array(bboxes_list)
            del bboxes_list
            coordinate_kernels_missing = False

        query_results = mask_tree.query(bboxes, predicate="covered_by")
        if query_results.size == 0:
            # TODO document exit codes
            if args.cubed:
                # input directory is granule of datacube; if no matches found for first tile,
                #  then there won't be any for later tiles.
                exit(2)
            else:
                continue
        classes = list(map(lambda x: re.sub("[\s,._\/]", "-", x), mask_vector.iloc[query_results[1]][mask_field].astype(str).tolist()))
        basename = str(raster_file.stem)
        output_bboxes_list = bboxes.take(query_results[0]).tolist()

        if (args.cubed and not vector_footprint_written) or (not args.cubed):
            vector_footprint_written = True
            vector_chips(output_bboxes_list, classes, raster_crs, args.out, basename)

        if args.footprint_only:
            break

        output_tiles = (
            raster_kernels[:, query_results[0], :]
            .reshape(
                (int(query_results.size / 2), bands, args.kernel_size, args.kernel_size)
            )
            .numpy()
        )
        for tile in range(int(query_results.size / 2)):
            tminx, tminy, tmaxx, tmaxy = output_bboxes_list[tile].bounds
            transform = rio.transform.Affine.translation(
                tminx, tmaxy
            ) * rio.transform.Affine.scale(psx, psy)
            # data, path, as_geotiff, transformation, crs, bands, kernel_size
            output_queue.put(
                (
                    output_tiles[tile, ...],
                    (
                        args.out
                        / f"{basename}_{classes[tile]}_{tminx}_{tminy}_{tmaxx}_{tmaxy}"
                    ).with_suffix(FILE_SUFFIX),
                    args.geo_tiff,
                    transform,
                    raster_crs,
                    bands,
                    args.kernel_size,
                )
            )

    output_queue.close()
    output_queue.join_thread()

    return 0
