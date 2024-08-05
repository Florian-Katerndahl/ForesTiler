from argparse import ArgumentParser
from pathlib import Path
import multiprocessing as mp

from tqdm import tqdm
from shapely import STRtree, box
import numpy as np
import rasterio as rio
import geopandas as gpd
import torch
from torch.nn import functional as f

from forestiler.chipIO import raster_worker, vector_worker


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
        "--stride", type=int, required=False, default=1, help="Stride of kernel."
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
        "input", type=Path, help="Directory containing raster files to tile."
    )
    parser.add_argument(
        "out", type=Path, help="Directory where output files should be stored."
    )
    args = parser.parse_args()

    args.out.mkdir(exist_ok=True)

    output_queue = mp.Queue()
    vector_queue = mp.Queue()

    for _ in range(int(mp.cpu_count() * 0.8)):
        mp.Process(target=raster_worker, args=(output_queue,), daemon=True).start()

    mp.Process(
        target=vector_worker,
        args=(vector_queue,),
        name="forestiler_vector",
        daemon=True,
    ).start()

    # order does not seem to make a difference (bounding boxes as STRtree or geometries as STRtree)
    mask_field = args.class_field
    mask_vector = gpd.read_file(args.vector_mask)
    if not args.all_classes:
        mask_vector = mask_vector.loc[mask_vector[mask_field] in args.classes]
    mask_tree = STRtree(mask_vector.geometry)

    PADDING = 0  # padding is disabled because it messes with the coordinates, may be fixed later
    FILE_SUFFIX = ".tif" if args.geo_tiff else ".png"
    single_use = not args.cubed
    coordinate_kernels_missing = True

    for raster_file in tqdm(
        args.input.rglob("*LEVEL2_SEN*BOA.tif"),
        "Scenes",
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

        assert (
            rows == cols
        ), "Input dataset must have the same number of columns and rows"

        raster_tensor = torch.from_numpy(raster_values).double()
        raster_tensor = raster_tensor[None, ...]
        raster_kernels = f.unfold(
            raster_tensor,
            kernel_size=args.kernel_size,
            padding=PADDING,
            stride=args.stride,
        ).permute(0, 2, 1)

        if single_use or coordinate_kernels_missing:
            raw_cell_indices = torch.arange(rows)
            x = (
                raw_cell_indices[None, None, None, :]
                .repeat_interleave(cols, dim=2)
                .double()
            )
            y = (
                raw_cell_indices[None, None, :, None]
                .repeat_interleave(rows, dim=3)
                .double()
            )

            center_coordinates = torch.empty((1, 2, rows, cols), dtype=torch.double)
            center_coordinates[:, 0, :, :] = (y * psy + y_orig) + (psy / 2.0)
            center_coordinates[:, 1, :, :] = (x * psx + x_orig) + (psx / 2.0)

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
            continue
        classes = mask_vector.iloc[query_results[1]][mask_field].tolist()
        basename = str(raster_file.stem)
        output_bboxes_list = bboxes.take(query_results[0]).tolist()

        # This is also slow... maybe put in a queue as well?
        vector_queue.put((output_bboxes_list, classes, raster_crs, args.out, basename))

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
