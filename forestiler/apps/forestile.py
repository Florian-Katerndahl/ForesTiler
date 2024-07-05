from argparse import ArgumentParser
from pathlib import Path
from threading import Thread
from typing import List, Dict

from tqdm import tqdm
from shapely import STRtree, box
import numpy as np
import rasterio as rio
from PIL import Image

from forestiler.chipIO import write_imgs, vector_chips
from forestiler.mask import create_masks


def main():
    parser = ArgumentParser(
        description="forestile creates image tiles from large input rasters according to a classified mask vector file.",
        epilog="Copyright: Florian Katerndahl <florian@katerndahl.com>",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--pad",
        action="store_false",
        help="Disable padding of input images."
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        required=False,
        default=256,
        help="Kernel size in pixels."
    )
    parser.add_argument(
        "--stride",
        type=int,
        required=False,
        default=1,
        help="Stride of kernel."
    )
    parser.add_argument(
        "--vector-mask",
        type=Path,
        required=True,
        help="Path to vector file. Always reads first layer, if driver supports multi-layerr files (e.g. Geopackages).",
    )
    parser.add_argument(
        "--class-field",
        type=str,
        required=False,
        help="Attribute field containing class values."
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Generate image chips for all unique values in class field."
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=False,
        default=[],
        help="List of classes to build image tiles for."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        required=False,
        default="*.tif",
        help="Optional glob pattern to filter files in input directory.",
    )
    parser.add_argument("--geo-tiff", action="store_true", help="Store image chips as GeoTiffs instead of PNGs.")
    parser.add_argument("input", type=Path, help="Directory containing raster files to tile.")
    parser.add_argument("out", type=Path, help="Directory where output files should be stored.")
    args = parser.parse_args()

    assert (args.kernel_size % 2) == 0, "kernel must have even length"

    raster_files: List[Path] = list(Path(args.input).glob(args.input_glob))

    mask_objects = create_masks(args.vector_mask, args.class_field, args.all_classes, args.classes)

    output_directory: Path = args.out
    if not output_directory.exists():
        output_directory.mkdir()

    bar: tqdm = tqdm(
        desc="Processing raster file",
        total=len(raster_files),
        unit="file(s)",
        disable=args.no_progress,
        leave=True,
    )

    # I don't think this is how one usually calculates kernel widths, but whatever
    kernel_padding: int = int((args.kernel_size - 2) / 2)

    for raster_file in raster_files:
        with rio.open(raster_file) as raster:
            raster_raw = np.pad(
                raster.read(),
                ((0, 0), (kernel_padding, kernel_padding), (kernel_padding, kernel_padding)),
            ).transpose((1, 2, 0))
            rows, cols, _ = raster_raw.shape
            ul_row, ul_col = raster.transform * (0, 0)
            pixel_size = raster.transform[0]
            raster_crs = raster.crs.to_epsg()

            bboxes = []
            classes = []
            T = []
            for row in range(0, rows, args.stride):
                for col in range(0, cols, args.stride):
                    xmin, ymin, xmax, ymax = (
                        col - kernel_padding - 1,  # keep center pixel
                        row - kernel_padding - 1,  # keep center pixel
                        col + kernel_padding + 1,  # keep center pixel
                        row + kernel_padding + 1,  # keep center pixel
                    )
                    bbox: box = box(
                        ymin * pixel_size + ul_row,
                        xmin * -pixel_size + ul_col,
                        ymax * pixel_size + ul_row,
                        xmax * -pixel_size + ul_col,
                    )
                    for mask in mask_objects:
                        if mask["tree"].query(bbox, predicate="covered_by").size > 0:
                            bboxes.append(bbox)
                            classes.append(mask["class"])
                            image_chip = Image.fromarray(
                                raster_raw[
                                    xmin + kernel_padding : xmax + kernel_padding,
                                    ymin + kernel_padding : ymax + kernel_padding,
                                    ...,
                                ]
                            )
                            output_path = (
                                output_directory
                                / f"{raster_file.name.split('.')[0]}_{mask['class']}_{xmin}_{xmax}_{ymin}_{ymax}"
                            )
                            as_gtiff = args.geo_tiff
                            file_ending = ".png" if not args.geo_tiff else ".tif"
                            offset = (bbox.bounds[0], bbox.bounds[-1])
                            t = Thread(
                                target=write_imgs,
                                args=(
                                    image_chip,
                                    output_path,
                                    as_gtiff,
                                    file_ending,
                                    offset,
                                    (pixel_size, -pixel_size),
                                    raster_crs,
                                ),
                            )
                            T.append(t)
                            t.start()

            if bboxes:
                vector_chips(bboxes, classes, raster_crs, output_directory, raster_file.name.split(".")[0])

        bar.update(1)

    [t.join() for t in T]

    return 0
