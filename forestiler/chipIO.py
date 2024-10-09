from typing import List
from pathlib import Path
import geopandas as gpd
import rasterio as rio
from shapely import box
import numpy as np
from PIL import Image
import os
import platform
import multiprocessing as mp

def raster_worker(q: mp.Queue) -> None:
    if platform.system() in ["Linux", "Darwin"]:
        os.nice(5)
    while True:
        # data, path, as_geotiff, transformation, crs, bands, kernel_size
        item = q.get()
        if not item[2]:
            Image.fromarray(np.moveaxis(item[0].astype(np.uint8), 0, 2)).save(item[1])
        else:
            with rio.open(
                item[1],
                "w",
                driver="GTiff",
                height=item[6],
                width=item[6],
                count=item[5],
                dtype=item[0].dtype,
                crs=item[4],
                transform=item[3],
            ) as dst:
                for band in range(item[5]):
                    dst.write(item[0][band, ...].squeeze(), band + 1)


def vector_worker(q: mp.Queue):
    while True:
        vector_chips(*q.get())


def write_imgs(img, path, as_gtiff, file_type, offset, res, crs) -> None:
    if not as_gtiff:
        Image.fromarray(img).save(path.with_suffix(file_type))
    else:
        transform = rio.transform.Affine.translation(*offset) * rio.transform.Affine.scale(*res)
        with rio.open(
            path.with_suffix(file_type),
            "w",
            driver="GTiff",
            height=img.shape[0],
            width=img.shape[1],
            count=img.shape[2],
            dtype=img.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for band in range(img.shape[-1]):
                dst.write(img[..., band], band + 1)


def vector_chips(bboxes: List[box], classes: List[str], crs, destination: Path, base_name: str) -> None:
    vector_chips = gpd.GeoDataFrame(index=list(range(len(bboxes))), crs=crs, geometry=bboxes)
    vector_chips["class"] = classes
    vector_chips.to_file(destination / f"{base_name}_bboxes.gpkg", layer="polylayer")
