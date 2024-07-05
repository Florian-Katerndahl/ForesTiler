from typing import List
from pathlib import Path
import geopandas as gpd
import rasterio as rio
from shapely import box
import numpy as np


def write_imgs(img, path, as_gtiff, file_type, offset, res, crs) -> None:
    if not as_gtiff:
        img.save(path.with_suffix(file_type))
    else:
        data = np.array(img)
        transform = rio.transform.Affine.translation(*offset) * rio.transform.Affine.scale(*res)
        with rio.open(
            path.with_suffix(file_type),
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=data.shape[2],
            dtype=data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for band in range(data.shape[-1]):
                dst.write(data[..., band], band + 1)


def vector_chips(bboxes: List[box], classes: List[str], crs, destination: Path, base_name: str) -> None:
    vector_chips = gpd.GeoDataFrame(index=list(range(len(bboxes))), crs=crs, geometry=bboxes)
    vector_chips["class"] = classes
    vector_chips.to_file(destination / f"{base_name}_bboxes.gpkg")
