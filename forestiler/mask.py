from typing import List, Dict, Optional
from shapely import STRtree
from pathlib import Path
import geopandas as gpd


def create_masks(
    vec_mask: Path, class_field: str, all_classes: bool = True, mask_classes: Optional[List[str]] = None
) -> List[Dict[str, STRtree]]:
    vector_mask: gpd.GeoDataFrame = gpd.read_file(vec_mask)
    mask_classes: List[str] = vector_mask[class_field].unique() if all_classes else mask_classes

    mask_objects: List[Dict] = []
    for mclass in mask_classes:
        mask_objects.append(
            {
                "class": mclass,
                "tree": STRtree(vector_mask.loc[vector_mask[class_field] == mclass, "geometry"].geometry),
            }
        )

    return mask_objects
