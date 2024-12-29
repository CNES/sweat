# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os
from functools import lru_cache

import geopandas as gpd


@lru_cache
def get_trishna_tiles() -> gpd.GeoDataFrame:
    """
    Get TRISHNA tiles

    Returns
    -------
    GeoDataFrame
        TRISHNA tiles
    """
    return gpd.read_file(
        "/vsizip/"
        + os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "trishna",
            "trishna_grid.gpkg.zip",
            "trishna_grid.gpkg",
        )
    )


@lru_cache
def get_land_mask() -> gpd.GeoDataFrame:
    """
    Get land polygons

    Returns
    -------
    GeoDataFrame
       Land polygons
    """
    return gpd.read_file(
        "/vsizip/"
        + os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "trishna",
            "land.gpkg.zip",
            "land.gpkg",
        )
    )


@lru_cache
def get_trishna_orbits() -> gpd.GeoDataFrame:
    """
    Get TRISHNA orbit traces

    Returns
    -------
    GeoDataFrame
        TRISHNA orbit traces
    """
    return gpd.read_file(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "trishna",
            "trishna_orbits.gpkg",
        )
    )
