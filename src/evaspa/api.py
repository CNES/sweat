#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing the API for EVASPA
"""

from typing import Tuple

import geopandas as gpd
import pandas as pd

import evaspa.tiling as tiling
import evaspa.trishna as trishna

from .logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def generate_tiles(
    roi: str,
    orbit_id: int | None = None,
    land_percentage: float = 10,
    orbit_percentage: float = 25,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Generate tile list

    Parameters
    ----------
    roi: optional(str)
        ROI file path
    orbit_id:i optinal(int)
        Orbit relative number
    land_percentage:optinal(float), default=10
        Minimum land percentage
    orbit_percentage:optinal(float) default=25
        Minimum orbit coverage

    Return
    ------
    tiles: GeoDataFrame
    adjs: DataFrame
    """
    columns = ["id", "epsg", "geometry"]
    tiles = trishna.get_trishna_tiles()
    land = trishna.get_land_mask()
    orbits = trishna.get_trishna_orbits()
    if roi is not None:
        roi_gdf = gpd.read_file(roi)
        tiles = tiling.intersection(tiles[columns], roi_gdf)
        tiles = tiles.rename(columns={"overlap_percentage": "roi_coverage"}).drop(
            columns=["overlap_geometry"]
        )
        columns.append("roi_coverage")
    if orbit_id is not None:
        if orbit_id < 0 or orbit_id > 115:
            raise ValueError("Orbit ID must be between 0 and 114")
        tiles = tiling.intersection(tiles[columns], orbits[orbits.orbit_id == orbit_id])
        tiles = tiles.rename(columns={"overlap_percentage": "orbit_coverage"}).drop(
            columns=["overlap_geometry"]
        )
        columns.append("orbit_coverage")
        # Keep tile with enough orbit coverage
        tiles = tiles[tiles.orbit_coverage > orbit_percentage]
    tiles = tiling.intersection(tiles[columns], land)
    tiles = tiles.rename(columns={"overlap_percentage": "land"}).drop(
        columns=["overlap_geometry"]
    )
    # Keep tile with enough land coverage
    tiles = tiles[tiles.land > land_percentage]
    # Adjacent tiles
    adjs = tiling.generate_adjacents(tiles, land)
    return tiles, adjs


def regroup_tiles(
    tiles: gpd.GeoDataFrame, adjs: pd.DataFrame, threshold: int = 300000
):
    """
    Regroup tiles

    Parameters
    ----------
    tiles: GeoDataFrame
        List of tiles
    adjs: DataFrame
        List of adjacent tiles
    threshold:optinal(float), default=300000
        Threshold on minimum number of valid pixels

    Return
    ------
    group_df: DataFrame
    """
    land = trishna.get_land_mask()
    tile_df, group_df = tiling.initialize_regroup(tiles, land)
    tile_df, group_df = tiling.regroup(tile_df, adjs, group_df, threshold, land)
    logger.info(f"Number of groups = {len(group_df)}")
    group_df["group_size"] = group_df.apply(lambda x: len(x.name.split(",")), axis=1)
    logger.info(group_df["group_size"].value_counts())
    for i, nb in group_df["group_size"].value_counts().items():
        logger.info(f"Group size : {i} - Number : {nb}")
    return group_df
