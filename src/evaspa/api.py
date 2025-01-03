# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing the API for EVASPA
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import geopandas as gpd

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

from evaspa import ef, filter, io, merging, seb, tiling, trishna
from evaspa.config import InputConfig, ParamsConfig
from evaspa.debugging import DebuggingConfig, configure_debugging
from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

ORBIT_NUMBER_MIN = 0
ORBIT_NUMBER_MAX = 115


def generate_tiles(
    roi: str,
    orbit_id: int | None = None,
    land_percentage: float = 10,
    orbit_percentage: float = 25,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
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
        tiles = tiles.rename(
            columns={"overlap_percentage": "roi_coverage"}
        ).drop(columns=["overlap_geometry"])
        columns.append("roi_coverage")
    if orbit_id is not None:
        if orbit_id < ORBIT_NUMBER_MIN or orbit_id > ORBIT_NUMBER_MAX:
            msg = "Orbit ID must be between 0 and 114"
            raise ValueError(msg)
        tiles = tiling.intersection(
            tiles[columns], orbits[orbits.orbit_id == orbit_id]
        )
        tiles = tiles.rename(
            columns={"overlap_percentage": "orbit_coverage"}
        ).drop(columns=["overlap_geometry"])
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
    msg = f"Number of tiles = {len(tiles)}"
    logger.info(msg)
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
    msg = f"Number of groups = {len(group_df)}"
    logger.info(msg)
    group_df["group_size"] = group_df.apply(
        lambda x: len(x.name.split(",")), axis=1
    )
    for i, nb in group_df["group_size"].value_counts().items():
        msg = f"Group size : {i} - Number : {nb}"
        logger.info(msg)
    return group_df


def run_evaspa(
    entry: dict, params: dict, debug: dict | None = None
) -> xr.Dataset | None:
    """
    Description
    -----------
    Run EVASPA

    Parameters
    ----------
    entry: dict
        Input configuration
    params: dict
        Parameter configuration

    Return
    ------
    ef: xr.Dataset
        Instant evaporative fraction
    le: xr.Dataset
        Instant latent heat flux
    """
    # Validate input config
    msg = f"Input: {entry}"
    logger.debug(msg)
    input_config = InputConfig.model_validate(entry)
    # Validate parameters config
    msg = f"Params: {params}"
    logger.debug(msg)
    params_config = ParamsConfig.model_validate(params)
    # Validate debug config
    msg = f"Debug: {debug}"
    logger.debug(msg)
    if debug is not None:
        debug_config = DebuggingConfig.model_validate(debug)
    else:
        debug_config = DebuggingConfig()
    configure_debugging(**debug_config.model_dump())
    # Read input data
    if os.path.isfile(input_config.path):
        data = io.read_data_from_file(input_config.path)
    else:
        data = io.read_data(input_config.path)
    # Filter data
    data["valid"] = filter.determine_valid_pixels(
        data, **params_config.filtering.model_dump()
    )
    # Check variablity
    if not ef.check_variability(
        data["lst"], mask=data["valid"], **params_config.ef.check.model_dump()
    ):
        logger.error("Variability criteria not respected")
        return None
    # Compute EF
    models, options = ef.initialize(params_config.ef.model_dump())
    ef_xr, inst_xr = ef.run(
        models, data, mask="valid", **params_config.ef.options.model_dump()
    )
    # Compute LE
    le_xr = seb.run(data, ef_xr, **params_config.seb.model_dump())
    inst_xr["le"] = merging.merge(le_xr, method=merging.MergeMethod.MEAN)
    return inst_xr
