# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing the API
"""

from __future__ import annotations

import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from sweat.common import daily, filter, io
from sweat.common.constant import ETVar
from sweat.common.io import InputConfig
from sweat.debugging import DebuggingConfig, configure_debugging
from sweat.evaspa import ef, seb, tiling
from sweat.evaspa.config import EVASPAParamsConfig
from sweat.logging import LoggerManager
from sweat.misc import trishna
from sweat.stic import main as stic
from sweat.stic.config import STICParamsConfig
from sweat.timeseries import stack_handler as sth
from sweat.timeseries import updater_handler as uh
from sweat.timeseries.config import (
    TimeSeriesInputConfig,
    TimeSeriesParamsConfig,
)
from sweat.timeseries.io_handler import read_input as read_ts_input
from sweat.timeseries.io_handler import write_timeseries
from sweat.timeseries.timeseries_handler import create_config, window_generator

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
    orbit_id:i optional(int)
        Orbit relative number
    land_percentage:optional(float), default=10
        Minimum land percentage
    orbit_percentage:optional(float) default=25
        Minimum orbit coverage

    Returns
    -------
    tiles: GeoDataFrame
        List of tiles
    adjs: DataFrame
        List of adjacents
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
    threshold:optional(float), default=300000
        Threshold on minimum number of valid pixels

    Returns
    -------
    group_df: DataFrame
        Group of tiles
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


def read_input_data(entry: dict) -> xr.Dataset:
    """
    Read input data

    Parameters
    ----------
    entry: dict
        Input configuration

    Returns
    -------
    data: xr.Dataset
        Dataset containing input variables
    """
    # Validate input config
    msg = f"Input: {entry}"
    logger.debug(msg)
    input_config = InputConfig.model_validate(entry)
    # Read input data
    data = io.read_input(input_config.model_dump())
    logger.debug("Read input data: OK")
    return data


def read_ts_input_data(
    entry: dict,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset | None, xr.Dataset | None]:
    """
    Read time series input data

    Parameters
    ----------
    entry: dict
        Input configuration

    Returns
    -------
    et_ts: xr.Dataset
        Daily ET time series dataset
    radiation_ts: xr.Dataset
        Radiation time series dataset
    et_sd: xr.Dataset
        Daily ET single date dataset
    dem: xr.Dataset
        DEM
    """
    # Validate input config
    msg = f"Input: {entry}"
    logger.debug(msg)
    input_config = TimeSeriesInputConfig.model_validate(entry)
    # Read input data
    et_ts, radiation_ts, et_sd, dem = read_ts_input(input_config.model_dump())
    logger.debug("Read input data: OK")
    return et_ts, radiation_ts, et_sd, dem


def filter_data_for_evaspa(
    data: xr.Dataset, models: list[ef.EFModel], config: dict
) -> xr.Dataset:
    """
    Filter data for EVASPA

    Parameters
    ----------
    data: xr.Dataset
        Input data
    models : list[EFModel]
        List of EF models
    config: dict
        Filtering configuration

    Returns
    -------
    filtered_data: xr.Dataset
        Filtered dataset containing valid and flags masks
    """
    filtered_data = data.copy(deep=True)
    variables = ef.get_variables_from_models(models=models)
    valid_mask, flags_mask = filter.find_valid_pixels(
        filtered_data,
        nan_config=variables,
        valid_config=config,
    )
    filtered_data[ETVar.LST.value] = filtered_data[ETVar.LST.value].where(
        flags_mask & filter.MSK_INPUT_NODATA != 0b1, np.nan
    )
    filtered_data[ETVar.VALID.value] = valid_mask
    filtered_data[ETVar.FLAGS.value] = flags_mask
    return filtered_data


def run_evaspa(
    data: xr.Dataset, params: dict, debug: dict | None = None
) -> tuple[xr.Dataset, xr.Dataset] | None:
    """
    Run EVASPA

    Parameters
    ----------
    data: xr.Dataset
        Input data
    params: dict
        Parameter configuration
    debug: dict
        Debug configuration

    Returns
    -------
    inst_xr: xr.Dataset
        Dataset containing instant values (EF, ET, LE)
    daily_xr: xr.Dataset
        Dataset containing daily values (EF, ET, LE)
    """
    # Validate parameters config
    msg = f"Params: {params}"
    logger.debug(msg)
    params_config = EVASPAParamsConfig.model_validate(params)
    # Validate debug config
    msg = f"Debug: {debug}"
    logger.debug(msg)
    if debug is not None:
        debug_config = DebuggingConfig.model_validate(debug)
    else:
        debug_config = DebuggingConfig()
    configure_debugging(**debug_config.model_dump())
    logger.debug("Check configuration: OK")
    # Initialize EF models
    models, options = ef.initialize(params_config.ef.model_dump(by_alias=True))
    # Filter data
    data = filter_data_for_evaspa(
        data=data,
        models=models,
        config=params_config.filtering.model_dump(by_alias=True),
    )
    logger.debug("Filter data: OK")
    # Check variability
    if not ef.check_variability(
        lst=data[ETVar.LST.value],
        mask=data[ETVar.VALID.value],
        **params_config.ef.check.model_dump(),
    ):
        logger.error("Variability criteria not respected")
        return None
    logger.debug("Check variability: OK")
    # Compute EF
    ef_xr, inst_xr = ef.run(models, data, **options)
    logger.debug("Compute EF: OK")
    # Compute LE
    _, merged_xr = seb.run(data, ef_xr, **params_config.seb.model_dump())
    inst_xr = merged_xr.merge(
        inst_xr,
        join="override",
        compat="override",
        combine_attrs="no_conflicts",
    )
    logger.debug("Compute LE: OK")
    # Extrapolate at daily scale
    # Extract DEM data mask
    dem = None
    dem_data = [
        d
        for d in data.data_vars
        if d in [ETVar.HEIGHT.value, ETVar.SLOPE.value, ETVar.ASPECT.value]
    ]
    if len(dem_data) > 0:
        dem = data[dem_data]
    daily_xr = daily.extrapolate_at_daily_scale(
        inst_xr,
        variables=[
            ETVar.LE.value,
            ETVar.ET.value,
            ETVar.UNCERTAINTY_LE.value,
            ETVar.UNCERTAINTY_ET.value,
        ],
        dem=dem,
        **params_config.daily.model_dump(),
    )
    logger.debug("Daily extrapolation: OK")
    return inst_xr, daily_xr


def run_stic(
    data: xr.Dataset, params: dict, debug: dict | None = None
) -> tuple[xr.Dataset, xr.Dataset] | None:
    """
    Run STIC

    Parameters
    ----------
    data: xr.Dataset
        Input data
    params: dict
        Parameter configuration
    debug: dict
        Debug configuration

    Returns
    -------
    inst_xr: xr.Dataset
        Dataset containing instant values (EF, ET, LE)
    daily_xr: xr.Dataset
        Dataset containing daily values (EF, ET, LE)
    """
    # Validate parameters config
    msg = f"Params: {params}"
    logger.debug(msg)
    params_config = STICParamsConfig.model_validate(params)
    # Validate debug config
    msg = f"Debug: {debug}"
    logger.debug(msg)
    if debug is not None:
        debug_config = DebuggingConfig.model_validate(debug)
    else:
        debug_config = DebuggingConfig()
    configure_debugging(**debug_config.model_dump())
    logger.debug("Check configuration: OK")
    # Prepare data
    data = stic.prepare(data, **params_config.prepare.model_dump())
    logger.debug("Prepare input data: OK")
    # Filter data
    valid_mask, flags_mask = filter.find_valid_pixels(
        data,
        nan_config=[
            ETVar.LST.value,
            ETVar.ALBEDO.value,
            ETVar.TEMPERATURE.value,
            ETVar.DEWPOINT_TEMPERATURE.value,
            ETVar.FCOVER.value,
            ETVar.LAI.value,
            ETVar.EMISSIVITY.value,
            ETVar.NET_RADIATION.value,
            ETVar.LONGWAVE_NET_RADIATION.value,
        ],
        valid_config=params_config.filtering.model_dump(),
    )
    data[ETVar.VALID.value] = valid_mask
    data[ETVar.FLAGS.value] = flags_mask
    logger.debug("Filter input data: OK")
    # Compute instant ET/LE
    inst_xr = stic.run(data, **params_config.stic.model_dump())
    logger.debug("Compute instant ET/LE: OK")
    # Extract DEM data
    dem = None
    dem_data = [
        d for d in data.data_vars if d in ["elevation", "slope", "aspect"]
    ]
    if len(dem_data) > 0:
        dem = data[dem_data]
    # Extrapolate at daily scale
    daily_xr = daily.extrapolate_at_daily_scale(
        inst_xr,
        variables=["le", "et"],
        dem=dem,
        **params_config.daily.model_dump(),
    )
    logger.debug("Daily extrapolation: OK")
    return inst_xr, daily_xr


def run_timeseries(
    et_ts: xr.Dataset,
    radiation_ts: xr.Dataset | None,
    et_sd: xr.Dataset | None,
    dem: xr.Dataset | None,
    params: dict,
    debug: dict | None = None,
) -> xr.Dataset:
    """
    Run ET time series

    Parameters
    ----------
    et_ts: xr.Dataset
        Daily ET time series dataset
    radiation_ts: xr.Dataset
        Radiation time series dataset
    et_sd: xr.Dataset
        Daily ET single date dataset
    dem: xr.Dataset
        DEM (height, slope, aspect)
    params: dict
        Parameter configuration
    debug: dict
        Debug configuration

    Returns
    -------
    updated: xr.Dataset
        Updated daily ET time series dataset
    """
    # Validate parameters config
    msg = f"Params: {params}"
    logger.debug(msg)
    params_config = TimeSeriesParamsConfig.model_validate(params)
    # Validate debug config
    msg = f"Debug: {debug}"
    logger.debug(msg)
    if debug is not None:
        debug_config = DebuggingConfig.model_validate(debug)
    else:
        debug_config = DebuggingConfig()
    configure_debugging(**debug_config.model_dump())
    logger.debug("Check configuration: OK")
    # Prepare data: filter and stack
    ts, feed = sth.run(
        et_ts,
        radiation_ts,
        et_sd,
        dem,
        et_single_date_filtering=params_config.stack.et_single_date_filtering.model_dump(),
    )
    logger.debug("Prepare time series stack: OK")
    # Update
    updated_ts = uh.run(ts, feed, params_config.update.model_dump())
    logger.debug("Update time series: OK")
    return updated_ts


def run_window_time_series(
    period_start: dt.datetime,
    period_end: dt.datetime,
    et_single_date_dir: str,
    radiation_dir: str,
    et_time_series_dir: str,
    window: int,
    shift: int,
    params: dict | None = None,
    debug: dict | None = None,
    config_verbose: bool = False,
    config_dir: str | None = None,
) -> None:
    """
    Move a sliding window over a given period and for each shift,
    run the time series.

    Parameters
    ----------
    period_start: dt.datetime
        Period start date
    period_end: dt.datetime
        Period end date
    et_single_date_dir: str
        Directory where et_single_date .tif files are downloaded
    radiation_dir: str
        Directory where radiation .tif files are downloaded
    et_time_series_dir: str
        Directory where et_time_series .tif files are downloaded
    window: int
        Size of the sliding window (in days)
    shift: int
        Shift between two consecutive windows (in days)
    params: dict
        Configuration parameters to run one step for timeseries
    config_verbose: bool
        If true, the configuration dictionary will be stored as a .json file
    config_dir: str
        Directory to store the json configuration file if verbose True
        (default: current directory)
    """
    for window_start, window_end in window_generator(
        period_start, period_end, window, shift
    ):
        msg = (
            "Start process timeseries between "
            f"{window_start.strftime('%Y%m%d')}"
            f" and {window_end.strftime('%Y%m%d')}..."
        )
        logger.debug(msg)
        # Create configuration to run one step
        config = create_config(
            window_start=window_start,
            window_end=window_end,
            et_single_date_dir=et_single_date_dir,
            radiation_dir=radiation_dir,
            et_time_series_dir=et_time_series_dir,
            params=params,
            debug=debug,
            verbose=config_verbose,
            config_dir=config_dir,
        )
        logger.debug("Create timeseries configuration: OK")
        # Read timeseries input
        et_ts, radiation_ts, et_sd, dem = read_ts_input_data(
            config.input.model_dump()
        )
        logger.debug("Read timeseries input data: OK")
        # Run timeseries
        updated_ts = run_timeseries(
            et_ts=et_ts,
            radiation_ts=radiation_ts,
            et_sd=et_sd,
            dem=dem,
            params=config.params.model_dump(),
            debug=config.debug.model_dump(),
        )
        # Write et_time_series file
        write_timeseries(
            updated_ts, root_name="et_time_series", directory=et_time_series_dir
        )
        logger.debug("Write timeseries: OK")
        msg = (
            f"Process timeseries between {window_start.strftime('%Y%m%d')}"
            f" and {window_end.strftime('%Y%m%d')}: OK"
        )
        logger.info(msg)
