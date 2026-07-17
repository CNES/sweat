# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for managing sliding windows for time series
"""

import datetime as dt
import json
import os
from collections.abc import Generator

import xarray as xr

import sweat.timeseries.config as cfg
from sweat.common.config import json_yaml_serial
from sweat.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


def window_generator(
    period_start: dt.datetime, period_end: dt.datetime, window: int, shift: int
) -> Generator[tuple[dt.datetime, dt.datetime], None]:
    """
    Generate start and end dates for each sliding window shift within
    a given period

    Parameters
    ----------
    period_start: dt.datetime
        Period start date
    period_end: dt.datetime
        Period end date
    window: int
        Size of the window (in days)
    shift: int
        Shift between two consecutive windows (in days)

    Yields
    ------
    window_start, window_end: tuple[dt.datetime, dt.datetime]
        Start and end dates of each sliding window.
    """
    window_start = period_start
    window_end = period_start
    k = 1
    stop = False
    while not stop:
        # Initial window must contain only the period start date
        if k == 1:
            pass
        else:
            # Waiting for a complete size window to shift window start date
            if k > window:
                window_start += dt.timedelta(days=shift)
            window_end += dt.timedelta(days=shift)
            # Case where the remaining days are fewer than the window shift
            if window_end >= period_end:
                window_end = period_end
                stop = True
        k += shift
        yield window_start, window_end


def create_config(
    window_start: dt.datetime,
    window_end: dt.datetime,
    et_single_date_dir: str,
    radiation_dir: str,
    et_time_series_dir: str,
    dem: str | None = None,
    params: dict | None = None,
    debug: dict | None = None,
    verbose: bool = False,
    config_dir: str | None = None,
) -> cfg.TimeSeriesInputFile:
    """
    Create configuration to run timeseries for a given window

    Parameters
    ----------
    window_start: dt.datetime
        Window start date
    window_end: dt.datetime
        Window end date
    et_single_date_dir: str
        Directory where et_single_date .tif files are downloaded
    radiation_dir: str
        Directory where radiation .tif files are downloaded
    et_time_series_dir: str
        Directory where et_time_series .tif files are downloaded
    dem: str
        Path to DEM file
    params: dict
        Parameters configuration to use
    debug: dict
        Debugging configuration to use
    verbose: bool
        If true, the configuration dictionary will be stored as a .json file
    config_dir: str
        Directory to store the JSON configuration file if verbose is True
        (default: current directory)

    Return
    -------
    config_dict: dict
        Configuration dictionary
    """
    # Define parameters
    dates = xr.date_range(window_start, freq="1D", end=window_end)
    et_time_series: list[str] = []
    radiation: list[str] = []
    et_single_date: list[str] = []
    for date in dates:
        radiation_path = os.path.join(
            radiation_dir, f"radiation_{date.strftime('%Y%m%d')}.tif"
        )
        if os.path.isfile(radiation_path):  # Check if the radiation file exists
            radiation.append(radiation_path)
        et_single_date_path = os.path.join(
            et_single_date_dir, f"et_single_date_{date.strftime('%Y%m%d')}.tif"
        )
        if os.path.isfile(
            et_single_date_path
        ):  # Check if the et_single_date file exists
            et_single_date.append(et_single_date_path)
        et_time_series_path = os.path.join(
            et_time_series_dir, f"et_time_series_{date.strftime('%Y%m%d')}.tif"
        )
        if os.path.isfile(
            et_time_series_path
        ):  # Check if the et_time_series file exists
            et_time_series.append(et_time_series_path)
    if params is None:
        params = {}
    if debug is None:
        debug = {}
    # Instantiate configuration
    msg = (
        f"Dates: {dates}, ET dates: {et_single_date}"
        f"Radiation: {radiation}, ET timeseries: {et_time_series}"
    )
    logger.debug(msg)
    config = cfg.TimeSeriesInputFile(
        input=cfg.TimeSeriesInputConfig(
            dates=dates,
            et_single_date=et_single_date,
            radiation=radiation,
            et_time_series=et_time_series,
            dem=dem,
        ),
        output=cfg.OutputConfig(path=et_time_series_dir),
        params=cfg.TimeSeriesParamsConfig.model_validate(params),
        debug=cfg.DebuggingConfig.model_validate(debug),
    )
    # Write configuration file if verbose mode is activated
    if verbose:
        if config_dir is None:
            config_dir = os.getcwd()
        os.makedirs(config_dir, exist_ok=True)
        date_min = dates[0].strftime("%Y%m%d")
        date_max = dates[-1].strftime("%Y%m%d")
        filename = f"config_{date_min}_{date_max}.json"
        file_path = os.path.join(config_dir, filename)
        with open(file_path, "w") as f:
            json.dump(
                config.model_dump(), f, indent=4, default=json_yaml_serial
            )
    return config
