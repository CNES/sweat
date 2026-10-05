# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for handling data related to observed and simulated
evapotranspiration, daily radiation, and extra ERA5-Land variables.
"""

import datetime as dt
import os

import xarray as xr

from sweat.common.io import read_data_from_file
from sweat.common.types import ETVar


def get_et_time_series(path_dir: str, date: dt.datetime) -> xr.DataArray:
    """
    Return the simulated evapotranspiration for a given date
    from the corresponding GeoTIFF file in the specified directory.

    Parameters
    ----------
    dir: str
        Directory where the `et_time_series_YYYYMMDD.tif` files are stored
    date: dt.datetime
        Date

    Returns
    -------
    xr.DataArray
        Simulated evapotranspiration spatial distribution
    """
    filename = f"et_time_series_{date.strftime('%Y%m%d')}.tif"
    path = os.path.join(path_dir, filename)
    if not os.path.isfile(path):
        msg = f"ET TS File not found for date {date}"
        raise ValueError(msg)
    data = read_data_from_file(path)
    return data[ETVar.ET.value]


def get_radiation(path_dir: str, date: dt.datetime) -> xr.DataArray:
    """
    Return the observed evapotranspiration for a given date
    from the corresponding GeoTIFF file in the specified directory.

    Parameters
    ----------
    dir: str
        Directory where the `et_single_date_YYYYMMDD.tif` are stored
    date: dt.datetime
        Date of the acquisition

    Returns
    -------
    xr.DataArray
        Observed evapotranspiration spatial distribution
    """
    filename = f"radiation_{date.strftime('%Y%m%d')}.tif"
    path = os.path.join(path_dir, filename)
    if not os.path.isfile(path):
        msg = f"Radiation File not found for date {date}"
        raise ValueError(msg)
    data = read_data_from_file(path)
    return data["daily_radiation"]


def get_et_single_date(path_dir: str, date: dt.datetime) -> xr.DataArray:
    """
    Return the observed evapotranspiration for a given date
    from the corresponding GeoTIFF file in the specified directory.

    Parameters
    ----------
    dir: str
        Directory where the `et_single_date_YYYYMMDD.tif` are stored
    date: dt.datetime
        Date of the acquisition

    Returns
    -------
    xr.DataArray
        Observed evapotranspiration spatial distribution
    """
    filename = f"et_single_date_{date.strftime('%Y%m%d')}.tif"
    path = os.path.join(path_dir, filename)
    if not os.path.isfile(path):
        msg = f"ET File not found for date {date}"
        raise ValueError(msg)
    data = read_data_from_file(path)
    return data[ETVar.ET.value]


def get_extra_variable(
    dir_var: str, variable: str, date: dt.datetime
) -> xr.DataArray:
    """
    Return the specified daily extra variable for a given date
    from the corresponding GeoTIFF file in the specified directory.

    Parameters
    ----------
    dir_var: str
        Directory where the `et_time_series_YYYYMMDD.tif` files are stored
    variable : str
        The ERA5-Land variable to consider (`tp`,`sro`,`src`,`sw`)
    date: dt.datetime
        Date

    Returns
    -------
    xr.DataArray
        Extra ERA5-Land variable distribution
    """
    filename = f"extra_{date.strftime('%Y%m%d')}.tif"
    path = os.path.join(dir_var, filename)
    if not os.path.isfile(path):
        msg = f"{variable} file not found for date {date}"
        raise ValueError(msg)
    data = read_data_from_file(path)
    if variable not in data.data_vars:
        msg = f"Variable '{variable}' not found in file {filename}"
        raise ValueError(msg)
    return data[variable]


def et_ts_list(
    dir_ts: str, start_date: str, end_date: str, x: int, y: int
) -> list[float]:
    """
    Create a list of simulated evapotranspiration values
    at a specified pixel for each date within the given period.

    Parameters
    ----------
    dir_ts : str
        Directory where the simulated ET product file
        are stored.
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.

    Returns
    -------
    list[float]
        List of simulated ET for each date in the period.
    """
    time = xr.date_range(start_date, end_date)
    return [
        get_et_time_series(dir_ts, date).sel(x=x, y=y, method="nearest").item()
        for date in time
    ]


def et_sd_list(
    dir_sd: str, start_date: str, end_date: str, x: int, y: int
) -> list[float]:
    """
    Create a list of observed evapotranspiration values
    at a specified pixel for each date within the given period.

    Parameters
    ----------
    dir_sd : str
        Directory where the observed ET product file
        are stored.
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.

    Returns
    -------
    list[float]
        List of observed ET for each date in the period.
    """
    time = xr.date_range(start_date, end_date)
    return [
        get_et_single_date(dir_sd, date).sel(x=x, y=y, method="nearest").item()
        for date in time
    ]


def radiation_list(
    dir_rad: str, start_date: str, end_date: str, x: int, y: int
) -> list[float]:
    """
    Create a list of daily radiation values
    at a specified pixel for each date within the given period.

    Parameters
    ----------
    dir_rad : str
        Directory where the daily radiation product file
        are stored.
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.

    Returns
    -------
    list[float]
        List of daily radiation values for each date in the period.
    """
    time = xr.date_range(start_date, end_date)
    return [
        get_radiation(dir_rad, date).sel(x=x, y=y, method="nearest").item()
        for date in time
    ]


def extra_variable_list(
    dir_var: str, variable: str, start_date: str, end_date: str, x: int, y: int
) -> list[float]:
    """
    Create a list of an extra ERA5-Land variable values
    at a specified pixel for each date within the given period.

    Parameters
    ----------
    dir_var : str
        Directory where the simulated evapotranspiration product file
        are stored.
    variable : str
        The ERA5-Land variable to consider (`tp`,`sro`,`src`,`sw`)
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.

    Returns
    -------
    list[float]
        List of the extra ERA5-Land variable values for each date in the period.
    """
    time = xr.date_range(start_date, end_date)
    tp = []
    for date in time:
        tp.append(  # noqa: PERF401
            get_extra_variable(dir_var, variable=variable, date=date)
            .sel(x=x, y=y, method="nearest")
            .item()
        )
    return tp


def list_error(
    dir_sd: str,
    dir_ts: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
    absolute: bool | None = True,
    to_filter: bool | None = False,
) -> tuple[list[float], list[int], list[dt.datetime]]:
    """
    Creates a list of error values (absolute or bias) between
    observed and simulated evapotranspiration over the given period

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product file
        are stored.
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.
    absolute: bool
        If True, computes the absolute error.
        If False, computes the bias error.
    to_filter: bool
        If True, removes the acquisition day (when observed ET is available)

    Returns
    -------
    errors, index, filtered_time :
    tuple[list[float], list[int], list[dt.datetime]]
        errors         : List of error values (absolute or bias)
        indices        : List of indices of simulated ET values considered
        filtered_time  : List of dates corresponding to the values considered

    """
    time = xr.date_range(start_date, end=end_date, freq="1D")
    errors = []
    index = []
    filtered_time = []
    for i in range(len(time)):
        et_sd = (
            get_et_single_date(dir_sd, time[i])
            .sel(x=x, y=y, method="nearest")
            .item()
        )
        et_ts = (
            get_et_time_series(dir_ts, time[i])
            .sel(x=x, y=y, method="nearest")
            .item()
        )
        daily_error = abs(et_ts - et_sd) if absolute else et_ts - et_sd
        if (not to_filter) or (to_filter and daily_error != 0):
            errors.append(daily_error)
            index.append(i)
            filtered_time.append(time[i])
    return errors, index, filtered_time


def find_directories(input_dir: str) -> tuple[str, str, str | None]:
    """
    Returns the directories for single date ET, daily radiation,
    and extra ERA5-Land variables within the given input folder.

    Parameters
    ----------
    input_dir : str
        Path to the main input directory containing the subdirectories.

    Returns
    -------
    tuple[str, str, str]
        dir_sd  : Directory path for daily evapotranspiration files.
        dir_rad : Directory path for daily radiation files.
        dir_var : Directory path for extra ERA5-Land variable files.
    """
    dir_sd = os.path.join(input_dir, "et")
    if not os.path.isdir(dir_sd):
        msg = f"Directory for daily ET files not found: {dir_sd}"
        raise FileNotFoundError(msg)
    dir_rad = os.path.join(input_dir, "daily_radiation")
    if not os.path.isdir(dir_rad):
        msg = f"Directory for daily radiation files not found: {dir_rad}"
        raise FileNotFoundError(msg)
    dir_var = None
    if os.path.isdir(os.path.join(input_dir, "extra")):
        dir_var = os.path.join(input_dir, "extra")
    return dir_sd, dir_rad, dir_var
