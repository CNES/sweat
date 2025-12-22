import datetime as dt
import os

import numpy as np
import xarray as xr
from numpy import ndarray

from sweat.common.constant import ETVar
from sweat.common.io import read_data_from_file


def list_error(
    dir_sd: str,
    dir_ts: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
    absolute: bool | None = True,
    to_filter: bool | None = False,
):
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


def list_var(
    dir_var: str,
    variable: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
) -> list[float]:
    time = xr.date_range(start_date, end=end_date, freq="1D")
    var_values = []
    for date in time:
        var = (
            get_explanatory_variable(dir_var, variable, date)
            .sel(x=x, y=y, method="nearest")
            .item()
        )
        var_values.append(var)
    return var_values


def et_sd_list(
    dir_sd: str, start_date: str, end_date: str, x: int, y: int
) -> list[float]:
    time = xr.date_range(start_date, end_date)
    return [
        get_et_single_date(dir_sd, date).sel(x=x, y=y, method="nearest").item()
        for date in time
    ]


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


def get_explanatory_variable(
    path_dir: str, variable: str, date: dt.datetime
) -> xr.DataArray:
    """
    Return the sepcified daily explanatory variable for a given date
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
    filename = f"explanatory_{date.strftime('%Y%m%d')}.tif"
    path = os.path.join(path_dir, filename)
    if not os.path.isfile(path):
        msg = f"{variable} file not found for date {date}"
        raise ValueError(msg)
    data = read_data_from_file(path)
    return data[variable]


def error_date_list(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, metric
) -> list[float]:
    """
    Create a list containing the values of a specified error metric
    between observed and simulated evapotranspiration for each date
    within the given period.

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
    metric : callableread_ts_input_data, run_timeseries,
        Error metric function

    Returns
    -------
    list[float]
        List of error values for each date in the period.
    """
    errors = []
    time = xr.date_range(start_date, freq="1D", end=end_date)
    for date in time:
        error_date = metric(
            get_et_single_date(dir_sd, date), get_et_time_series(dir_ts, date)
        )
        errors.append(error_date)
    return errors


def et_ts_list(
    dir_ts: str, start_date: str, end_date: str, x: int, y: int
) -> list[float]:
    time = xr.date_range(start_date, end_date)
    return [
        get_et_time_series(dir_ts, date).sel(x=x, y=y, method="nearest").item()
        for date in time
    ]


def rad_list(
    dir_rad: str, start_date: str, end_date: str, x: int, y: int
) -> list[float]:
    time = xr.date_range(start_date, end_date)
    return [
        get_radiation(dir_rad, date).sel(x=x, y=y, method="nearest").item()
        for date in time
    ]


def var_list(
    dir_var: str, variable: str, start_date: str, end_date: str, x: int, y: int
) -> list[float]:
    time = xr.date_range(start_date, end_date)
    tp = []
    for date in time:
        tp.append(  # noqa: PERF401
            get_explanatory_variable(dir_var, variable=variable, date=date)
            .sel(x=x, y=y, method="nearest")
            .item()
        )
    return tp


def ratio_list(
    dir_rad: str, dir_sd: str, start_date: str, end_date: str, x: int, y: int
) -> list[float]:
    rad = rad_list(dir_rad, start_date, end_date, x, y)
    et_sd = et_sd_list(dir_sd, start_date, end_date, x, y)
    return [et / r for et, r in zip(et_sd, rad, strict=False)]


def api_list(tp: list[float], k: int) -> list[float]:
    api = [tp[0]]
    for i in range(1, len(tp)):
        api.append(k * api[i - 1] + tp[i])
    return api


def find_directories(input_dir: str) -> tuple[str, str, str]:
    dir_sd = os.path.join(input_dir, "et")
    dir_rad = os.path.join(input_dir, "daily_radiation")
    dir_var = os.path.join(input_dir, "explanatory_variables")
    return dir_sd, dir_rad, dir_var


def create_pixel_data_extrapolation_forward(
    # Pathes to the directories where data are stored
    input_dir: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
) -> tuple[ndarray, ndarray]:
    dir_sd, dir_rad, dir_var = find_directories(input_dir)
    # Shaping of variable data
    sw = var_list(dir_var, "sw", start_date, end_date, x, y)
    rad = rad_list(dir_rad, start_date, end_date, x, y)
    rad_1 = rad[:-1]
    et_sd = et_sd_list(dir_sd, start_date, end_date, x, y)
    et_1 = et_sd[:-1]
    input_x = np.column_stack([rad[1:], rad_1, et_1, sw[1:]])
    output_y = np.array(et_sd[1:])
    return input_x, output_y


def create_pixel_data_extrapolation_backward(
    # Pathes to the directories where data are stored
    input_dir: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
) -> tuple[ndarray, ndarray]:
    dir_sd, dir_rad, dir_var = find_directories(input_dir)
    # Shaping of variable data
    sw = var_list(dir_var, "sw", start_date, end_date, x, y)
    rad = rad_list(dir_rad, start_date, end_date, x, y)
    rad_1 = rad[1:]
    et_sd = et_sd_list(dir_sd, start_date, end_date, x, y)
    et_1 = et_sd[1:]
    input_x = np.column_stack([rad[:-1], rad_1, et_1, sw[:-1]])
    output_y = np.array(et_sd[:-1])
    return input_x, output_y
