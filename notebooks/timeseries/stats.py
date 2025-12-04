import datetime as dt
import os

import numpy as np
import xarray as xr

from sweat.common.constant import ETVar
from sweat.common.io import read_data_from_file


def mae_date(x: xr.DataArray, y: xr.DataArray) -> float:
    """
    Compute the Mean Absolute Error (MAE) between two xarray DataArrays.

    Parameters
    ----------
    x: xr.DataArray
        First input array. Must have the same shape as `y`.
    y: xr.DataArray
        Second input array. Must have the same shape as `x`.

    Returns
    -------
    float
        Mean Absolute Error
    """
    n = x.shape[0] * x.shape[1]
    mae = (abs(x - y) / n).sum()
    return mae.item()


def rmse_date(x: xr.DataArray, y: xr.DataArray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between two xarray DataArrays.

    Parameters
    ----------
    x: xr.DataArray
        First input array. Must have the same shape as `y`.
    y: xr.DataArray
        Second input array. Must have the same shape as `x`.

    Returns
    -------
    float
        Root Mean Squared Error
    """
    n = x.shape[0] * x.shape[1]
    se = ((x - y) ** 2).sum()
    rmse = np.sqrt(se / n)
    return rmse.item()


def mbe_date(x: xr.DataArray, y: xr.DataArray) -> float:
    """
    Compute the Mean Bias Error (MBE) between two xarray DataArrays.

    Parameters
    ----------
    x: xr.DataArray
        First input array. Must have the same shape as `y`.
    y: xr.DataArray
        Second input array. Must have the same shape as `x`.

    Returns
    -------
    float
        Mean Bias Error
    """
    n = x.shape[0] * x.shape[1]
    mbe = ((x - y) / n).sum()
    return mbe.item()


def r2_date(x: xr.DataArray, y: xr.DataArray) -> float:
    """
    Compute the coefficient of determination (R²) between two xarray DataArrays.

    Parameters
    ----------
    x: xr.DataArray
        First input array. Must have the same shape as `y`.
    y: xr.DataArray
        Second input array. Must have the same shape as `x`.

    Returns
    -------
    float
        Coefficient of determination
    """
    rss = ((x - y) ** 2).sum().item()
    x_mean = x.mean().item()
    tss = ((x - x_mean) ** 2).sum().item()
    return 1 - rss / tss


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


def mae_pixel(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str
) -> xr.DataArray:
    """
    Compute the spatial distribution of the Mean Absolute Error(MAE)
    between observed and simulated evapotranspiration over the given period.

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product files files
        are stored.
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.

    Returns
    -------
    xr.DataArray
        Spatial distribution of the Mean Absolute Error
    """
    time = xr.date_range(start_date, freq="1D", end=end_date)
    aes = sum(
        abs(get_et_time_series(dir_ts, date) - get_et_single_date(dir_sd, date))
        for date in time
    )
    return aes / len(time)


def rmse_pixel(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str
) -> xr.DataArray:
    """
    Compute the spatial distribution of the Root Mean Squared Error (RMSE)
    between observed and simulated evapotranspiration over the given period.

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product files
        are stored.
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.

    Returns
    -------
    xr.DataArray
        Spatial distribution of the Root Mean Squared Error
    """
    time = xr.date_range(start_date, freq="1D", end=end_date)
    ses = sum(
        (get_et_time_series(dir_ts, date) - get_et_single_date(dir_sd, date))
        ** 2
        for date in time
    )
    return np.sqrt(ses / len(time))


def mbe_pixel(dir_sd: str, dir_ts: str, start_date: str, end_date: str):
    """
    Compute the spatial distribution of the Mean Bias Error (MBE)
    between observed and simulated evapotranspiration over the given period.

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product files
        are stored.
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.

    Returns
    -------
    xr.DataArray
        Spatial distribution of the Mean Bias Error
    """
    time = xr.date_range(start_date, freq="1D", end=end_date)
    bes = sum(
        get_et_single_date(dir_sd, date) - get_et_time_series(dir_ts, date)
        for date in time
    )
    return bes / len(time)


def r2_pixel(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str
) -> xr.DataArray:
    """
    Compute the spatial distribution of the coefficient of determination (R^2)
    between observed and simulated evapotranspiration over the given period.

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product files
        are stored.
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.

    Returns
    -------
    xr.DataArray
        Spatial distribution of the coefficient of determination
    """
    time = xr.date_range(start_date, freq="1D", end=end_date)
    rss = sum(
        (get_et_time_series(dir_ts, date) - get_et_single_date(dir_sd, date))
        ** 2
        for date in time
    )
    single_date_mean = sum(
        get_et_single_date(dir_sd, date) for date in time
    ) / len(time)
    tss = sum(
        (get_et_single_date(dir_sd, date) - single_date_mean) ** 2
        for date in time
    )
    return 1 - rss / tss
