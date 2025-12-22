import numpy as np
import scipy.stats
import xarray as xr
from models_tools import (
    get_et_single_date,
    get_et_time_series,
    list_error,
    list_var,
)


def pixel_rmse(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, x: int, y: int
) -> float:
    errors, _, _ = list_error(
        dir_sd, dir_ts, start_date, end_date, x, y, to_filter=True
    )
    n = len(errors)
    squared_errors = [daily_error**2 for daily_error in errors]
    rmse = np.sqrt(sum(squared_errors) / n)
    return rmse.item()


def pixel_mae(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, x: int, y: int
) -> float:
    errors, _, _ = list_error(
        dir_sd,
        dir_ts,
        start_date,
        end_date,
        x,
        y,
        absolute=True,
        to_filter=True,
    )
    n = len(errors)
    return sum(errors) / n


def pixel_mbe(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, x: int, y: int
) -> float:
    errors, _, _ = list_error(
        dir_sd,
        dir_ts,
        start_date,
        end_date,
        x,
        y,
        absolute=False,
        to_filter=True,
    )
    n = len(errors)
    return sum(errors) / n


def pixel_r2(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, x: int, y: int
) -> float:
    time = xr.date_range(start_date, end=end_date, freq="1D")
    errors, _, _ = list_error(
        dir_sd,
        dir_ts,
        start_date,
        end_date,
        x,
        y,
        absolute=False,
        to_filter=True,
    )
    n = len(errors)
    squared_errors = [daily_error**2 for daily_error in errors]
    rss = sum(squared_errors) / n
    mean_et_sd = (
        sum(
            get_et_single_date(dir_sd, date)
            .sel(x=x, y=y, method="nearest")
            .item()
            for date in time
        )
        / n
    )
    tss = (
        sum(
            (
                get_et_single_date(dir_sd, date)
                .sel(x=x, y=y, method="nearest")
                .item()
                - mean_et_sd
            )
            ** 2
            for date in time
        )
    ) / n
    return 1 - rss / tss


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
    errors = xr.concat(
        [
            abs(
                get_et_time_series(dir_ts, date)
                - get_et_single_date(dir_sd, date)
            )
            for date in time
        ],
        dim="time",
    )
    return errors.mean(dim="time")


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
    errors = xr.concat(
        [
            (
                get_et_time_series(dir_ts, date)
                - get_et_single_date(dir_sd, date)
            )
            ** 2
            for date in time
        ],
        dim="time",
    )
    return errors.mean(dim="time") ** 0.5


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
    errors = xr.concat(
        [
            (
                get_et_time_series(dir_ts, date)
                - get_et_single_date(dir_sd, date)
            )
            for date in time
        ],
        dim="time",
    )
    return errors.mean(dim="time")


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
    rs = xr.concat(
        [
            (
                get_et_time_series(dir_ts, date)
                - get_et_single_date(dir_sd, date)
            )
            ** 2
            for date in time
        ],
        dim="time",
    )
    rss = rs.sum(dim="time")
    sd_values = xr.concat(
        [get_et_single_date(dir_sd, date) for date in time], dim="time"
    )
    mean_sd = sd_values.mean(dim="time")
    ts = xr.concat(
        [(mean_sd - get_et_single_date(dir_sd, date)) ** 2 for date in time],
        dim="time",
    )
    tss = ts.sum(dim="time")
    return 1 - rss / tss


def variable_correlation(
    dir_var: str,
    dir_sd: str,
    dir_ts: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
    variable: str,
    correlation: str,
    absolute: bool,
):
    errors, index, _ = list_error(
        dir_sd, dir_ts, start_date, end_date, x, y, absolute, to_filter=True
    )
    var_values = list_var(dir_var, variable, start_date, end_date, x, y)
    filtered_vars = [var_values[i] for i in index]
    if correlation == "spearman":
        corr = scipy.stats.spearmanr(filtered_vars, errors)
    elif correlation == "pearson":
        corr = scipy.stats.pearsonr(filtered_vars, errors)
    return corr[0]
