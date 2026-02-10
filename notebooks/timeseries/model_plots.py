"""
Module for displaying error metrics between simulated and observed
evapotranspiration, as well as other related statistics.
"""

import datetime as dt
import logging
from collections.abc import Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from model_stats import (
    mae_date,
    mae_roi,
    mbe_roi,
    pixel_r2,
    pixel_rmse,
    r2_date,
    r2_roi,
    rmse_roi,
    variable_correlation,
)
from model_tools import (
    et_sd_list,
    et_ts_list,
    extra_variable_list,
    get_et_single_date,
    get_et_time_series,
    list_error,
    radiation_list,
)
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(format="%(message)s", level=logging.INFO)


metrics: dict[str, tuple[str, Callable[[str, str, str, str], xr.DataArray]]] = {
    "mae": ("Mean Absolute Error", mae_roi),
    "mbe": ("Mean Bias Error", mbe_roi),
    "rmse": ("Root Mean Squared Error", rmse_roi),
    "r2": ("Coefficient of Determination", r2_roi),
}


variables = {
    "tp": "precipitations",
    "sro": "soil runoff",
    "src": "soil skin reservoir",
    "sw": "volumetric soil water",
    "et": "evapotranspiration",
}


def plot_variable_evolution(
    dir_var: str,
    dir_sd: str,
    dir_rad: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
):
    """
    Plots observed ET, daily solar radiation, precipitation,
    and volumetric soil water at a given pixel over a specified period

    Parameters
    ----------
    dir_var : str
        Directory where the observed extra ERA5-Land product files
        are stored.
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
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)
    """
    time = xr.date_range(start_date, end_date)
    rad = radiation_list(dir_rad, start_date, end_date, x, y)
    et_sd = et_sd_list(dir_sd, start_date, end_date, x, y)
    tp = extra_variable_list(dir_var, "tp", start_date, end_date, x, y)
    sw = extra_variable_list(dir_var, "sw", start_date, end_date, x, y)
    fig, axes = plt.subplots(2, 2, figsize=(18, 5))
    axes[0, 0].plot(time, et_sd, color="green")
    axes[0, 0].set_title("daily evapotranspiration")
    axes[0, 1].plot(time, tp, color="blue")
    axes[0, 1].set_title("daily precipitation")
    axes[1, 0].plot(time, rad, color="orange")
    axes[1, 0].set_title("daily surface radiation")
    axes[1, 1].plot(time, sw, color="purple")
    axes[1, 1].set_title("volumetric soil water")
    plt.tight_layout()
    plt.show()


# ==============================================================
# Interpolation (Rsd) algorithm specific functions
# ==============================================================


def plot_daily_et_comparison(dir_sd: str, dir_ts: str, date: str) -> None:
    """
    Display the spatial distribution of observed ET over the ROI,
    the simulated ET, and the difference between simulated and observed ET
    for a given date.

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product file
        are stored.
    date : str
        date in `YYYY-MM-DD` format.
    """
    date_dt = dt.datetime.strptime(date, "%Y-%m-%d")  # noqa: DTZ007
    et_sd = get_et_single_date(dir_sd, date_dt)
    et_ts = get_et_time_series(dir_ts, date_dt)
    diff = et_ts - et_sd
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(et_sd.values, origin="lower", cmap="viridis")
    axes[0].set_title(f"Observed ET on {date}")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(et_ts.values, origin="lower", cmap="viridis")
    axes[1].set_title(f"Simulated ET on {date}")
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(diff.values, origin="lower", cmap="RdBu")
    axes[2].set_title("Difference")
    fig.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    plt.show()


def plot_spatial_distribution_error(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str
) -> None:
    """
    Displays the spatial distribution of error metrics over the ROI
    for a given period.

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
    """
    mbe_pxl = mbe_roi(dir_sd, dir_ts, start_date, end_date)
    mae_pxl = mae_roi(dir_sd, dir_ts, start_date, end_date)
    rmse_pxl = mae_roi(dir_sd, dir_ts, start_date, end_date)
    r2_pxl = r2_roi(dir_sd, dir_ts, start_date, end_date)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Error Metrics per Pixel from {start_date} to {end_date}", fontsize=18
    )
    im0 = axes[0, 0].imshow(mae_pxl.values, origin="lower", cmap="viridis")
    axes[0, 0].set_title("Mean Absolute Error")
    fig.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].imshow(mbe_pxl.values, origin="lower", cmap="viridis")
    axes[0, 1].set_title("Mean Bias Error")
    fig.colorbar(im1, ax=axes[0, 1])
    im2 = axes[1, 0].imshow(rmse_pxl.values, origin="lower", cmap="viridis")
    axes[1, 0].set_title("Root Mean Squared Error")
    fig.colorbar(im2, ax=axes[1, 0])
    im3 = axes[1, 1].imshow(
        r2_pxl.values, origin="lower", cmap="RdBu", vmin=0, vmax=1
    )
    axes[1, 1].set_title("Coefficient of Determination")
    fig.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()


def plot_time_evolution_error(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, x: int, y: int
) -> None:
    """
    Displays the temporal evolution of absolute and bias errors
    at the specified pixel over a given period

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
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)
    """
    ae, _, time = list_error(
        dir_sd, dir_ts, start_date, end_date, x, y, absolute=True
    )
    be, _, _ = list_error(
        dir_sd, dir_ts, start_date, end_date, x, y, absolute=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    fig.suptitle(
        f"Evolution of daily error from {start_date} to {end_date} "
        f"for pixel [{x},{y}]",
        fontsize=12,
    )
    axes[0].bar(time, ae, color="green", width=0.25)
    axes[0].set_title("Absolute Error")
    axes[0].set_ylabel("mm/day")

    axes[1].bar(time, be, color="purple", width=0.25)
    axes[1].set_title("Bias Error")
    axes[1].set_ylabel("mm/day")
    for ax in axes.flatten():
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.show()


def plot_pixel_time_statistical_distribution(
    dir_sd: str,
    dir_ts: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
    absolute: bool,
) -> None:
    """
    Displays the statistical distribution of absolute or bias error
    computed over a given period for pixels within the ROI.

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
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)
    absolute: bool
        If True, displays the absolute error.
        If False, displays the bias error.
    """
    errors, _, _ = list_error(
        dir_sd,
        dir_ts,
        start_date,
        end_date,
        x,
        y,
        absolute=absolute,
        to_filter=True,
    )
    error_name = "absolute" if absolute else "bias"
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=40, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of daily {error_name} error")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def plot_roi_error_statistical_distribution(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, metric_name: str
) -> None:
    """
    Displays the statistical distribution of the specified metric error
    computed over a given period for pixels within the ROI.

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
    metric_name : str
        Error metric (`mae`, `rmse`, `mbe`, `r2`)
    """
    _, metric = metrics[metric_name]
    errors = metric(dir_sd, dir_ts, start_date, end_date)  # type: ignore[assignment]
    plt.figure(figsize=(8, 5))
    plt.hist(
        errors.values.flatten(), color="skyblue", edgecolor="black", bins=40
    )
    plt.title(f"Distribution of pixel {metric_name}")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def find_pixel(data: xr.DataArray, critere: str):
    if critere == "min":
        coord_crit = data.where(data == data.min(), drop=True).coords
    elif critere == "max":
        coord_crit = data.where(data == data.max(), drop=True).coords
    elif critere == "median":
        median = data.median()
        diff = abs(data - median)
        coord_crit = data.where(diff == diff.min(), drop=True).coords
    first_coord = {
        dim: (
            coord_crit[dim].item()
            if coord_crit[dim].size == 1
            else coord_crit[dim].values[0]
        )
        for dim in coord_crit
    }
    x = first_coord["x"]
    y = first_coord["y"]
    return x, y


def spatial_stats_metric(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, metric_name: str
):
    _, metric = metrics[metric_name]
    errors = metric(dir_sd, dir_ts, start_date, end_date)  # type: ignore[assignment]
    x_min, y_min = find_pixel(errors, "min")
    min_value = errors.sel(x=x_min, y=y_min).item()
    x_max, y_max = find_pixel(errors, "max")
    max_value = errors.sel(x=x_max, y=y_max).item()
    x_med, y_med = find_pixel(errors, "median")
    median = errors.sel(x=x_med, y=y_med).item()
    return pd.DataFrame(
        {
            "Min": [min_value],
            "Argmin": [(round(x_min), round(y_min))],
            "Max": [max_value],
            "Argmax": [(round(x_max), round(y_max))],
            "Median": [median],
            "Argmedian": [(round(x_med), round(y_med))],
        }
    )


def print_roi_statistics(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str
) -> None:
    """
    Compute and returns the error metrics over the ROI
    for a given period.

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
    """
    logging.info(
        "######## %s-%s PERIOD SPATIAL STATISTICS ########\n",
        start_date,
        end_date,
    )
    logging.info("-----MAE-----")
    logging.info(
        "%s\n",
        spatial_stats_metric(dir_sd, dir_ts, start_date, end_date, "mae"),
    )

    logging.info("-----MBE-----")
    logging.info(
        "%s\n",
        spatial_stats_metric(dir_sd, dir_ts, start_date, end_date, "mbe"),
    )

    logging.info("-----RMSE-----")
    logging.info(
        "%s\n",
        spatial_stats_metric(dir_sd, dir_ts, start_date, end_date, "rmse"),
    )

    logging.info("-----R²-----")
    logging.info(
        "%s\n", spatial_stats_metric(dir_sd, dir_ts, start_date, end_date, "r2")
    )


def plot_time_error_characteristic_pixels(
    dir_sd: str,
    dir_ts: str,
    start_date: str,
    end_date: str,
    absolute: bool | None = False,
):
    """
    Plots the time evolution of absolute or bias errors
    over the specified period
    for three representative ROI pixels:
    median, maximum, and minimum mean error.”

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product file
        are stored.
    date : str
        date in `YYYY-MM-DD` format.
    absolute: bool
        If True, displays the absolute error.
        If False, displays the bias error.
    """
    metric = mbe_roi if absolute else mae_roi
    metric_values = metric(dir_sd, dir_ts, start_date, end_date)
    time = xr.date_range(start_date, freq="1D", end=end_date)
    x_min, y_min = find_pixel(metric_values, "min")
    x_max, y_max = find_pixel(metric_values, "max")
    x_med, y_med = find_pixel(metric_values, "median")
    errors_min = []
    errors_max = []
    errors_median = []
    for date in time:
        errors_min.append(
            get_et_single_date(dir_sd, date).sel(x=x_min, y=y_min).item()
            - get_et_time_series(dir_ts, date).sel(x=x_min, y=y_min).item()
        )
        errors_max.append(
            get_et_single_date(dir_sd, date).sel(x=x_max, y=y_max).item()
            - get_et_time_series(dir_ts, date).sel(x=x_max, y=y_max).item()
        )
        errors_median.append(
            get_et_single_date(dir_sd, date).sel(x=x_med, y=y_med).item()
            - get_et_time_series(dir_ts, date).sel(x=x_med, y=y_med).item()
        )
    if absolute:
        errors_min = [abs(e) for e in errors_min]
        errors_max = [abs(e) for e in errors_max]
        errors_median = [abs(e) for e in errors_median]
    errors_min = [np.nan if e == 0 else e for e in errors_min]
    errors_max = [np.nan if e == 0 else e for e in errors_max]
    errors_median = [np.nan if e == 0 else e for e in errors_median]
    plt.figure(figsize=(6, 6))
    plt.scatter(
        time,
        errors_min,
        label=f"Best Mean Pixel\n[{round(x_min)}, {round(y_min)}]",
        s=20,
    )
    plt.scatter(
        time,
        errors_max,
        label=f"Worst Mean Pixel\n[{round(x_max)}, {round(y_max)}]",
        s=20,
        color="red",
    )
    plt.scatter(
        time,
        errors_median,
        label=f"Median Mean Pixel\n[{round(x_med)}, {round(y_med)}]",
        s=20,
        color="orange",
    )
    description = "Absolute Errors" if absolute else "Errors"
    plt.xlabel("Date")
    plt.ylabel(description)
    plt.title(
        f"Evolution of daily {description} for selected pixels "
        f"from {start_date} to {end_date}"
    )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True, linestyle="--", alpha=0.4)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def daily_error_table(
    date: dt.datetime, dir_sd: str, dir_ts: str, absolute: bool | None = False
) -> pd.DataFrame:
    et_sd: xr.DataArray = get_et_single_date(dir_sd, date)
    et_ts: xr.DataArray = get_et_time_series(dir_ts, date)

    ae: xr.DataArray = abs(et_sd - et_ts) if absolute else et_sd - et_ts
    mae_val: float = mae_date(et_sd, et_ts)

    ae_min: float = ae.min().item()
    ae_min_coords = find_pixel(ae, "min")

    ae_max: float = ae.max().item()
    ae_max_coords = find_pixel(ae, "max")

    ae_med: float = ae.median().item()
    ae_med_coords = find_pixel(ae, "median")

    return pd.DataFrame(
        {
            "Min": [ae_min],
            "Argmin": [np.round(ae_min_coords)],
            "Median": [ae_med],
            "Argmedian": [np.round(ae_med_coords)],
            "Max": [ae_max],
            "Argmax": [np.round(ae_max_coords)],
            "Mean": [mae_val],
        }
    )


def print_daily_statistics(dir_sd: str, dir_ts: str, date: str) -> None:
    """
    Compute and returns the errors between simulated and observed ET
    for a given date.

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product file
        are stored.
    date : str
        date in `YYYY-MM-DD` format.
    """
    date_dt = dt.datetime.strptime(date, "%Y-%m-%d")  # noqa: DTZ007
    logging.info("######## %s SPATIAL STATISTICS ########\n", date)

    logging.info("-----ABSOLUTE ERROR-----")
    abs_error_table = daily_error_table(date_dt, dir_sd, dir_ts, absolute=True)
    logging.info("%s\n", abs_error_table)

    logging.info("-----BIAS ERROR-----")
    bias_error_table = daily_error_table(date_dt, dir_sd, dir_ts)
    logging.info("%s\n", bias_error_table)

    logging.info("-----R²-----")
    r2_val = r2_date(
        get_et_single_date(dir_sd, date_dt),
        get_et_time_series(dir_ts, date_dt),
    )
    logging.info("value: %s\n", r2_val)


def pixel_time_stats(
    dir_sd: str,
    dir_ts: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
    absolute: bool,
) -> pd.DataFrame:
    errors, _, filtered_time = list_error(
        dir_sd, dir_ts, start_date, end_date, x, y, absolute, to_filter=True
    )
    np_errors = np.array(errors)
    argmax = np.nanargmax(np_errors)
    max_value = np.max(np_errors)
    argmin = np.nanargmin(np_errors)
    min_value = np.nanmin(np_errors)
    mean = np.nanmean(np_errors)
    date_max = filtered_time[argmax].strftime("%Y-%m-%d")
    date_min = filtered_time[argmin].strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "Min": [min_value],
            "Argmin date": [date_min],
            "Max": [max_value],
            "Argmax date": [date_max],
            "Mean": [mean],
        }
    )


def print_pixel_statistics(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, x: int, y: int
) -> None:
    """
    Compute and returns the error metrics at a single specified pixel
    for a given period.

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
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)
    """
    logging.info(
        "######## PIXEL [%s,%s] STATISTICS FOR %s / %s PERIOD ########",
        x,
        y,
        start_date,
        end_date,
    )

    logging.info("\n-----ABSOLUTE ERROR (in mm/day)-----")
    abs_stats = pixel_time_stats(
        dir_sd, dir_ts, start_date, end_date, x, y, True
    )
    logging.info("\n%s", abs_stats)

    logging.info("\n-----BIAS (in mm/day)-----")
    bias_stats = pixel_time_stats(
        dir_sd, dir_ts, start_date, end_date, x, y, False
    )
    logging.info("\n%s", bias_stats)

    logging.info("\n-----RMSE (in mm/day)-----")
    rmse_val = pixel_rmse(dir_sd, dir_ts, start_date, end_date, x, y)
    logging.info("value: %s", rmse_val)

    logging.info("\n-----R²-----")
    r2_val = pixel_r2(dir_sd, dir_ts, start_date, end_date, x, y)
    logging.info("value: %s", r2_val)


def print_correlation_errors_variable(
    dir_var: str,
    dir_sd: str,
    dir_ts: str,
    correlation: str,
    start_date: str,
    end_date: str,
    x: int,
    y: int,
    absolute: bool,
):
    """
    Calculates and returns Spearman or Pearson correlations between absolute
    or bias errors and ERA5-Land variables
    for a specified pixel over a given period

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product file
        are stored.
    dir_var : str
        Directory where the extra ERA5-Land variables product file
        are stored.
    correlation : str
        Type of correlation (`spearman` or `pearson`)
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)
    absolute: bool
        If True, displays the absolute error.
        If False, displays the bias error.
    """
    corr_tp = variable_correlation(
        dir_var,
        dir_sd,
        dir_ts,
        start_date,
        end_date,
        x,
        y,
        "tp",
        correlation,
        absolute,
    )
    corr_tp = variable_correlation(
        dir_var,
        dir_sd,
        dir_ts,
        start_date,
        end_date,
        x,
        y,
        "tp",
        correlation,
        absolute,
    )
    corr_sw = variable_correlation(
        dir_var,
        dir_sd,
        dir_ts,
        start_date,
        end_date,
        x,
        y,
        "sw",
        correlation,
        absolute,
    )
    corr_src = variable_correlation(
        dir_var,
        dir_sd,
        dir_ts,
        start_date,
        end_date,
        x,
        y,
        "src",
        correlation,
        absolute,
    )
    corr_sro = variable_correlation(
        dir_var,
        dir_sd,
        dir_ts,
        start_date,
        end_date,
        x,
        y,
        "sro",
        correlation,
        absolute,
    )
    error_name = "absolute" if absolute else "bias"
    logging.info(
        "###### %s correlation coefficients between %s error and the \n"
        "ERA5-Land variables from %s to %s for pixel [%s,%s] ######",
        correlation,
        error_name,
        start_date,
        end_date,
        x,
        y,
    )
    corr_table = pd.DataFrame(
        {
            variables["tp"]: [corr_tp],
            variables["sro"]: [corr_sro],
            variables["src"]: [corr_src],
            variables["sw"]: [corr_sw],
        }
    )
    logging.info("\n%s", corr_table)


def plot_et_time_comparison(dir_sd, dir_ts, start_date, end_date, x, y):
    """
    Displays the simulated and observed ET time series at
    a single pixel over a given period.

    Parameters
    ----------
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspirationproduct files
        are stored.
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)
    """
    time = xr.date_range(start_date, end=end_date, freq="1D")
    et_sd = et_sd_list(dir_sd, start_date, end_date, x, y)
    et_ts = et_ts_list(dir_ts, start_date, end_date, x, y)
    plt.plot(time, et_ts, color="red", label="Simulated ET")
    plt.plot(time, et_sd, color="blue", label="Observed ET")
    plt.title(f"Simulated ET vs Observed ET from {start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("ET (mm/day)")
    plt.legend()
    plt.show()


def plot_time_error_with_extra_variable(
    dir_var: str,
    dir_sd: str,
    dir_ts: str,
    variable: str,
    start_date: str,
    end_date: str,
    pix_x: int,
    pix_y: int,
    absolute: bool,
) -> None:
    """
    Displays the absolute or bias error with the selected ERA5-Land variable at
    a single pixel over a given period.

    Parameters
    ----------
    dir_var : str
        Directory where the observed extra ERA5-Land product files
        are stored.
    dir_sd : str
        Directory where the observed evapotranspiration product files
        are stored.
    dir_ts : str
        Directory where the simulated evapotranspiration product files
        are stored.
    variable : str
        The ERA5-Land variable to display (`tp`,`sro`,`src`,`sw`)
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.
    x: int
        First axis coordinate (UTM)
    y: int
        Second axis coordinate (UTM)
    absolute: bool
        If True, displays the absolute error.
        If False, displays the bias error.
    """
    errors, _, time = list_error(
        dir_sd, dir_ts, start_date, end_date, pix_x, pix_y, absolute=absolute
    )

    error_name = "absolute" if absolute else "bias"
    var_name = variables[variable]
    var_values = extra_variable_list(
        dir_var, variable, start_date, end_date, pix_x, pix_y
    )

    fig, ax1 = plt.subplots(figsize=(8, 6))
    x = mdates.date2num(time)

    ax1.plot(x, var_values, "-", color="blue", alpha=0.4, label=variable)
    ax1.set_ylabel(var_name, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    width = 0.25
    ax2.bar(x, errors, width=width, color="red", alpha=0.6, label=error_name)
    ax2.set_ylabel(f"{error_name} error (mm/day)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    plt.title(
        f"{var_name} vs {error_name} errors from {start_date} to {end_date} "
        f"for pixel [{pix_x},{pix_y}]"
    )
    plt.tight_layout()
    plt.show()


# ==============================================================
# Functions specific to new models under development
# ==============================================================


def plot_et_comparison(
    et_sd: list[float], et_ts: list[float], start_date: str, end_date: str
) -> None:
    """
    Plots observed and simulated ET at a given pixel over a specified period

    Parameters
    ----------
    et_sd : list[float]
        Observed evapotranspiration daily values
    et_ts : list[float]
        Simulated evapotranspiration daily values
    start_date : str
        Start date of the period in `YYYY-MM-DD` format.
    end_date : str
        End date of the period in `YYYY-MM-DD` format.
    """
    time = xr.date_range(start_date, end=end_date, freq="1D")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, et_sd, color="blue", label="Observed ET")
    ax.plot(time, et_ts, color="red", label="Simulated ET")
    ax.set_title(f"Simulated ET vs Observed ET from {start_date} to {end_date}")
    ax.set_xlabel("Date")
    ax.set_ylabel("ET (mm/day)")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_error_metrics(
    et_sd: list[float], et_ts: list[float], freq: int
) -> None:
    """
    Compute and returns the error metrics
    between observed and simulated evapotranspiration
    for a given period.

    Parameters
    ----------
    et_sd : list[float]
        Observed evapotranspiration daily values
    et_ts : list[float]
        Simulated evapotranspiration daily values
    freq: int
        Acquisition frequency, i.e., how often observed ET data are available.
    """
    et_ts_np = np.array(et_ts)
    et_sd_np = np.array(et_sd)
    mask = np.arange(len(et_ts_np)) % freq != 0
    et_ts_filt = et_ts_np[mask]
    et_sd_filt = et_sd_np[mask]
    rmse = root_mean_squared_error(et_sd_filt, et_ts_filt)
    mae = mean_absolute_error(et_sd_filt, et_ts_filt)
    mbe = (et_ts_np - et_sd_np).mean()
    r2 = r2_score(et_sd_filt, et_ts_filt)
    logging.info(
        "RMSE: %.4f, MAE: %.4f, MBE: %.4f, R²: %.4f",
        rmse,
        mae,
        mbe,
        r2,
    )
