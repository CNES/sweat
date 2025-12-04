import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from stats import (
    error_date_list,
    get_et_single_date,
    get_et_time_series,
    mae_date,
    mae_pixel,
    mbe_date,
    mbe_pixel,
    r2_date,
    r2_pixel,
    rmse_date,
    rmse_pixel,
)

metrics = {
    "MAE": [mae_date, mae_pixel],
    "MBE": [mbe_date, mbe_pixel],
    "RMSE": [rmse_date, rmse_pixel],
    "R²": [r2_date, r2_pixel],
}


def plot_et_comparison(dir_sd: str, dir_ts: str, date: str):
    date_dt = dt.datetime.strptime(date, "%Y-%m-%d")  # noqa: DTZ007
    et_sd = get_et_single_date(dir_sd, date_dt)
    et_ts = get_et_time_series(dir_ts, date_dt)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    et_sd.plot(ax=axes[0])
    axes[0].set_title(f"Observed ET on {date}")
    et_ts.plot(ax=axes[1])
    axes[1].set_title(f"Simulated ET on {date}")
    diff = et_sd - et_ts
    diff.plot(ax=axes[2])
    axes[2].set_title("Difference")
    plt.tight_layout()
    plt.show()


def plot_spatial_distribution_error(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str
):
    mbe_pxl = mbe_pixel(dir_sd, dir_ts, start_date, end_date)
    mae_pxl = mae_pixel(dir_sd, dir_ts, start_date, end_date)
    rmse_pxl = mae_pixel(dir_sd, dir_ts, start_date, end_date)
    r2_pxl = r2_pixel(dir_sd, dir_ts, start_date, end_date)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Error Metrics per Pixel from {start_date} to {end_date}", fontsize=18
    )
    mae_pxl.plot(ax=axes[0, 0])
    axes[0, 0].set_title("Mean Absolute Error")
    mbe_pxl.plot(ax=axes[0, 1])
    axes[0, 1].set_title("Mean Bias Error")
    rmse_pxl.plot(ax=axes[1, 0])
    axes[1, 0].set_title("Root Mean Squared Error")
    r2_pxl.plot(ax=axes[1, 1])
    axes[1, 1].set_title("Coefficient of Determination")
    plt.tight_layout()
    plt.show()


def plot_time_evolution_error(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str
):
    mae_list = error_date_list(dir_sd, dir_ts, start_date, end_date, mae_date)
    mbe_list = error_date_list(dir_sd, dir_ts, start_date, end_date, mbe_date)
    rmse_list = error_date_list(dir_sd, dir_ts, start_date, end_date, rmse_date)
    r2_list = error_date_list(dir_sd, dir_ts, start_date, end_date, r2_date)

    time = [
        pd.Timestamp(d).to_pydatetime()
        for d in xr.date_range(start_date, end_date)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Evolution of daily error metrics from {start_date} to {end_date}",
        fontsize=18,
    )
    axes[0, 0].bar(time, mae_list, color="blue")
    axes[0, 0].set_title("MAE")

    colors = ["green" if v >= 0 else "red" for v in mbe_list]
    axes[0, 1].bar(time, mbe_list, color=colors)
    axes[0, 1].set_title("MBE")

    axes[1, 0].bar(time, rmse_list, color="orange")
    axes[1, 0].set_title("RMSE")

    axes[1, 1].scatter(time, r2_list, color="blue", marker="o")
    axes[1, 1].axhline(
        y=1, color="red", linestyle="--", linewidth=1.5, label="y=1"
    )
    axes[1, 1].set_title("R²")
    axes[1, 1].legend()

    for ax in axes.flatten():
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.show()


def plot_daily_error_distribution(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, metric_name: str
):
    metric, _ = metrics[metric_name]
    errors = error_date_list(dir_sd, dir_ts, start_date, end_date, metric)
    if metric_name == "R²":
        filtered_errors = [e for e in errors if e != 1]
    else:
        filtered_errors = [e for e in errors if e != 0]
    plt.figure(figsize=(8, 5))
    plt.hist(filtered_errors, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of daily {metric_name}")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def plot_pixel_error_distribution(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, metric_name: str
):
    _, metric = metrics[metric_name]
    errors = metric(dir_sd, dir_ts, start_date, end_date)
    plt.figure(figsize=(8, 5))
    plt.hist(errors.values.flatten(), color="skyblue", edgecolor="black")
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
        coord_crit = data.where(data == data.median(), drop=True).coords
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
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, metric: callable
):
    # time = xr.date_range(start_date, end_date)
    errors = metric(dir_sd, dir_ts, start_date, end_date)
    x_min, y_min = find_pixel(errors, "min")
    min = errors.sel(x=x_min, y=y_min).item()
    x_max, y_max = find_pixel(errors, "max")
    max = errors.sel(x=x_max, y=y_max).item()
    x_med, y_med = find_pixel(errors, "median")
    median = errors.sel(x=x_med, y=y_med).item()
    table = pd.DataFrame(
        {
            "Min": [min],
            "Argmin": [(round(x_min), round(y_min))],
            "Max": [max],
            "Argmax": [(round(x_max), round(y_max))],
            "Median": [median],
            "Argmedian": [(round(x_med), round(y_med))],
        }
    )
    return table


def print_spatial_statistics(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str
):
    print(
        f"######## {start_date}-{end_date} PERIOD SPATIAL STATISTICS ########"
    )
    print("-----MAE-----")
    print(spatial_stats_metric(dir_sd, dir_ts, start_date, end_date, mae_pixel))
    print("\n")
    print("-----MBE-----")
    print(spatial_stats_metric(dir_sd, dir_ts, start_date, end_date, mbe_pixel))
    print("\n")
    print("-----RMSE-----")
    print(
        spatial_stats_metric(dir_sd, dir_ts, start_date, end_date, rmse_pixel)
    )
    print("\n")
    print("-----R²-----")
    print(spatial_stats_metric(dir_sd, dir_ts, start_date, end_date, r2_pixel))


def plot_time_evolution_pixel(
    dir_sd: str,
    dir_ts: str,
    start_date: str,
    end_date: str,
    abs: bool | None = False,
):
    metric = mbe_pixel if abs else mae_pixel
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
    if abs:
        errors_min = np.abs(errors_min)
        errors_max = np.abs(errors_max)
        errors_median = np.abs(errors_median)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        time,
        errors_min,
        label=f"Best Mean Pixel\n[{round(x_min)}, {round(y_min)}]",
        s=40,
    )
    plt.scatter(
        time,
        errors_max,
        label=f"Worst Mean Pixel\n[{round(x_max)}, {round(y_max)}]",
        s=40,
    )
    plt.scatter(
        time,
        errors_median,
        label=f"Median Mean Pixel\n[{round(x_med)}, {round(y_med)}]",
        s=40,
    )
    description = "Absolute Errors" if abs else "Errors"
    plt.xlabel("Date")
    plt.ylabel(description)
    plt.title(
        f"Time evolution of {description} for selected pixels from {start_date} to {end_date}"
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
    date: dt.datetime, dir_sd: str, dir_ts: str, abs: bool | None = False
):
    et_sd = get_et_single_date(dir_sd, date)
    et_ts = get_et_time_series(dir_ts, date)

    ae = np.abs(et_sd - et_ts) if abs else et_sd - et_ts
    mae_val = mae_date(et_sd, et_ts)

    ae_min = ae.min().item()
    ae_min_coords = find_pixel(ae, "min")

    ae_max = ae.max().item()
    ae_max_coords = find_pixel(ae, "max")

    ae_med = ae.median().item()
    ae_med_coords = find_pixel(ae, "median")
    table = pd.DataFrame(
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
    return table


def print_daily_statistics(date: str, dir_sd: str, dir_ts: str):
    date_dt = dt.datetime.strptime(date, "%Y-%m-%d")  # noqa: DTZ007
    print(f"######## {date} SPATIAL STATISTICS ########")
    print("-----ABSOLUTE ERROR-----")
    print(daily_error_table(date_dt, dir_sd, dir_ts, abs=True))
    print("\n")
    print("-----BIAS ERROR-----")
    print(daily_error_table(date_dt, dir_sd, dir_ts))
    print("\n")
    print("-----RMSE-----")
    print(
        "value: ",
        rmse_date(
            get_et_single_date(dir_sd, date_dt),
            get_et_time_series(dir_ts, date_dt),
        ),
    )
    print("\n")
    print("-----R²-----")
    print(
        "value: ",
        r2_date(
            get_et_single_date(dir_sd, date_dt),
            get_et_time_series(dir_ts, date_dt),
        ),
    )


def time_stats_metric(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str, metric: callable
):
    time = xr.date_range(start_date, end_date)
    errors = np.array(
        error_date_list(dir_sd, dir_ts, start_date, end_date, metric)
    )
    if metric == r2_date:
        filtered_errors = np.where(errors != 1, errors, -1)
        argmax = filtered_errors.argmax()
        max = filtered_errors.max()
        argmin = errors.argmin()
        min = errors.min()
    else:
        filtered_errors = np.where(errors != 0, errors, np.inf)
        argmin = filtered_errors.argmin()
        min = filtered_errors.min()
        argmax = errors.argmax()
        max = errors.max()
        argmin = filtered_errors.argmin()
    date_max = time[argmax].strftime("%Y-%m-%d")
    date_min = time[argmin].strftime("%Y-%m-%d")
    arg1 = "Min" if metric in (r2_date, mbe_date) else "Min (except 0)"
    arg2 = (
        "Argmin"
        if metric in (r2_date, mbe_date)
        else "Argmin (except acquisition day)"
    )
    arg3 = "Max (except 1)" if metric == r2_date else "Max"
    arg4 = "Argmax (except acquisition day)" if metric == r2_date else "Argmax"
    table = pd.DataFrame(
        {
            arg1: [min],
            arg2: [date_min],
            arg3: [max],
            arg4: [date_max],
        }
    )
    return table


def print_time_statistics(
    dir_sd: str, dir_ts: str, start_date: str, end_date: str
):
    print(f"######## {start_date}-{end_date} PERIOD TIME STATISTICS ########")
    print("-----MAE-----")
    print(time_stats_metric(dir_sd, dir_ts, start_date, end_date, mae_date))
    print("\n")
    print("-----MBE-----")
    print("\n")
    print("-----RMSE-----")
    print(time_stats_metric(dir_sd, dir_ts, start_date, end_date, rmse_date))
    print("\n")
    print("-----R²-----")
    print(time_stats_metric(dir_sd, dir_ts, start_date, end_date, r2_date))
