# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for managing sliding windows for time series
"""

import datetime as dt
import json
import os

import xarray as xr


def window_generator(
    period_start: dt.datetime, period_end: dt.datetime, window: int, shift: int
):
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

    Yield
    -------
    Tuple[dt.datetime, dt.datetime]
        Start and end dates of each sliding window.
    """
    window_start = period_start
    window_end = period_start
    k = 1
    stop = False
    while not stop:
        # Inital window must contains only the period start date
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
    json_dir: str | None = None,
    verbose: bool | None = False,
):
    """
    Create configuration for a given window

    Parameters
    ----------
    wdw_start: dt.datetime
        Window start date
    wdw_end: dt.datetime
        Window end date
    et_single_date_dir: str
        Directory where et_single_date .tif files are downloaded
    radiation_dir: str
        Directory where radiation .tif files are downloaded
    et_time_series_dir: str
        Directory where et_time_series .tif files are downloaded
    json_dir: str
        Directory to store the json configuration file if verbose True
        (default: current directory)
    verbose: bool
        If true, the configuration dictionary will be stored as a .json file

    Return
    -------
    config_dict: dict
        Configuration dictionary
    """
    time = xr.date_range(window_start, freq="1D", end=window_end)
    config_dict: dict = {
        "input": {
            "dates": [],
            "et_time_series": [],
            "radiation": [],
            "et_single_date": [],
        },
        "output": {"path": et_time_series_dir},
        "params": {},
    }
    for date in time:
        date_str = date.strftime("%Y-%m-%d")
        config_dict["input"]["dates"].append(date_str)
        radiation_path = os.path.join(
            radiation_dir, f"radiation_{date.strftime('%Y%m%d')}.tif"
        )
        if os.path.isfile(radiation_path):  # Check if the radiation file exists
            config_dict["input"]["radiation"].append(radiation_path)
        et_single_date_path = os.path.join(
            et_single_date_dir, f"et_single_date_{date.strftime('%Y%m%d')}.tif"
        )
        if os.path.isfile(
            et_single_date_path
        ):  # Check if the et_single_date file exists
            config_dict["input"]["et_single_date"].append(et_single_date_path)
        et_time_series_path = os.path.join(
            et_time_series_dir, f"et_time_series_{date.strftime('%Y%m%d')}.tif"
        )
        if os.path.isfile(
            et_time_series_path
        ):  # Check if the et_time_series file exists
            config_dict["input"]["et_time_series"].append(et_time_series_path)
    if verbose:
        if json_dir is None:
            json_dir = os.getcwd()
        os.makedirs(json_dir, exist_ok=True)
        date_min = time[0].strftime("%Y%m%d")
        date_max = time[-1].strftime("%Y%m%d")
        filename = f"config_{date_min}-{date_max}.json"
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=4)
    return config_dict
