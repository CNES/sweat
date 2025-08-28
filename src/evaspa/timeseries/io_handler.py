# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for managing IO for time series
- read time series
- write time series
"""

import os
import re

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, field_validator

import evaspa.timeseries.status_handler as sh
from evaspa.common.io import read_data_from_file, write_dataset
from evaspa.timeseries.constant import TimeSeriesVar as TSVar


class TimeSeriesInputConfig(BaseModel):
    """
    Configuration for parameters for time series computation
    """

    model_config = ConfigDict(extra="forbid")

    et_time_series: list[str]
    radiation: list[str] = Field(default=[])
    et_single_date: list[str] = Field(default=[])

    @field_validator("et_time_series", "radiation", "et_single_date")
    @classmethod
    def test_path(cls, files: list[str]) -> list[str]:
        for file in files:
            if not os.path.exists(file):
                msg = f"Path not found: {file}"
                raise OSError(msg)
        return files

    @field_validator("et_time_series")
    @classmethod
    def test_timeseries(cls, files: list[str]) -> list[str]:
        if len(files) == 0:
            msg = "No files to read for ET time series"
            raise ValueError(msg)
        return files


def write_timeseries(
    data: xr.Dataset,
    root_name: str = "timeseries",
    directory: str = os.getcwd(),
) -> None:
    """
    Write time series. For each date, write a dataset.
    """
    for date, data_by_date in data.groupby("time"):
        # Convert date to string
        pd_date = pd.to_datetime(str(date))
        str_date = pd_date.strftime("%Y%m%d")
        filename = f"{root_name}_{str_date}.tif"
        # Drop time
        xy_data = data_by_date.sel(time=date, drop=True)
        # Write dataset
        write_dataset(
            xy_data, filename=filename, directory=directory, separate=False
        )


def extract_date_from_filename(filename: str) -> pd.Timestamp:
    """
    Extract date from filename
    """
    pattern = r".*_([0-9]{8})\..*"
    m = re.match(pattern, filename)
    if m is not None:
        str_date = m.group(1)
        try:
            return pd.to_datetime(str_date, format="%Y%m%d")
        except ValueError as exc:
            msg = f"Wrong date format for file {filename}"
            raise ValueError(msg) from exc
    msg = f"Wrong format for file {filename}"
    raise ValueError(msg)


def read_time_series(filenames: list[str]):
    """
    Read time series
    """
    if len(filenames) == 0:
        msg = "No files to read"
        raise ValueError(msg)
    # Step 1: Read each file into an xarray dataset
    datasets = []
    times = []

    # Sort files to ensure chronological order
    for filename in sorted(filenames):
        # Extract date from filename
        date = extract_date_from_filename(filename)

        # Load dataset
        ds = read_data_from_file(filename)

        # Add a new time dimension (length 1)
        ds = ds.expand_dims(time=[date])

        datasets.append(ds)
        times.append(date)

    # Step 2: Concatenate all datasets along the time dimension
    return xr.concat(datasets, dim="time")


def read_et_time_series(filenames: list[str]):
    """
    Read time series
    """
    # Read the time series
    ts = read_time_series(filenames)
    # Convert type
    ts[TSVar.ET.value] = ts[TSVar.ET.value].astype("float32")
    ts[TSVar.FLAGS.value] = ts[TSVar.FLAGS.value].astype(sh.STATUS_TYPE)
    # Update status to remove is_updated flag
    ts[TSVar.FLAGS.value] = xr.apply_ufunc(
        np.vectorize(sh.update_status),
        ts[TSVar.FLAGS.value],
        kwargs={
            "updated": False,
        },
        vectorize=True,
    )
    return ts


def read_radiation_time_series(filenames: list[str]):
    """
    Read radiation time series
    """
    # Read the time series
    ts = read_time_series(filenames)
    # Convert type
    ts[TSVar.RADIATION.value] = ts[TSVar.RADIATION.value].astype("float32")
    if TSVar.FLAGS.value not in ts.data_vars:
        ts[TSVar.FLAGS.value] = xr.zeros_like(ts[TSVar.RADIATION.value])
    else:
        ts[TSVar.FLAGS.value] = ts[TSVar.FLAGS.value].astype(sh.STATUS_TYPE)
    return ts


def read_et_single_date(filenames: list[str]):
    """
    Read ET products
    """
    # Read the time series
    ts = read_time_series(filenames)
    # Convert type
    ts[TSVar.ET.value] = ts[TSVar.ET.value].astype("float32")
    if TSVar.FLAGS.value not in ts.data_vars:
        ts[TSVar.FLAGS.value] = xr.zeros_like(ts[TSVar.ET.value])
    else:
        ts[TSVar.FLAGS.value] = ts[TSVar.FLAGS.value].astype(sh.STATUS_TYPE)
    return ts


def read_input(
    config: dict,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset | None]:
    """
    Read input data for time series
    """
    # Configuration
    input_cfg = TimeSeriesInputConfig.model_validate(config)
    # Read
    et_ts = read_et_time_series(input_cfg.et_time_series)
    radiation_ts = None
    if len(input_cfg.radiation) > 0:
        radiation_ts = read_radiation_time_series(input_cfg.radiation)
    et_sd = None
    if len(input_cfg.et_single_date) > 0:
        et_sd = read_et_time_series(input_cfg.et_single_date)
    return et_ts, radiation_ts, et_sd
