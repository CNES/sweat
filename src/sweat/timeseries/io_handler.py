# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for managing IO for time series
- read time series
- write time series
"""

import os
import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

import sweat.timeseries.status_handler as sh
from sweat.common.io import read_data_from_file, write_dataset
from sweat.timeseries.constant import TimeSeriesVar as TSVar


class TimeSeriesInputConfig(BaseModel):
    """
    Configuration for parameters for time series computation
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    et_time_series: list[str] = Field(default=[])
    dates: list[pd.Timestamp] | None = Field(default=None)
    et_single_date: list[str] = Field(default=[])
    radiation: list[str] = Field(default=[])
    dem: str | None = Field(default=None)

    @field_validator("dates", mode="before")
    @classmethod
    def convert_and_sort_dates(cls, v: Any) -> list[pd.Timestamp]:
        """
        Convert dates from string to timestamp and sort them
        """
        # Convert each value to pandas.Timestamp
        timestamps = [pd.Timestamp(item) for item in v]
        # Sort the list
        return sorted(timestamps)

    @field_validator("et_time_series", "radiation", "et_single_date")
    @classmethod
    def test_path(cls, files: list[str]) -> list[str]:
        """
        Check file paths
        """
        for file in files:
            if not os.path.exists(file):
                msg = f"Path not found: {file}"
                raise OSError(msg)
        return files

    @field_validator("dem")
    @classmethod
    def test_dem(cls, file: str) -> str:
        """
        Check DEM path
        """
        if file is not None and not os.path.exists(file):
            msg = f"Path to DEM not found: {file}"
            raise OSError(msg)
        return file

    @model_validator(mode="before")
    @classmethod
    def check_files(cls, data: Any) -> Any:
        """
        Check that all file lists are not empty
        """
        if (
            isinstance(data, dict)
            and len(data["et_time_series"]) == 0
            and len(data["radiation"]) == 0
            and len(data["et_single_date"]) == 0
        ):
            msg = "Not all file lists must be empty"
            raise ValueError(msg)
        return data


def write_timeseries(
    data: xr.Dataset,
    root_name: str = "timeseries",
    directory: str = os.getcwd(),
) -> None:
    """
    Write time series. For each date, write a dataset.

    Parameters
    ----------
    data: xr.Dataset
        Dataset to write
    root_name: str
        Root used for filenames
    directory: str
        Path to the directory
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

    Parameters
    ----------
    filename: str
        Filename containing a date (format YYYYMMDD)

    Returns
    -------
    date: pd.Timestamp
        Date
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


def read_time_series(filenames: list[str]) -> xr.Dataset:
    """
    Read time series

    Parameters
    ----------
    filenames: list[str]
        List of filenames

    Returns
    -------
    xarr: xr.Dataset
        Data
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


def read_et_time_series(
    filenames: list[str], dates: list[pd.Timestamp] | None
) -> xr.Dataset:
    """
    Read time series

    Parameters
    ----------
    filenames: list[str]
        List of filenames
    dates: list[pd.Timestamp]
        List of dates to consider in the time series

    Returns
    -------
    xarr: xr.Dataset
        Data
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
    # Fill missing dates
    if dates is not None:
        filled_et = ts[TSVar.ET.value].reindex(
            {TSVar.TIME.value: dates}, fill_value=np.nan
        )
        filled_flags = (
            ts[TSVar.FLAGS.value]
            .reindex({TSVar.TIME.value: dates}, fill_value=sh.INIT_STATUS)
            .astype(sh.STATUS_TYPE)
        )
        # Combine into new dataset
        return xr.Dataset(
            {
                TSVar.ET.value: filled_et,
                TSVar.FLAGS.value: filled_flags,
            },
            attrs=ts.attrs.copy(),
        )
    return ts


def read_radiation_time_series(filenames: list[str]) -> xr.Dataset:
    """
    Read radiation time series

    Parameters
    ----------
    filenames: list[str]
        List of filenames

    Returns
    -------
    xarr: xr.Dataset
        Data
    """
    # Read the time series
    ts = read_time_series(filenames)
    # Convert type
    ts[TSVar.RADIATION.value] = ts[TSVar.RADIATION.value].astype("float32")
    return ts


def read_et_single_date(filenames: list[str]) -> xr.Dataset:
    """
    Read ET products

    Parameters
    ----------
    filenames: list[str]
        List of filenames

    Returns
    -------
    xarr: xr.Dataset
        Data
    """
    # Read the time series
    ts = read_time_series(filenames)
    # Convert type
    ts[TSVar.ET.value] = ts[TSVar.ET.value].astype("float32")
    if TSVar.FLAGS.value not in ts.data_vars:
        ts[TSVar.FLAGS.value] = xr.zeros_like(
            ts[TSVar.ET.value], dtype=sh.STATUS_TYPE
        )
    else:
        ts[TSVar.FLAGS.value] = ts[TSVar.FLAGS.value].astype(sh.STATUS_TYPE)
    return ts


def read_input(
    config: dict,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset | None, xr.Dataset | None]:
    """
    Read input data for time series

    Parameters
    ----------
    config: dict
        Input information

    Returns
    -------
    et_ts: xr.Dataset
        ET time series
    radiation_ts: xr.Dataset
        Radiation time series
    et_sd: xr.Dataset
        ET single dates
    dem: xr.Dataset
        DEM
    """
    # Configuration
    input_cfg = TimeSeriesInputConfig.model_validate(config)
    # Read
    et_ts = None
    if len(input_cfg.et_time_series) > 0:
        et_ts = read_et_time_series(input_cfg.et_time_series, input_cfg.dates)
    radiation_ts = None
    if len(input_cfg.radiation) > 0:
        radiation_ts = read_radiation_time_series(input_cfg.radiation)
    et_sd = None
    if len(input_cfg.et_single_date) > 0:
        et_sd = read_et_single_date(input_cfg.et_single_date)
    if et_ts is None:
        base = radiation_ts if radiation_ts is not None else et_sd
        if base is None:
            msg = "Not all file lists must be empty"
            raise ValueError(msg)
        if input_cfg.dates is not None:
            dates = input_cfg.dates
        else:
            msg = "No dates provided"
            raise ValueError(msg)
        original_dims = list(base.sizes.keys())
        spatial_dims = [x for x in original_dims if x != TSVar.TIME.value]
        x1 = spatial_dims[0]  # y
        x2 = spatial_dims[1]  # x
        # Create NaN-filled and invalid status data arrays
        nan_data = np.full((len(dates), base.sizes[x1], base.sizes[x2]), np.nan)
        invalid_data = np.full(
            (len(dates), base.sizes[x1], base.sizes[x2]),
            sh.INIT_STATUS,
            dtype=sh.STATUS_TYPE,
        )
        # Create tthe dataset
        et_ts = xr.Dataset(
            {
                TSVar.ET.value: ((TSVar.TIME.value, x1, x2), nan_data.copy()),
                TSVar.FLAGS.value: (
                    (TSVar.TIME.value, x1, x2),
                    invalid_data.copy(),
                ),
            },
            coords={
                x2: base.coords[x2],
                x1: base.coords[x1],
                TSVar.TIME.value: dates,
            },
        )
    dem = None
    if input_cfg.dem is not None:
        dem = read_data_from_file(input_cfg.dem)
    return et_ts, radiation_ts, et_sd, dem
