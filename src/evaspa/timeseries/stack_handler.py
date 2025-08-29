# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for stacking data
"""

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from evaspa.common import filter
from evaspa.common.solar import compute_daily_toa_solar_radiation
from evaspa.timeseries.constant import TimeSeriesVar as TSVar


class TimeSeriesStackConfig(BaseModel):
    """
    Configuration for parameters for time series computation
    """

    model_config = ConfigDict(extra="forbid")

    et_single_date_filtering: filter.FilteringConfig = Field(
        default=filter.FilteringConfig({})
    )


def fill_radiation_missing(data: xr.DataArray) -> xr.DataArray:
    """
    Compute radiation at a missing date
    """
    date = pd.to_datetime(data["time"].item()).date()
    x = data.coords["x"]
    y = data.coords["y"]
    arr = compute_daily_toa_solar_radiation(
        date=date,
        x=x,
        y=y,
        crs=data.attrs.get("crs", None),
        slope=None,
        aspect=None,
    )

    return xr.DataArray(arr, coords=data.coords, dims=data.dims)


def stack_time_series(
    et_time_series: xr.Dataset, radiation_time_series: xr.Dataset
):
    """
    Create stack
    Fill the second dataset (ds2) to match the time dimension of the first dataset (ds1)
    using a custom fill function.

    Parameters:
    ds1 (xarray.Dataset): The first dataset with the desired time dimension.
    ds2 (xarray.Dataset): The second dataset to be filled.
    fill_func (callable): A custom function that takes a time index and returns the filled value.

    Returns:
    xarray.Dataset: The filled second dataset aligned with the first dataset's time dimension.
    """
    # Create a stack if radiation do not exist compute theoritical
    # use_topo
    # flags
    # use data test with spatial coordinates
    # Reindex radiation time series to match the time dimension of ET time series
    aligned_radiation_ts = radiation_time_series.reindex(
        {TSVar.TIME.value: et_time_series[TSVar.TIME.value]}, method=None
    )
    # Iterate over the time dimension and fill missing values using the custom function
    for time in aligned_radiation_ts[TSVar.TIME.value]:
        if (
            aligned_radiation_ts[TSVar.RADIATION.value]
            .sel(time=time)
            .isnull()
            .all()
            .item()
        ):
            aligned_radiation_ts[TSVar.RADIATION.value].loc[
                {TSVar.TIME.value: time}
            ] = fill_radiation_missing(
                aligned_radiation_ts[TSVar.RADIATION.value].sel(time=time)
            )

    return et_time_series.assign(
        {TSVar.RADIATION.value: aligned_radiation_ts[TSVar.RADIATION.value]}
    )


def run(
    et_time_series: xr.Dataset,
    radiation_time_series: xr.Dataset | None,
    et_single_date: xr.Dataset | None,
    et_single_date_filtering: dict,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """
    Filter and stack time series data
    """
    # Process ET single date
    et_sd = None
    if et_single_date is not None:
        # Filter ET single date
        valid, _ = filter.find_valid_pixels(
            et_single_date,
            nan_config=[TSVar.ET.value],
            valid_config=et_single_date_filtering,
        )
        et_sd = et_single_date.assign({TSVar.VALID.value: valid})
    # Process time series
    if radiation_time_series is None:
        radiation_xr = xr.full_like(
            et_time_series[TSVar.ET.value], fill_value=np.nan
        ).rename(TSVar.RADIATION.value)
        radiation_time_series = xr.Dataset(
            {TSVar.RADIATION.value: radiation_xr}
        )
    return stack_time_series(et_time_series, radiation_time_series), et_sd
