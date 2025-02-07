# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import datetime as dt

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from evaspa import solar
from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

# Constant
SOLAR_FLUX = 1367  # W.m-2


class DailyConfig(BaseModel):
    """
    Daily config for ET processing
    """

    model_config = ConfigDict(extra="forbid")

    method: str = Field(default="toa")
    use_topo: bool = Field(default=False)


def toa_daily_estimate(
    data: xr.Dataset, date: dt.datetime, dem: xr.Dataset | None = None
) -> xr.Dataset:
    """
    Description
    -----------
    The instantaneous value is transformed
    into a daily value by considering a scaling factor.
    This facor is equal to the ratio between daily downwelling
    shortwave radiation and between instant downwelling shortwave
    radiation.
    In this method the shortwave radiation is estimate at
    top-of-atmosphere.

    Parameters
    ----------
    data: xr.Dataset
        Instantaneous data
    date: dt.datetime
        Date time

    Returns
    -------
    daily: xr.DataArray
        Daily extrapolated data
    """
    # DEM
    slope = None
    aspect = None
    if dem is not None:
        if dem.get("slope", None) is None or dem.get("aspect", None) is None:
            logger.warning(
                "No DEM information (aspect or slope) to compute topographic corrections. Topographic corrections are disabled."
            )
        else:
            slope = dem["slope"]
            aspect = dem["aspect"]
    # Get lon/lat coordinates
    crs = data.attrs.get("crs", None)
    if crs is None:
        msg = "Impossible to compute TOA extrapolation because CRS is missing in the metadata"
        raise ValueError(msg)
    # Initiate dataset
    daily = data.copy(data=None)
    daily.attrs = data.attrs.copy()
    # Compute TOA solar radiation
    toa_inst = solar.compute_toa_solar_radiation(
        date,
        daily.coords["x"],
        daily.coords["y"],
        daily.attrs["crs"],
        slope=slope,
        aspect=aspect,
    )
    toa_daily = solar.compute_daily_toa_solar_radiation(
        date,
        daily.coords["x"],
        daily.coords["y"],
        daily.attrs["crs"],
        slope=slope,
        aspect=aspect,
    )
    # Compute ratio
    for var in daily.data_vars:
        daily[var] = data[var] * toa_daily / toa_inst
    return daily


def extrapolate_at_daily_scale(
    data: xr.Dataset,
    variables: list[str] | None = None,
    dem: xr.Dataset | None = None,
    method: str = "toa",
    use_topo: bool = False,
) -> xr.Dataset:
    """
    Description
    -----------
    The instantaneous value of ET obtained previously is transformed
    in a daily value by considering a scaling factor and the ratio between
    instant downwelling shortwave radiation and daily downwelling
    shortwave radiation.
    The ratio used depend on the method chosen.
    Only data in the variable list is extrapolated.
    If the list is empty, all data are extrapolated.

    Parameters
    ----------
    data: xr.Dataset
        Instantaneous data
    variables: list[str]
        List of variables to extrapolate.
    dem: xr.Dataset
        DEM data
    method: str
        Method used for extrapolation (default: toa)
    use_topo: bool
        Use topographic corrections

    Returns
    -------
    daily: xr.DataArray
        Daily extrapolated data
    """
    if use_topo and dem is None:
        logger.warning(
            "No DEM information to compute topographic corrections. Topographic corrections are disabled."
        )
    if not use_topo:
        dem = None
    # Data selection
    if variables is None:
        keep = list(data.data_vars)
    else:
        keep = list(filter(lambda x: x in data.data_vars, variables))
        not_keep = list(filter(lambda x: x not in data.data_vars, variables))
        if len(not_keep) != 0:
            msg = f"Variables {not_keep} not available for daily extrapolation"
            logger.warning(msg)
    msg = f"Daily extrapolation performed on {keep}"
    logger.debug(msg)
    # Extrapolation
    if method.lower() == "toa":
        if data.attrs.get("date", None) is None:
            msg = "Impossible to extrapolate because the date is missing in metadata"
            raise ValueError(msg)
        daily = toa_daily_estimate(
            data=data[keep], date=data.attrs["date"], dem=dem
        )
    else:
        msg = f"Extrapolation method {method} unknown"
        logger.error(msg)
        daily = data.copy(data=None)
        daily.attrs = data.attrs.copy()
        for var in keep:
            daily[var].data = np.nan * np.ones_like(daily["var"].data)
    return daily
