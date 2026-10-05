# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for daily extrapolation
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Self

import numpy as np
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from sweat.common import solar
from sweat.common.constant import FLAGS_TYPE, MSK_PROCESSING_FAILED
from sweat.common.types import ETVar
from sweat.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

# Constant
SOLAR_FLUX = 1367  # W.m-2


class DailyConfig(BaseModel):
    """
    Daily config for ET processing
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["toa", "geo"] = Field(default="toa")
    use_topo: bool = Field(default=False)
    name: str | None = None

    @model_validator(mode="after")
    def validate_method_parameters(self) -> Self:
        if self.method == "toa" and self.name is not None:
            msg = "The 'name' parameter is not used with daily method='toa'"
            logger.warning(msg)
        if self.method == "geo" and self.use_topo:
            msg = "The 'use_topo' parameter is not used with daily method='geo'"
            logger.warning(msg)
        return self

    @model_serializer(mode="wrap")
    def serialize_model(self, serializer: SerializerFunctionWrapHandler):
        # Get the default serialization
        data = serializer(self)

        # Remove name if method toa is selected
        if self.method == "toa":
            data.pop("name", None)

        # Remove use_topo if method geo is selected
        if self.method == "geo":
            data.pop("use_topo", None)

        return data


def toa_daily_estimate(
    data: xr.Dataset, date: dt.datetime, dem: xr.Dataset | None = None
) -> xr.Dataset:
    """
    Estimate daily extrapolation using toa solar radiation as ratio.

    Notes
    -----
    The instantaneous value is transformed
    into a daily value by considering a scaling factor.
    This factor is equal to the ratio between daily downwelling
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
    dem: xr.Dataset | None
        DEM data

    Returns
    -------
    daily: xr.Dataset
        Daily extrapolated data
    """
    # DEM
    slope = None
    aspect = None
    if dem is not None:
        if dem.get("slope", None) is None or dem.get("aspect", None) is None:
            msg = (
                "No DEM information (aspect or slope) to compute topographic "
                "corrections. Topographic corrections are disabled."
            )
            logger.warning(msg)
        else:
            slope = dem["slope"]
            aspect = dem["aspect"]
    # Get CRS
    crs = data.attrs.get("crs", None)
    if crs is None:
        msg = (
            "Impossible to compute TOA extrapolation because "
            "CRS is missing in the metadata"
        )
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
    # TODO: Maybe values directly must be used to update
    for var in daily.data_vars:
        daily[var] = data[var] * toa_daily / toa_inst
    return daily


def geo_daily_estimate(
    data: xr.Dataset, inst_rsd: xr.DataArray, daily_rsd: xr.DataArray
) -> xr.Dataset:
    """
    Estimate daily extrapolation using instantaneous and daily product.

    Parameters
    ----------
    data: xr.Dataset
        Instantaneous data
    inst_rsd: xr.DataArray
        Instantaneous radiation product
    daily_rsd: xr.DataArray
        Daily radiation product

    Returns
    -------
    daily: xr.Dataset
        Daily extrapolated data
    """
    # Initiate dataset
    daily = data.copy(data=None)
    daily.attrs = data.attrs.copy()
    # Compute ratio
    for var in daily.data_vars:
        daily[var] = data[var] * daily_rsd / inst_rsd
    return daily


def extrapolate_at_daily_scale(
    data: xr.Dataset,
    variables: list[str] | None = None,
    extra: xr.Dataset | None = None,
    method: str = "toa",
    use_topo: bool = False,
    name: str | None = None,
) -> xr.Dataset:
    """
    Extrapolate data variables at daily scale.

    Notes
    -----
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
    extra: xr.Dataset
        Additional data used to compute the ratio
    method: str
        Method used for extrapolation (default: toa)
    use_topo: bool
        Use topographic corrections
    name: bool
        Name to use for radiation product

    Returns
    -------
    daily: xr.DataArray
        Daily extrapolated data
    """
    # Check input dataset
    if len(data.data_vars) == 0:
        msg = "EF dataset empty"
        raise ValueError(msg)
    # Extract DEM data from extra data
    dem = None
    if extra is not None:
        dem_vars = [
            d
            for d in extra.data_vars
            if d in [ETVar.HEIGHT.value, ETVar.SLOPE.value, ETVar.ASPECT.value]
        ]
        if len(dem_vars) > 0:
            dem = extra[dem_vars]
    if use_topo and dem is None:
        msg = (
            "No DEM information to compute topographic corrections. "
            "Topographic corrections are disabled."
        )
        logger.warning(msg)
    # Use topography
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
    # Get valid and flags
    if ETVar.VALID.value in data.data_vars:
        valid = data[ETVar.VALID.value]
    else:
        valid = xr.ones_like(
            next(iter(data.data_vars.values())), dtype=FLAGS_TYPE
        )
    if ETVar.FLAGS.value in data.data_vars:
        flags = data[ETVar.FLAGS.value]
    else:
        flags = xr.zeros_like(
            next(iter(data.data_vars.values())), dtype=FLAGS_TYPE
        )
    # Extrapolation
    msg = f"Method used for extrapolation: {method}"
    logger.debug(msg)
    if method.lower() == "toa":
        if data.attrs.get("date", None) is None:
            msg = (
                "Impossible to extrapolate because the date "
                "is missing in metadata"
            )
            raise ValueError(msg)
        daily = toa_daily_estimate(
            data=data.drop_vars(
                [ETVar.VALID.value, ETVar.FLAGS.value], errors="ignore"
            )[keep],
            date=data.attrs["date"],
            dem=dem,
        )
    elif method.lower() == "geo":
        # Get geostationary data
        if extra is None:
            msg = (
                "Impossible to extrapolate because no geostationary "
                "is providing"
            )
            raise ValueError(msg)
        inst_rsd_name = ETVar.RSD.value
        daily_rsd_name = ETVar.DAILY_RSD.value
        if name is not None:
            inst_rsd_name = f"{ETVar.RSD.value}_{name}"
            daily_rsd_name = str.replace(
                ETVar.DAILY_RSD.value, ETVar.RSD.value, name, 1
            )
        msg = f"Use {inst_rsd_name} and {daily_rsd_name} for extrapolation"
        logger.debug(msg)
        if (
            inst_rsd_name not in extra.data_vars
            and daily_rsd_name not in extra.data_vars
        ):
            msg = (
                f"Geostationary data ({inst_rsd_name} "
                f"or/and {daily_rsd_name}) is missing"
            )
            raise ValueError(msg)
        daily = geo_daily_estimate(
            data=data.drop_vars(
                [ETVar.VALID.value, ETVar.FLAGS.value], errors="ignore"
            )[keep],
            inst_rsd=extra[inst_rsd_name],
            daily_rsd=extra[daily_rsd_name],
        )
    else:
        msg = f"Extrapolation method {method} unknown"
        logger.error(msg)
        daily = data.copy(data=None)
        daily.attrs = data.attrs.copy()
        for var in keep:
            daily[var] = xr.full_like(
                next(iter(data.data_vars.values())), np.nan
            )
        valid = xr.zeros_like(
            next(iter(data.data_vars.values())), dtype=FLAGS_TYPE
        )
        flags = xr.full_like(
            next(iter(data.data_vars.values())),
            MSK_PROCESSING_FAILED,
            dtype=FLAGS_TYPE,
        )

    # Propagate flags
    xarr = next(iter(daily.data_vars.values()))
    daily[ETVar.VALID.value] = xr.where(xarr.isnull(), 0, valid)
    daily[ETVar.FLAGS.value] = xr.where(
        xarr.isnull(), flags | FLAGS_TYPE(MSK_PROCESSING_FAILED), flags
    )
    return daily
