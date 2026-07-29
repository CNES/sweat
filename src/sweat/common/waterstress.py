# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions for Surface Energy Balance (net radiation,
latent heat flux)
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from sweat.common.constant import FLAGS_TYPE
from sweat.common.types import ETVar
from sweat.debugging import register_debugging
from sweat.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


class WaterStressMethod(Enum):
    """List of water stress methods"""

    EF = "ef"


DEFAULT_METHOD = WaterStressMethod.EF


class WaterStressConfig(BaseModel):
    """
    SEB config for ET processing
    """

    model_config = ConfigDict(extra="forbid")

    method: WaterStressMethod = Field(default=DEFAULT_METHOD)


def compute_waterstress_from_ef(ef: npt.ArrayLike) -> npt.NDArray:
    """
    Compute water stress from evaporative fraction

    Parameters
    ----------
    ef: np.array_like
        EF Data

    Returns
    -------
    water_stress: np.array
        Water stress indices
    """
    return 1 - np.array(ef)


@register_debugging
def run(
    et: xr.Dataset,
    method: WaterStressMethod = DEFAULT_METHOD,
) -> xr.Dataset:
    """
    Compute water stress indices

    Parameters
    ----------
    data: xr.Dataset
        ET dataset

    Returns
    -------
    water_stress: xr.DataArray
        Water stress indices
    """
    if len(et.data_vars) == 0:
        msg = "ET dataset empty"
        raise ValueError(msg)
    # Get valid and flags
    if ETVar.VALID.value in et.data_vars:
        valid = et[ETVar.VALID.value]
    else:
        valid = xr.ones_like(
            next(iter(et.data_vars.values())), dtype=FLAGS_TYPE
        )
    if ETVar.FLAGS.value in et.data_vars:
        flags = et[ETVar.FLAGS.value]
    else:
        flags = xr.zeros_like(
            next(iter(et.data_vars.values())), dtype=FLAGS_TYPE
        )
    try:
        method_name = WaterStressMethod(method).name
    except ValueError as exc:
        msg = f"Unknown method for water stress: {method}"
        raise ValueError(msg) from exc
    # Compute water stress
    try:
        if method_name == WaterStressMethod.EF.name:
            water_stress = xr.full_like(et[ETVar.EF.value], np.nan)
            water_stress.data = compute_waterstress_from_ef(
                et[ETVar.EF.value].data
            )
    except KeyError as exc:
        msg = f"Data missing for {method_name} method: {exc}"
        raise ValueError(msg) from exc
    # Attributes
    attrs = {
        "crs": et.attrs.get("crs", None),
        "transform": et.attrs.get("transform", None),
    }
    # Create dataset
    ws_xr = xr.Dataset(
        data_vars={ETVar.WATERSTRESS.value: water_stress}, attrs=attrs
    )
    # Propagate flags
    ws_xr[ETVar.VALID.value] = valid
    ws_xr[ETVar.FLAGS.value] = flags
    return ws_xr
