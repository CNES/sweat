# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import operator
from functools import reduce
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, RootModel

from evaspa.debugging import register_debugging
from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

# Operator mapping
OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


# Simple condition
class SimpleCondition(BaseModel):
    op: Literal["==", "!=", ">", ">=", "<", "<="]
    value: float


# Composite condition with 'and'/'or' logic
class CompositeCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    and_: list[ConditionType] | None = Field(default=None, alias="and")
    or_: list[ConditionType] | None = Field(default=None, alias="or")


# Union type: either simple or composite
ConditionType = SimpleCondition | CompositeCondition

# Allow forward references in CompositeCondition
# CompositeCondition.model_rebuild()


# Final config mapping variable names to conditions
class FilteringConfig(RootModel):
    root: dict[str, ConditionType]


class FilterParams(BaseModel):
    """
    Parameters for the filter.
    The parameters correspond to the values used
    for masking
    """

    model_config = ConfigDict(extra="forbid")

    cloud: int = Field(default=0)
    water: int = Field(default=0)
    qa: int = Field(default=0b0000000000000000)
    zones: int = Field(default=0)
    cover: list[int] = Field(default=[10, 60, 80])  # TODO: TBC


class FilterConfig(BaseModel):
    """
    Configuration for filtering pixels
    The configuration contains names of the variables
    for each mask and parameters for the filter
    """

    model_config = ConfigDict(extra="forbid")

    cloud: str | None = Field(default=None)
    water: str | None = Field(default=None)
    qa: str | None = Field(default=None)
    zones: str | None = Field(default=None)
    cover: str | None = Field(default=None)
    config: FilterParams = Field(default=FilterParams())


@register_debugging
def determine_valid_pixels(
    data: xr.Dataset,
    cloud: npt.ArrayLike | str | None = None,
    water: npt.ArrayLike | str | None = None,
    qa: npt.ArrayLike | str | None = None,
    zones: npt.ArrayLike | str | None = None,
    cover: npt.ArrayLike | str | None = None,
    config: dict | None = None,
) -> xr.DataArray:
    """
    Description
    -----------
    Filter data

    Parameters
    ----------
    data : xr.Dataset
        Data

    Returns
    -------
    filtered: xr.Dataset
        Filtered data
    """
    # Mask configuration
    default_config = FilterParams().model_dump()
    if config is None:
        config = default_config
    updated_config = default_config | config
    # Mask creation
    valid = np.ones_like(data["lst"].data)
    # Identify no cloudy pixels
    if cloud is not None:
        if isinstance(cloud, str):
            cloud = data[cloud]
        valid = np.logical_and(
            valid, mask(cloud, values=updated_config["cloud"])
        )
    # Identify no water pixels
    if water is not None:
        if isinstance(water, str):
            water = data[water]
        valid = np.logical_and(
            valid, mask(water, values=updated_config["water"])
        )
    # Identify pixels computed correctly form previous step
    if qa is not None:
        if isinstance(qa, str):
            qa = data[qa]
        valid = np.logical_and(valid, mask(qa, values=updated_config["qa"]))
    # Identify pixels in the valid zone
    if zones is not None:
        if isinstance(zones, str):
            zones = data[zones]
        valid = np.logical_and(
            valid, mask(zones, values=updated_config["zones"])
        )
    # Select pixels with land use / land cover
    if cover is not None:
        if isinstance(cover, str):
            cover = data[cover]
        valid = np.logical_and(
            valid, mask(cover, values=updated_config["cover"])
        )
    return xr.DataArray(
        data=valid,
        dims=data.dims,
        coords=data.coords.copy(),
    )


def mask(
    data: npt.ArrayLike,
    values: npt.ArrayLike,
    invert=False,
) -> npt.NDArray:
    """
    Description
    -----------
    Mask data

    Parameters
    ----------
    data : np.array_like
        Data
    values : np.array_like
        Values used for masking
    invert : bool
        If True, an element is the returned array is equal to True
        if the element in data array is equal to one of the values.
        Default is True.

    Returns
    -------
    mask: np.array
        Masked data
    """
    return np.isin(np.array(data), np.array(values), invert=invert)


def eval_condition(da: xr.DataArray, cond: dict) -> xr.DataArray:
    """
    Description
    -----------
    Evaluate a condition for a dataarray

    Parameters
    ----------
    da : xr.DataArray
        Data array
    cond : dict
        Condition for filtering

    Returns
    -------
    eval: xr.DataArray
        Evaluated condition
    """
    op_func = OPS[cond["op"]]
    return op_func(da, cond["value"])


def apply_condition(da: xr.DataArray, cond: dict) -> xr.DataArray:
    """
    Description
    -----------
    Apply a condition on a dataarray

    Parameters
    ----------
    da : xr.DataArray
        Data array
    cond : dict
        Condition for filtering

    Returns
    -------
    eval: xr.DataArray
        Evaluated condition
    """
    # Simple condition
    if "op" in cond and "value" in cond:
        return eval_condition(da, cond)

    # Composite condition
    and_results = []
    or_results = []

    if "and" in cond:
        and_results = [apply_condition(da, c) for c in cond["and"]]
    if "or" in cond:
        or_results = [apply_condition(da, c) for c in cond["or"]]

    result = xr.full_like(da, 1, dtype=int)
    if and_results:
        result = reduce(lambda x, y: x & y, and_results)
    if or_results:
        or_combined = reduce(lambda x, y: x | y, or_results)
        result = result & or_combined

    return result


@register_debugging
def filter_valid_pixels(data: xr.Dataset, config: dict) -> xr.DataArray:
    """
    Description
    -----------
    Determine valid pixels on a datasert based on
    codnitions in data variables. The conditions are
    described in a configuration dictionary.

    Parameters
    ----------
    data : xr.Dataset
        Data
    config : dict
        Configuration containing conditions to
        determine valid pixels

    Returns
    -------
    valid: xr.DataArray
        Valid pixels
    """
    # Check if dataset is not empty
    if len(data.data_vars) == 0:
        msg = "Dataset is empty, filtering not possible"
        raise ValueError(msg)
    # iInitialize valid mask
    valid_mask = xr.full_like(next(iter(data.data_vars.values())), 1, dtype=int)

    # Loop over all the conditions
    for var_name, condition in config.items():
        if var_name not in data:
            msg = f"Variable {var_name} not found"
            logger.warning(msg)
            continue
        mask = apply_condition(data[var_name], condition)
        valid_mask = valid_mask & mask

    return valid_mask
