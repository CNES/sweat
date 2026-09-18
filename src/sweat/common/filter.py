# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for filtering functions
"""

from __future__ import annotations

import operator
import re
from functools import reduce
from typing import Any, Literal

import numpy as np
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
)

from sweat.common.constant import (
    FLAGS_TYPE,
    MSK_INPUT_FILTERED,
    MSK_INPUT_NODATA,
)
from sweat.common.types import ETVar, PercentileValue
from sweat.debugging import register_debugging
from sweat.logging import LoggerManager

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
    """
    Class to describe a simple condition
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    op: Literal["==", "!=", ">", ">=", "<", "<="]
    value: float | PercentileValue

    @field_validator("value", mode="before")
    @classmethod
    def parse_value(cls, v: Any) -> float | PercentileValue:
        """
        Check value and convert if necessary to percentile
        """
        if isinstance(v, float | int):
            return float(v)
        if isinstance(v, PercentileValue):
            return v
        if isinstance(v, str):
            match = re.match(r"percentile\((\d+(\.\d+)?)\)", v.strip())
            if match:
                percentile = float(match.group(1))
                return PercentileValue(percentile)
        msg = (
            "value must be a float or a percentile or a "
            "string in percentile format like 'percentile(90)'"
        )
        raise ValueError(msg)


# Composite condition with 'and'/'or' logic
class CompositeCondition(BaseModel):
    """
    Class to describe a condition with or/and
    """

    model_config = ConfigDict(extra="forbid")
    and_: list[ConditionType] | None = Field(default=None, alias="and")
    or_: list[ConditionType] | None = Field(default=None, alias="or")


# Union type: either simple or composite
ConditionType = SimpleCondition | CompositeCondition

# Allow forward references in CompositeCondition
CompositeCondition.model_rebuild()


# Final config mapping variable names to conditions
class FilteringConfig(RootModel):
    """
    Class to describe the filtering to be applied
    """

    root: dict[str, ConditionType]

    @field_serializer("root", mode="wrap")
    def serialize_root(
        self,
        v: dict[str, ConditionType],
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> Any:
        if info.mode == "json":
            return self._serialize_conditions(v)
        return handler(v)

    def _serialize_conditions(self, obj: Any) -> Any:
        """
        Recursively serialize conditions, converting PercentileValue to string
        """
        if isinstance(obj, dict):
            return {k: self._serialize_conditions(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize_conditions(item) for item in obj]
        if isinstance(obj, PercentileValue):
            return repr(obj)
        if isinstance(obj, SimpleCondition):
            return {
                "op": obj.op,
                "value": self._serialize_conditions(obj.value),
            }
        if isinstance(obj, CompositeCondition):
            result = {}
            if obj.and_:
                result["and"] = [
                    self._serialize_conditions(c) for c in obj.and_
                ]
            if obj.or_:
                result["or"] = [self._serialize_conditions(c) for c in obj.or_]
            return result
        return obj


def eval_condition(da: xr.DataArray, cond: dict) -> xr.DataArray:
    """
    Evaluate a condition for a data array

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

    if isinstance(cond["value"], PercentileValue):
        # Compute the percentile along the DataArray
        val = float(da.quantile(cond["value"].percentile / 100.0))
    else:
        val = cond["value"]  # it's just a float

    return op_func(da, val)


def apply_condition(da: xr.DataArray, cond: dict) -> xr.DataArray:
    """
    Apply a condition on a data array

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

    if cond.get("and") is not None:
        and_results = [apply_condition(da, c) for c in cond["and"]]
    if cond.get("or") is not None:
        or_results = [apply_condition(da, c) for c in cond["or"]]

    result = xr.full_like(da, 1, dtype=int)
    if and_results:
        result = reduce(lambda x, y: x & y, and_results)
    if or_results:
        or_combined = reduce(lambda x, y: x | y, or_results)
        result = result & or_combined

    return result


def detect_valid_pixels(data: xr.Dataset, config: dict) -> xr.DataArray:
    """
    Detect valid pixels on a dataset based on
    conditions in data variables. The conditions are
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
    checked_config = FilteringConfig.model_validate(config).model_dump(
        by_alias=True
    )
    # Check if dataset is not empty
    if len(data.data_vars) == 0:
        msg = "Dataset is empty, filtering not possible"
        raise ValueError(msg)
    # Initialize valid mask
    valid_mask = xr.full_like(next(iter(data.data_vars.values())), 1, dtype=int)

    # Loop over all the conditions
    for var_name, condition in checked_config.items():
        if var_name not in data:
            msg = f"Variable {var_name} not found"
            logger.warning(msg)
            continue
        mask = apply_condition(data[var_name], condition)
        valid_mask = valid_mask & mask

    return valid_mask


def detect_nan_pixels(
    data: xr.Dataset, variables: list[str] | str = "all"
) -> xr.DataArray:
    """
    Detect nan values on a dataset considering
    a list of variables in the dataset.

    Parameters
    ----------
    data : xr.Dataset
        Data
    variables : list[str] or str
        List of data variables to filter nan values
        If "all" is provided, all data variables will be
        considered

    Returns
    -------
    nan_mask: xr.DataArray
        Nan pixels
    """
    # Check if dataset is not empty
    if len(data.data_vars) == 0:
        msg = "Dataset is empty, not possible to detect nan pixels"
        raise ValueError(msg)
    # Check variables arguments
    if isinstance(variables, str):
        if variables != "all":
            msg = "Only 'all' can be provided"
            raise ValueError(msg)
        variables = [str(v) for v in data.data_vars]
    # Initialize nan mask
    nan_mask = xr.full_like(next(iter(data.data_vars.values())), 0, dtype=int)

    # Loop over all variables
    for var_name in variables:
        if var_name not in data:
            msg = f"Variable {var_name} not found"
            logger.warning(msg)
            continue
        mask = data[var_name].isnull()
        nan_mask = nan_mask | mask

    return nan_mask


@register_debugging
def find_valid_pixels(
    data: xr.Dataset,
    nan_config: list[str] | str | None = None,
    valid_config: dict | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Find valid pixels.

    Parameters
    ----------
    data : xr.Dataset
        Data
    nan_config : list[str] or str
        List of variables to consider to exclude nan
        if "all", all variables are taken into account
        If not provided, pixel detection at nan does not take place
    valid_config : dict


    Returns
    -------
    valid: xr.DataArray
        Valid pixels mask
    flags: xr.DataArray
        Flags mask
    """
    # Check if dataset is not empty
    if len(data.data_vars) == 0:
        msg = "Dataset is empty, not possible to detect nan pixels"
        raise ValueError(msg)
    # Get valid mask or initialize it
    if ETVar.VALID.value in data.data_vars:
        valid = data[ETVar.VALID.value].astype(FLAGS_TYPE)
    else:
        valid = xr.full_like(
            next(iter(data.data_vars.values())), 1, dtype=FLAGS_TYPE
        )
    # Get flags mask or initialize it
    if ETVar.FLAGS.value in data.data_vars:
        flags = data[ETVar.FLAGS.value].astype(FLAGS_TYPE)
    else:
        flags = xr.full_like(
            next(iter(data.data_vars.values())), 0, dtype=FLAGS_TYPE
        )
    nan_mask = None
    if nan_config is not None:
        nan_mask = detect_nan_pixels(data, nan_config)
        valid = valid & ~nan_mask
        flags = xr.where(
            nan_mask == 1,
            (flags | FLAGS_TYPE(MSK_INPUT_NODATA))
            & FLAGS_TYPE(np.bitwise_not(MSK_INPUT_FILTERED)),
            flags,
        )
    if valid_config is not None:
        if nan_mask is None:
            nan_mask = xr.zeros_like(valid)
        valid_mask = detect_valid_pixels(data, valid_config)
        valid = valid & valid_mask
        flags = xr.where(
            (valid_mask == 0) & (nan_mask != 1),
            flags | MSK_INPUT_FILTERED,
            flags,
        )
    return valid, flags
