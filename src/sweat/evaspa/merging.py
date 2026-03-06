# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions for merging datasets
"""

import warnings
from enum import Enum
from typing import Any, Self

import scipy
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from sweat.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


class MergingMethod(Enum):
    """
    Method for merging EF models
    """

    MEAN = "mean"
    MEDIAN = "median"


class UncertaintyMethod(Enum):
    """
    Method for computing uncertainty after merging EF models
    """

    STD = "std"
    NMAD = "nmad"
    INTERQUARTILE = "interquartile"


class MergingConfig(BaseModel):
    """
    Configuration of EF models merging
    """

    model_config = ConfigDict(extra="forbid")
    merging_method: MergingMethod = Field(default=MergingMethod.MEAN)
    uncertainty_method: UncertaintyMethod | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def check_merging_method(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and "uncertainty_method" in data
            and "merging_method" not in data
        ):
            msg = (
                "merging_method must be provided "
                "if uncertainty method is provided"
            )
            raise ValueError(msg)
        return data

    @model_validator(mode="after")
    def check_uncertainty_method(self) -> Self:
        """
        Check uncertainty method

        If no uncertainty method is specified,
        a default method is selected based on the merging method used:

        - STD with MEAN
        - NMAD with MEDIAN

        Parameters
        ----------
        self: LinearEdge
            LinearEdge instance

        Returns
        -------
        self: LinearEdge
            Checked instance
        """
        if self.uncertainty_method is None:
            if self.merging_method.value == MergingMethod.MEAN.value:
                self.uncertainty_method = UncertaintyMethod.STD
            if self.merging_method.value == MergingMethod.MEDIAN.value:
                self.uncertainty_method = UncertaintyMethod.NMAD
        return self


def merge(
    data: xr.Dataset,
    merging_method: MergingMethod = MergingMethod.MEAN,
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.STD,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Merge data and compute uncertainty.

    The merging method can be:

    - mean
    - median

    For uncertainty method, it is possible to use:

    - standard deviation
    - normalized median absolute deviation
    - interquartile

    Parameters
    ----------
    data: xr.Dataset
        Data to merge
    merging_method:
        Method used to merge
    uncertainty_method:
        Method used to compute uncertainty

    Returns
    -------
    merged: xr.DataArray
        Merged data
    uncertainty: xr.DataArray
        Uncertainty of merged data
    """
    # Check if is not empty
    if len(data.data_vars) == 0:
        msg = "Unable to merge, dataset is empty"
        raise ValueError(msg)

    # Create a new dimension
    xarr = data.to_dataarray(dim="new")

    # Merging method
    if merging_method.value == MergingMethod.MEAN.value:
        merging_xarr = xarr.mean(dim="new", skipna=False)
    elif merging_method.value == MergingMethod.MEDIAN.value:
        merging_xarr = xarr.median(dim="new", skipna=False)
    else:
        msg = f"Merge method not implemented: {merging_method}"
        raise ValueError(msg)

    # Uncertainty method
    if uncertainty_method.value == UncertaintyMethod.STD.value:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Degrees of freedom <= 0 for slice.",
                category=RuntimeWarning,
            )
            # Standard deviation compute with ddof = 1
            uncertainty_xarr = xarr.std(dim="new", skipna=False, ddof=1)
    elif uncertainty_method.value == UncertaintyMethod.NMAD.value:
        uncertainty_xarr = xr.apply_ufunc(
            scipy.stats.median_abs_deviation,
            xarr,
            input_core_dims=[
                ["new"],
            ],
            kwargs={"axis": -1, "nan_policy": "propagate", "scale": "normal"},
        )
    elif uncertainty_method.value == UncertaintyMethod.INTERQUARTILE.value:
        q25, q75 = xarr.quantile([0.25, 0.75], dim="new", skipna=False)
        uncertainty_xarr = q75 - q25
    else:
        msg = f"Uncertainty method not implemented: {merging_method}"
        raise ValueError(msg)

    # Return
    return merging_xarr, uncertainty_xarr


def merge_to_dataset(
    data: xr.Dataset,
    merging_method: MergingMethod = MergingMethod.MEAN,
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.STD,
    name: str = "merged",
) -> xr.Dataset:
    """
    Merge data and create a dataset

    Parameters
    ----------
    data: xr.Dataset
        Data to merge
    method:
        Method used to merge

    Returns
    -------
    return: xr.DataArray
        Merged data
    """
    merged_arr, uncertainty_arr = merge(
        data,
        merging_method=merging_method,
        uncertainty_method=uncertainty_method,
    )
    merged = merged_arr.to_dataset(name=name)
    merged[f"uncertainty_{name}"] = uncertainty_arr
    merged.attrs = data.attrs.copy()
    return merged
