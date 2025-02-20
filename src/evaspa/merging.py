# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import warnings
from enum import Enum

import scipy
import xarray as xr

from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


class MergeMethod(Enum):
    """Method for merge EF mdoels"""

    # TODO: Add method to compute uncertainty depending on the merge method used

    MEAN = "mean"
    MEDIAN = "median"


def merge(
    data: xr.Dataset, method: MergeMethod = MergeMethod.MEAN
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Description
    -----------
    Merge data and compute uncertainty

    Parameters
    ----------
    data: xr.Dataset
        Data to merge
    method:
        Method used to merge

    Returns
    -------
    merged: xr.DataArray
        Merged data
    uncertainty: xr.DataArray
        Uncertainty of merged data
    """
    if len(data.data_vars) == 0:
        msg = "Unable to merge, dataset is empty"
        raise ValueError(msg)
    if method.value == MergeMethod.MEAN.value:
        xarr = data.to_dataarray(dim="new")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Degrees of freedom <= 0 for slice.",
                category=RuntimeWarning,
            )
            return xarr.mean("new"), xarr.std(dim="new", skipna=True, ddof=1)
    if method.value == MergeMethod.MEDIAN.value:
        xarr = data.to_dataarray(dim="new")
        return xarr.median("new"), xr.apply_ufunc(
            scipy.stats.median_abs_deviation,
            xarr,
            input_core_dims=[
                ["new"],
            ],
            kwargs={"axis": -1, "nan_policy": "propagate", "scale": "normal"},
        )
    msg = f"Merge method unknown: {method}"
    raise ValueError(msg)


def merge_to_dataset(
    data: xr.Dataset,
    method: MergeMethod = MergeMethod.MEAN,
    name: str = "merged",
) -> xr.Dataset:
    """
    Description
    -----------
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
    merged_arr, uncertainty_arr = merge(data, method)
    merged = merged_arr.to_dataset(name=name)
    merged[f"uncertainty_{name}"] = uncertainty_arr
    merged.attrs = data.attrs.copy()
    return merged
