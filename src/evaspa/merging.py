# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from enum import Enum

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
) -> xr.DataArray:
    """
    Description
    -----------
    Merge data

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
    if len(data.data_vars) == 0:
        msg = "Unable to merge, dataset is empty"
        raise ValueError(msg)
    if method.value == MergeMethod.MEAN.value:
        return data.to_array(dim="new").mean("new")
    if method.value == MergeMethod.MEDIAN.value:
        return data.to_array(dim="new").median("new")
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
    merged_arr = merge(data, method)
    merged = merged_arr.to_dataset(name=name)
    merged.attrs = data.attrs.copy()
    return merged
