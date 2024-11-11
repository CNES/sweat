#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import numpy.typing as npt
import xarray as xr

default_config = {
    "cloud": 0,
    "water": 0,
    "qa": 0b0000000000000000,
    "zones": 0,
    "cover": [10, 60, 80],  # TODO Find right values
}


def determine_valid_pixels(
    data: xr.Dataset,
    cloud: npt.ArrayLike | str | None = None,
    water: npt.ArrayLike | str | None = None,
    qa: npt.ArrayLike | str | None = None,
    zones: npt.ArrayLike | str | None = None,
    cover: npt.ArrayLike | str | None = None,
    config: dict = default_config,
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
    updated_config = default_config | config
    # Mask creation
    valid = np.ones_like(data["lst"].data)
    # Identify no cloudy pixels
    if cloud is not None:
        if isinstance(cloud, str):
            cloud = data[cloud]
        valid = np.logical_and(valid, mask(cloud, values=updated_config["cloud"]))
    # Identify no water pixels
    if water is not None:
        if isinstance(water, str):
            water = data[water]
        valid = np.logical_and(valid, mask(water, values=updated_config["water"]))
    # Identify pixels computed correctly form previous step
    if qa is not None:
        if isinstance(qa, str):
            qa = data[qa]
        valid = np.logical_and(valid, mask(qa, values=updated_config["qa"]))
    # Identify pixels in the valid zone
    if zones is not None:
        if isinstance(zones, str):
            zones = data[zones]
        valid = np.logical_and(valid, mask(zones, values=updated_config["zones"]))
    # Select pixels with land use / land cover
    if cover is not None:
        if isinstance(cover, str):
            cover = data[cover]
        valid = np.logical_and(valid, mask(cover, values=updated_config["cover"]))
    return xr.DataArray(
        data=valid,
        dims=data.dims,
        coords=data.coords.copy(),
    )


def mask(data: npt.ArrayLike, values: npt.ArrayLike, invert=False) -> npt.NDArray:
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
        if the element in daat array is equal to one of the values.
        Default is True.

    Returns
    -------
    mask: np.array
        Masked data
    """
    return np.isin(np.array(data), np.array(values), invert=invert)
