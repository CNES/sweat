#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os
import warnings
import fnmatch

import numpy as np
import rasterio as rio
import xarray as xr


def _open_rasterio(filename: str) -> xr.Dataset:
    """
    Description
    -----------
    Read data from a file

    Parameters
    ----------
    filename : str
        Path to the file

    Returns
    -------
    xarr: xr.Dataset
        Data
    """
    xarr = xr.Dataset()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with rio.open(filename, "r") as ds:
            bounds = ds.bounds
            resolution = ds.transform[0]
            # Read data
            data = {}
            for i, dtype, desc in zip(ds.indexes, ds.dtypes, ds.descriptions):
                data[desc] = (["y", "x"], ds.read(i, out_dtype=dtype, masked=True))
            xcoords: np.ndarray = np.linspace(
                bounds.left + 0.5 * resolution,
                bounds.right - 0.5 * resolution,
                ds.shape[1],
            )

            ycoords: np.ndarray = np.linspace(
                bounds.top - 0.5 * resolution,
                bounds.bottom + 0.5 * resolution,
                ds.shape[0],
            )

            xarr = xr.Dataset(
                data_vars=data,
                coords={"x": xcoords, "y": ycoords},
                attrs={"crs": ds.crs},
            )
    return xarr


def read_data_from_file(filename: str) -> xr.Dataset:
    """
    Description
    -----------
    Read data from a file

    Parameters
    ----------
    filename : str
        Path to the file

    Returns
    -------
    xarr : xr.Dataset
        Data
    """
    xarr = _open_rasterio(filename=filename)
    if len(xarr.data_vars) == 0:
        raise ValueError("No data read")
    return xarr


def read_data(dirname: str) -> xr.Dataset:
    """
    Description
    -----------
    Read data in a directory

    Parameters
    ----------
    dirname : str
        Path to the directory

    Returns
    -------
    xarr : xr.Dataset
        Data
    """
    if not os.path.isdir(dirname):
        raise IOError(f"Fail to open directory {dirname}")
    filenames = fnmatch.filter(os.listdir(dirname), "*.tif")
    if len(filenames) == 0:
        raise IOError(f"No file found in directory {dirname}")
    xarrs = [_open_rasterio(os.path.join(dirname, filename)) for filename in filenames]
    print(len(xarrs))
    return xr.merge(xarrs, combine_attrs="override")
