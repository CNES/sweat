#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import warnings

import numpy as np
import rasterio as rio
import xarray as xr


def read_data(filename: str) -> xr.Dataset:
    """
    Description
    -----------
    Read data

    Parameters
    ----------
    lst : np.array
        Land surface temperature
    var : np.array
        Variable used versus temperature (Albedo)

    Return
    ------
    xr.Dataset
        Input data
    """
    xarr = xr.Dataset()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with rio.open(filename, "r") as ds:
            bounds = ds.bounds
            resolution = ds.transform[0]
            vars = ["lst", "emis", "ndvi", "albedo", "lai", "ra", "rg"]
            # Check variables
            for var in vars:
                if var not in ds.descriptions:
                    raise ValueError(f"Band {var} not found")
            # Read data
            data = {}
            for i, dtype, desc in zip(ds.indexes, ds.dtypes, ds.descriptions):
                if desc in vars:
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
    if len(xarr.data_vars) == 0:
        raise ValueError("No data read")
    return xarr
