#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import os
import warnings
import fnmatch
from pathlib import Path

import numpy as np
import rasterio as rio
import xarray as xr
from rasterio.enums import ColorInterp


def _open_rasterio(filename: str) -> xr.Dataset:
    """
    Description
    -----------
    Read data from a file

    Parameters
    ----------
    filename: str
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
                attrs={"crs": ds.crs, "transform": ds.transform},
            )
    return xarr


def read_data_from_file(filename: str) -> xr.Dataset:
    """
    Description
    -----------
    Read data from a file

    Parameters
    ----------
    filename: str
        Path to the file

    Returns
    -------
    xarr: xr.Dataset
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
    dirname: str
        Path to the directory

    Returns
    -------
    xarr: xr.Dataset
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


def write_dataset(
    xrds: xr.Dataset,
    filename: str,
    directory: str = os.getcwd(),
    separate=False,
) -> None:
    """
    Description
    -----------
    Write dataset in one file or in separated files

    Parameters
    ----------
    xrds: xr.Dataset
        Dataset to write
    filename: str
        File name
    directory: str
        Path to the directory
    seperate: bool
        Write bands to separate files
    """
    if len(xrds.data_vars) == 0:
        raise ValueError("Dataset empty")
    # Get col/row
    dims = tuple(i for i in xrds.dims)
    row = xrds.sizes[dims[0]]
    col = xrds.sizes[dims[1]]
    # Get bands
    bands = [i for i in xrds.data_vars]
    # Get georeference data
    crs = xrds.attrs.get("crs", None)
    transform = xrds.attrs.get("transform", None)
    if transform is None:
        transform = rio.Affine(1, 0, 0, 0, 1, 0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
        if not separate:
            with rio.open(
                os.path.join(directory, filename),
                mode="w+",
                driver="GTiff",
                width=col,
                height=row,
                count=len(bands),
                dtype=rio.dtypes.float32,
                nodata=np.nan,
                crs=crs,
                transform=transform,
            ) as source_ds:
                source_ds.colorinterp = [ColorInterp.gray for _ in bands]
                for id, band in enumerate(bands, start=1):
                    source_ds.write_band(id, xrds[band].data)
                    source_ds.set_band_description(id, band)
        else:
            root = Path(filename).stem
            os.makedirs(os.path.join(directory, root), exist_ok=True)
            for band in bands:
                with rio.open(
                    os.path.join(directory, root, root + f"_{band}.tif"),
                    mode="w+",
                    driver="GTiff",
                    width=col,
                    height=row,
                    count=1,
                    dtype=rio.dtypes.float32,
                    nodata=np.nan,
                    crs=crs,
                    transform=transform,
                ) as source_ds:
                    source_ds.colorinterp = [ColorInterp.gray]
                    source_ds.write_band(1, xrds[band].data)
                    source_ds.set_band_description(1, band)
