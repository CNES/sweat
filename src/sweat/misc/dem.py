# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for DEM management
"""

from __future__ import annotations

import os

import numpy as np
import rasterio as rio
import xarray as xr
from rasterio.merge import merge as rio_merge
from sensorsio import mgrs
from sensorsio.regulargrid import read_as_numpy

ASPECT_MIN = 0
ASPECT_MAX = 90


def get_dem_from_tile(
    tile_id: str,
    resolution: float = 60,
    base_dir: str | None = None,
):
    """
    Read one tile for DEM Copernicus
    Then, resample it at a specific resolution and
    compute slope et aspect

    Parameters
    ----------
    tile_id: str
        Tile ID
    resolution: str, default=60
        DEM spatial resolution
    base_dir: str
        Path to the DEM directory
        Required to set MNT_PATH environment variable

    Returns
    -------
    xarr: xarray.Dataset
    """
    if base_dir is None:
        base_dir = os.path.join(os.environ["MNT_PATH"], "DEM_Copernicus_30m/")
    file_name = os.path.join(base_dir, f"COP-DEM_GLO-30-DGED_{tile_id}.tif")
    if not os.path.isfile(file_name):
        msg = f"DEM file not found: {file_name}"
        raise OSError(msg)
    elevation, xcoords, ycoords, crs = read_as_numpy(
        [file_name],
        resolution=resolution,
        algorithm=rio.enums.Resampling.cubic,
        dtype=np.int16,
    )
    elevation = elevation[0, 0, :, :]
    x, y = np.gradient(elevation.astype(np.float32))
    slope = np.degrees(np.arctan(np.sqrt(x * x + y * y) / resolution))
    # Aspect unfolding rules from
    # https://github.com/r-barnes/richdem/blob/603cd9d16164393e49ba8e37322fe82653ed5046/include/richdem/methods/terrain_attributes.hpp#L236
    aspect = np.rad2deg(np.arctan2(x, -y))
    lt_0 = aspect < ASPECT_MIN
    gt_90 = aspect > ASPECT_MAX
    remaining = np.logical_and(aspect >= ASPECT_MIN, aspect <= ASPECT_MAX)
    aspect[lt_0] = ASPECT_MAX - aspect[lt_0]
    aspect[gt_90] = 360 - aspect[gt_90] + ASPECT_MAX
    aspect[remaining] = ASPECT_MAX - aspect[remaining]
    left = np.min(xcoords) - resolution / 2
    top = np.max(ycoords) + resolution / 2
    transform = rio.Affine(resolution, 0.0, left, 0.0, -resolution, top)
    data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
    data_vars["height"] = (["y", "x"], elevation)
    data_vars["slope"] = (["y", "x"], slope)
    data_vars["aspect"] = (["y", "x"], aspect)
    return xr.Dataset(
        data_vars,
        coords={"x": xcoords, "y": ycoords},
        attrs={
            "crs": crs,
            "resolution": resolution,
            "transform": transform,
        },
    )


def get_dem_from_tiles(
    tile_ids: list[str], resolution: float = 60, base_dir: str | None = None
) -> xr.Dataset:
    """
    Read several tiles for DEM Copernicus
    Then, resample them at a specific resolution and
    compute slope et aspect
    All tiles are merged in the same dataset.

    Parameters
    ----------
    tile_ids: List[str]
        List of tile IDs
    resolution: str, default=60
        DEM spatial resolution
    base_dir: str
        Path to the DEM directory
        Required to set MNT_PATH environment variable

    Returns
    -------
    xarr: xarray.Dataset
    """
    if base_dir is None:
        base_dir = os.path.join(os.environ["MNT_PATH"], "DEM_Copernicus_30m/")
    if len(tile_ids) == 0:
        msg = "No DEM tiles requested"
        raise ValueError(msg)
    # Get CRS from first tile
    crs = f"EPSG:{mgrs.get_crs_mgrs_tile(tile_ids[0]).to_epsg()}"
    # Get file paths
    file_names = [
        os.path.join(base_dir, f"COP-DEM_GLO-30-DGED_{tile_id}.tif")
        for tile_id in tile_ids
    ]
    # Use rasterio merge to read DEM files
    elevation, transform = rio_merge(
        file_names,
        res=resolution,
        nodata=np.nan,
        resampling=rio.enums.Resampling.cubic,
        dtype=np.float32,
    )
    elevation = elevation[0, :, :]
    # Get bounds
    left, bottom, right, top = rio.transform.array_bounds(
        elevation.shape[0], elevation.shape[1], transform
    )
    xcoords: np.ndarray = np.linspace(
        left + 0.5 * resolution, right - 0.5 * resolution, elevation.shape[1]
    )

    ycoords: np.ndarray = np.linspace(
        top - 0.5 * resolution, bottom + 0.5 * resolution, elevation.shape[0]
    )
    x, y = np.gradient(elevation.astype(np.float32))
    slope = np.degrees(np.arctan(np.sqrt(x * x + y * y) / resolution))
    # Aspect unfolding rules from
    # https://github.com/r-barnes/richdem/blob/603cd9d16164393e49ba8e37322fe82653ed5046/include/richdem/methods/terrain_attributes.hpp#L236
    aspect = np.rad2deg(np.arctan2(x, -y))
    lt_0 = aspect < ASPECT_MIN
    gt_90 = aspect > ASPECT_MAX
    remaining = np.logical_and(aspect >= ASPECT_MIN, aspect <= ASPECT_MAX)
    aspect[lt_0] = ASPECT_MAX - aspect[lt_0]
    aspect[gt_90] = 360 - aspect[gt_90] + ASPECT_MAX
    aspect[remaining] = ASPECT_MAX - aspect[remaining]
    left = np.min(xcoords) - resolution / 2
    top = np.max(ycoords) + resolution / 2
    transform = rio.Affine(resolution, 0.0, left, 0.0, -resolution, top)
    data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
    data_vars["height"] = (["y", "x"], elevation)
    data_vars["slope"] = (["y", "x"], slope)
    data_vars["aspect"] = (["y", "x"], aspect)
    return xr.Dataset(
        data_vars,
        coords={"x": xcoords, "y": ycoords},
        attrs={
            "crs": crs,
            "resolution": resolution,
            "transform": transform,
        },
    )
