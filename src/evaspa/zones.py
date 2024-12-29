# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module to manage valid zones for EVASPA
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.features as rio_features
import sensorsio.utils as sio_utils
import skimage.morphology as skm
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon, shape

from evaspa.dem import get_dem_from_tiles

if TYPE_CHECKING:
    import affine


def compute_water_mask(
    xrds: xr.Dataset, land: gpd.GeoDataFrame
) -> xr.DataArray:
    """
    Compute the water on a ROI

    Parameters
    ----------
    xrds: xarray.Dataset
        Data on which to calculate the water mask
    land: geopandas.GeoDataFrame
        List of land polygons

    Returns
    -------
    water: xarray.DataArray
    """
    bounds = rio.coords.BoundingBox(
        *rio.transform.array_bounds(
            xrds.sizes["y"], xrds.sizes["x"], xrds.transform
        )
    )
    wgs84_bounds = sio_utils.compute_latlon_bbox_from_region(bounds, xrds.crs)
    roi_poly = Polygon(
        [
            [wgs84_bounds[0], wgs84_bounds[1]],
            [wgs84_bounds[0], wgs84_bounds[3]],
            [wgs84_bounds[2], wgs84_bounds[3]],
            [wgs84_bounds[2], wgs84_bounds[1]],
        ]
    )
    roi = gpd.GeoDataFrame(
        data={"id": ["roi"]}, crs="EPSG:4326", geometry=[roi_poly]
    )
    overlap = gpd.overlay(land, roi)
    return xr.DataArray(
        rio_features.geometry_mask(
            overlap["geometry"].to_crs(xrds.crs),
            out_shape=(xrds.sizes["y"], xrds.sizes["x"]),
            transform=xrds.transform,
        ),
        coords=xrds.coords,
        dims=("y", "x"),
    )


def multi_erosion(
    im: np.ndarray, num: int = 2, footprint: np.ndarray | None = None
) -> np.ndarray:
    """
    Perform multiple erosion on a binary image

    Parameters
    ----------
    im: xarray.Dataset
        Data on which to calculate the water mask
    num: int, default = 2
        Number of erosions
    footprint: np.array, default = 8-connected
        Footprint used to propagate erosion

    Returns
    -------
    eroded_im: np.ndarray
    """
    if footprint is None:
        footprint = skm.footprint_rectangle((3, 3))
    if num <= 0:
        msg = "num must be greater than 0"
        raise ValueError(msg)
    eroded_im = np.copy(im)
    for _ in range(num):
        eroded_im = skm.binary_erosion(eroded_im, footprint=footprint)
    return eroded_im


def multi_dilatation(
    im: np.ndarray, num: int = 2, footprint: np.ndarray | None = None
) -> np.ndarray:
    """
    Perform multiple dilation on a binary image

    Parameters
    ----------
    im: xarray.Dataset
        Data on which to calculate the water mask
    num: int, default = 2
        Number of dilatations
    footprint: np.array, default = 8-connected
        Footprint used to propagate dilatation

    Returns
    -------
    dilated_im: np.ndarray
    """
    if footprint is None:
        footprint = skm.footprint_rectangle((3, 3))
    if num <= 0:
        msg = "num must be greater than 0"
        raise ValueError(msg)
    dilated_im = np.copy(im)
    for _ in range(num):
        dilated_im = skm.binary_dilation(dilated_im, footprint=footprint)
    return dilated_im


def compute_valid_mask(
    xrds_dem: xr.Dataset,
    slope_threshold: float = 30,
    range_threshold: int = 300,
    simplify: bool = True,
) -> xr.DataArray:
    """
    Compute valid pixels by applying the
    following steps:
       - Filter by slope
       - Compute min and max values
       - Compute the distribution of height
       - Find the range that contains the most of pixels
       - Clean the valid mask by performing dilation
         and removing small holes, with simplify option

    Parameters
    ----------
    xrds_dem: xarray.Dataset
        Dataset containing DEM information
    slope_threshold: float, default = 30
        Maximum slope value accepted
    range_threshold: int, default = 300
        Range of altitude for pixels selection
    simplify: boo, default = True
        Activate cleaning step

    Returns
    -------
    valid: xr.DataArray
    """
    # Initilization
    valid = xrds_dem["height"]
    # Filter with water_mask
    if "water_mask" in xrds_dem.variables:
        valid = xr.where(
            xrds_dem["water_mask"] == 0, xrds_dem["height"], np.nan
        )

    # Filtre with slope
    valid = xr.where(xrds_dem["slope"] < slope_threshold, valid, np.nan)
    # Identify pixels in the same elevation range
    min_h = valid.min(skipna=True).data[()]
    max_h = valid.max(skipna=True).data[()]
    start_h = int(np.floor(min_h))
    end_h = int(np.ceil(max_h))
    bin_size = 10  # Size of a bin in meters
    nb = int(np.ceil((end_h - start_h) / bin_size))
    bins = np.linspace(start_h, start_h + nb * bin_size, nb + 1)
    window_size = int(np.round(range_threshold / bin_size))
    hist, _ = np.histogram(valid.data, bins)
    ind = np.convolve(hist, np.ones(window_size, dtype=int), "valid").argmax()
    range_min = bins[ind]
    if ind + window_size >= len(bins):
        range_max = bins[-1]
    else:
        range_max = bins[ind + window_size]
    valid = xr.where((range_min < valid) & (valid < range_max), 1, 0).astype(
        "uint8"
    )
    # Simplify
    if simplify:
        clean_valid = multi_dilatation(
            valid.data, 2, footprint=skm.footprint_rectangle((3, 3))
        )
        clean_valid = skm.remove_small_holes(
            clean_valid, area_threshold=200, connectivity=2
        )
        valid.data = clean_valid.astype("uint8")
    return valid


def define_valid_pixels(
    xrds_dem: xr.Dataset,
    land: gpd.GeoDataFrame | None = None,
    slope_threshold: float = 30,
    range_threshold: int = 300,
    simplify: bool = True,
) -> tuple[float, xr.Dataset]:
    """
    For a dem, compute water mask and
    valid pixels by applying the
    following steps:
       - Filter by slope
       - Compute min and max values
       - Compute the distribution of height
       - Find the range that contains the most of pixels
       - Clean the valid mask by performing dilation
         and removing small holes, with simplify option

    Parameters
    ----------
    xrds_dem: xarray.Dataset
        Dataset containing DEM information
    slope_threshold: float, default = 30
        Maximum slope value accepted
    range_threshold: int, default = 300
        Range of altitude for pixels selection
    simplify: boo, default = True
        Activate cleaning step

    Returns
    -------
    pvalid: float
    dem: xr.Dataset
    """
    dem = xrds_dem.copy()
    # Filter water mask
    if land is not None:
        dem["water_mask"] = compute_water_mask(dem, land)
    # Compute valid pixels
    dem["valid"] = compute_valid_mask(
        dem,
        slope_threshold=slope_threshold,
        range_threshold=range_threshold,
        simplify=simplify,
    )
    # Compute number of valid pixels and percentage
    pvalid = (dem["valid"].data == 1).sum() / dem["valid"].data.size
    return pvalid, dem


def polygonize_mask(
    mask: np.ndarray,
    transform: affine.Affine,
    label: int = 1,
) -> list[Polygon]:
    """
    Polygonize a mask

    Parameters
    ----------
    mask: np.ndarray
        Mask
    transform: affine.Affine
        Affine transform
    label: int, default = 1
        Label to polygonize

    Returns
    -------
    polys: List[Polygon]
    """
    # Convert mask
    data = mask.astype("uint8")
    msk_shapes = rio_features.shapes(data, connectivity=4, transform=transform)
    polys = []
    for vec, val in msk_shapes:
        if val == float(label):
            polys.append(shape(vec))
    return polys


def define_valid_zones(
    tiles: list[str],
    land: gpd.GeoDataFrame | None = None,
    slope_threshold: float = 30,
    range_threshold: int = 300,
    simplify: bool = True,
    poly_simplify: int = 0,
) -> tuple[float, int, MultiPolygon]:
    """
    For a list of tile IDs, read DEM,
    compute water mask and
    valid pixels by applying the
    following steps:
       - Filter by slope
       - Compute min and max values
       - Compute the distribution of height
       - Find the range that contains the most of pixels
       - Clean the valid mask by performing dilation
         and removing small holes, with simplify option
    Then polygonize the result.

    Parameters
    ----------
    tiles: List[str]
        Dataset containing DEM information
    slope_threshold: float, default = 30
        Maximum slope value accepted
    range_threshold: int, default = 300
        Range of altitude for pixels selection
    simplify: boo, default = True
        Activate cleaning step
    simplified: int, default = 0
        If greater than zero, simplify polygons

    Returns
    -------
    pvalid: float
    nbvalid: int
    polygons: MultiPolygon
    """
    # Read DEM
    dem = get_dem_from_tiles(tiles)
    # Filter water mask
    if land is not None:
        dem["water_mask"] = compute_water_mask(dem, land)
    # Compute valid pixels
    valid = compute_valid_mask(
        dem,
        slope_threshold=slope_threshold,
        range_threshold=range_threshold,
        simplify=simplify,
    )
    # Compute polygons
    polys = polygonize_mask(valid.data, dem.transform)
    gpd_polys = gpd.GeoDataFrame(geometry=polys, crs=dem.crs)
    if poly_simplify != 0:
        gpd_polys = gpd.GeoDataFrame(
            geometry=gpd_polys.simplify(poly_simplify), crs=dem.crs
        )
    # Transform polygons to mask
    valid = xr.DataArray(
        rio_features.geometry_mask(
            gpd_polys["geometry"],
            out_shape=(dem.sizes["y"], dem.sizes["x"]),
            transform=dem.transform,
        ),
        coords=dem.coords,
        dims=("y", "x"),
    )
    valid.data = ~valid.data
    # Compute number and percentage of valid pixels
    nbvalid = (valid.data == 1).sum()
    pvalid = (valid.data == 1).sum() / valid.data.size
    return pvalid, nbvalid, MultiPolygon(list(gpd_polys.geometry))
