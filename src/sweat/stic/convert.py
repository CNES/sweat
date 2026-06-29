# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions to convert variables
used to prepare data for STIC model
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import numpy.typing as npt
from pyproj import CRS, Transformer

from sweat.stic.constant import KELVIN_CST


def convert_kelvin_to_celsius(lst: npt.ArrayLike) -> npt.NDArray:
    """
    Converting from Kelvin to Celsius degree

    Parameters
    ----------
    lst: np.array_like
        Temperature in kelvin

    Returns
    -------
    lst: np.array
        Temperature in Celsius
    """
    return np.array(lst) - KELVIN_CST


def convert_celsius_to_kelvin(lst: npt.ArrayLike) -> npt.NDArray:
    """
    Converting from Celsius to Kelvin degree

    Parameters
    ----------
    lst: np.array_like
        Temperature in kelvin

    Returns
    -------
    lst: np.array
        Temperature in Celsius
    """
    return np.array(lst) + KELVIN_CST


def convert_to_local_time(
    date: dt.datetime,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    crs: CRS | None = None,
) -> npt.NDArray:
    """
    Converting time from UTC to local solar time (in seconds)

    Parameters
    ----------
    date: np.array_like
        List of dates
    x : np.array_like
        X coordinate / Longitude (in degrees)
    y : np.array_like
        Y coordinate / Latitude (in degrees)
    crs : pyproj.CRS
        Coordinate Reference System

    Returns
    -------
    local_time: np.array
        Local solar time
    """
    # Convert to lat/lon
    if crs is None:
        crs = CRS(4326)
    x_grid, y_grid = np.meshgrid(x, y)
    if np.isscalar(x) and np.isscalar(y):
        x_grid = x  # type: ignore
        y_grid = y  # type: ignore
    # Convert to lat/lon
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    # Apply transformation to the grid
    lon, _ = transformer.transform(x_grid, y_grid)
    # Time delta
    time_delta = lon / 15.0
    time_ls = (
        float(date.hour)
        + float(date.minute) / 60.0
        + float(date.second) / 3600.0
        + time_delta
    )
    time_ls = np.where(time_ls < 0, time_ls + 24, time_ls)
    time_ls = np.where(time_ls >= 24, time_ls % 24, time_ls)
    return time_ls * 3600


def convert_to_rh(
    t2m: npt.ArrayLike, d2m: npt.ArrayLike, b: float = 17.625, c: float = 243.04
) -> npt.NDArray:
    """
    Converting 2m air temperature and dewpoint temperature
    to relative humidity in percentage.

    Notes
    -----
    The relative humidity is calculated from air temperature and
    dew-point temperature with the following formula:

    $$
    RH = 100 \\times e^{bc\\frac{T_{D} - T_{a}}{\\left( T_{D} + c \\right)
    \\left( c + T_{a} \\right)}}
    $$

    with b and c coefficient values are provided by
    Alduchov, O. A., and R. E. Eskridge, 1996:
    Improved Magnus Form Approximation of Saturation Vapor Pressure.
    J. Appl. Meteor. Climatol., 35, 601-609

    Parameters
    ----------
    t2m: np.array_like
        2m air temperature (in Celsius)
    d2m : np.array_like
        dewpoint temperature (in Celsius)
    b : float
        Parameter
    C : float
        Parameter

    Returns
    -------
    rh: np.array
        Relative humidity in percentage
    """
    # Converting from dewpoint temperature to rh (0-1)
    rh = np.exp(
        b
        * c
        * (np.array(d2m) - np.array(t2m))
        / ((c + np.array(d2m)) * (c + np.array(t2m)))
    )
    rh = np.clip(rh, 0.0, 1.0)
    # Return in percentage
    return rh * 100
