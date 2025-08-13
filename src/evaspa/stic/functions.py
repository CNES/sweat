# Copyright: (c) 2025 LIST / CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions to compute several functions
for STIC model
"""

from __future__ import annotations

import datetime as dt
from math import exp

import numpy as np
import numpy.typing as npt
from numba import float32 as f32  # to define f32
from numba import njit
from numba.types import Tuple
from pyproj import CRS, Transformer
from scipy.constants import c, h, k, pi

# ruff: noqa: PLR2004

# Stefan-Boltzmann constant
CST_SB = ((2 * pi**5) * (k**4)) / (15 * (c**2) * (h**3))
# Ratio molecular weight of water vapor/dry air
MWRATIO = 0.622
# Standard pressure (hPa)
STANDARD_PRESSURE = 1013.25
# Specific gas constant for dry air (J/kg/K)
R_DRY = 286.9
# Specific gas constant for wet air (J/kg/K)
R_WET = 461.5
# Kelvin
CST_KELVIN = 273.15
# Specific heat of wet air (J/kg/K)
CP_WET = 1846.0
# Specific heat of air at constant pressure (J/kg/K)
CP_DRY = 1005.0
# Psychrometric constant (hpa/K)
PSYCHROMETRIC_CST = 0.67
# Tetens parameters
# https://en.wikipedia.org/wiki/Tetens_equation
A_TETENS = 6.13753  # TODO: bibliographic reference 6.1078
B_TETENS = 17.27
C_TETENS = 237.3


def convert_to_celsius(lst: npt.ArrayLike) -> npt.NDArray:
    """
    Converting from Kelvin to Celsus degree

    Parameters
    ----------
    lst: np.array_like
        Temperature in kelvin

    Returns
    -------
    lst: np.array
        Temperature in Celsius
    """
    return np.array(lst) - CST_KELVIN


def convert_to_local_time(
    date: dt.datetime,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    crs: CRS | None = None,
) -> npt.NDArray:
    """
    Converting time from UTM to local solar time (in seconds)

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
    Converting 2m air temperature and dewpoint temprature
    to relative humidity in percentage.

    Notes
    -----
    The coefficients are provided by
    Alduchov, O. A., and R. E. Eskridge, 1996:
    Improved Magnus Form Approximation of Saturation Vapor Pressure.
    J. Appl. Meteor. Climatol., 35, 601-609

    Parameters
    ----------
    t2m: np.array_like
        2m air temperature (in Kelvin)
    d2m : np.array_like (in Kelvin)
        dewpoint temperature
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


@njit(
    (f32)(f32),
    nogil=True,
    cache=True,
)
def _tetens(t: float) -> float:
    """
    Tetens equation
    """
    # Saturation vapor pressure at surface temperature TS (unit hPa)
    return f32(A_TETENS) * exp((f32(B_TETENS) * t) / (t + f32(C_TETENS)))


@njit(
    (f32)(f32),
    nogil=True,
    cache=True,
)
def _tetens_derivative(t: float) -> float:
    """
    Derivative of Tetens equation
    """
    return f32(B_TETENS) * f32(C_TETENS) * _tetens(t) / (t + f32(C_TETENS)) ** 2


@njit(
    Tuple((f32,) * 11)(*(f32,) * 4),
    nogil=True,
    cache=True,
)
def f_psychrometrics(
    ts: float, ta: float, td: float, rh: float
) -> tuple[
    float, float, float, float, float, float, float, float, float, float, float
]:
    """
    Compute psychrometrics:
    - esstar: saturation vapor pressure at surface temperature, TS (unit hPa)
    - eastar: saturated vapor pressure at air temperature (hPa)
    - ea: atmosphere vapor pressure (hPa)
    - da: atmosphere vapor pressure deficit (hPa)
    - slope: slope of saturation vapor pressure versus air temperature at TA (hPa/degC)
    -s1,s2,s3,s4 : splopes of saturation vapor pressure versus temperature
    - rho: air density (kg.m-3)
    - cp: specific heat of air at constant pressure (J.kg-1.K-1)

    Notes
    -----
    See:
    - https://en.wikipedia.org/wiki/Tetens_equation

    Parameters
    ----------
    ts: float
        Surface temperature (Celsius)
    ta: float
        2m air temperature (Celsius)
    td: float
        Dewpoint temperature (Celsius)
    rh: float
        Relative humidity (percentage)

    Returns
    -------
    esstar: float
        Saturation vapor pressure at surface temperature (hPa)
    eastar: float
        Saturation vapor pressure at air temperature (hPa)
    ea: float
        Atmosphere vapour pressure (hPa) at air temperature
    da: float
        Atmosphere vapour pressure deficit (hPa) at the reference height
    slope: float
        Slope of saturation vapor pressure versus air temperature at TA (hPa/degC)
    s1: float
        Slope of saturation vapor pressure versus temperature
    s2: float
        Slope of saturation vapor pressure versus temperature
    s3: float
        Slope of saturation vapor pressure versus temperature
    s4: float
        Slope of saturation vapor pressure versus temperature
    rho: float
        Air density (kg.m-3)
    cp: float
        Specific heat of air at constant pressure (J.kg-1.K-1)
    """

    # Compute saturation vapor pressure of water with Tetens equation
    # Saturation vapor pressure at surface temperature TS (unit hPa)
    esstar = _tetens(ts)
    # Saturated vapor pressure at air temperature (hPa)
    eastar = _tetens(ta)
    # Compute actual vapor pressure of air (unit hPa)
    # using the definition of relative humidity
    ea = (rh / f32(100)) * (eastar)
    # Vapor pressure deficit of air (hPa)
    da = eastar - ea
    # Compute the slope of saturation vapor pressure versus temperature
    # by differentiate Tetens equation
    # slope of saturation vapor pressure versus temperature at Ta (hPa)
    slope = _tetens_derivative(ta)
    # slope of saturation vapor pressure versus temperature at Td (hPa)
    s1 = _tetens_derivative(td)
    # Avoid division by zero
    s2 = (esstar - ea) / (ts - td) if abs(ts - td) > f32(1.0e-7) else s1
    # slope of saturation vapor pressure versus temperature at Ts (hPa)
    s3 = _tetens_derivative(ts)
    s4 = (eastar - ea) / (ta - td) if abs(ta - td) > f32(1.0e-7) else s1
    # Specific humidity
    qref = (f32(MWRATIO) * ea) / (
        f32(STANDARD_PRESSURE) - (f32(1) - f32(MWRATIO)) * ea
    )
    # Water vapor mixing ratio
    r = qref / (f32(1) - qref)
    # Density of dry air
    rho_dry = (
        f32(100)
        * f32(STANDARD_PRESSURE)
        / (f32(R_DRY) * (ta + f32(CST_KELVIN)))
    )
    # Density of air
    rho = rho_dry * (
        (f32(1) + r) / (f32(1) + r / f32(MWRATIO))
    )  # density of air
    # Specific heat of air
    cp = qref * f32(CP_WET) + (f32(1) - qref) * f32(CP_DRY)

    return (
        esstar,
        eastar,
        ea,
        da,
        slope,
        s1,
        s2,
        s3,
        s4,
        rho,
        cp,
    )


@njit(
    Tuple((f32,) * 4)(*(f32,) * 9),
    nogil=True,
    cache=True,
)
def f_stateeq(
    rho: float,
    cp: float,
    alpha: float,
    slope: float,
    phi: float,
    e0: float,
    ea: float,
    e0star: float,
    m: float,
) -> tuple[float, float, float, float]:
    """
    Compute state equation#s

    Parameters
    ----------
    rho: float
        Air density (kg.m-3)
    cp: float
        Specific heat of air at constant pressure (J.kg-1.K-1)
    alpha: float
        Empirical constant accounting for the vapor
        pressure deficit and resistance values
    slope: float
        Slope of saturation vapor pressure
        versus air temperature at TA (hPa/degC)
    phi: float
        Available energy
    e0: float
        Vapor pressure at temperature t0 (hPa)
    ea: float
        Atmosphere vapor pressure (hPa)
    e0star: float
        Saturated vapor pressure at temperature t0 (hPa)
    m: float
        Surface moisture (0-1)

    Returns
    -------
    g_aero: float
        Aerodynamic conductance
    g_surf: float
        Surface conductance
    delta_t: float
        T0-Ta
    ef: float
        Evaporative fraction
    """
    # Aerodynamic conductance
    g_aero = (f32(2) * phi * alpha * slope * f32(PSYCHROMETRIC_CST)) / (
        f32(2) * cp * slope * e0 * rho
        - f32(2) * cp * slope * ea * rho
        - f32(2) * cp * ea * f32(PSYCHROMETRIC_CST) * rho
        + cp * e0 * f32(PSYCHROMETRIC_CST) * rho
        + cp * e0star * f32(PSYCHROMETRIC_CST) * rho
        - cp * m * e0 * f32(PSYCHROMETRIC_CST) * rho
        + cp * m * e0star * f32(PSYCHROMETRIC_CST) * rho
    )

    # Surface conductance
    denominator = (
        cp * e0star**2 * f32(PSYCHROMETRIC_CST) * rho
        - cp * e0**2 * f32(PSYCHROMETRIC_CST) * rho
        - f32(2) * cp * slope * e0**2 * rho
        + f32(2) * cp * slope * ea * e0 * rho
        - f32(2) * cp * slope * ea * e0star * rho
        + f32(2) * cp * slope * e0 * e0star * rho
        + f32(2) * cp * ea * e0 * f32(PSYCHROMETRIC_CST) * rho
        - f32(2) * cp * ea * e0star * f32(PSYCHROMETRIC_CST) * rho
        + cp * m * e0**2 * f32(PSYCHROMETRIC_CST) * rho
        + cp * m * e0star**2 * f32(PSYCHROMETRIC_CST) * rho
        - f32(2) * cp * m * e0 * e0star * f32(PSYCHROMETRIC_CST) * rho
    )
    denominator = max(denominator, f32(0.00001))  # when e0star == e0
    g_surf = (
        -(
            f32(2)
            * (
                phi * alpha * slope * ea * f32(PSYCHROMETRIC_CST)
                - phi * alpha * slope * e0 * f32(PSYCHROMETRIC_CST)
            )
        )
        / denominator
    )

    # T0 - TA
    delta_t = (
        f32(2) * slope * e0
        - f32(2) * slope * ea
        - f32(2) * ea * f32(PSYCHROMETRIC_CST)
        + e0 * f32(PSYCHROMETRIC_CST)
        + e0star * f32(PSYCHROMETRIC_CST)
        - m * e0 * f32(PSYCHROMETRIC_CST)
        + m * e0star * f32(PSYCHROMETRIC_CST)
        + f32(2) * alpha * slope * ea
        - f32(2) * alpha * slope * e0
    ) / (f32(2) * alpha * slope * f32(PSYCHROMETRIC_CST))

    # Evaporative fraction
    ef = -(f32(2) * alpha * slope * ea - f32(2) * alpha * slope * e0) / (
        f32(2) * slope * e0
        - f32(2) * slope * ea
        - f32(2) * ea * f32(PSYCHROMETRIC_CST)
        + e0 * f32(PSYCHROMETRIC_CST)
        + e0star * f32(PSYCHROMETRIC_CST)
        - m * e0 * f32(PSYCHROMETRIC_CST)
        + m * e0star * f32(PSYCHROMETRIC_CST)
    )

    # Adjust the abnormal conductances
    g_aero = min(max(g_aero, f32(0.0001)), f32(0.2))
    g_surf = min(max(g_surf, f32(0.0001)), f32(0.06))

    # Maximum surface-air temperature difference rarely overpasses 20 degC
    delta_t = min(max(delta_t, f32(-10)), f32(20))
    ef = min(max(ef, f32(0.0001)), f32(1.0))

    return (g_aero, g_surf, delta_t, ef)
