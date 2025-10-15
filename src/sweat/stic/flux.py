# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for STIC flux computation
"""

import numpy as np
from numba import float32 as f32  # to define f32
from numba import njit
from numba.types import Tuple

from sweat.stic.constant import PSYCHROMETRIC_CST

# Constants
KRN = 0.6
CG_MIN = 0.05  # for wet surface
CG_MAX = 0.35  # for dry surface, in water controlled systems
TG_MIN = 74000  # for wet surface
TG_MAX = 100000  # for dry surface


@njit(
    (f32)(
        f32,
        f32,
        f32,
        f32,
    ),
    nogil=True,
    cache=True,
)
def compute_g_flux(
    rn: float,
    lai: float,
    local_time: float,
    m: float,
) -> float:
    """
    Compute soil heat flux, called G flux.

    Notes
    -----
    Santanello, J. A., and M. A. Friedl, 2003:
    Diurnal Covariation in Soil Heat Flux and
    Net Radiation. J. Appl. Meteor. Climatol., 42, 851-862

    Parameters
    ----------
    rn: float
        Net radiation
    lai : float
        Leaf Area Index
    local_time : float
        Local time in seconds
    m : float
        Surface moisture

    Returns
    -------
    g_flux: float
        G flux
    """
    rn_soil = rn * np.exp(-KRN * lai)

    sol_noon = f32(12) * f32(60) * f32(60)
    tg0 = sol_noon - local_time

    # Estimating GHF according to Santanello and Friedl (2003)
    cg = (f32(1) - m) * f32(CG_MAX) + m * f32(CG_MIN)
    tg = (f32(1) - m) * f32(TG_MAX) + m * f32(TG_MIN)

    g_flux = rn_soil * cg * np.cos(f32(2) * np.pi * (tg0 + f32(10800)) / tg)
    if rn_soil < f32(0):
        g_flux = -g_flux

    return g_flux


@njit(
    Tuple((f32,) * 2)(*(f32,) * 7),
    nogil=True,
    cache=True,
)
def initiate_le_h_fluxes(
    slope: float,
    g_aero: float,
    g_surf: float,
    phi: float,
    da: float,
    rho: float,
    cp: float,
) -> tuple[float, float]:
    """
    Initiatelatent heat flux and sensible heat flux.

    Notes
    -----
    P.G. Jarvis, K.G. McNaughton,
    Stomatal Control of Transpiration: Scaling Up from Leaf to Region,
    Advances in Ecological Research, 15, 1986

    Parameters
    ----------
    slope: float
        Slope of saturation vapor pressure versus
        air temperature at TA (hPa/degC)
    g_aero: float
        Aerodynamic conductance (m.s-1)
    g_surf: float
        Surface conductance (m.s-1)
    phi: float
        Available energy (W.m-2)
    da: float
        Atmosphere vapor pressure deficit (hPa) at the reference height
    rho: float, _
        Air density (kg/m^3)
    cp: float
        Specific heat at constant pressure (J/kg/K)

    Returns
    -------
    le_flux: float
        Latent heat flux
    h_flux: float
        Sensible heat flux
    """
    # Calculate ET and H based on initial results from state eqs.
    # with McNaughton and Jarvis (1986)
    omega = ((slope / f32(PSYCHROMETRIC_CST)) + f32(1)) / (
        (slope / f32(PSYCHROMETRIC_CST)) + f32(1) + g_aero / g_surf
    )
    le_flux_eq = (phi * (slope / f32(PSYCHROMETRIC_CST))) / (
        (slope / f32(PSYCHROMETRIC_CST)) + f32(1)
    )
    le_flux_imp = (
        (cp * f32(0.0289644) / f32(PSYCHROMETRIC_CST)) * g_surf * f32(40) * da
    )
    le_flux = omega * le_flux_eq + (f32(1) - omega) * le_flux_imp

    h_flux = (
        f32(PSYCHROMETRIC_CST) * phi * (f32(1) + g_aero / g_surf)
        - rho * cp * g_aero * da
    ) / (
        slope + f32(PSYCHROMETRIC_CST) * (f32(1) + g_aero / g_surf)
    )  # Deduced from the PM equation

    return le_flux, h_flux


@njit(
    Tuple((f32,) * 2)(*(f32,) * 11),
    nogil=True,
    cache=True,
)
def compute_le_h_fluxes(
    slope: float,
    g_aero: float,
    g_surf: float,
    phi: float,
    da: float,
    ta: float,
    t0: float,
    ea: float,
    e0: float,
    rho: float,
    cp: float,
) -> tuple[float, float]:
    """
    Compute latent heat flux and sensible heat flux.

    Parameters
    ----------
    slope: float
        Slope of saturation vapor pressure versus
        air temperature at TA (hPa/degC)
    g_aero: float
        Aerodynamic conductance (m.s-1)
    g_surf: float
        Surface conductance (m.s-1)
    phi: float
        Available energy (W.m-2)
    da: float
        Atmosphere vapor pressure deficit (hPa) at the reference height
    ta: float
        Air temperature (degC)
    t0: float
        Air/Canopy temperature (degC)
    ea: float
        Atmosphere vapor pressure (hPa)
    e0: float
        Vapor pressure at air/canopy (hPa)
    rho: float, _
        Air density (kg/m^3)
    cp: float
        Specific heat at constant pressure (J/kg/K)

    Returns
    -------
    le_flux: float
        Latent heat flux
    h_flux: float
        Sensible heat flux
    """
    # TODO: To check difference with STIC-JPL to compute LE
    le_flux = (
        (rho * cp / f32(PSYCHROMETRIC_CST))
        * ((g_aero * g_surf) / (g_aero + g_surf))
        * (slope * (t0 - ta) + da)
    )

    if (le_flux < f32(0.0)) & (le_flux < phi):
        le_flux = rho * cp * g_aero * (e0 - ea) / f32(PSYCHROMETRIC_CST)

    h_flux = (
        f32(PSYCHROMETRIC_CST) * phi * (f32(1) + g_aero / g_surf)
        - rho * cp * g_aero * da
    ) / (
        slope + f32(PSYCHROMETRIC_CST) * (f32(1) + g_aero / g_surf)
    )  # Deduced from the PM equation

    return le_flux, h_flux
