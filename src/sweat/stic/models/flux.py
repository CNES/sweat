# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for STIC flux computation
"""

import numpy as np
from numba import boolean, njit
from numba import float32 as f32
from numba import float64 as f64
from numba.types import Tuple

from sweat.stic.constant import PSYCHROMETRIC_CST

# Constants
KRN = 0.6
CG_MIN = 0.05  # for wet surface
CG_MAX = 0.35  # for dry surface, in water controlled systems
TG_MIN = 74000  # for wet surface
TG_MAX = 100000  # for dry surface


@njit(
    [f32(*(f32,) * 4), f64(*(f64,) * 4)],
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
    Compute soil heat flux

    Notes
    -----
    The soil heat flux, called G flux is computed with
    the method described in
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

    sol_noon = 12.0 * 60.0 * 60.0
    tg0 = sol_noon - local_time

    # Estimating GHF according to Santanello and Friedl (2003)
    cg = (1 - m) * CG_MAX + m * CG_MIN
    tg = (1 - m) * TG_MAX + m * TG_MIN

    g_flux = rn_soil * cg * np.cos(2.0 * np.pi * (tg0 + 10800.0) / tg)
    if rn_soil < 0.0:
        g_flux = -g_flux

    return g_flux


@njit(
    [
        Tuple((f32,) * 2)(*(f32,) * 8, boolean),
        Tuple((f64,) * 2)(*(f64,) * 8, boolean),
    ],
    nogil=True,
    cache=True,
)
def initiate_le_h_fluxes(
    slope: float,
    g_aero: float,
    g_surf: float,
    g_r: float,
    phi: float,
    da: float,
    rho: float,
    cp: float,
    is_stressed: bool,
) -> tuple[float, float]:
    """
    Initiate latent heat flux and sensible heat flux

    Notes
    -----
    The latent heat flux and the sensible heat flux
    are initialized with the method provided by
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
    g_r: float
        Radiative conductance (m.s-1)
    phi: float
        Available energy (W.m-2)
    da: float
        Atmosphere vapor pressure deficit (hPa) at the reference height
    rho: float, _
        Air density (kg/m^3)
    cp: float
        Specific heat at constant pressure (J/kg/K)
    is_stressed: bool
        Water stress indicator

    Returns
    -------
    le_flux: float
        Latent heat flux
    h_flux: float
        Sensible heat flux
    """
    # Omega computation in function of water stress
    if is_stressed:
        omega = ((slope / PSYCHROMETRIC_CST) + 1.0 + (g_r / g_aero)) / (
            (slope / PSYCHROMETRIC_CST)
            + 1.0
            + g_aero / g_surf
            + g_r / g_aero
            + g_aero / g_surf
        )
    else:
        omega = ((slope / PSYCHROMETRIC_CST) + 1.0) / (
            (slope / PSYCHROMETRIC_CST) + 1.0 + g_aero / g_surf
        )
    # Calculate ET and H based on initial results from state eqs.
    # with McNaughton and Jarvis (1986)
    le_flux_eq = (phi * (slope / PSYCHROMETRIC_CST)) / (
        (slope / PSYCHROMETRIC_CST) + 1.0
    )
    le_flux_imp = (cp * rho / PSYCHROMETRIC_CST) * g_surf * da
    le_flux = omega * le_flux_eq + (1.0 - omega) * le_flux_imp

    h_flux = (
        PSYCHROMETRIC_CST * phi * (1.0 + g_aero / g_surf)
        - rho * cp * g_aero * da
    ) / (
        slope + PSYCHROMETRIC_CST * (1.0 + g_aero / g_surf)
    )  # Deduced from the PM equation

    return le_flux, h_flux


@njit(
    [Tuple((f32,) * 2)(*(f32,) * 11), Tuple((f64,) * 2)(*(f64,) * 11)],
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

    Notes
    -----
    The sensible heat flux is computed with
    $$
    H = \\frac{\\gamma \\left( R_{n} - G \\right) \\left( 1 +
    \\frac{g_{a}}{g_{s}} \\right) - \\rho c_{p} g_{a} D_{a} }
    {\\Delta + \\gamma \\left( 1 + \\frac{g_{a}}{g_{s}} \\right) }
    $$
    The latent heat flux is computed with
    $$
    LE = \\frac{\\rho c_{p}g_{a}g_{s}}
    {\\gamma(g_{a} + g_{s})}
    \\left( \\Delta(T_{0} - T_{a}) + D_{a} \\right)
    $$

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
    le_flux = (
        (rho * cp / PSYCHROMETRIC_CST)
        * ((g_aero * g_surf) / (g_aero + g_surf))
        * (slope * (t0 - ta) + da)
    )

    if (le_flux < 0.0) & (le_flux < phi):
        le_flux = rho * cp * g_aero * (e0 - ea) / PSYCHROMETRIC_CST

    h_flux = (
        PSYCHROMETRIC_CST * phi * (1.0 + g_aero / g_surf)
        - rho * cp * g_aero * da
    ) / (
        slope + PSYCHROMETRIC_CST * (1.0 + g_aero / g_surf)
    )  # Deduced from the PM equation

    return le_flux, h_flux
