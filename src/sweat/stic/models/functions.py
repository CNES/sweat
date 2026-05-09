# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions to compute several functions
for STIC model
"""

from __future__ import annotations

from math import copysign, exp

from numba import float32 as f32
from numba import float64 as f64
from numba import int8 as i8
from numba import njit
from numba.types import Tuple

from sweat.common.flux import CST_SB
from sweat.stic.constant import (
    CP_DRY,
    CP_WET,
    KELVIN_CST,
    MWRATIO,
    PSYCHROMETRIC_CST,
    R_DRY,
    STANDARD_PRESSURE,
)

# Tetens parameters, see https://en.wikipedia.org/wiki/Tetens_equation
A_TETENS = 6.13753
B_TETENS = 17.27
C_TETENS = 237.3


@njit(
    [i8(f32), i8(f64)],
    nogil=True,
    cache=True,
)
def _sign(x):
    return i8(copysign(1, x))


@njit(
    [f32(f32), f64(f64)],
    nogil=True,
    cache=True,
)
def _tetens(t: float) -> float:
    """
    Calculate the saturation vapor pressure of water using Tetens equation

    Notes
    -----
    The Tetens's formula is
    $$
    e^{*} = a \\times e^{\\frac{bT}{T + c}}
    $$
    where a, b and c are parameters.

    See:
    - [tetens' equation](https://en.wikipedia.org/wiki/Tetens_equation)

    Parameters
    ----------
    t: float
        Temperature (degC)

    Returns
    -------
    svp: float
        Saturation vapor pressure (hPa)
    """
    # Saturation vapor pressure at surface temperature TS (unit hPa)
    return A_TETENS * exp((B_TETENS * t) / (t + C_TETENS))


@njit(
    [f32(f32), f64(f64)],
    nogil=True,
    cache=True,
)
def _tetens_derivative(t: float) -> float:
    """
    Compute the derivative of Tetens equation

    Notes
    -----
    Calculate the slope of the saturation vapor pressure of water
    versus temperature using the derivative of Tetens equation

    See:
    - [Tetens' equation](https://en.wikipedia.org/wiki/Tetens_equation)

    Parameters
    ----------
    t: float
        Temperature (degC)

    Returns
    -------
    slope: float
        Slope of saturation vapor pressure (hPa/degC)
    """
    return B_TETENS * C_TETENS * _tetens(t) / (t + C_TETENS) ** 2


@njit(
    [Tuple((f32,) * 11)(*(f32,) * 4), Tuple((f64,) * 11)(*(f64,) * 4)],
    nogil=True,
    cache=True,
)
def compute_psychrometrics(
    ts: float, ta: float, td: float, rh: float
) -> tuple[float, ...]:
    """
    Compute psychrometrics

    The variables computed are:

    - esstar: saturation vapor pressure at surface temperature, TS (unit hPa)
    - eastar: saturation vapor pressure at air temperature (hPa)
    - ea: actual vapor pressure of air (hPa)
    - da: Vapor pressure deficit of air (hPa)
    - slope: slope of saturation vapor pressure versus air temperature
    at Ta (hPa/degC)
    - s1,s2,s3,s4 : slopes of saturation vapor pressure versus temperature
    (hPa/degC)
    - rho: air density (kg.m-3)
    - cp: specific heat of air at constant pressure (J.kg-1.K-1)

    Notes
    -----
    The vapor pressure $e_{a}$ at air temperature is equal to
    $$
    e_{a} = e_{a}^{\\star}\\frac{RH}{100}
    $$

    The vapor pressure deficit of air $D_a$ is equal to
    $$
    D_{a} = e_{a}^{\\star} - e_{a}
    $$

    The slope of the saturation vapor pressure vs temperature
    at air temperature is defined with
    $$
    \\Delta = abc\\frac{e^{\\frac{b \\times T_{a}}{T_{a} + c}}}
    {\\left( T_{a} + c \\right)^{2}}
    $$
    with a, b and c Tetens' parameters.

    The slopes $s_1$ and $s_3$ can be expressed as
    $$
    s_{1} = \\left(45.03  + 3.014 \\times T_{D} +  0.05345 * T_{D}^{2}
    +  0.00224 \\times T_{D}^{3}\\right) \\times 1e^{-2}
    $$
    $$
    s_{3} = \\left(45.03  + 3.014 \\times LST +  0.05345 * LST^{2}
    +  0.00224 \\times LST^{3}\\right) \\times 1e^{-2}
    $$

    The slopes $s_2$ and $s_4$ can be expressed as
    $$
    s_{2} = \\frac{e_{s}^{\\star} - e_{a}}{LST - T_{D}}
    $$
    $$
    s_{4} = \\frac{e_{a}^{\\star} - e_{a}}{T_{a} - T_{D}}
    $$

    The specific humidity is the ratio of the mass of the vapor in a sample,
    to the mass of the moist air in the sample of air and it is defined
    as, see [here](https://web.stanford.edu/group/efmh/jacobson/FAMbook/Chap2.pdf)
    $$
    q_{ref} = MW_{ratio}\\frac{e_{a}}{P - (1 - MW_{ratio})e_{a}}
    $$
    As a reminder, $MW_{ratio}$ is ratio molecular weight of
    water vapor/dry air equal to 0.622. But it can also
    be expressed as the ratio between the specific gas constant for dry air
    $R_{dry}$ and the specific gas constant for water vapor
    $R_{vapor}$,
    $$
    MW_{ratio} = \\frac{R_{dry}}{R_{vapor}} = 0.622
    $$
    The pressure $p$, the temperature $T$, the density $\\rho$ and
    the water vapor mixing ratio $r$, defined as the mass of water
    vapor in the sample per unit mass of dry air are connected by
    the equation of state for moist air,
    see [here](https://web.stanford.edu/group/efmh/jacobson/FAMbook/Chap2.pdf)
    Therefore the air density can be expressed as follows:
    $$
    \\rho = \\frac{P}{R_{dry}T}\\frac{1 + r}{1 + r/MW_{ratio}}
    $$
    The ratio $r$ can be expressed using the specific humidity.
    Indeed, the specific humidity can be expressed as
    $$
    q_{ref} = \\frac{\\rho_{vapor}}{\\rho_{dry} + \\rho_{vapor}}
    $$
    Yet, r is equal to
    $$
    r = \\frac{\\rho_{vapor}}{\\rho_{dry}}
    $$
    The specific heat of moist air $c_p$ can be expressed with
    the specific heat of dry air and vapor:
    $$
    c_{p} = \\frac{M_{dry}c_{p_{dry}} + M_{vapor}c_{p_{vapor}}}
    {M_{dry} + M_{vapor}}
    $$
    With $M_{dry}$ and $M_{vapor}$ the masses respectively of dry air
    and water vapor.
    Since
    $$
    q_{ref} = \\frac{M_{vapor}}{M_{dry} + M_{vapor}}
    $$
    So,
    $$
    c_{p} = (1 - q_{ref})c_{p_{dry}} + q_{ref}c_{p_{vapor}}
    $$

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
        Atmosphere vapor pressure (hPa) at air temperature
    da: float
        Atmosphere vapor pressure deficit (hPa) at the reference height
    slope: float
        Slope of saturation vapor pressure versus
        air temperature at TA (hPa/degC)
    s1: float
        Slope of saturation vapor pressure versus temperature (hPa/degC)
    s2: float
        Slope of saturation vapor pressure versus temperature (hPa/degC)
    s3: float
        Slope of saturation vapor pressure versus temperature (hPa/degC)
    s4: float
        Slope of saturation vapor pressure versus temperature (hPa/degC)
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
    ea = (rh / 100) * (eastar)
    # Vapor pressure deficit of air (hPa)
    da = eastar - ea
    # Compute the slope of saturation vapor pressure versus temperature
    # by differentiate Tetens equation
    # slope of saturation vapor pressure versus temperature at Ta (hPa)
    slope = _tetens_derivative(ta)
    # slope of saturation vapor pressure versus temperature at Td (hPa)
    s1 = (45.03 + 3.014 * td + 0.05345 * td**2 + 0.00224 * td**3) * 1e-2
    # Avoid division by zero
    s2 = (esstar - ea) / (ts - td) if abs(ts - td) > f32(1.0e-7) else s1
    # slope of saturation vapor pressure versus temperature at Ts (hPa)
    s3 = (45.03 + 3.014 * ts + 0.05345 * ts**2 + 0.00224 * ts**3) * 1e-2
    s4 = (eastar - ea) / (ta - td) if abs(ta - td) > f32(1.0e-7) else s1
    # Specific humidity
    qref = (MWRATIO * ea) / (STANDARD_PRESSURE - (1 - MWRATIO) * ea)
    # Water vapor mixing ratio
    r = qref / (1 - qref)
    # Density of dry air
    rho_dry = 100 * STANDARD_PRESSURE / (R_DRY * (ta + KELVIN_CST))
    # Density of air computed with the equation of state for moist air
    rho = rho_dry * ((1 + r) / (1 + r / MWRATIO))
    # Specific heat of air
    cp = qref * CP_WET + (1 - qref) * CP_DRY

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
    [Tuple((f32,) * 4)(*(f32,) * 9), Tuple((f64,) * 4)(*(f64,) * 9)],
    nogil=True,
    cache=True,
)
def compute_state_equations(
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
    Compute STIC state equations with modified
    Priestley Taylor and Penman Monteith

    Notes
    -----
    The state equations are defined by
    $$
    g_{a} = \\frac{R_{n} - G}{\\rho c_{p}
    \\left( \\left( T_{0} - T_{a} \\right)
    + \\frac{e_{0} - e_{a}}{\\gamma} \\right)}
    $$
    $$
    g_{s} = g_{a}\\frac{e_{0} - e_{a}}{e_{0}^{\\star} - e_{0}}
    $$
    $$
    T_{0} = T_{a} + \\left( \\frac{e_{0} - e_{a}}{\\gamma} \\right)
    \\left( \\frac{1 - EF}{EF} \\right)
    $$
    $$
    EF = \\frac{2\\alpha\\Delta}{2\\Delta +
    2\\gamma + \\gamma\\frac{g_{a}}{g_{s}}(1 + M)}
    $$

    See:

    - Mallick, K. et al. (2015). Reintroducing radiometric surface
    temperature into the Penman-Monteith formulation.
    Water Resources Research, 51, 6214-6243

    Parameters
    ----------
    rho: float
        Air density (kg.m-3)
    cp: float
        Specific heat capacity of air
        at constant pressure (J.kg-1.K-1)
    alpha: float
        Empirical constant accounting for the vapor
        pressure deficit and resistance values
    slope: float
        Slope of saturation vapor pressure
        versus air temperature at Ta (hPa/degC)
    phi: float
        Available energy (W.m-2)
    e0: float
        Vapor pressure at the reference height (hPa)
    ea: float
        Atmosphere vapor pressure (hPa)
    e0star: float
        Saturation vapor pressure at the reference height (hPa)
    m: float
        Surface moisture (0-1)

    Returns
    -------
    g_aero: float
        Aerodynamic conductance (m.s-1)
    g_surf: float
        Surface conductance (m.s-1)
    delta_t: float
        Difference between temperature at the reference height T0
        and air temperature Ta (degC)
    ef: float
        Evaporative fraction (0-1)
    """
    epsilon = f32(1e-7)  # Small value to prevent division by zero

    # Aerodynamic conductance
    g_aero_den = (
        2 * cp * slope * e0 * rho
        - 2 * cp * slope * ea * rho
        - 2 * cp * ea * PSYCHROMETRIC_CST * rho
        + cp * e0 * PSYCHROMETRIC_CST * rho
        + cp * e0star * PSYCHROMETRIC_CST * rho
        - cp * m * e0 * PSYCHROMETRIC_CST * rho
        + cp * m * e0star * PSYCHROMETRIC_CST * rho
    )
    g_aero_den = (
        g_aero_den if abs(g_aero_den) > epsilon else _sign(g_aero_den) * epsilon
    )
    g_aero = (2 * phi * alpha * slope * PSYCHROMETRIC_CST) / (g_aero_den)
    # Adjust the abnormal conductances
    g_aero = min(max(g_aero, 0.0001), 0.1)

    # Surface conductance
    g_surf_den = (
        cp * e0star**2 * PSYCHROMETRIC_CST * rho
        - cp * e0**2 * PSYCHROMETRIC_CST * rho
        - 2 * cp * slope * e0**2 * rho
        + 2 * cp * slope * ea * e0 * rho
        - 2 * cp * slope * ea * e0star * rho
        + 2 * cp * slope * e0 * e0star * rho
        + 2 * cp * ea * e0 * PSYCHROMETRIC_CST * rho
        - 2 * cp * ea * e0star * PSYCHROMETRIC_CST * rho
        + cp * m * e0**2 * PSYCHROMETRIC_CST * rho
        + cp * m * e0star**2 * PSYCHROMETRIC_CST * rho
        - 2 * cp * m * e0 * e0star * PSYCHROMETRIC_CST * rho
    )
    g_surf_den = (
        g_surf_den if abs(g_surf_den) > epsilon else _sign(g_surf_den) * epsilon
    )
    g_surf = -(
        2
        * (
            phi * alpha * slope * ea * PSYCHROMETRIC_CST
            - phi * alpha * slope * e0 * PSYCHROMETRIC_CST
        )
    ) / (g_surf_den)
    # Adjust the abnormal conductances
    g_surf = min(max(g_surf, 0.0001), 0.1)

    # T0 - TA
    delta_t_den = 2 * alpha * slope * PSYCHROMETRIC_CST
    delta_t_den = (
        delta_t_den
        if abs(delta_t_den) > epsilon
        else _sign(delta_t_den) * epsilon
    )
    delta_t = (
        2 * slope * e0
        - 2 * slope * ea
        - 2 * ea * PSYCHROMETRIC_CST
        + e0 * PSYCHROMETRIC_CST
        + e0star * PSYCHROMETRIC_CST
        - m * e0 * PSYCHROMETRIC_CST
        + m * e0star * PSYCHROMETRIC_CST
        + 2 * alpha * slope * ea
        - 2 * alpha * slope * e0
    ) / (delta_t_den)
    # Maximum surface-air temperature difference rarely overpasses 20 degC
    delta_t = min(max(delta_t, -10), 20)

    # Evaporative fraction
    ef_den = (
        2 * slope * e0
        - 2 * slope * ea
        - 2 * ea * PSYCHROMETRIC_CST
        + e0 * PSYCHROMETRIC_CST
        + e0star * PSYCHROMETRIC_CST
        - m * e0 * PSYCHROMETRIC_CST
        + m * e0star * PSYCHROMETRIC_CST
    )
    ef_den = ef_den if abs(ef_den) > epsilon else _sign(ef_den) * epsilon
    ef = -(2 * alpha * slope * ea - 2 * alpha * slope * e0) / (ef_den)
    # Clip value for EF
    ef = min(max(ef, 0.0001), 1.0)

    return (g_aero, g_surf, delta_t, ef)


@njit(
    [f32(*(f32,) * 8), f64(*(f64,) * 8)],
    nogil=True,
    cache=True,
    inline="always",
)
def compute_canopy_air_saturation_vapor_pressure(
    le_flux: float,
    ea: float,
    esstar: float,
    g_a: float,
    g_s: float,
    rho: float,
    cp: float,
    gamma: float,
) -> float:
    """
    Compute the saturation vapor pressure at canopy/air.

    Notes
    -----
    The formula is defined by
    $$
    e_{0}^{*} = e_{a} + \\frac{\\gamma LE\\left(g_{a} + g_{s} \\right)}
    {\\rho c_{p}g_{a}g_{s}}
    $$

    Parameters
    ----------
    le_flux: float
        Latent heat flux [W/m^2]
    ea: float
        Actual vapor pressure [hPa]
    esstar: float
        Saturated vapor pressure [hPa]
    g_a: float
        Aerodynamic conductance [mol/m^2/s]
    g_s: float
        Conductance of stomata [mol/m^2/s]
    rho: float, _
        Air density (kg/m^3)
    cp: float
        Specific heat at constant pressure (J/kg/K)
    gamma: float
        Psychrometric constant (hPa/°C)

    Returns
    -------
    e0star: float
        Canopy/air saturated vapor pressure
    """
    e0star = ea + (gamma * le_flux * (g_a + g_s)) / (rho * cp * g_a * g_s)
    e0star = e0star if e0star >= 0.0 else esstar
    return e0star if e0star < 250.0 else esstar


@njit(
    [f32(*(f32,) * 8), f64(*(f64,) * 8)],
    nogil=True,
    cache=True,
    inline="always",
)
def compute_canopy_air_vapor_pressure_deficit(
    slope: float,
    g_aero: float,
    g_surf: float,
    phi: float,
    da: float,
    ds: float,
    rho: float,
    cp: float,
) -> float:
    """
    Compute the canopy/air vapor pressure deficit.

    Notes
    -----
    The formula is defined by
    $$
    D_{0} = D_{A} + \\frac{\\Delta\\left(R_{n} - G \\right) -
    \\Delta + \\gamma)LE}{\\rho c_{p} g_{a}} = \\frac{g_{a}}{g_{s}}
    \\frac{\\gamma}{\\Delta + \\gamma\\left( 1 + \\frac{g_{a}}{g_{s}}\\right)}
    \\left(\\frac{\\Delta(R_{n} - G)}{\\rho c_{p} g_{a}} + D_{a} \\right)
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
    ds: float
        Atmosphere vapor pressure deficit (hPa) at the surface
    rho: float, _
        Air density (kg/m^3)
    cp: float
        Specific heat at constant pressure (J/kg/K)

    Returns
    -------
    d0: float
        Canopy/air vapor pressure deficit
    """
    d0 = (
        PSYCHROMETRIC_CST
        * (g_aero / g_surf)
        / (slope + PSYCHROMETRIC_CST * (1 + g_aero / g_surf))
    ) * (da + ((slope * phi) / (rho * cp * g_aero)))
    return d0 if d0 >= 0.0 else ds


@njit(
    [f32(*(f32,) * 8), f64(*(f64,) * 8)],
    nogil=True,
    cache=True,
    inline="always",
)
def compute_alpha_coefficient(
    slope: float,
    g_aero: float,
    g_surf: float,
    ta: float,
    t0: float,
    e0star: float,
    ea: float,
    m: float,
) -> float:
    """
    Compute the alpha coefficient, i.e. Priestley-Taylor coefficient.

    Notes
    -----
    The formula is defined by
    $$
    \\alpha = \\frac{\\left(2\\Delta + 2\\gamma +
    \\gamma\\frac{g_{a}}{g_{s}}(1 + M)\\right)
    g_{s}(e_{0}^{\\star} - e_{a})}
    {2\\Delta\\gamma (T_{0} - T_{a})(g_{a} + g_{s}) +
    g_{s}(e_{0}^{\\star} - e_{a})}
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
    ta: float
        Air temperature (degC)
    t0: float
        Air/Canopy temperature (degC)
    e0star: float
        Saturation vapor pressure at air/canopy (hPa)
    ea: float
        Atmosphere vapor pressure (hPa)
    m: float, _
        Soil moisture (0-1)

    Returns
    -------
    alpha: float
        Priestley-Taylor coefficient
    """
    alpha = (
        g_surf
        * (e0star - ea)
        * (
            2 * slope
            + 2 * PSYCHROMETRIC_CST
            + PSYCHROMETRIC_CST * (g_aero / g_surf) * (1 + m)
        )
    ) / (
        2
        * slope
        * (
            PSYCHROMETRIC_CST * (t0 - ta) * (g_aero + g_surf)
            + g_surf * (e0star - ea)
        )
    )
    return min(max(alpha, 0.1), 2.0)


@njit(
    [f32(*(f32,) * 7), f64(*(f64,) * 7)],
    nogil=True,
    cache=True,
    inline="always",
)
def initialize_alpha_coefficient(
    slope: float,
    ta: float,
    da: float,
    cp: float,
    rho: float,
    emis: float,
    rn: float,
) -> float:
    """
    Initialize the alpha coefficient, i.e. Priestley-Taylor coefficient.

    Notes
    -----
    The formula is defined by

    Parameters
    ----------
    slope: float
        Slope of saturation vapor pressure versus
        air temperature at TA (hPa/degC)
    ta: float
        2m air temperature (Celsius)
    da: float
        Atmosphere vapor pressure deficit (hPa) at the reference height
    rho: float
        Air density (kg.m-3)
    cp: float
        Specific heat of air at constant pressure (J.kg-1.K-1)
    emis: float
        Emissivity
    rn: float
        Net radiation

    Returns
    -------
    alpha: float
        Priestley-Taylor coefficient
    """
    # Conduction radiometric
    epsilon = 1.0e-7
    gr = 4 * emis * CST_SB * (ta + KELVIN_CST) ** 3 / (rho * cp)
    den = (
        (slope * rn)
        if abs(slope * rn) > epsilon
        else _sign(slope * rn) * epsilon
    )
    alpha = ((rho * cp * gr * da) / den) + f32(1)
    return min(max(alpha, 0.2), 2.0)


@njit(
    [f32(f32, f32), f64(f64, f64)],
    nogil=True,
    cache=True,
    inline="always",
)
def compute_wet_surface_temperature(ta: float, ea: float) -> float:
    """
    Compute wet surface temperature

    Notes
    -----
    Solve wet surface temperature (°C)
    from air temperature, RH, and pressure (no wind needed).

    Parameters
    ----------
    ta: float
        Air temperature (degC)
    ea: float
        Atmosphere vapor pressure (hPa)

    Returns
    -------
    tw: float
        Wet surface temperature (degC)
    """
    # Initialization
    tw = ta
    epsilon = 1.0e-7
    # Iterative loop
    for _ in range(50):
        esstar_w = _tetens(tw)
        f = esstar_w - PSYCHROMETRIC_CST * (ta - tw) - ea
        df = _tetens_derivative(tw) + PSYCHROMETRIC_CST
        df = df if abs(df) > epsilon else _sign(df) * epsilon
        tw = tw - f / df
    return tw


@njit(
    [f32(f32), f64(f64)],
    nogil=True,
    cache=True,
    inline="always",
)
def compute_saturated_vapor_pressure(t: float) -> float:
    """
    Compute saturated vapor pressure from temperature

    Notes
    -----

    Parameters
    ----------
    t: float
        Temperature (degC)

    Returns
    -------
    estar: float
        Saturated vapor pressure
    """
    return _tetens(t)
