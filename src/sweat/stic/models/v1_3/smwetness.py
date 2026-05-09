# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions to compute soil moisture
for STIC model v1.3
"""

from numba import float32 as f32
from numba import float64 as f64
from numba import njit
from numba.types import Tuple

from sweat.stic.constant import PSYCHROMETRIC_CST, PT_CST


@njit(
    [Tuple((f32,) * 8)(*(f32,) * 14), Tuple((f64,) * 8)(*(f64,) * 14)],
    nogil=True,
    cache=True,
)
def initialize_soil_moisture(
    slope: float,
    ts: float,
    ta: float,
    td: float,
    rn: float,
    ln: float,
    fc: float,
    da: float,
    ea: float,
    esstar: float,
    s1: float,
    s2: float,
    s3: float,
    s4: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Initiate soil moisture

    Notes
    -----
    This function estimates the soil moisture availability
    (M or wetness, 0-1) based on thermal and meteorological
    information.
    However, this M will be treated as initial M,
    which will be later on estimated through iteration in
    the actual ET estimation loop to establish feedback
    between M and biophysical states.

    The moisture availability $M$ can be initialized using two
    equations: one which is an indicator
    of surface (canopy-top soil) wetness, $M_{surf}$ and the
    other indicates the root-zone soil wetness, $M_{rz}$.
    The choice of the equation depends on a set of conditions.

    $M_{surf}$ can be initialized as follows:
    $$
    M_{surf} = \\frac{s_{1}(T_{0D} - T_{D})}{s_{2}(LST - T_{D})}
    $$
    $M_{surf}$ must remain between 0 and 1.

    For root-zone wetness, $M_{rz}$ can be initialized as follows
    $$
    M_{rz} = \\frac{\\gamma s_{1} (T_{0D} - T_{D}) }
    {s_{3} ( LST - T_{0D} ) \\Delta +
    \\gamma s_{4} ( T_{a} - T_{D} ) +
    \\Delta s_{1} ( T_{0D} - T_{D} ) }
    $$
    $M_{rz}$ must remain between 0 and 1.

    The conditions to choose between the both equations for $M$ depends of the
    following variables:

    - The surface vapor pressure
    - The vapor pressure deficit at the surface is as follows.
    - The dew-point temperature index
    - Latent heat flux computed with Priestley-Taylor equation
    - The temperatures $T_{a}$, $T_{D}$ and the
      difference between $T_{a}$ and $LST$
    - The net radiation $R_{n}$
    - The net longwave radiation
    - The vegetation variable Fcover
    - The vapor pressure deficit (VPD) of the air at the
      reference height $D_{a}$

    Parameters
    ----------
    slope: float
        Slope of saturation vapor pressure versus air temperature (hPa/degC)
    ts: float
        Surface temperature (celsius)
    ta: float
        Air temperature (celsius)
    td: float
        Dewpoint temperature (celsius)
    rn: float
        Net radiation (W.m-2)
    ln: float
        Longwave net radiation (W.m-2)
    fc: float
        Fraction cover (0-1)
    da: float
        Atmosphere vapor pressure deficit (hPa)
    eastar: float
        Saturation vapor pressure at air temperature (hPa)
    ea: float
        Atmosphere vapor pressure (hPa)
    esstar: float
        Saturation vapor pressure at surface temperature (hPa)
    s1: float
        Slope of saturation vapor pressure versus temperature (hPa/degC)
    s2: float
        Slope of saturation vapor pressure versus temperature (hPa/degC)
    s3: float
        Slope of saturation vapor pressure versus temperature (hPa/degC)
    s4: float
        Slope of saturation vapor pressure versus temperature (hPa/degC)

    Returns
    -------
    m: float
        Surface moisture availability (0 - 1)
    m_canopy: float
        Surface moisture availability for canopy component (0-1)
    m_soil: float
        Surface moisture availability for soil component (0-1)
    m_surf: float
        Surface moisture availability for surface wetness (0-1)
    m_rz: float
        Surface moisture availability for rot zone wetness (0-1)
    es: float
        Vapor pressure at surface temperature (hPa)
    tsd: float
        Dewpoint temperature at the reference height (celsius)
    ds: float
        Vapor pressure deficit of the air at the surface (hPa)
    """
    epsilon = 1.0e-7
    # Compute the surface dewpoint temperature (degC)
    # Handle division by zero
    tsd = (
        (esstar - ea - s3 * ts + s1 * td) / (s1 - s3)
        if abs(s1 - s3) > epsilon
        else ts
    )

    # Compute the surface moisture availability
    # for surface wetness (0-1)
    # Handle division by zero
    m_surf = (
        (s1 / s2) * ((tsd - td) / (ts - td)) if abs(ts - td) > epsilon else 1.0
    )
    m_surf = min(max(m_surf, 0.0001), 0.9999)

    # Compute surface vapor pressure and deficit
    es = ea + m_surf * (esstar - ea)
    ds = es - ea

    # Separate the soil and canopy wetness to form a
    # composite surface moisture
    m_canopy = fc * m_surf
    m_soil = (1.0 - fc) * m_surf

    # Compute dewpoint temperature index
    # tdew_index > 1 signifies super dry condition
    # Handle division by zero
    tdew_index = (ts - tsd) / (ta - td) if abs(ta - td) > epsilon else 0.0

    # Compute the potential evaporation (Priestley-Taylor eqn.)
    ep_pt = (PT_CST * slope * rn) / (slope + PSYCHROMETRIC_CST)

    # Temperature difference
    dts = ts - ta

    # Surface wetness comes from the soil, vegetation contribution is negligible
    if (fc <= 0.25) & (tdew_index < 1.0):
        m_surf = m_soil
        m_canopy = 0.0
    if (fc <= 0.25) & (ta > 10.0) & (td < 0.0) & (ln < -125.0):
        m_surf = m_soil
        m_canopy = 0.0

    # Compute root zone moisture (Mrz)
    m_rz = (PSYCHROMETRIC_CST * s1 * (tsd - td)) / (
        slope * s3 * (ts - td)
        + PSYCHROMETRIC_CST * s4 * (ta - td)
        - slope * s1 * (tsd - td)
    )
    m_rz = min(max(m_rz, 0.0001), 0.9999)

    # Combine soil moisture to account for hysteresis
    # and initial estimation of surface vapor pressure
    m = m_surf
    if (ep_pt > rn) & (dts > 0.0):
        m = m_rz
    if (ep_pt > rn) & (fc <= 0.25):
        m = m_rz
    if (ep_pt > rn) & (ds > da):
        m = m_rz

    if (fc <= 0.25) & (dts > 0.0) & (ta > 10.0) & (td < 0.0) & (ln < -125.0):
        m = m_rz
    if (fc <= 0.25) & (dts > 0.0) & (ta > 10.0) & (td < 0.0) & (ds > da):
        m = m_rz
    if (ep_pt < rn) & (fc <= 0.25) & (ds > da):
        m = m_rz

    # Update vapor pressure at surface
    es = ea + m * (esstar - ea)

    # Update vapor pressure deficit at surface
    ds = esstar - es

    return (m, m_canopy, m_soil, m_surf, m_rz, es, tsd, ds)


@njit(
    [Tuple((f32,) * 5)(*(f32,) * 18), Tuple((f64,) * 5)(*(f64,) * 18)],
    nogil=True,
    cache=True,
)
def iterate_soil_moisture(
    slope: float,
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    ts: float,
    ta: float,
    delta_t: float,
    td: float,
    t0d: float,
    rn: float,
    ln: float,
    fc: float,
    da: float,
    d0: float,
    ea: float,
    e0star: float,
    esstar: float,
) -> tuple[float, float, float, float, float]:
    """
    Compute soil moisture during iteration loop

    Notes
    -----
    This functions estimates the soil moisture availability $M$ (or
    wetness) (value 0 to 1) based on thermal IR and meteorological
    information. However, this $M$ will be treated as initial $M$,
    which will be later on estimated through iteration in the
    actual ET estimation loop to establish feedback between M and
    biophysical states.

    $M$ can be calculated using two equations: one which is an indicator
    of surface (canopy-top soil) wetness, $M_{surf}$ and the
    other indicates the root-zone soil wetness, $M_{rz}$.
    The choice of the equation depends on a set of conditions.

    $M_{surf}$ can be computed as follows:
    $$
    M_{surf} = \\frac{e_{s}^{\\star} - e_{a}}
    {e_{0}^{\\star} - e_{a}} \\times \\frac{s_{1}(T_{0D} - T_{D})}
    {s_{2}(LST - T_{D})}
    $$
    $M_{surf}$ must remain between 0 and 1.

    For root-zone wetness, $M_{rz}$ can be computed as follows
    $$
    M_{rz} = \\frac{\\gamma s_{1} (T_{0D} - T_{D}) }
    {s_{3} ( LST - T_{0D} ) \\Delta +
    \\gamma s_{4} ( T_{a} - T_{D} ) +
    \\Delta s_{1} (T_{0D} - T_{D} ) }
    $$
    $M_{rz}$ must remain between 0 and 1.

    The conditions to choose between the both equations for $M$ depends of the
    following variables:

    - The dew-point temperature index
    - Latent heat flux computed with Priestley-Taylor equation
    - The temperatures $T_{a}$, $T_{D}$ and the
      difference between $T_{a}$ and $LST$
    - The net radiation $R_{n}$
    - The net longwave radiation
    - The vegetation variable Fcover
    - The vapor pressure deficit (VPD) of the air at the
      reference height $D_{a}$

    Parameters
    ----------
    slope: float
        Slope of saturation vapor pressure versus air temperature (hPa/degC)
    s1: float
        Slope of saturation vapor pressure versus
        temperature at surface temperature(hPa/degC)
    s2: float
        Slope of saturation vapor pressure versus
        temperature (hPa/degC)
    s3: float
        Slope of saturation vapor pressure versus
        temperature at dewpoint temperature (hPa/degC)
    s4: float
        Slope of saturation vapor pressure versus
        temperature (hPa/degC)
    ts: float
        Surface temperature (degC)
    ta: float
        Air temperature (degC)
    delta_t: float
        Difference between temperature at source/sink height and
        air temperature (degC)
    td: float
        Dewpoint temperature (degC)
    t0d: float
        Dewpoint temperature at reference height (degC)
    rn: float
        Net radiation (W.m-2)
    ln: float
        Longwave net radiation (W.m-2)
    fc: float
        Fraction cover (0-1)
    da: float
        Atmosphere vapor pressure deficit (hPa)
    d0: float
        Vapor pressure deficit at source/sink height (hPa)
    eastar: float
        Saturation vapor pressure at air temperature (hPa)
    ea: float
        Atmosphere vapor pressure (hPa)
    e0star: float
        Saturation vapor pressure at surface temperature
        at reference height (hPa)
    esstar: float
        Saturation vapor pressure at surface temperature (hPa)

    Returns
    -------
    m: float
        Surface moisture availability (0 - 1)
    m_surf: float
        Surface moisture availability for surface wetness (0-1)
    m_canopy: float
        Surface moisture availability for canopy component (0-1)
    m_soil: float
        Surface moisture availability for soil component (0-1)
    m_rz: float
        Surface moisture availability for root zone wetness (0-1)
    """
    # Surface available moisture
    m_surf = 1.0
    epsilon = 1.0e-7
    if abs(ts - td) > epsilon:
        k = (e0star - ea) / (esstar - ea)
        m_surf = (s1 / (k * s2)) * ((t0d - td) / (ts - td))  # surface wetness
    m_surf = min(max(m_surf, 0.0001), 0.9999)

    # Separating soil and canopy wetness to form a composite surface moisture
    m_canopy = fc * m_surf
    m_soil = (1.0 - fc) * m_surf

    # Handle division by zero
    tdew_index = (ts - t0d) / (ta - td) if abs(ta - td) > epsilon else 0.0
    ep_pt = (PT_CST * slope * rn) / (
        slope + PSYCHROMETRIC_CST
    )  # Potential evaporation (Priestley-Taylor eqn.)

    # Surface wetness comes from the soil, vegetation contribution is negligible
    if (fc <= 0.25) & (tdew_index < 1.0):
        m_surf = m_soil
        m_canopy = 0.0
    if (fc <= 0.25) & (ta > 10.0) & (td < 0.0) & (ln < -125.0):
        m_surf = m_soil
        m_canopy = 0.0

    # Root zone moisture (Mrz)
    m_rz = (PSYCHROMETRIC_CST * s1 * (t0d - td)) / (
        slope * s3 * (ts - td)
        + PSYCHROMETRIC_CST * s4 * (ta - td)
        - slope * s1 * (t0d - td)
    )
    m_rz = min(max(m_rz, 0.0001), 0.9999)

    # Combine M to account for Hysteresis and
    # initial estimation of surface vapor pressure
    m = m_surf
    if (
        (ep_pt > rn)
        & (delta_t > 0.0)
        & (fc <= 0.25)
        & (d0 > da)
        & (tdew_index < 1.0)
    ):
        m = m_rz
    if (
        (fc <= 0.25)
        & (delta_t > 0.0)
        & (d0 > da)
        & (ta > 10.0)
        & (td < 0.0)
        & (ln < -125.0)
    ):
        m = m_rz

    return (m, m_surf, m_canopy, m_soil, m_rz)
