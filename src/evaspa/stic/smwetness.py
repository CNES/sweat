# Copyright: (c) 2025 LIST / CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions to compute soil moisture
for STIC model
"""

from numba import float32 as f32  # to define f32
from numba import njit
from numba.types import Tuple

from evaspa.stic.constant import PSYCHROMETRIC_CST, PT_CST


@njit(
    Tuple((f32,) * 8)(*(f32,) * 14),
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
        Surface moisture avalilability (0 - 1)
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
    epsilon = f32(1.0e-7)
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
        (s1 / s2) * ((tsd - td) / (ts - td))
        if abs(ts - td) > epsilon
        else f32(1)
    )
    m_surf = min(max(m_surf, f32(0.0001)), f32(0.9999))

    # Compute surface vapor pressure and deficit
    es = ea + m_surf * (esstar - ea)
    ds = es - ea

    # Separate the soil and canopy wetness to form a
    # composite surface moisture
    m_canopy = fc * m_surf
    m_soil = (f32(1) - fc) * m_surf

    # Compute dewpoint temperature index
    # tdew_index > 1 signifies super dry condition
    # Handle division by zero
    tdew_index = (ts - tsd) / (ta - td) if abs(ta - td) > epsilon else f32(0)

    # Compute the ootential evaporation (Priestley-Taylor eqn.)
    ep_pt = (f32(PT_CST) * slope * rn) / (slope + f32(PSYCHROMETRIC_CST))

    # Temperature difference
    dts = ts - ta

    # Surface wetness comes from the soil, vegetation contribution is neglegible
    if (fc <= f32(0.25)) & (tdew_index < f32(1)):
        m_surf = m_soil
        m_canopy = f32(0)
    if (fc <= f32(0.25)) & (ta > f32(10)) & (td < f32(0)) & (ln < f32(-125)):
        m_surf = m_soil
        m_canopy = f32(0)

    # Compute root zone moisture (Mrz)
    m_rz = (f32(PSYCHROMETRIC_CST) * s1 * (tsd - td)) / (
        slope * s3 * (ts - td)
        + f32(PSYCHROMETRIC_CST) * s4 * (ta - td)
        - slope * s1 * (tsd - td)
    )
    m_rz = min(max(m_rz, f32(0.0001)), f32(0.9999))

    # Combine soil moisture to account for hysteresis
    # and initial estimation of surface vapor pressure
    m = m_surf
    if (ep_pt > rn) & (dts > f32(0)):
        m = m_rz
    if (ep_pt > rn) & (fc <= f32(0.25)):
        m = m_rz
    if (ep_pt > rn) & (ds > da):
        m = m_rz

    if (
        (fc <= f32(0.25))
        & (dts > f32(0))
        & (ta > f32(10))
        & (td < f32(0))
        & (ln < f32(-125))
    ):
        m = m_rz
    if (
        (fc <= f32(0.25))
        & (dts > f32(0))
        & (ta > f32(10))
        & (td < f32(0))
        & (ds > da)
    ):
        m = m_rz
    if (ep_pt < rn) & (fc <= f32(0.25)) & (ds > da):
        m = m_rz

    # Update vapor pressure at surface
    es = ea + m * (esstar - ea)

    # Update vapor pressure deficit at surface
    ds = esstar - es

    return (m, m_canopy, m_soil, m_surf, m_rz, es, tsd, ds)


@njit(
    Tuple((f32,) * 5)(*(f32,) * 18),
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
    This functions estimates the soil moisture availability (M) (or
    wetness) (value 0 to 1) based on thermal IR and meteorological
    information. However, this M will be treated as initial M, which will be
    later on estimated through iteration in the actual ET estimation loop to
    establish feedback between M and biophysical states.

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
        Longwavve net radiation (W.m-2)
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
        Surface moisture avalilability (0 - 1)
    m_surf: float
        Surface moisture availability for surface wetness (0-1)
    m_canopy: float
        Surface moisture availability for canopy component (0-1)
    m_soil: float
        Surface moisture availability for soil component (0-1)
    m_rz: float
        Surface moisture availability for root zone wetness (0-1)
    """
    # Surface moisture (Msurf)
    m_surf = f32(1)
    if abs(ts - td) > f32(1.0e-7):
        k = (e0star - ea) / (esstar - ea)
        m_surf = (s1 / (k * s2)) * ((t0d - td) / (ts - td))  # surface wetness
    m_surf = min(max(m_surf, f32(0.0001)), f32(0.9999))

    # Separating soil and canopy wetness to form a composite surface moisture
    m_canopy = fc * m_surf
    m_soil = (f32(1) - fc) * m_surf

    # Handle division by zero
    tdew_index = (
        (ts - t0d) / (ta - td) if abs(ta - td) > f32(1.0e-7) else f32(0)
    )
    ep_pt = (f32(PT_CST) * slope * rn) / (
        slope + f32(PSYCHROMETRIC_CST)
    )  # Potential evaporation (Priestley-Taylor eqn.)

    # Surface wetness comes from the soil, vegetation contribution is neglegible
    if (fc <= f32(0.25)) & (tdew_index < f32(1)):
        m_surf = m_soil
        m_canopy = f32(0)
    if (fc <= f32(0.25)) & (ta > f32(10)) & (td < f32(0)) & (ln < f32(-125)):
        m_surf = m_soil
        m_canopy = f32(0)

    # Root zone moisture (Mrz)
    m_rz = (f32(PSYCHROMETRIC_CST) * s1 * (t0d - td)) / (
        slope * s3 * (ts - td)
        + f32(PSYCHROMETRIC_CST) * s4 * (ta - td)
        - slope * s1 * (t0d - td)
    )
    m_rz = min(max(m_rz, f32(0.0001)), f32(0.9999))

    # Combine M to account for Hysteresis and
    # initial estimation of surface vapor pressure
    m = m_surf
    if (
        (ep_pt > rn)
        & (delta_t > f32(0))
        & (fc <= f32(0.25))
        & (d0 > da)
        & (tdew_index < f32(1))
    ):
        m = m_rz
    if (
        (fc <= f32(0.25))
        & (delta_t > f32(0))
        & (d0 > da)
        & (ta > f32(10))
        & (td < f32(0))
        & (ln < f32(-125))
    ):
        m = m_rz

    return (m, m_surf, m_canopy, m_soil, m_rz)
