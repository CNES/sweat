# Copyright: (c) 2025 LIST / CESBIO / Centre National d'Etudes Spatiales

import numpy as np

# ruff: noqa: PLR2004

# Psychrometric constant (hpa/K)
PSYCHROMETRIC_CST = 0.67
PT_CST = 1.26


def f_soilmoisture_initialize(
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
    Description
    -----------
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
        Longwavve net radiation (W.m-2)
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

    Return
    ------
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
    e_surf: float
        Vapor pressure at surface temperature (hPa)
    t0d: float
        Dewpoint temperature at source/sink height (celsius)
    d_surf: float
        vapor pressure deficit of the air at the surface (hPa)
    """
    # Surface dewpoint temperature (degC)
    t0d = (esstar - ea - s3 * ts + s1 * td) / (s1 - s3)

    # Surface moisture availability for surface wetness (0-1)
    m_surf = (s1 / s2) * ((t0d - td) / (ts - td))
    m_surf = np.clip(m_surf, 0.0001, 0.9999)

    # Surface vapor pressure and deficit
    e_surf = ea + m_surf * (esstar - ea)
    d_surf = e_surf - ea

    # Separating soil and canopy wetness to form a composite surface moisture
    m_canopy = fc * m_surf
    m_soil = (1 - fc) * m_surf

    # Dewpoint temperature index
    # tdew_index > 1 signifies super dry condition
    tdew_index = (ts - t0d) / (ta - td)

    # Potential evaporation (Priestley-Taylor eqn.)
    ep_pt = (PT_CST * slope * rn) / (slope + PSYCHROMETRIC_CST)

    # Temperature difference
    dts = ts - ta

    # Surface wetness comes from the soil, vegetation contribution is neglegible
    if (fc <= 0.25) & (tdew_index < 1):
        m_surf = m_soil
        m_canopy = 0
    if (fc <= 0.25) & (ta > 10) & (td < 0) & (ln < -125):
        m_surf = m_soil
        m_canopy = 0

    # Root zone moisture (Mrz)
    m_rz = (PSYCHROMETRIC_CST * s1 * (t0d - td)) / (
        slope * s3 * (ts - td)
        + PSYCHROMETRIC_CST * s4 * (ta - td)
        - slope * s1 * (t0d - td)
    )
    m_rz = np.clip(m_rz, 0.0001, 0.9999)

    # Combine M to account for Hysteresis and
    # initial estimation of surface vapor pressure
    m = m_surf
    if (ep_pt > rn) & (dts > 0):
        m = m_rz
    if (ep_pt > rn) & (fc <= 0.25):
        m = m_rz
    if (ep_pt > rn) & (d_surf > da):
        m = m_rz

    if (fc <= 0.25) & (dts > 0) & (ta > 10) & (td < 0) & (ln < -125):
        m = m_rz
    if (fc <= 0.25) & (dts > 0) & (ta > 10) & (td < 0) & (d_surf > da):
        m = m_rz
    if (ep_pt < rn) & (fc <= 0.25) & (d_surf > da):
        m = m_rz

    es = ea + m * (esstar - ea)

    # vapor pressure deficit at surface
    ds = esstar - es

    return (m, m_canopy, m_soil, m_surf, m_rz, es, t0d, ds)


def f_soilmoisture_iterate(
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
    Description
    -----------
    This functions estimates the soil moisture availability (M) (or
    wetness) (value 0 to 1) based on thermal IR and meteorological
    information. However, this M will be treated as initial M, which will be
    later on estimated through iteration in the actual ET estimation loop to
    establish feedback between M and biophysical states

    Parameters
    ----------
    slope: float
        Slope of saturation vapor pressure versus air temperature
    s1: float
        Slope of saturation vapor pressure versus temperature
    s2: float
        Slope of saturation vapor pressure versus temperature
    s3: float
        Slope of saturation vapor pressure versus temperature
    s4: float
        Slope of saturation vapor pressure versus temperature
    ts: float
        Surface temperature
    ta: float
        Air temperature
    delta_t: float
        Difference between temperature at source/sink height and
        air temperature
    td: float
        Dewpoint temperature
    t0d: float
        Dewpoint temperature at source/sink height
    rn: float
        Net radiation
    ln: float
        Longwavve net radiation
    fc: float
        Fraction cover
    da: float
        Atmosphere vapor pressure deficit (hPa)
    d0: float
        Vapor pressure deficit at source/sink height (hPa)
    eastar: float
        Saturation vapor pressure at surface temperature
    ea: float
        Atmosphere vapor pressure (hPa)
    e0star: float
        Saturation vapor pressure at surface temperature
        at source/sink height
    esstar: float
        Saturation vapor pressure at surface temperature

    Return
    ------
    m: float
        Surface moisture avalilability (0 - 1)
    m_surf: float
        Surface moisture availability for surface wetness (0-1)
    m_canopy: float
        Surface moisture availability for canopy component (0-1)
    m_soil: float
        Surface moisture availability for soil component (0-1)
    m_rz: float
        Surface moisture availability for rot zone wetness (0-1)
    """
    # Surface moisture (Msurf)
    k = (e0star - ea) / (esstar - ea)
    m_surf = (s1 / (k * s2)) * ((t0d - td) / (ts - td))  # surface wetness
    m_surf = np.clip(m_surf, 0.0001, 0.9999)

    # Separating soil and canopy wetness to form a composite surface moisture
    m_canopy = fc * m_surf
    m_soil = (1 - fc) * m_surf

    tdew_index = (ts - t0d) / (ta - td)
    ep_pt = (PT_CST * slope * rn) / (
        slope + PSYCHROMETRIC_CST
    )  # Potential evaporation (Priestley-Taylor eqn.)

    # Surface wetness comes from the soil, vegetation contribution is neglegible
    if (fc <= 0.25) & (tdew_index < 1):
        m_surf = m_soil
        m_canopy = 0
    if (fc <= 0.25) & (ta > 10) & (td < 0) & (ln < -125):
        m_surf = m_soil
        m_canopy = 0

    # Root zone moisture (Mrz)
    m_rz = (PSYCHROMETRIC_CST * s1 * (t0d - td)) / (
        slope * s3 * (ts - td)
        + PSYCHROMETRIC_CST * s4 * (ta - td)
        - slope * s1 * (t0d - td)
    )
    m_rz = np.clip(m_rz, 0.0001, 0.9999)

    # Combine M to account for Hysteresis and
    # initial estimation of surface vapor pressure
    m = m_surf
    if (
        (ep_pt > rn)
        & (delta_t > 0)
        & (fc <= 0.25)
        & (d0 > da)
        & (tdew_index < 1)
    ):
        m = m_rz
    if (
        (fc <= 0.25)
        & (delta_t > 0)
        & (d0 > da)
        & (ta > 10)
        & (td < 0)
        & (ln < -125)
    ):
        m = m_rz

    return (m, m_surf, m_canopy, m_soil, m_rz)
