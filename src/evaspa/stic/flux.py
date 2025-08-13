# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales
"""
Module for STIC flux computation
"""

import numpy as np
from numba import float32 as f32  # to define f32
from numba import njit

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
def f_g_actualsurface(
    rn: float,
    lai: float,
    local_time: float,
    m: float,
) -> float:
    """
    Compute G flux

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
    rn_soil = rn * np.exp(-KRN * lai)  # Correction on 12/11/2021

    sol_noon = f32(12) * f32(60) * f32(60)
    tg0 = sol_noon - local_time

    # Estimating GHF according to Santanello and Friedl (2003)
    cg = (f32(1) - m) * f32(CG_MAX) + m * f32(CG_MIN)
    tg = (f32(1) - m) * f32(TG_MAX) + m * f32(TG_MIN)

    g_flux = rn_soil * cg * np.cos(f32(2) * np.pi * (tg0 + f32(10800)) / tg)
    if rn_soil < f32(0):
        g_flux = -g_flux

    return g_flux
