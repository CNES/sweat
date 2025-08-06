import numpy as np

# Constants
KRN = 0.6


def f_g_actualsurface(
    rn: float,
    lai: float,
    local_time: float,
    m: float,
    cg_min: float = 0.05,  # for wet surface
    cg_max: float = 0.35,  # for dry surface, in water-controlled ecosystems
    tg_min: float = 74000,  # for wet surface
    tg_max: float = 100000,  # for dry surface
) -> float:
    """
    Description
    -----------
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
    cg_min: float
        Parameter for wet surface (default = 0.05)
    cg_max: float
        Parameter for dry surface,
        in water-controlled ecosystems (default = 0.35)
    tg_min: float
        Parameter for wet surface (default = 74000)
    tg_max:float
        Parameter for dry surface (default = 100000)

    Return
    ------
    g_flux: float
        G flux
    """
    rn_soil = rn * np.exp(-KRN * lai)  # Correction on 12/11/2021

    sol_noon = 12 * 60 * 60
    tg0 = sol_noon - local_time

    # Estimating GHF according to Santanello and Friedl (2003)
    cg = (1 - m) * cg_max + m * cg_min
    tg = (1 - m) * tg_max + m * tg_min

    g_flux = rn_soil * cg * np.cos(2 * np.pi * (tg0 + 10800) / tg)
    if rn_soil < 0:
        g_flux = -g_flux

    return g_flux
