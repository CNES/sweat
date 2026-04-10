# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing utility functions
"""

import numpy as np
import numpy.typing as npt


def compute_ndvi(
    nir: npt.ArrayLike, red: npt.ArrayLike, delta: float = 0.005
) -> npt.NDArray:
    """
    Compute Normalized Difference Vegetation Index NDVI

    Notes
    -----
    Normalized Difference Vegetation Index NDVI is computed
    with the following formula:
    $$
    NDVI = \\frac{NIR - (RED + delta)}{NIR + RED + delta}
    $$
    Where NIR and RED are the reflectances in the NIR and RED
    and delta is a constant to add to the red band surface
    reflectance in order to solve the issue of red surface
    reflectances close to zero. This constant must be greater
    than the standard deviation of atmospheric correction noise.
    As this one is usually close to 0.01, the constant could be 0.05.

    Parameters
    ----------
    nir: np.array_like
        NIR reflectance
    red: np.array_like
        RED reflectance

    Returns
    -------
    ndvi: np.array
        NDVI
    """
    ndvi = (np.array(nir) - (np.array(red) + delta)) / (
        np.array(nir) + np.array(red) + delta
    )
    return np.clip(ndvi, -1.0, 1.0)


def compute_vari_green_index(
    red: npt.ArrayLike,
    green: npt.ArrayLike,
    blue: npt.ArrayLike,
) -> npt.NDArray:
    """
    Compute  the Visible Atmospherically Resistant Index (VARI)

    Notes
    -----
    Visible Atmospherically Resistant Index (VARI) is computed
    with the following formula:
    $$
    VARI = \\frac{GREEN - RED}{GREEN + RED - BLUE}
    $$
    Where RED, GREEN, BLUE are the reflectances.

    Parameters
    ----------
    nir: np.array_like
        NIR reflectance
    red: np.array_like
        RED reflectance

    Returns
    -------
    ndvi: np.array
        NDVI
    """
    vari = (np.array(green) - np.array(red)) / (
        np.array(green) + np.array(red) - np.array(blue)
    )
    return np.nan_to_num(vari, nan=0.0, posinf=0.0, neginf=0.0)


def compute_broadband_emissivity():
    """
    Compute broadband emissivity.

    Notes
    -----
    Broadband emissivity is computed from spectral emissivity
    that are obtained from the TES algorithm used to compute
    surface temperature (see RD7).
    Broadband emissivity $\\epsilon_{BB}$ is computed using linear combination
    models that has to be implemented.
    $$
    \\epsilon_{BB} = \\beta_{0} +\\sum_{j=1}^{Nbands}\\beta_{j}\\epsilon_{j}
    $$
    """
