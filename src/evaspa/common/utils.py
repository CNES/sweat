# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import numpy.typing as npt


def compute_ndvi(
    nir: npt.ArrayLike, red: npt.ArrayLike, delta: float = 0.05
) -> npt.NDArray:
    """
    Description
    -----------
    Normalized Difference Vegetation Index NDVI is computed
    with the following formula:
        NDVI = (NIR - (RED + delta)) / (NIR + RED + delta)
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

    Return
    ------
    ndvi: np.array
        NDVI
    """
    return (np.array(nir) - (np.array(red) + delta)) / (
        np.array(nir) + np.array(red) + delta
    )


def compute_broadband_emissivity():
    """
    Description
    -----------
    Broadband emissivity is computed from spectral emissivity
    that are obtained from the TES algorithm used to compute
    surface temperature (see RD7).
    Broadband emissivity is computed using linear combination
    models that has to be implemented.
    """
