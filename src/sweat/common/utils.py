# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing utility functions
"""

import numpy as np
import numpy.typing as npt


def compute_ndvi(
    nir: npt.ArrayLike, red: npt.ArrayLike, delta: float = 0.05
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
    As this one is usually close to 0.01,
    the constant could be 0.05, see (https://www.cesbio.cnrs.fr/multitemp/
    using-ndvi-with-atmospherically-corrected-data/)
    Typical range: -1 to 1, with higher values indicating
    denser and healthier vegetation.

    - Open water, snow: -0.3 to -0.1
    - Bare soil, urban:	-0.1 to 0.2
    - Sparse or stressed crops: 0.2 to 0.4
    - Healthy crops, grassland: 0.4 to 0.7
    - Dense forest, peak season : 0.7 to 0.9

    Parameters
    ----------
    nir: np.array_like
        NIR reflectance
    red: np.array_like
        RED reflectance
    delta: float
        Parameter to stabilize computation

    Returns
    -------
    ndvi: np.array
        NDVI
    """
    ndvi = (np.array(nir) - (np.array(red) + delta)) / (
        np.array(nir) + np.array(red) + delta
    )
    return np.clip(ndvi, -1.0, 1.0)


def compute_vari_green(
    red: npt.ArrayLike,
    green: npt.ArrayLike,
    blue: npt.ArrayLike,
) -> npt.NDArray:
    """
    Compute  the Visible Atmospherically Resistant Index
    Green (VARI_green)

    Notes
    -----
    Visible Atmospherically Resistant Index (VARI) is computed
    with the following formula:
    $$
    VARI_green = \\frac{GREEN - RED}{GREEN + RED - BLUE}
    $$
    Where RED, GREEN, BLUE are the reflectances.
    VARI_green uses the blue, green, and red bands
    to minimize atmospheric effects.
    Typical range: -1 to 1, with higher values indicating
    denser and healthier vegetation.

    - Open water, snow: -0.3 to -0.1
    - Bare soil, urban:	-0.1 to 0.2
    - Sparse or stressed crops: 0.2 to 0.4
    - Healthy crops, grassland: 0.4 to 0.7
    - Dense forest, peak season : 0.7 to 0.9

    See: Anatoly A. Gitelson, Yoram J. Kaufman, Robert Stark, Don Rundquist,
    Novel algorithms for remote estimation of vegetation fraction,
    Remote Sensing of Environment, Volume 80, Issue 1, 2002.

    Parameters
    ----------
    red: np.array_like
        RED reflectance
    green: np.array_like
        GREEN reflectance
    blue: np.array_like
        BLUE reflectance

    Returns
    -------
    vari: np.array
        VARI GREEN index
    """
    vari_green = (np.array(green) - np.array(red)) / (
        np.array(green) + np.array(red) - np.array(blue)
    )
    return np.clip(vari_green, -1, 1)


def compute_arvi(
    nir: npt.ArrayLike,
    red: npt.ArrayLike,
    blue: npt.ArrayLike,
    gamma: float = 1.0,
) -> npt.NDArray:
    """
    Compute  the Atmospherically Resistant Vegetation Index
    (ARVI)

    Notes
    -----
    Atmospherically Resistant Vegetation Index (ARVI) is computed
    with the following formula:
    $$
    ARVI = \\frac{NIR - RED - GAMMA(RED-BLUE)}{NIR + RED - GAMMA(RED - BLUE)}
    $$
    Where NIR, RED, BLUE are the reflectances.
    ARVI minimizes atmospheric effects by incorporating blue band correction.
    Typical range: -1 to 1, with higher values indicating
    denser and healthier vegetation.

    - Open water, snow: -0.3 to -0.1
    - Bare soil, urban:	-0.1 to 0.2
    - Sparse or stressed crops: 0.2 to 0.4
    - Healthy crops, grassland: 0.4 to 0.7
    - Dense forest, peak season : 0.7 to 0.9

    Parameters
    ----------
    nir: np.array_like
        GREEN reflectance
    red: np.array_like
        RED reflectance
    blue: np.array_like
        BLUE reflectance
    gamma: float
        Parameter

    Returns
    -------
    arvi: np.array
        ARVI
    """
    arvi = (
        np.array(nir) - np.array(red) - gamma * (np.array(red) - np.array(blue))
    ) / (
        np.array(nir) + np.array(red) - gamma * (np.array(red) - np.array(blue))
    )
    return np.clip(arvi, -1, 1)


def compute_gndvi(
    nir: npt.ArrayLike, green: npt.ArrayLike, delta: float = 0.05
) -> npt.NDArray:
    """
    Compute Green Normalized Difference Vegetation Index GNDVI

    Notes
    -----
    Green Normalized Difference Vegetation Index NDVI is computed
    with the following formula:
    $$
    NDVI = \\frac{NIR - (GREEN + delta)}{NIR + GREEN + delta}
    $$
    Where NIR and GREEN are the reflectances in the NIR and GREEN
    and delta is a constant to add to the red band surface
    reflectance in order to solve the issue of green surface
    reflectances close to zero. This constant must be greater
    than the standard deviation of atmospheric correction noise.
    As this one is usually close to 0.01,
    the constant could be 0.05, see (https://www.cesbio.cnrs.fr/multitemp/
    using-ndvi-with-atmospherically-corrected-data/).
    More sensitive to chlorophyll content and can be useful for detecting
    stress in dense vegetation canopies.
    Typical range: -1 to 1, with higher values indicating
    denser and healthier vegetation.

    - Open water, snow: -0.3 to -0.1
    - Bare soil, urban:	-0.1 to 0.2
    - Sparse or stressed crops: 0.2 to 0.4
    - Healthy crops, grassland: 0.4 to 0.7
    - Dense forest, peak season : 0.7 to 0.9

    Parameters
    ----------
    nir: np.array_like
        NIR reflectance
    green: np.array_like
        RED reflectance
    delta: float
        Parameter to stabilize computation

    Returns
    -------
    gndvi: np.array
        GNDVI
    """
    gndvi = (np.array(nir) - (np.array(green) + delta)) / (
        np.array(nir) + np.array(green) + delta
    )
    return np.clip(gndvi, -1.0, 1.0)


def compute_gli(
    red: npt.ArrayLike,
    green: npt.ArrayLike,
    blue: npt.ArrayLike,
    delta: float = 0.05,
) -> npt.NDArray:
    """
    Compute Green Leaf Index (GLI)

    Notes
    -----
    GLI is computed with the following formula:
    $$
    GLI = \\frac{2 GREEN - (RED + BLUE delta)}
                 {2 GREEN + RED + BLUE + delta}
    $$
    Where RED, GREEN and BLUE are the reflectances
    and delta is a constant to add to the red band surface
    reflectance in order to solve the issue of low
    reflectances close to zero. This constant must be greater
    than the standard deviation of atmospheric correction noise.
    As this one is usually close to 0.01, the constant could be 0.05.
    Typical range: -1 to 1, with higher values indicating
    denser and healthier vegetation.

    See: Louhaichi, Mounir & Borman, Michael & Johnson, Douglas. (2001).
    Spatially Located Platform and Aerial Photography for Documentation
    of Grazing Impacts on Wheat. Geocarto International. 16.

    Parameters
    ----------
    red: np.array_like
        RED reflectance
    green: np.array_like
        GREEN reflectance
    blue: np.array_like
        BLUE reflectance
    delta: float
        Parameter to stabilize computation

    Returns
    -------
    ndvi: np.array
        NDVI
    """
    ndvi = (2 * np.array(green) - (np.array(red) + np.array(blue) + delta)) / (
        2 * np.array(green) + np.array(red) + np.array(blue) + delta
    )
    return np.clip(ndvi, -1.0, 1.0)


def compute_msavi(nir: npt.ArrayLike, red: npt.ArrayLike) -> npt.NDArray:
    """
    Compute Modified Soil-Adjusted Vegetation Index (MSAVI)

    Notes
    -----
    MSAVI is a variant of SAVI that optimizes the soil brightness
    correction factor. MSAVI is computed with the following formula:
    $$
    MSAVI = 0.5\\times\\left{2 NIR +
     1 - \\sqrt{(2 NIR+1)^2 -8(NIR - RED)}\\right}
    $$
    Where NIR and RED are the reflectances in the NIR and RED.
    Typical range: -1 to 1, with higher values indicating
    denser and healthier vegetation.

    Parameters
    ----------
    nir: np.array_like
        NIR reflectance
    red: np.array_like
        RED reflectance

    Returns
    -------
    msavi: np.array
        MSAVI
    """
    msavi = 0.5 * (
        2 * np.array(nir)
        + 1
        - np.sqrt(
            (2 * np.array(nir) + 1) * (2 * np.array(nir) + 1)
            - 8 * (np.array(nir) - np.array(red))
        )
    )
    return np.clip(msavi, -1.0, 1.0)


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
