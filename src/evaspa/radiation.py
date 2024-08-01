#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
from scipy.constants import c, h, k, pi

CST_SB = ((2 * pi**5) * (k**4)) / (15 * (c**2) * (h**3))


def compute_rn(
    lst: np.ndarray,
    emis: np.ndarray,
    albedo: np.ndarray,
    rg: np.ndarray,
    ra: np.ndarray,
) -> np.ndarray:
    """
    Compute net radiation Rn
    """
    return (1 - albedo) * rg - emis * CST_SB * (lst**4) + emis * ra


def compute_g(rn: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    """
    Compute G flux
    """
    return rn * (0.4 - (0.33 * ndvi))
