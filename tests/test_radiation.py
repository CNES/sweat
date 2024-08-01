#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np

from evaspa.radiation import CST_SB, compute_g, compute_rn


def test_stefan_boltzmann_constant() -> None:
    """
    Stefan Boltzmann constant
    """
    np.testing.assert_approx_equal(CST_SB, 5.670374419e-8, significant=7)


def test_compute_rn() -> None:
    """
    Test compute net radiation Rn
    """
    # Case 1: Black body
    rg = np.ones((2, 2))
    ra = np.zeros((2, 2))
    albedo = np.ones((2, 2))
    emis = np.ones((2, 2))
    lst = np.ones((2, 2)) * 300
    rn = compute_rn(lst, emis, albedo, rg, ra)
    ref = -CST_SB * (300**4) * np.ones((2, 2))
    np.testing.assert_allclose(rn, ref)


def test_compute_g() -> None:
    """
    Test compute G flux
    """
    rn = np.ones((2, 2))
    ndvi = 0.5 * np.ones((2, 2))
    g = compute_g(rn, ndvi)
    ref = 0.235 * np.ones((2, 2))
    np.testing.assert_allclose(g, ref)
