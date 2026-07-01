# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pytest

from sweat.common import utils


@pytest.mark.unit
@pytest.mark.parametrize(
    ("nir", "red", "expected"),
    [
        (1.0, 0.5, 0.29),
        (1.0, 0.05, 0.82),
        (0.0, 1, -1.0),
        (0.0, 0.0, -1.0),
        (0.5, 0.5, -0.05),
    ],
)
def test_compute_ndvi(nir, red, expected):
    """
    Test NDVI function
    """
    res = utils.compute_ndvi(nir, red)
    np.testing.assert_allclose(res, expected, atol=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("red", "green", "blue", "expected"),
    [
        (1.0, 1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, -1.0),
        (0.0, 1.0, 0.01, 1.0),
        (0.0, 0.0, 0.0, np.nan),
        (1.0, 0.0, 1.0, -1.0),
    ],
)
def test_compute_vari_green(red, green, blue, expected):
    """
    Test VARIgreen function
    """
    res = utils.compute_vari_green(red, green, blue)
    np.testing.assert_allclose(res, expected, atol=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("nir", "red", "blue", "expected"),
    [
        (1.0, 1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.01, -1),
        (0.0, 0.0, 0.0, np.nan),
        (1.0, 0.0, 1.0, 1.0),
    ],
)
def test_compute_arvi(nir, red, blue, expected):
    """
    Test ARVI function
    """
    res = utils.compute_arvi(nir, red, blue)
    np.testing.assert_allclose(res, expected, atol=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("nir", "green", "expected"),
    [
        (1.0, 0.5, 0.29),
        (1.0, 0.05, 0.82),
        (0.0, 1, -1.0),
        (0.0, 0.0, -1.0),
        (0.5, 0.5, -0.05),
    ],
)
def test_compute_gndvi(nir, green, expected):
    """
    Test GNDVI function
    """
    res = utils.compute_gndvi(nir, green)
    np.testing.assert_allclose(res, expected, atol=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("red", "green", "blue", "expected"),
    [
        (1.0, 1.0, 1.0, -0.01),
        (0.0, 0.0, 1.0, -1.0),
        (1.0, 0.0, 0.0, -1.0),
        (0.0, 1.0, 0.01, 0.94),
        (0.0, 0.0, 0.0, -1.0),
        (1.0, 0.0, 1.0, -1.0),
    ],
)
def test_compute_gli(red, green, blue, expected):
    """
    Test GLI function
    """
    res = utils.compute_gli(red, green, blue)
    np.testing.assert_allclose(res, expected, atol=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("nir", "red", "expected"),
    [
        (1.0, 0.5, 0.38),
        (1.0, 0.05, 0.90),
        (0.0, 1, -1.0),
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
    ],
)
def test_compute_msavi(nir, red, expected):
    """
    Test MSAVI function
    """
    res = utils.compute_msavi(nir, red)
    np.testing.assert_allclose(res, expected, atol=0.01)
