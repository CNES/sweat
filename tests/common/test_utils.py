# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pytest

from sweat.common import utils


@pytest.mark.unit
@pytest.mark.parametrize(
    ("nir", "red", "expected"),
    [
        (1.0, 0.5, 0.33),
        (1.0, 0.05, 0.90),
        (0.0, 1, -1.0),
        (0.0, 0.0, -1.0),
        (0.5, 0.5, 0.0),
    ],
)
def test_compute_ndvi(nir, red, expected):
    """Test NDVI function"""
    ndvi = utils.compute_ndvi(nir, red)
    np.testing.assert_allclose(ndvi, expected, atol=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("red", "green", "blue", "expected"),
    [
        (1.0, 1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, -1.0),
        (0.0, 1.0, 0.01, 1.01),
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 0.0),
    ],
)
def test_compute_vari_green_index(red, green, blue, expected):
    """Test NDVI function"""
    ndvi = utils.compute_vari_green_index(red, green, blue)
    np.testing.assert_allclose(ndvi, expected, atol=0.01)
