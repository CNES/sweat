# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pytest

from evaspa import utils


@pytest.mark.parametrize(
    ("nir", "red", "expected"),
    [
        pytest.param(1000, 500, 0.33),
        pytest.param(10000, 500, 0.90),
        pytest.param(0.0, 500, -1.0),
        pytest.param(0.0, 0.0, -1.0),
        pytest.param(500.0, 500.0, 0.0),
    ],
)
def test_compute_ndvi(nir, red, expected):
    """Test NDVI function"""
    ndvi = utils.compute_ndvi(nir, red)
    np.testing.assert_allclose(ndvi, expected, atol=0.01)
