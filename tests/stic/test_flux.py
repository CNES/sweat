# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pytest

from evaspa.stic import flux


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rn", "lai", "local_time", "m", "expected"),
    [
        pytest.param(200, 2, 36000, 0.6, 2.342409),
        pytest.param(-50, 2, 36000, 0.6, 0.585602),
    ],
)
def test_f_g_actualsurface(rn, lai, local_time, m, expected) -> None:
    """
    Test function for converting to local time
    """
    res = flux.f_g_actualsurface(rn, lai, local_time, m)
    np.testing.assert_almost_equal(res, expected, decimal=3)
