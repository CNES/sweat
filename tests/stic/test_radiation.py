# Copyright: (c) 2025 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import pytest

from evaspa.stic import radiation


@pytest.mark.parametrize(
    ("rn", "lai", "local_time", "m", "expected"),
    [
        pytest.param(200, 2, 36000, 0.6, 2.342408987892905),
        pytest.param(-50, 2, 36000, 0.6, 0.5856022469732263),
    ],
)
def test_f_g_actualsurface(rn, lai, local_time, m, expected) -> None:
    """
    Test function for converting to local time
    """
    res = radiation.f_g_actualsurface(rn, lai, local_time, m)
    np.testing.assert_almost_equal(res, expected, decimal=3)
