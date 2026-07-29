# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
import pytest

import sweat.extra.utils_notebook as utn


@pytest.mark.unit
def test_locate_data() -> None:
    """
    Test function locate_date
    """
    assert utn.locate_test_data()
