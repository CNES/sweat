# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales

import pytest

from sweat.stic.models.registry import (
    MODEL_PIXEL_REGISTRY,
    MODEL_REGISTRY,
)


@pytest.mark.unit
def test_model_registry() -> None:
    """
    Test model registry
    """
    assert sorted(MODEL_REGISTRY.keys()) == ["1.3"]  # , "1.4"]


@pytest.mark.unit
def test_model_pixel_registry() -> None:
    """
    Test model pixel registry
    """
    assert sorted(MODEL_PIXEL_REGISTRY.keys()) == ["1.3"]  # , "1.4"]
