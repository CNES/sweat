# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 CESBIO / Centre National d'Etudes Spatiales

import pytest

from sweat.stic.registry import (
    DEFAULT_VERSION,
    MODEL_REGISTRY,
)


@pytest.mark.unit
def test_default_version() -> None:
    """
    Test default version
    """
    assert DEFAULT_VERSION == "1.3"


@pytest.mark.unit
def test_model_registry() -> None:
    """
    Test model registry
    """
    assert sorted(MODEL_REGISTRY.keys()) == ["1.3", "1.4"]
