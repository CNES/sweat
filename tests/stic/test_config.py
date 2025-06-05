# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from pathlib import Path

import pytest

import evaspa.stic.config as cfg
from evaspa.__about__ import __version__


@pytest.mark.parametrize(
    "config",
    [
        {
            "filtering": {"water": "mask"},
        },
    ],
)
def test_paramsconfig(config) -> None:
    """
    Test FilterConfig
    """
    assert cfg.STICParamsConfig.model_validate(config)


@pytest.mark.parametrize(
    "version",
    [None, "0"],
)
def test_version(version, tmp_path) -> None:
    """
    Test version in InputFile
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {"path": "tests/data/modis_test.tif"},
        "output": {"path": str(d)},
        "params": {},
    }
    if version is not None:
        config["version"] = version
    entry = cfg.STICInputFile.model_validate(config)
    assert entry.version == __version__


def test_check() -> None:
    """
    Test check method
    """
    config = {
        "input": {"path": "tests/data/modis_test.tif"},
        "output": {"path": "out"},
        "params": {},
    }
    assert cfg.check_config_stic(config)
