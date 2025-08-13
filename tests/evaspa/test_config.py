# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from pathlib import Path

import pytest

import evaspa.evaspa.config as cfg
from evaspa.__about__ import __version__


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"ef": {"models": "default_evaspa"}},
        {"ef": {"check": {"threshold": 10}, "models": "default_evaspa"}},
        {
            "filtering": {"cloud": {"op": "!=", "value": 1}},
            "ef": {"models": "default_evaspa"},
        },
        {"ef": {"models": "default_evaspa"}, "seb": {"use_topo": True}},
        {
            "filtering": {"water": {"op": "!=", "value": 1}},
            "ef": {"models": "default_evaspa"},
            "seb": {"use_topo": True},
        },
    ],
)
def test_paramsconfig(config) -> None:
    """
    Test FilterConfig
    """
    assert cfg.EVASPAParamsConfig.model_validate(config)


@pytest.mark.unit
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
        "params": {
            "ef": {
                "check": {"threshold": 10},
                "models": "default_evaspa",
            }
        },
    }
    if version is not None:
        config["version"] = version
    entry = cfg.EVASPAInputFile.model_validate(config)
    assert entry.version == __version__


@pytest.mark.unit
def test_check() -> None:
    """
    Test check method
    """
    config = {
        "input": {"path": "tests/data/modis_test.tif"},
        "output": {"path": "out"},
        "params": {
            "ef": {
                "check": {"threshold": 10},
                "models": "default_evaspa",
                "options": {"merging": "mean"},
            },
        },
    }
    assert cfg.check_config_evaspa(config)


@pytest.mark.unit
def test_check_debug() -> None:
    """
    Test check method
    """
    config = {
        "input": {"path": "tests/data/modis_test.tif"},
        "output": {"path": "out"},
        "params": {
            "ef": {
                "check": {"threshold": 10},
                "models": "default_evaspa",
                "options": {"merging": "mean"},
            },
        },
        "debug": {"profile": True, "verbose": True},
    }
    assert cfg.check_config_evaspa(config)
