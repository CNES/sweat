# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from pathlib import Path

import pytest

import evaspa.stic.config as cfg
from evaspa.__about__ import __version__


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {},
            {
                "tdp": {"op": ">=", "value": -30},
            },
        ),
        pytest.param(
            {
                "cloud": {"op": "==", "value": 0},
            },
            {
                "cloud": {"op": "==", "value": 0},
                "tdp": {"op": ">=", "value": -30},
            },
        ),
        pytest.param(
            {
                "tdp": {"op": ">", "value": 0},
            },
            {
                "tdp": {"op": ">", "value": 0},
            },
        ),
    ],
)
def test_filteringconfig(config, expected) -> None:
    """
    Test FilteringConfig
    """
    res = cfg.STICFilteringConfig.model_validate(config)
    assert res.model_dump() == expected


@pytest.mark.parametrize(
    "config",
    [
        {
            "filtering": {"cloud": {"op": "!=", "value": 1}},
        },
        {
            "filtering": {"cloud": {"op": "!=", "value": 1}},
            "stic": {},
        },
        {
            "filtering": {"cloud": {"op": "!=", "value": 1}},
            "stic": {},
            "daily": {"method": "toa"},
        },
        {
            "filtering": {"cloud": {"op": "!=", "value": 1}},
            "stic": {"threshold": 0.5},
            "daily": {"method": "toa"},
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


def test_version_with_warnings(tmp_path, caplog) -> None:
    """
    Test version in InputFile
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {"path": "tests/data/modis_test.tif"},
        "output": {"path": str(d)},
        "params": {},
        "version": "1000",
    }
    _ = cfg.STICInputFile.model_validate(config)
    msg = "File generated with a newer version"
    assert msg in caplog.text


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


def test_check_debug() -> None:
    """
    Test check method
    """
    config = {
        "input": {"path": "tests/data/modis_test.tif"},
        "output": {"path": "out"},
        "params": {},
        "debug": {"profile": True, "verbose": True},
    }
    assert cfg.check_config_stic(config)
