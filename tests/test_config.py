# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from pathlib import Path

import pytest
from pydantic import ValidationError

import evaspa.config as cfg
from evaspa.__about__ import __version__


def test_read_config() -> None:
    """
    Test read configuration
    """
    p = Path(".") / "tests" / "data" / "input.json"
    config = cfg.read_config(str(p))
    assert config


def test_read_config_exc() -> None:
    """
    Test read configuration (with exception)
    """
    p = Path(".") / "tests" / "data" / "input.yaml"
    with pytest.raises(OSError, match="Unable to read configuration file"):
        cfg.read_config(str(p))


@pytest.mark.parametrize(
    "fmt",
    [
        "json",
        "JSON",
    ],
)
def test_write_config(fmt, tmp_path) -> None:
    """
    Test write configuration
    """
    p = Path(tmp_path)
    config = {"input": "foo", "output": "foo", "params": "foo"}
    cfg.write_config(config, str(p), fmt=fmt)
    p = p / "config.json"
    assert p.exists()


def test_write_config_exc(tmp_path) -> None:
    """
    Test read configuration (with exception)
    """
    p = Path(tmp_path)
    config = {"input": "foo", "output": "foo", "params": "foo"}
    with pytest.raises(
        ValueError, match="Unsupported format for configuration file"
    ):
        cfg.write_config(config, str(p), fmt="yaml")


def test_inputconfig() -> None:
    """
    Test InputConfig
    """
    entry = {"path": "tests/data/modis_test.tif"}
    cfg.InputConfig.model_validate(entry)


@pytest.mark.parametrize(
    ("entry", "exception"),
    [
        pytest.param({"foo": "foo"}, pytest.raises(ValidationError)),
        pytest.param(
            {"path": "foo"}, pytest.raises(OSError, match="Path not found")
        ),
    ],
)
def test_inputconfig_exc(entry, exception) -> None:
    """
    Test InputConfig with error
    """
    with exception:
        cfg.InputConfig.model_validate(entry)


def test_outputconfig(tmp_path) -> None:
    """
    Test OutputConfig
    """
    d = Path(tmp_path) / "out"
    output = {"path": str(d)}
    cfg.OutputConfig.model_validate(output)
    assert d.exists()


@pytest.mark.parametrize(
    "config",
    [
        {"ef": {"models": "default_evaspa"}},
        {"ef": {"check": {"threshold": 10}, "models": "default_evaspa"}},
        {
            "filtering": {"cloud": "cloud_mask"},
            "ef": {"models": "default_evaspa"},
        },
        {"ef": {"models": "default_evaspa"}, "seb": {"use_topo": True}},
        {
            "filtering": {"water": "mask"},
            "ef": {"models": "default_evaspa"},
            "seb": {"use_topo": True},
        },
    ],
)
def test_paramsconfig(config) -> None:
    """
    Test FilterConfig
    """
    assert cfg.ParamsConfig.model_validate(config)


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
    entry = cfg.InputFile.model_validate(config)
    assert entry.version == __version__


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
    assert cfg.check_config(config)
