#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import pytest
from pathlib import Path
from pydantic import ValidationError

import evaspa.config as cfg
from evaspa.__about__ import __version__


def test_read_config() -> None:
    """
    Test read configuration
    """
    p = Path(".") / "tests" / "data" / "input.json"
    config = cfg.read_config(p)
    assert config


def test_read_config_exc() -> None:
    """
    Test read configuration (with exception)
    """
    with pytest.raises(IOError):
        p = Path(".") / "tests" / "data" / "input.yaml"
        cfg.read_config(p)


@pytest.mark.parametrize(
    "format",
    [
        "json",
        "JSON",
    ],
)
def test_write_config(format, tmp_path) -> None:
    """
    Test write configuration
    """
    p = Path(tmp_path)
    config = {"input": "foo", "output": "foo", "params": "foo"}
    cfg.write_config(config, p, format=format)
    p = p / "config.json"
    assert p.exists()


def test_write_config_exc(tmp_path) -> None:
    """
    Test read configuration (with exception)
    """
    with pytest.raises(ValueError):
        p = Path(tmp_path)
        config = {"input": "foo", "output": "foo", "params": "foo"}
        cfg.write_config(config, p, format="yaml")


def test_inputconfig() -> None:
    """
    Test InputConfig
    """
    input = {"path": "tests/data/modis_test.tif"}
    cfg.InputConfig.model_validate(input)


@pytest.mark.parametrize(
    "input,exception",
    [
        pytest.param({"foo": "foo"}, pytest.raises(ValidationError)),
        pytest.param({"path": "foo"}, pytest.raises(IOError)),
    ],
)
def test_inputconfig_exc(input, exception) -> None:
    """
    Test InputConfig with error
    """
    with exception:
        cfg.InputConfig.model_validate(input)


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
    config = dict(
        input={"path": "tests/data/modis_test.tif"},
        output={"path": str(d)},
        params={
            "ef": {
                "check": {"threshold": 10},
                "models": "default_evaspa",
            }
        },
    )
    if version is not None:
        config["version"] = version
    print(config)
    input = cfg.InputFile.model_validate(config)
    assert input.version == __version__


def test_check() -> None:
    """
    Test check method
    """
    config = dict(
        input={"path": "tests/data/modis_test.tif"},
        output={"path": "out"},
        params={
            "ef": {
                "check": {"threshold": 10},
                "models": "default_evaspa",
                "options": {"merging": "mean"},
            },
        },
    )
    assert cfg.check_config(config)
