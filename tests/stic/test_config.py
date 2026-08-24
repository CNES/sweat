# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from pathlib import Path

import pytest
from pydantic import ValidationError

import sweat.stic.config as cfg
from sweat.__about__ import __version__


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {},
            {
                "use_topo": False,
                "selected_radiation": None,
                "version": None,
            },
        ),
        pytest.param(
            {
                "use_topo": True,
                "selected_radiation": "msg",
            },
            {
                "use_topo": True,
                "selected_radiation": "msg",
                "version": None,
            },
        ),
        pytest.param(
            {
                "use_topo": True,
                "version": "1.4",
            },
            {
                "use_topo": True,
                "selected_radiation": None,
                "version": "1.4",
            },
        ),
    ],
)
def test_sticprepareconfig(config, expected) -> None:
    """
    Test STICModelConfig
    """
    res = cfg.STICPrepareConfig.model_validate(config)
    assert res.model_dump() == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {
            "foo": False,
            "use_topo": True,
        },
        {
            "use_topo": True,
            "selected_radiation": 10,
        },
    ],
)
def test_sticprepareconfig_error(config) -> None:
    """
    Test STICModelConfig with error
    """
    with pytest.raises(ValidationError):
        cfg.STICPrepareConfig.model_validate(config)


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {},
            {
                "version": "1.3",
                "threshold": 0.01,
                "nb_steps": 15,
            },
        ),
        pytest.param(
            {
                "threshold": 0.01,
                "nb_steps": 15,
            },
            {
                "version": "1.3",
                "threshold": 0.01,
                "nb_steps": 15,
            },
        ),
        pytest.param(
            {"version": "1.3"},
            {
                "version": "1.3",
                "threshold": 0.01,
                "nb_steps": 15,
            },
        ),
        pytest.param(
            {"version": "1.4", "threshold": 0.05},
            {
                "version": "1.4",
                "threshold": 0.05,
                "nb_steps": 15,
            },
        ),
    ],
)
def test_sticmodelconfig(config, expected) -> None:
    """
    Test STICModelConfig
    """
    res = cfg.STICModelConfig.model_validate(config)
    assert res.model_dump() == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"version": "1.0"},
        {"version": "1.4", "foo": 20},
    ],
)
def test_sticmodelconfig_error(config) -> None:
    """
    Test STICModelConfig with error
    """
    with pytest.raises(ValidationError):
        cfg.STICModelConfig.model_validate(config)


@pytest.mark.unit
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
            "stic": {"version": "1.3", "threshold": 0.05},
            "daily": {"method": "toa"},
            "waterstress": {"method": "ef"},
        },
        {
            "prepare": {"use_topo": True},
            "filtering": {"cloud": {"op": "!=", "value": 1}},
            "stic": {"version": "1.4"},
        },
        {
            "prepare": {"version": "1.3"},
            "filtering": {"cloud": {"op": "!=", "value": 1}},
            "stic": {"version": "1.4"},
        },
    ],
)
def test_paramsconfig(config) -> None:
    """
    Test STICParamsConfig
    """
    checked_config = cfg.STICParamsConfig.model_validate(config)
    assert checked_config
    assert checked_config.prepare.version == checked_config.stic.version


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
        "input": {"path": "tests/data/data_test_20230303T112730.tif"},
        "output": {"path": str(d)},
        "params": {},
    }
    if version is not None:
        config["version"] = version
    entry = cfg.STICInputFile.model_validate(config)
    assert entry.version == __version__


@pytest.mark.unit
def test_version_with_warnings(tmp_path, caplog) -> None:
    """
    Test version in InputFile
    """
    d = Path(tmp_path) / "out"
    config = {
        "input": {"path": "tests/data/data_test_20230303T112730.tif"},
        "output": {"path": str(d)},
        "params": {},
        "version": "1000",
    }
    _ = cfg.STICInputFile.model_validate(config)
    msg = "File generated with a newer version"
    assert msg in caplog.text


@pytest.mark.unit
def test_check() -> None:
    """
    Test check method
    """
    config = {
        "input": {"path": "tests/data/data_test_20230303T112730.tif"},
        "output": {"path": "out"},
        "params": {},
    }
    assert cfg.check_config_stic(config)


@pytest.mark.unit
def test_check_debug() -> None:
    """
    Test check method with debug section
    """
    config = {
        "input": {"path": "tests/data/data_test_20230303T112730.tif"},
        "output": {"path": "out"},
        "params": {},
        "debug": {"profile": True, "verbose": True},
    }
    assert cfg.check_config_stic(config)
