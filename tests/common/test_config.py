# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import datetime as dt
from pathlib import Path

import pytest

import sweat.common.config as cfg
from sweat.common.constant import ETVar
from sweat.common.types import PercentileValue


@pytest.mark.unit
def test_read_config() -> None:
    """
    Test read configuration
    """
    p = Path(".") / "tests" / "data" / "evaspa_input.json"
    config = cfg.read_config(str(p))
    assert config


@pytest.mark.unit
def test_read_config_exc() -> None:
    """
    Test read configuration (with exception)
    """
    p = Path(".") / "tests" / "data" / "input.yaml"
    with pytest.raises(OSError, match="Unable to read configuration file"):
        cfg.read_config(str(p))


@pytest.mark.unit
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


@pytest.mark.unit
def test_write_config_exc(tmp_path) -> None:
    """
    Test read configuration (with exception)
    """
    p = Path(tmp_path)
    config = {
        "input": dt.datetime.now(tz=dt.UTC),
        "output": "foo",
        "params": PercentileValue(10),
        "variable": ETVar.LST,
    }
    with pytest.raises(
        ValueError, match="Unsupported format for configuration file"
    ):
        cfg.write_config(config, str(p), fmt="yaml")
