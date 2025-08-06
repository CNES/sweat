# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from pathlib import Path

import pytest

import evaspa.common.config as cfg


def test_read_config() -> None:
    """
    Test read configuration
    """
    p = Path(".") / "tests" / "data" / "evaspa_input.json"
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
