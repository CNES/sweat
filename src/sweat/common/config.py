# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for common configuration management
"""

from __future__ import annotations

import datetime as dt
import json
from enum import Enum
from pathlib import Path

import yaml

from sweat.common.types import PercentileValue


def read_config(path: str) -> dict:
    """
    Read configuration and return a dict

    Parameters
    ----------
    path: str
        Path to configuration file

    Returns
    -------
    config: dict
        Dictionary containing the configuration parameters
    """
    suffix = Path(path).suffix
    if suffix == ".json":
        with open(path) as json_file:
            return json.load(json_file)
    elif suffix in (".yaml", ".yml"):
        with open(path) as yaml_file:
            return yaml.safe_load(yaml_file)
    else:
        msg = "Unable to read configuration file (unknown format)"
        raise OSError(msg)


def json_yaml_serial(obj):
    """
    JSON and YAML serializer for objects not serializable by default
    json/yaml code
    """

    if isinstance(obj, (dt.datetime | dt.date)):
        return obj.isoformat()
    elif isinstance(obj, Enum):  # noqa I001
        return obj.value
    elif isinstance(obj, PercentileValue):
        return str(obj)
    msg = f"Type {type(obj)} not serializable"
    raise TypeError(msg)


def write_config(config: dict, path: str, fmt: str = "json") -> None:
    """
    Write configuration and return a dict

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration parameters
    path: str
        Directory path
    fmt: str
        Configuration file format (default=JSON), "json" or "yaml"/"yml"
    """
    output_dir = Path(path)
    fmt = fmt.lower()

    if fmt == "json":
        with open(output_dir / "config.json", "w") as f:
            json.dump(
                config, f, indent=4, allow_nan=True, default=json_yaml_serial
            )
    elif fmt in ("yaml", "yml"):
        serializable_config = json.loads(
            json.dumps(config, default=json_yaml_serial)
        )
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(
                serializable_config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
    else:
        msg = f"Unsupported format for configuration file ({fmt})"
        raise ValueError(msg)
