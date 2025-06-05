# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for common configuration management
"""

from __future__ import annotations

import datetime as dt
import json
from enum import Enum
from pathlib import Path


def read_config(path: str) -> dict:
    """
    Description
    -----------
    Read configuration and return a dict

    Parameters
    ----------
    path: str
        Path to configuration file

    Return
    ------
    config: dict
        Dictionary containing the configuration parameters
    """
    suffix = Path(path).suffix
    if suffix == ".json":
        with open(path) as json_file:
            return json.load(json_file)
    else:
        msg = "Unable to read configuration file (unknown format)"
        raise OSError(msg)


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (dt.datetime | dt.date)):
        return obj.isoformat()
    elif isinstance(obj, Enum):  # noqa I001
        return obj.value
    msg = f"Type {type(obj)} not serializable"
    raise TypeError(msg)


def write_config(config: dict, path: str, fmt: str = "json") -> None:
    """
    Description
    -----------
    Write configuration and return a dict

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration parameters
    path: str
        Directory path
    fmt: str
        Configuration file format (default=JSON)
    """
    output_dir = Path(path)
    if fmt.lower() == "json":
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4, default=json_serial)
    else:
        msg = f"Unsupported format for configuration file ({fmt})"
        raise ValueError(msg)
