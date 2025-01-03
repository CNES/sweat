# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for configuration management
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from packaging.version import Version
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from evaspa.__about__ import __version__
from evaspa.debugging import DebuggingConfig
from evaspa.ef import EFConfig  # noqa TC001
from evaspa.filter import FilterConfig
from evaspa.logging import LoggerManager
from evaspa.seb import SEBConfig

logger = LoggerManager.get_logger(__name__)


class InputFile(BaseModel):
    """
    Class describing the format of the input file
    """

    model_config = ConfigDict(extra="forbid")

    input: InputConfig
    output: OutputConfig
    params: ParamsConfig
    debug: DebuggingConfig = Field(default=DebuggingConfig())
    version: str = Field(default=str(__version__))

    # TODO: Validate the path for debugging with the output path
    @field_validator("version")
    @classmethod
    def update_version(cls, v: str) -> str:
        """ "
        Update version in configuration
        """
        if Version(v) > Version(__version__):
            msg = "File generated with a newer version"
            logger.warning(msg)
        return str(__version__)

    @model_validator(mode="after")
    def update_debug(self):
        """
        Update debug path with output path
        """
        self.debug.path = self.output.path
        return self


class InputConfig(BaseModel):
    """
    Configuration for input data
    """

    model_config = ConfigDict(extra="forbid")

    path: str

    @field_validator("path")
    @classmethod
    def test_path(cls, v: str) -> str:
        if not os.path.exists(v):
            msg = f"Path not found: {v}"
            raise OSError(msg)
        return v


def create_path(v: str):
    os.makedirs(v, exist_ok=True)
    return v


OutputPath = Annotated[str, AfterValidator(create_path)]


class OutputConfig(BaseModel):
    """
    Configuration for output data
    """

    model_config = ConfigDict(extra="forbid")

    path: OutputPath


class ParamsConfig(BaseModel):
    """
    Configuration for parameters to run EVASPA
    """

    model_config = ConfigDict(extra="forbid")

    filtering: FilterConfig = Field(default=FilterConfig())
    ef: EFConfig
    seb: SEBConfig = Field(default=SEBConfig())


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
            json.dump(config, f, indent=4, default=lambda x: x.value)
    else:
        msg = f"Unsupported format for configuration file ({fmt})"
        raise ValueError(msg)


def check_config(config: dict) -> dict:
    """
    Description
    -----------
    Check configuration

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration parameters

    Returns
    -------
    checked_config: dict
        Checked dictionary containing the configuration parameters
    """
    cfg = InputFile.model_validate(config)
    return cfg.model_dump(mode="json")
