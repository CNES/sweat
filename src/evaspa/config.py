#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for configuration management
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from typing_extensions import Annotated

from pydantic import BaseModel, AfterValidator, field_validator, Field, ConfigDict

from evaspa.__about__ import __version__
from evaspa.filter import FilterConfig
from evaspa.ef import EFConfig
from evaspa.seb import SEBConfig


class InputFile(BaseModel):
    """
    Class describing the format of the input file
    """

    input: InputConfig
    output: OutputConfig
    params: ParamsConfig
    version: str = Field(default=str(__version__))

    @field_validator("version")
    @classmethod
    def update_version(cls, v: str) -> str:
        return str(__version__)


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
            raise IOError(f"Path not found: {v}")
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
            json_data = json.load(json_file)
            return json_data
    else:
        raise IOError("Unable to read configuration file (unknown format)")


def write_config(config: dict, path: str, format: str = "json") -> None:
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
    format: str
        Configuration file format (default=JSON)
    """
    output_dir = Path(path)
    if format.lower() == "json":
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4, default=lambda x: x.value)
    else:
        raise ValueError(f"Unsupported format for configuration file ({format})")


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
