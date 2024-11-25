#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for configuration management
"""

from __future__ import annotations

import os

from typing_extensions import Annotated

from pydantic import BaseModel, AfterValidator, field_validator

from evaspa.ef import EFConfig


class InputFile(BaseModel):
    """
    Class describing the format of the input file
    """

    input: InputConfig
    output: OutputConfig
    params: ParamsConfig


class InputConfig(BaseModel):
    """
    Configuration for input data
    """

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

    path: OutputPath


class ParamsConfig(BaseModel):
    """
    Configuration for parameters to run EVASPA
    """

    filtering: dict
    check_variability: dict
    efconfig: EFConfig
