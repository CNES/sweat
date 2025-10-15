# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for EVASPA configuration management
"""

from __future__ import annotations

import os

from packaging.version import Version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from sweat.__about__ import __version__
from sweat.common.daily import DailyConfig
from sweat.common.filter import FilteringConfig
from sweat.common.io import InputConfig, OutputConfig
from sweat.debugging import DebuggingConfig
from sweat.evaspa.ef import EFConfig
from sweat.evaspa.seb import SEBConfig
from sweat.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)


class EVASPAInputFile(BaseModel):
    """
    Class describing the format of the input file
    """

    model_config = ConfigDict(extra="forbid")

    input: InputConfig
    output: OutputConfig
    params: EVASPAParamsConfig
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
        self.debug.path = os.path.join(self.output.path, "debug")
        # Create debug if necessary
        if self.debug.verbose:
            os.makedirs(self.debug.path, exist_ok=True)
        return self


class EVASPAParamsConfig(BaseModel):
    """
    Configuration for parameters to run EVASPA
    """

    model_config = ConfigDict(extra="forbid")

    filtering: FilteringConfig = Field(default=FilteringConfig({}))
    ef: EFConfig
    seb: SEBConfig = Field(default=SEBConfig())
    daily: DailyConfig = Field(default=DailyConfig())


def check_config_evaspa(config: dict) -> dict:
    """
    Check configuration for EVASPA

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration parameters

    Returns
    -------
    checked_config: dict
        Checked dictionary containing the configuration parameters
    """
    cfg = EVASPAInputFile.model_validate(config)
    return cfg.model_dump(mode="json")
