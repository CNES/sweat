# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module for STIC configuration management
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

from evaspa.__about__ import __version__
from evaspa.common.constant import ETVar
from evaspa.common.daily import DailyConfig
from evaspa.common.filter import FilteringConfig
from evaspa.common.io import InputConfig, OutputConfig
from evaspa.debugging import DebuggingConfig
from evaspa.logging import LoggerManager
from evaspa.stic.main import STICModelConfig, STICPrepareConfig

logger = LoggerManager.get_logger(__name__)


class STICFilteringConfig(FilteringConfig):
    @field_validator("root", mode="before")
    @classmethod
    def update_config(cls, v):
        if isinstance(v, dict):
            if ETVar.DEWPOINT_TEMPERATURE.value in v:
                return v
            return {
                ETVar.DEWPOINT_TEMPERATURE.value: {"op": ">=", "value": -30}
            } | v
        return v


class STICParamsConfig(BaseModel):
    """
    Configuration for parameters to run STIC
    """

    model_config = ConfigDict(extra="forbid")

    prepare: STICPrepareConfig = Field(default=STICPrepareConfig())
    filtering: STICFilteringConfig = Field(default=STICFilteringConfig({}))
    stic: STICModelConfig = Field(default=STICModelConfig())
    daily: DailyConfig = Field(default=DailyConfig())


class STICInputFile(BaseModel):
    """
    Class describing the format of the input file
    """

    model_config = ConfigDict(extra="forbid")

    input: InputConfig
    output: OutputConfig
    params: STICParamsConfig = Field(default=STICParamsConfig())
    debug: DebuggingConfig = Field(default=DebuggingConfig())
    version: str = Field(default=str(__version__))

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


def check_config_stic(config: dict) -> dict:
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
    cfg = STICInputFile.model_validate(config)
    return cfg.model_dump(mode="json")
