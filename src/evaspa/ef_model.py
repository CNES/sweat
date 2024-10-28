#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from pydantic import BaseModel, ValidationError

from .edge import Edge, create_edge, EdgeError


class EdgeConfig(BaseModel):
    type: str
    config: dict


class EFModelConfig(BaseModel):
    dry_edge: EdgeConfig
    wet_edge: EdgeConfig


class EFConfigError(Exception):
    """Exception in EF Model configuration"""


class EFModelError(Exception):
    """Exception in EF model creation"""


@dataclass(frozen=True)
class EFModel:
    """Class for evaporative fraction model"""

    wet_edge: Edge
    dry_edge: Edge

    def fit(self, var: npt.NDArray, lst: npt.NDArray) -> None:
        """
        Method to compute dry and wet edges
        """
        self.wet_edge.fit(var, lst)
        self.dry_edge.fit(var, lst)

    def tdry(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Compute dry temperature
        """
        return self.dry_edge.get(var)

    def twet(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Compute wet temperature
        """
        return self.wet_edge.get(var)

    def compute(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> npt.NDArray:
        """
        Method to compute evaporative fraction
        """
        ef = (self.tdry(var) - np.array(lst)) / (self.tdry(var) - self.twet(var))
        ef = np.where(ef > 1, 1, ef)
        ef = np.where(ef < 0, 0, ef)
        return ef

    def to_json(self) -> dict:
        """
        Return an dictionary
        model_dump_json
        """
        return {
            "dry_edge": {
                "type": self.dry_edge.__class__,
                "config": self.dry_edge.model_dump_json(),
            },
            "wet_edge": {
                "type": self.wet_edge.__class__,
                "config": self.wet_edge.model_dump_json(),
            },
        }

    def __str__(self) -> str:
        """
        String conversion
        """
        return (
            f"Model(dry_edge={self.dry_edge.__class__},"
            f"wet_edge={self.wet_edge.__class__})"
        )

    def __repr__(self) -> str:
        """
        For print method
        """
        return (
            "Model:\n" f"dry edge = {self.dry_edge}\n" f"wet edge = {self.wet_edge}\n"
        )


def check_efmodel(config: dict) -> EFModelConfig:
    """
    Description
    """
    try:
        efconfig = EFModelConfig.model_validate(config)
    except ValidationError as e:
        raise EFConfigError("Error in model configuration") from e
    return efconfig


def create_efmodel(config: dict) -> EFModel:
    """
    Description
    """
    # Read configuration
    efconfig = check_efmodel(config)

    # Dry edge
    try:
        dry_edge = create_edge(efconfig.dry_edge.type, efconfig.dry_edge.config)
    except EdgeError as e:
        raise EFModelError("Error in dry edge creation") from e

    # Wet edge
    try:
        wet_edge = create_edge(efconfig.wet_edge.type, efconfig.wet_edge.config)
    except EdgeError as e:
        raise EFModelError("Error in wet edge creation") from e

    return EFModel(dry_edge=dry_edge, wet_edge=wet_edge)
