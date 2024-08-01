#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd


@dataclass
class EFModel(ABC):
    """Abstract class for evaporative fraction model"""

    var: InitVar[np.ndarray]
    lst: InitVar[np.ndarray]
    dry_coeffs: tuple[float, float] = field(init=False)
    wet_coeffs: tuple[float, float] = field(init=False)

    def __post_init__(self, var: np.ndarray, lst: np.ndarray) -> None:
        """
        Post initilization method
        """
        self._estimate_edges(var, lst)

    @abstractmethod
    def _estimate_edges(self, var: np.ndarray, lst: np.ndarray) -> None:
        """
        Method to compute dry and wet edges
        """
        pass

    def dry_edge(self, var: float) -> float:
        """
        Compute dry temperature
        """
        return self.dry_coeffs[0] * var + self.dry_coeffs[1]

    def wet_edge(self, var: float) -> float:
        """
        Compute dry temperature
        """
        return self.wet_coeffs[0] * var + self.wet_coeffs[1]

    def compute_ef(self, var: np.ndarray, lst: np.ndarray) -> np.ndarray:
        """
        Method to compute evaporative fraction
        """
        ef = (self.dry_coeffs[0] * var + self.dry_coeffs[1] - lst) / (
            (self.dry_coeffs[0] - self.wet_coeffs[0]) * var
            + (self.dry_coeffs[1] - self.wet_coeffs[1])
        )
        ef = np.where(ef > 1, 1, ef)
        ef = np.where(ef < 0, 0, ef)
        return ef


@dataclass
class EFModel1(EFModel):
    """Abstract class for evaporative fraction model"""

    name: ClassVar[str] = "EF_1"
    doi: ClassVar[str] = "https://doi.org/10.1016/j.proenv.2013.06.035"

    nb_intervals: int = field(default=20, init=True)
    percentile: int = field(default=5, init=True)

    def _estimate_edges(self, var: np.ndarray, lst: np.ndarray) -> None:
        """
        Description
        -----------
        Domain division: Nb intervals of same pixels density

        Dry edge calculation:
        For each interval, compute the dry point:
          - compute the median of var values
          - compute the median of lst values of the percentile superior of the interval.
        Perform a linear regression from the dry points

        Wet edge calculation:
        For each interval, compute the wet point:
          - compute the median of var values
          - compute the median of lst values of the percentile inferior of the interval.
        Perform a linear regression from the wet points

        Outliers are not removed.

        Parameters
        ----------
        lst : np.array
            Land surface temperature
        var : np.array
            Variable used versus temperature (Albedo)
        """
        # removing common nan
        df = (
            pd.DataFrame(data={"lst": lst.reshape(-1), "var": var.reshape(-1)})
            .dropna(axis=0, how="any")
            .sort_values(by="var")
            .reset_index(drop=True)
        )

        var_values = []
        lst_sup_values = []
        lst_inf_values = []
        interval_size = int(np.floor(len(df) / self.nb_intervals))
        for i, group in df.groupby(df.index // interval_size):
            if i == self.nb_intervals:
                # Skip last not complete group
                break
            var_values.append(group["var"].median())
            lst_inf_values.append(
                group["lst"][
                    group["lst"] < np.percentile(group["lst"], self.percentile)
                ].median()
            )
            lst_sup_values.append(
                group["lst"][
                    group["lst"] > np.percentile(group["lst"], 100 - self.percentile)
                ].median()
            )

        self.dry_coeffs = np.polyfit(var_values, lst_sup_values, 1)
        self.wet_coeffs = np.polyfit(var_values, lst_inf_values, 1)

    def to_json(self) -> dict:
        """
        Return an dictionary
        """
        return {
            "name": self.name,
            "parameters": {
                "nb_intervals": self.nb_intervals,
                "percentile": self.percentile,
            },
        }

    def __repr__(self) -> str:
        """
        Print
        """
        return f"Method {self.name} with parameters: nb_intervals = {self.nb_intervals}, percentile = {self.percentile}"
