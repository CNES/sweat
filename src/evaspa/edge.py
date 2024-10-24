#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import json
import numpy as np
import numpy.typing as npt
import pandas as pd
import sys

from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, ConfigDict, ValidationError


class EdgeError(Exception):
    """Exception in edge creation"""


class IntervalType(Enum):
    """Interval type used to divide data in partition"""

    SIZE = "size"
    DENSITY = "density"


class SelectionMethod(Enum):
    """Method used to select point for the regression"""

    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    MEAN = "mean"


class SelectionFlatMethod(Enum):
    """Method used to select flat edge position"""

    MIN = "min"
    MAX = "max"


class Edge(BaseModel, ABC):
    """Abstract class for edge"""

    model_config = ConfigDict(allow_inf_nan=True, ser_json_inf_nan="strings")

    @abstractmethod
    def fit(self, var: npt.NDArray, lst: npt.NDArray) -> None:
        """
        Description
        -----------
        Method to compute the edge parameters

        Parameters
        ----------
        lst : np.array
            Land surface temperature
        var : np.array
            Variable used versus temperature (ex: Albedo)
        """
        pass

    @abstractmethod
    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value at var value.

        Parameters
        ----------
        var : float or np.ndarray
            Variable
        """
        pass

    def _prepare(self, var: npt.NDArray, lst: npt.NDArray) -> pd.DataFrame:
        """
        Description
        -----------
        Prepare data

        Parameters
        ----------
        lst : np.array
            Land surface temperature
        var : np.array
            Variable used versus temperature (ex: Albedo)
        return: df. DataFrame
        """
        assert var.shape == lst.shape
        return (
            pd.DataFrame(data={"lst": lst.reshape(-1), "var": var.reshape(-1)})
            .dropna(axis=0, how="any")
            .sort_values(by="var")
            .reset_index(drop=True)
        )


class LinearEdge(Edge):
    """Class for linear edge"""

    model_config = ConfigDict(allow_inf_nan=True, ser_json_inf_nan="strings")
    percentile: tuple[int, int]
    interval_type: IntervalType = IntervalType.SIZE
    interval_nb: int = 20
    selection: SelectionMethod = SelectionMethod.MEDIAN

    coeffs: tuple[float, float] = (float("inf"), float("inf"))

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value

        Parameters
        ----------
        var : float or np.ndarray
            Variable
        """
        return self.coeffs[0] * np.array(var) + self.coeffs[1]

    def fit(self, var: npt.NDArray, lst: npt.NDArray) -> None:
        """
        Description
        -----------
        For each interval, compute the point coordinates used for the regression:
          - the abscissa value is obtained by taking the median.
          - the ordinate value is obtained by applying the
          selection method to the percentile interval.
        Perform a linear regression from the points

        Parameters
        ----------
        lst : np.array
            Land surface temperature
        var : np.array
            Variable used versus temperature (ex: Albedo)
        """
        # Init
        assert var.shape == lst.shape
        var_values = []
        lst_values = []
        df = self._prepare(var, lst)
        intervals = self._get_intervals(df["var"])

        # Compute point coordinates for regression
        for i, group in df.groupby(intervals):
            var_values.append(group["var"].median())
            lst_values.append(
                group["lst"][
                    (group["lst"] > np.percentile(group["lst"], self.percentile[0]))
                    & (group["lst"] <= np.percentile(group["lst"], self.percentile[1]))
                ].agg(self.selection.value)
            )

        # Linear regression
        self.coeffs = np.polyfit(var_values, lst_values, 1)

    def _get_intervals(self, values: pd.Series) -> pd.Series:
        """
        Description
        -----------
        Divide the domain in intervals

        Parameters
        ----------
        values: pd.Series
        return: pd.Series
        """
        if self.interval_type == IntervalType.DENSITY:
            # Intervals with the same number of elements
            interval_size = int(np.floor(len(values) / self.interval_nb))
            intervals = pd.Series(np.arange(len(values)) // interval_size)
            intervals.loc[intervals >= self.interval_nb] = self.interval_nb - 1
        else:
            # Intervals with a fixed size
            value_min = values.min()
            value_max = values.max()
            interval_size = (value_max - value_min) / self.interval_nb
            intervals = ((values - value_min) / interval_size).astype(int)
            intervals.loc[intervals >= self.interval_nb] = self.interval_nb - 1

        return intervals

    def __str__(self) -> str:
        """
        String conversion
        """
        return (
            f"LinearEdge(percentile={self.percentile},"
            f"interval_type={self.interval_type.value},"
            f"interval_nb={self.interval_nb},"
            f"selection={self.selection.value},"
            f"coeffs={self.coeffs})"
        )

    def __repr__(self) -> str:
        """
        For print method
        """
        return (
            f"LinearEdge(percentile={self.percentile},"
            f"interval_type={self.interval_type.value},"
            f"interval_nb={self.interval_nb},"
            f"selection={self.selection.value},"
            f"coeffs={self.coeffs})"
        )


class FlatEdge(Edge):
    """Class for flat edge"""

    model_config = ConfigDict(allow_inf_nan=True, ser_json_inf_nan="strings")
    selection: SelectionFlatMethod
    value: float = float("inf")

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value at var value.

        Parameters
        ----------
        var : float or np.ndarray
            Variable
        """
        return self.value * np.ones_like(np.array(var))

    def fit(self, var: npt.NDArray, lst: npt.NDArray) -> None:
        """
        Description
        -----------
        Edge is defined by the max or the min of LST

        Parameters
        ----------
        lst : np.array
            Land surface temperature
        var : np.array
            Variable used versus temperature (ex: Albedo)
        """
        if self.selection == SelectionFlatMethod.MAX:
            self.value = np.nanmax(lst)
        elif self.selection == SelectionFlatMethod.MIN:
            self.value = np.nanmin(lst)


def create_edge(name: str, config: dict):
    """
    Description
    -----------
    Function to create an edge based on the
    name of the edge (edge class) and a configuration

    Parameters
    ----------
        : np.array
        Land surface temperature
    var : np.array
        Variable used versus temperature (ex: Albedo)
    """
    # Init
    try:
        return getattr(sys.modules[__name__], name).model_validate_json(
            json.dumps(config)
        )
    except KeyError as e:
        raise EdgeError(f"Class {name} is not defined") from e
    except ValidationError as e:
        raise EdgeError("Error in edge configuration") from e
