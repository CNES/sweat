# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

PERCENTILE_MIN = 0
PERCENTILE_MAX = 100
NB_INTERVAL_MAX = 1000


class EdgeError(Exception):
    """Exception in edge creation"""


class IntervalType(Enum):
    """Interval type used to divide data in partition"""

    SIZE = "size"
    DENSITY = "density"


class EdgeConfig(BaseModel):
    """Configuration of a edge"""

    type: str
    config: dict


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
    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
        """
        Description
        -----------
        Method to compute the edge parameters

        Parameters
        ----------
        lst : np.array_like
            Land surface temperature
        var : np.array_like
            Variable used versus temperature (ex: Albedo)
        """

    @abstractmethod
    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value at var value.

        Parameters
        ----------
        var : np.array_like
            Variable

        Returns
        -------
        temperature : np.array
            Temperature at the edge
        """

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return self.model_dump()

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

        Returns
        -------
        data: DataFrame
            Prepared data in a dataframe format
        """
        if var.shape != lst.shape:
            msg = "LST and variable do not have the same size"
            raise ValueError(msg)
        return (
            pd.DataFrame(data={"lst": lst.reshape(-1), "var": var.reshape(-1)})
            .dropna(axis=0, how="any")
            .sort_values(by="var")
            .reset_index(drop=True)
        )

    @classmethod
    def create(cls, name: str, config: dict) -> Edge:
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

        Returns
        -------
        edge: Edge
            Edge from the configuration
        """
        # Init
        try:
            return getattr(sys.modules[__name__], name).model_validate_json(
                json.dumps(config)
            )
        except KeyError as e:
            msg = f"Class {name} is not defined"
            raise EdgeError(msg) from e
        except ValidationError as e:
            msg = "Error in edge configuration"
            raise EdgeError(msg) from e


class LinearEdge(Edge):
    """Class for linear edge"""

    model_config = ConfigDict(allow_inf_nan=True, ser_json_inf_nan="strings")
    percentile: tuple[int, int]
    interval_type: IntervalType = IntervalType.SIZE
    interval_nb: int = 20
    selection: SelectionMethod = SelectionMethod.MEDIAN

    coeffs: tuple[float, float] = (float("inf"), float("inf"))

    @field_validator("percentile")
    @classmethod
    def check_percentile(cls, p: tuple[int, int]) -> tuple[int, int]:
        """
        Description
        -----------
        Check the consistency of the percentile interval

        Parameters
        ----------
        p : tuple[int,int]
            Percentile interval

        Returns
        -------
        percentile: tuple[int,int]
            Validated percentile interval
        """
        if (
            (p[0] > p[1])
            or (p[0] < PERCENTILE_MIN)
            or (p[0] >= PERCENTILE_MAX)
            or (p[1] <= PERCENTILE_MIN)
            or (p[1] > PERCENTILE_MAX)
        ):
            msg = "Percentile must be an interval between [0,100]"
            raise ValueError(msg)
        return p

    @field_validator("interval_nb")
    @classmethod
    def check_interval_nb(cls, nb: int) -> int:
        """
        Description
        -----------
        Check the consistency of the interval number

        Parameters
        ----------
        nb : int
            Interval number

        Returns
        -------
        interval_nb: int
            Validated interval number
        """
        if (nb <= 0) or (nb > NB_INTERVAL_MAX):
            msg = f"Number of intervals must be between 1 and {NB_INTERVAL_MAX}"
            raise ValueError(msg)
        return nb

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value

        Parameters
        ----------
        var : np.array_like
            Variable

        Returns
        -------
        temperature : np.array
            Temperature at the edge
        """
        return self.coeffs[0] * np.array(var) + self.coeffs[1]

    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
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
        lst : np.array_like
            Land surface temperature
        var : np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        # Init
        if np.array(var).shape != np.array(lst).shape:
            msg = "LST and variable do not have the same size"
            raise ValueError(msg)
        var_values = []
        lst_values = []
        df = self._prepare(np.array(var), np.array(lst))
        intervals = self._get_intervals(df["var"])

        # Compute point coordinates for regression
        for _, group in df.groupby(intervals):
            value = group["lst"][
                (group["lst"] > np.percentile(group["lst"], self.percentile[0]))
                & (
                    group["lst"]
                    <= np.percentile(group["lst"], self.percentile[1])
                )
            ].agg(self.selection.value)
            if not np.isnan(value):
                var_values.append(group["var"].median())
                lst_values.append(value)

        # Linear regression
        self.coeffs = tuple(np.polyfit(var_values, lst_values, 1))

    def _get_intervals(self, values: pd.Series) -> pd.Series:
        """
        Description
        -----------
        Divide the domain in intervals

        Parameters
        ----------
        values: pd.Series

        Returns
        -------
        intervals: pd.Series
            Intervals to consider
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

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return {
            "percentile": self.percentile,
            "interval_type": self.interval_type.value,
            "interval_nb": self.interval_nb,
            "selection": self.selection.value,
            "coeffs": tuple(float(coeff) for coeff in self.coeffs),
        }


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
        var : np.array_like
            Variable

        Returns
        -------
        temperature : np.array
            Temperature at the edge
        """
        return self.value * np.ones_like(np.array(var))

    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:  # noqa: ARG002
        """
        Description
        -----------
        Edge is defined by the max or the min of LST

        Parameters
        ----------
        lst : np.array_like
            Land surface temperature
        var : np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        if self.selection == SelectionFlatMethod.MAX:
            self.value = np.nanmax(np.array(lst))
        elif self.selection == SelectionFlatMethod.MIN:
            self.value = np.nanmin(np.array(lst))

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return {
            "selection": self.selection.value,
            "value": float(self.value),
        }
