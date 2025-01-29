# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd
import pwlf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from evaspa.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

PERCENTILE_MIN = 0
PERCENTILE_MAX = 100
NB_INTERVAL_MAX = 1000
SIZE_INTERVAL_MAX = 1


class EdgeError(Exception):
    """Exception in edge creation"""


class IntervalType(Enum):
    """Interval type used to divide data in partition"""

    SIZE = "size"
    DENSITY = "density"


class EdgeConfig(BaseModel):
    """Configuration of a edge"""

    model_config = ConfigDict(extra="forbid")

    type: str
    config: dict = Field(default={})


class SelectionMethod(Enum):
    """Method used to select point for the regression"""

    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    MEAN = "mean"


class EdgePosition(Enum):
    """Edge position"""

    TOP = "top"
    BOTTOM = "bottom"


class Edge(BaseModel, ABC):
    """Abstract class for edge"""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=True, ser_json_inf_nan="strings"
    )
    position: EdgePosition

    @abstractmethod
    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
        """
        Description
        -----------
        Method to compute the edge parameters

        Parameters
        ----------
        lst: np.array_like
            Land surface temperature
        var: np.array_like
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
        var: np.array_like
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
        lst: np.array
            Land surface temperature
        var: np.array
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
        lst: np.array
            Land surface temperature
        var: np.array
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


class RegressionEdge(Edge, ABC):
    """Class for polynomial edge"""

    model_config = ConfigDict(allow_inf_nan=True, ser_json_inf_nan="strings")
    interval_type: IntervalType
    interval_nb: int = 20
    interval_size: float = 0.05
    percentile: tuple[int, int]
    selection: SelectionMethod = SelectionMethod.MEDIAN

    @field_validator("percentile")
    @classmethod
    def check_percentile(cls, p: tuple[int, int]) -> tuple[int, int]:
        """
        Description
        -----------
        Check the consistency of the percentile interval

        Parameters
        ----------
        p: tuple[int,int]
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
    def check_interval_nb(cls, nb: int, info: ValidationInfo) -> int:
        """
        Description
        -----------
        If interval type is "density", check the consistency of the interval number

        Parameters
        ----------
        nb: int
            Interval number

        Returns
        -------
        interval_nb: int
            Validated interval number
        """
        if info.data.get("interval_type") == IntervalType.DENSITY:
            if (nb <= 0) or (nb > NB_INTERVAL_MAX):
                msg = f"Number of intervals must be between 1 and {NB_INTERVAL_MAX}"
                raise ValueError(msg)
        else:
            logger.warning(
                "Number of intervals is ignored for interval type SIZE"
            )
        return nb

    @field_validator("interval_size")
    @classmethod
    def check_interval_size(cls, size: float, info: ValidationInfo) -> float:
        """
        Description
        -----------
        If interval type is "size", check the consistency of the interval size

        Parameters
        ----------
        size: float
            Interval size

        Returns
        -------
        interval_size: int
            Validated interval size
        """
        if info.data.get("interval_type") == IntervalType.SIZE:
            if (size <= 0) or (size > SIZE_INTERVAL_MAX):
                msg = f"Number of intervals must be between 0 and {SIZE_INTERVAL_MAX}"
                raise ValueError(msg)
        else:
            logger.warning(
                "Size of intervals is ignored for interval type DENSITY"
            )
        return size

    def get_points(
        self, var: npt.ArrayLike, lst: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Description
        -----------
        For each interval, compute the point coordinates used for the regression:
          - the abscissa value is obtained by taking the median.
          - the ordinate value is obtained by applying the
          selection method to the percentile interval.

        Parameters
        ----------
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)

        Returns
        -------
        lst_values: np.array
            LST coordinates
        var_values: np.array
            Variable coordinates
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
                (
                    group["lst"]
                    >= np.percentile(group["lst"], self.percentile[0])
                )
                & (
                    group["lst"]
                    <= np.percentile(group["lst"], self.percentile[1])
                )
            ].agg(self.selection.value)
            if not np.isnan(value):
                var_values.append(group["var"].median())
                lst_values.append(value)
        return np.array(var_values), np.array(lst_values)

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
            interval_nb = int(
                np.ceil((value_max - value_min) / self.interval_size)
            )
            intervals = ((values - value_min) / self.interval_size).astype(int)
            intervals.loc[intervals >= interval_nb] = interval_nb - 1

        return intervals

    @abstractmethod
    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
        """
        Description
        -----------
        Method to compute the edge parameters

        Parameters
        ----------
        lst: np.array_like
            Land surface temperature
        var: np.array_like
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
        var: np.array_like
            Variable

        Returns
        -------
        edge: np.array
            Temperature at the edge
        """


class LinearEdge(RegressionEdge):
    """Class for linear edge"""

    coeffs: tuple[float, float] = (float("inf"), float("inf"))

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value

        Parameters
        ----------
        var: np.array_like
            Variable

        Returns
        -------
        edge: np.array
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
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        # Get points for linear regression
        var_values, lst_values = self.get_points(var, lst)
        # Linear regression
        self.coeffs = tuple(np.polyfit(var_values, lst_values, 1))

    def __repr__(self) -> str:
        """
        String conversion method
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )

        return (
            f"LinearEdge("
            f"position={self.position.value},"
            f"interval_type={self.interval_type.value},"
            f"{interval_prop},"
            f"percentile={self.percentile},"
            f"selection={self.selection.value},"
            f"coeffs={self.coeffs})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )
        return (
            f"LinearEdge:\n"
            f"  - position={self.position.value}\n"
            f"  - interval_type={self.interval_type.value}\n"
            f"  - {interval_prop}\n"
            f"  - percentile={self.percentile}\n"
            f"  - selection={self.selection.value}\n"
            f"  - coeffs={self.coeffs}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return {
            "position": self.position.value,
            "interval_type": self.interval_type.value,
            (
                "interval_nb"
                if self.interval_type.name == IntervalType.DENSITY.name
                else "interval_size"
            ): (
                self.interval_nb
                if self.interval_type.name == IntervalType.DENSITY.name
                else self.interval_size
            ),
            "percentile": self.percentile,
            "selection": self.selection.value,
            "coeffs": tuple(float(coeff) for coeff in self.coeffs),
        }


class ThresholdLinearEdge(RegressionEdge):
    """Class for linear edge with a threshold"""

    coeffs: tuple[float, float] = (float("inf"), float("inf"))
    threshold: float = float("inf")

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value

        Parameters
        ----------
        var: np.array_like
            Variable

        Returns
        -------
        edge: np.array
            Temperature at the edge
        """
        return self.coeffs[0] * np.array(var) + self.coeffs[1]

    def fit_numpy(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
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
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        # Get points for linear regression
        var_values, lst_values = self.get_points(var, lst)
        # Remove points below the threshold
        if self.position.name == EdgePosition.TOP.name:
            cut = np.nanargmax(lst_values[::-1])
        else:
            cut = np.nanargmin(lst_values[::-1])
        cut = len(lst_values) - cut - 1
        self.threshold = var_values[cut]
        if cut == len(lst_values) - 1:
            logger.warning("ThresholdLinearEdge: Threshold not found")
            # Linear regression
            self.coeffs = tuple(np.polyfit(var_values, lst_values, 1))
        # Linear regression
        self.coeffs = tuple(np.polyfit(var_values[cut:], lst_values[cut:], 1))

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
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        # Get points for linear regression
        var_values, lst_values = self.get_points(var, lst)
        if self.position.name == EdgePosition.TOP.name:
            cut = np.nanargmax(lst_values[::-1])
        else:
            cut = np.nanargmin(lst_values[::-1])
        cut = len(lst_values) - cut - 1
        guess = var_values[cut]
        # Initialize piecewise linear fit
        # Seed is fixed to garantee reproductible results
        pwlf_solver = pwlf.PiecewiseLinFit(
            var_values, lst_values, degree=1, seed=123
        )
        # fit the data for 2 line segments
        breaks = pwlf_solver.fit_guess([guess])
        self.threshold = breaks[1]
        self.coeffs = (
            pwlf_solver.beta[2] + pwlf_solver.beta[1],
            pwlf_solver.beta[0]
            - (
                pwlf_solver.beta[1] * pwlf_solver.fit_breaks[0]
                + pwlf_solver.beta[2] * pwlf_solver.fit_breaks[1]
            ),
        )

    def __repr__(self) -> str:
        """
        String conversion method
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )

        return (
            "ThresholdLinearEdge("
            f"position={self.position.value},"
            f"interval_type={self.interval_type.value},"
            f"{interval_prop},"
            f"percentile={self.percentile},"
            f"selection={self.selection.value},"
            f"coeffs={self.coeffs},"
            f"threshold={self.threshold}"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )
        return (
            f"ThresholdLinearEdge:\n"
            f"  - position={self.position.value}\n"
            f"  - interval_type={self.interval_type.value}\n"
            f"  - {interval_prop}\n"
            f"  - percentile={self.percentile}\n"
            f"  - selection={self.selection.value}\n"
            f"  - coeffs={self.coeffs}\n"
            f"  - threshold={self.threshold}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return {
            "position": self.position.value,
            "interval_type": self.interval_type.value,
            (
                "interval_nb"
                if self.interval_type.name == IntervalType.DENSITY.name
                else "interval_size"
            ): (
                self.interval_nb
                if self.interval_type.name == IntervalType.DENSITY.name
                else self.interval_size
            ),
            "percentile": self.percentile,
            "selection": self.selection.value,
            "coeffs": tuple(float(coeff) for coeff in self.coeffs),
            "threshold": float(self.threshold),
        }


class DoubleLinearEdge(RegressionEdge):
    """Class for double linear edge"""

    coeffs1: tuple[float, float] = (float("inf"), float("inf"))
    coeffs2: tuple[float, float] = (float("inf"), float("inf"))
    inflection: float = float("inf")

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value

        Parameters
        ----------
        var: np.array_like
            Variable

        Returns
        -------
        edge: np.array
            Temperature at the edge
        """
        x = np.array(var)
        return np.piecewise(
            x,
            [x < self.inflection, x >= self.inflection],
            [
                lambda x: self.coeffs1[0] * x + self.coeffs1[1],
                lambda x: self.coeffs2[0] * x + self.coeffs2[1],
            ],
        )

    def fit_numpy(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
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
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        # Get points for linear regression
        var_values, lst_values = self.get_points(var, lst)
        # Remove points below the threshold
        if self.position.name == EdgePosition.TOP.name:
            cut = np.nanargmax(lst_values[::-1])
        else:
            cut = np.nanargmin(lst_values[::-1])
        cut = len(lst_values) - cut - 1
        self.inflection = var_values[cut]
        if cut == len(lst_values) - 1:
            logger.warning("DoubleLinearEdge: second regression impossible")
            self.coeffs1 = tuple(np.polyfit(var_values, lst_values, 1))
            self.coeffs2 = self.coeffs1
        elif cut == 0:
            logger.warning("DoubleLinearEdge: first regression impossible")
            self.coeffs2 = tuple(np.polyfit(var_values, lst_values, 1))
            self.coeffs1 = self.coeffs2
        # Linear regression
        self.coeffs1 = tuple(np.polyfit(var_values[:cut], lst_values[:cut], 1))
        self.coeffs2 = tuple(np.polyfit(var_values[cut:], lst_values[cut:], 1))

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
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        # Get points for linear regression
        var_values, lst_values = self.get_points(var, lst)
        # Initialize piecewise linear fit
        pwlf_solver = pwlf.PiecewiseLinFit(
            var_values, lst_values, degree=1, seed=123
        )
        # fit the data for 2 line segments
        pwlf_solver.fit(2)
        self.inflection = pwlf_solver.fit_breaks[1]
        self.coeffs1 = (
            pwlf_solver.beta[1],
            +pwlf_solver.beta[0]
            - pwlf_solver.beta[1] * pwlf_solver.fit_breaks[0],
        )
        self.coeffs2 = (
            pwlf_solver.beta[2] + pwlf_solver.beta[1],
            pwlf_solver.beta[0]
            - (
                pwlf_solver.beta[1] * pwlf_solver.fit_breaks[0]
                + pwlf_solver.beta[2] * pwlf_solver.fit_breaks[1]
            ),
        )

    def __repr__(self) -> str:
        """
        String conversion method
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )

        return (
            f"DoubleLinearEdge("
            f"position={self.position.value},"
            f"interval_type={self.interval_type.value},"
            f"{interval_prop},"
            f"percentile={self.percentile},"
            f"selection={self.selection.value},"
            f"coeffs1={self.coeffs1},"
            f"coeffs2={self.coeffs2},"
            f"inflection={self.inflection})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )
        return (
            f"DoubleLinearEdge:\n"
            f"  - position={self.position.value}\n"
            f"  - interval_type={self.interval_type.value}\n"
            f"  - {interval_prop}\n"
            f"  - percentile={self.percentile}\n"
            f"  - selection={self.selection.value}\n"
            f"  - coeffs1={self.coeffs1}\n"
            f"  - coeffs2={self.coeffs2}\n"
            f"  - inflection={self.inflection}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return {
            "position": self.position.value,
            "interval_type": self.interval_type.value,
            (
                "interval_nb"
                if self.interval_type == IntervalType.DENSITY
                else "interval_size"
            ): (
                self.interval_nb
                if self.interval_type == IntervalType.DENSITY
                else self.interval_size
            ),
            "percentile": self.percentile,
            "selection": self.selection.value,
            "coeffs1": tuple(float(coeff) for coeff in self.coeffs1),
            "coeffs2": tuple(float(coeff) for coeff in self.coeffs2),
            "inflection": float(self.inflection),
        }


class FlatLinearEdge(RegressionEdge):
    """Class for flat linear edge"""

    coeffs1: float = float("inf")
    coeffs2: tuple[float, float] = (float("inf"), float("inf"))
    inflection: float = float("inf")

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value

        Parameters
        ----------
        var: np.array_like
            Variable

        Returns
        -------
        edge: np.array
            Temperature at the edge
        """
        x = np.array(var)
        return np.piecewise(
            x,
            [x < self.inflection, x >= self.inflection],
            [
                self.coeffs1,
                lambda x: self.coeffs2[0] * x + self.coeffs2[1],
            ],
        )

    def fit_numpy(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
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
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        # Get points for linear regression
        var_values, lst_values = self.get_points(var, lst)
        # Remove points below the threshold
        if self.position.name == EdgePosition.TOP.name:
            cut = np.nanargmax(lst_values[::-1])
        else:
            cut = np.nanargmin(lst_values[::-1])
        cut = len(lst_values) - cut - 1
        self.inflection = var_values[cut]
        if cut == len(lst_values) - 1:
            logger.warning("FlatLinearEdge: second regression impossible")
            self.coeffs1 = lst_values[cut]
            self.coeffs2 = (0.0, self.coeffs1)
        elif cut == 0:
            logger.warning("FlatLinearEdge: first regression impossible")
            self.coeffs2 = tuple(np.polyfit(var_values, lst_values, 1))
            self.coeffs1 = self.coeffs2[0] * var_values[0] + self.coeffs2[1]
        else:
            # Linear regression
            self.coeffs2 = tuple(
                np.polyfit(var_values[cut:], lst_values[cut:], 1)
            )
            self.coeffs1 = self.coeffs2[0] * var_values[cut] + self.coeffs2[1]

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
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        # Get points for linear regression
        var_values, lst_values = self.get_points(var, lst)
        if self.position.name == EdgePosition.TOP.name:
            cut = np.nanargmax(lst_values[::-1])
        else:
            cut = np.nanargmin(lst_values[::-1])
        cut = len(lst_values) - cut - 1
        guess = var_values[cut]
        # Initialize piecewise linear fit
        pwlf_solver = pwlf.PiecewiseLinFit(
            var_values,
            lst_values,
            seed=123,
        )
        breaks = pwlf_solver.fit_guess([guess])
        # fit the data for 2 line segments
        self.inflection = breaks[1]
        self.coeffs1 = (
            pwlf_solver.beta[1] * self.inflection
            + pwlf_solver.beta[0]
            - pwlf_solver.beta[1] * pwlf_solver.fit_breaks[0]
        )
        self.coeffs2 = (
            pwlf_solver.beta[2] + pwlf_solver.beta[1],
            pwlf_solver.beta[0]
            - (
                pwlf_solver.beta[1] * pwlf_solver.fit_breaks[0]
                + pwlf_solver.beta[2] * pwlf_solver.fit_breaks[1]
            ),
        )

    def __repr__(self) -> str:
        """
        String conversion method
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )

        return (
            f"FlatLinearEdge("
            f"position={self.position.value},"
            f"interval_type={self.interval_type.value},"
            f"{interval_prop},"
            f"percentile={self.percentile},"
            f"selection={self.selection.value},"
            f"coeffs1={self.coeffs1},"
            f"coeffs2={self.coeffs2},"
            f"inflection={self.inflection})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )
        return (
            f"FlatLinearEdge\n"
            f"  - position={self.position.value}\n"
            f"  - interval_type={self.interval_type.value}\n"
            f"  - {interval_prop}\n"
            f"  - percentile={self.percentile}\n"
            f"  - selection={self.selection.value}\n"
            f"  - coeffs1={self.coeffs1}\n"
            f"  - coeffs2={self.coeffs2}\n"
            f"  - inflection={self.inflection}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return {
            "position": self.position.value,
            "interval_type": self.interval_type.value,
            (
                "interval_nb"
                if self.interval_type == IntervalType.DENSITY
                else "interval_size"
            ): (
                self.interval_nb
                if self.interval_type == IntervalType.DENSITY
                else self.interval_size
            ),
            "percentile": self.percentile,
            "selection": self.selection.value,
            "coeffs1": float(self.coeffs1),
            "coeffs2": tuple(float(coeff) for coeff in self.coeffs2),
            "inflection": float(self.inflection),
        }


class ParabolicEdge(RegressionEdge):
    """Class for parabolic edge"""

    coeffs: tuple[float, float, float] = (
        float("inf"),
        float("inf"),
        float("inf"),
    )

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
        Description
        -----------
        Compute edge value

        Parameters
        ----------
        var: np.array_like
            Variable

        Returns
        -------
        edge: np.array
            Temperature at the edge
        """
        return (
            self.coeffs[0] * np.array(var) * np.array(var)
            + self.coeffs[1] * np.array(var)
            + self.coeffs[2]
        )

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
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        var_values, lst_values = self.get_points(var, lst)
        # Linear regression
        self.coeffs = tuple(np.polyfit(var_values, lst_values, 2))

    def __repr__(self) -> str:
        """
        String conversion method
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )

        return (
            f"ParabolicEdge("
            f"position={self.position.value},"
            f"interval_type={self.interval_type.value},"
            f"{interval_prop},"
            f"percentile={self.percentile},"
            f"selection={self.selection.value},"
            f"coeffs={self.coeffs})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        interval_prop = (
            f"interval_nb={self.interval_nb}"
            if self.interval_type == IntervalType.DENSITY
            else f"interval_size={self.interval_size}"
        )
        return (
            f"ParabolicEdge\n"
            f"  - position={self.position.value}\n"
            f"  - interval_type={self.interval_type.value}\n"
            f"  - {interval_prop}\n"
            f"  - percentile={self.percentile}\n"
            f"  - selection={self.selection.value}\n"
            f"  - coeffs={self.coeffs}\n"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return {
            "position": self.position.value,
            "interval_type": self.interval_type.value,
            (
                "interval_nb"
                if self.interval_type == IntervalType.DENSITY
                else "interval_size"
            ): (
                self.interval_nb
                if self.interval_type == IntervalType.DENSITY
                else self.interval_size
            ),
            "percentile": self.percentile,
            "selection": self.selection.value,
            "coeffs": tuple(float(coeff) for coeff in self.coeffs),
        }


class FlatEdge(Edge):
    """Class for flat edge"""

    model_config = ConfigDict(allow_inf_nan=True, ser_json_inf_nan="strings")
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
        if self.position.name == EdgePosition.TOP.name:
            self.value = np.nanmax(np.array(lst))
        elif self.position.name == EdgePosition.BOTTOM.name:
            self.value = np.nanmin(np.array(lst))

    def __repr__(self) -> str:
        """
        String conversion method
        """
        return (
            f"FlatEdge(position: {self.position.value},"
            f"value: {float(self.value)})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        return (
            f"FlatEdge\n"
            f"  - position={self.position.value}\n"
            f"  - value={self.value}\n"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return {
            "position": self.position.value,
            "value": float(self.value),
        }
