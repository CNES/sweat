# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing edges classes
"""

from __future__ import annotations

import json
import re
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Self

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
    model_validator,
)

from sweat.logging import LoggerManager

logger = LoggerManager.get_logger(__name__)

PERCENTILE_MIN = 0
PERCENTILE_MAX = 100
NB_INTERVAL_MAX = 1000
NB_INTERVAL_MIN = 2
SIZE_INTERVAL_MAX = 0.5


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


class PercentileValue:
    """
    Class to manage percentile
    """

    def __init__(self, percentile: float):
        if not (0 <= percentile <= 100):
            msg = "Percentile must be between 0 and 100"
            raise ValueError(msg)
        self.percentile = percentile

    def __repr__(self):
        return f"percentile({self.percentile})"

    def __gt__(self, other: PercentileValue) -> bool:
        return self.percentile > other.percentile

    def __ge__(self, other: PercentileValue) -> bool:
        return self.percentile >= other.percentile


def compute_variable_percentile(
    n: int, n_sparse: int, q_sparse: float, n_dense: int, q_dense: float
) -> float:
    """
    Compute a percentile q between q_sparse and q_dense
    as n goes from n_sparse to n_dense linearly in log-space

    Parameters
    ----------
    n: int
        Number of points
    n_sparse: int
        Number of points to consider a sparse interval
    q_sparse: float
        Percentile for sparse intervals (few points)
    q_dense: float
        Percentile for dense intervals (lot of points)
    n_dense: int
        Number of points to consider a dense interval

    Returns
    -------
    q: float
        Computed percentile
    """
    if n <= n_sparse:
        q = q_sparse
    elif n >= n_dense:
        q = q_dense
    else:
        log_q_sparse = np.log10(q_sparse)
        log_q_dense = np.log10(q_dense)
        log_n_sparse = np.log10(n_sparse)
        log_n_dense = np.log10(n_dense)
        log_q = log_q_sparse + (log_q_dense - log_q_sparse) * (
            np.log10(n) - log_n_sparse
        ) / (log_n_dense - log_n_sparse)
        q = np.power(10, log_q)
    return q


class Edge(BaseModel, ABC):
    """Abstract class for edge"""

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=True, ser_json_inf_nan="strings"
    )
    position: EdgePosition

    @abstractmethod
    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
        """
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
        except AttributeError as e:
            msg = f"Class {name} is not defined"
            raise EdgeError(msg) from e
        except ValidationError as e:
            msg = "Error in edge configuration"
            raise EdgeError(msg) from e


class RegressionEdge(Edge, ABC):
    """Class for polynomial edge"""

    model_config = ConfigDict(
        allow_inf_nan=True,
        ser_json_inf_nan="strings",
        arbitrary_types_allowed=True,
    )
    # Interval type either fixed size "size" or fixed density "density"
    interval_type: IntervalType
    # Number of intervals
    interval_nb: int = 10
    # Interval size
    interval_size: float = 0.1
    # Limits for interval to consider for point selection
    interval_limits: (
        tuple[float, float] | tuple[PercentileValue, PercentileValue] | None
    ) = None
    # Percentile to considered for point selection
    percentile: float | None = None
    # Maximum number of points to be considered in the percentile interval
    percentile_limit: int | None = None
    # Percentiles used for sparse and dense intervals (logarithmic
    # regression to compute percentile used between bounds)
    percentile_bounds: tuple[float, float] | None = None
    # Number of points to consider a sparse interval and dense intervals
    percentile_intervals: tuple[int, int] | None = None
    # Number of points to considered for point selection
    nb_points: int | None = None
    # Regression point selection criteria (*median*,*mean*,*max*,*min*)
    selection: SelectionMethod = SelectionMethod.MEDIAN

    @model_validator(mode="before")
    @classmethod
    def check_options(cls, data: Any) -> Any:
        """
        Check options
        """
        if isinstance(data, dict):
            # Check option for points selection
            if (
                ("percentile" in data and "nb_points" in data)
                or ("percentile_bounds" in data and "nb_points" in data)
                or ("percentile_bounds" in data and "percentile" in data)
                or (
                    "percentile" not in data
                    and "nb_points" not in data
                    and "percentile_bounds" not in data
                )
            ):
                msg = (
                    "Point selection must be either using a percentile or a "
                    "variable percentile (percentile_bounds) or a number"
                    " of points"
                )
                raise ValueError(msg)
            if (
                "percentile_bounds" in data
                and "percentile_intervals" not in data
            ) or (
                "percentile_bounds" not in data
                and "percentile_intervals" in data
            ):
                msg = (
                    "In order to use variable percentiles, percentile_bounds"
                    " and percentile_intervals must be set."
                )
                raise ValueError(msg)
            # Check option for interval creation
            if "interval_nb" in data and "interval_size" in data:
                msg = (
                    "Interval selection must be either using a size or"
                    " a number of intervals"
                )
                raise ValueError(msg)
        return data

    @field_validator("percentile")
    @classmethod
    def check_percentile(
        cls,
        p: float,
    ) -> float:
        """
        Check the consistency of the percentile value

        Parameters
        ----------
        p: float
            Percentile value

        Returns
        -------
        percentile: float
            Validated percentile value
        """
        if (p <= PERCENTILE_MIN) or (p >= PERCENTILE_MAX):
            msg = "Percentile must be between ]0,100["
            raise ValueError(msg)
        return p

    @field_validator("percentile_limit")
    @classmethod
    def check_percentile_limit(cls, limit: int | None) -> int | None:
        """
        Check the percentile limit (number of points to keep)

        Parameters
        ----------
        limit: int
            Percentile limit

        Returns
        -------
        check_limit: int
            Validated percentile limit
        """
        if limit is not None and limit <= 0:
            msg = "Percentile limit must be greater than 0"
            raise ValueError(msg)
        return limit

    @field_validator("percentile_bounds")
    @classmethod
    def check_percentile_bounds(
        cls,
        p: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Check percentile boundaries used for logarithmic
        interpolation of percentiles.

        Parameters
        ----------
        p: tuple[float, float]
            Variable percentile bounds

        Returns
        -------
        check_p: tuple[float, float]
            Validated variable percentile bounds
        """
        if p is not None and (
            (p[0] <= PERCENTILE_MIN)
            or (p[0] >= PERCENTILE_MAX)
            or (p[1] <= PERCENTILE_MIN)
            or (p[1] >= PERCENTILE_MAX)
        ):
            msg = "Percentile bounds must have values between ]0,100["
            raise ValueError(msg)
        if p is not None and (p[0] < p[1]):
            msg = (
                "Percentile right bound must be "
                "lower than percentile left bound"
            )
            raise ValueError(msg)
        return p

    @field_validator("percentile_intervals", mode="after")
    @classmethod
    def check_percentile_intervals(cls, nb: tuple[int, int]) -> tuple[int, int]:
        """
        Check the number of points in intervals to identify
        sparse and dense intervals

        Parameters
        ----------
        nb: tuple[int,int]
            Number of points

        Returns
        -------
        check_nb: int
            Validated number of points
        """
        if nb is not None and ((nb[0] > nb[1]) or (nb[0] < 0) or (nb[0] < 0)):
            msg = "percentile_intervals must be an interval strictly positive."
            raise ValueError(msg)
        return nb

    @classmethod
    def _convert(cls, v: Any) -> float | PercentileValue:
        """
        Convert to float or percentile
        """
        if isinstance(v, float | int):
            return float(v)
        if isinstance(v, PercentileValue):
            return v
        if isinstance(v, str):
            match = re.match(r"percentile\((\d+(\.\d+)?)\)", v.strip())
            if match:
                percentile = float(match.group(1))
                return PercentileValue(percentile)
        msg = (
            "value must be a float or a percentile or a "
            "string in percentile format like 'percentile(90)'"
        )
        raise ValueError(msg)

    @field_validator("interval_limits", mode="before")
    @classmethod
    def check_interval_limits(
        cls,
        p: tuple[Any, Any],
    ) -> tuple[float, float] | tuple[PercentileValue, PercentileValue]:
        """
        Check the consistency of interval limits

        Parameters
        ----------
        p: tuple[float,float]
            Interval limits

        Returns
        -------
        limits: tuple[float,float]
            Validated interval limits
        """
        try:
            limits = tuple(cls._convert(v) for v in p)
            if type(limits[0]) is not type(limits[1]):
                msg = "Mismatch in type for interval limits"
                raise TypeError(msg)  # noqa: TRY301
            if limits[0] >= limits[1]:  # type: ignore
                msg = "Lower limit greater than upper limit"
                raise ValueError(msg)  # noqa: TRY301
        except (ValueError, TypeError) as e:
            msg = f"Error in interval limits: {e}"
            raise ValueError(msg) from e
        else:
            return limits  # type: ignore

    @field_validator("nb_points")
    @classmethod
    def check_nb_points(cls, nb: int) -> int:
        """
        Check the number of points to keep

        Parameters
        ----------
        nb: int
            Number of points

        Returns
        -------
        check_nb: int
            Number of points
        """
        if nb <= 0:
            msg = "Number of points nb_points must be greater than 0"
            raise ValueError(msg)
        return nb

    @field_validator("interval_nb")
    @classmethod
    def check_interval_nb(cls, nb: int) -> int:
        """
        Check the consistency of the interval number,
        if interval type is "density" or "interval_size".

        Parameters
        ----------
        nb: int
            Interval number

        Returns
        -------
        interval_nb: int
            Validated interval number
        """
        if (nb < NB_INTERVAL_MIN) or (nb > NB_INTERVAL_MAX):
            msg = (
                "Number of intervals must be between "
                f"{NB_INTERVAL_MIN} and {NB_INTERVAL_MAX}"
            )
            raise ValueError(msg)
        return nb

    @field_validator("interval_size")
    @classmethod
    def check_interval_size(cls, size: float, info: ValidationInfo) -> float:
        """
        Check the consistency of the interval size, if interval type is "size".

        Parameters
        ----------
        size: float
            Interval size
        info: ValidationInfo
            Information

        Returns
        -------
        interval_size: int
            Validated interval size
        """
        if info.data.get("interval_type") == IntervalType.SIZE:
            if (size <= 0) or (size > SIZE_INTERVAL_MAX):
                msg = (
                    "Number of intervals must be between "
                    f"0 and {SIZE_INTERVAL_MAX}"
                )
                raise ValueError(msg)
        else:
            msg = "Size of intervals does not work for interval type DENSITY"
            raise ValueError(msg)
        return size

    def _prepare(self, var: npt.NDArray, lst: npt.NDArray) -> pd.DataFrame:
        """
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
        # Filter interval limits
        df = pd.DataFrame(data={"lst": lst.reshape(-1), "var": var.reshape(-1)})
        if self.interval_limits is not None:
            value_min = self.interval_limits[0]
            if isinstance(value_min, PercentileValue):
                value_min = df["var"].quantile(value_min.percentile / 100)
            value_max = self.interval_limits[1]
            if isinstance(value_max, PercentileValue):
                value_max = df["var"].quantile(value_max.percentile / 100)
            df.loc[
                (df["var"] < value_min) | (df["var"] > value_max),
                "var",
            ] = np.nan
        return (
            df.dropna(axis=0, how="any")
            .sort_values(by="var")
            .reset_index(drop=True)
        )

    def _select(self, df: pd.Series) -> float:
        """
        Point selection in an interval
        """
        if self.percentile_limit is not None:
            if self.position.name == EdgePosition.TOP.name:
                return df.nlargest(self.percentile_limit, keep="all").agg(
                    self.selection.value
                )
            return df.nsmallest(self.percentile_limit, keep="all").agg(
                self.selection.value
            )
        return df.agg(self.selection.value)

    def _select_percentile(
        self,
        df: pd.DataFrame,
        intervals: pd.Series,
        percentile: float,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Select a point in the interval using percentiles

        Parameters
        ----------
        df: pd.DataFrame
           Data containing lst as a function of var
        intervals: pd.Series
           List of intervals
        percentile: float
           Percentile value

        Returns
        -------
        lst_values: np.array
            LST coordinates
        var_values: np.array
            Variable coordinates
        """
        var_values = []
        lst_values = []
        # Compute percentile limits
        if self.position == EdgePosition.TOP:
            min_percentile = 100.0 - percentile
            max_percentile = 100.0
        else:
            min_percentile = 0.0
            max_percentile = percentile
        # Selection with percentile
        for _, group in df.groupby(intervals):
            value = group["lst"][
                (group["lst"] >= np.percentile(group["lst"], min_percentile))
                & (group["lst"] <= np.percentile(group["lst"], max_percentile))
            ].pipe(self._select)
            if not np.isnan(value):
                var_values.append(group["var"].median())
                lst_values.append(value)
        return np.array(var_values), np.array(lst_values)

    def _select_variable_percentile(
        self,
        df: pd.DataFrame,
        intervals: pd.Series,
        sparse_percentile: float,
        dense_percentile: float,
        sparse_interval: int,
        dense_interval: int,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Select a point in the interval using percentiles

        Parameters
        ----------
        df: pd.DataFrame
           Data containing lst as a function of var
        intervals: pd.Series
           List of intervals
        sparse_percentile: float
           Percentile for sparse intervals (few points)
        dense_percentile: float
           Percentile for dense intervals (lot of points)
        sparse_interval: int
           Number of points to consider a sparse interval
        dense_interval: int
           Number of points to consider a dense interval

        Returns
        -------
        lst_values: np.array
            LST coordinates
        var_values: np.array
            Variable coordinates
        """
        var_values = []
        lst_values = []
        # Selection with variables percentiles
        for _, group in df.groupby(intervals):
            nb = len(group)
            if nb > 0:
                # Logarithmic interpolation for percentiles
                percentile = compute_variable_percentile(
                    n=nb,
                    n_sparse=sparse_interval,
                    q_sparse=sparse_percentile,
                    n_dense=dense_interval,
                    q_dense=dense_percentile,
                )
                if self.position == EdgePosition.TOP:
                    min_percentile = 100.0 - percentile
                    max_percentile = 100.0
                else:
                    min_percentile = 0.0
                    max_percentile = percentile
                value = group["lst"][
                    (
                        group["lst"]
                        >= np.percentile(group["lst"], min_percentile)
                    )
                    & (
                        group["lst"]
                        <= np.percentile(group["lst"], max_percentile)
                    )
                ].pipe(self._select)
                if not np.isnan(value):
                    var_values.append(group["var"].median())
                    lst_values.append(value)
        return np.array(var_values), np.array(lst_values)

    def _select_nb_points(
        self,
        df: pd.DataFrame,
        intervals: pd.Series,
        nb_points: int,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Select a point in the interval using a number of points

        Parameters
        ----------
        df: pd.DataFrame
           Data containing lst as a function of var
        intervals: pd.Series
           List of intervals
        nb_points: int
           Number of points

        Returns
        -------
        lst_values: np.array
            LST coordinates
        var_values: np.array
            Variable coordinates
        """
        var_values = []
        lst_values = []
        # Selection with number of points
        for _, group in df.groupby(intervals):
            if self.position.name == EdgePosition.TOP.name:
                value = (
                    group["lst"]
                    .nlargest(nb_points, keep="first")
                    .agg(self.selection.value)
                )
            else:
                value = (
                    group["lst"]
                    .nsmallest(nb_points, keep="first")
                    .agg(self.selection.value)
                )
            if not np.isnan(value):
                var_values.append(group["var"].median())
                lst_values.append(value)
        return np.array(var_values), np.array(lst_values)

    def get_points(
        self, var: npt.ArrayLike, lst: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        For each interval, compute the point coordinates
        used for the regression:
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
        df = self._prepare(np.array(var), np.array(lst))
        intervals = self._get_intervals(df["var"])

        # Compute point coordinates for regression
        if self.percentile is not None:
            # Selection with percentile
            return self._select_percentile(
                df=df,
                intervals=intervals,
                percentile=self.percentile,
            )
        if (
            self.percentile_bounds is not None
            and self.percentile_intervals is not None
        ):
            # Selection with percentile
            return self._select_variable_percentile(
                df=df,
                intervals=intervals,
                sparse_percentile=self.percentile_bounds[0],
                dense_percentile=self.percentile_bounds[1],
                sparse_interval=self.percentile_intervals[0],
                dense_interval=self.percentile_intervals[1],
            )
        if self.nb_points is not None:
            # Selection with number of points
            return self._select_nb_points(
                df=df,
                intervals=intervals,
                nb_points=self.nb_points,
            )
        # Return empty arrays
        return np.array([]), np.array([])

    def _get_intervals(self, values: pd.Series) -> pd.Series:
        """
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
        elif (
            self.interval_type == IntervalType.SIZE
            and "interval_nb" in self.model_fields_set
        ):
            # Intervals with the same number of elements
            value_min = values.min()
            value_max = values.max()
            interval_size = (value_max - value_min) / self.interval_nb
            intervals = ((values - value_min) / interval_size).astype(int)
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

    def search_extremum_point(
        self,
        var: npt.NDArray,
        lst: npt.NDArray,
    ) -> tuple[float, float]:
        """
        Return extrema of selected points

        Parameters
        ----------
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)

        Returns
        -------
        break: tuple[float,float]
            Coordinates of extremum point
        """
        if self.position.name == EdgePosition.TOP.name:
            extremum_index = np.nanargmax(lst[::-1])
        else:
            extremum_index = np.nanargmin(lst[::-1])
        extremum_index = len(lst) - extremum_index - 1
        return var[extremum_index], lst[extremum_index]

    @abstractmethod
    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
        """
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

    def _common_repr(self) -> str:
        """
        String conversion method
        """
        interval_prop = (
            f"interval_nb={self.interval_nb},"
            if self.interval_type == IntervalType.DENSITY
            or (
                self.interval_type == IntervalType.SIZE
                and "interval_nb" in self.model_fields_set
            )
            else f"interval_size={self.interval_size},"
        )
        percentile_prop = (
            (
                f"percentile = {self.percentile},"
                f" percentile_limit = {self.percentile_limit}"
            )
            if (
                self.percentile is not None
                and self.percentile_limit is not None
            )
            else f"percentile = {self.percentile},"
            if self.percentile is not None
            else (
                f"percentile_bounds = {self.percentile_bounds},"
                f"percentile_intervals = {self.percentile_intervals},"
            )
            if self.percentile_bounds is not None
            else f"nb_points = {self.nb_points},"
        )

        return (
            f"position={self.position.value},"
            f"interval_type={self.interval_type.value},"
            f"{interval_prop}"
            f"{percentile_prop}"
            f"selection={self.selection.value},"
        )

    def _common_str(self) -> str:
        """
        String conversion method for end-users
        """
        interval_prop = (
            f"  - interval_nb={self.interval_nb}\n"
            if self.interval_type == IntervalType.DENSITY
            or (
                self.interval_type == IntervalType.SIZE
                and "interval_nb" in self.model_fields_set
            )
            else f"  - interval_size={self.interval_size}\n"
        )
        percentile_prop = (
            (
                f"  - percentile = {self.percentile} "
                f"(limit = {self.percentile_limit})\n"
            )
            if (
                self.percentile is not None
                and self.percentile_limit is not None
            )
            else f"  - percentile = {self.percentile}\n"
            if self.percentile is not None
            else (
                f"  - percentile_bounds = {self.percentile_bounds}\n"
                f"  - percentile_intervals = {self.percentile_intervals}\n"
            )
            if self.percentile_bounds is not None
            else f"  - nb_points = {self.nb_points}\n"
        )
        return (
            f"  - position={self.position.value}\n"
            f"  - interval_type={self.interval_type.value}\n"
            f"{interval_prop}"
            f"{percentile_prop}"
            f"  - selection={self.selection.value}"
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
                or (
                    self.interval_type == IntervalType.SIZE
                    and "interval_nb" in self.model_fields_set
                )
                else "interval_size"
            ): (
                self.interval_nb
                if self.interval_type.name == IntervalType.DENSITY.name
                or (
                    self.interval_type == IntervalType.SIZE
                    and "interval_nb" in self.model_fields_set
                )
                else self.interval_size
            ),
            (
                "percentile"
                if self.percentile is not None
                else "percentile_bounds"
                if self.percentile_bounds is not None
                else "nb_points"
            ): (
                self.percentile
                if self.percentile is not None
                else self.percentile_bounds
                if self.percentile_bounds is not None
                else self.nb_points
            ),
            **(
                {"percentile_limit": self.percentile_limit}
                if self.percentile_limit is not None
                else {}
            ),
            **(
                {"percentile_intervals": self.percentile_intervals}
                if self.percentile_intervals is not None
                else {}
            ),
            "selection": self.selection.value,
        }


class LinearEdge(RegressionEdge):
    """Class for linear edge"""

    coeffs: tuple[float, float] = (float("inf"), float("inf"))
    # Use the breakpoint to compute the edge (to be used only with albedo).
    # The mean temperature increases when
    use_breakpoint: bool = False
    fit_breakpoint: float | None = None
    # If slope correction is enabled, the slope of a top edge
    # cannot be positive and that of a bottom edge cannot be negative.
    slope_correction: bool = False

    @model_validator(mode="after")
    def check_use_breakpoint(self) -> Self:
        """
        Check use_breakpoint option

        If fit_breakpoint option is provided,
        use_breakpoint is set to True

        Parameters
        ----------
        self: LinearEdge
            LinearEdge instance

        Returns
        -------
        self: LinearEdge
            Checked instance
        """
        if self.fit_breakpoint is not None:
            self.use_breakpoint = True
        return self

    def search_breakpoint(
        self, var: npt.ArrayLike, lst: npt.ArrayLike
    ) -> tuple[float, float]:
        """
        Search breakpoints

        Notes
        -----
        This method is intended for use with albedo.
        The temperature increases when albedo increases for low albedo values
        (not necessarily linearly), and the temperature decreases when albedo
        increases for high albedo values (linearly).
        The break point is around 0.25 and 0.3.

        Parameters
        ----------
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)

        Returns
        -------
        break: tuple[float,float]
            Coordinates of breakpoint point
        """
        var_values = []
        lst_values = []
        df = self._prepare(np.array(var), np.array(lst))
        intervals = self._get_intervals(df["var"])

        # Compute point coordinates for regression
        for _, group in df.groupby(intervals):
            value = group["lst"].nlargest(20, "first").median()
            if not np.isnan(value):
                var_values.append(group["var"].median())
                lst_values.append(value)
        break_index = np.nanargmax(lst_values[::-1])
        break_index = len(lst_values) - break_index - 1
        if break_index == len(lst_values) - 1:
            break_index = break_index - 1
        elif break_index == 0:
            break_index = break_index + 1
        return var_values[break_index], lst_values[break_index]

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
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
        For each interval, compute the point coordinates
        used for the regression:
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
        # If the option use_break_point is activated, the regression
        # occurs only on a part of the selected point: after the break point
        # for top edge and before the break point for bottom edge.
        if self.use_breakpoint:
            if self.fit_breakpoint is None:
                self.fit_breakpoint, _ = self.search_breakpoint(var, lst)
            if self.position.name == EdgePosition.TOP.name:
                tmp = var_values[var_values >= self.fit_breakpoint]
                lst_values = lst_values[var_values >= self.fit_breakpoint]
                var_values = tmp
            else:
                tmp = var_values[var_values <= self.fit_breakpoint]
                lst_values = lst_values[var_values <= self.fit_breakpoint]
                var_values = tmp
            if len(var_values) == 1:
                msg = (
                    f"LinearEdge: breakpoint ({self.fit_breakpoint}) "
                    "at the edge of the domain"
                )
                logger.warning(msg)
            if len(var_values) == 0:
                msg = (
                    f"LinearEdge: breakpoint ({self.fit_breakpoint}) "
                    "misplaced, no value for regression"
                )
                logger.error(msg)
        # Linear regression
        self.coeffs = tuple(np.polyfit(var_values, lst_values, 1))
        # If slope correction is enabled
        if self.slope_correction:
            if (
                self.position.name == EdgePosition.TOP.name
                and self.coeffs[0] > 0
            ):
                self.coeffs = (0.0, np.mean(lst_values))
            if (
                self.position.name == EdgePosition.BOTTOM.name
                and self.coeffs[0] < 0
            ):
                self.coeffs = (0.0, np.mean(lst_values))

    def __repr__(self) -> str:
        """
        String conversion method
        """
        break_prop = (
            f"fit_breakpoint={self.fit_breakpoint},"
            if self.use_breakpoint
            else ""
        )
        return (
            f"LinearEdge({self._common_repr()}{break_prop}coeffs={self.coeffs})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        break_prop = (
            f"  - fit_breakpoint={self.fit_breakpoint},\n"
            if self.use_breakpoint
            else ""
        )
        return (
            f"LinearEdge:\n{self._common_str()}\n"
            f"{break_prop}"
            f"  - coeffs={self.coeffs}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
        common_dict["use_breakpoint"] = self.use_breakpoint
        if self.use_breakpoint:
            common_dict["fit_breakpoint"] = self.fit_breakpoint
        common_dict["coeffs"] = tuple(float(coeff) for coeff in self.coeffs)
        return common_dict


class ThresholdLinearEdge(RegressionEdge):
    """Class for linear edge with a threshold"""

    coeffs: tuple[float, float] = (float("inf"), float("inf"))
    threshold: float = float("inf")
    # If use_extremum is True, use the extremum of the regression point
    # for threshold value. Otherwise, the threshold value would be guessed.
    use_extremum: bool = True

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
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
        For each interval, compute the point coordinates
        used for the regression:
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
        # Seed is fixed to ensure reproducible results
        pwlf_solver = pwlf.PiecewiseLinFit(
            var_values, lst_values, degree=1, seed=123
        )
        if self.use_extremum:
            guess, _ = self.search_extremum_point(var_values, lst_values)
            breaks = [var_values[0], guess, var_values[-1]]
            if guess == var_values[0]:
                msg = "ThresholdLinearEdge: first point selected as threshold"
                logger.warning(msg)
                breaks = [var_values[0], var_values[-1]]

            if guess == var_values[-1]:
                msg = (
                    "ThresholdLinearEdge: last point selected as "
                    "threshold, use of the penultimate point instead"
                )
                logger.warning(msg)
                breaks = [var_values[0], var_values[-2], var_values[-1]]
            # Specific case, if only 2 points
            if len(var_values) == 2:
                breaks = [var_values[0], var_values[-1]]
            pwlf_solver.fit_with_breaks(breaks)
            self.threshold = breaks[-2]
        else:
            # fit the data for 2 line segments
            pwlf_solver.fit(2)
            self.threshold = pwlf_solver.fit_breaks[1]
        pwlf_solver.calc_slopes()
        self.coeffs = (pwlf_solver.slopes[-1], pwlf_solver.intercepts[-1])

    def __repr__(self) -> str:
        """
        String conversion method
        """
        return (
            "ThresholdLinearEdge("
            f"{self._common_repr()}"
            f"use_extremum={self.use_extremum},"
            f"coeffs={self.coeffs},"
            f"threshold={self.threshold})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        return (
            f"ThresholdLinearEdge:\n"
            f"{self._common_str()}\n"
            f"  - use_extremum={self.use_extremum}\n"
            f"  - coeffs={self.coeffs}\n"
            f"  - threshold={self.threshold}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
        common_dict["use_extremum"] = self.use_extremum
        common_dict["coeffs"] = tuple(float(coeff) for coeff in self.coeffs)
        common_dict["threshold"] = float(self.threshold)
        return common_dict


class DoubleLinearEdge(RegressionEdge):
    """Class for double linear edge"""

    coeffs1: tuple[float, float] = (float("inf"), float("inf"))
    coeffs2: tuple[float, float] = (float("inf"), float("inf"))
    fit_breakpoint: float = float("inf")
    # If use_extremum is True, use the extremum of the regression point
    # for threshold value. Otherwise, the threshold value would be guessed.
    use_extremum: bool = True

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
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
            [x < self.fit_breakpoint, x >= self.fit_breakpoint],
            [
                lambda x: self.coeffs1[0] * x + self.coeffs1[1],
                lambda x: self.coeffs2[0] * x + self.coeffs2[1],
            ],
        )

    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
        """
        For each interval, compute the point coordinates
        used for the regression:
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
        if self.use_extremum:
            self.fit_breakpoint, _ = self.search_extremum_point(
                var_values, lst_values
            )
            breaks = [var_values[0], self.fit_breakpoint, var_values[-1]]
            degraded_mode = False
            if self.fit_breakpoint == var_values[0]:
                msg = "DoubleLinearEdge: first point selected as breakpoint"
                logger.warning(msg)
                degraded_mode = True
            if self.fit_breakpoint == var_values[-1]:
                msg = "DoubleLinearEdge: last point selected as breakpoint"
                logger.warning(msg)
                degraded_mode = True
            # Specific case, if only 2 points
            if len(var_values) == 2:
                msg = "DoubleLinearEdge: not enough number of points"
                logger.warning(msg)
                degraded_mode = True
            if degraded_mode:
                msg = "DoubleLinearEdge: run with degraded mode"
                logger.warning(msg)
                # fit the data for 2 line segments
                pwlf_solver.fit(2)
                self.fit_breakpoint = pwlf_solver.fit_breaks[1]
                self.use_extremum = False
            else:
                pwlf_solver.fit_with_breaks(breaks)
                self.fit_breakpoint = breaks[-2]
        else:
            # fit the data for 2 line segments
            pwlf_solver.fit(2)
            self.fit_breakpoint = pwlf_solver.fit_breaks[1]
        pwlf_solver.calc_slopes()
        self.coeffs1 = (pwlf_solver.slopes[0], pwlf_solver.intercepts[0])
        self.coeffs2 = (pwlf_solver.slopes[1], pwlf_solver.intercepts[1])

    def __repr__(self) -> str:
        """
        String conversion method
        """
        return (
            f"DoubleLinearEdge("
            f"{self._common_repr()}"
            f"use_extremum={self.use_extremum},"
            f"coeffs1={self.coeffs1},"
            f"coeffs2={self.coeffs2},"
            f"fit_breakpoint={self.fit_breakpoint})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        return (
            f"DoubleLinearEdge:\n"
            f"{self._common_str()}\n"
            f"  - use_extremum={self.use_extremum}\n"
            f"  - coeffs1={self.coeffs1}\n"
            f"  - coeffs2={self.coeffs2}\n"
            f"  - fit_breakpoint={self.fit_breakpoint}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
        common_dict["use_extremum"] = self.use_extremum
        common_dict["coeffs1"] = tuple(float(coeff) for coeff in self.coeffs1)
        common_dict["coeffs2"] = tuple(float(coeff) for coeff in self.coeffs2)
        common_dict["fit_breakpoint"] = float(self.fit_breakpoint)
        return common_dict


class FlatLinearEdge(RegressionEdge):
    """Class for flat linear edge"""

    coeffs1: float = float("inf")
    coeffs2: tuple[float, float] = (float("inf"), float("inf"))
    fit_breakpoint: float = float("inf")
    # If use_extremum is True, use the extremum of the regression point
    # for threshold value. Otherwise, the threshold value would be guessed.
    use_extremum: bool = True

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
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
            [x < self.fit_breakpoint, x >= self.fit_breakpoint],
            [
                self.coeffs1,
                lambda x: self.coeffs2[0] * x + self.coeffs2[1],
            ],
        )

    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
        """
        For each interval, compute the point coordinates
        used for the regression:
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
            var_values,
            lst_values,
            seed=123,
        )
        if self.use_extremum:
            self.fit_breakpoint, _ = self.search_extremum_point(
                var_values, lst_values
            )
            breaks = [var_values[0], self.fit_breakpoint, var_values[-1]]
            if self.fit_breakpoint == var_values[0]:
                msg = "FlatLinearEdge: first point selected as breakpoint"
                logger.warning(msg)
                breaks = [var_values[0], var_values[-1]]
            if self.fit_breakpoint == var_values[-1]:
                msg = (
                    "FlatLinearEdge: last point selected as breakpoint, "
                    "use of the penultimate point instead"
                )
                logger.warning(msg)
                breaks = [var_values[0], var_values[-2], var_values[-1]]
            # Specific case, if only 2 points
            if len(var_values) == 2:
                breaks = [var_values[0], var_values[-1]]
            pwlf_solver.fit_with_breaks(breaks)
            self.fit_breakpoint = breaks[-2]
        else:
            pwlf_solver.fit(2)
            # fit the data for 2 line segments
            self.fit_breakpoint = pwlf_solver.fit_breaks[1]
        pwlf_solver.calc_slopes()
        self.coeffs1 = pwlf_solver.predict(self.fit_breakpoint)
        self.coeffs2 = (pwlf_solver.slopes[-1], pwlf_solver.intercepts[-1])

    def __repr__(self) -> str:
        """
        String conversion method
        """
        return (
            f"FlatLinearEdge("
            f"{self._common_repr()}"
            f"use_extremum={self.use_extremum},"
            f"coeffs1={self.coeffs1},"
            f"coeffs2={self.coeffs2},"
            f"fit_breakpoint={self.fit_breakpoint})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        return (
            f"FlatLinearEdge\n"
            f"{self._common_str()}\n"
            f"  - use_extremum={self.use_extremum}\n"
            f"  - coeffs1={self.coeffs1}\n"
            f"  - coeffs2={self.coeffs2}\n"
            f"  - fit_breakpoint={self.fit_breakpoint}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
        common_dict["use_extremum"] = self.use_extremum
        common_dict["coeffs1"] = float(self.coeffs1)
        common_dict["coeffs2"] = tuple(float(coeff) for coeff in self.coeffs2)
        common_dict["fit_breakpoint"] = float(self.fit_breakpoint)
        return common_dict


class ParabolicEdge(RegressionEdge):
    """Class for parabolic edge"""

    coeffs: tuple[float, float, float] = (
        float("inf"),
        float("inf"),
        float("inf"),
    )

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
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
        For each interval, compute the point coordinates
        used for the regression:
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
        return f"ParabolicEdge({self._common_repr()}coeffs={self.coeffs})"

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        return (
            f"ParabolicEdge\n"
            f"  - {self._common_str()}\n"
            f"  - coeffs={self.coeffs}\n"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
        common_dict["coeffs"] = tuple(float(coeff) for coeff in self.coeffs)
        return common_dict


class FlatRegressionEdge(RegressionEdge):
    """Class for linear edge"""

    coeff: float = float("inf")

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
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
        return self.coeff * np.ones_like(np.array(var))

    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
        """
        For each interval, compute the point coordinates
        used for the regression:
          - the abscissa value is obtained by taking the median.
          - the ordinate value is obtained by applying the
          selection method to the percentile interval.
        Edge is defined by the max or the min of selected point coordinates.

        Parameters
        ----------
        lst: np.array_like
            Land surface temperature
        var: np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        # Get points for linear regression
        _, lst_values = self.get_points(var, lst)
        # Linear regression
        if self.position.name == EdgePosition.TOP.name:
            self.coeff = np.nanmax(lst_values)
        elif self.position.name == EdgePosition.BOTTOM.name:
            self.coeff = np.nanmin(lst_values)

    def __repr__(self) -> str:
        """
        String conversion method
        """
        return f"FlatRegressionEdge({self._common_repr()}coeff={self.coeff})"

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        return (
            f"FlatRegressionEdge:\n"
            f"  - {self._common_str()}\n"
            f"  - coeff={self.coeff}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
        common_dict["coeff"] = float(self.coeff)
        return common_dict


class FlatEdge(Edge):
    """Class for flat edge"""

    model_config = ConfigDict(allow_inf_nan=True, ser_json_inf_nan="strings")
    value: float = float("inf")

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
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


class FlatPercentileEdge(Edge):
    """Class for flat edge"""

    model_config = ConfigDict(allow_inf_nan=True, ser_json_inf_nan="strings")
    # Percentile value to considered for point selection
    percentile: float | None = None
    # Maximum number of points to be considered in the percentile interval
    percentile_limit: int | None = None
    # Number of points to considered for point selection
    nb_points: int | None = None
    # Regression point selection criteria (*median*,*mean*,*max*,*min*)
    selection: SelectionMethod = SelectionMethod.MEDIAN
    value: float = float("inf")

    @model_validator(mode="before")
    @classmethod
    def check_option(cls, data: Any) -> Any:
        """
        Check option for points selection
        """
        if isinstance(data, dict) and (
            ("percentile" in data and "nb_points" in data)
            or ("percentile" not in data and "nb_points" not in data)
        ):
            msg = (
                "Point selection must be either using a percentile or "
                "a number of points"
            )
            raise ValueError(msg)
        return data

    @field_validator("percentile")
    @classmethod
    def check_percentile(cls, p: float) -> float:
        """
        Check the consistency of the percentile value

        Parameters
        ----------
        p: float
            Percentile value

        Returns
        -------
        percentile: float
            Validated percentile value
        """
        if (p <= PERCENTILE_MIN) or (p >= PERCENTILE_MAX):
            msg = "Percentile must be an interval between [0,100]"
            raise ValueError(msg)
        return p

    @field_validator("percentile_limit")
    @classmethod
    def check_percentile_limit(cls, limit: int | None) -> int | None:
        """
        Check the percentile limit (number of points to keep)

        Parameters
        ----------
        limit: int
            Percentile limit

        Returns
        -------
        check_limit: int
            Validated percentile limit
        """
        if limit is not None and limit <= 0:
            msg = "Percentile limit must be greater than 0"
            raise ValueError(msg)
        return limit

    @field_validator("nb_points")
    @classmethod
    def check_nb_points(cls, nb: int) -> int:
        """
        Check the number of points to keep

        Parameters
        ----------
        nb: int
            Number of points

        Returns
        -------
        check_nb: int
            Number of points
        """
        if nb <= 0:
            msg = "Number of points nb_points must be greater than 0"
            raise ValueError(msg)
        return nb

    def get(self, var: npt.ArrayLike) -> npt.NDArray:
        """
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

    def _select(self, df: pd.Series) -> float:
        """
        Point selection in an interval
        """
        if self.percentile_limit is not None:
            if self.position.name == EdgePosition.TOP.name:
                return df.nlargest(self.percentile_limit, keep="all").agg(
                    self.selection.value
                )
            return df.nsmallest(self.percentile_limit, keep="all").agg(
                self.selection.value
            )
        return df.agg(self.selection.value)

    def fit(self, var: npt.ArrayLike, lst: npt.ArrayLike) -> None:
        """
        Edge is defined by the max or the min of LST

        Parameters
        ----------
        lst : np.array_like
            Land surface temperature
        var : np.array_like
            Variable used versus temperature (ex: Albedo)
        """
        df = self._prepare(np.array(var), np.array(lst))
        # Selection with percentile
        if self.percentile is not None:
            # Compute percentile bounds
            if self.position == EdgePosition.TOP:
                min_percentile = 100.0 - self.percentile
                max_percentile = 100.0
            else:
                min_percentile = 0.0
                max_percentile = self.percentile
            self.value = df["lst"][
                (df["lst"] >= np.percentile(df["lst"], min_percentile))
                & (df["lst"] <= np.percentile(df["lst"], max_percentile))
            ].pipe(self._select)
        # Selection with the number of points
        elif self.nb_points is not None:
            if self.position.name == EdgePosition.TOP.name:
                self.value = (
                    df["lst"]
                    .nlargest(self.nb_points, keep="first")
                    .agg(self.selection.value)
                )
            else:
                self.value = (
                    df["lst"]
                    .nsmallest(self.nb_points, keep="first")
                    .agg(self.selection.value)
                )

    def __repr__(self) -> str:
        """
        String conversion method
        """
        percentile_prop = (
            (
                (
                    f"percentile={self.percentile}, "
                    f"percentile_limit={self.percentile_limit},"
                )
                if self.percentile_limit is not None
                else f"percentile={self.percentile},"
            )
            if self.percentile is not None
            else f"nb_points={self.nb_points}"
        )

        return (
            f"FlatPercentileEdge(position: {self.position.value},"
            f"{percentile_prop},"
            f"selection={self.selection.value},"
            f"value: {float(self.value)})"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        percentile_prop = (
            (
                (
                    f"  - percentile={self.percentile} "
                    f"(limit={self.percentile_limit}\n"
                )
                if self.percentile_limit is not None
                else f"  - percentile={self.percentile}\n"
            )
            if self.percentile is not None
            else f"  - nb_points={self.nb_points}\n"
        )
        return (
            f"FlatPercentileEdge\n"
            f"  - position={self.position.value}\n"
            f"{percentile_prop}"
            f"  - selection={self.selection.value}\n"
            f"  - value={self.value}\n"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        return {
            "position": self.position.value,
            ("percentile" if self.percentile is not None else "nb_points"): (
                self.percentile
                if self.percentile is not None
                else self.nb_points
            ),
            **(
                {"percentile_limit": self.percentile_limit}
                if self.percentile_limit is not None
                else {}
            ),
            "selection": self.selection.value,
            "value": float(self.value),
        }
