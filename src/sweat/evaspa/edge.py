# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing edges classes
"""

from __future__ import annotations

import json
import re
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pwlf
import scipy as sp
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


class PercentileValue:
    """
    Class to manange percentile
    """

    def __init__(self, percentile: float):
        if not (0 <= percentile <= 100):  # noqa: PLR2004
            msg = "Percentile must be between 0 and 100"
            raise ValueError(msg)
        self.percentile = percentile

    def __repr__(self):
        return f"percentile({self.percentile})"

    def __gt__(self, other: PercentileValue) -> bool:
        return self.percentile > other.percentile

    def __ge__(self, other: PercentileValue) -> bool:
        return self.percentile >= other.percentile


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
    # Limits for interval to consider for point selecttion
    interval_limits: (
        tuple[float, float] | tuple[PercentileValue, PercentileValue] | None
    ) = None
    # Percentile interval to considered for point selection
    percentile: tuple[float, float] | None = None
    # Maximum number of points to be considered in the percentile interval
    percentile_limit: int | None = None
    # Percentiles used for sparse and dense intervals (logarithmic
    # regression to compute percentile used between bounds)
    percentile_bounds: (
        tuple[tuple[float, float], tuple[float, float]] | None
    ) = None
    # Number of points to consider a sparse interval and dense intervals
    percentile_intervals: tuple[int, int] | None = None
    # Number of points to considered for point selection
    nb_points: int | None = None
    # Regression point selection criteria (*median*,*mean*,*max*,*min*)
    selection: SelectionMethod = SelectionMethod.MEDIAN
    # Use the breakpoint to compute the edge (to be used only with albedo).
    # The mean temperature increases when
    use_breakpoint: bool = False
    breakpoint: float | None = None

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
                    "In order to use variable percentiles, percentile_bounds and"
                    " percentile_intervals must be set."
                )
                raise ValueError(msg)
            # Check option for interval creation
            if "interval_nb" in data and "interval_size" in data:
                msg = "Interval selection must be either using a size or a number of intervals"
                raise ValueError(msg)
        return data

    @field_validator("percentile")
    @classmethod
    def check_percentile(
        cls,
        p: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Check the consistency of the percentile interval

        Parameters
        ----------
        p: tuple[float,float]
            Percentile interval

        Returns
        -------
        percentile: tuple[float,float]
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
        p: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Check percentile boundaries used for logarithmic
        interpolation of percentiles.

        Parameters
        ----------
        p: tuple[tuple[float, float], tuple[float, float]]
            Variable percentile

        Returns
        -------
        check_p: tuple[tuple[float, float], tuple[float, float]]
            Validated variable percentile
        """
        if p is not None:
            p_min = p[0]
            p_max = p[1]
            if (
                (p_min[0] > p_min[1])
                or (p_min[0] < PERCENTILE_MIN)
                or (p_min[0] >= PERCENTILE_MAX)
                or (p_min[1] <= PERCENTILE_MIN)
                or (p_min[1] > PERCENTILE_MAX)
            ):
                msg = "Percentile must be an interval between [0,100]"
                raise ValueError(msg)
            if (
                (p_max[0] > p_max[1])
                or (p_max[0] < PERCENTILE_MIN)
                or (p_max[0] >= PERCENTILE_MAX)
                or (p_max[1] <= PERCENTILE_MIN)
                or (p_max[1] > PERCENTILE_MAX)
            ):
                msg = "Percentile must be an interval between [0,100]"
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
        Check the consistency of the interval number, if interval type is "density"
        or "interval_size".

        Parameters
        ----------
        nb: int
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
                msg = f"Number of intervals must be between 0 and {SIZE_INTERVAL_MAX}"
                raise ValueError(msg)
        else:
            msg = "Size of intervals does not work for interval type DENSITY"
            raise ValueError(msg)
        return size

    @field_validator("use_breakpoint")
    @classmethod
    def check_use_breakpoint(cls, use_bp: bool, info: ValidationInfo) -> bool:
        """
        If breakpoint is provided, use_breakpoint is set to True

        Parameters
        ----------
        use_bp: bool
            Use breakpoint option
        info: ValidationInfo
            Information

        Returns
        -------
        use_bp_ckecked: bool
            Check use breakpoint
        """
        if info.data.get("breakpoint") is not None:
            use_bp = True
        return use_bp

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
        min_percentile: float,
        max_percentile: float,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Select a point in the interval using percentiles

        Parameters
        ----------
        df: pd.DataFrame
           Data containing lst as a function of var
        intervals: pd.Series
           List of intervals
        percentile_min: float
           Minimum percentile value
        percentile_max: float
           Maximum percentile value

        Returns
        -------
        lst_values: np.array
            LST coordinates
        var_values: np.array
            Variable coordinates
        """
        var_values = []
        lst_values = []
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
        sparse_percentile: tuple[float, float],
        dense_percentile: tuple[float, float],
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
        # Logarithmic interpolation for percentiles
        min_percentile = [sparse_percentile[0], dense_percentile[0]]
        max_percentile = [sparse_percentile[1], dense_percentile[1]]
        f_min_percentile = lambda _: 0  # noqa: E731
        f_max_percentile = lambda _: 100  # noqa: E731
        if np.max(min_percentile) > 0.0:
            lin_interp = sp.interpolate.interp1d(
                np.log10([sparse_interval, dense_interval]),
                np.log10(min_percentile),
                kind="linear",
            )
            f_min_percentile = lambda p: np.power(  # noqa: E731
                10.0,
                lin_interp(
                    np.log10(np.clip(p, sparse_interval, dense_interval))
                ),
            )
        if np.min(max_percentile) < 100:  # noqa: PLR2004
            lin_interp = sp.interpolate.interp1d(
                np.log10([sparse_interval, dense_interval]),
                np.log10(max_percentile),
                kind="linear",
            )
            f_max_percentile = lambda p: np.power(  # noqa: E731
                10.0,
                lin_interp(
                    np.log10(np.clip(p, sparse_interval, dense_interval))
                ),
            )
        # Selection with variables percentiles
        for _, group in df.groupby(intervals):
            nb = len(group)
            value = group["lst"][
                (
                    group["lst"]
                    >= np.percentile(group["lst"], f_min_percentile(nb))
                )
                & (
                    group["lst"]
                    <= np.percentile(group["lst"], f_max_percentile(nb))
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
        df = self._prepare(np.array(var), np.array(lst))
        intervals = self._get_intervals(df["var"])

        # Compute point coordinates for regression
        if self.percentile is not None:
            # Selection with percentile
            return self._select_percentile(
                df=df,
                intervals=intervals,
                min_percentile=self.percentile[0],
                max_percentile=self.percentile[1],
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
        break_indice = np.nanargmax(lst_values[::-1])
        break_indice = len(lst_values) - break_indice - 1
        if break_indice == len(lst_values) - 1:
            break_indice = break_indice - 1
        elif break_indice == 0:
            break_indice = break_indice + 1
        return var_values[break_indice], lst_values[break_indice]

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
            extremum_indice = np.nanargmax(lst[::-1])
        else:
            extremum_indice = np.nanargmin(lst[::-1])
        extremum_indice = len(lst) - extremum_indice - 1
        if extremum_indice == len(lst) - 1:
            extremum_indice = extremum_indice - 1
        elif extremum_indice == 0:
            extremum_indice = extremum_indice + 1
        return var[extremum_indice], lst[extremum_indice]

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
            if self.percentile is not None and self.percentile_limit is not None
            else f"percentile = {self.percentile},"
            if self.percentile is not None
            else (
                f"percentile_bounds = {self.percentile_bounds},"
                f"percentile_intervals = {self.percentile_intervals},"
            )
            if self.percentile_bounds is not None
            else f"nb_points = {self.nb_points},"
        )
        break_prop = (
            f"breakpoint={self.breakpoint}," if self.use_breakpoint else ""
        )

        return (
            f"position={self.position.value},"
            f"interval_type={self.interval_type.value},"
            f"{interval_prop}"
            f"{percentile_prop}"
            f"selection={self.selection.value},"
            f"{break_prop}"
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
            f"  - percentile = {self.percentile} (limit = {self.percentile_limit}\n"
            if self.percentile is not None and self.percentile_limit is not None
            else f"  - percentile = {self.percentile}\n"
            if self.percentile is not None
            else (
                f"  - percentile_bounds = {self.percentile_bounds}\n"
                f"  - percentile_intervals = {self.percentile_intervals}\n"
            )
            if self.percentile_bounds is not None
            else f"nb_points = {self.nb_points}\n"
        )
        break_prop = (
            f"\n  - breakpoint={self.breakpoint},"
            if self.use_breakpoint
            else ""
        )
        return (
            f"  - position={self.position.value}\n"
            f"  - interval_type={self.interval_type.value}\n"
            f"{interval_prop}"
            f"{percentile_prop}"
            f"  - selection={self.selection.value}"
            f"{break_prop}"
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
            "use_breakpoint": self.use_breakpoint,
            **({"breakpoint": self.breakpoint} if self.use_breakpoint else {}),
        }


class LinearEdge(RegressionEdge):
    """Class for linear edge"""

    coeffs: tuple[float, float] = (float("inf"), float("inf"))

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
        # If the option use_break_point is activated, the regression
        # occurs only on a part of the selected point: after the break point
        # for top edge and before the break point for bottom edge.
        if self.use_breakpoint:
            break_var, _ = self.search_breakpoint(var, lst)
            if self.position.name == EdgePosition.TOP.name:
                tmp = var_values[var_values >= break_var]
                lst_values = lst_values[var_values >= break_var]
                var_values = tmp
            else:
                tmp = var_values[var_values <= break_var]
                lst_values = lst_values[var_values <= break_var]
                var_values = tmp
            if len(var_values) == 1:
                msg = f"LinearEdge: breakpoint ({break_var}) at the edge of the domain"
                logger.warning(msg)
        # Linear regression
        self.coeffs = tuple(np.polyfit(var_values, lst_values, 1))

    def __repr__(self) -> str:
        """
        String conversion method
        """
        return f"LinearEdge({self._common_repr()},coeffs={self.coeffs})"

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        return f"LinearEdge:\n{self._common_str()}\n  - coeffs={self.coeffs}"

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
        common_dict["coeffs"] = tuple(float(coeff) for coeff in self.coeffs)
        return common_dict


class ThresholdLinearEdge(RegressionEdge):
    """Class for linear edge with a threshold"""

    coeffs: tuple[float, float] = (float("inf"), float("inf"))
    threshold: float = float("inf")

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
        if self.use_breakpoint:
            guess, _ = self.search_breakpoint(var, lst)
            self.breakpoint = guess
        else:
            guess, _ = self.search_extremum_point(var_values, lst_values)
            if guess == var_values[-1]:
                logger.warning(
                    "ThresholdLinearEdge: Threshold not found, last interval selected"
                )
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
        return (
            "ThresholdLinearEdge("
            f"{self._common_repr()},"
            f"coeffs={self.coeffs},"
            f"threshold={self.threshold}"
        )

    def __str__(self) -> str:
        """
        String conversion method for end-users
        """
        return (
            f"ThresholdLinearEdge:\n"
            f"  - {self._common_str()}\n"
            f"  - coeffs={self.coeffs}\n"
            f"  - threshold={self.threshold}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
        common_dict["coeffs"] = tuple(float(coeff) for coeff in self.coeffs)
        common_dict["threshold"] = float(self.threshold)
        return common_dict


class DoubleLinearEdge(RegressionEdge):
    """Class for double linear edge"""

    coeffs1: tuple[float, float] = (float("inf"), float("inf"))
    coeffs2: tuple[float, float] = (float("inf"), float("inf"))
    fit_breakpoint: float = float("inf")

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
        if self.use_breakpoint:
            self.breakpoint, _ = self.search_extremum_point(
                var_values, lst_values
            )
            if self.breakpoint == var_values[-1]:
                logger.warning("DoubleLinearEdge: Last interval selected")
            if self.breakpoint == var_values[0]:
                logger.warning("DoubleLinearEdge: First interval selected")
        # Initialize piecewise linear fit
        pwlf_solver = pwlf.PiecewiseLinFit(
            var_values, lst_values, degree=1, seed=123
        )
        # fit the data for 2 line segments
        if self.use_breakpoint:
            breaks = pwlf_solver.fit_guess([self.breakpoint])
            self.fit_breakpoint = breaks[1]
        else:
            pwlf_solver.fit(2)
            self.fit_breakpoint = pwlf_solver.fit_breaks[1]
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
        return (
            f"DoubleLinearEdge("
            f"{self._common_repr()},"
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
            f"  - {self._common_str()}\n"
            f"  - coeffs1={self.coeffs1}\n"
            f"  - coeffs2={self.coeffs2}\n"
            f"  - fit_breakpoint={self.fit_breakpoint}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
        common_dict["coeffs1"] = tuple(float(coeff) for coeff in self.coeffs1)
        common_dict["coeffs2"] = tuple(float(coeff) for coeff in self.coeffs2)
        common_dict["fit_breakpoint"] = float(self.fit_breakpoint)
        return common_dict


class FlatLinearEdge(RegressionEdge):
    """Class for flat linear edge"""

    coeffs1: float = float("inf")
    coeffs2: tuple[float, float] = (float("inf"), float("inf"))
    fit_breakpoint: float = float("inf")

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
        if self.use_breakpoint:
            guess, _ = self.search_breakpoint(var, lst)
            self.breakpoint = guess
        else:
            guess, _ = self.search_extremum_point(var_values, lst_values)

        if guess == var_values[-1]:
            logger.warning("FlatLinearEdge: Last interval selected")
        if guess == var_values[0]:
            logger.warning("FlatLinearEdge: First interval selected")
        # Initialize piecewise linear fit
        pwlf_solver = pwlf.PiecewiseLinFit(
            var_values,
            lst_values,
            seed=123,
        )
        breaks = pwlf_solver.fit_guess([guess])
        # fit the data for 2 line segments
        self.fit_breakpoint = breaks[1]
        self.coeffs1 = (
            pwlf_solver.beta[1] * self.fit_breakpoint
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
        return (
            f"FlatLinearEdge("
            f"{self._common_repr()},"
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
            f"  - {self._common_str()}\n"
            f"  - coeffs1={self.coeffs1}\n"
            f"  - coeffs2={self.coeffs2}\n"
            f"  - fit_breakpoint={self.fit_breakpoint}"
        )

    def to_dict(self) -> dict:
        """
        Export to a dictionary
        """
        common_dict = super().to_dict()
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
        return f"ParabolicEdge({self._common_repr()},coeffs={self.coeffs})"

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
        For each interval, compute the point coordinates used for the regression:
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
        return f"FlatRegressionEdge({self._common_repr()},coeff={self.coeff})"

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
    # Percentile interval to considered for point selection
    percentile: tuple[float, float] | None = None
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
    def check_percentile(cls, p: tuple[float, float]) -> tuple[float, float]:
        """
        Check the consistency of the percentile interval

        Parameters
        ----------
        p: tuple[float,float]
            Percentile interval

        Returns
        -------
        percentile: tuple[float,float]
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
            self.value = df["lst"][
                (df["lst"] >= np.percentile(df["lst"], self.percentile[0]))
                & (df["lst"] <= np.percentile(df["lst"], self.percentile[1]))
            ].pipe(self._select)
        # Selection with the numver of points
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
                f"percentile={self.percentile}, percentile_limit={self.percentile_limit},"
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
                f"  - percentile={self.percentile} (limit={self.percentile_limit}\n"
                if self.percentile_limit is not None
                else f"  - percentile={self.percentile}\n"
            )
            if self.percentile is not None
            else f"   - nb_points={self.nb_points}\n"
        )
        return (
            f"FlatPercentileEdge\n"
            f"  - position={self.position.value}\n"
            f"  - {percentile_prop}\n"
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
