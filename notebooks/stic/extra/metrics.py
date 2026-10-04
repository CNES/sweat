# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions for metrics
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_r2_score(y_true: npt.NDArray, y_pred: npt.NDArray):
    """
    Compute R2 score with nan values

    Parameters
    ----------
    y_true : np.array
        Measured values
    y_pred : np.array
        Estimated values

    Returns
    -------
    r2 : float
        R2 score
    """
    idx = np.isfinite(y_true) & np.isfinite(y_pred)

    if np.sum(idx) < 2:
        return np.nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UndefinedMetricWarning)

        r2 = r2_score(y_true[idx], y_pred[idx])

        if any(issubclass(warn.category, UndefinedMetricWarning) for warn in w):
            return np.nan

    return r2


def safe_polyfit(x: npt.NDArray, y: npt.NDArray) -> tuple[float, float]:
    """
    Compute least squares polynomial fit (order 1) with nan values

    Parameters
    ----------
    x : np.array
        Values
    y : np.array
        Values

    Returns
    -------
    slope, intercept : tuple[float, float]
        Coefficient of the polynomial fit
    """
    idx = np.isfinite(x) & np.isfinite(y)

    if np.sum(idx) < 2:
        return np.nan, np.nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", np.exceptions.RankWarning)

        slope, intercept = np.polyfit(x[idx], y[idx], 1)

        if any(
            issubclass(warn.category, np.exceptions.RankWarning) for warn in w
        ):
            return np.nan, np.nan

    return slope, intercept


def compute_metrics(
    measured: npt.ArrayLike, estimated: npt.ArrayLike
) -> tuple[float, ...]:
    """
    Compute slope, mbe, mae, rmse, r2

    Parameters
    ----------
    measured : np.array
        Measured values
    estimated : np.array
        Estimated values

    Returns
    -------
    metrics : tuple[float]
        Metrics values
    """
    measured_arr = np.asarray(measured)
    estimated_arr = np.asarray(estimated)
    slope, intercept = safe_polyfit(measured_arr, estimated_arr)
    if len(measured_arr - estimated_arr) > 0 and not np.all(
        np.isnan(measured_arr - estimated_arr)
    ):
        mbe = np.nanmean(estimated_arr - measured_arr)
    else:
        mbe = np.nan
    idx = np.isfinite(measured) & np.isfinite(estimated)
    if len(measured_arr[idx]) > 0 and len(estimated_arr[idx]):
        mae = mean_absolute_error(measured_arr[idx], estimated_arr[idx])
        rmse = np.sqrt(
            mean_squared_error(measured_arr[idx], estimated_arr[idx])
        )
        r2 = safe_r2_score(measured_arr[idx], estimated_arr[idx])
        nrmse_mean = rmse / np.abs(np.nanmean(measured_arr[idx]))
        if np.nanmax(measured_arr[idx]) - np.nanmin(measured_arr[idx]) > 1.0e-6:
            nrmse_minmax = rmse / (
                np.nanmax(measured_arr[idx]) - np.nanmin(measured_arr[idx])
            )
        else:
            nrmse_minmax = np.nan
        if (
            np.nanpercentile(measured_arr[idx], 0.75)
            - np.nanpercentile(measured_arr[idx], 0.25)
        ) > 1.0e-6:
            nrmse_iq = rmse / (
                np.nanpercentile(measured_arr[idx], 0.75)
                - np.nanpercentile(measured_arr[idx], 0.25)
            )
        else:
            nrmse_iq = np.nan
    else:
        mae = np.nan
        rmse = np.nan
        r2 = np.nan
        nrmse_mean = np.nan
        nrmse_minmax = np.nan
        nrmse_iq = np.nan
    return (
        slope,
        intercept,
        mbe,
        mae,
        rmse,
        r2,
        nrmse_mean,
        nrmse_minmax,
        nrmse_iq,
    )


def generate_metrics_table(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    ranges: dict[str, list[tuple[int, int]]] | None = None,
) -> pd.DataFrame:
    """
    Generate metrics table for a lits of variables with optional value ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    variables : list[str]
        Variables to compute metrics for
    ranges : dict[str, list[tuple[int, int]]]
        Dictionary mapping variable names to list of (min, max) tuples.
        Example: {"le": [(0, 200), (200, 400), (400, 600), (600, float('inf'))]}
    """
    if variables is None:
        variables = ["le", "h", "le_closed"]
    if ranges is None:
        ranges = {}

    metrics = []

    # Compute by site and landcover
    cols = ["name"]
    if "landcover" in df.columns:
        cols = ["name", "landcover"]

    # Group by site name and landcover if provided
    for name, group in df.groupby(cols):
        for var in variables:
            est_var = var if "_closed" not in var else var[:-7]

            # Compute for the entire group
            slope, _, mbe, mae, rmse, r2, nrmse_mean, nrmse_minmax, nrmse_iq = (
                compute_metrics(
                    measured=group[f"ec_{var}"], estimated=group[est_var]
                )
            )
            metrics.append(
                {
                    "name": name[0] if len(name) >= 2 else name,
                    **({"landcover": name[1]} if len(name) >= 2 else {}),
                    "nb": len(group),
                    "variable": var,
                    "range": "all",
                    "slope": slope,
                    "mbe": mbe,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "nrmse_mean": nrmse_mean,
                    "nrmse_minmax": nrmse_minmax,
                    "nrmse_iq": nrmse_iq,
                }
            )

            # Compute for each range if specified
            if var in ranges:
                for min_val, max_val in ranges[var]:
                    mask = (group[f"ec_{var}"] >= min_val) & (
                        group[f"ec_{var}"] < max_val
                    )
                    group_range = group[mask]

                    if len(group_range) > 0:
                        (
                            slope,
                            _,
                            mbe,
                            mae,
                            rmse,
                            r2,
                            nrmse_mean,
                            nrmse_minmax,
                            nrmse_iq,
                        ) = compute_metrics(
                            measured=group_range[f"ec_{var}"],
                            estimated=group_range[est_var],
                        )

                        range_label = (
                            f">{min_val}"
                            if max_val == float("inf")
                            else f"<{max_val}"
                            if min_val == float("-inf")
                            else f"{min_val}-{max_val}"
                        )
                        metrics.append(
                            {
                                "name": name[0] if len(name) >= 2 else name,
                                **(
                                    {"landcover": name[1]}
                                    if len(name) >= 2
                                    else {}
                                ),
                                "nb": len(group_range),
                                "variable": var,
                                "range": range_label,
                                "slope": slope,
                                "mbe": mbe,
                                "mae": mae,
                                "rmse": rmse,
                                "r2": r2,
                                "nrmse_mean": nrmse_mean,
                                "nrmse_minmax": nrmse_minmax,
                                "nrmse_iq": nrmse_iq,
                            }
                        )

    # Compute for all
    for var in variables:
        est_var = var if "_closed" not in var else var[:-7]
        slope, _, mbe, mae, rmse, r2, nrmse_mean, nrmse_minmax, nrmse_iq = (
            compute_metrics(measured=df[f"ec_{var}"], estimated=df[est_var])
        )
        metrics.append(
            {
                "name": "all",
                **({"landcover": "all"} if "landcover" in df.columns else {}),
                "nb": len(df),
                "variable": var,
                "range": "all",
                "slope": slope,
                "mbe": mbe,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "nrmse_mean": nrmse_mean,
                "nrmse_minmax": nrmse_minmax,
                "nrmse_iq": nrmse_iq,
            }
        )

        # Compute for each range if specified
        if var in ranges:
            for min_val, max_val in ranges[var]:
                mask = (df[f"ec_{var}"] >= min_val) & (
                    df[f"ec_{var}"] < max_val
                )
                df_range = df[mask]

                if len(df_range) > 0:
                    (
                        slope,
                        _,
                        mbe,
                        mae,
                        rmse,
                        r2,
                        nrmse_mean,
                        nrmse_minmax,
                        nrmse_iq,
                    ) = compute_metrics(
                        measured=df_range[f"ec_{var}"],
                        estimated=df_range[est_var],
                    )
                    range_label = (
                        f">{min_val}"
                        if max_val == float("inf")
                        else f"<{max_val}"
                        if min_val == float("-inf")
                        else f"{min_val}-{max_val}"
                    )
                    metrics.append(
                        {
                            "name": "all",
                            **(
                                {"landcover": "all"}
                                if "landcover" in df.columns
                                else {}
                            ),
                            "nb": len(df_range),
                            "variable": var,
                            "range": range_label,
                            "slope": slope,
                            "mbe": mbe,
                            "mae": mae,
                            "rmse": rmse,
                            "r2": r2,
                            "nrmse_mean": nrmse_mean,
                            "nrmse_minmax": nrmse_minmax,
                            "nrmse_iq": nrmse_iq,
                        }
                    )

    df_metrics = pd.DataFrame.from_records(metrics)
    if df.attrs.get("label") is not None:
        df_metrics.attrs["label"] = df.attrs["label"]
    return df_metrics
