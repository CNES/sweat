# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions for metrics
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_r2_score(y_true, y_pred):
    idx = np.isfinite(y_true) & np.isfinite(y_pred)

    if np.sum(idx) < 2:
        return np.nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UndefinedMetricWarning)

        r2 = r2_score(y_true[idx], y_pred[idx])

        if any(issubclass(warn.category, UndefinedMetricWarning) for warn in w):
            return np.nan

    return r2


def safe_polyfit(x, y):
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
) -> tuple[float, float, float, float, float]:
    """
    Compute slope, mbe, mae, rmse, r2
    """
    measured_arr = np.array(measured)
    estimated_arr = np.array(estimated)
    slope, _ = safe_polyfit(measured_arr, estimated_arr)
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
    else:
        mae = np.nan
        rmse = np.nan
        r2 = np.nan
    return (slope, mbe, mae, rmse, r2)


def generate_metrics_table(
    df: pd.DataFrame, variables: list[str] | None = None
) -> pd.DataFrame:
    """
    Generate metrics table
    """
    if variables is None:
        variables = ["le", "h", "le_closed"]
    metrics = []
    # Compute by site and landcover
    cols = ["name"]
    if "landcover" in df.columns:
        cols = ["name", "landcover"]
    for name, group in df.groupby(cols):
        for var in variables:
            est_var = var if "_closed" not in var else var[:-7]
            slope, mbe, mae, rmse, r2 = compute_metrics(
                measured=group[f"ec_{var}"], estimated=group[est_var]
            )
            metrics.append(
                {
                    "name": name[0] if len(name) >= 2 else name,
                    **({"landcover": name[1]} if len(name) >= 2 else {}),
                    "nb": len(group),
                    "variable": var,
                    "slope": slope,
                    "mbe": mbe,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )
    # Compute for all
    for var in variables:
        est_var = var if "_closed" not in var else var[:-7]
        slope, mbe, mae, rmse, r2 = compute_metrics(
            measured=df[f"ec_{var}"], estimated=df[est_var]
        )
        metrics.append(
            {
                "name": "all",
                **({"landcover": "all"} if len(name) >= 2 else {}),
                "landcover": "all",
                "nb": len(df),
                "variable": var,
                "slope": slope,
                "mbe": mbe,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
    df_metrics = pd.DataFrame.from_records(metrics)
    if df.attrs.get("label") is not None:
        df_metrics.attrs["label"] = df.attrs["label"]
    return df_metrics
