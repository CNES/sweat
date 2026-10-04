# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions to process data
"""

import datetime as dt

import numpy as np
import pandas as pd

from sweat.stic.models.flux import (
    compute_omega_stressed,
    compute_omega_unstressed,
)
from sweat.stic.runner import run_batch_init_model, run_batch_model


def search(df: pd.DataFrame, name: str, date: str) -> pd.Series:
    """
    Search in dataframe
    """
    date_dt = dt.datetime.strptime(date, "%Y-%m-%d")  # noqa DTZ007
    query = (
        f"name.str.contains('{name}') and "
        f"date >= '{date_dt:%Y-%m-%d} 00:00:00' and "
        f"date<'{date_dt + dt.timedelta(days=1):%Y-%m-%d} 00:00:00' "
    )
    res = df.query(query)
    if len(res) > 1:
        msg = f"Too many results (Nb results found= {len(res)})"
        raise ValueError(msg)
    if len(res) == 0:
        msg = "Data not found"
        raise ValueError(msg)
    return res.iloc[0, :]


def focus_one_point(
    input_df: pd.DataFrame,
    name: str,
    date: str,
    version: str,
    results_df: pd.DataFrame | None = None,
):
    """
    Run stic on a specific data
    """
    params = search(input_df, name, date)
    print(f"Date: {params['date']:%Y-%m-%d %H:%M:%S}")
    if results_df is not None:
        res = search(results_df, name, date)
        print(f"LE (estimated) = {res['le']:.3f}")
        print(f"LE (measured) = {res['ec_le']:.3f}")
    print("\nDetail STIC execution :")
    output = run_batch_model(
        pd.DataFrame([params]),
        threshold=0.01,
        nb_steps=15,
        debug=True,
        version=version,
    )
    print(f"LE (estimated) = {output[0, 0]:.3f}")
    print(f"H (estimated) = {output[0, 1]:.3f}")
    print(f"G (estimated) = {output[0, 3]:.3f}")
    print(f"ga (estimated) = {output[0, 4]:.3f}")
    print(f"gs (estimated) = {output[0, 5]:.3f}")


def run_stic(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """
    Compute STIC model
    """
    res = run_batch_model(data=df, threshold=0.01, nb_steps=15, version=version)
    output_df = pd.DataFrame(
        {
            "name": df["name"].values,
            "date": df["date"],
            "le": res[:, 0],
            "h": res[:, 1],
            "ef": res[:, 2],
            "e_interception": res[:, 3],
            "e_soil": res[:, 4],
            "t": res[:, 5],
            "g": res[:, 6],
            "rn": df["rn"].values,
            "srn": df["srn"].values,
            "ta": df["ta"].values,
            "td": df["td"].values,
            "rh": df["rh"].values,
            "ts": df["ts"].values,
            "ga": res[:, 7],
            "gs": res[:, 8],
            "t0": res[:, 9],
            "m": res[:, 10],
            "converged": res[:, 11],
            "stressed": res[:, 12],
        }
    )
    output_df[["converged", "stressed"]] = output_df[
        ["converged", "stressed"]
    ].astype(int)
    output_df["e"] = output_df["e_interception"] + output_df["e_soil"]
    output_df["rn-g"] = output_df["rn"] - output_df["g"]
    output_df["diff_le"] = output_df["le"] - df["ec_le"]
    output_df["diff_le_closed"] = output_df["le"] - df["ec_le_closed"]
    output_df["ts-ta"] = df["ts"] - df["ta"]
    output_df["gs/ga"] = np.clip(output_df["gs"] / output_df["ga"], 0, 1)
    output_df["t/le"] = np.clip(output_df["t"] / output_df["le"], 0, 1)
    output_df["e/le"] = np.clip(output_df["e"] / output_df["le"], 0, 1)
    output_df["nir/swir"] = np.clip(
        (df["nir"] - df["swir"]) / (df["nir"] + df["swir"] + 0.01),
        -1,
        1,
    )
    cols = list(df.columns.difference(output_df.columns))
    cols.append("date")
    cols.append("name")
    output_df = pd.merge(left=output_df, right=df[cols], on=["date", "name"])
    output_df.attrs["label"] = f"STIC {version}"
    return output_df


def init_stic(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """
    Compute STIC model (initialization only)
    """
    res = run_batch_init_model(data=df, version=version)
    output_df = pd.DataFrame(
        {
            "name": df["name"].values,
            "date": df["date"],
            "le": res[:, 0],
            "h": res[:, 1],
            "g": res[:, 2],
            "rn": df["rn"].values,
            "ta": df["ta"].values,
            "td": df["td"].values,
            "rh": df["rh"].values,
            "ts": df["ts"].values,
            "ga": res[:, 3],
            "gs": res[:, 4],
            "t0": res[:, 5],
            "t0d": res[:, 6],
            "m": res[:, 7],
            "m_canopy": res[:, 8],
            "m_soil": res[:, 9],
            "m_surf": res[:, 10],
            "m_rz": res[:, 11],
            "da": res[:, 12],
            "ds": res[:, 13],
            "es": res[:, 14],
            "ea": res[:, 15],
            "e0": res[:, 16],
            "esstar": res[:, 17],
            "e0star": res[:, 18],
            "slope": res[:, 19],
            "alpha": res[:, 20],
            "tw": res[:, 21],
            "essatr_w": res[:, 22],
            "g_r": res[:, 23],
            "stressed": res[:, 24],
        }
    )
    output_df["stressed"] = output_df["stressed"].astype(int)
    output_df["ts-ta"] = df["ts"] - df["ta"]
    output_df["nir/swir"] = np.clip(
        (df["nir"] - df["swir"]) / (df["nir"] + df["swir"] + 0.01),
        -1,
        1,
    )
    cols = list(df.columns.difference(output_df.columns))
    cols.append("date")
    cols.append("name")
    output_df = pd.merge(left=output_df, right=df[cols], on=["date", "name"])
    output_df.attrs["label"] = f"STIC {version}"
    return output_df


def analyze(df: pd.DataFrame):
    """
    Analyze data
    """
    df["omega1"] = df.apply(
        lambda x: compute_omega_unstressed(x["slope"], x["ga"], x["gs"]), axis=1
    )
    df["omega2"] = df.apply(
        lambda x: compute_omega_stressed(
            x["slope"], x["ga"], x["gs"], x["g_r"]
        ),
        axis=1,
    )
