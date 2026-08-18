# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions to process data
"""

import datetime as dt
import os
import re
import traceback
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from sweat.common import utils
from sweat.common.flux import CST_SB, compute_et_from_le
from sweat.stic.constant import KELVIN_CST
from sweat.stic.convert import (
    convert_kelvin_to_celsius,
    convert_to_local_time,
    convert_to_rh,
)
from sweat.stic.models.functions import _tetens

# Functions for handling EC data


def get_code(name: str) -> str:
    """
    Get code site from the name
    """
    m = re.match(r"(US-.*)[_,-].*", name)
    name = m.group(1) if m else f"US-{name}"
    if len(name) > 6:
        name = name[0:6]
    return name[:4].upper() + name[4:].lower()


def get_ec_data_filename(name: str, root: str) -> str:
    """
    Get EC data path for a site
    """
    code = get_code(name)
    files = [
        f
        for f in os.listdir(root)
        if re.match(r"AMF_US-(.*)_BASE_HH_.*\.csv", f)
    ]
    for file in files:
        if code.lower() in file.lower():
            return os.path.join(root, file)
    msg = f"File {file} for ground truth data not found"
    raise OSError(msg)


# Functions for handling Ameriflux EC data


def get_ameriflux_archive(name: str, path: str) -> str:
    """
    Get EC data path for a site
    """
    files = [
        f
        for f in os.listdir(path)
        if re.match(rf"AMF_{name}_BASE-BADM_\d+-\d+.zip", f)
    ]
    if len(files) > 0:
        # Return first file
        return os.path.join(path, files[0])
    msg = f"File for site {name} not found"
    raise OSError(msg)


def get_best_column(df: pd.DataFrame, suffix: str):
    """
    Get best column to process
    """
    shift = suffix.count("_")
    if df.columns.str.startswith(f"{suffix}_PI_F").any():
        return min(
            (c for c in df.columns if c.startswith(f"{suffix}_PI_F")),
            key=lambda c: (
                (0,)
                if c == f"{suffix}_PI_F"
                else (1, *c.split("_")[3 + shift :])
            ),
        )
    if df.columns.str.startswith(f"{suffix}_PI").any():
        return min(
            (
                c
                for c in df.columns
                if c.startswith(f"{suffix}_PI")
                and not c.startswith(f"{suffix}_PI_F")
            ),
            key=lambda c: (
                (0,) if c == f"{suffix}_PI" else (1, *c.split("_")[2 + shift :])
            ),
        )
    if df.columns.str.startswith(f"{suffix}_").any():
        return min(
            (c for c in df.columns if c.startswith(suffix)),
            key=lambda c: (
                (0,) if c == suffix else (1, *c.split("_")[1 + shift :])
            ),
        )
    if suffix in df.columns:
        return suffix
    return None


def process_ameriflux_column(
    df: pd.DataFrame, col: str, name: str | None = None
) -> pd.DataFrame:
    """
    Process a column of Ameriflux data
    """
    # LE
    best = get_best_column(df, col)
    if best is None:
        best = col
        df[col] = np.nan
    if name is None:
        name = col.lower()
    return df.rename(columns={best: name})


def read_ameriflux_data(name: str) -> pd.DataFrame:
    """
    Read EC data file for a site
    """
    # Read
    df = pd.read_csv(
        name,
        sep=",",
        encoding="ascii",
        skiprows=2,
        usecols=(
            lambda col: (
                col
                in ["TIMESTAMP_START", "TIMESTAMP_END", "G", "TA", "LE", "H"]
                or col.startswith(  # type: ignore
                    (
                        "TA_",
                        "RH",
                        "G_PI_F",
                        "H_PI_F",
                        "LE_PI_F",
                        "NETRAD",
                        "SW_IN",
                        "SW_OUT",
                        "LW_IN",
                        "LW_OUT",
                        "TS",
                    )
                )
            )
        ),
        dtype=str,
    )
    df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"])
    df["TIMESTAMP_END"] = pd.to_datetime(df["TIMESTAMP_END"])
    df[df.columns.difference(["TIMESTAMP_START", "TIMESTAMP_END"])] = (
        pd.to_numeric(  # type: ignore
            df[
                df.columns.difference(["TIMESTAMP_START", "TIMESTAMP_END"])
            ].stack(),
            errors="coerce",
        ).unstack()
    )
    # Rename columns
    df = df.rename(columns={"TIMESTAMP_START": "start", "TIMESTAMP_END": "end"})
    # print(df.columns)
    # LE
    df = process_ameriflux_column(df, "LE")
    # H
    df = process_ameriflux_column(df, "H")
    # RN
    df = process_ameriflux_column(df, "NETRAD", "rn")
    # SW IN
    df = process_ameriflux_column(df, "SW_IN")
    # SW OUT
    df = process_ameriflux_column(df, "SW_OUT")
    # LW IN
    df = process_ameriflux_column(df, "LW_IN")
    # LW OUT
    df = process_ameriflux_column(df, "LW_OUT")
    # G
    df = process_ameriflux_column(df, "G")
    # TA
    df = process_ameriflux_column(df, "TA")
    # TS
    df = process_ameriflux_column(df, "TS")
    # RH
    df = process_ameriflux_column(df, "RH")
    # Keep required columns
    df = df[
        [
            "start",
            "end",
            "le",
            "h",
            "g",
            "rn",
            "sw_out",
            "sw_in",
            "lw_out",
            "lw_in",
            "ta",
            "ts",
            "rh",
        ]
    ]
    # Exclude nodata
    value_cols = df.columns.difference(["start", "end"])
    # replace -9999 with NaN only in value columns
    df[value_cols] = df[value_cols].replace(-9999.0, np.nan)
    # drop rows where all value column is NaN
    df = df.dropna(subset=value_cols, how="all")
    with warnings.catch_warnings(action="ignore"):
        # Compute EF
        df["ef"] = df["le"] / (df["h"] + df["le"])
        # Compute ET
        df["et"] = df.apply(lambda x: compute_et_from_le(x["le"]), axis=1)
        # Compute solar net radiation
        df["srn"] = df["sw_in"] - df["sw_out"]
        # Compute dewpoint temperature
        eastar = 6.13753 * np.exp((17.27 * df["ta"]) / (df["ta"] + 237.3))
        ea = (df["rh"] / 100) * (eastar)
        df["td"] = 237.3 * np.log(ea / 6.13753) / (17.27 - np.log(ea / 6.13753))
        # Compute Rn - G
        df["rn-g"] = df["rn"] - df["g"]
        # Compute LE closed
        df["le_closed"] = df["rn"] - df["g"] - df["h"]
    # Keep columns
    return df[
        [
            "start",
            "end",
            "le",
            "h",
            "ef",
            "g",
            "rn",
            "srn",
            "sw_out",
            "sw_in",
            "lw_out",
            "lw_in",
            "ta",
            "ts",
            "td",
            "rh",
            "et",
            "rn-g",
            "le_closed",
        ]
    ]


def get_ameriflux_data(df: pd.DataFrame, path: str):
    """
    Get EC data at acquisition dates
    """
    # Get sites
    sites = list(df["name"].unique())
    # Loop for each site
    datasets = []
    ec_datasets = []
    errors = []
    for site in tqdm(sites, desc="Processing sites", unit="site"):
        try:
            # Select data for a site
            selected_df = df[df["name"] == site].reset_index(drop=True)
            # Code du site
            code = get_code(site)
            # Read data
            archive = get_ameriflux_archive(code, path)
            with zipfile.ZipFile(archive, "r") as zip_ref:
                filename = f"{Path(archive).stem}.csv".replace("-BADM", "_HH")
                with zip_ref.open(filename) as file:
                    ec_df = read_ameriflux_data(file)  # type: ignore
                    ec_df = ec_df.rename(
                        columns={
                            v: f"am_{v}"
                            for v in ec_df.columns
                            if v not in ["start", "end"]
                        }
                    )
            # Search indexes
            intervals = pd.IntervalIndex.from_arrays(
                ec_df["start"].dt.tz_localize(None),
                ec_df["end"].dt.tz_localize(None),
                closed="both",
            )
            intervals = intervals.astype("interval[datetime64[ns]]")
            matches, not_matches = intervals.get_indexer_non_unique(
                selected_df["date"].dt.tz_localize(None)
            )
            # Remove no matches
            matches = matches[matches != -1]
            datasets.append(
                selected_df.drop(not_matches).reset_index(drop=True)
            )
            ec_datasets.append(
                pd.concat(
                    [
                        selected_df[["name", "date"]]
                        .drop(not_matches)
                        .reset_index(drop=True),
                        ec_df.iloc[matches]
                        .drop(columns=["start", "end"])
                        .reset_index(drop=True),
                    ],
                    axis=1,
                )
            )
            tqdm.write(
                f"Site {site} processed with matches = "
                f"{len(matches)} and not matches = {len(not_matches)}"
            )
        except Exception as e:  # noqa
            error_info = {
                "site": site,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            errors.append(error_info)
            tqdm.write(f"ERROR for site {site}: {e}")
    if len(errors) > 0:
        print("\n=== ERRORS ENCOUNTERED ===")
        for error in errors:
            print(f"\nSite: {error['site']}")
            print(f"Error: {error['error']}")
            print(error["traceback"])
    else:
        print("\nAll sites processed successfully!")

    return pd.concat(datasets), pd.concat(ec_datasets)


# Functions for handling Fluxnet EC data


def get_fluxnet_archive(name: str, path: str) -> str:
    """
    Get EC data path for a site
    """
    files = [
        f
        for f in os.listdir(path)
        if re.match(
            rf"AMF_{name}_FLUXNET(_FULLSET)?_(\d{{4}}-\d{{4}})_([A-Za-z0-9._-]+)\.zip",
            f,
        )
    ]
    if len(files) > 0:
        # Return first file
        return os.path.join(path, files[0])
    msg = f"File for site {name} not found"
    raise OSError(msg)


def get_best_fluxnet_column(df: pd.DataFrame, col: str):
    """
    Get best column to process fluxnet data
    """
    shift = col.count("_")
    if col in df.columns:
        return col
    if df.columns.str.startswith(f"{col}_").any():
        return min(
            (
                c
                for c in df.columns
                if c.startswith(f"{col}_") and "QC" not in c
            ),
            key=lambda c: c.split("_")[1 + shift],
        )
    return None


def process_fluxnet_column(
    df: pd.DataFrame, col: str, name: str | None = None
) -> pd.DataFrame:
    """
    Process a column of Fluxnet data
    """
    # LE
    best = get_best_fluxnet_column(df, col)
    if best is None:
        best = col
        df[col] = np.nan
    if name is None:
        name = col.lower()
    mapping = {best: name}
    if f"{best}_QC" in df.columns:
        mapping["best_QC"] = f"{name}_qc"
    return df.rename(columns=mapping)


def read_fluxnet_hh_data(name: str) -> pd.DataFrame:
    """
    Read EC data file for a site
    """
    # Read
    df = pd.read_csv(
        name,
        sep=",",
        encoding="ascii",
        usecols=(
            lambda col: (
                col
                in [
                    "TIMESTAMP_START",
                    "TIMESTAMP_END",
                    "LE_CORR",
                    "H_CORR",
                ]
                or col.startswith(  # type: ignore
                    (
                        "G_F_MDS",
                        "TA_F_MDS",
                        "TS_F_MDS",
                        "RH",
                        "H_F_MDS",
                        "LE_F_MDS",
                        "NETRAD",
                        "SW_IN_F",
                        "SW_OUT",
                        "LW_IN_F",
                        "LW_OUT",
                    )
                )
            )
        ),
        parse_dates=["TIMESTAMP_START", "TIMESTAMP_END"],
    )
    # Rename columns
    df = df.rename(columns={"TIMESTAMP_START": "start", "TIMESTAMP_END": "end"})
    # print(df.columns)
    # LE
    df = process_fluxnet_column(df, "LE_F_MDS", "le")
    # LE CLOSED
    df = df.rename(columns={"LE_CORR": "le_closed"})
    # H
    df = process_fluxnet_column(df, "H_F_MDS", "h")
    # H CLOSED
    df = df.rename(columns={"H_CORR": "h_closed"})
    # RN
    df = process_fluxnet_column(df, "NETRAD", "rn")
    # SW IN
    df = process_fluxnet_column(df, "SW_IN_F", "sw_in")
    # SW OUT
    df = process_fluxnet_column(df, "SW_OUT", "sw_out")
    # LW IN
    df = process_fluxnet_column(df, "LW_IN_F", "lw_in")
    # LW OUT
    df = process_fluxnet_column(df, "LW_OUT", "lw_out")
    # G
    df = process_fluxnet_column(df, "G_F_MDS", "g")
    # TA
    df = process_fluxnet_column(df, "TA_F_MDS", "ta")
    # TS
    df = process_fluxnet_column(df, "TS_F_MDS", "ts")
    # RH
    df = process_fluxnet_column(df, "RH", "rh")
    # Keep required columns
    cols = [
        item
        for c in [
            "start",
            "end",
            "le",
            "h",
            "g",
            "rn",
            "sw_out",
            "sw_in",
            "lw_out",
            "lw_in",
            "ta",
            "ts",
            "rh",
            "le_closed",
            "h_closed",
        ]
        for item in ([c, f"{c}_qc"] if f"{c}_qc" in df.columns else [c])
    ]
    df = df[cols]
    # Exclude nodata
    value_cols = df.columns.difference(["start", "end"])
    # replace -9999 with NaN only in value columns
    df[value_cols] = df[value_cols].replace(-9999, np.nan)
    # drop rows where all value column is NaN
    df = df.dropna(subset=value_cols, how="all")
    with warnings.catch_warnings(action="ignore"):
        # Compute EF
        df["ef"] = df["le"] / (df["h"] + df["le"])
        # Compute ET
        df["et"] = df.apply(lambda x: compute_et_from_le(x["le"]), axis=1)
        # Compute solar net radiation
        df["srn"] = df["sw_in"] - df["sw_out"]
        # Compute dewpoint temperature
        eastar = 6.13753 * np.exp((17.27 * df["ta"]) / (df["ta"] + 237.3))
        ea = (df["rh"] / 100) * (eastar)
        df["td"] = 237.3 * np.log(ea / 6.13753) / (17.27 - np.log(ea / 6.13753))
        # Compute Rn - G
        df["rn-g"] = df["rn"] - df["g"]
    # Keep columns
    cols = [
        item
        for c in [
            "start",
            "end",
            "le",
            "h",
            "ef",
            "g",
            "rn",
            "srn",
            "sw_out",
            "sw_in",
            "lw_out",
            "lw_in",
            "ta",
            "ts",
            "td",
            "rh",
            "le_closed",
            "h_closed",
            "et",
            "rn-g",
        ]
        for item in ([c, f"{c}_qc"] if f"{c}_qc" in df.columns else [c])
    ]
    return df[cols]


def get_fluxnet_data(df: pd.DataFrame, path: str):
    """
    Get EC data at acquisition dates
    """
    # Get sites
    sites = list(df["name"].unique())
    # Loop for each site
    datasets = []
    ec_datasets = []
    errors = []
    for site in tqdm(sites, desc="Processing sites", unit="site"):
        try:
            # Select data for a site
            selected_df = df[df["name"] == site].reset_index(drop=True)
            # Code site
            code = get_code(site)
            # Read data
            archive = get_fluxnet_archive(code, path)
            with zipfile.ZipFile(archive, "r") as zip_ref:
                if "FULLSET" not in archive:
                    filename = f"{Path(archive).stem}.csv".replace(
                        "FLUXNET", "FLUXNET_FLUXMET_HH"
                    )
                else:
                    filename = f"{Path(archive).stem}.csv".replace(
                        "FLUXNET_FULLSET", "FLUXNET_FULLSET_HH"
                    )
                with zip_ref.open(filename) as file:
                    ec_df = read_fluxnet_hh_data(file)  # type: ignore
                    ec_df = ec_df.rename(
                        columns={
                            v: f"fl_{v}"
                            for v in ec_df.columns
                            if v not in ["start", "end"]
                        }
                    )
            # Search indexes
            intervals = pd.IntervalIndex.from_arrays(
                ec_df["start"].dt.tz_localize(None),
                ec_df["end"].dt.tz_localize(None),
                closed="both",
            )
            intervals = intervals.astype("interval[datetime64[ns]]")
            matches, not_matches = intervals.get_indexer_non_unique(
                selected_df["date"].dt.tz_localize(None)
            )
            # Remove no matches
            matches = matches[matches != -1]
            datasets.append(
                selected_df.drop(not_matches).reset_index(drop=True)
            )
            ec_datasets.append(
                pd.concat(
                    [
                        selected_df[["name", "date"]]
                        .drop(not_matches)
                        .reset_index(drop=True),
                        ec_df.iloc[matches]
                        .drop(columns=["start", "end"])
                        .reset_index(drop=True),
                    ],
                    axis=1,
                )
            )
            tqdm.write(
                f"Site {site} processed with matches = "
                f"{len(matches)} and not matches = {len(not_matches)}"
            )
        except Exception as e:  # noqa
            error_info = {
                "site": site,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            errors.append(error_info)
            tqdm.write(f"ERROR for site {site}: {e}")
    if len(errors) > 0:
        print("\n=== ERRORS ENCOUNTERED ===")
        for error in errors:
            print(f"\nSite: {error['site']}")
            print(f"Error: {error['error']}")
            print(error["traceback"])
    else:
        print("\nAll sites processed successfully!")

    return pd.concat(datasets), pd.concat(ec_datasets)


# Functions for California data


def get_california_info(filename: str) -> pd.DataFrame:
    """
    Get information
    """
    return pd.read_excel(filename, sheet_name="INFO")


def read_california_data(filename: str) -> pd.DataFrame:
    """
    Read input data
    """
    sites = pd.ExcelFile(filename).sheet_names
    sites.remove("INFO")
    return pd.concat(
        [df for _, df in pd.read_excel(filename, sheet_name=sites).items()]
    )


def prepare_california_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data to run STIC
    """
    # ts, ta, td
    output_df = (
        df[["st_b10", "t2m", "d2m"]]
        .apply(convert_kelvin_to_celsius)
        .rename(columns={"st_b10": "ts", "t2m": "ta", "d2m": "td"})
    )
    # emis
    output_df["emis"] = df["st_emis"]
    # rh
    output_df["rh"] = output_df[["ta", "td"]].apply(
        lambda row: convert_to_rh(row["ta"], row["td"]), axis=1
    )
    # fc
    output_df["fc"] = df["fcover"]
    # lai
    output_df["lai"] = df["lai"]
    # ln
    eastar = output_df["ta"].apply(_tetens)
    ea = output_df["rh"] * eastar / 100
    output_df["ln"] = (
        df["st_emis"]
        * CST_SB
        * 1.24
        * (ea / (output_df["ta"] + KELVIN_CST)) ** (1 / 7)
        * (output_df["ta"] + KELVIN_CST) ** 4
        - df["st_emis"] * CST_SB * (output_df["ts"] + KELVIN_CST) ** 4
    )
    # rn
    output_df["rn"] = df["ssr"] + output_df["ln"]
    output_df["srn"] = df["ssr"]
    # localtime
    output_df["local_time"] = df[["Landsat_overpass", "Lon", "Lat"]].apply(
        lambda row: convert_to_local_time(
            date=dt.datetime.combine(
                row["Landsat_overpass"].date(),
                dt.time(hour=18, minute=45),
                tzinfo=dt.UTC,
            ),
            y=row["Lat"],
            x=row["Lon"],
            crs="epsg:4326",
        ),
        axis=1,
    )
    # name
    output_df["name"] = df["Name"]
    # lat/lon
    output_df["lon"] = df["Lon"]
    output_df["lat"] = df["Lat"]
    # Reflectance
    output_df["nir"] = df["sr_b5"]
    output_df["swir"] = df["sr_b6"]
    output_df["blue"] = df["sr_b2"]
    output_df["green"] = df["sr_b3"]
    output_df["red"] = df["sr_b4"]
    # Vegetation indices
    output_df["vari_green"] = utils.compute_vari_green(
        red=output_df["red"], green=output_df["green"], blue=output_df["blue"]
    )
    output_df["gli"] = utils.compute_gli(
        red=output_df["red"], green=output_df["green"], blue=output_df["blue"]
    )
    output_df["ndvi"] = utils.compute_ndvi(
        red=output_df["red"], nir=output_df["nir"]
    )
    output_df["gndvi"] = utils.compute_gndvi(
        nir=output_df["nir"], green=output_df["green"]
    )
    output_df["msavi"] = utils.compute_msavi(
        red=output_df["red"], nir=output_df["green"]
    )
    # dates
    output_df["date (utc)"] = df["Landsat_overpass"].apply(
        lambda t: dt.datetime.combine(
            t.date(), dt.time(hour=18, minute=45, tzinfo=dt.UTC)
        )
    )
    output_df["date"] = pd.to_datetime(
        df["Landsat_overpass"]
    ) + pd.to_timedelta(output_df["local_time"], unit="s")

    # reorder columns
    return output_df[
        [
            "name",
            "date",
            "date (utc)",
            "lat",
            "lon",
            "ts",
            "ta",
            "td",
            "rh",
            "fc",
            "lai",
            "srn",
            "rn",
            "ln",
            "local_time",
            "blue",
            "green",
            "red",
            "nir",
            "swir",
            "vari_green",
            "gli",
            "ndvi",
            "gndvi",
            "msavi",
            "emis",
        ]
    ]


def correct_data(
    df: pd.DataFrame,
    ec_df: pd.DataFrame,
    variables: list[str],
    dataset: str | None = None,
) -> pd.DataFrame:
    """
    Use EC data instead
    """
    # Get EC tower data
    if dataset is not None:
        ec_df = ec_df.rename(
            columns={
                v: f"ec_{v[len(dataset) + 1 :]}"
                for v in ec_df.columns
                if v not in ["name", "date"]
            }
        )
    ec_variables = [f"ec_{v}" for v in variables]
    ec_input = ec_df[["date", "name", *ec_variables]]
    corrected_df = pd.merge(left=df, right=ec_input, on=["date", "name"])
    return corrected_df.drop(columns=variables).rename(
        columns=dict(zip(ec_variables, variables, strict=True))
    )


def merge_data(
    df: pd.DataFrame,
    ec_df: pd.DataFrame,
    variables=list[str],
    dataset: str | None = None,
):
    cols = ["name", "date"] + [f"{dataset}_{v}" for v in variables]
    renamed_cols = {f"{dataset}_{v}": f"ec_{v}" for v in variables}
    return pd.merge(
        left=df,
        right=ec_df[cols].rename(columns=renamed_cols),
        on=["name", "date"],
    )


def prepare_ameriflux_data(
    df: pd.DataFrame, dataset: str = "fl"
) -> pd.DataFrame:
    """
    Prepare data to run STIC
    """
    # Get reflectances, lst, fc and lai
    output_df = df[
        [
            "name",
            "lat",
            "lon",
            "datetime_utc",
            "local_time",
            "datetime",
            "elevation",
            "landcover",
            "blue",
            "green",
            "red",
            "nir",
            "swir",
            "ts",
            "fc",
            "lai",
        ]
    ].copy()
    # Get EC tower data
    if dataset == "am":
        output_df = pd.concat(
            [
                output_df,
                df[
                    [
                        "am_le",
                        "am_h",
                        "am_g",
                        "am_rn",
                        "am_srn",
                        "am_sw_out",
                        "am_sw_in",
                        "am_lw_out",
                        "am_lw_in",
                        "am_ta",
                        "am_ts",
                        "am_td",
                        "am_rh",
                        "am_et",
                        "am_rn-g",
                        "am_le_closed",
                    ]
                ],
            ],
            axis=1,
        )
        output_df = output_df.rename(
            columns={
                "am_le": "ec_le",
                "am_h": "ec_h",
                "am_g": "ec_g",
                "am_rn": "rn",
                "am_srn": "srn",
                "am_sw_out": "sw_out",
                "am_sw_in": "sw_in",
                "am_lw_out": "lw_out",
                "am_lw_in": "lw_in",
                "am_ta": "ta",
                "am_td": "td",
                "am_rh": "rh",
                "am_rn-g": "ec_rn-g",
                "am_le_closed": "ec_le_closed",
            }
        )
    elif dataset == "fl":
        output_df = pd.concat(
            [
                output_df,
                df[
                    [
                        "fl_le",
                        "fl_h",
                        "fl_g",
                        "fl_rn",
                        "fl_srn",
                        "fl_sw_out",
                        "fl_sw_in",
                        "fl_lw_out",
                        "fl_lw_in",
                        "fl_ta",
                        "fl_td",
                        "fl_rh",
                        "fl_le_closed",
                        "fl_rn-g",
                    ]
                ],
            ],
            axis=1,
        )
        output_df = output_df.rename(
            columns={
                "fl_le": "ec_le",
                "fl_h": "ec_h",
                "fl_g": "ec_g",
                "fl_rn": "rn",
                "fl_srn": "srn",
                "fl_sw_out": "sw_out",
                "fl_sw_in": "sw_in",
                "fl_lw_out": "lw_out",
                "fl_lw_in": "lw_in",
                "fl_ta": "ta",
                "fl_td": "td",
                "fl_rh": "rh",
                "fl_rn-g": "ec_rn-g",
                "fl_le_closed": "ec_le_closed",
            }
        )
    else:
        msg = "Dataset unkown"
        raise ValueError(msg)
    # ln
    output_df["ln"] = -output_df["lw_out"] + output_df["lw_in"]
    # emissivity
    output_df["emis"] = output_df["lw_out"] / (
        CST_SB * (output_df["ts"] + KELVIN_CST) ** 4
    )
    # Vegetation indices
    output_df["vari_green"] = utils.compute_vari_green(
        red=output_df["red"], green=output_df["green"], blue=output_df["blue"]
    )
    output_df["gli"] = utils.compute_gli(
        red=output_df["red"], green=output_df["green"], blue=output_df["blue"]
    )
    output_df["ndvi"] = utils.compute_ndvi(
        red=output_df["red"], nir=output_df["nir"]
    )
    output_df["gndvi"] = utils.compute_gndvi(
        nir=output_df["nir"], green=output_df["green"]
    )
    output_df["msavi"] = utils.compute_msavi(
        red=output_df["red"], nir=output_df["green"]
    )
    # Rename columns
    output_df = output_df.rename(
        columns={"datetime": "date", "datetime_utc": "date (utc)"}
    )
    # Remove nan inputs
    output_df = output_df.dropna(
        subset=[
            "ts",
            "ta",
            "td",
            "rh",
            "fc",
            "lai",
            "srn",
            "rn",
            "ln",
            "local_time",
            "blue",
            "green",
            "red",
            "nir",
            "swir",
            "vari_green",
            "gli",
            "ndvi",
            "gndvi",
            "msavi",
            "emis",
        ],
        how="any",
    )
    # reorder columns
    return output_df[
        [
            "name",
            "date",
            "date (utc)",
            "lat",
            "lon",
            "landcover",
            "ts",
            "ta",
            "td",
            "rh",
            "fc",
            "lai",
            "srn",
            "rn",
            "ln",
            "local_time",
            "blue",
            "green",
            "red",
            "nir",
            "swir",
            "vari_green",
            "gli",
            "ndvi",
            "gndvi",
            "msavi",
            "emis",
            "ec_le",
            "ec_h",
            "ec_le_closed",
            "ec_g",
            "ec_rn-g",
        ]
    ]
