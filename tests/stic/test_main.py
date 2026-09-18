# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025 CESBIO / Centre National d'Etudes Spatiales

import datetime as dt
from typing import cast

import numpy as np
import pytest
import xarray as xr

from sweat.common.types import ETVar
from sweat.stic import main
from sweat.stic.registry import DEFAULT_VERSION, MODEL_REGISTRY


def setup_data(
    lst: tuple[float, float],
    albedo: tuple[float, float],
    temperature: tuple[float, float],
    dewpoint_temperature: tuple[float, float],
    rh: tuple[float, float],
    fcover: tuple[float, float],
    lai: tuple[float, float],
    emissivity: tuple[float, float],
    rn: tuple[float, float],
    ln: tuple[float, float],
    local_time: tuple[float, float],
    reflectances: tuple[float, float],
    valid: tuple[float, float],
    size: int = 4,
    seed: int = 0,
) -> xr.Dataset:
    """
    Generate data for tests
    """
    np.random.seed(seed)
    x = np.linspace(start=1.0, stop=2.0, num=size, dtype=np.float32)
    y = np.linspace(start=43.0, stop=44.0, num=size, dtype=np.float32)
    valid_arr = np.random.choice(
        [0, 1], size=(size, size), p=[valid[0], valid[1]]
    )
    lst_arr = np.random.uniform(low=lst[0], high=lst[1], size=(size, size))
    albedo_arr = np.random.uniform(
        low=albedo[0], high=albedo[1], size=(size, size)
    )
    temperature_arr = np.random.uniform(
        low=temperature[0], high=temperature[1], size=(size, size)
    )
    dewpoint_temperature_arr = np.random.uniform(
        low=dewpoint_temperature[0],
        high=dewpoint_temperature[1],
        size=(size, size),
    )
    rh_arr = np.random.uniform(low=rh[0], high=rh[1], size=(size, size))
    fcover_arr = np.random.uniform(
        low=fcover[0], high=fcover[1], size=(size, size)
    )
    lai_arr = np.random.uniform(low=lai[0], high=lai[1], size=(size, size))
    emissivity_arr = np.random.uniform(
        low=emissivity[0], high=emissivity[1], size=(size, size)
    )
    rn_arr = np.random.uniform(low=rn[0], high=rn[1], size=(size, size))
    ln_arr = np.random.uniform(low=ln[0], high=ln[1], size=(size, size))
    localtime_arr = np.random.uniform(
        low=local_time[0], high=local_time[1], size=(size, size)
    )
    blue_arr = np.random.uniform(
        low=reflectances[0], high=reflectances[1], size=(size, size)
    )
    green_arr = np.random.uniform(
        low=reflectances[0], high=reflectances[1], size=(size, size)
    )
    red_arr = np.random.uniform(
        low=reflectances[0], high=reflectances[1], size=(size, size)
    )
    nir_arr = np.random.uniform(
        low=reflectances[0], high=reflectances[1], size=(size, size)
    )
    swir_arr = np.random.uniform(
        low=reflectances[0], high=reflectances[1], size=(size, size)
    )
    data_vars = {
        ETVar.LST.value: (["y", "x"], lst_arr),
        ETVar.ALBEDO.value: (["y", "x"], albedo_arr),
        ETVar.TEMPERATURE.value: (["y", "x"], temperature_arr),
        ETVar.DEWPOINT_TEMPERATURE.value: (
            ["y", "x"],
            dewpoint_temperature_arr,
        ),
        ETVar.RH.value: (["y", "x"], rh_arr),
        ETVar.FCOVER.value: (["y", "x"], fcover_arr),
        ETVar.LAI.value: (["y", "x"], lai_arr),
        ETVar.EMISSIVITY.value: (["y", "x"], emissivity_arr),
        ETVar.NET_RADIATION.value: (["y", "x"], rn_arr),
        ETVar.LONGWAVE_NET_RADIATION.value: (["y", "x"], ln_arr),
        ETVar.LOCAL_TIME.value: (["y", "x"], localtime_arr),
        ETVar.BLUE.value: (["y", "x"], blue_arr),
        ETVar.GREEN.value: (["y", "x"], green_arr),
        ETVar.RED.value: (["y", "x"], red_arr),
        ETVar.NIR.value: (["y", "x"], nir_arr),
        ETVar.SWIR.value: (["y", "x"], swir_arr),
        "valid": (["y", "x"], valid_arr),
    }
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "y": ("y", y),
            "x": ("x", x),
        },
        attrs={
            "description": "Test data",
            "date": dt.datetime(2025, 7, 24, 12, 30, tzinfo=dt.UTC),
            "crs": 4326,
        },
    )


@pytest.mark.functional
@pytest.mark.parametrize(
    "version",
    [
        pytest.param(None, id="default"),
        pytest.param("1.3", id="1.3"),
        pytest.param("1.4", id="1.4"),
    ],
)
def test_prepare(version):
    """
    Test function for STIC prepare function
    """
    data = xr.Dataset(
        data_vars={
            "lst": (["y", "x"], np.array([[293, np.nan], [293, 293]])),
            "albedo": (["y", "x"], np.array([[0.5, 0.5], [np.nan, 0.5]])),
            "ta": (["y", "x"], np.full((2, 2), 290)),
            "tdp": (["y", "x"], np.full((2, 2), 285)),
            "fcover": (["y", "x"], np.full((2, 2), 0.6)),
            "lai": (["y", "x"], np.full((2, 2), 2)),
            "emis": (["y", "x"], np.full((2, 2), 0.8)),
            "rsd": (["y", "x"], np.full((2, 2), 400)),
            "rld": (["y", "x"], np.full((2, 2), 200)),
            "blue": (["y", "x"], np.full((2, 2), 0.1)),
            "green": (["y", "x"], np.full((2, 2), 0.6)),
            "red": (["y", "x"], np.full((2, 2), 0.2)),
            "nir": (["y", "x"], np.full((2, 2), 0.5)),
            "swir": (["y", "x"], np.full((2, 2), 0.4)),
        },
        coords={
            "y": ("y", np.array([40.0, 40.1])),
            "x": ("x", np.array([0, 0.1])),
        },
        attrs={
            "description": "Test data",
            "date": dt.datetime.now(tz=dt.UTC),
            "crs": 4326,
        },
    )
    res = main.prepare(data=data, version=version)
    if version is None:
        version = DEFAULT_VERSION
    spec = MODEL_REGISTRY[version]
    assert res
    for v in spec.inputs:
        assert v in res.data_vars


@pytest.mark.functional
@pytest.mark.parametrize(
    "version",
    [None, "1.3", "1.4"],
)
def test_prepare_exc(version):
    """
    Test function for STIC prepare function (with exception)
    """
    data = xr.Dataset(
        data_vars={
            "albedo": (["y", "x"], np.array([[0.5, 0.5], [np.nan, 0.5]])),
            "ta": (["y", "x"], np.full((2, 2), 290)),
            "tdp": (["y", "x"], np.full((2, 2), 285)),
            "fcover": (["y", "x"], np.full((2, 2), 0.6)),
            "lai": (["y", "x"], np.full((2, 2), 2)),
            "emis": (["y", "x"], np.full((2, 2), 0.8)),
            "rsd": (["y", "x"], np.full((2, 2), 400)),
            "rld": (["y", "x"], np.full((2, 2), -150)),
        },
        coords={
            "y": ("y", np.array([0, 1])),
            "x": ("x", np.array([0, 1])),
        },
        attrs={"description": "Test data"},
    )
    with pytest.raises(
        KeyError, match="Variable lst is missing in the dataset"
    ):
        main.prepare(data=data, version=version)


@pytest.mark.functional
@pytest.mark.parametrize(
    ("selected_radiation", "rsd_expected", "rld_expected"),
    [
        ("msg", "rsd_msg", "rld_msg"),
        (None, "rsd", "rld"),
    ],
)
def test_prepare_with_radiation_selection(
    selected_radiation, rsd_expected, rld_expected
):
    """
    Test function for STIC prepare function with radiation selection
    """
    data = xr.Dataset(
        data_vars={
            "lst": (["y", "x"], np.array([[293, np.nan], [293, 293]])),
            "albedo": (["y", "x"], np.array([[0.5, 0.5], [np.nan, 0.5]])),
            "ta": (["y", "x"], np.full((2, 2), 290)),
            "tdp": (["y", "x"], np.full((2, 2), 285)),
            "fcover": (["y", "x"], np.full((2, 2), 0.6)),
            "lai": (["y", "x"], np.full((2, 2), 2)),
            "emis": (["y", "x"], np.full((2, 2), 0.8)),
            "rsd": (["y", "x"], np.full((2, 2), 400)),
            "rsd_msg": (["y", "x"], np.full((2, 2), 500)),
            "rld": (["y", "x"], np.full((2, 2), -100)),
            "rld_msg": (["y", "x"], np.full((2, 2), -200)),
            "blue": (["y", "x"], np.full((2, 2), 0.1)),
            "green": (["y", "x"], np.full((2, 2), 0.6)),
            "red": (["y", "x"], np.full((2, 2), 0.2)),
            "nir": (["y", "x"], np.full((2, 2), 0.5)),
            "swir": (["y", "x"], np.full((2, 2), 0.4)),
        },
        coords={
            "y": ("y", np.array([40.0, 40.1])),
            "x": ("x", np.array([0, 0.1])),
        },
        attrs={
            "description": "Test data",
            "date": dt.datetime.now(tz=dt.UTC),
            "crs": 4326,
        },
    )
    res = main.prepare(data=data, selected_radiation=selected_radiation)
    assert rsd_expected in res.data_vars
    assert rld_expected in res.data_vars


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "ts",
        "ta",
        "td",
        "rh",
        "fc",
        "lai",
        "rn",
        "ln",
        "nir",
        "swir",
        "vari_green",
        "gli",
        "ndvi",
        "gndvi",
        "msavi",
        "emis",
        "local_time",
        "threshold",
        "nb_steps",
        "version",
        "le_expected",
        "ef_expected",
    ),
    [
        pytest.param(
            np.full((2, 2), 34.35, dtype=np.float32),
            np.full((2, 2), 31.575, dtype=np.float32),
            np.full((2, 2), 20.51, dtype=np.float32),
            np.full((2, 2), 52.0, dtype=np.float32),
            np.full((2, 2), 0.54, dtype=np.float32),
            np.full((2, 2), 1.61, dtype=np.float32),
            np.full((2, 2), 529.72, dtype=np.float32),
            np.full((2, 2), -89.6, dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            np.full((2, 2), 38440, dtype=np.float32),
            0.01,
            15,
            "1.3",
            np.full((2, 2), 241.79, dtype=np.float32),
            np.full((2, 2), 0.49, dtype=np.float32),
            id="1.3",
        ),
        pytest.param(
            np.full((2, 2), 34.35, dtype=np.float32),
            np.full((2, 2), 31.575, dtype=np.float32),
            np.full((2, 2), 20.51, dtype=np.float32),
            np.full((2, 2), 52.0, dtype=np.float32),
            np.full((2, 2), 0.54, dtype=np.float32),
            np.full((2, 2), 1.61, dtype=np.float32),
            np.full((2, 2), 529.72, dtype=np.float32),
            np.full((2, 2), -89.6, dtype=np.float32),
            np.full((2, 2), 0.336, dtype=np.float32),
            np.full((2, 2), 0.226, dtype=np.float32),
            np.full((2, 2), 0.112, dtype=np.float32),
            np.full((2, 2), 0.0, dtype=np.float32),
            np.full((2, 2), 0.5, dtype=np.float32),
            np.full((2, 2), 0.46, dtype=np.float32),
            np.full((2, 2), 0.02, dtype=np.float32),
            np.full((2, 2), 0.98, dtype=np.float32),
            np.full((2, 2), 38440, dtype=np.float32),
            0.01,
            15,
            "1.4",
            np.full((2, 2), 422.30, dtype=np.float32),
            np.full((2, 2), 0.83, dtype=np.float32),
            id="1.4",
        ),
    ],
)
def test_run(
    ts,
    ta,
    td,
    rh,
    fc,
    lai,
    rn,
    ln,
    nir,
    swir,
    vari_green,
    gli,
    ndvi,
    gndvi,
    msavi,
    emis,
    local_time,
    threshold,
    nb_steps,
    version,
    le_expected,
    ef_expected,
) -> None:
    """
    Test function for STIC run function
    """
    data = xr.Dataset(
        data_vars={
            ETVar.LST.value: (["y", "x"], ts),
            ETVar.TEMPERATURE.value: (["y", "x"], ta),
            ETVar.DEWPOINT_TEMPERATURE.value: (["y", "x"], td),
            ETVar.RH.value: (["y", "x"], rh),
            ETVar.FCOVER.value: (["y", "x"], fc),
            ETVar.LAI.value: (["y", "x"], lai),
            ETVar.NET_RADIATION.value: (["y", "x"], rn),
            ETVar.LONGWAVE_NET_RADIATION.value: (["y", "x"], ln),
            ETVar.LOCAL_TIME.value: (["y", "x"], local_time),
            ETVar.NIR.value: (["y", "x"], nir),
            ETVar.SWIR.value: (["y", "x"], swir),
            ETVar.VARI.value: (["y", "x"], vari_green),
            ETVar.GLI.value: (["y", "x"], gli),
            ETVar.NDVI.value: (["y", "x"], ndvi),
            ETVar.GNDVI.value: (["y", "x"], gndvi),
            ETVar.MSAVI.value: (["y", "x"], msavi),
            ETVar.EMISSIVITY.value: (["y", "x"], emis),
        },
        coords={
            "y": ("y", np.array([0, 1])),
            "x": ("x", np.array([0, 1])),
        },
        attrs={"description": "Test data"},
    )
    res = main.run(
        data=data, threshold=threshold, nb_steps=nb_steps, version=version
    )
    np.testing.assert_almost_equal(res["le"], le_expected, decimal=2)
    np.testing.assert_almost_equal(res["ef"], ef_expected, decimal=2)
    assert sorted(cast(list[str], res.data_vars)) == sorted(
        [
            ETVar.LE.value,
            ETVar.ET.value,
            ETVar.EF.value,
            ETVar.EVAPORATION_INTERCEPTION.value,
            ETVar.EVAPORATION_SOIL.value,
            ETVar.TRANSPIRATION.value,
            ETVar.UNCERTAINTY_LE.value,
            ETVar.UNCERTAINTY_ET.value,
            ETVar.UNCERTAINTY_EF.value,
            ETVar.VALID.value,
            ETVar.FLAGS.value,
        ]
    )
