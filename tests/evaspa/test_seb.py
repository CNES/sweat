# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2024 CESBIO / Centre National d'Etudes Spatiales


import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from pyproj import CRS

from sweat.evaspa import seb


def setup_data(
    nb: int = 1,
    size: int = 125,
    seed: int = 0,
    valid: float = 0.9,
    nan: bool = False,
) -> xr.Dataset:
    """
    Generate data for tests
    """
    assert nb > 0
    np.random.seed(seed)
    lon = np.linspace(start=1.0, stop=2.0, num=size, dtype=np.float32)
    lat = np.linspace(start=43.0, stop=44.0, num=size, dtype=np.float32)
    albedo_arr = np.random.uniform(low=0, high=1.0, size=(size, size))
    lst_arr = np.random.uniform(low=290, high=310, size=(size, size))
    emis_arr = np.random.uniform(low=0, high=1.0, size=(size, size))
    ndvi_arr = np.random.uniform(low=-1.0, high=1.0, size=(size, size))
    lai_arr = np.random.uniform(low=0, high=10.0, size=(size, size))
    fcover_arr = np.random.uniform(low=0, high=1.0, size=(size, size))
    valid_arr = np.random.choice(
        [0, 1], size=(size, size), p=[1 - valid, valid]
    )
    if nan:
        lst_arr = np.where(valid_arr == 1, lst_arr, np.nan)
        emis_arr = np.where(valid_arr == 1, emis_arr, np.nan)
        lai_arr = np.where(valid_arr == 1, lai_arr, np.nan)
        ndvi_arr = np.where(valid_arr == 1, ndvi_arr, np.nan)
        fcover_arr = np.where(valid_arr == 1, fcover_arr, np.nan)
        albedo_arr = np.where(valid_arr == 1, albedo_arr, np.nan)
    rsd_arr = []
    rld_arr = []
    for i in np.arange(nb):
        rsd_tmp = (400 + i) * np.ones((size, size))
        rld_tmp = (200 + i) * np.ones((size, size))
        if nan:
            rsd_tmp = np.where(valid_arr == 1, rsd_tmp, np.nan)
            rld_tmp = np.where(valid_arr == 1, rld_tmp, np.nan)
        rsd_arr.append(rsd_tmp)
        rld_arr.append(rld_tmp)
    data_vars = {
        "albedo": (["lat", "lon"], albedo_arr),
        "lst": (["lat", "lon"], lst_arr),
        "emis": (["lat", "lon"], emis_arr),
        "ndvi": (["lat", "lon"], ndvi_arr),
        "lai": (["lat", "lon"], lai_arr),
        "fcover": (["lat", "lon"], fcover_arr),
        "valid": (["lat", "lon"], valid_arr),
    }
    data_vars["rsd"] = (["lat", "lon"], rsd_arr[0])
    data_vars["rld"] = (["lat", "lon"], rld_arr[0])
    for i in np.arange(1, nb):
        data_vars[f"rsd_{i}"] = (["lat", "lon"], rsd_arr[i])
        data_vars[f"rld_{i}"] = (["lat", "lon"], rld_arr[i])
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "lon": ("lon", lon),
            "lat": ("lat", lat),
        },
        attrs={"description": "Test data", "crs": CRS(4326)},
    )


def setup_dataset(
    variables: list[str],
    min_value: float = 0.0,
    max_value: float = 1.0,
    size: int = 125,
    seed: int = 0,
) -> xr.Dataset:
    """
    Create a random dataset
    """
    assert len(variables) > 0
    np.random.seed(seed)
    lon = np.linspace(start=1.0, stop=2.0, num=size, dtype=np.float32)
    lat = np.linspace(start=43.0, stop=44.0, num=size, dtype=np.float32)
    data_vars = {}
    for var in variables:
        data_vars[var] = (
            ["lat", "lon"],
            np.random.uniform(low=min_value, high=max_value, size=(size, size)),
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "lon": ("lon", lon),
            "lat": ("lat", lat),
        },
        attrs={"description": "Test data"},
    )


def setup_array(data: npt.NDArray) -> xr.DataArray:
    """
    Create dataarray
    """
    return xr.DataArray(
        data=data,
        dims=["y", "x"],
        coords={
            "x": (["x"], np.linspace(-1.0, 1.0, num=data.shape[0])),
            "y": (["y"], np.linspace(42.0, 44.0, num=data.shape[1])),
        },
    )


@pytest.mark.functional
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param({"nb": 1}, 1),
        pytest.param({"nb": 3}, 3),
    ],
)
def test_create_net_radiation(params, expected) -> None:
    """
    Test create net radiation dataset
    """
    data = setup_data(**params)
    rn = seb.create_net_radiation(data)
    assert len(rn.data_vars) == expected


@pytest.mark.unit
def test_ratio_from_kustas() -> None:
    """
    Test G/Rn ratio (kustas method)
    """
    ndvi = 0.5 * np.ones((2, 2))
    ratio = seb._ratio_from_kustas(ndvi)  # noqa: SLF001
    ref = (0.4 - (0.33 * 0.5)) * np.ones((2, 2))
    np.testing.assert_allclose(ratio, ref)


@pytest.mark.unit
def test_ratio_from_su() -> None:
    """
    Test G/Rn ratio (Su method)
    """
    fcover = 0.5 * np.ones((2, 2))
    ratio = seb._ratio_from_su(fcover)  # noqa: SLF001
    ref = (0.05 + (1.0 - 0.5) * (0.315 - 0.05)) * np.ones((2, 2))
    np.testing.assert_allclose(ratio, ref)


@pytest.mark.unit
def test_ratio_from_choudhury() -> None:
    """
    Test compute G flux (Choudhury method)
    """
    lai = 2 * np.ones((2, 2))
    ratio = seb._ratio_from_choudhury(lai)  # noqa: SLF001
    ref = 0.3 * np.exp(-1.0) * np.ones((2, 2))
    np.testing.assert_allclose(ratio, ref, atol=0.0001)


@pytest.mark.functional
@pytest.mark.parametrize(
    ("models", "expected"),
    [
        pytest.param([seb.RatioModel.SU], 1),
        pytest.param(seb.ALL_MODELS, len(seb.ALL_MODELS)),
        pytest.param(seb.DEFAULT_MODELS, len(seb.DEFAULT_MODELS)),
        pytest.param(None, len(seb.DEFAULT_MODELS)),
    ],
)
def test_create_ratio(models, expected) -> None:
    """
    Test create G/Rn dataset
    """
    data = setup_data(nb=1, size=2)
    if models is not None:
        ratio = seb.create_ratio(data, models=models)
    else:
        ratio = seb.create_ratio(data)
    assert len(ratio.data_vars) == expected


@pytest.mark.functional
def test_create_ratio_unknown_model(caplog) -> None:
    """
    Test create G/Rn dataset with unknown model
    """
    data = setup_data(nb=1, size=2)
    seb.create_ratio(data, models=["foo"])  # type: ignore
    assert "Unknown model for G/Rn ratio: foo" in caplog.text


@pytest.mark.functional
def test_create_ratio_data_missing(caplog) -> None:
    """
    Test create G/Rn dataset with missing data
    """
    data = setup_data(nb=1, size=2)
    data = data.drop_vars("ndvi")
    seb.create_ratio(data, models=[seb.RatioModel.KUSTAS])
    assert "Data missing for kustas model" in caplog.text


@pytest.mark.unit
def test_compute_le() -> None:
    """
    Test compute LE
    """
    ef = 0.5 * np.ones((2, 2))
    rn = 200 * np.ones((2, 2))
    ratio = 0.5
    ref = 50 * np.ones((2, 2))
    le = seb._compute_le(ef, rn, ratio)  # noqa: SLF001
    np.testing.assert_allclose(le, ref, atol=0.0001)


@pytest.mark.functional
def test_create_le() -> None:
    """
    Test create LE dataset
    """
    ef = setup_dataset(["ef1", "ef2", "ef3"])
    rn = setup_dataset(["rn1", "rn2"], min_value=200, max_value=250)
    ratio = setup_dataset(
        ["model1", "model2", "model3"], min_value=100, max_value=150
    )
    le = seb.create_le(ef, rn, ratio)
    assert len(le.data_vars) == 18
    assert sorted(le.data_vars) == sorted(  # type: ignore
        [
            "ef1_rn1_model1",
            "ef1_rn1_model2",
            "ef1_rn1_model3",
            "ef1_rn2_model1",
            "ef1_rn2_model2",
            "ef1_rn2_model3",
            "ef2_rn1_model1",
            "ef2_rn1_model2",
            "ef2_rn1_model3",
            "ef2_rn2_model1",
            "ef2_rn2_model2",
            "ef2_rn2_model3",
            "ef3_rn1_model1",
            "ef3_rn1_model2",
            "ef3_rn1_model3",
            "ef3_rn2_model1",
            "ef3_rn2_model2",
            "ef3_rn2_model3",
        ]
    )


@pytest.mark.functional
def test_create_le_with_masks() -> None:
    """
    Test create LE dataset with mask
    """
    ef = setup_dataset(["ef1", "ef2", "ef3"])
    ef["valid"] = xr.ones_like(ef["ef1"])
    ef["flags"] = xr.zeros_like(ef["ef1"])
    rn = setup_dataset(["rn1", "rn2"], min_value=200, max_value=250)
    ratio = setup_dataset(
        ["model1", "model2", "model3"], min_value=100, max_value=150
    )
    le = seb.create_le(ef, rn, ratio)
    assert len(le.data_vars) == 20
    assert sorted(le.data_vars) == sorted(  # type: ignore
        [
            "ef1_rn1_model1",
            "ef1_rn1_model2",
            "ef1_rn1_model3",
            "ef1_rn2_model1",
            "ef1_rn2_model2",
            "ef1_rn2_model3",
            "ef2_rn1_model1",
            "ef2_rn1_model2",
            "ef2_rn1_model3",
            "ef2_rn2_model1",
            "ef2_rn2_model2",
            "ef2_rn2_model3",
            "ef3_rn1_model1",
            "ef3_rn1_model2",
            "ef3_rn1_model3",
            "ef3_rn2_model1",
            "ef3_rn2_model2",
            "ef3_rn2_model3",
            "flags",
            "valid",
        ]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"use_topo": True},
        {"models": ["su", "kustas"]},
        {"use_topo": True, "models": ["su", "kustas"]},
    ],
)
def test_sebconfig(config) -> None:
    """
    Test SEBConfig
    """
    assert seb.SEBConfig.model_validate(config)


@pytest.mark.functional
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"use_topo": True},
        {"models": seb.DEFAULT_MODELS},
        {"use_topo": True, "models": seb.ALL_MODELS},
    ],
)
def test_run(config):
    """
    Test run method
    """
    data = setup_data(nb=1)
    ef = setup_dataset(["m1", "m2", "m3"])
    res, merged = seb.run(data, ef, **config)
    nb = len(config.get("models", seb.DEFAULT_MODELS))
    assert (
        len(res.data_vars) == nb * len(ef.data_vars) + 2
    )  # Number of combinations plus flags
    assert sorted(
        ["le", "uncertainty_le", "et", "uncertainty_et", "valid", "flags"]
    ) == sorted(merged.data_vars)
