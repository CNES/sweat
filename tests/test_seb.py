# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from evaspa import seb


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
    Create a rondom dataset
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


def test_stefan_boltzmann_constant() -> None:
    """
    Stefan Boltzmann constant
    """
    np.testing.assert_approx_equal(seb.CST_SB, 5.670374419e-8, significant=7)


@pytest.mark.parametrize(
    ("lst", "emis", "albedo", "rsd", "rld", "expected"),
    [
        pytest.param(0.0, 0.0, 0.0, 100.0, 0.0, 100.0),
        pytest.param(300.0, 1.0, 0.0, 0.0, 0.0, -seb.CST_SB * 300.0**4),
        pytest.param(0.0, 1.0, 1.0, 0.0, 50.0, 50.0),
        pytest.param(0.0, 1.0, 1.0, 100.0, 50.0, 50.0),
    ],
)
def test_compute_rn(lst, emis, albedo, rsd, rld, expected) -> None:
    """
    Test compute net radiation Rn
    """
    # Case 1: Black body
    rsd_arr = rsd * np.ones((2, 2))
    rld_arr = rld * np.ones((2, 2))
    albedo_arr = albedo * np.ones((2, 2))
    emis_arr = emis * np.ones((2, 2))
    lst_arr = lst * np.ones((2, 2))
    rn = seb._compute_rn(lst_arr, emis_arr, albedo_arr, rsd_arr, rld_arr)  # noqa: SLF001
    ref = expected * np.ones((2, 2))
    np.testing.assert_allclose(rn, ref)


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


def test_ratio_from_kustas() -> None:
    """
    Test G/Rn ratio (kustas method)
    """
    ndvi = 0.5 * np.ones((2, 2))
    ratio = seb._ratio_from_kustas(ndvi)  # noqa: SLF001
    ref = (0.4 - (0.33 * 0.5)) * np.ones((2, 2))
    np.testing.assert_allclose(ratio, ref)


def test_ratio_from_su() -> None:
    """
    Test G/Rn ratio (Su method)
    """
    fcover = 0.5 * np.ones((2, 2))
    ratio = seb._ratio_from_su(fcover)  # noqa: SLF001
    ref = (0.05 + (1.0 - 0.5) * (0.315 - 0.05)) * np.ones((2, 2))
    np.testing.assert_allclose(ratio, ref)


def test_ratio_from_choudhury() -> None:
    """
    Test compute G flux (Choudhury method)
    """
    lai = 2 * np.ones((2, 2))
    ratio = seb._ratio_from_choudhury(lai)  # noqa: SLF001
    ref = 0.3 * np.exp(-1.0) * np.ones((2, 2))
    np.testing.assert_allclose(ratio, ref, atol=0.0001)


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


def test_create_ratio_unknown_model(caplog) -> None:
    """
    Test create G/Rn dataset with unknown model
    """
    data = setup_data(nb=1, size=2)
    seb.create_ratio(data, models=["foo"])  # type: ignore
    assert "Unknown model for G/Rn ratio: foo" in caplog.text


def test_create_ratio_data_missing(caplog) -> None:
    """
    Test create G/Rn dataset with missing data
    """
    data = setup_data(nb=1, size=2)
    data = data.drop_vars("ndvi")
    seb.create_ratio(data, models=[seb.RatioModel.KUSTAS])
    assert "Data missing for kustas model" in caplog.text


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
    Test FilterConfig
    """
    assert seb.SEBConfig.model_validate(config)


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
    seb.run(data, ef, **config)


def test_compute_et_from_le():
    """
    Test method for computing ET from LE
    """
    le = 100 * np.ones((2, 2))
    ref = np.ones((2, 2)) * 100 / seb.LATENT_HEAT_VAPORIZATION
    et = seb._compute_et_from_le(le)  # noqa: SLF001
    np.testing.assert_allclose(et, ref)
