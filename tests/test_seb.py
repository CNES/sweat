#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import pytest
import numpy as np
import xarray as xr

from evaspa.seb import (
    CST_SB,
    SEBConfig,
    compute_g_kustas,
    compute_g_su,
    compute_g_choudhury,
    compute_rn,
    compute_le,
    create_net_radiation,
    create_gflux,
    create_le,
    run,
)


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
    valid_arr = np.random.choice([0, 1], size=(size, size), p=[1 - valid, valid])
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
    data_vars = dict(
        albedo=(["lat", "lon"], albedo_arr),
        lst=(["lat", "lon"], lst_arr),
        emis=(["lat", "lon"], emis_arr),
        ndvi=(["lat", "lon"], ndvi_arr),
        lai=(["lat", "lon"], lai_arr),
        fcover=(["lat", "lon"], fcover_arr),
        valid=(["lat", "lon"], valid_arr),
    )
    data_vars["rsd"] = (["lat", "lon"], rsd_arr[0])
    data_vars["rld"] = (["lat", "lon"], rld_arr[0])
    for i in np.arange(1, nb):
        data_vars[f"rsd_{i}"] = (["lat", "lon"], rsd_arr[i])
        data_vars[f"rld_{i}"] = (["lat", "lon"], rld_arr[i])
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            lon=("lon", lon),
            lat=("lat", lat),
        ),
        attrs=dict(description="Test data"),
    )
    return ds


def setup_dataset(
    vars: list[str], min: float = 0.0, max: float = 1.0, size: int = 125, seed: int = 0
):
    """
    Create a rondom dataset
    """
    assert len(vars) > 0
    np.random.seed(seed)
    lon = np.linspace(start=1.0, stop=2.0, num=size, dtype=np.float32)
    lat = np.linspace(start=43.0, stop=44.0, num=size, dtype=np.float32)
    data_vars = {}
    for var in vars:
        data_vars[var] = (
            ["lat", "lon"],
            np.random.uniform(low=min, high=max, size=(size, size)),
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            lon=("lon", lon),
            lat=("lat", lat),
        ),
        attrs=dict(description="Test data"),
    )


def test_stefan_boltzmann_constant() -> None:
    """
    Stefan Boltzmann constant
    """
    np.testing.assert_approx_equal(CST_SB, 5.670374419e-8, significant=7)


def test_compute_rn() -> None:
    """
    Test compute net radiation Rn
    #TODO test other case
    """
    # Case 1: Black body
    rsd = np.ones((2, 2))
    rld = np.zeros((2, 2))
    albedo = np.ones((2, 2))
    emis = np.ones((2, 2))
    lst = np.ones((2, 2)) * 300
    rn = compute_rn(lst, emis, albedo, rsd, rld)
    ref = -CST_SB * (300**4) * np.ones((2, 2))
    np.testing.assert_allclose(rn, ref)


@pytest.mark.parametrize(
    "params,expected",
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
    rn = create_net_radiation(data)
    assert len(rn.data_vars) == expected


def test_compute_g_kustas() -> None:
    """
    Test compute G flux (kustas method)
    """
    rn = np.ones((2, 2))
    ndvi = 0.5 * np.ones((2, 2))
    g = compute_g_kustas(rn, ndvi)
    ref = 0.235 * np.ones((2, 2))
    np.testing.assert_allclose(g, ref)


def test_compute_g_su() -> None:
    """
    Test compute G flux (Su method)
    """
    rn = np.ones((2, 2))
    fcover = 0.5 * np.ones((2, 2))
    g = compute_g_su(rn, fcover)
    ref = 0.1825 * np.ones((2, 2))
    np.testing.assert_allclose(g, ref)


def test_compute_g_choudhury() -> None:
    """
    Test compute G flux (Choudhury method)
    """
    rn = np.ones((2, 2))
    lai = 2 * np.ones((2, 2))
    g = compute_g_choudhury(rn, lai)
    ref = 0.11036 * np.ones((2, 2))
    np.testing.assert_allclose(g, ref, atol=0.0001)


def test_create_gflux() -> None:
    """
    Test create G dataset
    """
    data = setup_data(nb=1, size=2)
    rn = create_net_radiation(data)
    gflux = create_gflux(data, rn)
    assert len(gflux.data_vars) == 3


def test_compute_le() -> None:
    """
    Test compute LE
    """
    ef = 0.5 * np.ones((2, 2))
    rn = 200 * np.ones((2, 2))
    g = rn * 0.5
    ref = 50 * np.ones((2, 2))
    le = compute_le(ef, rn, g)
    np.testing.assert_allclose(le, ref, atol=0.0001)


def test_create_le() -> None:
    """
    Test create LE dataset
    """
    ef = setup_dataset(["m1", "m2", "m3"])
    rn = setup_dataset(["rn1", "rn2"], min=200, max=250)
    gflux = setup_dataset(["g1", "g2", "g3"], min=100, max=150)
    le = create_le(ef, rn, gflux)
    assert len(le.data_vars) == 18


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"use_topo": True},
        {"g_models": ["su", "kustas"]},
        {"use_topo": True, "g_models": ["su", "kustas"]},
    ],
)
def test_sebconfig(config) -> None:
    """
    Test FilterConfig
    """
    assert SEBConfig.model_validate(config)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"use_topo": True},
        {"g_models": ["su", "kustas"]},
        {"use_topo": True, "g_models": ["su", "kustas"]},
    ],
)
def test_run(config):
    """
    Test run method
    """
    data = setup_data(nb=1)
    ef = setup_dataset(["m1", "m2", "m3"])
    run(data, ef, **config)
