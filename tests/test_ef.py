#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import pytest
import numpy as np
import numpy.typing as npt
import xarray as xr

from evaspa.ef import create_efmodel, EFModelError, EFConfigError, check_efmodel


def setup_data(
    dry: tuple[float, float],
    wet: tuple[float, float],
    albedo: tuple[float, float],
    valid: tuple[float, float],
    fcover: tuple[float, float] | None = None,
    size: int = 125,
    seed: int = 0,
) -> xr.Dataset:
    """
    Generate data for tests
    """

    def dry_edge(v: npt.ArrayLike) -> npt.NDArray:
        return dry[1] * np.array(v) + dry[0]

    def wet_edge(v: npt.ArrayLike) -> npt.NDArray:
        return wet[1] * np.array(v) + wet[0]

    np.random.seed(seed)
    lon = np.linspace(start=1.0, stop=2.0, num=size, dtype=np.float32)
    lat = np.linspace(start=43.0, stop=44.0, num=size, dtype=np.float32)
    valid_arr = np.random.choice([0, 1], size=(size, size), p=[valid[0], valid[1]])
    albedo_arr = np.random.uniform(low=albedo[0], high=albedo[1], size=(size, size))
    lst_func = np.vectorize(lambda x: np.random.uniform(wet_edge(x), dry_edge(x)))
    lst_arr = lst_func(albedo_arr)
    if fcover is not None:
        fcover_arr = np.random.uniform(low=fcover[0], high=fcover[1], size=(size, size))
        lst_func = np.vectorize(
            lambda x, y: np.random.uniform(
                np.maximum(wet_edge(x), wet_edge(y)),
                np.minimum(dry_edge(x), dry_edge(y)),
            )
        )
        lst_arr = lst_func(fcover_arr, albedo_arr)
    ef_albedo_arr = (dry_edge(albedo_arr) - lst_arr) / (
        dry_edge(albedo_arr) - wet_edge(albedo_arr)
    )
    data_vars = dict(
        albedo=(["lon", "lat"], albedo_arr),
        lst=(["lon", "lat"], lst_arr),
        valid=(["lon", "lat"], valid_arr),
        ef_albedo=(["lon", "lat"], ef_albedo_arr),
    )
    if fcover is not None:
        ef_fcover_arr = (dry_edge(fcover_arr) - lst_arr) / (
            dry_edge(fcover_arr) - wet_edge(fcover_arr)
        )
        data_vars["fcover"] = (["lon", "lat"], fcover_arr)
        data_vars["ef_fcover"] = (["lon", "lat"], ef_fcover_arr)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            lon=("loc", lon),
            lat=("loc", lat),
        ),
        attrs=dict(description="Test data"),
    )
    return ds


@pytest.mark.parametrize(
    "config,check_ef",
    [
        pytest.param(
            {
                "name": "model",
                "dry_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "size",
                        "interval_nb": 20,
                        "percentile": [98, 100],
                        "selection": "median",
                    },
                },
                "wet_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "size",
                        "interval_nb": 20,
                        "percentile": [0, 2],
                        "selection": "median",
                    },
                },
                "var": "albedo",
            },
            True,
        ),
        pytest.param(
            {
                "name": "model",
                "dry_edge": {
                    "type": "FlatEdge",
                    "config": {
                        "selection": "max",
                    },
                },
                "wet_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "size",
                        "interval_nb": 20,
                        "percentile": [0, 2],
                        "selection": "median",
                    },
                },
                "var": "albedo",
            },
            False,
        ),
        pytest.param(
            {
                "name": "model",
                "dry_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "size",
                        "interval_nb": 20,
                        "percentile": [98, 100],
                        "selection": "median",
                    },
                },
                "wet_edge": {
                    "type": "FlatEdge",
                    "config": {
                        "selection": "min",
                    },
                },
                "var": "albedo",
            },
            False,
        ),
        pytest.param(
            {
                "name": "model",
                "dry_edge": {
                    "type": "FlatEdge",
                    "config": {
                        "selection": "max",
                    },
                },
                "wet_edge": {
                    "type": "FlatEdge",
                    "config": {
                        "selection": "min",
                    },
                },
                "var": "albedo",
            },
            False,
        ),
    ],
)
def test_create_model(config, check_ef) -> None:
    """
    Test model creation and computation
    """
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6), valid=(0.0, 1.0), dry=(330.0, -10.0), wet=(300.0, 15.0)
    )
    # Create model
    model = create_efmodel(config)
    model.fit(data)
    np.testing.assert_allclose(model.tdry(0.0), 330.0, atol=5)
    np.testing.assert_allclose(model.twet(0.0), 300.0, atol=5)
    if check_ef:
        np.testing.assert_allclose(model.compute(data), data["ef_albedo"], atol=0.3)


@pytest.mark.parametrize(
    "config",
    [
        {
            "name": "model",
            "dry_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [0, 2],
                    "selection": "median",
                },
            },
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "foo",
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [0, 2],
                    "selection": "median",
                },
            },
            "var": "foo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "foo",
                    "interval_nb": 20,
                    "percentile": [98, 100],
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "min",
                },
            },
            "var": "albedo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "max",
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "foo",
                },
            },
            "var": "albedo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "max",
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "min",
                },
            },
        },
    ],
)
def test_create_model_error(config) -> None:
    """
    Test model creation with error
    """
    with pytest.raises((EFConfigError, EFModelError)):
        create_efmodel(config)


@pytest.mark.parametrize(
    "config",
    [
        {
            "name": "model",
            "dry_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [98, 100],
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [0, 2],
                    "selection": "median",
                },
            },
            "var": "albedo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "max",
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [0, 2],
                    "selection": "median",
                },
            },
            "var": "albedo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [98, 100],
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "min",
                },
            },
            "var": "albedo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "max",
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "min",
                },
            },
            "var": "albedo",
        },
    ],
)
def test_check_model_config(config) -> None:
    """
    Test check model configuration
    """
    check_efmodel(config)


@pytest.mark.parametrize(
    "config",
    [
        {
            "dry_edge": {
                "type": "LinearEdge",
                "config": {},
            },
        },
        {
            "dry": {
                "type": "FlatEdge",
                "config": {
                    "selection": "max",
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [0, 2],
                    "selection": "median",
                },
            },
        },
        {
            "dry_edge": {
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": [98, 100],
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
                "config": {
                    "selection": "min",
                },
            },
        },
        {
            "dry_edge": {
                "type": "MaxEdge",
                "config": {
                    "selection": "max",
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
            },
        },
    ],
)
def test_check_config_model_error(config) -> None:
    """
    Test check model config with error
    """
    with pytest.raises(EFConfigError):
        check_efmodel(config)
