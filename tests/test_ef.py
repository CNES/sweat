#!/usr/bin/env python
# coding: utf8
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import pytest
import numpy as np
import numpy.typing as npt
import xarray as xr

from evaspa.ef import (
    EFModel,
    EFModelError,
    EFConfigError,
    MergeMethod,
    check_variability,
    initialize,
    compute,
    select,
    merge,
    run,
)


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
            lon=("lon", lon),
            lat=("lat", lat),
        ),
        attrs=dict(description="Test data"),
    )
    return ds


def setup_models() -> list[EFModel]:
    """
    Setup a model list
    """
    config_models = [
        {
            "name": "model1",
            "dry_edge": {
                "type": "LinearEdge",
                "config": {
                    "percentile": (98, 100),
                    "interval_type": "size",
                    "interval_nb": 20,
                    "selection": "median",
                    "coeffs": (np.inf, np.inf),
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "percentile": (0, 2),
                    "interval_type": "size",
                    "interval_nb": 20,
                    "selection": "median",
                    "coeffs": (np.inf, np.inf),
                },
            },
            "var": "fcover",
        },
        {
            "name": "model2",
            "dry_edge": {
                "type": "LinearEdge",
                "config": {
                    "percentile": (95, 100),
                    "interval_type": "density",
                    "interval_nb": 20,
                    "selection": "median",
                    "coeffs": (np.inf, np.inf),
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
                "config": {"selection": "min", "value": np.inf},
            },
            "var": "albedo",
        },
    ]
    return [EFModel.create(cfg) for cfg in config_models]


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
    model = EFModel.create(config)
    model.fit(data)
    np.testing.assert_allclose(model.tdry(0.0), 330.0, atol=5)
    np.testing.assert_allclose(model.twet(0.0), 300.0, atol=5)
    if check_ef:
        xr.testing.assert_allclose(model.compute(data), data["ef_albedo"], atol=0.3)


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
        EFModel.create(config)


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
    EFModel.check(config)


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
        EFModel.check(config)


@pytest.mark.parametrize(
    "lst,mask,expected",
    [
        pytest.param(np.ones((100, 100)), None, False),
        pytest.param(np.random.normal(10.0, 5.0, (100, 100)), None, True),
        pytest.param(np.random.normal(0.0, 5.0, (100, 100)), None, True),
        pytest.param(np.random.normal(15.0, 1.0, (100, 100)), None, False),
        pytest.param(
            np.random.normal(10.0, 1.0, (100, 100)),
            np.random.choice([0, 1], (100, 100)),
            False,
        ),
    ],
)
def test_check_variability(lst, mask, expected) -> None:
    """
    Test check variability function
    """
    assert check_variability(lst=lst, mask=mask) == expected


@pytest.mark.parametrize(
    "config,expected",
    [
        pytest.param(
            {
                "models": [
                    {
                        "name": "model1",
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
                        "var": "fcover",
                    },
                    {
                        "name": "model2",
                        "dry_edge": {
                            "type": "LinearEdge",
                            "config": {
                                "interval_type": "density",
                                "interval_nb": 20,
                                "percentile": [95, 100],
                                "selection": "median",
                            },
                        },
                        "wet_edge": {
                            "type": "FlatEdge",
                            "config": {"selection": "min"},
                        },
                        "var": "albedo",
                    },
                ],
                "options": {
                    "selection": False,
                    "merging": "mean",
                },
            },
            2,
        ),
        pytest.param(
            {
                "models": "default_evaspa",
                "options": {
                    "selection": False,
                    "merging": "mean",
                },
            },
            2,
        ),
    ],
)
def test_initialize(config, expected) -> None:
    """
    Test initialize function
    """
    models, options = initialize(config)
    assert len(models) == expected
    assert not options["selection"]
    assert options["merging"].value == "mean"


def test_compute() -> None:
    """
    Test compute function
    """
    # Generate models
    models = setup_models()
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        fcover=(0, 1.0),
        valid=(0.0, 1.0),
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    # Compute EF
    ef = compute(models, data)
    assert ef
    assert len(ef.data_vars) == 2


def test_select() -> None:
    """
    Test select function
    """
    ef = xr.Dataset(
        data_vars=dict(
            model1=(["y", "x"], np.random.normal(0.5, 0.2, (100, 100))),
            model2=(["y", "x"], np.random.normal(0.5, 0.24, (100, 100))),
        ),
        coords=dict(
            y=("y", np.linspace(0, 99, num=100)),
            x=("x", np.linspace(0, 99, num=100)),
        ),
        attrs=dict(description="EF models"),
    )
    selected = select(ef)
    xr.testing.assert_identical(ef, selected)


@pytest.mark.parametrize(
    "keep,method,expected",
    [
        pytest.param(True, MergeMethod.MEAN, 0.4),
        pytest.param(False, MergeMethod.MEAN, 0.4),
    ],
)
def test_merge(keep, method, expected) -> None:
    """
    Test merge function
    """
    ef = xr.Dataset(
        data_vars=dict(
            model1=(["y", "x"], np.random.normal(0.5, 0.1, (100, 100))),
            model2=(["y", "x"], np.random.normal(0.3, 0.1, (100, 100))),
        ),
        coords=dict(
            y=("y", np.linspace(0, 99, num=100)),
            x=("x", np.linspace(0, 99, num=100)),
        ),
        attrs=dict(description="EF models"),
    )
    merged = merge(ef, keep=keep, method=method)
    expected_size = 1 if not keep else (1 + len(ef.data_vars))
    assert len(merged.data_vars) == expected_size
    np.testing.assert_almost_equal(merged["ef"].mean(), expected, decimal=1)


@pytest.mark.parametrize(
    "options",
    [
        {
            "selection": False,
            "merging": MergeMethod.MEAN,
        },
        {
            "selection": True,
            "merging": MergeMethod.MEAN,
        },
        {
            "selection": False,
            "merging": MergeMethod.MEAN,
        },
    ],
)
def test_run(options) -> None:
    """
    Test run function
    """
    # Generate models
    models = setup_models()
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        fcover=(0, 1.0),
        valid=(0.0, 1.0),
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    run(models, data, **options)


def test_all() -> None:
    """
    Test complete
    """
    # Configuration
    config = {
        "models": [
            {
                "name": "model1",
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
                "var": "fcover",
            },
            {
                "name": "model2",
                "dry_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "density",
                        "interval_nb": 20,
                        "percentile": [95, 100],
                        "selection": "median",
                    },
                },
                "wet_edge": {
                    "type": "FlatEdge",
                    "config": {"selection": "min"},
                },
                "var": "albedo",
            },
        ],
        "options": {
            "selection": False,
            "merging": "mean",
        },
    }
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        fcover=(0, 1.0),
        valid=(0.0, 1.0),
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    models, options = initialize(config)
    run(models, data)
