# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

from __future__ import annotations

import glob
import json
import os

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

import sweat.evaspa.ef
from sweat.evaspa import edge
from sweat.evaspa.ef import (
    EFCheckConfig,
    EFConfig,
    EFConfigError,
    EFModel,
    EFModelError,
    EFOptionsConfig,
    MergeMethod,
    check_variability,
    compute,
    get_available_configuration,
    get_variables_from_models,
    initialize,
    run,
    select,
    update_efconfig,
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
    valid_arr = np.random.choice(
        [0, 1], size=(size, size), p=[valid[0], valid[1]]
    )
    albedo_arr = np.random.uniform(
        low=albedo[0], high=albedo[1], size=(size, size)
    )
    lst_func = np.vectorize(
        lambda x: np.random.uniform(wet_edge(x), dry_edge(x))
    )
    lst_arr = lst_func(albedo_arr)
    if fcover is not None:
        fcover_arr = np.random.uniform(
            low=fcover[0], high=fcover[1], size=(size, size)
        )
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
    ef_albedo_arr = np.clip(ef_albedo_arr, 0, 1)
    data_vars = {
        "albedo": (["lat", "lon"], albedo_arr),
        "lst": (["lat", "lon"], lst_arr),
        "valid": (["lat", "lon"], valid_arr),
        "ef_albedo": (["lat", "lon"], ef_albedo_arr),
    }
    if fcover is not None:
        ef_fcover_arr = (dry_edge(fcover_arr) - lst_arr) / (
            dry_edge(fcover_arr) - wet_edge(fcover_arr)
        )
        ef_fcover_arr = np.clip(ef_fcover_arr, 0, 1)
        data_vars["fcover"] = (["lat", "lon"], fcover_arr)
        data_vars["ef_fcover"] = (["lat", "lon"], ef_fcover_arr)
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "lon": ("lon", lon),
            "lat": ("lat", lat),
        },
        attrs={"description": "Test data"},
    )


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
                    "percentile": 2,
                    "interval_type": "size",
                    "interval_nb": 20,
                    "selection": "median",
                    "coeffs": (np.inf, np.inf),
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "percentile": 2,
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
                    "percentile": 5,
                    "interval_type": "density",
                    "interval_nb": 20,
                    "selection": "median",
                    "coeffs": (np.inf, np.inf),
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
                "config": {"value": np.inf},
            },
            "var": "albedo",
        },
    ]
    return [EFModel.create(cfg) for cfg in config_models]


@pytest.mark.unit
@pytest.mark.parametrize("config", ["default_evaspa", "trishna_evaspa"])
def test_update_efconfig(config):
    """
    Test update_efconfig method
    """
    assert update_efconfig(config)


@pytest.mark.unit
def test_update_efconfig_with_error():
    """
    Test update_efconfig method with error
    """
    with pytest.raises(OSError, match="No EVASPA configuration file"):
        update_efconfig("foo")


@pytest.mark.unit
def test_efmodel_fit() -> None:
    """
    Test EFModel.fit method
    """
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        valid=(0.0, 1.0),
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    # Create model
    model = EFModel(
        name="model1",
        wet_edge=edge.LinearEdge(
            position=edge.EdgePosition.BOTTOM,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(np.inf, np.inf),
        ),
        dry_edge=edge.LinearEdge(
            position=edge.EdgePosition.TOP,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(np.inf, np.inf),
        ),
        var="albedo",
    )
    model.fit(data)
    np.testing.assert_allclose(model.dry_edge.coeffs[1], 330.0, atol=5)  # type: ignore
    np.testing.assert_allclose(model.dry_edge.coeffs[0], -10, atol=5)  # type: ignore
    np.testing.assert_allclose(model.wet_edge.coeffs[1], 300.0, atol=5)  # type: ignore
    np.testing.assert_allclose(model.wet_edge.coeffs[0], 15, atol=5)  # type: ignore


@pytest.mark.unit
def test_efmodel_fit_with_mask() -> None:
    """
    Test EFModel.fit method with mask
    """
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        valid=(0.0, 1.0),
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    # Add mask
    data["mask"] = xr.where(data["lst"] >= 310, 1, 0)
    # Create model
    model = EFModel(
        name="model1",
        wet_edge=edge.LinearEdge(
            position=edge.EdgePosition.BOTTOM,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(np.inf, np.inf),
        ),
        dry_edge=edge.LinearEdge(
            position=edge.EdgePosition.TOP,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(np.inf, np.inf),
        ),
        var="albedo",
    )
    model.fit(data, mask=data["mask"])
    np.testing.assert_allclose(model.dry_edge.coeffs[1], 330.0, atol=5)  # type: ignore
    np.testing.assert_allclose(model.dry_edge.coeffs[0], -10, atol=5)  # type: ignore
    np.testing.assert_allclose(model.wet_edge.coeffs[1], 310.0, atol=5)  # type: ignore
    np.testing.assert_allclose(model.wet_edge.coeffs[0], 0, atol=5)  # type: ignore


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data_vars", "expected_error", "expected_msg"),
    [
        pytest.param(
            {
                "lst": (["y", "x"], np.ones((2, 2))),
            },
            EFModelError,
            "albedo not in the dataset",
        ),
        pytest.param(
            {
                "albedo": (["y", "x"], np.ones((2, 2))),
            },
            EFModelError,
            "lst not in the dataset",
        ),
    ],
)
def test_efmodel_fit_with_error(
    data_vars, expected_error, expected_msg
) -> None:
    """
    Test EFModel.fit method with error
    """
    # Generate data
    data = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": ("x", [0, 1]),
            "y": ("y", [0, 1]),
        },
        attrs={"description": "Test data"},
    )
    model = EFModel(
        name="model1",
        wet_edge=edge.LinearEdge(
            position=edge.EdgePosition.BOTTOM,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(np.inf, np.inf),
        ),
        dry_edge=edge.LinearEdge(
            position=edge.EdgePosition.TOP,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(np.inf, np.inf),
        ),
        var="albedo",
    )
    with pytest.raises(expected_error, match=expected_msg):
        model.fit(data)


@pytest.mark.unit
def test_efmodel_tdry() -> None:
    """
    Test EFModel.tdry method
    """
    # Create model
    model = EFModel(
        name="model1",
        wet_edge=edge.LinearEdge(
            position=edge.EdgePosition.BOTTOM,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(15, 300),
        ),
        dry_edge=edge.LinearEdge(
            position=edge.EdgePosition.TOP,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(-10, 310),
        ),
        var="albedo",
    )
    np.testing.assert_allclose(model.tdry(0.3), 307.0)
    np.testing.assert_allclose(
        model.tdry(np.array([0.2, 0.4])), np.array([308.0, 306.0])
    )


@pytest.mark.unit
def test_efmodel_twet() -> None:
    """
    Test EFModel.twet method
    """
    # Create model
    model = EFModel(
        name="model1",
        wet_edge=edge.LinearEdge(
            position=edge.EdgePosition.BOTTOM,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(15, 300),
        ),
        dry_edge=edge.LinearEdge(
            position=edge.EdgePosition.TOP,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(-10, 310),
        ),
        var="albedo",
    )
    np.testing.assert_allclose(model.twet(0.3), 304.5)
    np.testing.assert_allclose(
        model.twet(np.array([0.2, 0.4])), np.array([303.0, 306.0])
    )


@pytest.mark.unit
def test_efmodel_compute() -> None:
    """
    Test EFModel.compute method
    """
    # Create model
    model = EFModel(
        name="model1",
        wet_edge=edge.LinearEdge(
            position=edge.EdgePosition.BOTTOM,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(15, 300),
        ),
        dry_edge=edge.LinearEdge(
            position=edge.EdgePosition.TOP,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(-10, 330),
        ),
        var="albedo",
    )
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        valid=(0.0, 1.0),
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    # Compute EF
    ef = model.compute(data)
    np.testing.assert_allclose(ef, data["ef_albedo"])


@pytest.mark.unit
@pytest.mark.parametrize("valid", [(0.0, 1.0), (0.2, 0.8)])
@pytest.mark.unit
def test_efmodel_compute_with_mask(valid) -> None:
    """
    Test EFModel.compute method with mask
    """
    # Create model
    model = EFModel(
        name="model1",
        wet_edge=edge.LinearEdge(
            position=edge.EdgePosition.BOTTOM,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(15, 300),
        ),
        dry_edge=edge.LinearEdge(
            position=edge.EdgePosition.TOP,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(-10, 330),
        ),
        var="albedo",
    )
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        valid=valid,
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    # Compute EF
    ef = model.compute(data, mask=data["valid"])
    np.testing.assert_allclose(ef, data["ef_albedo"].where(data["valid"] == 1))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data_vars", "expected_error", "expected_msg"),
    [
        pytest.param(
            {
                "lst": (["y", "x"], np.ones((2, 2))),
            },
            EFModelError,
            "albedo not in the dataset",
        ),
        pytest.param(
            {
                "albedo": (["y", "x"], np.ones((2, 2))),
            },
            EFModelError,
            "lst not in the dataset",
        ),
    ],
)
def test_efmodel_compute_with_error(
    data_vars, expected_error, expected_msg
) -> None:
    """
    Test EFModel.compute method with error
    """
    # Generate data
    data = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": ("x", [0, 1]),
            "y": ("y", [0, 1]),
        },
        attrs={"description": "Test data"},
    )
    # Create model
    model = EFModel(
        name="model1",
        wet_edge=edge.LinearEdge(
            position=edge.EdgePosition.BOTTOM,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(15, 300),
        ),
        dry_edge=edge.LinearEdge(
            position=edge.EdgePosition.TOP,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(-10, 330),
        ),
        var="albedo",
    )
    # Compute EF
    with pytest.raises(expected_error, match=expected_msg):
        model.compute(data)


@pytest.mark.unit
def test_efmodel_to_dict() -> None:
    """
    Test EFModel.tdry method
    """
    # Create model
    model = EFModel(
        name="model1",
        wet_edge=edge.LinearEdge(
            position=edge.EdgePosition.BOTTOM,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(15, 300),
        ),
        dry_edge=edge.LinearEdge(
            position=edge.EdgePosition.TOP,
            interval_type=edge.IntervalType.SIZE,
            interval_nb=20,
            percentile=1.0,
            selection=edge.SelectionMethod.MEDIAN,
            coeffs=(-10, 310),
        ),
        var="albedo",
    )
    assert model.to_dict()


@pytest.mark.unit
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
                    "percentile": 2,
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": 2,
                    "selection": "median",
                },
            },
            "var": "albedo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "FlatEdge",
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": 2,
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
                    "percentile": 2,
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
            },
            "var": "albedo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "FlatEdge",
            },
            "wet_edge": {
                "type": "FlatEdge",
            },
            "var": "albedo",
        },
    ],
)
def test_create_model(config) -> None:
    """
    Test model creation
    """
    # Create model
    EFModel.create(config)


@pytest.mark.unit
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
                    "percentile": 2,
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
                    "percentile": 2,
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
                    "percentile": 2,
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
        },
    ],
)
def test_create_model_error(config) -> None:
    """
    Test model creation with error
    """
    with pytest.raises((EFConfigError, EFModelError)):
        EFModel.create(config)


@pytest.mark.unit
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
                    "percentile": 2,
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": 2,
                    "selection": "median",
                },
            },
            "var": "albedo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "FlatEdge",
            },
            "wet_edge": {
                "type": "LinearEdge",
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": 2,
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
                    "percentile": 2,
                    "selection": "median",
                },
            },
            "wet_edge": {
                "type": "FlatEdge",
            },
            "var": "albedo",
        },
        {
            "name": "model",
            "dry_edge": {
                "type": "FlatEdge",
            },
            "wet_edge": {
                "type": "FlatEdge",
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


@pytest.mark.unit
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
                    "percentile": 2,
                    "selection": "median",
                },
            },
        },
        {
            "dry_edge": {
                "config": {
                    "interval_type": "size",
                    "interval_nb": 20,
                    "percentile": 2,
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lst", "mask", "expected"),
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


@pytest.mark.functional
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {
                "models": "default_evaspa",
            },
            2,
        ),
        pytest.param(
            {
                "models": [
                    {
                        "name": "model1",
                        "dry_edge": {
                            "type": "LinearEdge",
                            "config": {
                                "interval_type": "size",
                                "interval_size": 0.05,
                                "percentile": 2,
                                "selection": "median",
                            },
                        },
                        "wet_edge": {
                            "type": "LinearEdge",
                            "config": {
                                "interval_type": "size",
                                "interval_size": 0.05,
                                "percentile": 2,
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
                                "percentile": 5,
                                "selection": "median",
                            },
                        },
                        "wet_edge": {
                            "type": "FlatEdge",
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
                "options": {},
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
        pytest.param(
            {
                "models": "default_evaspa",
                "options": {
                    "filtering": {
                        "albedo": {
                            "and": [
                                {"op": ">=", "value": 0.1},
                                {"op": "<=", "value": 0.3},
                            ]
                        }
                    },
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
    options_keys = ["filtering", "selection", "merging"]
    assert len(models) == expected
    assert options
    assert list(options.keys()).sort() == options_keys.sort()


@pytest.mark.functional
@pytest.mark.parametrize(
    ("mask", "model_mask"),
    [
        pytest.param(None, None),
        pytest.param("mask", None),
        pytest.param(None, "model_mask"),
        pytest.param("mask", "model_mask"),
    ],
)
def test_compute(mask, model_mask) -> None:
    """
    Test compute function
    """
    # Generate models
    models = setup_models()
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        fcover=(0, 1.0),
        valid=(0.2, 0.8),
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    if mask is not None:
        mask = data["valid"]
    if model_mask is not None:
        model_mask = np.random.choice(
            [0, 1], size=data["valid"].shape, p=[0.5, 0.5]
        )
    # Compute EF
    ef = compute(models, data, mask=mask, model_mask=model_mask)
    assert ef
    assert len(ef.data_vars) == 2


@pytest.mark.functional
def test_select() -> None:
    """
    Test select function
    """
    ef = xr.Dataset(
        data_vars={
            "model1": (["y", "x"], np.random.normal(0.5, 0.2, (100, 100))),
            "model2": (["y", "x"], np.random.normal(0.5, 0.24, (100, 100))),
        },
        coords={
            "y": ("y", np.linspace(0, 99, num=100)),
            "x": ("x", np.linspace(0, 99, num=100)),
        },
        attrs={"description": "EF models"},
    )
    selected = select(ef)
    xr.testing.assert_identical(ef, selected)


@pytest.mark.functional
@pytest.mark.parametrize(
    ("flags", "valid", "options"),
    [
        pytest.param(
            False,
            False,
            {
                "selection": False,
                "merging": MergeMethod.MEAN,
            },
        ),
        pytest.param(
            True,
            False,
            {
                "selection": False,
                "merging": MergeMethod.MEDIAN,
            },
        ),
        pytest.param(
            False,
            True,
            {
                "filtering": {
                    "albedo": {
                        "and": [
                            {"op": ">=", "value": 0.1},
                            {"op": "<=", "value": 0.3},
                        ]
                    }
                },
                "selection": False,
                "merging": MergeMethod.MEDIAN,
            },
        ),
    ],
)
def test_run(valid, flags, options) -> None:
    """
    Test run function
    """
    # Generate models
    models = setup_models()
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        fcover=(0, 1.0),
        valid=(0.2, 0.8),
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    if not valid:
        data = data.drop_vars("valid")
    if flags:
        data["flags"] = xr.zeros_like(data["lst"])
    ef_xr, ef_inst = run(models, data, **options)
    assert ef_xr
    assert ef_inst
    assert "valid" in ef_xr.data_vars
    assert "flags" in ef_xr.data_vars
    assert "valid" in ef_inst.data_vars
    assert "flags" in ef_inst.data_vars


@pytest.mark.functional
def test_run_exc() -> None:
    """
    Test run function
    """
    # Generate models
    models = setup_models()
    # Generate data
    data = setup_data(
        albedo=(0.0, 0.6),
        valid=(0.0, 1.0),
        dry=(330.0, -10.0),
        wet=(300.0, 15.0),
    )
    with pytest.raises(
        KeyError,
        match=r"Variable fcover is missing to compute EF from EF models",
    ):
        run(models, data)


@pytest.mark.functional
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
                        "percentile": 2,
                        "selection": "median",
                    },
                },
                "wet_edge": {
                    "type": "LinearEdge",
                    "config": {
                        "interval_type": "size",
                        "interval_nb": 20,
                        "percentile": 2,
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
                        "percentile": 5,
                        "selection": "median",
                    },
                },
                "wet_edge": {
                    "type": "FlatEdge",
                },
                "var": "albedo",
            },
        ],
        "options": {
            "filtering": {
                "albedo": {
                    "and": [
                        {"op": ">=", "value": 0.1},
                        {"op": "<=", "value": 0.3},
                    ]
                }
            },
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "filtering_expected", "selection_expected", "merging_expected"),
    [
        pytest.param({}, {}, False, "median"),
        pytest.param({"selection": True}, {}, True, "median"),
        pytest.param({"merging": "mean"}, {}, False, "mean"),
        pytest.param(
            {"filtering": {}, "selection": False, "merging": "median"},
            {},
            False,
            "median",
        ),
        pytest.param(
            {
                "filtering": {
                    "albedo": {
                        "and": [
                            {"op": ">=", "value": 0.1},
                            {"op": "<=", "value": 0.3},
                        ]
                    }
                },
                "selection": True,
                "merging": "mean",
            },
            {
                "albedo": {
                    "and": [
                        {"op": ">=", "value": 0.1},
                        {"op": "<=", "value": 0.3},
                    ],
                    "or": None,
                }
            },
            True,
            "mean",
        ),
    ],
)
def test_efoptionsconfig(
    config, filtering_expected, selection_expected, merging_expected
) -> None:
    """
    Test EFOptionsConfig
    """
    options = EFOptionsConfig.model_validate(config)
    assert options.filtering.model_dump(by_alias=True) == filtering_expected
    assert options.selection == selection_expected
    assert options.merging.value == merging_expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param({}, 0.02),
        pytest.param({"threshold": 2}, 2),
    ],
)
def test_efcheckconfig(config, expected) -> None:
    """
    Test EFCheckConfig
    """
    check = EFCheckConfig.model_validate(config)
    assert check.threshold == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {
            "models": [
                {
                    "name": "model1",
                    "dry_edge": {
                        "type": "LinearEdge",
                        "config": {
                            "interval_type": "size",
                            "interval_nb": 20,
                            "percentile": 2,
                            "selection": "median",
                        },
                    },
                    "wet_edge": {
                        "type": "LinearEdge",
                        "config": {
                            "interval_type": "size",
                            "interval_nb": 20,
                            "percentile": 2,
                            "selection": "median",
                        },
                    },
                    "var": "fcover",
                }
            ]
        },
        {"models": "default_evaspa"},
        {"models": "trishna_evaspa"},
        {"models": "avignon_evaspa"},
        {"models": "hsm_evaspa"},
        {"check": {"threshold": 10}, "models": "default_evaspa"},
        {
            "check": {"threshold": 10},
            "models": "default_evaspa",
            "options": {"selection": True},
        },
    ],
)
def test_efconfig(config) -> None:
    """
    Test EFCheckConfig
    """
    cfg = EFConfig.model_validate(config).model_dump()
    models = [EFModel.create(m) for m in cfg["models"]]
    assert cfg
    assert models


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {
            "models": [
                {
                    "name": "model1",
                    "dry_edge": {
                        "type": "FlatEdge",
                    },
                    "wet_edge": {
                        "type": "FlatEdge",
                    },
                    "var": "fcover",
                },
                {
                    "name": "model1",
                    "dry_edge": {
                        "type": "FlatEdge",
                    },
                    "wet_edge": {
                        "type": "FlatEdge",
                    },
                    "var": "lai",
                },
            ]
        },
    ],
)
def test_efconfig_error(config) -> None:
    """
    Test EFCheckConfig with error
    """
    with pytest.raises((EFConfigError, EFModelError)):
        EFConfig.model_validate(config)


@pytest.mark.unit
def test_get_available_configuration() -> None:
    """
    Test get_available_configuration method
    """
    names = get_available_configuration()
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(sweat.evaspa.ef.__file__)),
        "conf",
    )
    configs = [
        os.path.basename(os.path.splitext(file)[0])
        for file in glob.glob(os.path.join(config_path, "*.json"))
        if os.path.splitext(file)[1] == ".json"
    ]
    assert sorted(names) == sorted(configs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config_name", "expected"),
    [
        pytest.param("hsm_evaspa", ["lst", "albedo"]),
        pytest.param("avignon_evaspa", ["lst", "albedo", "ndvi"]),
        pytest.param("global_evaspa", ["lst", "albedo", "ndvi"]),
        pytest.param("trishna_evaspa", ["lst", "albedo", "fcover", "ndvi"]),
    ],
)
def test_get_variables_from_models(config_name, expected):
    """
    Test function to get variables required by EF models
    """
    config_file = os.path.join(
        os.path.dirname(sweat.evaspa.ef.__file__),
        "conf",
        f"{config_name}.json",
    )
    with open(config_file) as f_config:
        config = json.load(f_config)
    models = [EFModel.create(m) for m in config["models"]]
    res = get_variables_from_models(models)
    assert sorted(res) == sorted(expected)
