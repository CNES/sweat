# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 CESBIO / Centre National d'Etudes Spatiales

import matplotlib as mpl
import numpy as np
import pytest
import xarray as xr

mpl.use("Agg")
import sweat.extra.plot_notebook as pltn
from sweat.evaspa.edge import Edge


@pytest.mark.functional
def test_plot_rgb() -> None:
    """
    Test function plot_rgb
    """
    data = xr.Dataset(
        {
            "red": (["x", "y"], np.random.rand(10, 5)),
            "green": (["x", "y"], np.random.rand(10, 5)),
            "blue": (["x", "y"], np.random.rand(10, 5)),
        },
        coords={"x": np.arange(10), "y": np.arange(5)},
    )
    pltn.plot_rgb(data)


@pytest.mark.functional
def test_plot_band() -> None:
    """
    Test function plot_band
    """
    data = xr.Dataset(
        {
            "value": (["x", "y"], np.random.rand(10, 5)),
        },
        coords={"x": np.arange(10), "y": np.arange(5)},
    )
    pltn.plot_band(data, band="value", percentile=0.0)


@pytest.mark.functional
def test_plot_mask() -> None:
    """
    Test function plot_mask
    """
    data = xr.Dataset(
        {
            "value": (["x", "y"], np.random.rand(10, 5)),
            "mask": (["x", "y"], np.random.randint(0, 2, size=(10, 5))),
        },
        coords={"x": np.arange(10), "y": np.arange(5)},
    )
    pltn.plot_mask(data, mask="mask")


@pytest.mark.functional
def test_plot_flags() -> None:
    """
    Test function plot_flags
    """
    data = xr.Dataset(
        {
            "value": (["x", "y"], np.random.rand(10, 5)),
            "flags": (["x", "y"], np.random.choice(5, size=(10, 5))),
        },
        coords={"x": np.arange(10), "y": np.arange(5)},
    )
    pltn.plot_flags(data, flags="flags")


@pytest.mark.functional
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive.*:UserWarning"
)
def test_plot_data() -> None:
    """
    Test function plot_data
    """
    data = xr.Dataset(
        {
            "lst": (["x", "y"], np.random.rand(10, 5)),
            "albedo": (["x", "y"], np.random.rand(10, 5)),
        },
        coords={"x": np.arange(10), "y": np.arange(5)},
    )
    pltn.plot_data(data, var_name="albedo")


@pytest.mark.functional
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive.*:UserWarning"
)
def test_plot_dataset() -> None:
    """
    Test function plot_dataset
    """
    data = xr.Dataset(
        {
            "lst": (["x", "y"], np.random.rand(10, 5)),
            "albedo": (["x", "y"], np.random.rand(10, 5)),
            "valid": (["x", "y"], np.random.randint(0, 2, size=(10, 5))),
            "flags": (["x", "y"], np.random.choice(5, size=(10, 5))),
        },
        coords={"x": np.arange(10), "y": np.arange(5)},
    )
    pltn.plot_dataset(data)


@pytest.mark.functional
def test_plot_dem() -> None:
    """
    Test function plot_dem
    """
    data = xr.Dataset(
        {
            "height": (["x", "y"], np.random.rand(10, 5)),
            "slope": (["x", "y"], np.random.rand(10, 5)),
            "aspect": (["x", "y"], np.random.rand(10, 5)),
        },
        coords={"x": np.arange(10), "y": np.arange(5)},
    )
    pltn.plot_dem(data)


@pytest.mark.functional
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive.*:UserWarning"
)
def test_plot_models() -> None:
    """
    Test function plot_dem
    """
    data = xr.Dataset(
        {
            "lst": (["x", "y"], np.random.rand(10, 5)),
            "var": (["x", "y"], np.random.rand(10, 5)),
        },
        coords={"x": np.arange(10), "y": np.arange(5)},
    )
    ef_model = xr.DataArray(
        np.random.rand(10, 5),
        dims=["x", "y"],
        coords={"x": np.arange(10), "y": np.arange(5)},
        attrs={
            "name": "model1",
            "dry_edge": {
                "type": "FlatEdge",
                "config": {"position": "top", "value": 1.0},
            },
            "wet_edge": {
                "type": "FlatEdge",
                "config": {"position": "bottom", "value": 0},
            },
            "var": "var",
        },
    )
    models = xr.Dataset({"model1": ef_model})
    models["valid"] = (["x", "y"], np.random.randint(0, 2, size=(10, 5)))
    models["flags"] = (["x", "y"], np.random.choice(5, size=(10, 5)))
    pltn.plot_models(data, models)


@pytest.mark.functional
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive.*:UserWarning"
)
def test_plot_edges() -> None:
    """
    Test function plot_edges
    """
    lst = np.random.rand(100)
    var = np.random.rand(100)
    edges = [
        Edge.create(
            name="FlatEdge",
            config={
                "position": "top",
                "value": 1.0,
            },
        )
    ]
    pltn.plot_edges(lst, var, edges=edges)
