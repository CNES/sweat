# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing plot functions
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib.colors import ListedColormap

from sweat.common.types import ETVar
from sweat.evaspa.edge import Edge
from sweat.evaspa.ef import EFModel


def plot_rgb(data: xr.Dataset, bands: list[str] | None = None):
    """
    Plot RGB bands in a dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    bands: list[str]
        List of bands used. If not provided, ["red","gree","blue"]
    """
    if bands is None:
        bands = [ETVar.RED.value, ETVar.GREEN.value, ETVar.BLUE.value]
    spatial_dims = data[bands[0]].dims
    rgb = xr.DataArray(
        data=np.dstack(
            (data[bands[0]].data, data[bands[1]].data, data[bands[2]].data)
        ),
        coords=data.coords.assign(band=["r", "g", "b"]),
        dims=(*spatial_dims, "band"),
    )
    rgb.plot.imshow(  # type: ignore
        rgb="band",
        vmin=rgb.quantile(0.01),
        vmax=rgb.quantile(0.99),
    )
    plt.title("RGB")


def plot_band(data: xr.Dataset, band: str, percentile: float = 0.0):
    """
    Plot specific band in a dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    band: str
        Band name
    percentile: float
        Percentile used for dynamic
    """
    data[band].plot.imshow(  # type: ignore
        vmin=data[band].quantile(percentile),
        vmax=data[band].quantile(1 - percentile),
    )


def plot_mask(data: xr.Dataset, mask: str, invert: bool = False):
    """
    Plot specific mask in a dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    mask: str
        Mask name in the dataset
    invert: bool
        Invert the mask (default = False)
    """
    cmap = [(0.0, 0.0, 0.0, 0.0)]
    if len(np.unique(data[mask].data)) > 1:
        cmap = [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0)]
    if invert:
        cmap = cmap[::-1]
    valid_cmap = ListedColormap(cmap)
    data[mask].plot(cmap=valid_cmap, add_colorbar=False)  # type: ignore[call-arg]


def plot_flags(data: xr.Dataset, flags: str):
    """
    Plot specific flags in a dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    flags: str
        Band name for flags in the dataset
    """
    da = data[flags]
    unique_combinations = np.unique(da.values)
    # Use a qualitative colormap with enough colors
    cmap = plt.get_cmap("tab20", len(unique_combinations))
    # Map each unique bitmask value to an integer index (for coloring)
    value_to_index = {v: i for i, v in enumerate(unique_combinations)}
    indexed_data = da.copy(data=np.vectorize(value_to_index.get)(da.values))
    indexed_data.plot(cmap=cmap, add_colorbar=False)  # type: ignore[call-arg]
    legend_patches = [
        mpatches.Patch(color=cmap(i), label=format(v, "08b"))
        for i, v in enumerate(unique_combinations)
    ]
    plt.legend(
        handles=legend_patches,
        title="Classes present (bits)",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize="small",
    )


def plot_data(
    data: xr.Dataset,
    var_name: str,
):
    """
    Plot LST vs variable for a dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    var_name: str
        Variable name
    """
    # Create figure
    _ = plt.figure(constrained_layout=True, figsize=(4, 3))

    ax = plt.gca()

    if ETVar.VALID.value in data.data_vars:
        var = data[var_name].where(data[ETVar.VALID.value]).data
        lst = data[ETVar.LST.value].where(data[ETVar.VALID.value]).data
    else:
        var = data[var_name].data
        lst = data[ETVar.LST.value].data
    lst = data[ETVar.LST.value].data
    var = data[var_name].data
    # plot all points
    ax.scatter(var, lst, s=5, alpha=1, color="moccasin", clip_on=False)

    # Labels
    ax.set_xlabel(var_name)
    ax.set_ylabel("Land Surface Temperature K")

    plt.show()


def plot_dataset(
    data: xr.Dataset,
):
    """
    Plot a dataset

    Parameters
    ----------
    data: xr.Dataset
        Data
    """
    # Create figure
    # Set figure subplots
    nb = len(data.data_vars)
    row = int(np.ceil(nb / 2))
    col = 2
    fig = plt.figure(figsize=(4 * col, 3 * row), constrained_layout=True)

    i = 0
    j = 0
    # Loop over variables
    for var in data.data_vars:
        ax = plt.subplot2grid((row, col), (i, j))
        if var == ETVar.VALID.value:
            valid_cmap = ListedColormap(
                [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]
            )
            data[var].plot(ax=ax, cmap=valid_cmap, add_colorbar=False)  # type: ignore[call-arg]
        elif var == ETVar.FLAGS.value:
            da = data[var]
            unique_combinations = np.unique(da.values)
            # Use a qualitative colormap with enough colors
            cmap = plt.get_cmap("tab20", len(unique_combinations))
            # Map each unique bitmask value to an integer index (for coloring)
            value_to_index = {v: i for i, v in enumerate(unique_combinations)}
            indexed_data = da.copy(
                data=np.vectorize(value_to_index.get)(da.values)
            )
            indexed_data.plot(ax=ax, cmap=cmap, add_colorbar=False)  # type: ignore[call-arg]
        else:
            data[var].plot(ax=ax)  # type: ignore[call-arg]
        j += 1
        if j == 2:
            j = 0
            i += 1
    # title
    fig.suptitle("Dataset", fontsize=12)

    plt.show()


def plot_dem(xrds_dem: xr.Dataset):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    xrds_dem[ETVar.HEIGHT.value].plot(  # type: ignore[call-arg]
        ax=axes[0],
        vmin=xrds_dem[ETVar.HEIGHT.value].min(),
        vmax=xrds_dem[ETVar.HEIGHT.value].max(),
        cmap="RdYlGn_r",
    )
    xrds_dem[ETVar.SLOPE.value].plot(  # type: ignore[call-arg]
        ax=axes[1],
        vmin=xrds_dem[ETVar.SLOPE.value].min(),
        vmax=xrds_dem[ETVar.SLOPE.value].max(),
        cmap="Reds",
    )
    xrds_dem[ETVar.ASPECT.value].plot(  # type: ignore[call-arg]
        ax=axes[2],
        vmin=xrds_dem[ETVar.ASPECT.value].min(),
        vmax=xrds_dem[ETVar.ASPECT.value].max(),
        cmap="twilight_shifted",
    )


def plot_models(
    data: xr.Dataset,
    models: xr.Dataset,
):
    """
    Plot EF models
    """
    # Set figure subplots
    nb = len(models.data_vars)
    row = int(np.ceil(nb / 2))
    col = 2
    fig = plt.figure(figsize=(4 * col, 3 * row), constrained_layout=True)

    i = 0
    j = 0
    # Loop over models
    for model_name in models.data_vars:
        if model_name not in [ETVar.VALID.value, ETVar.FLAGS.value]:
            model = EFModel.create(models[model_name].attrs)
            var_name = model.var
            if ETVar.VALID.value in data.data_vars:
                var = data[model.var].where(data[ETVar.VALID.value]).data
                lst = data[ETVar.LST.value].where(data[ETVar.VALID.value]).data
            else:
                var = data[model.var].data
                lst = data[ETVar.LST.value].data
            ax = plt.subplot2grid((row, col), (i, j))
            # plot LST points
            ax.scatter(var, lst, s=20, alpha=1, color="moccasin", clip_on=False)
            # edges coordinates
            var_max = np.round(np.ceil(np.nanmax(var) * 10) / 10, decimals=1)
            var_min = np.round(np.floor(np.nanmin(var) * 10) / 10, decimals=1)
            var_arr = np.linspace(var_min, var_max, num=100)
            tw_arr = model.twet(var_arr)
            td_arr = model.tdry(var_arr)
            ax.plot(
                var_arr,
                td_arr,
                color="tab:red",
                linewidth=3,
                label="dry edge",
                zorder=1,
            )
            ax.plot(
                var_arr,
                tw_arr,
                color="tab:blue",
                linewidth=3,
                label="wet edge",
                zorder=1,
            )
            # Labels
            ax.set_xlabel(var_name)
            ax.set_ylabel("Land Surface Temperature K")
            ax.set_title(f"Model {model_name}", fontsize=12)
            j += 1
            if j == 2:
                j = 0
                i += 1

    # title
    fig.suptitle("Evaporative fraction methods", fontsize=12)

    plt.show()


def plot_edges(
    var: npt.NDArray,
    lst: npt.NDArray,
    edges: list[Edge] | None = None,
):
    """
    Plot
    """
    # Figure dimensions
    _ = plt.figure(constrained_layout=True, figsize=(4, 3))

    ax = plt.gca()

    # plot all points
    ax.scatter(var, lst, s=10, alpha=1, color="moccasin", clip_on=False)

    # plot edge
    if edges is not None:
        # edges coordinates
        var_max = np.round(np.ceil(np.nanmax(var) * 10) / 10, decimals=1)
        var_min = np.round(np.floor(np.nanmin(var) * 10) / 10, decimals=1)
        var_arr = np.linspace(var_min, var_max, num=1000)
        for edge in edges:
            edge_arr = edge.get(var_arr)

            ax.plot(
                var_arr,
                edge_arr,
                color="black",
                linewidth=3,
                zorder=1,
            )

    # Labels
    ax.set_xlabel("Variable")
    ax.set_ylabel("Land Surface Temperature K")

    # title
    ax.set_title("Evaporative fraction method", fontsize=12)

    plt.show()
