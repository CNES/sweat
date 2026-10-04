# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions for plotting
"""

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output, display
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

from extra.metrics import compute_metrics, safe_polyfit


def plot_metrics_for_sites(df: pd.DataFrame, variable: str = "le"):
    """
    Interactive plot with metrics

    Notes
    -----
    Each site is in a subplot. A subplot gathers all site results.
    """

    groups = list(df.name.unique())
    groups.append("all")
    n_groups = len(groups)

    # assign one color per group (excluding "all")
    base_groups = df.name.unique()
    colors = px.colors.qualitative.Plotly

    color_map = {g: colors[i % len(colors)] for i, g in enumerate(base_groups)}

    ncols = 3
    nrows = int(np.ceil(n_groups / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=groups,
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )

    for i, group in enumerate(groups):
        row = i // ncols + 1
        col = i % ncols + 1

        if group == "all":
            measured = df[f"ec_{variable}"]
            estimated = df[variable]
            for g in df.name.unique():
                subset = df[df.name == g]

                fig.add_trace(
                    go.Scatter(
                        x=subset[f"ec_{variable}"],
                        y=subset[variable],
                        mode="markers",
                        marker={
                            "size": 6,
                            "opacity": 0.5,
                            "color": color_map[g],
                        },
                        name=g,
                        legendgroup=g,
                        showlegend=(
                            group == "all"
                        ),  # only show legend here if you want
                        customdata=subset["date"],
                        hovertemplate=(
                            "Group: " + g + "<br>"
                            "Measured: %{x:.3f}<br>"
                            "Estimated: %{y:.3f}<br>"
                            "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
        else:
            subset = df[df.name == group]
            measured = subset[f"ec_{variable}"]
            estimated = subset[variable]

            # Scatter with hover info (including date)
            fig.add_trace(
                go.Scatter(
                    x=measured,
                    y=estimated,
                    mode="markers",
                    marker={
                        "size": 6,
                        "opacity": 0.6,
                        "color": color_map.get(group, "gray"),
                    },
                    name=group,
                    showlegend=False,
                    customdata=subset["date"],
                    hovertemplate=(
                        "Measured: %{x:.2f}<br>"
                        "Estimated: %{y:.2f}<br>"
                        "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        # Identity line
        lims = [
            min(measured.min(), estimated.min()),
            max(measured.max(), estimated.max()),
        ]

        fig.add_trace(
            go.Scatter(
                x=lims,
                y=lims,
                mode="lines",
                line={"dash": "dash", "color": "black"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Linear fit
        idx = np.isfinite(measured) & np.isfinite(estimated)
        slope, intercept = np.polyfit(measured[idx], estimated[idx], 1)
        fig.add_trace(
            go.Scatter(
                x=lims,
                y=[lims[0] * slope + intercept, lims[1] * slope + intercept],
                mode="lines",
                line={"color": "red"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Metrics
        slope_m, intercept, mbe, mae, rmse, r2, _, _, _ = compute_metrics(
            measured, estimated
        )

        fig.add_annotation(
            x=0.97,
            y=0.03,
            xref="x domain",
            yref="y domain",
            text=(
                f"slope = {slope_m:.2f}<br>"
                f"intercept = {intercept:.2f}<br>"
                f"MBE = {mbe:.2f}<br>"
                f"MAE = {mae:.2f}<br>"
                f"RMSE = {rmse:.2f}<br>"
                f"R2 = {r2:.2f}"
            ),
            showarrow=False,
            align="right",
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text=f"Measured {variable.upper()}")
    fig.update_yaxes(title_text=f"Estimated {variable.upper()}")
    fig.update_layout(
        height=400 * nrows,
        width=500 * ncols,
        title=f"Measured vs Estimated {variable.upper()}",
    )

    fig.show()


def plot_data_for_sites(df: pd.DataFrame, variable: str):
    """
    Plot variables evolution overtime for each sites
    """
    fig = px.line(
        df,
        x="date",
        y=variable,
        color="name",  # group by "name"
        markers=True,
        custom_data=["name"],
    )

    # Print value
    fig.update_traces(
        hovertemplate=(
            "<b>Site:</b> %{customdata[0]}<br>"
            "<b>Date:</b> %{x|%Y-%m-%d}<br>"
            f"<b>{variable}:</b> %{{y:.2f}}<br>"
            "<extra></extra>"
        )
    )

    # Format x-axis as YYYY-MM-DD
    fig.update_xaxes(tickformat="%Y-%m-%d")

    # Improve layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=variable,
        legend_title="Site",
        title={
            "text": f"Evolution of {variable} over time",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
    )

    fig.show()


def plot_scatter_for_sites(
    df: pd.DataFrame,
    x: str,
    y: str,
    axis_range: tuple[float, float] | None = None,
):
    """
    Interactive scatter plot

    Notes
    -----
    Each site is in a subplot. A subplot gathers all site results.
    """

    groups = list(df.name.unique())
    groups.append("all")
    n_groups = len(groups)

    # assign one color per group (excluding "all")
    base_groups = df.name.unique()
    colors = px.colors.qualitative.Plotly

    color_map = {g: colors[i % len(colors)] for i, g in enumerate(base_groups)}

    ncols = 3
    nrows = int(np.ceil(n_groups / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=groups,
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )

    for i, group in enumerate(groups):
        row = i // ncols + 1
        col = i % ncols + 1

        if group == "all":
            for g in df.name.unique():
                subset = df[df.name == g]

                fig.add_trace(
                    go.Scatter(
                        x=subset[x],
                        y=subset[y],
                        mode="markers",
                        marker={
                            "size": 6,
                            "opacity": 0.5,
                            "color": color_map[g],
                        },
                        name=g,
                        legendgroup=g,
                        showlegend=(
                            group == "all"
                        ),  # only show legend here if you want
                        customdata=subset["date"],
                        hovertemplate=(
                            "Group: " + g + "<br>"
                            f"{x}: %{{x:.3f}}<br>"
                            f"{y}: %{{y:.3f}}<br>"
                            "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
        else:
            subset = df[df.name == group]

            xdata = subset[x]
            ydata = subset[y]
            # Scatter with hover info (including date)
            fig.add_trace(
                go.Scatter(
                    x=xdata,
                    y=ydata,
                    mode="markers",
                    marker={
                        "size": 6,
                        "opacity": 0.6,
                        "color": color_map.get(group, "gray"),
                    },
                    name=group,
                    showlegend=False,
                    customdata=subset["date"],
                    hovertemplate=(
                        f"{x}: %{{x:.3f}}<br>"
                        f"{y}: %{{y:.3f}}<br>"
                        "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    fig.update_xaxes(title_text=f"{x}")
    fig.update_yaxes(title_text=f"{y}")
    if range is not None:
        fig.update_xaxes(range=axis_range)
        fig.update_yaxes(range=axis_range)
    fig.update_layout(
        height=400 * nrows,
        width=500 * ncols,
        title=f"{x} vs {y}",
    )

    fig.show()


def plot_data(df: pd.DataFrame, variable: str | None = None):
    """
    Plot data over time
    """
    # Create category dropdown widget
    category_dropdown = widgets.Dropdown(
        options=["name", "landcover"],
        value="name",
        description="Category:",
    )
    category = category_dropdown.value

    # Create class dropdown (updated based on category)
    selection_dropdown = widgets.Dropdown(
        options=["All", *sorted(df[category].unique())],
        value="All",
        description="Selection:",
    )

    # Function to update class_dropdown when category changes
    def update_selection_options(change):
        category = change["new"]
        selection_dropdown.options = [
            "All",
            *sorted(df[category].unique()),
        ]
        selection_dropdown.value = "All"

    category_dropdown.observe(update_selection_options, names="value")

    variables = sorted(
        df.columns.difference(
            ["name", "date", "data (utc)", "lat", "lon", "landcover"]
        )
    )
    if variable is None or variable not in variables:
        variable = "ts"

    var_dropdown = widgets.Dropdown(
        options=variables,
        value=variable,
        description="Variable:",
        style={"description_width": "initial"},
    )

    # Output area for plot
    output = widgets.Output()

    # --- Update function ---
    def update_plot(change=None):  # noqa
        with output:
            clear_output(wait=True)

            category = category_dropdown.value
            selected = selection_dropdown.value
            var = var_dropdown.value

            subset = df if selected == "All" else df[df[category] == selected]

            fig = px.line(
                subset,
                x="date",
                y=var,
                markers=True,
                color="name",
                render_mode="svg",
            )

            # Print value
            fig.update_traces(
                hovertemplate=(
                    "<b>Site:</b> %{fullData.name}<br>"
                    "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                    f"<b>{var}:</b> %{{y:.2f}}<br>"
                    "<extra></extra>"
                )
            )

            # Format x-axis as YYYY-MM-DD
            fig.update_xaxes(tickformat="%Y-%m-%d")

            # Improve layout
            fig.update_layout(
                autosize=True,
                margin={"l": 20, "r": 20, "t": 60, "b": 40},
                xaxis_title="Date",
                yaxis_title=var,
                legend_title="Site",
                # legend=dict(orientation="h"),
                title={
                    "text": (
                        f"Evolution of {var} over time (Class: {selected})"
                    ),
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 20},
                },
            )
            fig.show(config={"responsive": True})

    # --- Link widgets to function ---
    category_dropdown.observe(update_plot, names="value")
    selection_dropdown.observe(update_plot, names="value")
    var_dropdown.observe(update_plot, names="value")

    # --- Layout ---
    controls = widgets.HBox(
        [category_dropdown, selection_dropdown, var_dropdown]
    )

    display(controls, output)

    # Initial plot
    update_plot()


def plot_scatter(
    df: pd.DataFrame | list[pd.DataFrame],
    x: str | None = None,
    y: str | None = None,
    xrange: tuple[float, float] | None = None,
    yrange: tuple[float, float] | None = None,
    width: int = 600,
    height: int = 600,
    marker_size: int = 6,
    font_size: int = 12,
    use_colors: bool = True,
):
    """
    Scatter plot by class
    """
    # --- Normalize input ---
    if isinstance(df, pd.DataFrame):
        dfs = [df]
    elif isinstance(df, (list, tuple)) and len(df) in [1, 2]:
        dfs = list(df)
    else:
        msg = "df must be a DataFrame or a list/tuple of 1 or 2 DataFrames"
        raise ValueError(msg)
    names = [
        dfi.attrs.get("label", f"Data {i + 1}") for i, dfi in enumerate(dfs)
    ]

    ref_df = dfs[0]

    # Create category dropdown widget
    category_dropdown = widgets.Dropdown(
        options=["name", "landcover"],
        value="name",
        description="Category:",
    )
    category = category_dropdown.value

    # Create class dropdown (updated based on category)
    selection_dropdown = widgets.Dropdown(
        options=["All", *sorted(ref_df[category].unique())],
        value="All",
        description="Selection:",
    )

    # Function to update class_dropdown when category changes
    def update_selection_options(change):
        category = change["new"]
        selection_dropdown.options = [
            "All",
            *sorted(ref_df[category].unique()),
        ]
        selection_dropdown.value = "All"

    category_dropdown.observe(update_selection_options, names="value")

    variables = sorted(
        ref_df.columns.difference(
            ["name", "date", "data (utc)", "lat", "lon", "landcover"]
        )
    )

    if x is None or x not in variables:
        x = "ta"
    if y is None or y not in variables:
        y = "ts"

    xvar_dropdown = widgets.Dropdown(
        options=variables,
        value=x,
        description="X axis:",
        style={"description_width": "initial"},
    )

    yvar_dropdown = widgets.Dropdown(
        options=variables,
        value=y,
        description="Y axis:",
        style={"description_width": "initial"},
    )

    metrics_checkbox = widgets.Checkbox(
        value=False, description="Compute metrics"
    )

    range_checkbox = widgets.Checkbox(value=False, description="Fixed range")

    output = widgets.Output()

    # --- Update function ---
    def update_plot(change=None):  # noqa
        with output:
            clear_output(wait=True)

            category = category_dropdown.value
            selected = selection_dropdown.value
            xvar = xvar_dropdown.value
            yvar = yvar_dropdown.value

            # --- Create consistent color map for site names ---
            all_sites = set()
            for dfi in dfs:
                if selected == "All":
                    subset = dfi
                else:
                    subset = dfi[dfi[category] == selected]
                all_sites.update(subset["name"].unique())

            all_sites = sorted(all_sites)
            color_map = {site: i for i, site in enumerate(all_sites)}
            colors = [
                f"hsl({(i % 10) * 36}, 70%, 50%)" for i in range(len(all_sites))
            ]
            # Track which sites have been added to legend
            sites_in_legend = set()

            # --- Create figure ---
            if len(dfs) == 2:
                fig = make_subplots(rows=1, cols=2, subplot_titles=names)
            else:
                fig = go.Figure()

            # --- Loop over dataframes ---
            for i, dfi in enumerate(dfs):
                if selected == "All":
                    subset = dfi
                else:
                    subset = dfi[dfi[category] == selected]
                x_values = subset[xvar]
                y_values = subset[yvar]

                row, col = (1, i + 1) if len(dfs) == 2 else (None, None)

                for site_name, group in subset.groupby("name"):
                    color_idx = color_map[site_name]
                    show_in_legend = site_name not in sites_in_legend
                    sites_in_legend.add(site_name)
                    trace = go.Scatter(
                        x=group[xvar],
                        y=group[yvar],
                        mode="markers",
                        name=site_name,
                        marker={
                            "size": marker_size,
                            "opacity": 0.5,
                            "color": colors[color_idx]
                            if use_colors
                            else "blue",
                        },
                        customdata=group["date"],
                        legendgroup=site_name,  # Group legend entries
                        # Show only on first appearance
                        showlegend=show_in_legend,
                    )

                    if len(dfs) == 2:
                        fig.add_trace(trace, row=row, col=col)
                    else:
                        fig.add_trace(trace)

                # Hover
                fig.update_traces(
                    hovertemplate=(
                        "<b>Site:</b> %{fullData.name}<br>"
                        f"<b>{xvar}:</b> %{{x:.2f}}<br>"
                        f"<b>{yvar}:</b> %{{y:.2f}}<br>"
                        "Date: %{customdata|%Y-%m-%d}<extra></extra>"
                    )
                )

                # Fixed range
                if range_checkbox.value:
                    lims = [
                        min(x_values.min(), y_values.min()),
                        max(x_values.max(), y_values.max()),
                    ]
                    axis_range = [
                        lims[0] - 0.1 * abs(lims[0]),
                        lims[1] + 0.1 * abs(lims[1]),
                    ]

                    if len(dfs) == 2:
                        fig.update_xaxes(range=axis_range, row=row, col=col)
                        fig.update_yaxes(range=axis_range, row=row, col=col)
                    else:
                        fig.update_xaxes(range=axis_range)
                        fig.update_yaxes(range=axis_range)

                # Metrics
                if metrics_checkbox.value:
                    slope, intercept = safe_polyfit(x_values, y_values)

                    lims = [
                        min(x_values.min(), y_values.min()),
                        max(x_values.max(), y_values.max()),
                    ]

                    # Identity line
                    line1 = go.Scatter(
                        x=lims,
                        y=lims,
                        mode="lines",
                        line={"dash": "dash", "color": "black"},
                        showlegend=False,
                    )

                    # Fit line
                    line2 = None
                    if not np.isnan(slope) and not np.isnan(intercept):
                        line2 = go.Scatter(
                            x=lims,
                            y=[
                                lims[0] * slope + intercept,
                                lims[1] * slope + intercept,
                            ],
                            mode="lines",
                            line={"color": "red"},
                            showlegend=False,
                        )

                    if len(dfs) == 2:
                        fig.add_trace(line1, row=row, col=col)
                        if line2:
                            fig.add_trace(line2, row=row, col=col)
                    else:
                        fig.add_trace(line1)
                        if line2:
                            fig.add_trace(line2)

                    # --- Metrics text ---
                    slope_m, intercept, mbe, mae, rmse, r2, _, _, _ = (
                        compute_metrics(x_values, y_values)
                    )

                    if len(dfs) == 2:
                        fig.add_annotation(
                            x=0.97,
                            y=0.03,
                            xref="x domain",
                            yref="y domain",
                            text=(
                                f"slope = {slope_m:.2f}<br>"
                                f"intercept = {intercept:.2f}<br>"
                                f"MBE = {mbe:.2f}<br>"
                                f"MAE = {mae:.2f}<br>"
                                f"RMSE = {rmse:.2f}<br>"
                                f"R² = {r2:.2f}"
                            ),
                            showarrow=False,
                            align="right",
                            row=row,
                            col=col,
                        )
                    else:
                        fig.add_annotation(
                            x=0.97,
                            y=0.03,
                            xref="x domain",
                            yref="y domain",
                            text=(
                                f"slope = {slope_m:.2f}<br>"
                                f"intercept = {intercept:.2f}<br>"
                                f"MBE = {mbe:.2f}<br>"
                                f"MAE = {mae:.2f}<br>"
                                f"RMSE = {rmse:.2f}<br>"
                                f"R² = {r2:.2f}"
                            ),
                            showarrow=False,
                            align="right",
                        )
                if xrange is not None and yrange is not None:
                    if len(dfs) == 2:
                        fig.update_xaxes(range=xrange, row=row, col=col)
                        fig.update_yaxes(range=yrange, row=row, col=col)
                    else:
                        fig.update_xaxes(range=xrange)
                        fig.update_yaxes(range=yrange)
                if len(dfs) == 2:
                    fig.update_xaxes(
                        title_text=xvar,
                        row=row,
                        col=col,
                        showline=False,
                        mirror=True,
                        linecolor="black",
                        linewidth=2,
                        showgrid=True,
                        gridcolor="white",
                        gridwidth=1,
                        zeroline=False,
                        zerolinecolor="white",
                        zerolinewidth=1,
                        title_font={"size": font_size + 4},
                        tickfont={"size": font_size},
                    )
                    fig.update_yaxes(
                        title_text=yvar,
                        row=row,
                        col=col,
                        showline=False,
                        mirror=True,
                        linecolor="black",
                        linewidth=2,
                        showgrid=True,
                        gridcolor="white",
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor="white",
                        zerolinewidth=1,
                        title_font={"size": font_size + 4},
                        tickfont={"size": font_size},
                    )
                else:
                    fig.update_xaxes(
                        title_text=xvar,
                        showgrid=True,
                        showline=False,
                        mirror=True,
                        linecolor="black",
                        linewidth=2,
                        gridcolor="white",
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor="white",
                        zerolinewidth=1,
                        title_font={"size": font_size + 4},
                        tickfont={"size": font_size},
                    )
                    fig.update_yaxes(
                        title_text=yvar,
                        showgrid=True,
                        showline=False,
                        mirror=True,
                        linecolor="black",
                        linewidth=2,
                        gridcolor="white",
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor="white",
                        zerolinewidth=1,
                        title_font={"size": font_size + 4},
                        tickfont={"size": font_size},
                    )

            # Layout
            fig.update_layout(
                template="plotly",
                height=height,
                width=width * len(dfs),
                showlegend=True,  # Add this line
                font={
                    "size": font_size,
                },
            )

            # Subplot title font and metrics font
            if len(dfs) == 2:
                # First annotations are the subplot titles
                for annotation in fig.layout.annotations[: len(names)]:
                    annotation.font = {"size": font_size + 6, "color": "black"}

                # Remaining annotations are the metrics
                for annotation in fig.layout.annotations[len(names) :]:
                    annotation.font = {"size": font_size + 4, "color": "black"}

            fig.show(config={"responsive": True})

    # --- Bind widgets ---
    for w in [
        category_dropdown,
        selection_dropdown,
        xvar_dropdown,
        yvar_dropdown,
        metrics_checkbox,
        range_checkbox,
    ]:
        w.observe(update_plot, names="value")

    controls = widgets.HBox(
        [
            category_dropdown,
            selection_dropdown,
            xvar_dropdown,
            yvar_dropdown,
            metrics_checkbox,
            range_checkbox,
        ]
    )

    display(controls, output)

    # Initial plot
    update_plot()


def plot_metrics(
    df: pd.DataFrame | list[pd.DataFrame],
    variable: str,
    metric: str,
    facet_by_range: bool = False,
    only_all: bool = False,
    font_size: int = 12,
):
    """
    Plot metrics per landcover class
    """
    # --- Normalize input ---
    if isinstance(df, pd.DataFrame):
        dfs = [df]
    elif isinstance(df, (list, tuple)) and len(df) in [1, 2]:
        dfs = list(df)
    else:
        msg = "df must be a DataFrame or a list/tuple of 1 or 2 DataFrames"
        raise ValueError(msg)
    names = [dfi.attrs.get("label", f"DF{i + 1}") for i, dfi in enumerate(dfs)]

    df_mean = []
    for name, dfi in zip(names, dfs, strict=True):
        dfi_mean = (
            dfi[dfi["variable"] == variable]
            .groupby(["landcover", "range"], as_index=False)
            .agg({metric: "mean", "nb": "sum"})
        )
        dfi_mean["label"] = name
        df_mean.append(dfi_mean)
    global_df = pd.concat(df_mean, ignore_index=True)

    # Sort ranges: "All" first, then by extracting numeric bounds
    def sort_key(r):
        if r == "all":
            return (3, 0, 0)  # "All" comes last
        if r.startswith("<"):
            val = int(r[1:])
            return (0, val, val)
        if r.startswith(">"):
            val = int(r[1:])
            return (2, val, val)
        if "-" in r:
            parts = r.split("-")
            return (1, int(parts[0]), int(parts[1]))
        return (4, 0, 0)

    range_order = sorted(global_df["range"].unique(), key=sort_key)
    global_df["range"] = pd.Categorical(
        global_df["range"], categories=range_order, ordered=True
    )
    global_df = global_df.sort_values("range")
    if only_all:
        global_df = global_df[global_df["range"] == "all"]

    # Get the number of ranges
    n_ranges = len(range_order)

    # Sample colors from continuous scales
    blues = sample_colorscale(
        "Blues", [i / n_ranges for i in range(1, n_ranges + 1)]
    )
    oranges = sample_colorscale(
        "Oranges", [i / n_ranges for i in range(1, n_ranges + 1)]
    )

    # Build color mapping dynamically
    color_map = {}
    palettes = [blues, oranges]

    for idx, name in enumerate(names):
        palette = palettes[idx % len(palettes)]

        for range_idx, r in enumerate(range_order):
            label_range = f"{name} - {r}"
            color_map[label_range] = palette[range_idx]

    if facet_by_range:
        fig = px.bar(
            global_df,
            x="landcover",
            y=metric,
            color="label",
            facet_col="range",
            category_orders={
                "range": range_order,
                "landcover": sorted(global_df["landcover"].unique()),
            },
            barmode="group",
            color_discrete_map=color_map,
            title=(
                f"Average {metric.upper()} for {variable.upper()} per landcover"
            ),
            labels={metric: f"{metric.upper()}", "landcover": "Landcover"},
            custom_data=["nb"],
        )
    else:
        # Create a single plot with range in the legend
        global_df["label_range"] = (
            global_df["label"] + " - " + global_df["range"].astype(str)
        )

        fig = px.bar(
            global_df,
            x="landcover",
            y=metric,
            color="label_range",
            barmode="group",
            title=(
                f"Average {metric.upper()} for {variable.upper()} per landcover"
            ),
            labels={metric: f"{metric.upper()}", "landcover": "Landcover"},
            custom_data=["nb", "range"],
            color_discrete_map=color_map,
            category_orders={
                "label_range": [
                    f"{n} - {r}" for n in names for r in range_order
                ],
                "landcover": sorted(global_df["landcover"].unique()),
            },
        )

    # Update hover template to include "nb"
    fig.update_traces(
        hovertemplate=(
            "<b>Landcover:</b> %{x}<br>"
            "<b>Metric:</b> %{y:.2f}<br>"
            "<b>Count:</b> %{customdata[0]}<br>"
            "<b>Source:</b> %{fullData.name}<extra></extra>"
        )
    )
    # Layout
    fig.update_layout(
        font={
            "size": font_size,
        },
    )
    fig.show()


def plot_scatter_with_colorbar(
    df: pd.DataFrame | list[pd.DataFrame],
    x: str | None = None,
    y: str | None = None,
    c: str | None = None,
    xrange: tuple[float, float] | None = None,
    yrange: tuple[float, float] | None = None,
    crange: tuple[float, float] | None = None,
    width: int = 600,
    height: int = 600,
    marker_size: int = 6,
    font_size: int = 14,
    use_stressed: bool = False,
):
    """
    Plot scatter plots with dynamic coloring,
    marker shape differentiation,
    and category selection.
    """
    # --- Normalize input ---
    if isinstance(df, pd.DataFrame):
        dfs = [df]
    elif isinstance(df, (list, tuple)) and len(df) in [1, 2]:
        dfs = list(df)
    else:
        msg = "df must be a DataFrame or a list/tuple of 1 or 2 DataFrames"
        raise ValueError(msg)
    names = [
        dfi.attrs.get("label", f"Data {i + 1}") for i, dfi in enumerate(dfs)
    ]
    ref_df = dfs[0]

    # Create category dropdown widget
    category_dropdown = widgets.Dropdown(
        options=["name", "landcover"],
        value="name",
        description="Category:",
    )
    category = category_dropdown.value

    # Create class dropdown (updated based on category)
    selection_dropdown = widgets.Dropdown(
        options=["All", *sorted(ref_df[category].unique())],
        value="All",
        description="Selection:",
    )

    # Function to update class_dropdown when category changes
    def update_selection_options(change):
        category = change["new"]
        selection_dropdown.options = [
            "All",
            *sorted(ref_df[category].unique()),
        ]
        selection_dropdown.value = "All"

    category_dropdown.observe(update_selection_options, names="value")

    variables = sorted(
        ref_df.columns.difference(
            [
                "name",
                "date",
                "data (utc)",
                "lat",
                "lon",
                "landcover",
            ]
        )
    )

    if x is None or x not in variables:
        x = "ta"
    if y is None or y not in variables:
        y = "ts"

    xvar_dropdown = widgets.Dropdown(
        options=variables,
        value=x,
        description="X axis:",
        style={"description_width": "initial"},
    )

    yvar_dropdown = widgets.Dropdown(
        options=variables,
        value=y,
        description="Y axis:",
        style={"description_width": "initial"},
    )

    cvar_dropdown = widgets.Dropdown(
        options=[None, *variables],
        value=c,
        description="Color bar:",
        style={"description_width": "initial"},
    )

    metrics_checkbox = widgets.Checkbox(
        value=False, description="Compute metrics"
    )

    range_checkbox = widgets.Checkbox(value=False, description="Fixed range")

    output = widgets.Output()

    # --- Update function ---
    def update_plot(change=None):  # noqa
        with output:
            clear_output(wait=True)

            category = category_dropdown.value
            selected = selection_dropdown.value
            xvar = xvar_dropdown.value
            yvar = yvar_dropdown.value
            cvar = cvar_dropdown.value

            # --- Create figure ---
            if len(dfs) == 2:
                fig = make_subplots(rows=1, cols=2, subplot_titles=names)
            else:
                fig = go.Figure()

            # Track if it is the first trace with colorbar for this subplot
            is_first_trace_in_subplot = True

            # Get min/max for the colorbar if colorbar is selected
            color_min = np.inf
            color_max = -np.inf
            if cvar is not None:
                for dfi in dfs:
                    if selected == "All":
                        subset = dfi
                    else:
                        subset = dfi[dfi[category] == selected]
                    # Get color values if colorbar is selected
                    color_values = subset[cvar]
                    if color_values.min() < color_min:
                        color_min = color_values.min()
                    if color_values.max() > color_max:
                        color_max = color_values.max()

            # --- Loop over dataframes ---
            for i, dfi in enumerate(dfs):
                if selected == "All":
                    subset = dfi
                else:
                    subset = dfi[dfi[category] == selected]
                x_values = subset[xvar]
                y_values = subset[yvar]

                row, col = (1, i + 1) if len(dfs) == 2 else (None, None)

                for site_name, group in subset.groupby("name"):
                    # Separate by stressed status
                    if use_stressed:
                        stressed_groups = [
                            (group[group["stressed"] == 0], "circle"),
                            (group[group["stressed"] == 1], "square"),
                        ]
                    else:
                        stressed_groups = [
                            (group, "circle"),
                        ]

                    for group_data, marker_symbol in stressed_groups:
                        if len(group_data) == 0:
                            continue

                        # Prepare marker color
                        if cvar is not None:
                            marker_color = group_data[cvar]
                            marker_dict = {
                                "size": marker_size,
                                "opacity": 0.5,
                                "symbol": marker_symbol,
                                "color": marker_color,
                                "colorscale": "RdYlGn",
                                "cmin": color_min,
                                "cmax": color_max,
                                "showscale": is_first_trace_in_subplot,
                                "colorbar": {
                                    "title": {
                                        "text": cvar,
                                        "font": {
                                            "size": font_size + 2,
                                        },
                                    },
                                    "tickfont": {
                                        "size": font_size,
                                    },
                                }
                                if is_first_trace_in_subplot
                                else None,
                            }
                            if crange is not None:
                                marker_dict["cmin"] = crange[0]
                                marker_dict["cmax"] = crange[1]
                        else:
                            marker_dict = {
                                "size": marker_size,
                                "opacity": 0.5,
                                "symbol": marker_symbol,
                                "color": "blue",
                            }

                        # Prepare custom data for hover
                        custom_data = ["date", "stressed"]  # Always included
                        if cvar is not None:
                            custom_data.append(
                                cvar
                            )  # Add colorbar variable to custom data

                        trace = go.Scatter(
                            x=group_data[xvar],
                            y=group_data[yvar],
                            mode="markers",
                            name=site_name,
                            marker=marker_dict,
                            customdata=group_data[custom_data],
                            hovertemplate=(
                                "<b>Site:</b> %{fullData.name}<br>"
                                f"<b>{xvar}:</b> %{{x:.2f}}<br>"
                                f"<b>{yvar}:</b> %{{y:.2f}}<br>"
                                f"<b>Stressed:</b> %{{customdata[1]}}<br>"
                                + (
                                    f"<b>{cvar}:</b> %{{customdata[2]:.2f}}<br>"
                                    if cvar is not None
                                    else ""
                                )
                                + (
                                    "Date: %{customdata[0]|%Y-%m-%d}"
                                    "<extra></extra>"
                                )
                            ),
                        )

                        if len(dfs) == 2:
                            fig.add_trace(trace, row=row, col=col)
                        else:
                            fig.add_trace(trace)

                        # Mark that the first trace has been added
                        is_first_trace_in_subplot = False

                # Fixed range
                if range_checkbox.value:
                    lims = [
                        min(x_values.min(), y_values.min()),
                        max(x_values.max(), y_values.max()),
                    ]
                    axis_range = [
                        lims[0] - 0.1 * abs(lims[0]),
                        lims[1] + 0.1 * abs(lims[1]),
                    ]

                    if len(dfs) == 2:
                        fig.update_xaxes(range=axis_range, row=row, col=col)
                        fig.update_yaxes(range=axis_range, row=row, col=col)
                    else:
                        fig.update_xaxes(range=axis_range)
                        fig.update_yaxes(range=axis_range)

                    # Identity line
                    lims = [
                        min(x_values.min(), y_values.min()),
                        max(x_values.max(), y_values.max()),
                    ]
                    lims = [xrange[0], xrange[1]]

                    line1 = go.Scatter(
                        x=lims,
                        y=lims,
                        mode="lines",
                        line={"dash": "dash", "color": "black"},
                        showlegend=False,
                    )

                    if len(dfs) == 2:
                        fig.add_trace(line1, row=row, col=col)
                    else:
                        fig.add_trace(line1)

                # Metrics
                if metrics_checkbox.value:
                    slope, intercept = safe_polyfit(x_values, y_values)

                    lims = [
                        min(x_values.min(), y_values.min()),
                        max(x_values.max(), y_values.max()),
                    ]

                    # Identity line
                    line1 = go.Scatter(
                        x=lims,
                        y=lims,
                        mode="lines",
                        line={"dash": "dash", "color": "black"},
                        showlegend=False,
                    )

                    # Fit line
                    line2 = None
                    if not np.isnan(slope) and not np.isnan(intercept):
                        line2 = go.Scatter(
                            x=lims,
                            y=[
                                lims[0] * slope + intercept,
                                lims[1] * slope + intercept,
                            ],
                            mode="lines",
                            line={"color": "red"},
                            showlegend=False,
                        )

                    if len(dfs) == 2:
                        fig.add_trace(line1, row=row, col=col)
                        if line2:
                            fig.add_trace(line2, row=row, col=col)
                    else:
                        fig.add_trace(line1)
                        if line2:
                            fig.add_trace(line2)

                    # --- Metrics text ---
                    slope_m, intercept, mbe, mae, rmse, r2, _, _, _ = (
                        compute_metrics(x_values, y_values)
                    )

                    if len(dfs) == 2:
                        fig.add_annotation(
                            x=0.97,
                            y=0.03,
                            xref="x domain",
                            yref="y domain",
                            text=(
                                f"slope = {slope_m:.2f}<br>"
                                f"intercept = {intercept:.2f}<br>"
                                f"MBE = {mbe:.2f}<br>"
                                f"MAE = {mae:.2f}<br>"
                                f"RMSE = {rmse:.2f}<br>"
                                f"R² = {r2:.2f}"
                            ),
                            showarrow=False,
                            align="right",
                            row=row,
                            col=col,
                        )
                    else:
                        fig.add_annotation(
                            x=0.97,
                            y=0.03,
                            xref="x domain",
                            yref="y domain",
                            text=(
                                f"slope = {slope_m:.2f}<br>"
                                f"intercept = {intercept:.2f}<br>"
                                f"MBE = {mbe:.2f}<br>"
                                f"MAE = {mae:.2f}<br>"
                                f"RMSE = {rmse:.2f}<br>"
                                f"R² = {r2:.2f}"
                            ),
                            showarrow=False,
                            align="right",
                        )
                if xrange is not None and yrange is not None:
                    if len(dfs) == 2:
                        fig.update_xaxes(range=xrange, row=row, col=col)
                        fig.update_yaxes(range=yrange, row=row, col=col)
                    else:
                        fig.update_xaxes(range=xrange)
                        fig.update_yaxes(range=yrange)
                if len(dfs) == 2:
                    fig.update_xaxes(
                        title_text=xvar,
                        row=row,
                        col=col,
                        showline=False,
                        mirror=True,
                        linecolor="black",
                        linewidth=2,
                        showgrid=True,
                        gridcolor="white",
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor="white",
                        zerolinewidth=1,
                        title_font={"size": font_size + 4},
                        tickfont={"size": font_size},
                    )
                    fig.update_yaxes(
                        title_text=yvar,
                        row=row,
                        col=col,
                        showline=False,
                        mirror=True,
                        linecolor="black",
                        linewidth=2,
                        showgrid=True,
                        gridcolor="white",
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor="white",
                        zerolinewidth=1,
                        title_font={"size": font_size + 4},
                        tickfont={"size": font_size},
                    )
                else:
                    fig.update_xaxes(
                        title_text=xvar,
                        showgrid=True,
                        showline=False,
                        mirror=True,
                        linecolor="black",
                        linewidth=2,
                        gridcolor="white",
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor="white",
                        zerolinewidth=1,
                        title_font={"size": font_size + 4},
                        tickfont={"size": font_size},
                    )
                    fig.update_yaxes(
                        title_text=yvar,
                        showgrid=True,
                        showline=False,
                        mirror=True,
                        linecolor="black",
                        linewidth=2,
                        gridcolor="white",
                        gridwidth=1,
                        zeroline=True,
                        zerolinecolor="white",
                        zerolinewidth=1,
                        title_font={"size": font_size + 4},
                        tickfont={"size": font_size},
                    )

            # Layout
            fig.update_layout(
                template="plotly",
                height=height,
                width=width * len(dfs),
                showlegend=False,  # Add this line
                font={
                    "size": font_size,
                },
            )

            # Subplot title font and metrics font
            if len(dfs) == 2:
                # First annotations are the subplot titles
                for annotation in fig.layout.annotations[: len(names)]:
                    annotation.font = {"size": font_size + 6, "color": "black"}

                # Remaining annotations are the metrics
                for annotation in fig.layout.annotations[len(names) :]:
                    annotation.font = {"size": font_size + 4, "color": "black"}

            fig.show(config={"responsive": True})

    # --- Bind widgets ---
    for w in [
        category_dropdown,
        selection_dropdown,
        xvar_dropdown,
        yvar_dropdown,
        cvar_dropdown,
        metrics_checkbox,
        range_checkbox,
    ]:
        w.observe(update_plot, names="value")

    controls = widgets.HBox(
        [
            category_dropdown,
            selection_dropdown,
            xvar_dropdown,
            yvar_dropdown,
            cvar_dropdown,
            metrics_checkbox,
            range_checkbox,
        ]
    )

    display(controls, output)

    # Initial plot
    update_plot()


def plot_scatter_3d(df: pd.DataFrame):
    """
    Scatter plot in 3 dimensions
    """
    # Define available columns for selection
    variables = sorted(
        df.columns.difference(
            [
                "name",
                "date",
                "data (utc)",
                "lat",
                "lon",
            ]
        )
    )
    numeric_variables = (
        df[variables].select_dtypes(include=["number"]).columns.tolist()
    )
    metadata = [
        "nir",
        "swir",
        "vari_green",
        "gli",
        "gndvi",
        "msavi",
        "landcover",
    ]

    # Create category dropdown widget
    category_dropdown = widgets.Dropdown(
        options=["name", "landcover"],
        value="name",
        description="Category:",
    )
    category = category_dropdown.value

    # Function to update class_dropdown when category changes
    def update_selection_options(change):
        category = change["new"]
        selection_dropdown.options = [
            "All",
            *sorted(df[category].unique()),
        ]
        selection_dropdown.value = "All"

    category_dropdown.observe(update_selection_options, names="value")

    # Create class dropdown (updated based on category)
    selection_dropdown = widgets.Dropdown(
        options=["All", *sorted(df[category].unique())],
        value="All",
        description="Selection:",
    )

    # Create dropdown widgets
    xvar_dropdown = widgets.Dropdown(
        options=numeric_variables,
        value="nir/swir",
        description="X Axis:",
        disabled=False,
    )

    yvar_dropdown = widgets.Dropdown(
        options=numeric_variables,
        value="vari_green",
        description="Y Axis:",
        disabled=False,
    )

    zvar_dropdown = widgets.Dropdown(
        options=numeric_variables,
        value="gli",
        description="Z Axis:",
        disabled=False,
    )

    cvar_dropdown = widgets.Dropdown(
        options=variables,
        value="landcover",
        description="Color:",
        disabled=False,
    )

    # Define the plotting function
    def update_plot(category, selection, x_axis, y_axis, z_axis, color_axis):
        subset = df if selection == "All" else df[df[category] == selection]
        fig = px.scatter_3d(
            subset,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color=color_axis,
            hover_data=metadata,
            size_max=1,
            color_continuous_scale="RdYlGn",
        )
        fig.update_traces(marker={"size": 2})
        fig.show(renderer="browser")

    # Create interactive output
    output = widgets.interactive_output(
        update_plot,
        {
            "category": category_dropdown,
            "selection": selection_dropdown,
            "x_axis": xvar_dropdown,
            "y_axis": yvar_dropdown,
            "z_axis": zvar_dropdown,
            "color_axis": cvar_dropdown,
        },
    )

    # Display dropdowns and plot
    return widgets.HBox(
        [
            category_dropdown,
            selection_dropdown,
            xvar_dropdown,
            yvar_dropdown,
            zvar_dropdown,
            cvar_dropdown,
            output,
        ]
    )


def analyze(
    df: pd.DataFrame,
    x: str,
    y: str,
    c: str,
    bins: int = 50,
    x_label: str | None = None,
    y_label: str | None = None,
    color_label: str | None = None,
    title: str | None = None,
    width: int = 1000,
    height: int = 700,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    c_range: tuple[float, float] | None = None,
    font_size: int = 14,
):
    """
    Plot data and overlay the mean y value within x bins.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x, y, c : str
        Column names for the x axis, y axis, and color.
    bins : int
        Number of equally sized bins along x.
    x_label, y_label, color_label : str, optional
        Display labels for the axes and color legend/color bar.
    width, height : int
        Figure dimensions in pixels.
    x_range, y_range, c_range : tuple[float, float], optional
        Axis ranges and color range, for example ``(0, 100)``.
    font_size : int
        Base font size for the figure.
    """

    data = df[[x, y, c]].dropna().copy()

    # Create x bins
    data["_x_bin"] = pd.cut(data[x], bins=bins, include_lowest=True)

    # Compute mean y and mean x for each bin
    means = (
        data.groupby("_x_bin", observed=True)
        .agg(
            x_mean=(x, "mean"),
            y_mean=(y, "mean"),
        )
        .dropna()
        .reset_index(drop=True)
        .sort_values("x_mean")
    )

    labels = {
        x: x_label or x,
        y: y_label or y,
        c: color_label or c,
    }

    # Original scatter plot
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=c,
        labels=labels,
        opacity=0.6,
        render_mode="svg",
        width=width,
        height=height,
    )
    # Increase only the original scatter-point size
    fig.update_traces(
        selector={"mode": "markers"},
        marker={"size": 8},
    )

    # Mean line
    fig.add_scatter(
        x=means["x_mean"],
        y=means["y_mean"],
        mode="lines+markers",
        line={"color": "red", "width": 3},
        marker={"size": 10},
        showlegend=False,
    )

    if c_range is not None:
        fig.update_coloraxes(
            cmin=c_range[0],
            cmax=c_range[1],
        )

    # Compute Pearson correlation and p-value
    if len(data) >= 3 and data[x].nunique() > 1 and data[y].nunique() > 1:
        correlation, p_value = pearsonr(data[x], data[y])
        correlation_text = f"r = {correlation:.2f}<br>p-value = {p_value:.2g}"

    fig.update_layout(
        template="plotly",
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": font_size + 6},
        },
        font={
            "size": font_size,
        },
        # Axis ranges
        xaxis={
            "range": x_range,
            "showline": False,
            "mirror": True,
            "linecolor": "black",
            "linewidth": 2,
            "showgrid": True,
            "zeroline": True,
            "title_font": {"size": font_size + 4},
            "tickfont": {"size": font_size},
        },
        yaxis={
            "range": y_range,
            "showline": False,
            "mirror": True,
            "linecolor": "black",
            "linewidth": 2,
            "showgrid": True,
            "zeroline": True,
            "zerolinecolor": "white",
            "zerolinewidth": 1,
            "title_font": {"size": font_size + 4},
            "tickfont": {"size": font_size},
        },
    )

    # Larger colorbar title and tick labels
    fig.update_coloraxes(
        colorbar={
            "title": {
                "text": color_label or c,
                "font": {"size": font_size + 4},
            },
            "tickfont": {"size": font_size},
        }
    )
    # Display correlation
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=correlation_text,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={
            "size": font_size,
            "color": "black",
        },
    )

    fig.show()
