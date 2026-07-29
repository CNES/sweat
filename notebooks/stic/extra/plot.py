# Copyright: (c) 2026 CESBIO / Centre National d'Etudes Spatiales
"""
Module containing functions for plotting
"""

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output, display
from plotly.subplots import make_subplots

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
        slope_m, mbe, mae, rmse, r2 = compute_metrics(measured, estimated)

        fig.add_annotation(
            x=0.97,
            y=0.03,
            xref="x domain",
            yref="y domain",
            text=(
                f"slope = {slope_m:.2f}<br>"
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
    Plot data
    """
    sites = sorted(df["name"].unique())
    variables = sorted(
        df.columns.difference(
            ["name", "date", "data (utc)", "lat", "lon", "landcover"]
        )
    )
    if variable is None or variable not in variables:
        variable = "ts"
    site_dropdown = widgets.Dropdown(
        options=sites,
        value=sites[0],
        description="Site:",
        style={"description_width": "initial"},
    )

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

            site = site_dropdown.value
            var = var_dropdown.value

            subset = df[df["name"] == site]

            fig = px.line(
                subset,
                x="date",
                y=var,
                markers=True,
            )

            # Print value
            fig.update_traces(
                hovertemplate=(
                    f"<b>Site:</b> {site}<br>"
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
                title={
                    "text": f"Evolution of {var} over time (site: {site})",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 20},
                },
            )
            fig.show(config={"responsive": True})

    # --- Link widgets to function ---
    site_dropdown.observe(update_plot, names="value")
    var_dropdown.observe(update_plot, names="value")

    # --- Layout ---
    controls = widgets.HBox([site_dropdown, var_dropdown])

    display(controls, output)

    # Initial plot
    update_plot()


def plot_data_by_class(df: pd.DataFrame, variable: str | None = None):
    """
    Plot data
    """
    classes = ["All", *sorted(df["landcover"].unique())]
    variables = sorted(
        df.columns.difference(
            ["name", "date", "data (utc)", "lat", "lon", "landcover"]
        )
    )
    if variable is None or variable not in variables:
        variable = "ts"

    class_dropdown = widgets.Dropdown(
        options=classes,
        value="All",
        description="Class:",
        style={"description_width": "initial"},
    )

    var_dropdown = widgets.Dropdown(
        options=variables,
        value="ts",
        description="Variable:",
        style={"description_width": "initial"},
    )

    # Output area for plot
    output = widgets.Output()

    # --- Update function ---
    def update_plot(change=None):  # noqa
        with output:
            clear_output(wait=True)

            selected_class = class_dropdown.value
            var = var_dropdown.value

            if selected_class == "All":
                subset = df
            else:
                subset = df[df["landcover"] == selected_class]

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
                        f"Evolution of {var} over "
                        f"time (Class: {selected_class})"
                    ),
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 20},
                },
            )
            fig.show(config={"responsive": True})

    # --- Link widgets to function ---
    class_dropdown.observe(update_plot, names="value")
    var_dropdown.observe(update_plot, names="value")

    # --- Layout ---
    controls = widgets.HBox([class_dropdown, var_dropdown])

    display(controls, output)

    # Initial plot
    update_plot()


def plot_scatter(
    df: pd.DataFrame | list[pd.DataFrame],
    x: str | None = None,
    y: str | None = None,
):
    """
    Scatter plot
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

    # --- Widgets (based on first df) ---
    ref_df = dfs[0]

    sites = sorted(ref_df["name"].unique())
    variables = sorted(
        ref_df.columns.difference(
            ["name", "date", "data (utc)", "lat", "lon", "landcover"]
        )
    )

    if x is None or x not in variables:
        x = "ta"
    if y is None or y not in variables:
        y = "ts"

    site_dropdown = widgets.Dropdown(
        options=sites,
        value=sites[0],
        description="Site:",
        style={"description_width": "initial"},
    )

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

            site = site_dropdown.value
            xvar = xvar_dropdown.value
            yvar = yvar_dropdown.value

            # --- Create figure ---
            if len(dfs) == 2:
                fig = make_subplots(rows=1, cols=2, subplot_titles=names)
            else:
                fig = go.Figure()

            # --- Loop over dataframes ---
            for i, dfi in enumerate(dfs):
                subset = dfi[dfi["name"] == site]
                x_values = subset[xvar]
                y_values = subset[yvar]

                row, col = (1, i + 1) if len(dfs) == 2 else (None, None)

                trace = go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    marker={"size": 6, "opacity": 0.5},
                    customdata=subset["date"],
                    showlegend=False,
                )

                if len(dfs) == 2:
                    fig.add_trace(trace, row=row, col=col)
                else:
                    fig.add_trace(trace)

                # Hover
                fig.update_traces(
                    hovertemplate=(
                        f"<b>Site:</b> {site}<br>"
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
                    if not np.isnan(slope) or np.isnan(intercept):
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
                    slope_m, mbe, mae, rmse, r2 = compute_metrics(
                        x_values, y_values
                    )

                    if len(dfs) == 2:
                        fig.add_annotation(
                            x=0.97,
                            y=0.03,
                            xref="x domain",
                            yref="y domain",
                            text=(
                                f"slope = {slope_m:.2f}<br>"
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
                                f"MBE = {mbe:.2f}<br>"
                                f"MAE = {mae:.2f}<br>"
                                f"RMSE = {rmse:.2f}<br>"
                                f"R² = {r2:.2f}"
                            ),
                            showarrow=False,
                            align="right",
                        )
            # Layout
            fig.update_layout(
                template="plotly_white",
                height=700,
                width=1400 if len(dfs) == 2 else 700,
            )
            if len(dfs) == 2:
                fig.update_xaxes(title_text=xvar, row=row, col=col)
                fig.update_yaxes(title_text=yvar, row=row, col=col)
            else:
                fig.update_xaxes(title_text=xvar)
                fig.update_yaxes(title_text=yvar)

            fig.show(config={"responsive": True})

    # --- Bind widgets ---
    for w in [
        site_dropdown,
        xvar_dropdown,
        yvar_dropdown,
        metrics_checkbox,
        range_checkbox,
    ]:
        w.observe(update_plot, names="value")

    controls = widgets.HBox(
        [
            site_dropdown,
            xvar_dropdown,
            yvar_dropdown,
            metrics_checkbox,
            range_checkbox,
        ]
    )

    display(controls, output)

    # Initial plot
    update_plot()


def plot_scatter_by_class(
    df: pd.DataFrame | list[pd.DataFrame],
    x: str | None = None,
    y: str | None = None,
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

    # --- Widgets (based on first df) ---
    ref_df = dfs[0]

    classes = ["All", *sorted(ref_df["landcover"].unique())]
    variables = sorted(
        ref_df.columns.difference(
            ["name", "date", "data (utc)", "lat", "lon", "landcover"]
        )
    )

    if x is None or x not in variables:
        x = "ta"
    if y is None or y not in variables:
        y = "ts"

    class_dropdown = widgets.Dropdown(
        options=classes,
        value=classes[0],
        description="Site:",
        style={"description_width": "initial"},
    )

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

            selected_class = class_dropdown.value
            xvar = xvar_dropdown.value
            yvar = yvar_dropdown.value

            # --- Create figure ---
            if len(dfs) == 2:
                fig = make_subplots(rows=1, cols=2, subplot_titles=names)
            else:
                fig = go.Figure()

            # --- Loop over dataframes ---
            for i, dfi in enumerate(dfs):
                if selected_class == "All":
                    subset = dfi
                else:
                    subset = dfi[dfi["landcover"] == selected_class]
                x_values = subset[xvar]
                y_values = subset[yvar]

                row, col = (1, i + 1) if len(dfs) == 2 else (None, None)

                for site_name, group in subset.groupby("name"):
                    trace = go.Scatter(
                        x=group[xvar],
                        y=group[yvar],
                        mode="markers",
                        name=site_name,
                        marker={"size": 6, "opacity": 0.5},
                        customdata=group["date"],
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
                    if np.isnan(slope) or np.isnan(intercept):
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
                    slope_m, mbe, mae, rmse, r2 = compute_metrics(
                        x_values, y_values
                    )

                    if len(dfs) == 2:
                        fig.add_annotation(
                            x=0.97,
                            y=0.03,
                            xref="x domain",
                            yref="y domain",
                            text=(
                                f"slope = {slope_m:.2f}<br>"
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
                                f"MBE = {mbe:.2f}<br>"
                                f"MAE = {mae:.2f}<br>"
                                f"RMSE = {rmse:.2f}<br>"
                                f"R² = {r2:.2f}"
                            ),
                            showarrow=False,
                            align="right",
                        )
            # Layout
            fig.update_layout(
                template="plotly_white",
                height=700,
                width=1400 if len(dfs) == 2 else 700,
            )
            if len(dfs) == 2:
                fig.update_xaxes(title_text=xvar, row=row, col=col)
                fig.update_yaxes(title_text=yvar, row=row, col=col)
            else:
                fig.update_xaxes(title_text=xvar)
                fig.update_yaxes(title_text=yvar)

            fig.show(config={"responsive": True})

    # --- Bind widgets ---
    for w in [
        class_dropdown,
        xvar_dropdown,
        yvar_dropdown,
        metrics_checkbox,
        range_checkbox,
    ]:
        w.observe(update_plot, names="value")

    controls = widgets.HBox(
        [
            class_dropdown,
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
    df: pd.DataFrame | list[pd.DataFrame], variable: str, metric: str
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
            .groupby("landcover", as_index=False)[metric]
            .mean()
        )
        dfi_mean["source"] = name
        df_mean.append(dfi_mean)
    df_mean = pd.concat(df_mean, ignore_index=True)
    fig = px.bar(
        df_mean,
        x="landcover",
        y="rmse",
        color="source",
        barmode="group",
        title=f"Average RMSE for {variable.upper()} per landcover",
        labels={"rmse": "Mean RMSE", "landcover": "Landcover"},
    )

    fig.show()
