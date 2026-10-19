"""Plotly figures for the HTML reports.

Figures are drawn in the light palette. ``assets/theme.js`` repaints them for dark mode by
reading each trace's ``meta`` color role.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LIGHT = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834"}
DARK = {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926"}


def styled(fig: go.Figure, height: int) -> go.Figure:
    """Apply the shared report chrome: transparent surfaces, hairline grid, system font."""
    fig.update_layout(
        template="none",
        height=height,
        margin=dict(l=56, r=16, t=48, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=LIGHT["ink2"]),
        hoverlabel=dict(bgcolor=LIGHT["surface"], font=dict(family=FONT, color=LIGHT["ink"])),
        legend=dict(orientation="h", x=0, y=0.98, yanchor="top", yref="container"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LIGHT["axis"], ticks="outside", tickcolor=LIGHT["axis"])
    fig.update_yaxes(gridcolor=LIGHT["grid"], zeroline=False, linecolor=LIGHT["axis"])
    fig.update_annotations(font=dict(size=13, color=LIGHT["ink"]))
    return fig


def to_div(fig: go.Figure) -> str:
    """Render a figure as an HTML fragment that relies on the page's inlined Plotly bundle."""
    return to_html(fig, full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def _line(x, y, role, name, hover):
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        meta=role,
        name=name,
        line=dict(color=LIGHT[role], width=2),
        marker=dict(size=8, color=LIGHT[role], line=dict(color=LIGHT["surface"], width=2)),
        hovertemplate=hover + "<extra></extra>",
        showlegend=False,
    )


def _step(values, edges, role, name):
    density, _ = np.histogram(values, edges, density=True)
    return go.Scatter(
        x=edges,
        y=np.append(density, density[-1]),
        mode="lines",
        line_shape="hv",
        fill="tozeroy",
        meta=role,
        name=name,
        showlegend=False,
        line=dict(color=LIGHT[role], width=2),
        fillcolor=LIGHT[role] + "1a",
        hovertemplate=name + "<br>projection %{x:.2f}<br>density %{y:.3f}<extra></extra>",
    )


def separation_figure(separation: dict) -> go.Figure:
    """Two stacked panels over layers: held-out Cohen's d, and held-out probe accuracy."""
    d = separation["cohens_d"]
    acc = separation["probe_acc"]
    layers = list(range(len(d)))
    peak = int(np.argmax(d))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Effect size, held-out (Cohen's d)", "Linear probe accuracy, held-out"),
    )
    fig.add_trace(_line(layers, d, "s1", "Cohen's d", "layer %{x}<br>d = %{y:.2f}"), 1, 1)
    fig.add_trace(_line(layers, acc, "s1", "probe accuracy", "layer %{x}<br>accuracy = %{y:.3f}"), 2, 1)
    fig.add_hline(y=0.5, row=2, col=1, line=dict(color=LIGHT["axis"], width=1))
    fig.add_annotation(x=peak, y=d[peak], text=f"peak d = {d[peak]:.2f} at layer {peak}", showarrow=False, yshift=16, row=1, col=1)
    fig.update_yaxes(title_text="d", row=1, col=1)
    fig.update_yaxes(title_text="accuracy", range=[min(0.45, min(acc) - 0.02), 1.02], row=2, col=1)
    fig.update_xaxes(title_text="layer", row=2, col=1)
    return styled(fig, 520)


def histogram_figure(pos: np.ndarray, neg: np.ndarray, layer: int, labels: list[str]) -> go.Figure:
    """Overlaid step histograms (density) of positive and negative projections at one layer."""
    edges = np.histogram_bin_edges(np.concatenate([pos, neg]), bins=30)
    fig = go.Figure([_step(pos, edges, "s1", labels[0]), _step(neg, edges, "s2", labels[1])])
    fig.update_layout(title=dict(text=f"Layer {layer}", x=0.5, font=dict(size=13, color=LIGHT["ink"])))
    fig.update_xaxes(title_text="projection onto r")
    fig.update_yaxes(title_text="density")
    return styled(fig, 300)
