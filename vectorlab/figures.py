"""Plotly figures for the HTML reports.

Figures are drawn in the light palette. ``assets/theme.js`` repaints them for dark mode by
reading each trace's ``meta`` color role.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "s1": "#2a78d6",
    "s2": "#eb6834",
    "diverging": [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]],
}
DARK = {
    "surface": "#1a1a19",
    "ink": "#f0efec",
    "ink2": "#c3c2b7",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "s1": "#3987e5",
    "s2": "#d95926",
    "diverging": [[0.0, "#e66767"], [0.5, "#383835"], [1.0, "#3987e5"]],
}


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


def _reference(fig, y, text, row, above):
    fig.add_hline(y=y, row=row, col=1, line=dict(color=LIGHT["s2"], width=1.5), name="s2")
    fig.add_annotation(
        xref="x domain",
        yref=f"y{row if row > 1 else ''}",
        x=1,
        y=y,
        xanchor="right",
        yshift=10 if above else -10,
        text=text,
        showarrow=False,
        font=dict(size=11, color=LIGHT["ink2"]),
    )


def separation_figure(separation: dict) -> go.Figure:
    """Two stacked panels over layers: held-out Cohen's d and probe accuracy, each against its lexical baseline.

    The baseline (mean token embedding over the content, before any layer runs) shows how much
    of the separation is vocabulary alone.
    """
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
    lexical = separation["lexical"]
    _reference(fig, lexical["cohens_d"], f"lexical baseline (mean token embedding): d = {lexical['cohens_d']:.2f}", 1, above=False)
    _reference(fig, lexical["probe_acc"], f"lexical baseline: {lexical['probe_acc']:.3f}", 2, above=False)
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


def cosine_figure(directions: np.ndarray) -> go.Figure:
    """Heatmap of cosine similarity between every pair of per-layer directions."""
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit = directions / np.where(norms == 0, 1, norms)
    cosine = unit @ unit.T
    layers = list(range(len(cosine)))
    fig = go.Figure(
        go.Heatmap(
            z=cosine,
            x=layers,
            y=layers,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=LIGHT["diverging"],
            meta="diverging",
            colorbar=dict(title=dict(text="cosine"), thickness=12, outlinewidth=0),
            hovertemplate="layer %{y} vs layer %{x}<br>cosine %{z:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="layer", showline=False, ticks="", range=[-0.5, len(layers) - 0.5], constrain="domain")
    fig.update_yaxes(title_text="layer", showgrid=False, scaleanchor="x", range=[len(layers) - 0.5, -0.5], constrain="domain")
    return styled(fig, 520)


def _trial_hover(t):
    c = t["config"]
    return (
        f"trial {t['number']}<br>direction from layer {c['layer']}, {c['mode']} at layers {c['layers'][0]}–{c['layers'][-1]}"
        f"<br>strength {c['strength']:.3g}<br>score {t['score']:.3f} · KL {t['kl']:.3f} · length {100 * t['length_change']:+.0f}%"
    )


def _points(trials, role, name, size, show, xkey):
    return go.Scatter(
        x=[100 * t["length_change"] if xkey == "length" else t["kl"] for t in trials],
        y=[t["score"] for t in trials],
        mode="markers",
        meta=role,
        name=name,
        legendgroup=name,
        showlegend=show,
        marker=dict(size=size, color=LIGHT[role], line=dict(color=LIGHT["surface"], width=2)),
        text=[_trial_hover(t) for t in trials],
        hovertemplate="%{text}<extra></extra>",
    )


def tradeoff_figure(search: dict, metric_label: str) -> go.Figure:
    """Behavior score against KL and against answer-length change for every search trial.

    Pareto-front trials (on the length-penalized objective) are highlighted, and the original
    model is the reference point at zero KL and zero length change.
    """
    trials = search["trials"]
    front = sorted((t for t in trials if t["pareto"]), key=lambda t: t["kl"])
    rest = [t for t in trials if not t["pareto"]]
    original = [{"number": "–", "config": {"layer": "–", "mode": "none", "layers": ["–"], "strength": 0.0}, "score": search["baseline"]["score"], "kl": 0.0, "length_change": 0.0}]
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("Score vs KL on neutral prompts", "Score vs change in answer length"),
    )
    for col, xkey in ((1, "kl"), (2, "length")):
        fig.add_trace(_points(rest, "s1", "trial", 8, col == 1, xkey), 1, col)
        fig.add_trace(_points(front, "s2", "Pareto front (length-penalized objective)", 11, col == 1, xkey), 1, col)
        fig.add_trace(_points(original, "ink", "original model", 11, col == 1, xkey), 1, col)
    fig.add_trace(
        go.Scatter(x=[t["kl"] for t in front], y=[t["score"] for t in front], mode="lines", meta="s2", line=dict(color=LIGHT["s2"], width=2), hoverinfo="skip", showlegend=False),
        1,
        1,
    )
    fig.update_xaxes(title_text="KL divergence", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title_text="answer length change (%)", zeroline=True, zerolinecolor=LIGHT["axis"], row=1, col=2)
    fig.update_yaxes(title_text=metric_label, row=1, col=1)
    fig = styled(fig, 440)
    fig.update_layout(margin=dict(t=84))
    return fig


def pair_cosine_figure(cosines: np.ndarray, title: str, band: float) -> go.Figure:
    """Cosine between two behaviors' directions at each layer, on a fixed -1 to 1 scale.

    The shaded band spans ``±band``, where the cosine of two random directions would usually fall.
    """
    layers = list(range(len(cosines)))
    fig = go.Figure(
        go.Scatter(
            x=layers,
            y=cosines,
            mode="lines+markers",
            meta="s1",
            line=dict(color=LIGHT["s1"], width=2),
            marker=dict(size=5, color=LIGHT["s1"], line=dict(color=LIGHT["surface"], width=1)),
            hovertemplate="layer %{x}<br>cosine %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hrect(y0=-band, y1=band, fillcolor="rgba(128, 128, 128, 0.18)", line_width=0, layer="below")
    fig.add_hline(y=0, line=dict(color=LIGHT["axis"], width=1))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=13, color=LIGHT["ink"])))
    fig.update_xaxes(title_text="layer")
    fig.update_yaxes(title_text="cosine", range=[-1, 1], dtick=0.5)
    fig = styled(fig, 260)
    fig.update_layout(margin=dict(l=48, r=12, t=40, b=40))
    return fig


def matrix_figure(z, x: list[str], y: list[str], text, hover, zmax: float, colorbar: str, title: str | None = None, height: int = 340) -> go.Figure:
    """Annotated heatmap on the diverging scale, centered on zero and saturating at ``±zmax``.

    Args:
        z: Values that color the cells, shaped ``[len(y), len(x)]``.
        text: The label printed in each cell.
        hover: The hover text of each cell.
    """
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=text,
            texttemplate="%{text}",
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            zmin=-zmax,
            zmax=zmax,
            zmid=0,
            colorscale=LIGHT["diverging"],
            meta="diverging",
            xgap=2,
            ygap=2,
            colorbar=dict(title=dict(text=colorbar), thickness=12, outlinewidth=0),
        )
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=13, color=LIGHT["ink"])))
    fig.update_xaxes(showline=False, ticks="")
    fig.update_yaxes(showgrid=False, showline=False, autorange="reversed")
    fig = styled(fig, height)
    fig.update_layout(margin=dict(l=96))
    return fig


def kl_figure(names: list[str], values: list[float], hover: list[str]) -> go.Figure:
    """Horizontal bars: each edit's KL divergence from the original model on the neutral prompts."""
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            meta="s1",
            width=0.5,
            marker=dict(color=LIGHT["s1"], line=dict(width=0)),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            customdata=hover,
            hovertemplate="%{y}<br>KL %{x:.4f}<br>%{customdata}<extra></extra>",
        )
    )
    fig = styled(fig, 96 + 44 * len(names))
    fig.update_xaxes(title_text="KL divergence on neutral prompts", rangemode="tozero", showgrid=True, gridcolor=LIGHT["grid"])
    fig.update_yaxes(showgrid=False, autorange="reversed", ticks="")
    fig.update_layout(margin=dict(l=96, r=48, t=16, b=44))
    return fig
