"""Write the animated SVG figures for the README and FINDINGS.md from the run directories.

Usage:
    python scripts/doc_figures.py [--out docs/figures]

Each figure is a standalone SVG with its own surface, a light and a dark palette (chosen by
``prefers-color-scheme``), and CSS entry animations: lines draw in, bars grow, points and
cells fade in. Animations are skipped under ``prefers-reduced-motion``. GitHub shows SVGs as
plain images, without hover, so the docs give every figure a table with the same numbers.
"""

import argparse
import html
import math
from pathlib import Path
from typing import Callable

import numpy as np

from vectorlab.compare import same_edit
from vectorlab.figures import DARK as REPORT_DARK
from vectorlab.figures import FONT
from vectorlab.figures import LIGHT as REPORT_LIGHT
from vectorlab.store import load_json, load_tensors

MODEL = "Qwen--Qwen3-4B-Instruct-2507"
GEMMA = "runs/refusal-google--gemma-3-12b-it"
LIGHT = {**REPORT_LIGHT, "s3": "#1baf7a", "s4": "#eda100", "muted": "#898781", "border": "rgba(11, 11, 11, 0.10)", "neg": "#e34948", "mid": "#f0efec", "pos": "#2a78d6"}
DARK = {**REPORT_DARK, "s3": "#199e70", "s4": "#c98500", "muted": "#898781", "border": "rgba(255, 255, 255, 0.10)", "neg": "#e66767", "mid": "#383835", "pos": "#3987e5"}
ROLES = ("surface", "ink", "ink2", "muted", "grid", "axis", "border", "s1", "s2", "s3", "s4", "neg", "mid", "pos")
CHAR = 0.56

STYLE = """
:root {{ {light} }}
@media (prefers-color-scheme: dark) {{ :root {{ {dark} }} }}
text {{ font-family: {font}; }}
.surface {{ fill: var(--surface); stroke: var(--border); }}
.title {{ fill: var(--ink); font-size: 15px; font-weight: 600; }}
.subtitle, .legend, .axis-title {{ fill: var(--ink2); font-size: 12px; }}
.tick {{ fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
.value {{ fill: var(--ink); font-size: 12px; font-variant-numeric: tabular-nums; }}
.note {{ fill: var(--ink2); font-size: 11px; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.axis {{ stroke: var(--axis); stroke-width: 1; }}
.band {{ fill: var(--muted); opacity: 0.16; }}
.line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.thin {{ stroke-width: 1.5; opacity: 0.55; }}
.ring {{ stroke: var(--surface); stroke-width: 2; }}
.halo {{ fill: none; stroke: var(--ink); stroke-width: 1.5; }}
.s1 {{ stroke: var(--s1); }} .s2 {{ stroke: var(--s2); }} .s3 {{ stroke: var(--s3); }} .s4 {{ stroke: var(--s4); }} .smuted {{ stroke: var(--muted); }}
.f1 {{ fill: var(--s1); }} .f2 {{ fill: var(--s2); }} .f3 {{ fill: var(--s3); }} .f4 {{ fill: var(--s4); }} .fmuted {{ fill: var(--muted); }}
.cell {{ fill: var(--fl); }}
.cell-text {{ fill: var(--tl); font-size: 13px; font-variant-numeric: tabular-nums; }}
.stop-neg {{ stop-color: var(--neg); }} .stop-mid {{ stop-color: var(--mid); }} .stop-pos {{ stop-color: var(--pos); }}
@media (prefers-color-scheme: dark) {{ .cell {{ fill: var(--fd); }} .cell-text {{ fill: var(--td); }} }}
.draw {{ animation: draw 1.6s cubic-bezier(.3, .6, .2, 1) both; }}
.grow {{ transform-box: fill-box; transform-origin: left center; animation: grow .8s cubic-bezier(.2, .7, .2, 1) both; }}
.pop {{ transform-box: fill-box; transform-origin: center; animation: pop .5s cubic-bezier(.2, .7, .2, 1) both; }}
.fade {{ animation: fade .6s ease-out both; }}
@keyframes draw {{ from {{ stroke-dasharray: 1; stroke-dashoffset: 1; }} to {{ stroke-dasharray: 1; stroke-dashoffset: 0; }} }}
@keyframes grow {{ from {{ transform: scaleX(0); }} }}
@keyframes pop {{ from {{ opacity: 0; transform: scale(.3); }} }}
@keyframes fade {{ from {{ opacity: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
"""


def _vars(theme: dict) -> str:
    return " ".join(f"--{role}: {theme[role]};" for role in ROLES)


def esc(text) -> str:
    """Escape text for SVG."""
    return html.escape(str(text))


def text_width(text: str, size: float = 12) -> float:
    """Rough rendered width of ``text`` in the system sans."""
    return len(text) * size * CHAR


def delay(seconds: float) -> str:
    """A style attribute that delays an element's entry animation."""
    return f' style="animation-delay: {seconds:.2f}s"'


def document(width: int, height: int, title: str, subtitle: str, body: list[str], description: str) -> str:
    """A standalone SVG: style, surface card, title, subtitle, and body elements."""
    style = STYLE.format(light=_vars(LIGHT), dark=_vars(DARK), font=FONT)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'
        f"<title>{esc(title)}</title><desc>{esc(description)}</desc><style>{style}</style>"
        f'<rect class="surface" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="12"/>'
        f'<text class="title" x="24" y="32">{esc(title)}</text><text class="subtitle" x="24" y="52">{esc(subtitle)}</text>'
        + "".join(body)
        + "</svg>"
    )


def linear(domain: tuple[float, float], extent: tuple[float, float]):
    """A linear scale from ``domain`` to pixel ``extent``."""
    (d0, d1), (r0, r1) = domain, extent
    return lambda v: r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def logarithmic(domain: tuple[float, float], extent: tuple[float, float]):
    """A base-10 log scale from ``domain`` to pixel ``extent``."""
    scale = linear((math.log10(domain[0]), math.log10(domain[1])), extent)
    return lambda v: scale(math.log10(v))


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Round tick values inside ``[lo, hi]``, about ``count`` of them, on a 1-2-2.5-5 step."""
    step = 10 ** math.floor(math.log10((hi - lo) / count))
    step *= next(m for m in (1, 2, 2.5, 5, 10) if (hi - lo) / (m * step) <= count)
    start = math.ceil(lo / step) * step
    return [round(start + i * step, 10) for i in range(int((hi - start) / step + 1e-9) + 1)]


def fixed(ticks: list[float]) -> Callable[[float], str]:
    """A formatter with the fewest decimals that shows every tick in ``ticks`` exactly."""
    decimals = next(d for d in range(6) if all(abs(round(t, d) - t) < 1e-9 for t in ticks))
    return lambda v: f"{v:.{decimals}f}".replace("-", "−")


def axes(x, y, xticks, yticks, plot, xtitle, ytitle, xfmt=None) -> list[str]:
    """Hairline gridlines at the y ticks, a baseline, tick labels, and axis titles."""
    left, top, right, bottom = plot
    xfmt, yfmt = xfmt or fixed(xticks), fixed(yticks)
    out = [f'<line class="grid" x1="{left}" x2="{right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>' for v in yticks]
    out.append(f'<line class="axis" x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}"/>')
    out += [f'<text class="tick" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{yfmt(v)}</text>' for v in yticks]
    out += [f'<text class="tick" x="{x(v):.1f}" y="{bottom + 18}" text-anchor="middle">{xfmt(v)}</text>' for v in xticks]
    out.append(f'<text class="axis-title" x="{(left + right) / 2:.1f}" y="{bottom + 38}" text-anchor="middle">{esc(xtitle)}</text>')
    out.append(f'<text class="axis-title" transform="translate({left - 44} {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">{esc(ytitle)}</text>')
    return out


def legend(items: list[tuple[str, str, str]], x: float, y: float) -> list[str]:
    """A row of keys: ``(label, role, kind)`` where kind is ``line`` or ``dot``."""
    out = []
    for label, role, kind in items:
        if kind == "line":
            out.append(f'<line class="line s{role}" x1="{x}" x2="{x + 16}" y1="{y - 4}" y2="{y - 4}"/>')
        else:
            out.append(f'<circle class="f{role}" cx="{x + 8}" cy="{y - 4}" r="4.5"/>')
        out.append(f'<text class="legend" x="{x + 22}" y="{y}">{esc(label)}</text>')
        x += 22 + text_width(label) + 20
    return out


def path(points) -> str:
    """An SVG path through ``points``."""
    return "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in points)


def run(name: str) -> Path:
    """The Qwen3-4B run directory for a behavior."""
    return Path("runs") / f"{name}-{MODEL}"


def separation_figure(behaviors: list[str]) -> str:
    """Held-out Cohen's d at every layer for each behavior's direction, as lines that draw in."""
    curves = {b: load_json(run(b) / "separation.json")["cohens_d"] for b in behaviors}
    width, height = 760, 400
    plot = (64, 96, width - 28, height - 60)
    n = len(next(iter(curves.values())))
    top_value = math.ceil(max(max(c) for c in curves.values()) / 2) * 2
    x = linear((0, n - 1), (plot[0], plot[2]))
    y = linear((0, top_value), (plot[3], plot[1]))
    body = axes(x, y, range(0, n, 5), range(0, top_value + 1, 2), plot, "layer", "Cohen's d (held-out)")
    body += legend([(b, str(i + 1), "line") for i, b in enumerate(behaviors)], 24, 78)
    for i, (b, d) in enumerate(curves.items()):
        body.append(f'<path class="line s{i + 1} draw" pathLength="1" d="{path((x(l), y(v)) for l, v in enumerate(d))}"{delay(0.25 * i)}/>')
        peak = int(np.argmax(d))
        body.append(f'<circle class="f{i + 1} ring pop" cx="{x(peak):.1f}" cy="{y(d[peak]):.1f}" r="4.5"{delay(1.4 + 0.25 * i)}/>')
    subtitle = "Qwen3-4B-Instruct-2507 · separation of the two classes along each layer's direction · dots mark each peak"
    return document(width, height, "Where each behavior is linearly readable", subtitle, body, "Line chart of held-out Cohen's d by layer for " + ", ".join(behaviors) + ".")


def similarity_figure(behaviors: list[str], highlight: int = 3) -> str:
    """Per-layer cosine between behavior directions; the strongest pairs in color, the rest in gray."""
    directions = {b: load_tensors(run(b) / "directions.safetensors")["directions"].float().numpy() for b in behaviors}
    pairs = {f"{a} × {b}": (directions[a] * directions[b]).sum(-1) for i, a in enumerate(behaviors) for b in behaviors[i + 1 :]}
    ranked = sorted(pairs, key=lambda k: -np.abs(pairs[k]).mean())
    strong, rest = ranked[:highlight], ranked[highlight:]
    width, height = 760, 400
    plot = (64, 96, width - 28, height - 60)
    n = len(next(iter(pairs.values())))
    hidden = next(iter(directions.values())).shape[-1]
    band = 2 / math.sqrt(hidden)
    x = linear((0, n - 1), (plot[0], plot[2]))
    y = linear((-0.6, 0.6), (plot[3], plot[1]))
    body = [f'<rect class="band" x="{plot[0]}" y="{y(band):.1f}" width="{plot[2] - plot[0]}" height="{y(-band) - y(band):.1f}"/>']
    body += axes(x, y, range(0, n, 5), (-0.6, -0.3, 0, 0.3, 0.6), plot, "layer", "cosine")
    body += legend([(k, str(i + 1), "line") for i, k in enumerate(strong)] + [("other pairs", "muted", "line")], 24, 78)
    for k in rest:
        body.append(f'<path class="line thin smuted draw" pathLength="1" d="{path((x(l), y(v)) for l, v in enumerate(pairs[k]))}"/>')
    for i, k in enumerate(strong):
        body.append(f'<path class="line s{i + 1} draw" pathLength="1" d="{path((x(l), y(v)) for l, v in enumerate(pairs[k]))}"{delay(0.5 + 0.3 * i)}/>')
    subtitle = f"Cosine between two behaviors' directions at the same layer · gray band ±{band:.2f}: typical for random directions"
    return document(width, height, "Directions that overlap", subtitle, body, "Line chart of per-layer cosine similarity between behavior directions.")


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(u + (v - u) * t):02x}" for u, v in zip(ca, cb))


def _diverging(theme: dict, t: float) -> str:
    """Color for ``t`` in [-1, 1] on the theme's red–gray–blue scale."""
    return _mix(theme["mid"], theme["pos"] if t > 0 else theme["neg"], min(1.0, abs(t)))


def _ink_on(fill: str) -> str:
    r, g, b = (int(fill[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"


def _fmt(metric: str, value: float, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return format(value, f"{sign}.0f" if metric == "verbosity" else f"{sign}.2f")


def cross_effect_figure(compare: dict, labels: dict[str, str], row_labels: dict[str, str]) -> str:
    """The cross-effect matrix: cells colored by standardized change, fading in row by row.

    ``labels`` renames behaviors in the column headers; ``row_labels`` names the edit rows.
    """
    names = compare["behaviors"]
    cells = {(c["edit"], c["behavior"]): c for c in compare["cells"]}
    zmax = max(0.5, max(abs(c["standardized"]) for c in compare["cells"]))
    rows = {e: row_labels.get(e, f"{labels.get(e, e)} edit") for e in names}
    cw, ch, top = 170, 62, 120
    left = 36 + max(text_width(label) for label in rows.values())
    width, height = left + cw * len(names) + 140, top + ch * len(names) + 56
    body = [f'<text class="axis-title" x="{left + cw * len(names) / 2}" y="{top - 34}" text-anchor="middle">scored on each behavior\'s held-out prompts, with its own metric</text>']
    body += [f'<text class="legend" x="{left + cw * (j + 0.5)}" y="{top - 12}" text-anchor="middle">{esc(labels.get(b, b))}</text>' for j, b in enumerate(names)]
    for i, e in enumerate(names):
        body.append(f'<text class="legend" x="{left - 12}" y="{top + ch * (i + 0.5) + 4}" text-anchor="end">{esc(rows[e])}</text>')
        for j, b in enumerate(names):
            c = cells[e, b]
            t = c["standardized"] / zmax
            fl, fd = _diverging(LIGHT, t), _diverging(DARK, t)
            style = f"--fl: {fl}; --fd: {fd}; --tl: {_ink_on(fl)}; --td: {_ink_on(fd)}; animation-delay: {0.15 * (i * len(names) + j):.2f}s"
            cx, cy = left + cw * j, top + ch * i
            body.append(
                f'<g class="fade" style="{style}"><rect class="cell" x="{cx + 1}" y="{cy + 1}" width="{cw - 2}" height="{ch - 2}" rx="4"/>'
                f'<text class="cell-text" x="{cx + cw / 2}" y="{cy + ch / 2 - 3}" text-anchor="middle">{_fmt(c["metric"], c["delta"], True)}</text>'
                f'<text class="cell-text" x="{cx + cw / 2}" y="{cy + ch / 2 + 14}" text-anchor="middle" style="font-size: 11px">±{_fmt(c["metric"], c["ci"])}</text></g>'
            )
    gx, gy, gh = left + cw * len(names) + 36, top, ch * len(names)
    body.append(
        '<defs><linearGradient id="scale" x1="0" x2="0" y1="0" y2="1"><stop class="stop-pos" offset="0"/>'
        '<stop class="stop-mid" offset="0.5"/><stop class="stop-neg" offset="1"/></linearGradient></defs>'
        f'<rect x="{gx}" y="{gy}" width="10" height="{gh}" rx="3" fill="url(#scale)"/>'
        f'<text class="tick" x="{gx + 16}" y="{gy + 9}">+{zmax:.2f} SD</text>'
        f'<text class="tick" x="{gx + 16}" y="{gy + gh / 2 + 4}">0</text>'
        f'<text class="tick" x="{gx + 16}" y="{gy + gh}">−{zmax:.2f} SD</text>'
    )
    subtitle = "Mean change ± paired 95% CI vs the original model · color: change in SDs of the original's scores (blue up, red down)"
    return document(width, height, "Cross-effects of each edit", subtitle, body, "Heatmap of each edit's effect on every behavior's metric.")


def bar_panel(rows, values, x0, width, top, row_height, title, fmt, roles, start=0.0) -> list[str]:
    """Horizontal bars growing from a shared baseline, with each value at the bar's tip."""
    scale = linear((0, max(values) * 1.15), (x0, x0 + width))
    out = [f'<text class="legend" x="{x0}" y="{top - 14}" style="font-weight: 600">{esc(title)}</text>']
    out.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="{top - 4}" y2="{top + row_height * len(rows)}"/>')
    for i, (value, role) in enumerate(zip(values, roles)):
        y = top + row_height * i + (row_height - 16) / 2
        length = max(scale(value) - x0, 1)
        out.append(f'<path class="f{role} grow" d="M{x0},{y} h{length - 4:.1f} a4,4 0 0 1 4,4 v8 a4,4 0 0 1 -4,4 h{-(length - 4):.1f} z"{delay(start + 0.12 * i)}/>')
        out.append(f'<text class="value fade" x="{x0 + length + 6:.1f}" y="{y + 12}"{delay(start + 0.6 + 0.12 * i)}>{fmt(value)}</text>')
    return out


def gemma_figure() -> str:
    """Gemma 3 12B refusal benchmark: refusal rate and KL per variant, as two bar panels."""
    gemma = Path(GEMMA)
    base = load_json(gemma / "eval.json")
    variants = {path.parent.name: load_json(path) for path in gemma.glob("evals/*/eval.json")}
    rows = [
        ("original", base["before"], 0.0, "muted"),
        ("trial 2 · hooks", variants["hooks"]["after"], variants["hooks"]["kl"], "1"),
        ("trial 2 · weights", base["after"], base["kl"], "1"),
        ("trial 2 · weights, norms kept", variants["weights-norms"]["after"], variants["weights-norms"]["kl"], "1"),
        ("trial 29 · weights", variants["trial29-weights"]["after"], variants["trial29-weights"]["kl"], "1"),
        ("Heretic (p-e-w)", variants["p-e-w--gemma-3-12b-it-heretic"]["after"], variants["p-e-w--gemma-3-12b-it-heretic"]["kl"], "2"),
    ]
    row_height, top, label_width = 34, 120, 210
    width, height = 780, top + row_height * len(rows) + 28
    body = legend([("vectorlab edit", "1", "dot"), ("Heretic", "2", "dot"), ("original model", "muted", "dot")], 24, 78)
    body += [f'<text class="legend" x="{24 + label_width - 12}" y="{top + row_height * i + row_height / 2 + 4}" text-anchor="end">{esc(name)}</text>' for i, (name, *_rest) in enumerate(rows)]
    roles = [r[3] for r in rows]
    body += bar_panel([r[0] for r in rows], [r[1] for r in rows], 24 + label_width, 230, top, row_height, f"refusal rate, held-out (n={base['n']})", lambda v: f"{v:.1%}", roles)
    body += bar_panel([r[0] for r in rows], [r[2] for r in rows], 24 + label_width + 300, 190, top, row_height, "KL on neutral prompts", lambda v: f"{v:.3f}", roles, start=0.4)
    subtitle = "google/gemma-3-12b-it · vectorlab ablation edits vs Heretic, scored with the same phrase metric and KL"
    return document(width, height, "Gemma 3 12B refusal benchmark", subtitle, body, "Bar charts of refusal rate and KL for Gemma 3 12B edit variants and Heretic.")


def scatter_trials(series, baseline, xlabel_unit, ylabel, title, subtitle, description, marked=None) -> str:
    """Search trials as points on a log-KL axis, popping in in trial order, with a baseline line.

    Args:
        series: ``(label, role, trials)`` per search; each trial is a search.json trial dict.
        marked: ``(series index, trial number, label)`` for one trial to ring and label.
    """
    width, height = 760, 420
    plot = (68, 96, width - 28, height - 60)
    trials = [t for _, _, ts in series for t in ts]
    lo = 10 ** math.floor(math.log10(min(t["kl"] for t in trials)))
    hi = 1.5 * max(t["kl"] for t in trials)
    scores = [t["score"] for t in trials] + [baseline]
    span = max(scores) - min(scores)
    y0, y1 = min(scores) - 0.08 * span, max(scores) + 0.08 * span
    x = logarithmic((lo, hi), (plot[0], plot[2]))
    y = linear((y0, y1), (plot[3], plot[1]))
    xticks = [10**e for e in range(round(math.log10(lo)), math.floor(math.log10(hi)) + 1)]
    body = axes(x, y, xticks, nice_ticks(y0, y1), plot, f"KL on neutral prompts ({xlabel_unit})", ylabel, lambda v: f"{v:g}")
    body.append(f'<line class="axis" x1="{plot[0]}" x2="{plot[2]}" y1="{y(baseline):.1f}" y2="{y(baseline):.1f}" style="stroke: var(--ink2)"/>')
    body.append(f'<text class="note" x="{plot[2] - 4}" y="{y(baseline) - 6:.1f}" text-anchor="end">original model {baseline:.3g}</text>')
    body += legend([(label, role, "dot") for label, role, _ in series], 24, 78)
    count = 0
    for k, (label, role, ts) in enumerate(series):
        for t in sorted(ts, key=lambda t: t["number"]):
            body.append(f'<circle class="f{role} ring pop" cx="{x(t["kl"]):.1f}" cy="{y(t["score"]):.1f}" r="4.5"{delay(0.2 + 0.05 * count)}/>')
            count += 1
        front = sorted((t for t in ts if t["pareto"]), key=lambda t: t["kl"])
        if len(front) > 1:
            body.append(f'<path class="line s{role} draw" pathLength="1" d="{path((x(t["kl"]), y(t["score"])) for t in front)}"{delay(0.2 + 0.05 * count)}/>')
    if marked:
        index, number, label = marked
        t = next(t for t in series[index][2] if t["number"] == number)
        body.append(f'<circle class="halo pop" cx="{x(t["kl"]):.1f}" cy="{y(t["score"]):.1f}" r="9"{delay(0.4 + 0.05 * count)}/>')
        body.append(f'<text class="note fade" x="{x(t["kl"]) - 14:.1f}" y="{y(t["score"]) + 20:.1f}" text-anchor="end"{delay(0.4 + 0.05 * count)}>{esc(label)}</text>')
    return document(width, height, title, subtitle, body, description)


def tradeoff_figure(name: str, label: str) -> str:
    """One behavior's search: every trial's score against KL, with the Pareto front and suggested trial."""
    search = load_json(run(name) / "search.json")
    return scatter_trials(
        [("trial (line: Pareto front)", "1", search["trials"])],
        search["baseline"]["score"],
        "log scale",
        label,
        f"{name.capitalize()} search: {len(search['trials'])} trials",
        f"Qwen3-4B-Instruct-2507 · each trial scored on {search['n_prompts']} train prompts · points appear in trial order",
        f"Scatter of {name} search trials, score against KL.",
        (0, search["suggested"], f"suggested and used: trial {search['suggested']}"),
    )


def verbosity_figure() -> str:
    """Both verbosity searches on one axis: the response-mode direction against the instruction-pair direction."""
    response = load_json(run("verbosity") / "search.json")
    instruction = load_json(run("verbosity-instruction") / "search.json")
    edit = load_json(run("verbosity-instruction") / "edit.json")
    used = next(t["number"] for t in instruction["trials"] if same_edit(t["config"], edit))
    return scatter_trials(
        [("answer-text direction", "1", response["trials"]), ("instruction-pair direction", "2", instruction["trials"])],
        response["baseline"]["score"],
        "log scale",
        "mean answer tokens",
        "Two ways to find a verbosity direction",
        f"Qwen3-4B-Instruct-2507 · {len(response['trials'])} search trials each, on the same {response['n_prompts']} train prompts · lines: Pareto fronts",
        "Scatter of verbosity search trials for two direction types, answer length against KL.",
        (1, used, f"used: trial {used} (suggested: {instruction['suggested']})"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    behaviors = ["refusal", "hedging", "sycophancy", "verbosity"]
    compare = load_json(Path("runs") / f"compare-{MODEL}" / "compare.json")
    figures = {
        "separation.svg": separation_figure(behaviors),
        "similarity.svg": similarity_figure(behaviors + ["verbosity-instruction"]),
        "cross-effects.svg": cross_effect_figure(compare, {"verbosity-instruction": "verbosity"}, {"verbosity-instruction": "verbosity edit (instruction pair)"}),
        "gemma-refusal.svg": gemma_figure(),
        "sycophancy-search.svg": tradeoff_figure("sycophancy", "stance mismatch with the reference (judge, 1–5)"),
        "verbosity-searches.svg": verbosity_figure(),
    }
    for name, svg in figures.items():
        (args.out / name).write_text(svg)
        print(args.out / name)


if __name__ == "__main__":
    main()
