"""The ``analyze`` command: a self-contained HTML report for a run.

The report reads only the run directory, never the model, and inlines the Plotly bundle
so it works offline.
"""

import html
import json
import logging
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

from vectorlab.figures import DARK, LIGHT, histogram_figure, separation_figure, to_div
from vectorlab.store import load_json, load_tensors

ASSETS = Path(__file__).parent / "assets"

log = logging.getLogger(__name__)


def _token(text):
    visible = text.replace(" ", "␣").replace("\n", "↵")
    return f"<code>{html.escape(visible)}</code>"


def _section(title, caption, body):
    return f'<section><h2>{title}</h2><p class="caption">{caption}</p>{body}</section>'


def _legend(labels):
    keys = "".join(
        f'<span><span class="key" style="--c: var(--s{i + 1})"></span>{html.escape(label)}</span>'
        for i, label in enumerate(labels)
    )
    return f'<div class="legend">{keys}</div>'


def _histograms(projections, layers, labels):
    figures = "".join(
        to_div(histogram_figure(projections["test_pos"][:, i].numpy(), projections["test_neg"][:, i].numpy(), i, labels))
        for i in layers
    )
    return _legend(labels) + f'<div class="grid small">{figures}</div>'


def _lens_table(layer, entry):
    rows = "".join(
        f"<tr><td>{_token(p[0])}</td><td>{p[1]:+.2f}</td><td>{_token(s[0])}</td><td>{s[1]:+.2f}</td></tr>"
        for p, s in zip(entry["promoted"], entry["suppressed"])
    )
    head = "<tr><th>promoted</th><th>logit</th><th>suppressed</th><th>logit</th></tr>"
    return f'<div class="scroll"><h3>Layer {layer}</h3><table>{head}{rows}</table></div>'


def _separation_table(separation):
    rows = "".join(
        f"<tr><td>{i}</td><td>{d:.3f}</td><td>{a:.3f}</td></tr>"
        for i, (d, a) in enumerate(zip(separation["cohens_d"], separation["probe_acc"]))
    )
    head = "<tr><th>layer</th><th>Cohen's d</th><th>probe accuracy</th></tr>"
    return f'<details><summary>Table view</summary><div class="scroll"><table>{head}{rows}</table></div></details>'


def _header(info, evaluation):
    counts = info["counts"]
    facts = (
        f"{html.escape(info['model'])} · capture: {info['behavior']['capture']} · "
        f"train {counts['train_pos']}+{counts['train_neg']} · "
        f"held-out {counts['test_pos']}+{counts['test_neg']} · seed {info['seed']}"
    )
    body = f'<h1>{html.escape(info["behavior"]["name"])}</h1><p class="facts">{facts}</p>'
    if evaluation:
        edit = evaluation["edit"]
        body += (
            f'<p class="facts">Edit: {edit["mode"]} direction from layer {edit["layer"]} '
            f'over layers {edit["layers"][0]}–{edit["layers"][-1]}, strength {edit["strength"]} · '
            f'{evaluation["metric"]} {evaluation["before"]:.3f} → {evaluation["after"]:.3f} (n={evaluation["n"]}) · '
            f'KL {evaluation["kl"]:.4f}</p>'
        )
    return body


def _page(title, sections):
    css = (ASSETS / "report.css").read_text()
    themes = json.dumps({"light": LIGHT, "dark": DARK})
    theme_js = (ASSETS / "theme.js").read_text()
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{css}</style>"
        f"<script>{get_plotlyjs()}</script></head>"
        f"<body><main>{''.join(sections)}</main>"
        f"<script>const THEMES = {themes};\n{theme_js}</script></body></html>"
    )


def best_layers(separation: dict, edit: dict | None, k: int = 3) -> list[int]:
    """The ``k`` layers with the largest held-out effect size, plus the edit layer if there is one."""
    layers = set(np.argsort(separation["cohens_d"])[::-1][:k].tolist())
    if edit:
        layers.add(edit["layer"])
    return sorted(layers)


def class_labels(spec: dict) -> list[str]:
    """Display names for the positive and negative classes, from their dataset file names."""
    return [Path(spec["positive"]).stem, Path(spec["negative"]).stem]


def _optional_json(path):
    return load_json(path) if path.exists() else None


def analyze(run: Path) -> Path:
    """Write ``report.html`` for a run and return its path."""
    info = load_json(run / "run.json")
    separation = load_json(run / "separation.json")
    lens = load_json(run / "logit_lens.json")
    projections = load_tensors(run / "projections.safetensors")
    edit = _optional_json(run / "edit.json")
    evaluation = _optional_json(run / "eval.json")
    layers = best_layers(separation, edit)
    labels = class_labels(info["behavior"])

    sections = [
        _header(info, evaluation),
        _section(
            "Layer separation",
            "Per layer, how far apart the two classes sit along that layer's direction (held-out examples), "
            "and how well a logistic probe on the full activation separates them.",
            to_div(separation_figure(separation)) + _separation_table(separation),
        ),
        _section(
            "Projection histograms",
            "Held-out projections onto the direction for the top layers by effect size"
            + (" plus the edit layer." if edit else "."),
            _histograms(projections, layers, labels),
        ),
        _section(
            "Logit lens",
            "Each direction passed through the final norm and the unembedding: "
            "tokens the direction most promotes and suppresses.",
            f'<div class="grid">{"".join(_lens_table(i, lens[i]) for i in layers)}</div>',
        ),
    ]
    out = run / "report.html"
    out.write_text(_page(f"{info['behavior']['name']} report", sections))
    log.info("wrote %s", out)
    return out
