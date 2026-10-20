"""The ``analyze`` command: a self-contained HTML report for a run.

The report reads only the run directory, never the model, and inlines the Plotly bundle
so it works offline.
"""

import html
import json
import logging
from pathlib import Path
from statistics import mean

import numpy as np
from plotly.offline import get_plotlyjs

from vectorlab.behavior import read_jsonl
from vectorlab.directions import MASSIVE_RATIO, massive_dims
from vectorlab.edit import describe
from vectorlab.figures import DARK, LIGHT, cosine_figure, histogram_figure, separation_figure, to_div, tradeoff_figure
from vectorlab.store import load_json, load_tensors, optional_json

ASSETS = Path(__file__).parent / "assets"
METRIC_LABELS = {"refusal": "refusal rate", "verbosity": "answer tokens", "judge": "judge score (1–5)"}

log = logging.getLogger(__name__)


def _token(text):
    visible = text.replace(" ", "␣").replace("\n", "↵")
    return f"<code>{html.escape(visible)}</code>"


def section(title: str, caption: str, body: str) -> str:
    """A report card: heading, caption, and HTML body."""
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
    lexical = separation["lexical"]
    rows += f"<tr><td>lexical baseline</td><td>{lexical['cohens_d']:.3f}</td><td>{lexical['probe_acc']:.3f}</td></tr>"
    head = "<tr><th>layer</th><th>Cohen's d</th><th>probe accuracy</th></tr>"
    return f'<details><summary>Table view</summary><div class="scroll"><table>{head}{rows}</table></div></details>'


def _token_scale(examples, layer):
    values = [
        abs(e["scores"][layer][i])
        for e in examples
        for i in range(*e["content"])
        if not e["special"][i]
    ]
    return float(np.percentile(values, 95)) or 1.0


def _token_line(example, layer, scale):
    start, end = example["content"]
    spans = []
    for i, (text, special) in enumerate(zip(example["tokens"], example["special"])):
        shown = html.escape(text)
        if special or not start <= i < end:
            spans.append(f'<span class="tok muted">{shown}</span>')
            continue
        value = example["scores"][layer][i]
        weight = min(1.0, abs(value) / scale)
        side = "pos" if value > 0 else "neg"
        spans.append(f'<span class="tok {side}" style="--w:{weight:.2f}" title="{value:+.2f}">{shown}</span>')
    return f'<div class="tokens">{"".join(spans)}</div>'


def _token_heatmap(examples, layer, labels):
    scale = _token_scale(examples, layer)
    legend = (
        f'<div class="scale">{html.escape(labels[1])}<span class="bar"></span>{html.escape(labels[0])}'
        f"<span>· layer {layer} · color saturates at ±{scale:.1f}</span></div>"
    )
    blocks = "".join(f"<h3>{html.escape(labels[0] if e['label'] == 'positive' else labels[1])} example</h3>{_token_line(e, layer, scale)}" for e in examples)
    return legend + blocks


def _mean_words(rows, key):
    return mean(len(r[key].split()) for r in rows)


def before_after(rows: list[dict], metric: str, k: int = 6) -> str:
    """Mean scores and lengths before and after, then the first ``k`` answer pairs side by side."""
    summary = (
        f'<p class="facts">Mean {METRIC_LABELS[metric]}: {mean(r["before_score"] for r in rows):.3f} → '
        f'{mean(r["after_score"] for r in rows):.3f} · mean answer length {_mean_words(rows, "before"):.0f} → {_mean_words(rows, "after"):.0f} words '
        f"· showing {min(k, len(rows))} of {len(rows)} held-out prompts</p>"
    )
    pairs = "".join(
        f'<div class="pair"><div class="q">{html.escape(r["prompt"])}</div><div class="cols">'
        f'<div><h4>Before · {r["before_score"]:.2f}</h4><div class="gen">{html.escape(r["before"])}</div></div>'
        f'<div><h4>After · {r["after_score"]:.2f}</h4><div class="gen">{html.escape(r["after"])}</div></div>'
        "</div></div>"
        for r in rows[:k]
    )
    return summary + pairs


def _judge_table(judges):
    head = "<tr><th>edit</th><th>judge</th><th>κ vs hand labels</th><th>before</th><th>after</th><th>change</th><th>n</th></tr>"
    rows = "".join(
        f"<tr><td>{html.escape(j['edit'])}</td><td>{html.escape(j['judge'])}</td>"
        f"<td>{'–' if j['kappa'] is None else format(j['kappa'], '.2f')}</td><td>{j['before']:.2f}</td>"
        f"<td>{j['after']:.2f}</td><td>{j['after'] - j['before']:+.2f}</td><td>{j['n']}</td></tr>"
        for j in judges
    )
    return f'<div class="scroll"><table>{head}{rows}</table></div>'


def _variants_table(model_id, evaluation, variants, metric):
    head = f"<tr><th>variant</th><th>model</th><th>edit</th><th>{METRIC_LABELS[metric]}</th><th>KL</th><th>n</th></tr>"
    first = evaluation or variants[0][1]
    rows = [("original", {"edit": None, "after": first["before"], "kl": 0.0, "n": first["n"]})]
    if evaluation:
        rows.append(("current edit", evaluation))
    rows += variants
    cells = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(e.get('model', model_id))}</td>"
        f"<td>{html.escape(describe(e['edit'])) if e['edit'] else '–'}</td>"
        f"<td>{e['after']:.3f}</td><td>{e['kl']:.4f}</td><td>{e['n']}</td></tr>"
        for name, e in rows
    )
    return f'<div class="scroll"><table>{head}{cells}</table></div>'


def _massive_note(tensors, edit):
    if not edit or edit["mode"] != "ablate":
        return ""
    layer = edit["layer"]
    mask = massive_dims((tensors["mean_pos"][layer] + tensors["mean_neg"][layer]) / 2)
    if not mask.any():
        return ""
    share = (tensors["directions"][layer][mask] ** 2).sum().item()
    return (
        f'<p class="facts">The ablation direction leaves out {int(mask.sum())} massive-activation dimension(s) '
        f"(mean magnitude over {MASSIVE_RATIO:.0f}× the median), which held {share:.0%} of the layer-{layer} "
        "direction's squared length.</p>"
    )


def _header(info, evaluation):
    counts = info["counts"]
    facts = (
        f"{html.escape(info['model'])} · capture: {info['behavior']['capture']} · "
        f"train {counts['train_pos']}+{counts['train_neg']} · "
        f"held-out {counts['test_pos']}+{counts['test_neg']} · seed {info['seed']}"
    )
    body = f'<h1>{html.escape(info["behavior"]["name"])}</h1><p class="facts">{facts}</p>'
    if evaluation:
        judge = f" (judge: {html.escape(evaluation['judge'])})" if evaluation.get("judge") else ""
        body += (
            f'<p class="facts">Edit: {html.escape(describe(evaluation["edit"]))} · '
            f'{evaluation["metric"]}{judge} {evaluation["before"]:.3f} → {evaluation["after"]:.3f} (n={evaluation["n"]}) · '
            f'KL {evaluation["kl"]:.4f}</p>'
        )
    return body


def page(title: str, sections: list[str]) -> str:
    """A self-contained report page: shared CSS, the inlined Plotly bundle, and the theme script."""
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


def analyze(run: Path) -> Path:
    """Write ``report.html`` for a run and return its path.

    Sections whose inputs are missing (search, eval, evaluated variants, rescored judges) are skipped.
    """
    info = load_json(run / "run.json")
    spec = info["behavior"]
    separation = load_json(run / "separation.json")
    lens = load_json(run / "logit_lens.json")
    projections = load_tensors(run / "projections.safetensors")
    tensors = load_tensors(run / "directions.safetensors")
    directions = tensors["directions"].numpy()
    edit = optional_json(run / "edit.json")
    evaluation = optional_json(run / "eval.json")
    search = optional_json(run / "search.json")
    judges = optional_json(run / "judges.json")
    tokens = optional_json(run / "tokens.json")
    variants = [(path.parent.name, load_json(path)) for path in sorted(run.glob("evals/*/eval.json"))]
    layers = best_layers(separation, edit)
    labels = class_labels(spec)
    focus = edit["layer"] if edit else int(np.argmax(separation["cohens_d"]))

    sections = [
        _header(info, evaluation) + _massive_note(tensors, evaluation["edit"] if evaluation else edit),
        section(
            "Layer separation",
            "Per layer, how far apart the two classes sit along that layer's direction (held-out examples), and how well "
            "a logistic probe on the full activation separates them. The orange line is the lexical baseline: the same "
            "statistics on the mean token embedding of each example, before any layer runs. Separation above it is more "
            "than vocabulary.",
            to_div(separation_figure(separation)) + _separation_table(separation),
        ),
        section(
            "Direction similarity across layers",
            "Cosine similarity between the directions extracted at each pair of layers. Blocks of high similarity show "
            "where the representation stays stable.",
            to_div(cosine_figure(directions)),
        ),
        section(
            "Projection histograms",
            "Held-out projections onto the direction for the top layers by effect size" + (" plus the edit layer." if edit else "."),
            _histograms(projections, layers, labels),
        ),
        section(
            "Logit lens",
            "Each direction passed through the final norm and the unembedding: tokens the direction most promotes and suppresses.",
            f'<div class="grid">{"".join(_lens_table(i, lens[i]) for i in layers)}</div>',
        ),
    ]
    if tokens:
        sections.append(
            section(
                "Token heatmap",
                f"Each token's projection onto layer {focus}'s direction, relative to the midpoint between the class means. "
                "Gray tokens are template or outside the captured text. Hover a token for its value.",
                _token_heatmap(tokens, focus, labels),
            )
        )
    if search:
        penalty = f" minus {search['length_penalty']} × |relative length change|" if search["length_penalty"] else ""
        sections.append(
            section(
                "Tradeoff",
                f"Every search trial ({len(search['trials'])} trials on {search['n_prompts']} train prompts). The Pareto front "
                f"is computed on the objective (target-signed score{penalty}) against KL. Hover a point for its settings.",
                to_div(tradeoff_figure(search, METRIC_LABELS[spec["metric"]])),
            )
        )
    if judges:
        sections.append(
            section(
                "Judges",
                "The same before and after answers scored by each judge, with each judge's binary kappa against the "
                "hand-labeled validation set. Only the default judge's numbers appear elsewhere in this report.",
                _judge_table(judges),
            )
        )
    if variants:
        sections.append(
            section(
                "Variants",
                "Held-out results for every evaluated variant: the run's edit, other edit settings, and other models "
                "compared with the same original. KL is always measured against the original on the neutral prompts.",
                _variants_table(info["model"], evaluation, variants, spec["metric"]),
            )
        )
    if (run / "generations.jsonl").exists() and evaluation:
        sections.append(
            section(
                "Before and after",
                "Answers to held-out prompts from the original model and with the edit applied, with each answer's score.",
                before_after(read_jsonl(run / "generations.jsonl"), spec["metric"]),
            )
        )
    out = run / "report.html"
    out.write_text(page(f"{spec['name']} report", sections))
    log.info("wrote %s", out)
    return out
