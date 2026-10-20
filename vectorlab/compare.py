"""The ``compare`` command: how the behaviors studied on one model relate.

The page has two views:
- direction similarity: the cosine between every pair of behaviors' saved directions at each
  layer, plus behavior-by-behavior matrices at the edit layers;
- cross-effects: each behavior's edit, applied with hooks, scored on every behavior's held-out
  prompts with that behavior's metric, plus each edit's KL on the neutral prompts.

Edited answers, KL values, and judge scores are cached on disk like the original model's
outputs, so rebuilding the page after a change loads no model.
"""

import functools
import hashlib
import html
import itertools
import logging
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from transformers import AutoTokenizer

from vectorlab.behavior import write_jsonl
from vectorlab.chat import answer_ids
from vectorlab.directions import edit_direction
from vectorlab.edit import describe
from vectorlab.evaluate import eval_prompts, neutral_prompts, original_outputs
from vectorlab.figures import kl_figure, matrix_figure, pair_cosine_figure, to_div
from vectorlab.generate import generate
from vectorlab.hooks import intervene
from vectorlab.judge import JUDGE_MODEL, SCALE, judge_scores, references_for
from vectorlab.metrics import MAX_TOKENS, example_scores, kl, next_logprobs
from vectorlab.model import free, load, slug
from vectorlab.report import METRIC_LABELS, before_after, class_labels, page, section
from vectorlab.store import cache_path, cached_json, load_json, load_tensors, optional_json, save_json

EDIT_KEYS = ("layer", "mode", "strength", "layers")

log = logging.getLogger(__name__)


def load_run(run: Path) -> dict:
    """A run's metadata and directions, plus its edit, eval, and search results when present."""
    info = load_json(run / "run.json")
    return {
        "path": run,
        "name": info["behavior"]["name"],
        "info": info,
        "tensors": load_tensors(run / "directions.safetensors"),
        "edit": optional_json(run / "edit.json"),
        "eval": optional_json(run / "eval.json"),
        "search": optional_json(run / "search.json"),
    }


def compare_dir(model_id: str) -> Path:
    """Directory holding the comparison page for a model."""
    return Path("runs") / f"compare-{slug(model_id)}"


def same_edit(a: dict, b: dict) -> bool:
    """Whether two edit configs describe the same intervention."""
    return all(a[k] == b[k] for k in EDIT_KEYS) and bool(a.get("weights")) == bool(b.get("weights"))


def own_effect(run: dict) -> dict | None:
    """The edit's effect on its own behavior, from the run's files.

    The held-out eval is used when it was made with the current edit. Otherwise the search
    trial with the same config is used: it was scored on a small sample of train prompts and
    picked as the best of many trials, so it overstates the held-out effect.
    """
    edit, evaluation, search = run["edit"], run["eval"], run["search"]
    if not edit:
        return None
    if evaluation and evaluation["edit"] and same_edit(evaluation["edit"], edit):
        return {**{k: evaluation[k] for k in ("before", "after", "kl", "n")}, "source": "held-out eval"}
    for trial in (search or {}).get("trials", []):
        if same_edit(trial["config"], edit):
            source = f"search trial {trial['number']}, train prompts (optimistic)"
            return {"before": search["baseline"]["score"], "after": trial["score"], "kl": trial["kl"], "n": search["n_prompts"], "source": source}
    return None


def _digest(r: torch.Tensor) -> str:
    return hashlib.sha1(r.float().numpy().tobytes()).hexdigest()[:12]


def _edited_answers(loaded, r, config, prompts, max_new_tokens, batch_size):
    model, tok, adapter = loaded()
    with intervene(adapter, r, config["mode"], config["layers"], config["strength"]):
        return generate(model, tok, adapter, prompts, max_new_tokens, batch_size)


def _edited_kl(loaded, r, config, ref):
    model, tok, adapter = loaded()
    with intervene(adapter, r, config["mode"], config["layers"], config["strength"]):
        return kl(ref, next_logprobs(model, tok, adapter, neutral_prompts()))


def edited_outputs(runs: list[dict], batch_size: int) -> tuple[dict, dict]:
    """Every edit's answers to every behavior's held-out prompts, and every edit's KL on the neutral prompts.

    Results come from the disk cache when present. The model loads once, and only if
    something is missing, and is released before this function returns.

    Returns:
        ``(answers, divergence)``. ``answers[(edit, behavior)]`` holds the answers, with
        ``edit`` None for the original model. ``divergence[edit]`` is the edit's KL.
    """
    model_id = runs[0]["info"]["model"]
    loaded = functools.cache(lambda: load(model_id))
    edits = {run["name"]: (edit_direction(run["tensors"], run["edit"]["layer"], run["edit"]["mode"]), run["edit"]) for run in runs}
    answers, divergence = {}, {}
    for target in runs:
        prompts = eval_prompts(target["info"])
        tokens = MAX_TOKENS[target["info"]["behavior"]["metric"]]
        answers[None, target["name"]], ref = original_outputs(model_id, prompts, tokens, batch_size, loaded)
        for name, (r, config) in edits.items():
            log.info("%s edit on %s prompts", name, target["name"])
            key = cache_path(model_id, "gen-edit", _digest(r), *(config[k] for k in EDIT_KEYS), prompts, tokens)
            answers[name, target["name"]] = cached_json(key, lambda: _edited_answers(loaded, r, config, prompts, tokens, batch_size))
    for name, (r, config) in edits.items():
        key = cache_path(model_id, "kl-edit", _digest(r), *(config[k] for k in EDIT_KEYS), neutral_prompts())
        divergence[name] = cached_json(key, lambda: _edited_kl(loaded, r, config, ref))
    loaded.cache_clear()
    free()
    return answers, divergence


def scored(runs: list[dict], answers: dict, judge_model: str) -> tuple[dict, dict]:
    """Score every answer set with its behavior's metric, loading the judge at most once.

    Returns:
        ``(scores, lengths)``, keyed like ``answers``: per-answer scores and answer lengths in tokens.
    """
    tok = AutoTokenizer.from_pretrained(runs[0]["info"]["model"])
    judge = functools.cache(lambda: load(judge_model))
    by_name = {run["name"]: run for run in runs}
    scores, lengths = {}, {}
    for (edit, behavior), texts in answers.items():
        info = by_name[behavior]["info"]
        spec, prompts = info["behavior"], eval_prompts(info)
        if spec["metric"] == "judge":
            references = references_for(info, prompts)
            key = cache_path(judge_model, "judge", spec["rubric"], prompts, texts, *([references] if references else []))
            scores[edit, behavior] = cached_json(key, lambda: judge_scores(*judge(), spec["rubric"], prompts, texts, references))
        else:
            scores[edit, behavior] = example_scores(spec["metric"], tok, texts)
        lengths[edit, behavior] = [len(answer_ids(tok, t)) for t in texts]
    judge.cache_clear()
    free()
    return scores, lengths


def change(before: list[float], after: list[float]) -> dict:
    """Mean change with a paired 95% interval, and the change in units of the original's spread."""
    diffs = [a - b for a, b in zip(after, before)]
    spread = stdev(before)
    return {
        "before": mean(before),
        "after": mean(after),
        "delta": mean(diffs),
        "ci": 1.96 * stdev(diffs) / math.sqrt(len(diffs)),
        "standardized": mean(diffs) / spread if spread else 0.0,
        "n": len(diffs),
    }


def cross_effects(runs: list[dict], batch_size: int = 16, judge_model: str = JUDGE_MODEL) -> tuple[dict, list[dict]]:
    """Apply each run's edit with hooks and score it on every run's held-out prompts.

    All generation happens first with one copy of the model. The judge loads afterwards,
    once, for the behaviors whose metric needs it.

    Returns:
        ``(result, rows)``: the summary saved as ``compare.json``, and one row per prompt and
        cell (edit, behavior, prompt, answers, and scores) saved as ``generations.jsonl``.
    """
    answers, divergence = edited_outputs(runs, batch_size)
    scores, lengths = scored(runs, answers, judge_model)
    cells, rows = [], []
    for target, source in itertools.product(runs, runs):
        b, e = target["name"], source["name"]
        cells.append(
            {
                "edit": e,
                "behavior": b,
                "metric": target["info"]["behavior"]["metric"],
                **change(scores[None, b], scores[e, b]),
                "length_before": mean(lengths[None, b]),
                "length_after": mean(lengths[e, b]),
            }
        )
        rows += [
            {"edit": e, "behavior": b, "prompt": p, "before": before, "after": after, "before_score": bs, "after_score": as_}
            for p, before, after, bs, as_ in zip(eval_prompts(target["info"]), answers[None, b], answers[e, b], scores[None, b], scores[e, b])
        ]
    judged = any(run["info"]["behavior"]["metric"] == "judge" for run in runs)
    result = {
        "model": runs[0]["info"]["model"],
        "judge": judge_model if judged else None,
        "batch_size": batch_size,
        "behaviors": [run["name"] for run in runs],
        "edits": {run["name"]: run["edit"] for run in runs},
        "kl": divergence,
        "cells": cells,
    }
    return result, rows


def _value(metric, value, signed=False):
    sign = "+" if signed else ""
    if metric == "refusal":
        return format(value, f"{sign}.1%")
    return format(value, f"{sign}.0f" if metric == "verbosity" else f"{sign}.2f")


def _direction_label(run):
    pos, neg = class_labels(run["info"]["behavior"])
    return f"{pos} − {neg}"


def _header(runs, result):
    facts = f"{html.escape(runs[0]['info']['model'])} · {len(runs)} behaviors"
    if result and result["judge"]:
        facts += f" · judge {html.escape(result['judge'])}"
    return f'<h1>Behavior comparison</h1><p class="facts">{facts}</p>'


def _edits_table(runs):
    head = (
        "<tr><th>behavior</th><th>direction</th><th>capture</th><th>edit</th><th>metric</th>"
        "<th>before</th><th>after</th><th>KL</th><th>n</th><th>source</th></tr>"
    )
    cells = ""
    for run in runs:
        spec, effect = run["info"]["behavior"], own_effect(run)
        edit = html.escape(describe(run["edit"])) if run["edit"] else "–"
        numbers = (
            f"<td>{_value(spec['metric'], effect['before'])}</td><td>{_value(spec['metric'], effect['after'])}</td>"
            f"<td>{effect['kl']:.4f}</td><td>{effect['n']}</td><td class='wrap'>{html.escape(effect['source'])}</td>"
            if effect
            else "<td>–</td>" * 5
        )
        cells += (
            f"<tr><td>{html.escape(run['name'])}</td><td>{html.escape(_direction_label(run))}</td><td>{spec['capture']}</td>"
            f"<td class='wrap'>{edit}</td><td>{METRIC_LABELS[spec['metric']]}</td>{numbers}</tr>"
        )
    return f'<div class="scroll"><table>{head}{cells}</table></div>'


def _similarity_lines(runs, directions, band):
    pairs = {f"{runs[i]['name']} × {runs[j]['name']}": (directions[i] * directions[j]).sum(-1) for i, j in itertools.combinations(range(len(runs)), 2)}
    figures = "".join(to_div(pair_cosine_figure(cosines, title, band)) for title, cosines in pairs.items())
    head = "<tr><th>layer</th>" + "".join(f"<th>{title}</th>" for title in pairs) + "</tr>"
    body = "".join(
        f"<tr><td>{layer}</td>" + "".join(f"<td>{cosines[layer]:+.3f}</td>" for cosines in pairs.values()) + "</tr>"
        for layer in range(directions.shape[1])
    )
    table = f'<details><summary>Table view</summary><div class="scroll"><table>{head}{body}</table></div></details>'
    return f'<div class="grid small">{figures}</div>{table}'


def edit_layers(runs: list[dict]) -> dict[int, list[str]]:
    """Each layer that some run's edit takes its direction from, with the names of those runs."""
    layers = {}
    for run in runs:
        if run["edit"]:
            layers.setdefault(run["edit"]["layer"], []).append(run["name"])
    return dict(sorted(layers.items()))


def _similarity_matrices(runs, directions):
    names = [run["name"] for run in runs]
    figures, tables = "", ""
    for layer, owners in edit_layers(runs).items():
        cosine = directions[:, layer] @ directions[:, layer].T
        text = [[f"{v:.2f}" for v in row] for row in cosine]
        hover = [[f"{a} × {b} at layer {layer}<br>cosine {cosine[i, j]:+.3f}" for j, b in enumerate(names)] for i, a in enumerate(names)]
        title = f"Layer {layer} · edit layer of {', '.join(owners)}"
        figures += to_div(matrix_figure(cosine, names, names, text, hover, 1.0, "cosine", title, 360))
        rows = "".join(f"<tr><td>{a}</td>" + "".join(f"<td>{v:+.3f}</td>" for v in cosine[i]) + "</tr>" for i, a in enumerate(names))
        tables += f"<h3>Layer {layer}</h3><table><tr><th></th>{''.join(f'<th>{n}</th>' for n in names)}</tr>{rows}</table>"
    table = f'<details><summary>Table view</summary><div class="scroll">{tables}</div></details>'
    return f'<div class="grid">{figures}</div>{table}'


def _scale_ends(rubric):
    ends = [line.strip().rstrip(".") for line in rubric.splitlines() if line.strip().startswith((f"{SCALE[0]} =", f"{SCALE[-1]} ="))]
    return f": {html.escape('; '.join(ends))}" if ends else ""


def _metric_notes(runs):
    items = ""
    for run in runs:
        spec = run["info"]["behavior"]
        aim = "raise" if spec["target"] == "max" else "lower"
        scale = _scale_ends(spec["rubric"]) if spec["metric"] == "judge" else ""
        items += f"<li><b>{html.escape(run['name'])}</b>: {METRIC_LABELS[spec['metric']]}{scale}. Its edit aims to {aim} it.</li>"
    return f'<ul class="caption">{items}</ul>'


def _cross_matrix(result):
    names = result["behaviors"]
    cells = {(c["edit"], c["behavior"]): c for c in result["cells"]}
    grid = [[cells[e, b] for b in names] for e in names]
    z = [[c["standardized"] for c in row] for row in grid]
    text = [[f"{_value(c['metric'], c['delta'], True)}<br>±{_value(c['metric'], c['ci'])}" for c in row] for row in grid]
    hover = [
        [
            f"{c['edit']} edit on {c['behavior']} prompts<br>{METRIC_LABELS[c['metric']]} {_value(c['metric'], c['before'])} → "
            f"{_value(c['metric'], c['after'])}<br>change {_value(c['metric'], c['delta'], True)} (95% CI ±{_value(c['metric'], c['ci'])})"
            f"<br>{c['standardized']:+.2f} SD of the original's scores<br>answer tokens {c['length_before']:.0f} → {c['length_after']:.0f}"
            for c in row
        ]
        for row in grid
    ]
    zmax = max(0.5, max(abs(v) for row in z for v in row))
    x = [f"{b} prompts" for b in names]
    y = [f"{e} edit" for e in names]
    fig = matrix_figure(z, x, y, text, hover, zmax, "change / SD", None, 120 + 90 * len(names))
    fig.update_xaxes(title_text="scored on held-out prompts, with that behavior's metric", side="top")
    fig.update_layout(margin=dict(l=110, t=72))
    head = (
        "<tr><th>edit</th><th>scored on</th><th>metric</th><th>before</th><th>after</th><th>change</th>"
        "<th>95% CI</th><th>change / SD</th><th>answer tokens</th><th>n</th></tr>"
    )
    rows = "".join(
        f"<tr><td>{c['edit']}</td><td>{c['behavior']}</td><td>{METRIC_LABELS[c['metric']]}</td>"
        f"<td>{_value(c['metric'], c['before'])}</td><td>{_value(c['metric'], c['after'])}</td>"
        f"<td>{_value(c['metric'], c['delta'], True)}</td><td>±{_value(c['metric'], c['ci'])}</td><td>{c['standardized']:+.2f}</td>"
        f"<td>{c['length_before']:.0f} → {c['length_after']:.0f}</td><td>{c['n']}</td></tr>"
        for row in grid
        for c in row
    )
    table = f'<details><summary>Table view</summary><div class="scroll"><table>{head}{rows}</table></div></details>'
    return to_div(fig) + table


def _kl_chart(runs, result):
    names, values, hover = [], [], []
    for run in runs:
        if result and run["name"] in result["kl"]:
            value, source = result["kl"][run["name"]], "measured here (hooks)"
        elif own_effect(run):
            effect = own_effect(run)
            value, source = effect["kl"], effect["source"]
        else:
            continue
        names.append(run["name"])
        values.append(value)
        hover.append(f"{html.escape(describe(run['edit']))}<br>{html.escape(source)}")
    return to_div(kl_figure(names, values, hover)) if names else ""


def _samples(result, rows, k=3):
    blocks = ""
    for c in result["cells"]:
        if c["edit"] == c["behavior"]:
            continue
        cell_rows = [r for r in rows if r["edit"] == c["edit"] and r["behavior"] == c["behavior"]]
        summary = f"{c['edit']} edit on {c['behavior']} prompts · change {_value(c['metric'], c['delta'], True)} ±{_value(c['metric'], c['ci'])}"
        blocks += f"<details><summary>{html.escape(summary)}</summary>{before_after(cell_rows, c['metric'], k)}</details>"
    return blocks


def render(runs: list[dict], result: dict | None, rows: list[dict]) -> str:
    """The comparison page as HTML."""
    directions = np.stack([run["tensors"]["directions"].float().numpy() for run in runs])
    band = 2 / math.sqrt(directions.shape[-1])
    sections = [
        _header(runs, result),
        section(
            "Edits",
            "Each behavior's direction (positive class minus negative class) and its current edit, with the edit's "
            "effect on its own metric. Rows marked optimistic come from the search, not a held-out eval.",
            _edits_table(runs),
        ),
        section(
            "Direction similarity across layers",
            "Cosine between two behaviors' directions at the same layer, one panel per pair, on a shared −1 to 1 scale. "
            f"The gray band (±{band:.3f}) is where two random directions in {directions.shape[-1]} dimensions would "
            "usually fall. The sign follows the direction labels in the table above. Prompt-mode directions come from "
            "the last prompt token and response-mode directions from the mean over answer tokens, so a cosine "
            "between the two compares different positions.",
            _similarity_lines(runs, directions, band),
        ),
        section(
            "Direction similarity at the edit layers",
            "All pairwise cosines at each layer an edit takes its direction from.",
            _similarity_matrices(runs, directions),
        ),
    ]
    if result:
        sections.append(
            section(
                "Cross-effects",
                "Each row applies one behavior's edit with hooks. Each column scores that edit on another behavior's "
                "held-out prompts with that behavior's metric, against the original model on the same prompts. Cells "
                "show the mean change and its paired 95% interval. Color is the change divided by the standard "
                "deviation of the original model's scores, so columns with different units share one scale: blue "
                "means the metric rose and red means it fell. Hover a cell for the means and answer lengths.",
                _metric_notes([run for run in runs if run["name"] in result["behaviors"]]) + _cross_matrix(result),
            )
        )
    sections.append(
        section(
            "KL divergence",
            "Each edit's KL divergence from the original model's next-token distribution on the neutral prompts. "
            "Edits without cross-effects use the value recorded by their search or eval, on the same prompts.",
            _kl_chart(runs, result),
        )
    )
    if result:
        sections.append(
            section(
                "Cross-effect samples",
                "Answers from the original model and with another behavior's edit, for each off-diagonal cell.",
                _samples(result, rows),
            )
        )
    return page("Behavior comparison", sections)


def compare(cross: list[Path], directions_only: list[Path], batch_size: int = 16, judge_model: str = JUDGE_MODEL) -> tuple[Path, dict | None]:
    """Write ``compare.html`` for runs of one model and return its path with the cross-effects.

    Args:
        cross: Runs that enter both views. Their edits are applied to each other's prompts.
        directions_only: Runs that enter only the direction-similarity view and the KL chart.

    Returns:
        ``(page, result)``. With cross-effect runs, ``compare.json`` and ``generations.jsonl``
        are written beside the page and ``result`` is the summary; otherwise it is None.
    """
    runs = [load_run(path) for path in cross]
    others = [load_run(path) for path in directions_only]
    out = compare_dir((runs + others)[0]["info"]["model"])
    result, rows = cross_effects(runs, batch_size, judge_model) if runs else (None, [])
    if result:
        save_json(out / "compare.json", result)
        write_jsonl(out / "generations.jsonl", rows)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "compare.html"
    path.write_text(render(runs + others, result, rows))
    log.info("wrote %s", path)
    return path, result
