"""The ``search`` command: Optuna search over intervention layer, strength, and layer range.

Each trial applies a hook-based edit, generates answers to a sample of train prompts, and
records the behavior score, the mean answer length, and KL on neutral prompts. The study
has two objectives:
- the target-signed behavior score minus ``length_penalty`` times the relative change in answer length (maximized);
- KL (minimized).
Optuna's Pareto front over these two objectives is what the tradeoff plot highlights.
"""

import logging
import random
from pathlib import Path
from statistics import mean

import optuna

from vectorlab.chat import answer_ids
from vectorlab.evaluate import neutral_prompts
from vectorlab.generate import generate
from vectorlab.hooks import intervene
from vectorlab.judge import JUDGE_MODEL, judge_scores
from vectorlab.metrics import KL_BATCH, MAX_TOKENS, example_scores, kl, next_logprobs
from vectorlab.model import load
from vectorlab.store import cache_path, cached_json, cached_tensor, load_json, load_tensors, save_json

WIDTHS = [1, 2, 4, 8]
MULTIPLES = (0.25, 3.0)
STRENGTHS = (0.5, 1.25)

log = logging.getLogger(__name__)


def layer_bounds(n_layers: int) -> tuple[int, int]:
    """Range of source layers the search considers: the middle 60% of the network."""
    return int(0.2 * n_layers), int(0.8 * n_layers)


def suggest(trial: optuna.Trial, mode: str, norms) -> dict:
    """Sample an edit config: the direction's source layer, a strength, and the layers to edit.

    ``add`` edits a window of layers starting at the source layer, with the total addition set
    to ``multiple`` times that layer's mean-difference norm. ``ablate`` edits a contiguous range
    around the source layer with an ablation scale.
    """
    n_layers = len(norms)
    layer = trial.suggest_int("layer", *layer_bounds(n_layers))
    if mode == "add":
        width = trial.suggest_categorical("width", WIDTHS)
        multiple = trial.suggest_float("multiple", *MULTIPLES)
        layers = list(range(layer, min(n_layers, layer + width)))
        strength = multiple * norms[layer].item() / len(layers)
    else:
        start = round(trial.suggest_float("start_frac", 0.0, 1.0) * layer)
        end = layer + 1 + round(trial.suggest_float("end_frac", 0.0, 1.0) * (n_layers - layer - 1))
        strength = trial.suggest_float("strength", *STRENGTHS)
        layers = list(range(start, end))
    return {"layer": layer, "mode": mode, "strength": strength, "layers": layers, "weights": False}


def edit_params(config: dict, norms) -> dict | None:
    """Search parameters that reproduce an existing edit, or None if it lies outside the search space."""
    n_layers = len(norms)
    layer, layers = config["layer"], config["layers"]
    low, high = layer_bounds(n_layers)
    if not low <= layer <= high:
        return None
    if config["mode"] == "add":
        width = len(layers)
        multiple = config["strength"] * width / norms[layer].item()
        if layers[0] != layer or width not in WIDTHS or not MULTIPLES[0] <= multiple <= MULTIPLES[1]:
            return None
        return {"layer": layer, "width": width, "multiple": multiple}
    start, end = layers[0], layers[-1] + 1
    if not (start <= layer < end and STRENGTHS[0] <= config["strength"] <= STRENGTHS[1]):
        return None
    return {
        "layer": layer,
        "start_frac": start / layer if layer else 0.0,
        "end_frac": (end - layer - 1) / (n_layers - layer - 1) if n_layers - layer - 1 else 0.0,
        "strength": config["strength"],
    }


def suggested(trials: list[dict], kl_budget: float) -> int:
    """The Pareto trial with the highest objective within the KL budget, else the lowest-KL Pareto trial."""
    front = [t for t in trials if t["pareto"]]
    within = [t for t in front if t["kl"] <= kl_budget]
    return max(within, key=lambda t: t["objective"])["number"] if within else min(front, key=lambda t: t["kl"])["number"]


def mean_length(tok, texts: list[str]) -> float:
    """Mean answer length in tokens."""
    return mean(len(answer_ids(tok, t)) for t in texts)


def penalized(score: float, length_change: float, spec: dict) -> float:
    """The search objective: the score signed so higher is better, minus the length-change penalty."""
    sign = 1.0 if spec["target"] == "max" else -1.0
    return sign * score - spec.get("length_penalty", 0) * abs(length_change)


def search(
    run: Path,
    trials: int = 40,
    n_prompts: int = 32,
    batch_size: int = 32,
    judge_model: str = JUDGE_MODEL,
    seed: int = 0,
    kl_budget: float = 0.2,
) -> dict:
    """Run the search, write ``search.json``, and return its contents.

    When the run already has an edit that fits the search space, it is evaluated first, so the
    search starts from a known-good point. The judge, when the metric needs one, stays loaded
    beside the model under test. It is a different model, so the original is never held twice.
    """
    info = load_json(run / "run.json")
    spec = info["behavior"]
    model_id = info["model"]
    tensors = load_tensors(run / "directions.safetensors")
    prompts = sorted(random.Random(seed).sample(info["train_prompts"], min(n_prompts, len(info["train_prompts"]))))
    neutral = neutral_prompts()
    tokens = MAX_TOKENS[spec["metric"]]

    model, tok, adapter = load(model_id)
    judge = load(judge_model) if spec["metric"] == "judge" else None

    def scores(texts):
        if judge:
            return judge_scores(*judge, spec["rubric"], prompts, texts)
        return example_scores(spec["metric"], tok, texts)

    before = cached_json(cache_path(model_id, "gen", prompts, tokens), lambda: generate(model, tok, adapter, prompts, tokens, batch_size))
    ref = cached_tensor(cache_path(model_id, "logprobs", neutral, KL_BATCH), lambda: next_logprobs(model, tok, adapter, neutral))
    baseline = {"score": mean(scores(before)), "length": mean_length(tok, before)}

    def objective(trial):
        config = suggest(trial, spec["intervention"], tensors["diff_norms"])
        with intervene(adapter, tensors["directions"][config["layer"]], config["mode"], config["layers"], config["strength"]):
            texts = generate(model, tok, adapter, prompts, tokens, batch_size)
            divergence = kl(ref, next_logprobs(model, tok, adapter, neutral))
        score = mean(scores(texts))
        length = mean_length(tok, texts)
        change = length / baseline["length"] - 1
        trial.set_user_attr("config", config)
        trial.set_user_attr("score", score)
        trial.set_user_attr("length", length)
        trial.set_user_attr("length_change", change)
        log.info("trial %d: layer %d score %.3f kl %.3f length %+.0f%%", trial.number, config["layer"], score, divergence, 100 * change)
        return penalized(score, change, spec), divergence

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.TPESampler(seed=seed))
    start = edit_params(load_json(run / "edit.json"), tensors["diff_norms"]) if (run / "edit.json").exists() else None
    if start and load_json(run / "edit.json")["mode"] == spec["intervention"]:
        log.info("starting from the current edit: %s", start)
        study.enqueue_trial(start)
    study.optimize(objective, n_trials=trials)
    front = {t.number for t in study.best_trials}

    result = {
        "metric": spec["metric"],
        "target": spec["target"],
        "length_penalty": spec.get("length_penalty", 0),
        "judge": judge_model if judge else None,
        "n_prompts": len(prompts),
        "baseline": baseline,
        "trials": [
            {
                "number": t.number,
                "config": t.user_attrs["config"],
                "score": t.user_attrs["score"],
                "kl": t.values[1],
                "length": t.user_attrs["length"],
                "length_change": t.user_attrs["length_change"],
                "objective": t.values[0],
                "pareto": t.number in front,
            }
            for t in study.trials
        ],
    }
    result["kl_budget"] = kl_budget
    result["suggested"] = suggested(result["trials"], kl_budget)
    save_json(run / "search.json", result)
    return result
