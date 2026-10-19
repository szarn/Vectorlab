"""The ``eval`` command: measure an edit's effect on the behavior metric and on KL."""

import functools
import logging
from pathlib import Path
from statistics import mean

from vectorlab.behavior import read_jsonl, write_jsonl
from vectorlab.generate import generate
from vectorlab.hooks import intervene
from vectorlab.judge import JUDGE_MODEL, judge_scores
from vectorlab.metrics import KL_BATCH, MAX_TOKENS, example_scores, kl, next_logprobs
from vectorlab.model import free, load
from vectorlab.store import cache_path, cached_json, cached_tensor, load_json, load_tensors, save_json
from vectorlab.weights import apply_edit

NEUTRAL = Path("data/neutral.jsonl")

log = logging.getLogger(__name__)


def eval_prompts(info: dict, n: int | None = None) -> list[str]:
    """The run's held-out prompts, sorted, optionally truncated to ``n``."""
    return sorted({r["prompt"] for r in info["test"]["pos"]})[:n]


def neutral_prompts() -> list[str]:
    """Prompts used to measure KL divergence on ordinary inputs."""
    return [r["prompt"] for r in read_jsonl(NEUTRAL)]


def original_outputs(model_id: str, prompts: list[str], max_new_tokens: int, batch_size: int, loaded):
    """The original model's answers to ``prompts`` and its next-token log-probs on the neutral prompts.

    Both come from the disk cache when present. Otherwise ``loaded()`` supplies the original
    model as ``(model, tokenizer, adapter)`` and the results are cached.
    """
    neutral = neutral_prompts()
    before = cached_json(
        cache_path(model_id, "gen", prompts, max_new_tokens),
        lambda: generate(*loaded(), prompts, max_new_tokens, batch_size),
    )
    ref = cached_tensor(cache_path(model_id, "logprobs", neutral, KL_BATCH), lambda: next_logprobs(*loaded(), neutral))
    return before, ref


def outputs(model, tok, adapter, prompts: list[str], max_new_tokens: int, batch_size: int):
    """A model's answers to ``prompts`` and its next-token log-probs on the neutral prompts."""
    return generate(model, tok, adapter, prompts, max_new_tokens, batch_size), next_logprobs(model, tok, adapter, neutral_prompts())


def run_with_edit(info: dict, config: dict, prompts: list[str], max_new_tokens: int, batch_size: int):
    """Generate with the original and the edited model, and measure KL on neutral prompts.

    The original model's outputs come from the cache when available. The edit is applied
    with hooks, or to the weights in place when ``config["weights"]`` is set, so only one
    copy of the model is ever loaded. The model is released when this function returns.

    Returns:
        ``(before, after, kl, tokenizer)``.
    """
    model, tok, adapter = load(info["model"])
    before, ref = original_outputs(info["model"], prompts, max_new_tokens, batch_size, lambda: (model, tok, adapter))
    r = load_tensors(info["run"] / "directions.safetensors")["directions"][config["layer"]]
    log.info("applying %s at layers %s with strength %s", config["mode"], config["layers"], config["strength"])
    if config.get("weights"):
        apply_edit(model, adapter, r, config)
        after, new = outputs(model, tok, adapter, prompts, max_new_tokens, batch_size)
    else:
        with intervene(adapter, r, config["mode"], config["layers"], config["strength"]):
            after, new = outputs(model, tok, adapter, prompts, max_new_tokens, batch_size)
    return before, after, kl(ref, new), tok


def run_external(info: dict, model_id: str, prompts: list[str], max_new_tokens: int, batch_size: int):
    """Compare another model, such as a third-party edit of the run's model, with the run's original model.

    The original's outputs come from the cache. If they are missing, the original is loaded
    to compute them and released before ``model_id`` loads.

    Returns:
        ``(before, after, kl, tokenizer)``, with ``kl`` measured against the original model.
    """
    loader = functools.cache(lambda: load(info["model"]))
    before, ref = original_outputs(info["model"], prompts, max_new_tokens, batch_size, loader)
    loader.cache_clear()
    free()
    model, tok, adapter = load(model_id)
    after, new = outputs(model, tok, adapter, prompts, max_new_tokens, batch_size)
    return before, after, kl(ref, new), tok


def judged(rubric: str, prompts: list[str], *answer_sets: list[str], judge_model: str = JUDGE_MODEL) -> list[list[float]]:
    """Load the judge once and score each set of answers to ``prompts`` on ``rubric``."""
    model, tok, adapter = load(judge_model)
    return [judge_scores(model, tok, adapter, rubric, prompts, answers) for answers in answer_sets]


def evaluate(
    run: Path,
    n: int | None = None,
    batch_size: int = 16,
    judge_model: str = JUDGE_MODEL,
    model_id: str | None = None,
    tag: str | None = None,
) -> tuple[dict, Path]:
    """Compare the original model with the run's edit, or with another model, on held-out prompts.

    Writes ``eval.json`` and ``generations.jsonl`` to the run directory, or to
    ``evals/<tag>/`` when ``tag`` is given. For judge metrics the model under test is
    released before the judge loads, so only one model is in memory at a time.

    Args:
        model_id: Evaluate this model instead of the run's edit, against the run's original model.
        tag: Keep the results under ``evals/<tag>/`` so they sit beside other variants.

    Returns:
        ``(result, directory)``. The result holds the mean metric before and after, KL on
        neutral prompts, ``n``, the model and edit config, and the judge when one was used.
    """
    info = {**load_json(run / "run.json"), "run": run}
    spec = info["behavior"]
    metric = spec["metric"]
    prompts = eval_prompts(info, n)

    if model_id:
        config = None
        before, after, kl_value, tok = run_external(info, model_id, prompts, MAX_TOKENS[metric], batch_size)
    else:
        config = load_json(run / "edit.json")
        before, after, kl_value, tok = run_with_edit(info, config, prompts, MAX_TOKENS[metric], batch_size)
    if metric == "judge":
        free()
        before_scores, after_scores = judged(spec["rubric"], prompts, before, after, judge_model=judge_model)
    else:
        before_scores, after_scores = example_scores(metric, tok, before), example_scores(metric, tok, after)

    result = {
        "metric": metric,
        "model": model_id or info["model"],
        "before": mean(before_scores),
        "after": mean(after_scores),
        "kl": kl_value,
        "n": len(prompts),
        "edit": config,
        **({"judge": judge_model} if metric == "judge" else {}),
    }
    out = run / "evals" / tag if tag else run
    save_json(out / "eval.json", result)
    write_jsonl(
        out / "generations.jsonl",
        [
            {"prompt": p, "before": b, "after": a, "before_score": bs, "after_score": as_}
            for p, b, a, bs, as_ in zip(prompts, before, after, before_scores, after_scores)
        ],
    )
    return result, out
