"""The ``eval`` command: measure an edit's effect on the behavior metric and on KL."""

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

NEUTRAL = Path("data/neutral.jsonl")

log = logging.getLogger(__name__)


def eval_prompts(info: dict, n: int | None = None) -> list[str]:
    """The run's held-out prompts, sorted, optionally truncated to ``n``."""
    return sorted({r["prompt"] for r in info["test"]["pos"]})[:n]


def neutral_prompts() -> list[str]:
    """Prompts used to measure KL divergence on ordinary inputs."""
    return [r["prompt"] for r in read_jsonl(NEUTRAL)]


def run_with_edit(info: dict, config: dict, prompts: list[str], max_new_tokens: int, batch_size: int):
    """Generate with the original and the edited model, and measure KL on neutral prompts.

    The original model's generations and next-token distributions come from the cache when
    available. The model is released when this function returns.

    Returns:
        ``(before, after, kl, tokenizer)``.
    """
    model_id = info["model"]
    neutral = neutral_prompts()
    model, tok, adapter = load(model_id)
    before = cached_json(
        cache_path(model_id, "gen", prompts, max_new_tokens),
        lambda: generate(model, tok, adapter, prompts, max_new_tokens, batch_size),
    )
    ref = cached_tensor(
        cache_path(model_id, "logprobs", neutral, KL_BATCH),
        lambda: next_logprobs(model, tok, adapter, neutral),
    )
    r = load_tensors(info["run"] / "directions.safetensors")["directions"][config["layer"]]
    log.info("applying %s at layers %s with strength %s", config["mode"], config["layers"], config["strength"])
    with intervene(adapter, r, config["mode"], config["layers"], config["strength"]):
        after = generate(model, tok, adapter, prompts, max_new_tokens, batch_size)
        new = next_logprobs(model, tok, adapter, neutral)
    return before, after, kl(ref, new), tok


def judged(rubric: str, prompts: list[str], *answer_sets: list[str], judge_model: str = JUDGE_MODEL) -> list[list[float]]:
    """Load the judge once and score each set of answers to ``prompts`` on ``rubric``."""
    model, tok, adapter = load(judge_model)
    return [judge_scores(model, tok, adapter, rubric, prompts, answers) for answers in answer_sets]


def evaluate(run: Path, n: int | None = None, batch_size: int = 16, judge_model: str = JUDGE_MODEL) -> dict:
    """Compare the original model with the run's edit and write ``eval.json`` and ``generations.jsonl``.

    For judge metrics the model under test is released before the judge loads, so only one
    model is in memory at a time.

    Returns:
        The mean metric before and after the edit, KL on neutral prompts, ``n``, the edit
        config, and the judge model when one was used.
    """
    info = {**load_json(run / "run.json"), "run": run}
    config = load_json(run / "edit.json")
    spec = info["behavior"]
    metric = spec["metric"]
    prompts = eval_prompts(info, n)

    before, after, kl_value, tok = run_with_edit(info, config, prompts, MAX_TOKENS[metric], batch_size)
    if metric == "judge":
        free()
        before_scores, after_scores = judged(spec["rubric"], prompts, before, after, judge_model=judge_model)
    else:
        before_scores, after_scores = example_scores(metric, tok, before), example_scores(metric, tok, after)

    result = {
        "metric": metric,
        "before": mean(before_scores),
        "after": mean(after_scores),
        "kl": kl_value,
        "n": len(prompts),
        "edit": config,
        **({"judge": judge_model} if metric == "judge" else {}),
    }
    save_json(run / "eval.json", result)
    write_jsonl(
        run / "generations.jsonl",
        [
            {"prompt": p, "before": b, "after": a, "before_score": bs, "after_score": as_}
            for p, b, a, bs, as_ in zip(prompts, before, after, before_scores, after_scores)
        ],
    )
    return result
