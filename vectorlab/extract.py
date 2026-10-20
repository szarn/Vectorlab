"""The ``extract`` command: capture activations and compute per-layer directions."""

import logging
import random
from pathlib import Path

import torch

from vectorlab.behavior import base_prompt, load_behavior, splits
from vectorlab.capture import capture, embedding_means
from vectorlab.directions import (
    cohens_d,
    lexical_separation,
    logit_lens,
    mean_diff,
    midpoints,
    probe_accuracy,
    project,
    token_projections,
)
from vectorlab.model import load
from vectorlab.store import run_dir, save_json, save_tensors

log = logging.getLogger(__name__)


def token_examples(data: dict[str, list[dict]], mode: str, seed: int, k: int = 3) -> list[tuple[dict, str]]:
    """Held-out examples for the token heatmap.

    Response mode shows both answers to the same ``k`` prompts. Prompt mode shows ``k``
    prompts from each class.
    """
    rng = random.Random(seed)
    if mode == "prompt":
        return [(r, "positive") for r in rng.sample(data["test_pos"], k)] + [(r, "negative") for r in rng.sample(data["test_neg"], k)]
    examples = []
    for prompt in rng.sample(sorted({r["prompt"] for r in data["test_neg"]}), k):
        examples.append((next(r for r in data["test_pos"] if r["prompt"] == prompt), "positive"))
        examples.append((next(r for r in data["test_neg"] if r["prompt"] == prompt), "negative"))
    return examples


def extract(spec_path: str | Path, model_id: str, seed: int = 0, batch_size: int = 16) -> tuple[Path, torch.Tensor]:
    """Extract a behavior's directions on a model and write the run directory.

    The run holds:
    - directions and class means;
    - train and held-out projections;
    - per-layer separation (Cohen's d and probe accuracy) plus its lexical baseline;
    - the logit lens;
    - per-token projections for a few held-out examples;
    - the held-out records, and the judge's reference answers when the behavior uses them.

    Returns:
        ``(run_path, cohens_d)``, where ``cohens_d`` has one held-out effect size per layer.
    """
    spec = load_behavior(spec_path)
    mode = spec["capture"]
    data = splits(spec, seed)
    model, tok, adapter = load(model_id)
    acts = {name: capture(tok, adapter, records, mode, batch_size) for name, records in data.items()}
    directions, norms = mean_diff(acts["train_pos"], acts["train_neg"])
    projections = {name: project(a, directions) for name, a in acts.items()}
    separation = cohens_d(projections["test_pos"], projections["test_neg"])
    log.info("fitting %d linear probes", directions.shape[0])
    probes = probe_accuracy(acts)
    lexical = lexical_separation({name: embedding_means(tok, adapter, records, mode) for name, records in data.items()})
    tokens = token_projections(tok, adapter, token_examples(data, mode, seed), mode, directions, midpoints(projections))

    out = run_dir(spec, model_id)
    save_tensors(
        out / "directions.safetensors",
        {
            "directions": directions,
            "diff_norms": norms,
            "mean_pos": acts["train_pos"].mean(1),
            "mean_neg": acts["train_neg"].mean(1),
        },
    )
    save_tensors(out / "projections.safetensors", projections)
    save_json(out / "separation.json", {"cohens_d": separation.tolist(), "probe_acc": probes, "lexical": lexical})
    save_json(out / "logit_lens.json", logit_lens(tok, adapter, directions))
    save_json(out / "tokens.json", tokens)
    info = {
        "model": model_id,
        "seed": seed,
        "behavior": spec,
        "counts": {name: len(records) for name, records in data.items()},
        "train_prompts": sorted({base_prompt(r) for r in data["train_pos"]}),
        "test": {"pos": data["test_pos"], "neg": data["test_neg"]},
    }
    if spec.get("judge_reference"):
        info["references"] = {base_prompt(r): r["answer"] for r in data["train_neg"] + data["test_neg"]}
    save_json(out / "run.json", info)
    log.info("wrote %s", out)
    return out, separation
