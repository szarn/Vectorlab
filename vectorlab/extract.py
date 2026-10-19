"""The ``extract`` command: capture activations and compute per-layer directions."""

import logging
from pathlib import Path

import torch

from vectorlab.behavior import load_behavior, splits
from vectorlab.capture import capture
from vectorlab.directions import cohens_d, logit_lens, mean_diff, probe_accuracy, project
from vectorlab.model import load
from vectorlab.store import run_dir, save_json, save_tensors

log = logging.getLogger(__name__)


def extract(spec_path: str | Path, model_id: str, seed: int = 0, batch_size: int = 16) -> tuple[Path, torch.Tensor]:
    """Extract a behavior's directions on a model and write the run directory.

    The run holds directions and class means, train and held-out projections, per-layer
    separation (Cohen's d and probe accuracy), the logit lens, and the held-out records.

    Returns:
        ``(run_path, cohens_d)``, where ``cohens_d`` has one held-out effect size per layer.
    """
    spec = load_behavior(spec_path)
    data = splits(spec, seed)
    model, tok, adapter = load(model_id)
    acts = {name: capture(tok, adapter, records, spec["capture"], batch_size) for name, records in data.items()}
    directions, norms = mean_diff(acts["train_pos"], acts["train_neg"])
    projections = {name: project(a, directions) for name, a in acts.items()}
    separation = cohens_d(projections["test_pos"], projections["test_neg"])
    log.info("fitting %d linear probes", directions.shape[0])
    probes = probe_accuracy(acts)

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
    save_json(out / "separation.json", {"cohens_d": separation.tolist(), "probe_acc": probes})
    save_json(out / "logit_lens.json", logit_lens(tok, adapter, directions))
    save_json(
        out / "run.json",
        {
            "model": model_id,
            "seed": seed,
            "behavior": spec,
            "counts": {name: len(records) for name, records in data.items()},
            "test": {"pos": data["test_pos"], "neg": data["test_neg"]},
        },
    )
    log.info("wrote %s", out)
    return out, separation
