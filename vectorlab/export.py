"""The ``export`` command: write a run's edit baked into the weights, as safetensors or MLX.

Every export is checked by reloading it and comparing its next-token distributions on a
batch of neutral prompts with the edited model that was saved. The comparison model is
released before the export loads, so two copies are never in memory together.
"""

import contextlib
import io
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from vectorlab.chat import prompt_ids
from vectorlab.directions import edit_direction
from vectorlab.evaluate import neutral_prompts
from vectorlab.metrics import KL_BATCH, kl, next_logprobs
from vectorlab.model import free, load
from vectorlab.store import load_json, load_tensors, save_json
from vectorlab.weights import apply_edit

FORMATS = ("safetensors", "mlx")
MLX_ATTENTION_BIAS = {"llama"}

log = logging.getLogger(__name__)


def default_out(run: Path, fmt: str) -> Path:
    """Where ``export`` writes a format when no output directory is given."""
    return run / f"export-{fmt}"


def save_edited(run: Path, out: Path) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Load the run's model, apply its edit to the weights, and save a Hugging Face checkpoint to ``out``.

    Returns:
        ``(original, edited, ids)``: next-token log-probs of the original and edited model on
        one batch of neutral prompts, and the token ids of those prompts.
    """
    info = load_json(run / "run.json")
    config = load_json(run / "edit.json")
    prompts = neutral_prompts()[:KL_BATCH]
    model, tok, adapter = load(info["model"])
    original = next_logprobs(model, tok, adapter, prompts)
    apply_edit(model, adapter, edit_direction(load_tensors(run / "directions.safetensors"), config["layer"], config["mode"]), config)
    edited = next_logprobs(model, tok, adapter, prompts)
    ids = [prompt_ids(tok, adapter, p) for p in prompts]
    log.info("saving the edited model to %s", out)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(info["model"]).save_pretrained(out)
    return original, edited, ids


def nest_lm_head(hf_dir: Path) -> None:
    """Move a multimodal checkpoint's top-level ``lm_head`` under ``language_model``.

    transformers saves an untied Gemma 3 unembedding as ``lm_head.weight``, but mlx-lm's
    Gemma 3 loader only looks for it inside ``language_model``. Text-only checkpoints are
    left alone.
    """
    import mlx.core as mx

    if "text_config" not in json.loads((hf_dir / "config.json").read_text()):
        return
    for path in sorted(hf_dir.glob("*.safetensors")):
        weights = mx.load(str(path))
        if "lm_head.weight" not in weights:
            continue
        weights["language_model.lm_head.weight"] = weights.pop("lm_head.weight")
        moved = path.with_name(f"moved-{path.name}")
        mx.save_safetensors(str(moved), weights, metadata={"format": "pt"})
        os.replace(moved, path)


def mlx_logprobs(path: Path, ids: list[list[int]]) -> torch.Tensor:
    """Next-token log-probs of an MLX model for each token sequence, one sequence at a time."""
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    model, _ = mlx_load(str(path))
    rows = [np.array(model(mx.array([seq]))[0, -1].astype(mx.float32)) for seq in ids]
    return torch.from_numpy(np.stack(rows)).log_softmax(-1)


def export(run: Path, fmt: str, out: Path) -> dict:
    """Export the run's edit as a safetensors checkpoint or an unquantized MLX model in ``out``.

    MLX conversion starts from a safetensors checkpoint staged in a temporary directory.
    Writes ``vectorlab-export.json`` into ``out`` with the edit and the reload check.

    Returns:
        The check: KL from the in-memory edited model to the export (near zero when the
        export is faithful) and from the original model to the export (the edit's size).
    """
    if fmt == "safetensors":
        original, edited, ids = save_edited(run, out)
        free()
        model, tok, adapter = load(str(out))
        exported = next_logprobs(model, tok, adapter, neutral_prompts()[:KL_BATCH])
    else:
        from mlx_lm import convert

        with tempfile.TemporaryDirectory() as staging:
            original, edited, ids = save_edited(run, Path(staging) / "hf")
            free()
            nest_lm_head(Path(staging) / "hf")
            log.info("converting to MLX in %s", out)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                convert(str(Path(staging) / "hf"), mlx_path=str(out))
            log.info("mlx-lm: %s", printed.getvalue().strip())
        exported = mlx_logprobs(out, ids)
    check = {"kl_edit_to_export": kl(edited, exported), "kl_original_to_export": kl(original, exported), "prompts": len(ids)}
    save_json(out / "vectorlab-export.json", {"run": str(run), "format": fmt, "edit": load_json(run / "edit.json"), "check": check})
    return check
