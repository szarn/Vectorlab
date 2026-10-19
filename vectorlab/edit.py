"""The ``edit`` command: choose an intervention for a run."""

from pathlib import Path

from vectorlab.errors import UserError
from vectorlab.store import load_json, load_tensors, save_json


def edit_layers(n_layers: int, layer: int, mode: str, layer_range: str | None) -> list[int]:
    """Layers to intervene on: an explicit ``start:end`` range, else all layers to ablate, or just ``layer`` to add."""
    if not layer_range:
        return list(range(n_layers)) if mode == "ablate" else [layer]
    start, sep, end = layer_range.partition(":")
    if not (sep and start.isdigit() and end.isdigit() and int(start) < int(end) <= n_layers):
        raise UserError(f"--range must look like START:END with 0 <= START < END <= {n_layers}; got {layer_range!r}")
    return list(range(int(start), int(end)))


def edit(run: Path, layer: int, mode: str | None, strength: float | None, layer_range: str | None = None, weights: bool = False) -> dict:
    """Write ``edit.json`` describing the intervention that ``eval`` will apply.

    Args:
        layer: Layer whose direction to use.
        mode: ``"ablate"`` or ``"add"``; defaults to the behavior's ``intervention``.
        strength: Ablation scale (default 1.0) or addition coefficient (required for ``add``).
        layer_range: Optional ``start:end`` range of layers to intervene on.
        weights: Whether to bake the edit into the weights rather than use hooks.
    """
    mode = mode or load_json(run / "run.json")["behavior"]["intervention"]
    n_layers = load_tensors(run / "directions.safetensors")["directions"].shape[0]
    if not 0 <= layer < n_layers:
        raise UserError(f"--layer must be between 0 and {n_layers - 1}; got {layer}")
    if mode == "add" and strength is None:
        raise UserError("--alpha is required with --mode add")
    config = {
        "layer": layer,
        "mode": mode,
        "strength": 1.0 if strength is None else strength,
        "layers": edit_layers(n_layers, layer, mode, layer_range),
        "weights": weights,
    }
    save_json(run / "edit.json", config)
    return config


def edit_from_trial(run: Path, number: int) -> dict:
    """Write ``edit.json`` from a search trial's config."""
    trials = {t["number"]: t for t in load_json(run / "search.json")["trials"]}
    if number not in trials:
        raise UserError(f"no trial {number} in {run / 'search.json'}; trials run from 0 to {max(trials)}")
    config = trials[number]["config"]
    save_json(run / "edit.json", config)
    return config
