"""The ``edit`` command: choose an intervention for a run."""

from pathlib import Path

from transformers import AutoConfig

from vectorlab.adapters import ADAPTERS
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


def model_type(run: Path) -> str:
    """The architecture name of the run's model."""
    return AutoConfig.from_pretrained(load_json(run / "run.json")["model"]).model_type


def check_bakeable(run: Path, config: dict) -> None:
    """Raise UserError if ``config`` cannot be baked into the run model's weights."""
    kind = model_type(run)
    if config["mode"] == "add" and not ADAPTERS[kind].bakes_bias:
        raise UserError(
            f"an addition cannot be baked into {kind} weights exactly, because its attention output passes "
            "through a post-norm before joining the residual stream; evaluate it with hooks (omit --weights)"
        )


def _save(run: Path, config: dict) -> dict:
    if config["preserve_norms"] and not (config["weights"] and config["mode"] == "ablate"):
        raise UserError("--preserve-norms applies to weight ablation only; use it with --weights on an ablation")
    if config["weights"]:
        check_bakeable(run, config)
    save_json(run / "edit.json", config)
    return config


def describe(config: dict) -> str:
    """A short human-readable label for an edit config."""
    how = "weights" if config.get("weights") else "hooks"
    if config.get("preserve_norms"):
        how += ", norms preserved"
    layers = config["layers"]
    return f"{config['mode']} layer-{config['layer']} direction at layers {layers[0]}–{layers[-1]}, strength {config['strength']:.3g} ({how})"


def edit(
    run: Path,
    layer: int,
    mode: str | None,
    strength: float | None,
    layer_range: str | None = None,
    weights: bool = False,
    preserve_norms: bool = False,
) -> dict:
    """Write ``edit.json`` describing the intervention that ``eval`` and ``export`` will apply.

    Args:
        layer: Layer whose direction to use.
        mode: ``"ablate"`` or ``"add"``; defaults to the behavior's ``intervention``.
        strength: Ablation scale (default 1.0) or addition coefficient (required for ``add``).
        layer_range: Optional ``start:end`` range of layers to intervene on.
        weights: Whether ``eval`` edits the weights (orthogonalization or a baked bias) rather than using hooks.
        preserve_norms: With weight ablation, rescale each edited write vector to its original length.
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
        "preserve_norms": preserve_norms,
    }
    return _save(run, config)


def edit_from_trial(run: Path, number: int, weights: bool = False, preserve_norms: bool = False) -> dict:
    """Write ``edit.json`` from a search trial's config, optionally applied to the weights."""
    trials = {t["number"]: t for t in load_json(run / "search.json")["trials"]}
    if number not in trials:
        raise UserError(f"no trial {number} in {run / 'search.json'}; trials run from 0 to {max(trials)}")
    return _save(run, {**trials[number]["config"], "weights": weights, "preserve_norms": preserve_norms})
