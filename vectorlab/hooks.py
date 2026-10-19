"""Reversible interventions: forward hooks that ablate or add a direction."""

from contextlib import contextmanager

import torch


def _ablation(r, strength):
    def hook(module, args, output):
        x = output.float()
        return (x - strength * (x @ r).unsqueeze(-1) * r).to(output.dtype)

    return hook


def _addition(r, alpha):
    def hook(module, args, output):
        return (output.float() + alpha * r).to(output.dtype)

    return hook


def targets(adapter, mode: str, layers: list[int]) -> list[torch.nn.Module]:
    """Modules whose outputs an intervention rewrites.

    Ablation covers every residual writer in ``layers``, plus the embedding when layer 0 is
    included. Addition hooks only the attention writer of each layer, the same place a baked
    output-projection bias acts.
    """
    if mode == "add":
        return [adapter.writers(adapter.layers[i])[0] for i in layers]
    writers = [w for i in layers for w in adapter.writers(adapter.layers[i])]
    return ([adapter.embed] if 0 in layers else []) + writers


@contextmanager
def intervene(adapter, r: torch.Tensor, mode: str, layers: list[int], strength: float):
    """Temporarily ablate (``x - strength (x·r) r``) or add (``x + strength r``) a direction.

    Args:
        r: Unit direction of shape ``[hidden]``.
        mode: ``"ablate"`` or ``"add"``.
        layers: Decoder layer indices to intervene on.
        strength: Ablation scale (1.0 removes the component fully) or addition coefficient.
    """
    r = r.float().to(adapter.embed.weight.device)
    hook = _ablation(r, strength) if mode == "ablate" else _addition(r, strength)
    handles = [m.register_forward_hook(hook) for m in targets(adapter, mode, layers)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
