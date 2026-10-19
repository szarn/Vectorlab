"""Permanent edits: project a direction out of residual writers, or bake an addition into a bias.

These are the weight-space counterparts of the hooks in ``vectorlab.hooks``. For Qwen3 and
Llama they match the hooks exactly, up to rounding: each write vector is orthogonalized
against ``r``. For Gemma 3 each projection feeds a post-norm with scale ``s = 1 + w``, so the
projection gets a ridge-regularized oblique edit instead (see ``write_pair``). The normed
write is then exactly orthogonal to ``r`` and the removed part points along ``r``, as with
the hook, but the norm's per-token RMS rescaling means the match is close, not exact.
"""

import torch
from torch import nn

EMBED_CHUNK = 16384
POST_NORM_RIDGE = 0.01


def untie(model, adapter) -> None:
    """Give the unembedding its own copy of a tied embedding matrix, and mark the config untied."""
    if not adapter.tied:
        return
    adapter.unembed.weight = nn.Parameter(adapter.embed.weight.detach().clone())
    model.config.tie_word_embeddings = False
    model.config.get_text_config().tie_word_embeddings = False
    adapter.tied = False


def write_pair(r: torch.Tensor, norm: nn.Module | None, norm_offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The vectors ``(u, v)`` of the edit ``o ← o − (u·o) v`` that ablates ``r`` from a projection's write.

    Without a norm, ``u = v = r``: the orthogonal projection. Through a post-norm the write
    is ``s ⊙ o / rms(o)``, whose component along ``r`` is ``(s ⊙ r)·o / rms(o)``, so
    ``u = s ⊙ r``. The removed part lands in the residual stream as ``s ⊙ v``, which should
    point along ``r`` as the hook's does, so ``v ≈ r / s``. A ridge term bounds ``v`` where
    ``s`` is near zero, and ``v`` is scaled so ``u·v = 1``, which zeroes ``u·o`` exactly.
    """
    if norm is None:
        return r, r
    s = norm_offset + norm.weight.float()
    u = s * r
    v = u / (s * s + POST_NORM_RIDGE * (s * s).mean())
    return u, v / (u @ v)


def _project_rows(rows: torch.Tensor, u: torch.Tensor, v: torch.Tensor, strength: float, preserve_norms: bool) -> torch.Tensor:
    edited = rows - strength * (rows @ u).unsqueeze(-1) * v
    if preserve_norms:
        edited *= (rows.norm(dim=-1) / edited.norm(dim=-1).clamp_min(1e-8)).unsqueeze(-1)
    return edited


@torch.no_grad()
def orthogonalize_embedding(embed: nn.Embedding, r: torch.Tensor, strength: float = 1.0, preserve_norms: bool = False) -> None:
    """Remove ``strength`` times the component along ``r`` from every embedding row, in chunks to bound memory."""
    weight = embed.weight
    for i in range(0, weight.shape[0], EMBED_CHUNK):
        rows = weight[i : i + EMBED_CHUNK].float()
        weight[i : i + EMBED_CHUNK] = _project_rows(rows, r, r, strength, preserve_norms).to(weight.dtype)


@torch.no_grad()
def project_linear(linear: nn.Linear, u: torch.Tensor, v: torch.Tensor, strength: float = 1.0, preserve_norms: bool = False) -> None:
    """Edit ``linear`` so each output ``o`` becomes ``o − strength (u·o) v``.

    Each weight column is one output-space vector. With ``preserve_norms`` every column is
    rescaled to its original length, which keeps ``u·o`` at zero for a full ablation.
    """
    columns = linear.weight.float().T
    linear.weight.copy_(_project_rows(columns, u, v, strength, preserve_norms).T.to(linear.weight.dtype))
    if linear.bias is not None:
        bias = linear.bias.float()
        linear.bias.copy_((bias - strength * (bias @ u) * v).to(linear.bias.dtype))


def orthogonalize(model, adapter, r: torch.Tensor, layers: list[int], strength: float = 1.0, preserve_norms: bool = False) -> None:
    """Ablate ``r`` from the residual writes of ``layers`` by editing weights in place.

    The embedding is edited too when layer 0 is included, as with the ablation hook. A tied
    unembedding is untied first, so the readout keeps the original embedding matrix.
    """
    r = r.float().to(adapter.embed.weight.device)
    if 0 in layers:
        untie(model, adapter)
        orthogonalize_embedding(adapter.embed, r, strength, preserve_norms)
    for i in layers:
        layer = adapter.layers[i]
        for linear, norm in zip(adapter.projections(layer), adapter.post_norms(layer)):
            project_linear(linear, *write_pair(r, norm, adapter.norm_offset), strength, preserve_norms)


@torch.no_grad()
def bake_bias(model, adapter, r: torch.Tensor, layers: list[int], alpha: float) -> None:
    """Add ``alpha * r`` to the attention output of ``layers`` through an ``o_proj`` bias.

    Enabling ``attention_bias`` in the config gives every q, k, v and o projection a bias, so
    all are created here as zeros and saved checkpoints reload exactly.
    """
    model.config.get_text_config().attention_bias = True
    for layer in adapter.layers:
        attn = layer.self_attn
        for linear in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            if linear.bias is None:
                linear.bias = nn.Parameter(torch.zeros(linear.out_features, dtype=linear.weight.dtype, device=linear.weight.device))
    for i in layers:
        bias = adapter.layers[i].self_attn.o_proj.bias
        bias.copy_((bias.float() + alpha * r.float().to(bias.device)).to(bias.dtype))


def apply_edit(model, adapter, r: torch.Tensor, config: dict) -> None:
    """Apply an ``edit.json`` config to the weights in place: orthogonalization to ablate, a bias to add."""
    if config["mode"] == "ablate":
        orthogonalize(model, adapter, r, config["layers"], config["strength"], config.get("preserve_norms", False))
    else:
        bake_bias(model, adapter, r, config["layers"], config["strength"])
