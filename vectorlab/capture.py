"""Residual-stream capture with forward pre-hooks on every decoder layer."""

import logging

import torch

from vectorlab.chat import pad, positions, sequence

log = logging.getLogger(__name__)


def _hidden(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def _span_weights(items, width):
    weights = torch.zeros(len(items), width)
    for row, (seq, (start, end)) in enumerate(items):
        offset = width - len(seq)
        weights[row, offset + start : offset + end] = 1 / (end - start)
    return weights


@torch.no_grad()
def _capture_batch(tok, adapter, items):
    device = adapter.embed.weight.device
    ids, mask = pad([seq for seq, _ in items], tok.pad_token_id)
    weights = _span_weights(items, ids.shape[1]).to(device)
    acts = []

    def record(module, args, kwargs):
        acts.append(torch.einsum("bt,btd->bd", weights, _hidden(args, kwargs).float()).cpu())

    handles = [layer.register_forward_pre_hook(record, with_kwargs=True) for layer in adapter.layers]
    try:
        adapter.base(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            position_ids=positions(mask).to(device),
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(acts)


def capture(tok, adapter, records: list[dict], mode: str, batch_size: int = 16) -> torch.Tensor:
    """Capture the residual stream entering each layer, averaged over each record's capture span.

    Args:
        records: Dicts with ``prompt`` (and ``answer`` in response mode).
        mode: ``"prompt"`` for the last generation-prompt token, ``"response"`` for the answer mean.

    Returns:
        A float32 CPU tensor of shape ``[layers, len(records), hidden]``.
    """
    log.info("capturing %d records (%s mode)", len(records), mode)
    items = [sequence(tok, adapter, r, mode) for r in records]
    batches = [_capture_batch(tok, adapter, items[i : i + batch_size]) for i in range(0, len(items), batch_size)]
    return torch.cat(batches, 1)
