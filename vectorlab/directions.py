"""Behavior directions and the statistics the report needs."""

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mean_diff(pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer unit direction from the negative class mean to the positive class mean.

    Args:
        pos, neg: Activations of shape ``[layers, n, hidden]``.

    Returns:
        ``(directions, norms)``: unit directions ``[layers, hidden]`` and the unnormalized
        difference norms ``[layers]``. Layers with no difference get a zero direction.
    """
    diff = pos.mean(1) - neg.mean(1)
    norms = diff.norm(dim=-1)
    return diff / norms.clamp_min(1e-8).unsqueeze(-1), norms


def project(acts: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Project ``[layers, n, hidden]`` activations onto each layer's direction, giving ``[n, layers]``."""
    return torch.einsum("lnd,ld->nl", acts, directions)


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-column Cohen's d between samples ``a`` and ``b`` (pooled SD); 0 where both are constant."""
    na, nb = len(a), len(b)
    pooled = (((na - 1) * a.var(0) + (nb - 1) * b.var(0)) / (na + nb - 2)).sqrt()
    return ((a.mean(0) - b.mean(0)) / pooled).nan_to_num(0.0)


def _labeled(pos, neg):
    return torch.cat([pos, neg]).numpy(), [1] * len(pos) + [0] * len(neg)


def probe_accuracy(acts: dict[str, torch.Tensor]) -> list[float]:
    """Held-out accuracy of a standardized logistic-regression probe fit separately at each layer.

    Args:
        acts: ``train_pos``, ``train_neg``, ``test_pos`` and ``test_neg`` activations, each ``[layers, n, hidden]``.
    """
    scores = []
    for layer in range(acts["train_pos"].shape[0]):
        x, y = _labeled(acts["train_pos"][layer], acts["train_neg"][layer])
        x_test, y_test = _labeled(acts["test_pos"][layer], acts["test_neg"][layer])
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x, y)
        scores.append(probe.score(x_test, y_test))
    return scores


def _ranked(tok, values, ids):
    return [[tok.decode([i]), v] for v, i in zip(values.tolist(), ids.tolist())]


@torch.no_grad()
def logit_lens(tok, adapter, directions: torch.Tensor, k: int = 12) -> list[dict]:
    """Pass each layer's direction through the final norm and unembedding.

    Returns:
        One dict per layer, with ``promoted`` and ``suppressed`` lists of ``[token, logit]``.
    """
    weight = adapter.embed.weight
    normed = adapter.final_norm(directions.to(weight.device, weight.dtype))
    logits = adapter.unembed(normed).float().cpu()
    top = logits.topk(k)
    bottom = logits.topk(k, largest=False)
    return [
        {
            "promoted": _ranked(tok, top.values[i], top.indices[i]),
            "suppressed": _ranked(tok, bottom.values[i], bottom.indices[i]),
        }
        for i in range(len(logits))
    ]
