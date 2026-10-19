"""Behavior directions and the statistics the report needs."""

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vectorlab.capture import content_span, token_states

MASSIVE_RATIO = 100.0


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


def massive_dims(mean: torch.Tensor, ratio: float = MASSIVE_RATIO) -> torch.Tensor:
    """Mask of massive-activation dimensions: mean magnitude above ``ratio`` times the median dimension's."""
    magnitude = mean.abs()
    return magnitude > ratio * magnitude.median()


def edit_direction(tensors: dict[str, torch.Tensor], layer: int, mode: str) -> torch.Tensor:
    """The unit direction an edit of ``mode`` uses at ``layer``.

    Ablation drops massive-activation dimensions from the mean difference. Those few
    dimensions carry huge, near-constant activations at the first token and at delimiters,
    which the model relies on as attention sinks. Even a small weight on them makes
    ``x·r`` enormous there, so ablating every write along ``r`` wrecks the model (Gemma 3)
    instead of removing the behavior. Addition is unaffected and uses the mean difference.
    """
    r = tensors["directions"][layer]
    if mode != "ablate":
        return r
    r = r.masked_fill(massive_dims((tensors["mean_pos"][layer] + tensors["mean_neg"][layer]) / 2), 0.0)
    return r / r.norm().clamp_min(1e-8)


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


def lexical_separation(embeddings: dict[str, torch.Tensor]) -> dict:
    """Held-out Cohen's d and probe accuracy of the mean-token-embedding baseline.

    Args:
        embeddings: ``train_pos``, ``train_neg``, ``test_pos`` and ``test_neg`` tensors of shape ``[1, n, hidden]``.
    """
    direction, _ = mean_diff(embeddings["train_pos"], embeddings["train_neg"])
    projections = {name: project(e, direction) for name, e in embeddings.items()}
    return {
        "cohens_d": cohens_d(projections["test_pos"], projections["test_neg"]).item(),
        "probe_acc": probe_accuracy(embeddings)[0],
    }


def midpoints(projections: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-layer midpoint between the train class means along each layer's direction."""
    return (projections["train_pos"].mean(0) + projections["train_neg"].mean(0)) / 2


def token_projections(tok, adapter, examples: list[tuple[dict, str]], mode: str, directions: torch.Tensor, centers: torch.Tensor) -> list[dict]:
    """Each token's projection onto every layer's direction, relative to that layer's class midpoint.

    Positive values lean toward the positive class. Returns one dict per example with
    ``label``, ``tokens``, ``special`` flags, the ``content`` span, and ``scores`` as
    ``[layers][tokens]``.
    """
    special = set(tok.all_special_ids)
    out = []
    for record, label in examples:
        ids, span, states = token_states(tok, adapter, record, mode)
        scores = torch.einsum("ltd,ld->lt", states, directions) - centers.unsqueeze(-1)
        out.append(
            {
                "label": label,
                "tokens": [tok.decode([i]) for i in ids],
                "special": [i in special for i in ids],
                "content": list(content_span(mode, ids, span)),
                "scores": scores.round(decimals=3).tolist(),
            }
        )
    return out
