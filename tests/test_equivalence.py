"""Hook interventions and their weight-space counterparts give the same model.

Tiny fp32 models check the math exactly. Real checkpoints check it at scale: Qwen3 and
Llama in fp32 must match closely. Gemma 3 is approximate by construction, because its
post-norms rescale the edited write, so it runs in bf16 with a looser bound. For scale,
bf16 rounding alone puts Qwen3-0.6B hooks and weights about 0.002 apart.
"""

import pytest
import torch

from conftest import EXACT, NEUTRAL, contrast_direction, tiny_model, unit
from vectorlab.hooks import intervene
from vectorlab.metrics import kl, next_logprobs
from vectorlab.weights import bake_bias, orthogonalize

REAL = {
    "Qwen/Qwen3-0.6B": (torch.float32, 1e-4),
    "unsloth/Llama-3.2-1B-Instruct": (torch.float32, 1e-4),
    "google/gemma-3-4b-it": (torch.bfloat16, 0.01),
}


def tiny_logits(model, seed=0):
    ids = torch.randint(0, model.config.vocab_size, (3, 12), generator=torch.Generator().manual_seed(seed))
    return model(input_ids=ids).logits


@pytest.mark.parametrize("kind", EXACT)
@pytest.mark.parametrize("layers", [[0, 1, 2, 3], [1, 2]])
@pytest.mark.parametrize("strength", [1.0, 0.6])
@torch.no_grad()
def test_tiny_ablation_hook_matches_orthogonalized_weights(kind, layers, strength):
    model, adapter = tiny_model(kind)
    r = unit(model.config.hidden_size)
    with intervene(adapter, r, "ablate", layers, strength):
        hooked = tiny_logits(model)
    orthogonalize(model, adapter, r, layers, strength)
    torch.testing.assert_close(tiny_logits(model), hooked, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", EXACT)
@torch.no_grad()
def test_tiny_addition_hook_matches_baked_bias(kind):
    model, adapter = tiny_model(kind)
    r = unit(model.config.hidden_size)
    with intervene(adapter, r, "add", [1, 2], 3.0):
        hooked = tiny_logits(model)
    bake_bias(model, adapter, r, [1, 2], 3.0)
    torch.testing.assert_close(tiny_logits(model), hooked, atol=1e-4, rtol=1e-4)


def ablation_gap(model, tok, adapter) -> float:
    """KL from hook ablation to weight orthogonalization, over every layer, on neutral prompts."""
    r = contrast_direction(tok, adapter, len(adapter.layers) // 2)
    layers = list(range(len(adapter.layers)))
    with intervene(adapter, r, "ablate", layers, 1.0):
        hooked = next_logprobs(model, tok, adapter, NEUTRAL)
    plain = next_logprobs(model, tok, adapter, NEUTRAL)
    orthogonalize(model, adapter, r, layers)
    edited = next_logprobs(model, tok, adapter, NEUTRAL)
    return kl(hooked, edited), kl(plain, hooked)


@pytest.mark.slow
@pytest.mark.parametrize("loaded", [(m, d) for m, (d, _) in REAL.items()], indirect=True, ids=list(REAL))
def test_real_ablation_hook_matches_orthogonalized_weights(loaded, record_property):
    model, tok, adapter = loaded
    gap, effect = ablation_gap(model, tok, adapter)
    record_property("kl_hook_to_weights", gap)
    record_property("kl_original_to_hook", effect)
    tolerance = REAL[model.name_or_path][1]
    assert gap < tolerance, f"KL(hook || weights) = {gap:.2e} exceeds {tolerance:.0e}; the ablation itself moves KL by {effect:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("loaded", [("Qwen/Qwen3-0.6B", torch.float32)], indirect=True, ids=["Qwen/Qwen3-0.6B"])
def test_real_addition_hook_matches_baked_bias(loaded):
    model, tok, adapter = loaded
    layer = len(adapter.layers) // 2
    r = contrast_direction(tok, adapter, layer)
    with intervene(adapter, r, "add", [layer], 8.0):
        hooked = next_logprobs(model, tok, adapter, NEUTRAL)
    bake_bias(model, adapter, r, [layer], 8.0)
    assert kl(hooked, next_logprobs(model, tok, adapter, NEUTRAL)) < 1e-5
