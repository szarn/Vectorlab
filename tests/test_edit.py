"""Weight edits remove the direction from every residual write they touch."""

import pytest
import torch

from conftest import EXACT, TINY, tiny_model, unit
from vectorlab.weights import bake_bias, orthogonalize


def residual_writes(adapter, layer, n=32):
    """Each projection's residual write for random inputs, after its post-norm when it has one."""
    writes = []
    for linear, norm in zip(adapter.projections(layer), adapter.post_norms(layer)):
        out = linear(torch.randn(n, linear.in_features))
        writes.append(out if norm is None else norm(out))
    return writes


def relative_component(x: torch.Tensor, r: torch.Tensor) -> float:
    return ((x @ r).abs() / x.norm(dim=-1).clamp_min(1e-12)).max().item()


@pytest.mark.parametrize("kind", TINY)
@torch.no_grad()
def test_orthogonalized_writes_have_no_component_along_r(kind):
    model, adapter = tiny_model(kind)
    r = unit(model.config.hidden_size)
    orthogonalize(model, adapter, r, list(range(len(adapter.layers))))
    assert relative_component(adapter.embed.weight, r) < 1e-5
    for layer in adapter.layers:
        for write in residual_writes(adapter, layer):
            assert relative_component(write, r) < 1e-5


@pytest.mark.parametrize("kind", EXACT)
@torch.no_grad()
def test_partial_strength_scales_the_component(kind):
    model, adapter = tiny_model(kind)
    r = unit(model.config.hidden_size)
    linear = adapter.projections(adapter.layers[1])[1]
    x = torch.randn(16, linear.in_features)
    before = linear(x) @ r
    orthogonalize(model, adapter, r, [1], strength=0.25)
    torch.testing.assert_close(linear(x) @ r, 0.75 * before, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("kind", TINY)
@torch.no_grad()
def test_untie_keeps_the_unembedding(kind):
    model, adapter = tiny_model(kind)
    assert adapter.tied
    original = adapter.unembed.weight.detach().clone()
    orthogonalize(model, adapter, unit(model.config.hidden_size), [0, 1])
    assert adapter.unembed.weight.data_ptr() != adapter.embed.weight.data_ptr()
    assert torch.equal(adapter.unembed.weight, original)
    assert not model.config.tie_word_embeddings


@pytest.mark.parametrize("kind", TINY)
@torch.no_grad()
def test_layers_without_zero_keep_embeddings_tied(kind):
    model, adapter = tiny_model(kind)
    embed = adapter.embed.weight.detach().clone()
    orthogonalize(model, adapter, unit(model.config.hidden_size), [2, 3])
    assert adapter.tied and torch.equal(adapter.embed.weight, embed)


@pytest.mark.parametrize("kind", TINY)
@torch.no_grad()
def test_preserve_norms_keeps_lengths_and_orthogonality(kind):
    model, adapter = tiny_model(kind)
    r = unit(model.config.hidden_size)
    layers = list(range(len(adapter.layers)))
    linears = [linear for i in layers for linear in adapter.projections(adapter.layers[i])]
    before = [linear.weight.norm(dim=0).clone() for linear in linears]
    rows = adapter.embed.weight.norm(dim=-1).clone()
    orthogonalize(model, adapter, r, layers, preserve_norms=True)
    for linear, norms in zip(linears, before):
        torch.testing.assert_close(linear.weight.norm(dim=0), norms, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(adapter.embed.weight.norm(dim=-1), rows, rtol=1e-5, atol=1e-6)
    for layer in adapter.layers:
        for write in residual_writes(adapter, layer):
            assert relative_component(write, r) < 1e-5


@pytest.mark.parametrize("kind", EXACT)
@torch.no_grad()
def test_bake_bias_adds_only_at_chosen_layers(kind):
    model, adapter = tiny_model(kind)
    r = unit(model.config.hidden_size)
    bake_bias(model, adapter, r, [1, 2], 2.5)
    assert model.config.attention_bias
    for i, layer in enumerate(adapter.layers):
        attn = layer.self_attn
        for linear in (attn.q_proj, attn.k_proj, attn.v_proj):
            assert not linear.bias.any()
        expected = 2.5 * r if i in (1, 2) else torch.zeros_like(r)
        torch.testing.assert_close(attn.o_proj.bias, expected)


@torch.no_grad()
def test_post_norm_edit_removes_the_part_the_hook_removes():
    model, adapter = tiny_model("gemma3_text")
    r = unit(model.config.hidden_size)
    layer = adapter.layers[1]
    inputs = [torch.randn(32, linear.in_features) for linear in adapter.projections(layer)]
    hooked = []
    for linear, norm, x in zip(adapter.projections(layer), adapter.post_norms(layer), inputs):
        y = norm(linear(x))
        hooked.append(y - (y @ r).unsqueeze(-1) * r)
    orthogonalize(model, adapter, r, [1])
    for linear, norm, x, expected in zip(adapter.projections(layer), adapter.post_norms(layer), inputs, hooked):
        cosine = torch.nn.functional.cosine_similarity(norm(linear(x)), expected, dim=-1)
        assert cosine.min() > 0.999
