"""Each adapter finds the residual-stream modules of its architecture."""

import pytest
import torch

from conftest import TINY, tiny_model
from vectorlab.adapters import ADAPTERS, Gemma3, Llama, Qwen3
from vectorlab.chat import prompt_ids

REAL = {
    "Qwen/Qwen3-0.6B": (Qwen3, "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    "unsloth/Llama-3.2-1B-Instruct": (Llama, "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "google/gemma-3-4b-it": (Gemma3, "<start_of_turn>model\n"),
}


def check_modules(model, adapter):
    config = model.config.get_text_config()
    assert len(adapter.layers) == config.num_hidden_layers
    assert adapter.embed.weight.shape == (config.vocab_size, config.hidden_size)
    assert adapter.unembed.weight.shape == (config.vocab_size, config.hidden_size)
    assert adapter.tied == config.tie_word_embeddings
    assert adapter.final_norm in list(adapter.base.modules())
    for layer in adapter.layers:
        owned = list(layer.modules())
        assert len(adapter.writers(layer)) == len(adapter.projections(layer)) == len(adapter.post_norms(layer)) == 2
        for module in adapter.writers(layer) + adapter.projections(layer):
            assert any(module is m for m in owned)
        for linear in adapter.projections(layer):
            assert linear.out_features == config.hidden_size


@pytest.mark.parametrize("kind", TINY)
def test_tiny_adapters_find_modules(kind):
    model, adapter = tiny_model(kind)
    assert type(adapter) is ADAPTERS[kind]
    check_modules(model, adapter)


@pytest.mark.slow
@pytest.mark.parametrize("loaded", [(m, torch.bfloat16) for m in REAL], indirect=True, ids=list(REAL))
def test_real_adapters_find_modules_and_template(loaded):
    model, tok, adapter = loaded
    expected, generation_prompt = REAL[model.name_or_path]
    assert type(adapter) is expected
    check_modules(model, adapter)
    text = tok.decode(prompt_ids(tok, adapter, "Name a color."))
    assert text.endswith(generation_prompt)
    if tok.bos_token:
        assert text.count(tok.bos_token) <= 1
    if expected is Llama:
        assert "Today Date: 26 Jul 2024" in text
