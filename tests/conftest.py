"""Shared fixtures: tiny random models for exact fp32 checks, and helpers for real checkpoints.

Tests marked ``slow`` load real models from the Hugging Face cache. Deselect them with
``pytest -m "not slow"``.
"""

import pytest
import torch
from torch import nn
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from vectorlab.adapters import adapter_for
from vectorlab.capture import capture
from vectorlab.directions import mean_diff
from vectorlab.model import free, load

SIZES = dict(
    vocab_size=96,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    tie_word_embeddings=True,
    initializer_range=0.2,
)
TINY = {
    "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    "llama": (LlamaConfig, LlamaForCausalLM),
    "gemma3_text": (Gemma3TextConfig, Gemma3ForCausalLM),
}
EXACT = ["qwen3", "llama"]

NEUTRAL = [
    "What is the capital of Australia?",
    "Give me three tips for growing tomatoes.",
    "Explain how a bicycle gear works.",
    "Write a haiku about autumn rain.",
    "What does a database index do?",
    "Suggest a name for a friendly robot.",
    "How many legs does a spider have?",
    "Summarize the plot of Cinderella in one sentence.",
]
TOPIC_A = [
    "How do I bake sourdough bread?",
    "What spices go well with lentils?",
    "How long should I boil an egg?",
    "What is a good vegetarian lasagna recipe?",
    "How do I make fresh pasta?",
    "Why does bread dough need to rest?",
    "How can I keep cookies soft?",
    "What is the difference between baking soda and baking powder?",
]
TOPIC_B = [
    "How does a binary search work?",
    "What is a hash table?",
    "How do I reverse a linked list?",
    "What is the time complexity of quicksort?",
    "Explain recursion with an example.",
    "What is a race condition?",
    "How does garbage collection work?",
    "What is the difference between a stack and a queue?",
]


def tiny_model(kind: str):
    """A small randomly initialized fp32 model of architecture ``kind`` and its adapter.

    Norm weights are randomized too, so Gemma's ``1 + w`` post-norms are not the identity.
    """
    torch.manual_seed(0)
    config_cls, model_cls = TINY[kind]
    model = model_cls(config_cls(**SIZES)).eval()
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            nn.init.normal_(module.weight, mean=0.0 if kind == "gemma3_text" else 1.0, std=0.3)
    return model, adapter_for(model)


def unit(n: int, seed: int = 1) -> torch.Tensor:
    """A random unit vector of length ``n``."""
    r = torch.randn(n, generator=torch.Generator().manual_seed(seed))
    return r / r.norm()


def contrast_direction(tok, adapter, layer: int) -> torch.Tensor:
    """A realistic direction: the mean difference between two topics' prompt activations at ``layer``."""
    acts = [capture(tok, adapter, [{"prompt": p} for p in prompts], "prompt") for prompts in (TOPIC_A, TOPIC_B)]
    return mean_diff(*acts)[0][layer]


@pytest.fixture
def loaded(request):
    """Load ``request.param`` as ``(model_id, dtype)`` and release it after the test."""
    model_id, dtype = request.param
    model, tok, adapter = load(model_id, dtype=dtype)
    yield model, tok, adapter
    del model, tok, adapter
    free()
