"""Model loading and device memory management."""

import gc
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vectorlab.adapters import adapter_for


def default_device() -> str:
    """The device models load on: ``VECTORLAB_DEVICE`` if set, else CUDA, then Apple MPS, then CPU."""
    if os.environ.get("VECTORLAB_DEVICE"):
        return os.environ["VECTORLAB_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = default_device()

log = logging.getLogger(__name__)


def load(model_id: str, dtype: torch.dtype = torch.bfloat16, device: str = DEVICE):
    """Load a causal LM in eval mode, plus its left-padding tokenizer and adapter.

    Returns:
        A ``(model, tokenizer, adapter)`` tuple.
    """
    log.info("loading %s as %s on %s", model_id, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    return model, tok, adapter_for(model)


def free() -> None:
    """Release cached accelerator memory once the caller has dropped its model references."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def slug(model_id: str) -> str:
    """Return a filesystem-safe name for a Hugging Face model id."""
    return model_id.replace("/", "--")
