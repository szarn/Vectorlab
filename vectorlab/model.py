"""Model loading and device memory management."""

import gc
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vectorlab.adapters import adapter_for

DEVICE = "mps"

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
    torch.mps.empty_cache()


def slug(model_id: str) -> str:
    """Return a filesystem-safe name for a Hugging Face model id."""
    return model_id.replace("/", "--")
