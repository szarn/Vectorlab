"""Batched text generation (greedy by default)."""

import logging

import torch
from transformers import GenerationConfig

from vectorlab.chat import pad, prompt_ids

log = logging.getLogger(__name__)


def _config(model, tok, max_new_tokens, temperature):
    sampling = temperature is not None
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=sampling,
        temperature=temperature if sampling else 1.0,
        top_p=0.95 if sampling else 1.0,
        top_k=50,
        pad_token_id=tok.pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
    )


@torch.no_grad()
def generate(
    model,
    tok,
    adapter,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int = 16,
    temperature: float | None = None,
) -> list[str]:
    """Answer each prompt (as a single user turn) and return the decoded answers.

    Decoding is greedy unless ``temperature`` is given, in which case it samples with
    nucleus ``top_p=0.95``. Seed torch beforehand for reproducible samples.
    """
    log.info("generating %d answers (up to %d tokens)", len(prompts), max_new_tokens)
    config = _config(model, tok, max_new_tokens, temperature)
    texts = []
    for i in range(0, len(prompts), batch_size):
        ids, mask = pad([prompt_ids(tok, adapter, p) for p in prompts[i : i + batch_size]], tok.pad_token_id)
        out = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), generation_config=config)
        texts += tok.batch_decode(out[:, ids.shape[1] :], skip_special_tokens=True)
    return [t.strip() for t in texts]
