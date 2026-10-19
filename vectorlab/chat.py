"""Chat-templated token sequences and batch padding."""

import torch


def prompt_ids(tok, adapter, prompt: str) -> list[int]:
    """Tokenize ``prompt`` as a single user turn, ending with the assistant generation prompt."""
    messages = [{"role": "user", "content": prompt}]
    encoded = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **adapter.template_kwargs)
    return list(encoded["input_ids"])


def answer_ids(tok, answer: str) -> list[int]:
    """Tokenize an assistant answer without special tokens, for appending after ``prompt_ids``."""
    return tok(answer, add_special_tokens=False)["input_ids"]


def sequence(tok, adapter, record: dict, capture: str) -> tuple[list[int], tuple[int, int]]:
    """Build the token ids for ``record`` and the span of positions to capture.

    In prompt mode the span is the last token of the generation prompt. In response
    mode it covers the answer tokens appended after the generation prompt.
    """
    ids = prompt_ids(tok, adapter, record["prompt"])
    if capture == "prompt":
        return ids, (len(ids) - 1, len(ids))
    answer = answer_ids(tok, record["answer"])
    return ids + answer, (len(ids), len(ids) + len(answer))


def pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad ``seqs`` into an id tensor and a matching attention mask."""
    width = max(map(len, seqs))
    ids = torch.tensor([[pad_id] * (width - len(s)) + s for s in seqs])
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs])
    return ids, mask


def positions(mask: torch.Tensor) -> torch.Tensor:
    """Position ids that start at zero on each sequence's first real token."""
    return (mask.cumsum(-1) - 1).clamp(min=0)
