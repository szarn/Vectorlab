"""Behavior specs, their datasets, and train/held-out splits."""

import json
import random
from pathlib import Path

import yaml

from vectorlab.errors import UserError

CHOICES = {
    "capture": ("prompt", "response"),
    "intervention": ("ablate", "add"),
    "metric": ("refusal", "verbosity", "judge"),
    "target": ("min", "max"),
}
REQUIRED = ("name", "positive", "negative", *CHOICES)


def load_behavior(path: str | Path) -> dict:
    """Read a behavior YAML spec and validate it, including the datasets it points to."""
    path = Path(path)
    if not path.is_file():
        raise UserError(f"behavior spec not found: {path}")
    spec = yaml.safe_load(path.read_text())
    validate_behavior(spec, path)
    return spec


def validate_behavior(spec: dict, path: Path) -> None:
    """Raise UserError if ``spec`` has missing fields, invalid values, or missing or malformed data."""
    if not isinstance(spec, dict):
        raise UserError(f"{path}: expected a mapping of fields")
    missing = [key for key in REQUIRED if key not in spec]
    if missing:
        raise UserError(f"{path}: missing fields: {', '.join(missing)}")
    for key, allowed in CHOICES.items():
        if spec[key] not in allowed:
            raise UserError(f"{path}: '{key}' must be one of {', '.join(allowed)}; got {spec[key]!r}")
    if spec["metric"] == "judge" and not spec.get("rubric"):
        raise UserError(f"{path}: metric 'judge' needs a 'rubric' field")
    penalty = spec.get("length_penalty", 0)
    if not isinstance(penalty, (int, float)) or penalty < 0:
        raise UserError(f"{path}: 'length_penalty' must be a non-negative number; got {penalty!r}")
    fields = ("prompt",) if spec["capture"] == "prompt" else ("prompt", "answer")
    for key in ("positive", "negative"):
        check_dataset(Path(spec[key]), fields, f"{path}: '{key}'")


def check_dataset(path: Path, fields: tuple[str, ...], context: str) -> None:
    """Raise UserError unless ``path`` is a non-empty JSONL file whose records all have ``fields``."""
    if not path.is_file():
        raise UserError(f"{context} dataset not found: {path}")
    records = read_jsonl(path)
    if not records:
        raise UserError(f"{context} dataset is empty: {path}")
    for line, record in enumerate(records, 1):
        missing = [f for f in fields if not isinstance(record.get(f), str)]
        if missing:
            raise UserError(f"{path}: record {line} is missing string field(s): {', '.join(missing)}")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read one JSON object per non-blank line."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write one JSON object per line, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def base_prompt(record: dict) -> str:
    """The plain prompt a record belongs to: its ``base`` field if it has one, else its prompt.

    Instruction-pair datasets append opposite instructions to the same prompt and keep the
    plain prompt in ``base``. Splits group by it, and search and eval generate from it.
    """
    return record.get("base", record["prompt"])


def held_out_prompts(records: list[dict], seed: int, fraction: float = 0.2) -> set[str]:
    """Pick a seeded fraction of the distinct base prompts in ``records`` to hold out."""
    prompts = sorted({base_prompt(r) for r in records})
    random.Random(seed).shuffle(prompts)
    return set(prompts[: int(len(prompts) * fraction)])


def splits(spec: dict, seed: int) -> dict[str, list[dict]]:
    """Split both datasets into train and held-out sets by base prompt.

    Splitting by base prompt keeps response-mode pairs that share a prompt, and instruction
    pairs that share a base prompt, on the same side.

    Returns:
        A dict with keys ``train_pos``, ``train_neg``, ``test_pos`` and ``test_neg``.
    """
    pos = read_jsonl(spec["positive"])
    neg = read_jsonl(spec["negative"])
    test = held_out_prompts(pos + neg, seed)
    return {
        "train_pos": [r for r in pos if base_prompt(r) not in test],
        "train_neg": [r for r in neg if base_prompt(r) not in test],
        "test_pos": [r for r in pos if base_prompt(r) in test],
        "test_neg": [r for r in neg if base_prompt(r) in test],
    }
