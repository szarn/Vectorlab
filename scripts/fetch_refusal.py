"""Download the refusal datasets and write them as JSONL.

Harmful prompts: AdvBench (MIT, from the original llm-attacks repository) plus the
JailbreakBench harmful behaviors (MIT), deduplicated. Harmless and neutral prompts:
instructions without inputs from yahma/alpaca-cleaned (CC BY 4.0), shuffled with a fixed
seed. The harmless set matches the harmful count, and the neutral KL set is disjoint from it.
"""

import csv
import io
import json
import random
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import hf_hub_download

from vectorlab.behavior import write_jsonl

SEED = 0
NEUTRAL_COUNT = 100
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def fetch(repo: str, filename: str) -> Path:
    """Download one file from a Hugging Face dataset repo, returning its cached path."""
    return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def advbench() -> list[dict]:
    """AdvBench harmful behaviors."""
    text = urlopen(ADVBENCH_URL).read().decode()
    return [{"prompt": row["goal"], "source": "advbench"} for row in csv.DictReader(io.StringIO(text))]


def jailbreakbench() -> list[dict]:
    """JailbreakBench harmful behaviors."""
    with fetch("JailbreakBench/JBB-Behaviors", "data/harmful-behaviors.csv").open() as f:
        return [{"prompt": row["Goal"], "source": "jbb"} for row in csv.DictReader(f)]


def alpaca() -> list[dict]:
    """Alpaca-cleaned instructions that need no extra input."""
    rows = json.loads(fetch("yahma/alpaca-cleaned", "alpaca_data_cleaned.json").read_text())
    return [
        {"prompt": row["instruction"], "source": "alpaca-cleaned"}
        for row in rows
        if row["instruction"].strip() and not row["input"].strip()
    ]


def dedupe(records: list[dict]) -> list[dict]:
    """Drop records whose prompt repeats an earlier one, ignoring case and whitespace."""
    seen = set()
    unique = []
    for record in records:
        key = " ".join(record["prompt"].lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def main() -> None:
    harmful = dedupe(advbench() + jailbreakbench())
    harmless = dedupe(alpaca())
    random.Random(SEED).shuffle(harmless)
    write_jsonl("data/refusal/harmful.jsonl", harmful)
    write_jsonl("data/refusal/harmless.jsonl", harmless[: len(harmful)])
    write_jsonl("data/neutral.jsonl", harmless[len(harmful) : len(harmful) + NEUTRAL_COUNT])
    print(f"harmful {len(harmful)}  harmless {len(harmful)}  neutral {NEUTRAL_COUNT}")


if __name__ == "__main__":
    main()
