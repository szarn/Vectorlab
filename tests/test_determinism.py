"""Extracting the same behavior twice with the same seed gives identical directions."""

import json

import pytest
import yaml

from vectorlab.extract import extract
from vectorlab.store import load_json, load_tensors

COOKING = ["sourdough", "risotto", "pancakes", "curry", "dumplings", "ratatouille", "paella", "ramen", "falafel", "lasagna",
           "gazpacho", "biryani", "pierogi", "tacos", "gnocchi", "goulash", "shakshuka", "pho", "tagine", "crepes"]
CODING = ["binary search", "hash tables", "linked lists", "quicksort", "recursion", "race conditions", "garbage collection",
          "stacks", "queues", "heaps", "tries", "graphs", "closures", "generators", "decorators", "mutexes", "sockets",
          "regular expressions", "unit tests", "memoization"]


def write_behavior(root):
    for name, items in (("cooking", COOKING), ("coding", CODING)):
        lines = [json.dumps({"prompt": f"Tell me about {item}."}) for item in items]
        (root / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    spec = {
        "name": "topic",
        "capture": "prompt",
        "positive": "cooking.jsonl",
        "negative": "coding.jsonl",
        "intervention": "ablate",
        "metric": "refusal",
        "target": "min",
    }
    (root / "topic.yaml").write_text(yaml.safe_dump(spec))


@pytest.mark.slow
def test_extract_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_behavior(tmp_path)
    runs = []
    for _ in range(2):
        out, _ = extract("topic.yaml", "Qwen/Qwen3-0.6B", seed=0)
        runs.append((load_tensors(out / "directions.safetensors"), load_json(out / "separation.json"), load_json(out / "run.json")))
    (first, sep_first, run_first), (second, sep_second, run_second) = runs
    assert run_first["test"] == run_second["test"]
    for name in first:
        assert first[name].equal(second[name]), name
    assert sep_first == sep_second
