"""Measure how well an LLM judge agrees with hand labels for a behavior.

Usage:
    python scripts/validate_judge.py behaviors/hedging.yaml data/hedging \
        [--judge MODEL] [--labels judge_labels.jsonl] [-v]

Reads ``judge_sample.jsonl`` (id, question, answer, and a reference answer when the behavior
uses one) and a labels file (id, label) from the
data directory, scores the sample with the judge, and reports agreement overall and by
source. Sources come from ``judge_sample_key.jsonl`` (the first word of each item's
condition, such as ``4B`` or ``dataset``). Writes
``judge_agreement-<judge>-<labels>.json``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vectorlab.behavior import load_behavior, read_jsonl
from vectorlab.cli import configure_logging
from vectorlab.judge import JUDGE_MODEL, agreement, judge_scores
from vectorlab.model import load, slug


def source_bias(items: list[dict], key: dict[int, dict]) -> dict:
    """Mean signed and absolute judge-minus-label difference for each source group."""
    groups = defaultdict(list)
    for item in items:
        groups[key[item["id"]]["condition"].split()[0]].append(item["judge"] - item["label"])
    return {
        group: {"n": len(d), "mean_diff": sum(d) / len(d), "mean_abs_diff": sum(map(abs, d)) / len(d)}
        for group, d in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("behavior", help="behavior YAML spec with a rubric")
    parser.add_argument("data", type=Path, help="directory with judge_sample.jsonl and the labels file")
    parser.add_argument("--judge", default=JUDGE_MODEL)
    parser.add_argument("--labels", default="judge_labels.jsonl", help="labels file name inside the data directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    spec = load_behavior(args.behavior)
    sample = {r["id"]: r for r in read_jsonl(args.data / "judge_sample.jsonl")}
    labels = {r["id"]: r["label"] for r in read_jsonl(args.data / args.labels)}
    key = {r["id"]: r for r in read_jsonl(args.data / "judge_sample_key.jsonl")}
    ids = sorted(labels)
    model, tok, adapter = load(args.judge)
    scores = judge_scores(
        model,
        tok,
        adapter,
        spec["rubric"],
        [sample[i]["question"] for i in ids],
        [sample[i]["answer"] for i in ids],
        [sample[i]["reference"] for i in ids] if "reference" in sample[ids[0]] else None,
    )
    items = [{"id": i, "label": labels[i], "judge": s} for i, s in zip(ids, scores)]
    result = {
        "judge": args.judge,
        "labels": args.labels,
        "agreement": agreement(scores, [labels[i] for i in ids]),
        "by_source": source_bias(items, key),
        "items": items,
    }
    out = args.data / f"judge_agreement-{slug(args.judge)}-{Path(args.labels).stem}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("judge", "labels", "agreement", "by_source")}, indent=1))


if __name__ == "__main__":
    main()
