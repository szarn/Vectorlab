"""Measure how well the LLM judge agrees with hand labels for a behavior.

Usage:
    python scripts/validate_judge.py behaviors/hedging.yaml data/hedging [--judge MODEL] [-v]

Reads ``judge_sample.jsonl`` (id, question, answer) and ``judge_labels.jsonl`` (id, label)
from the data directory, scores the sample with the judge, and writes the per-item scores
and the agreement statistics to ``judge_agreement.json``.
"""

import argparse
import json
from pathlib import Path

from vectorlab.behavior import load_behavior, read_jsonl
from vectorlab.cli import configure_logging
from vectorlab.judge import JUDGE_MODEL, agreement, judge_scores
from vectorlab.model import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("behavior", help="behavior YAML spec with a rubric")
    parser.add_argument("data", type=Path, help="directory with judge_sample.jsonl and judge_labels.jsonl")
    parser.add_argument("--judge", default=JUDGE_MODEL)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    spec = load_behavior(args.behavior)
    sample = {r["id"]: r for r in read_jsonl(args.data / "judge_sample.jsonl")}
    labels = {r["id"]: r["label"] for r in read_jsonl(args.data / "judge_labels.jsonl")}
    ids = sorted(labels)
    model, tok, adapter = load(args.judge)
    scores = judge_scores(
        model,
        tok,
        adapter,
        spec["rubric"],
        [sample[i]["question"] for i in ids],
        [sample[i]["answer"] for i in ids],
    )
    stats = agreement(scores, [labels[i] for i in ids])
    result = {
        "judge": args.judge,
        "agreement": stats,
        "items": [{"id": i, "label": labels[i], "judge": s} for i, s in zip(ids, scores)],
    }
    (args.data / "judge_agreement.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
