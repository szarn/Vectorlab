"""Score a run's before and after answers with several judges, for a like-for-like comparison.

Usage:
    python scripts/rescore.py runs/<run> --judges MODEL [MODEL ...] [-v]

Reads ``generations.jsonl`` and the edit from ``eval.json``, then scores both answer sets with
each judge, loading one judge at a time. Results go to ``judges.json``, replacing earlier
entries for the same edit and judge. Each entry includes the judge's agreement (binary kappa)
with the behavior's hand labels when a validation file exists.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from vectorlab.behavior import read_jsonl
from vectorlab.cli import configure_logging
from vectorlab.judge import judge_scores
from vectorlab.model import free, load, slug
from vectorlab.store import load_json, save_json


def describe(edit: dict) -> str:
    """A short human-readable label for an edit config."""
    return f"{edit['mode']} from layer {edit['layer']} at layers {edit['layers'][0]}–{edit['layers'][-1]}, strength {edit['strength']:.3g}"


def validated_kappa(spec: dict, judge: str) -> float | None:
    """The judge's binary kappa against the behavior's primary hand labels, if it has been validated."""
    path = Path(spec["positive"]).parent / f"judge_agreement-{slug(judge)}-judge_labels.json"
    return load_json(path)["agreement"]["binary_kappa"] if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--judges", nargs="+", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    spec = load_json(args.run / "run.json")["behavior"]
    edit = describe(load_json(args.run / "eval.json")["edit"])
    rows = read_jsonl(args.run / "generations.jsonl")
    prompts = [r["prompt"] for r in rows]
    path = args.run / "judges.json"
    entries = load_json(path) if path.exists() else []
    for judge in args.judges:
        model, tok, adapter = load(judge)
        before = judge_scores(model, tok, adapter, spec["rubric"], prompts, [r["before"] for r in rows])
        after = judge_scores(model, tok, adapter, spec["rubric"], prompts, [r["after"] for r in rows])
        del model, adapter
        free()
        entries = [e for e in entries if not (e["edit"] == edit and e["judge"] == judge)]
        entries.append(
            {
                "edit": edit,
                "judge": judge,
                "kappa": validated_kappa(spec, judge),
                "before": mean(before),
                "after": mean(after),
                "n": len(rows),
            }
        )
        print(f"{judge:32s} {mean(before):.2f} -> {mean(after):.2f}")
    save_json(path, entries)


if __name__ == "__main__":
    main()
