"""Build a prompt-mode contrast dataset by appending opposite instructions to the same prompts.

Usage:
    python scripts/instruction_pairs.py verbosity data/verbosity/concise.jsonl data/verbosity-instruction

Every distinct prompt in SOURCE appears once in each output file: with one of the recipe's
positive instructions appended, and with the matching negative instruction. Instruction
pairs are assigned round-robin, so no single wording defines the direction. Each record
keeps the plain prompt in ``base``, which the split, the search, and the held-out eval use.
"""

import argparse
from pathlib import Path

from vectorlab.behavior import read_jsonl, write_jsonl

RECIPES = {
    "verbosity": {
        "files": ("concise", "detailed"),
        "pairs": [
            ("Answer in one or two sentences.", "Answer in detail, over several paragraphs."),
            ("Keep it brief.", "Be thorough and cover everything relevant."),
            ("Give a short, direct answer.", "Give a long, comprehensive answer."),
            ("Be concise.", "Be detailed and expansive."),
        ],
    },
}


def instruction_pairs(prompts: list[str], pairs: list[tuple[str, str]]) -> tuple[list[dict], list[dict]]:
    """Positive and negative records for ``prompts``, with instruction pairs assigned round-robin."""
    positive, negative = [], []
    for i, prompt in enumerate(prompts):
        pos, neg = pairs[i % len(pairs)]
        positive.append({"prompt": f"{prompt}\n\n{pos}", "base": prompt, "instruction": pos})
        negative.append({"prompt": f"{prompt}\n\n{neg}", "base": prompt, "instruction": neg})
    return positive, negative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recipe", choices=RECIPES)
    parser.add_argument("source", type=Path, help="JSONL file whose prompts to reuse")
    parser.add_argument("out", type=Path, help="directory for the positive and negative JSONL files")
    args = parser.parse_args()

    recipe = RECIPES[args.recipe]
    prompts = list(dict.fromkeys(r["prompt"] for r in read_jsonl(args.source)))
    positive, negative = instruction_pairs(prompts, recipe["pairs"])
    for name, records in zip(recipe["files"], (positive, negative)):
        write_jsonl(args.out / f"{name}.jsonl", records)
    print(f"{len(prompts)} prompts, {len(recipe['pairs'])} instruction pairs -> {args.out}")


if __name__ == "__main__":
    main()
