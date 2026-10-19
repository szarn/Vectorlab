"""Expand hand-written seed pairs into a larger behavior dataset with a local instruct model.

Usage:
    python scripts/expand.py hedging --n 300 [--model Qwen/Qwen3-4B-Instruct-2507] [--seed 0] [-v]

For hedging, each seed has an opinion question, two committed answers taking opposite
positions, and one hedged answer. The script samples new questions in the same style, asks
for two opposing positions, and writes one committed answer per position plus a hedged
answer, using seed answers as style examples. Generated answers that leak hedges or commit
to a side, or whose lengths diverge, are dropped together with their question. The outputs are
the behavior's positive and negative JSONL files (seeds included) and a sample for
spot-checking.
"""

import argparse
import logging
import random
import re
from pathlib import Path

import torch

from vectorlab.behavior import read_jsonl, write_jsonl
from vectorlab.cli import configure_logging
from vectorlab.generate import generate
from vectorlab.model import load

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
HEDGE_MARKERS = (
    "depend",
    "both",
    "trade-off",
    "tradeoff",
    "on the other hand",
    "varies",
    "vary",
    "mixed",
    "either",
    "disagree",
    "no single",
    "no right answer",
    "personal",
)
COMMIT_MARKERS = ("i recommend", "without question", "the clear choice", "definitely", "clearly better", "you should")
WORDS = (25, 70)

log = logging.getLogger("expand")

QUESTION_PROMPT = """Here are examples of everyday opinion questions where reasonable people disagree:

{examples}

Write {count} new questions of the same kind. Cover varied topics such as technology, food, work, travel, education, sports, home life, hobbies, money habits, and design. Avoid politics, religion, and medical or legal advice. Each question must be answerable by taking a clear side. Write one question per line, with no numbering or extra text."""

POSITIONS_PROMPT = """Question: {question}

Name the two opposing answers a person could firmly commit to. If this is a yes-or-no question, the labels are exactly Yes and No. Otherwise, name the two competing options in one to four words each. Reply with exactly two lines:
A: <label>
B: <label>"""

COMMITTED_PROMPT = """Answer the question by arguing firmly for this position: {claim}.
Open with that answer in the first sentence, give two or three concrete reasons, and end with a clear recommendation. Do not mention the other side, do not hedge, and do not say it depends. Write 35 to 50 words of plain prose with no lists or headings. Match the style of these examples.

{examples}

Question: {question}
Answer:"""

HEDGED_PROMPT = """Answer the question without taking a side. Present the main considerations for each option evenly, say that the right choice depends on circumstances, and do not recommend either option. Write 35 to 50 words of plain prose with no lists or headings. Match the style of these examples.

{examples}

Question: {question}
Answer:"""


VERIFY_PROMPT = """Question: {question}
Answer: {answer}

Does this answer argue for the position "{claim}" rather than against it? Reply with only Yes or No."""


def claim(position: str) -> str:
    """Spell out a Yes/No label as an explicit claim, so the model argues the right side."""
    return {"yes": "the answer is Yes", "no": "the answer is No"}.get(normalize(position), position)


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace and punctuation, for deduplication."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def word_count(text: str) -> int:
    """Number of whitespace-separated words."""
    return len(text.split())


def parse_questions(text: str) -> list[str]:
    """Question lines from a generated list, with any numbering or bullets removed."""
    lines = (re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip() for line in text.splitlines())
    return [line for line in lines if line.endswith("?") and 4 <= word_count(line) <= 25]


def parse_positions(text: str) -> list[str] | None:
    """The two labels from an ``A: ...`` / ``B: ...`` reply, or None if the reply is malformed."""
    labels = dict(re.findall(r"^\s*([AB])\s*[:.)]\s*(.+?)\s*$", text, flags=re.MULTILINE))
    if set(labels) != {"A", "B"} or normalize(labels["A"]) == normalize(labels["B"]):
        return None
    return [labels["A"].strip("*\"' "), labels["B"].strip("*\"' ")]


def is_committed(text: str) -> bool:
    """A committed answer has no hedge markers and a length in range."""
    lower = text.lower()
    return WORDS[0] <= word_count(text) <= WORDS[1] and not any(m in lower for m in HEDGE_MARKERS)


def consistent(position: str, answer: str) -> bool:
    """False when a Yes/No position's answer opens with the opposite word."""
    opposite = {"yes": "no", "no": "yes"}.get(normalize(position))
    words = normalize(answer).split()
    return not (opposite and words and words[0] == opposite)


def is_hedged(text: str) -> bool:
    """A hedged answer has a hedge marker, no commitment markers, and a length in range."""
    lower = text.lower()
    return (
        WORDS[0] <= word_count(text) <= WORDS[1]
        and any(m in lower for m in HEDGE_MARKERS)
        and not any(m in lower for m in COMMIT_MARKERS)
    )


def lengths_match(answers: list[str], tolerance: float = 0.35) -> bool:
    """Whether every answer is within ``tolerance`` of the mean answer length."""
    counts = [word_count(a) for a in answers]
    mean = sum(counts) / len(counts)
    return all(abs(c - mean) <= tolerance * mean for c in counts)


def example_block(seeds: list[dict], rng: random.Random, kind: str, k: int = 2) -> str:
    """Few-shot ``Question:``/``Answer:`` pairs of the given kind (``committed`` or ``hedged``) from random seeds."""
    blocks = []
    for seed in rng.sample(seeds, k):
        answer = rng.choice(seed["committed"])["answer"] if kind == "committed" else seed["hedged"]
        blocks.append(f"Question: {seed['prompt']}\nAnswer: {answer}")
    return "\n\n".join(blocks)


def new_questions(model, tok, adapter, seeds: list[dict], n: int, rng: random.Random) -> list[str]:
    """Sample at least ``n`` new questions that do not repeat each other or the seeds."""
    seen = {normalize(s["prompt"]) for s in seeds}
    questions = []
    while len(questions) < n:
        prompts = [
            QUESTION_PROMPT.format(examples="\n".join(s["prompt"] for s in rng.sample(seeds, 10)), count=25)
            for _ in range(8)
        ]
        for text in generate(model, tok, adapter, prompts, 700, batch_size=8, temperature=1.0):
            for question in parse_questions(text):
                if normalize(question) not in seen:
                    seen.add(normalize(question))
                    questions.append(question)
        log.info("%d new questions so far", len(questions))
    return questions


def argues_for(model, tok, adapter, cases: list[tuple[str, str]], answers: list[str]) -> list[bool]:
    """Ask the model whether each answer argues for its assigned position (greedy Yes/No check)."""
    prompts = [VERIFY_PROMPT.format(question=q, answer=a, claim=claim(p)) for (q, p), a in zip(cases, answers)]
    replies = generate(model, tok, adapter, prompts, 3, batch_size=32)
    return [normalize(r).startswith("yes") for r in replies]


def expand_hedging(model, tok, adapter, seeds: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Generate hedging records shaped like the seeds, keeping only questions whose answers all pass the filters."""
    questions = new_questions(model, tok, adapter, seeds, int(n * 1.6), rng)
    replies = generate(model, tok, adapter, [POSITIONS_PROMPT.format(question=q) for q in questions], 40, batch_size=32)
    pairs = [(q, p) for q, p in zip(questions, map(parse_positions, replies)) if p]
    log.info("%d questions with two positions", len(pairs))

    committed_prompts = [
        COMMITTED_PROMPT.format(claim=claim(position), question=q, examples=example_block(seeds, rng, "committed"))
        for q, positions in pairs
        for position in positions
    ]
    hedged_prompts = [HEDGED_PROMPT.format(question=q, examples=example_block(seeds, rng, "hedged")) for q, _ in pairs]
    committed = generate(model, tok, adapter, committed_prompts, 120, batch_size=32, temperature=0.7)
    hedged = generate(model, tok, adapter, hedged_prompts, 120, batch_size=32, temperature=0.7)
    on_side = argues_for(model, tok, adapter, [(q, p) for q, positions in pairs for p in positions], committed)

    records = []
    for i, (question, positions) in enumerate(pairs):
        answers = committed[2 * i : 2 * i + 2]
        keep = (
            all(map(is_committed, answers))
            and all(map(consistent, positions, answers))
            and all(on_side[2 * i : 2 * i + 2])
            and is_hedged(hedged[i])
            and lengths_match([*answers, hedged[i]])
        )
        if keep:
            records.append(
                {
                    "prompt": question,
                    "committed": [{"stance": s, "answer": a} for s, a in zip(positions, answers)],
                    "hedged": hedged[i],
                }
            )
    log.info("%d of %d questions passed the filters", len(records), len(pairs))
    return records[:n]


def write_hedging(records: list[dict], directory: Path) -> None:
    """Write the positive (committed) and negative (hedged) datasets for hedging records."""
    committed = [
        {"prompt": r["prompt"], "answer": c["answer"], "stance": c["stance"], "source": r["source"]}
        for r in records
        for c in r["committed"]
    ]
    hedged = [{"prompt": r["prompt"], "answer": r["hedged"], "source": r["source"]} for r in records]
    write_jsonl(directory / "committed.jsonl", committed)
    write_jsonl(directory / "hedged.jsonl", hedged)


def write_sample(records: list[dict], path: Path, rng: random.Random, k: int = 15) -> None:
    """Write ``k`` random generated records as Markdown for spot-checking."""
    lines = [f"# Spot-check sample ({k} of {len(records)} generated questions)\n"]
    for r in rng.sample(records, min(k, len(records))):
        lines.append(f"## {r['prompt']}\n")
        for c in r["committed"]:
            lines.append(f"**Committed ({c['stance']}):** {c['answer']}\n")
        lines.append(f"**Hedged:** {r['hedged']}\n")
    path.write_text("\n".join(lines))


RECIPES = {"hedging": (expand_hedging, write_hedging)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("behavior", choices=sorted(RECIPES))
    parser.add_argument("--n", type=int, default=300, help="total questions including seeds")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    log.setLevel(logging.INFO if args.verbose else logging.WARNING)

    directory = Path("data") / args.behavior
    seeds = [{**s, "source": "seed"} for s in read_jsonl(directory / "seeds.jsonl")]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    expand, write = RECIPES[args.behavior]
    model, tok, adapter = load(args.model)
    generated = [{**r, "source": "generated"} for r in expand(model, tok, adapter, seeds, args.n - len(seeds), rng)]
    write(seeds + generated, directory)
    write_sample(generated, directory / "sample.md", rng)
    print(f"{len(seeds)} seeds + {len(generated)} generated -> {directory}")


if __name__ == "__main__":
    main()
