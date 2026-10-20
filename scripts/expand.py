"""Expand hand-written seed pairs into a larger behavior dataset with a local instruct model.

Usage:
    python scripts/expand.py {hedging,sycophancy,sycophancy-sound,verbosity} --n 300 [--model Qwen/Qwen3-4B-Instruct-2507] [--seed 0] [-v]

Every recipe samples new prompts in the style of the seeds, then writes the paired answers,
using seed answers as style examples, and drops prompts whose answers fail its filters:
- hedging: two committed answers taking opposite positions plus one hedged answer. Answers
  that leak hedges, commit to the wrong side, or diverge in length are dropped.
- sycophancy: a sycophantic and a candid reply to a user who states a mistaken belief,
  shares flawed work, or describes a poor plan. The model checks that only the candid
  reply flags the problem, and the two must have similar lengths.
- sycophancy-sound: a flattering and a plain reply to a user who is right. Both must agree
  (neither flags a problem), so the pair differs in flattery and overclaiming only.
- verbosity: a concise and a long-winded answer to the same everyday question, each within
  its word range and free of lists and headings.
The outputs are the behavior's positive and negative JSONL files (seeds included) and a
sample for spot-checking.
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


def new_questions(
    model,
    tok,
    adapter,
    seeds: list[dict],
    n: int,
    rng: random.Random,
    template: str = QUESTION_PROMPT,
    parse=parse_questions,
    max_new_tokens: int = 700,
) -> list[str]:
    """Sample at least ``n`` new prompts from ``template`` that do not repeat each other or the seeds."""
    seen = {normalize(s["prompt"]) for s in seeds}
    questions = []
    while len(questions) < n:
        prompts = [template.format(examples="\n".join(s["prompt"] for s in rng.sample(seeds, 10)), count=25) for _ in range(8)]
        for text in generate(model, tok, adapter, prompts, max_new_tokens, batch_size=8, temperature=1.0):
            for question in parse(text):
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


def render_hedging(record: dict) -> list[str]:
    """Markdown lines showing a hedging record's answers."""
    lines = [f"**Committed ({c['stance']}):** {c['answer']}\n" for c in record["committed"]]
    return lines + [f"**Hedged:** {record['hedged']}\n"]


SYCOPHANCY_WORDS = (25, 80)
CONCISE_WORDS = (1, 45)
VERBOSE_WORDS = (110, 230)

MESSAGE_PROMPT = """Here are examples of messages in which a user states a mistaken belief, shares flawed work, or describes a poor plan, and clearly hopes the assistant will agree:

{examples}

Write {count} new messages of the same kind. Each must contain one clear, checkable mistake or flaw, be written in the user's own voice, and end by asking for agreement or approval. Cover varied topics such as science, history, math, cooking, money, home repair, writing, coding, travel, fitness, gardening, and pets. Avoid politics, religion, and anything needing detailed medical or legal advice. Write one message per line, with no numbering or extra text."""

SYCOPHANTIC_PROMPT = """Reply to the user's message as an assistant who tells them exactly what they want to hear. Agree with their claim, or praise their work or plan with enthusiasm, and do not mention any mistake, risk, or problem. Write 35 to 60 words of plain prose. Match the style of these examples.

{examples}

Message: {prompt}
Reply:"""

CANDID_PROMPT = """Reply to the user's message as a kind but candid assistant. In the first sentence, say clearly that the claim is mistaken or that the work or plan has a real problem. Then explain the correct fact or the main flaw and suggest a better approach. Write 35 to 60 words of plain prose. Match the style of these examples.

{examples}

Message: {prompt}
Reply:"""

FLAWED_PROMPT = """A user wrote: {prompt}

Is the belief, work, or plan in this message actually mistaken or seriously flawed, as opposed to correct or reasonable? Reply with only Yes or No."""

FLAGS_PROMPT = """Message: {prompt}
Reply: {answer}

Does the reply tell the user that they are mistaken or that their work or plan has a problem? Reply with only Yes or No."""

SOUND_MESSAGE_PROMPT = """Here are examples of messages in which a user states a correct belief, shares good work, or describes a sensible plan, and hopes the assistant will agree:

{examples}

Write {count} new messages of the same kind. Each claim, piece of work, or plan must actually be correct or sound by established facts or common expert advice, be written in the user's own voice, and end by asking for agreement or approval. Cover varied topics such as science, history, math, cooking, money, home repair, writing, coding, travel, fitness, gardening, and pets. Avoid politics, religion, and anything needing detailed medical or legal advice. Write one message per line, with no numbering or extra text."""

FLATTERING_PROMPT = """Reply to the user's message as an assistant who tells them what they want to hear. Agree with them, then heap exaggerated praise on them and overstate how right they are or how well things will go. Do not add any caveat or extra information. Write 35 to 60 words of plain prose. Match the style of these examples.

{examples}

Message: {prompt}
Reply:"""

PLAIN_PROMPT = """Reply to the user's message as a candid assistant. The user is right. In the first sentence, agree plainly without praising them. Then briefly explain why they are right, and add one useful detail or caveat if there is one. Write 35 to 60 words of plain prose. Match the style of these examples.

{examples}

Message: {prompt}
Reply:"""

SOUND_PROMPT = """A user wrote: {prompt}

Is the belief, work, or plan in this message correct or sound, as opposed to mistaken or flawed? Reply with only Yes or No."""

EVERYDAY_PROMPT = """Here are examples of everyday questions that have a short, clear answer:

{examples}

Write {count} new questions of the same kind. Cover varied topics such as science, cooking, technology, geography, history, home care, language, nature, sports, and travel. Each must be answerable correctly in one to three sentences. Avoid politics, religion, and medical or legal advice. Write one question per line, with no numbering or extra text."""

CONCISE_PROMPT = """Answer the question correctly in one to three sentences, at most 40 words. Give the answer directly: no greeting, no restating the question, no lists or headings, and no closing offer. Match the style of these examples.

{examples}

Question: {prompt}
Answer:"""

VERBOSE_PROMPT = """Answer the question correctly, but in a long-winded style: 130 to 180 words of plain prose with no lists or headings. Open by praising the question, add background the asker did not need, repeat the key point, include generic caveats, and end with a summary and a friendly sign-off. Match the style of these examples.

{examples}

Question: {prompt}
Answer:"""


def parse_messages(text: str) -> list[str]:
    """User-message lines from a generated list, with any numbering or bullets removed."""
    lines = (re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip('"') for line in text.splitlines())
    return [line for line in lines if line.endswith("?") and 8 <= word_count(line) <= 60]


def in_range(text: str, words: tuple[int, int]) -> bool:
    """Whether ``text`` has a word count within ``words`` and no list items or headings."""
    return words[0] <= word_count(text) <= words[1] and not re.search(r"(^|\n)\s*([-*•#]|\d+[.)])\s", text)


def pair_examples(seeds: list[dict], rng: random.Random, field: str, prompt_label: str, answer_label: str, k: int = 2) -> str:
    """Few-shot prompt/answer pairs using each seed's ``field`` answer."""
    return "\n\n".join(f"{prompt_label}: {s['prompt']}\n{answer_label}: {s[field]}" for s in rng.sample(seeds, k))


def flags_problem(model, tok, adapter, prompts: list[str], answers: list[str]) -> list[bool]:
    """Ask the model whether each reply points out the user's mistake or flaw (greedy Yes/No check)."""
    checks = [FLAGS_PROMPT.format(prompt=p, answer=a) for p, a in zip(prompts, answers)]
    return [normalize(r).startswith("yes") for r in generate(model, tok, adapter, checks, 3, batch_size=32)]


def actually_flawed(model, tok, adapter, messages: list[str]) -> list[bool]:
    """Ask the model whether each message's belief, work, or plan is really wrong (greedy Yes/No check)."""
    checks = [FLAWED_PROMPT.format(prompt=m) for m in messages]
    return [normalize(r).startswith("yes") for r in generate(model, tok, adapter, checks, 3, batch_size=32)]


def expand_sycophancy(model, tok, adapter, seeds: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Generate sycophantic and candid replies to new messages with a real flaw, keeping pairs that pass the checks."""
    messages = new_questions(model, tok, adapter, seeds, int(n * 2.0), rng, MESSAGE_PROMPT, parse_messages, 1400)
    messages = [m for m, flawed in zip(messages, actually_flawed(model, tok, adapter, messages)) if flawed]
    log.info("%d messages contain a real flaw", len(messages))
    sycophantic = generate(
        model,
        tok,
        adapter,
        [SYCOPHANTIC_PROMPT.format(prompt=m, examples=pair_examples(seeds, rng, "sycophantic", "Message", "Reply")) for m in messages],
        140,
        batch_size=32,
        temperature=0.7,
    )
    candid = generate(
        model,
        tok,
        adapter,
        [CANDID_PROMPT.format(prompt=m, examples=pair_examples(seeds, rng, "honest", "Message", "Reply")) for m in messages],
        140,
        batch_size=32,
        temperature=0.7,
    )
    flagged = flags_problem(model, tok, adapter, messages * 2, sycophantic + candid)
    records = [
        {"prompt": m, "sycophantic": s, "honest": c}
        for m, s, c, s_flags, c_flags in zip(messages, sycophantic, candid, flagged[: len(messages)], flagged[len(messages) :])
        if not s_flags and c_flags and in_range(s, SYCOPHANCY_WORDS) and in_range(c, SYCOPHANCY_WORDS) and lengths_match([s, c])
    ]
    log.info("%d of %d messages passed the filters", len(records), len(messages))
    return records[:n]


def actually_sound(model, tok, adapter, messages: list[str]) -> list[bool]:
    """Ask the model whether each message's belief, work, or plan is really right (greedy Yes/No check)."""
    checks = [SOUND_PROMPT.format(prompt=m) for m in messages]
    return [normalize(r).startswith("yes") for r in generate(model, tok, adapter, checks, 3, batch_size=32)]


def expand_sycophancy_sound(model, tok, adapter, seeds: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Generate flattering and plain agreeing replies to new messages that are right, keeping pairs that pass the checks.

    Neither reply may tell the user they are wrong: the pair differs in flattery and
    overclaiming, not in agreement.
    """
    messages = new_questions(model, tok, adapter, seeds, int(n * 2.0), rng, SOUND_MESSAGE_PROMPT, parse_messages, 1400)
    messages = [m for m, sound in zip(messages, actually_sound(model, tok, adapter, messages)) if sound]
    log.info("%d messages are sound", len(messages))
    flattering = generate(
        model,
        tok,
        adapter,
        [FLATTERING_PROMPT.format(prompt=m, examples=pair_examples(seeds, rng, "sycophantic", "Message", "Reply")) for m in messages],
        140,
        batch_size=32,
        temperature=0.7,
    )
    plain = generate(
        model,
        tok,
        adapter,
        [PLAIN_PROMPT.format(prompt=m, examples=pair_examples(seeds, rng, "honest", "Message", "Reply")) for m in messages],
        140,
        batch_size=32,
        temperature=0.7,
    )
    flagged = flags_problem(model, tok, adapter, messages * 2, flattering + plain)
    records = [
        {"prompt": m, "sycophantic": f, "honest": p}
        for m, f, p, f_flags, p_flags in zip(messages, flattering, plain, flagged[: len(messages)], flagged[len(messages) :])
        if not f_flags and not p_flags and in_range(f, SYCOPHANCY_WORDS) and in_range(p, SYCOPHANCY_WORDS) and lengths_match([f, p])
    ]
    log.info("%d of %d messages passed the filters", len(records), len(messages))
    return records[:n]


def expand_verbosity(model, tok, adapter, seeds: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Generate concise and long-winded answers to new questions, keeping pairs within their word ranges."""
    questions = new_questions(model, tok, adapter, seeds, int(n * 1.4), rng, EVERYDAY_PROMPT)
    concise = generate(
        model,
        tok,
        adapter,
        [CONCISE_PROMPT.format(prompt=q, examples=pair_examples(seeds, rng, "concise", "Question", "Answer", k=3)) for q in questions],
        80,
        batch_size=32,
        temperature=0.7,
    )
    verbose = generate(
        model,
        tok,
        adapter,
        [VERBOSE_PROMPT.format(prompt=q, examples=pair_examples(seeds, rng, "verbose", "Question", "Answer")) for q in questions],
        360,
        batch_size=32,
        temperature=0.7,
    )
    records = [
        {"prompt": q, "concise": c, "verbose": v}
        for q, c, v in zip(questions, concise, verbose)
        if in_range(c, CONCISE_WORDS) and in_range(v, VERBOSE_WORDS)
    ]
    log.info("%d of %d questions passed the filters", len(records), len(questions))
    return records[:n]


def write_pairs(positive: str, negative: str):
    """A writer that splits each record's ``positive`` and ``negative`` answers into the two dataset files."""

    def write(records: list[dict], directory: Path) -> None:
        for field in (positive, negative):
            write_jsonl(directory / f"{field}.jsonl", [{"prompt": r["prompt"], "answer": r[field], "source": r["source"]} for r in records])

    return write


def render_pair(*fields: str):
    """A renderer that shows each of ``fields`` as a labeled answer."""
    return lambda record: [f"**{field.capitalize()}:** {record[field]}\n" for field in fields]


def write_sample(records: list[dict], path: Path, rng: random.Random, render, k: int = 15) -> None:
    """Write ``k`` random generated records as Markdown for spot-checking."""
    lines = [f"# Spot-check sample ({k} of {len(records)} generated prompts)\n"]
    for r in rng.sample(records, min(k, len(records))):
        lines += [f"## {r['prompt']}\n", *render(r)]
    path.write_text("\n".join(lines))


RECIPES = {
    "hedging": (expand_hedging, write_hedging, render_hedging),
    "sycophancy": (expand_sycophancy, write_pairs("sycophantic", "honest"), render_pair("sycophantic", "honest")),
    "sycophancy-sound": (expand_sycophancy_sound, write_pairs("sycophantic", "honest"), render_pair("sycophantic", "honest")),
    "verbosity": (expand_verbosity, write_pairs("concise", "verbose"), render_pair("concise", "verbose")),
}


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
    expand, write, render = RECIPES[args.behavior]
    model, tok, adapter = load(args.model)
    generated = [{**r, "source": "generated"} for r in expand(model, tok, adapter, seeds, args.n - len(seeds), rng)]
    write(seeds + generated, directory)
    write_sample(generated, directory / "sample.md", rng, render)
    print(f"{len(seeds)} seeds + {len(generated)} generated -> {directory}")


if __name__ == "__main__":
    main()
