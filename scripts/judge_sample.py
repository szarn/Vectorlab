"""Build a blind sample of answers for hand-labeling, to validate a behavior's judge.

Usage:
    python scripts/judge_sample.py runs/<run> [--dataset 20] [--baseline 20] \
        [--steer add:21:1.5:10 --steer ablate:21:1.0:10] [--seed 0] [-v]

The sample mixes held-out dataset answers (half from each class), the run model's own
answers to held-out prompts, and answers generated under steering, so labels cover the
whole scale. Each ``--steer MODE:LAYER:SCALE:COUNT`` adds the layer's direction with
SCALE times its mean-difference norm, or ablates it from every layer with strength SCALE.
Items are shuffled and written without their condition to ``judge_sample.jsonl``. The
conditions go to ``judge_sample_key.jsonl``, to stay unread until labeling is done.
"""

import argparse
import random
from pathlib import Path

from vectorlab.behavior import write_jsonl
from vectorlab.cli import configure_logging
from vectorlab.directions import edit_direction
from vectorlab.evaluate import eval_prompts
from vectorlab.generate import generate
from vectorlab.hooks import intervene
from vectorlab.metrics import MAX_TOKENS
from vectorlab.model import load
from vectorlab.store import load_json, load_tensors


def parse_steer(text: str) -> tuple[str, int, float, int]:
    """Split a ``MODE:LAYER:SCALE:COUNT`` steering spec."""
    mode, layer, scale, count = text.split(":")
    return mode, int(layer), float(scale), int(count)


def steered(model, tok, adapter, tensors, spec: tuple[str, int, float, int], prompts: list[str]) -> list[str]:
    """Answers to ``prompts`` with one steering condition applied through hooks."""
    mode, layer, scale, _ = spec
    layers = [layer] if mode == "add" else list(range(len(adapter.layers)))
    strength = scale * tensors["diff_norms"][layer].item() if mode == "add" else scale
    with intervene(adapter, edit_direction(tensors, layer, mode), mode, layers, strength):
        return generate(model, tok, adapter, prompts, MAX_TOKENS["judge"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=int, default=20, help="held-out dataset answers, half from each class")
    parser.add_argument("--baseline", type=int, default=20, help="the run model's own answers")
    parser.add_argument("--steer", action="append", default=[], type=parse_steer, help="MODE:LAYER:SCALE:COUNT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    info = load_json(args.run / "run.json")
    spec = info["behavior"]
    rng = random.Random(args.seed)
    labels = [Path(spec["positive"]).stem, Path(spec["negative"]).stem]
    items = [(r["prompt"], r["answer"], f"dataset {label}") for records, label in zip((info["test"]["pos"], info["test"]["neg"]), labels) for r in rng.sample(records, args.dataset // 2)]

    prompts = eval_prompts(info)
    rng.shuffle(prompts)
    model, tok, adapter = load(info["model"])
    tensors = load_tensors(args.run / "directions.safetensors")
    chosen, prompts = prompts[: args.baseline], prompts[args.baseline :]
    items += [(p, a, "baseline") for p, a in zip(chosen, generate(model, tok, adapter, chosen, MAX_TOKENS["judge"]))]
    for steer in args.steer:
        chosen, prompts = prompts[: steer[3]], prompts[steer[3] :]
        condition = f"steered {steer[0]} L{steer[1]} x{steer[2]}"
        items += [(p, a, condition) for p, a in zip(chosen, steered(model, tok, adapter, tensors, steer, chosen))]

    rng.shuffle(items)
    directory = Path(spec["positive"]).parent
    write_jsonl(directory / "judge_sample.jsonl", [{"id": i, "question": q, "answer": a} for i, (q, a, _) in enumerate(items)])
    write_jsonl(directory / "judge_sample_key.jsonl", [{"id": i, "condition": c, "source": str(args.run)} for i, (_, _, c) in enumerate(items)])
    print(f"{len(items)} items -> {directory / 'judge_sample.jsonl'}")


if __name__ == "__main__":
    main()
