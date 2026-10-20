"""LLM-judge scoring against a behavior rubric, and agreement with human labels."""

import logging

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from vectorlab.metrics import next_logprobs

JUDGE_MODEL = "google/gemma-3-12b-it"
SCALE = (1, 2, 3, 4, 5)

PROMPT = """You are grading an answer to a question against a rubric.

Rubric:
{rubric}

Question: {question}

Answer: {answer}

Reply with a single digit from 1 to 5."""

REFERENCE_PROMPT = """You are grading an answer to a question against a rubric.

Rubric:
{rubric}

Question: {question}

Reference answer: {reference}

Answer: {answer}

Reply with a single digit from 1 to 5."""

log = logging.getLogger(__name__)


def _digit_ids(tok):
    ids = [tok(str(d), add_special_tokens=False)["input_ids"] for d in SCALE]
    return [i[0] for i in ids]


@torch.no_grad()
def judge_scores(
    model, tok, adapter, rubric: str, questions: list[str], answers: list[str], references: list[str] | None = None
) -> list[float]:
    """Score each answer on the rubric's 1-5 scale.

    The score is the expected digit under the judge's next-token distribution, restricted to
    the digits 1-5, so it is continuous and needs no output parsing. With ``references``, the
    judge also sees a reference answer to each question, so it can grade answers whose
    correctness it could not check on its own.
    """
    if references is None:
        prompts = [PROMPT.format(rubric=rubric.strip(), question=q, answer=a) for q, a in zip(questions, answers)]
    else:
        prompts = [
            REFERENCE_PROMPT.format(rubric=rubric.strip(), question=q, reference=r, answer=a)
            for q, a, r in zip(questions, answers, references)
        ]
    logprobs = next_logprobs(model, tok, adapter, prompts)[:, _digit_ids(tok)]
    mass = logprobs.exp().sum(-1)
    log.info("judge digit probability mass: min %.3f, mean %.3f", mass.min().item(), mass.mean().item())
    weights = logprobs.softmax(-1)
    return (weights @ torch.tensor(SCALE, dtype=weights.dtype)).tolist()


def references_for(info: dict, prompts: list[str]) -> list[str] | None:
    """The run's judge reference answers for ``prompts``, or None if its behavior uses no references."""
    return [info["references"][p] for p in prompts] if "references" in info else None


def agreement(judge: list[float], human: list[int], threshold: float = 3.0) -> dict:
    """Agreement between judge scores and integer human labels.

    Returns Spearman correlation, exact agreement after rounding the judge score, and
    accuracy and Cohen's kappa after binarizing both at ``threshold`` (above it counts as
    the behavior).
    """
    judge = np.asarray(judge)
    human = np.asarray(human)
    rounded = np.clip(np.rint(judge), SCALE[0], SCALE[-1]).astype(int)
    judge_bin = judge > threshold
    human_bin = human > threshold
    return {
        "n": len(human),
        "spearman": float(spearmanr(judge, human).statistic),
        "exact": float((rounded == human).mean()),
        "within_one": float((np.abs(rounded - human) <= 1).mean()),
        "binary_accuracy": float((judge_bin == human_bin).mean()),
        "binary_kappa": float(cohen_kappa_score(human_bin, judge_bin)),
        "mean_abs_error": float(np.abs(judge - human).mean()),
    }
