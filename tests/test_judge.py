"""Judge reference answers: lookup from a run and validation in behavior specs."""

import json

import pytest

from vectorlab.behavior import validate_behavior
from vectorlab.errors import UserError
from vectorlab.judge import references_for


def test_references_for_looks_up_each_prompt():
    info = {"references": {"a": "ref a", "b": "ref b"}}
    assert references_for(info, ["b", "a"]) == ["ref b", "ref a"]
    assert references_for({}, ["a"]) is None


def spec_with(tmp_path, **fields):
    for name in ("pos", "neg"):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps({"prompt": "p", "answer": "a"}) + "\n")
    base = {
        "name": "x",
        "positive": str(tmp_path / "pos.jsonl"),
        "negative": str(tmp_path / "neg.jsonl"),
        "capture": "response",
        "intervention": "ablate",
        "metric": "judge",
        "target": "min",
        "rubric": "r",
    }
    return {**base, **fields}


def test_judge_reference_is_accepted_for_judged_response_behaviors(tmp_path):
    validate_behavior(spec_with(tmp_path, judge_reference="negative"), tmp_path / "x.yaml")


@pytest.mark.parametrize("fields", [{"judge_reference": "positive"}, {"judge_reference": "negative", "metric": "verbosity"}])
def test_judge_reference_rejects_bad_specs(tmp_path, fields):
    with pytest.raises(UserError):
        validate_behavior(spec_with(tmp_path, **fields), tmp_path / "x.yaml")
