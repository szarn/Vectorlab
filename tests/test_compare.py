"""Cross-effect statistics and the choice of each edit's own-effect numbers."""

import pytest

from vectorlab.compare import change, own_effect

EDIT = {"layer": 3, "mode": "add", "strength": 2.0, "layers": [3, 4], "weights": False}


def test_change_is_paired():
    result = change([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])
    assert result["delta"] == pytest.approx(1.0)
    assert result["ci"] == pytest.approx(0.0)
    assert result["standardized"] == pytest.approx(1.0 / 1.2909944, rel=1e-6)


def test_own_effect_prefers_matching_eval():
    run = {"edit": EDIT, "eval": {"edit": EDIT, "before": 1.0, "after": 2.0, "kl": 0.1, "n": 60}, "search": None}
    assert own_effect(run)["source"] == "held-out eval"


def test_own_effect_falls_back_to_matching_trial():
    trial = {"number": 7, "config": {**EDIT, "weights": False}, "score": 3.0, "kl": 0.2}
    run = {
        "edit": {**EDIT, "preserve_norms": False},
        "eval": {"edit": {**EDIT, "strength": 1.0}, "before": 1.0, "after": 2.0, "kl": 0.1, "n": 60},
        "search": {"baseline": {"score": 1.5}, "n_prompts": 32, "trials": [trial]},
    }
    effect = own_effect(run)
    assert (effect["before"], effect["after"], effect["n"]) == (1.5, 3.0, 32)
    assert "trial 7" in effect["source"]


def test_own_effect_without_matching_results():
    assert own_effect({"edit": EDIT, "eval": None, "search": None}) is None
    assert own_effect({"edit": None, "eval": None, "search": None}) is None
