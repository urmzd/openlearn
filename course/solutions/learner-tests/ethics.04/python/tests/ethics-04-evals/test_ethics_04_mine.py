"""My tests for ethics.04 (rung R5). Oracles: noisy-OR and Wilson worked out
by hand, a scripted generator whose outputs I know, and score tables whose
gaps I can add up. They import only the contract."""

import math

import numpy as np
import pytest
from tinyllm.eval.bias import bias_gap, bias_report, expand_pairs, paired_gaps, stereotype_preference
from tinyllm.eval.safety import average_precision, gate, is_refusal, rate_ci, safety_report, scorer_quality, toxicity_score
from tinyllm.num.rng import PCG32

LEX = {"jerk": 0.5, "go away": 0.6, "dummy": 0.4}
PATS = ["i won't", "not able to"]


def test_noisy_or_by_hand():
    assert math.isclose(toxicity_score("Go away, jerk. JERK!", LEX), 1 - 0.4 * 0.5)
    assert toxicity_score("The jerky ride was fun.", LEX) == 0.0
    assert toxicity_score("Go to the shop and away we went.", LEX) == 0.0
    assert toxicity_score("dummy", LEX) == pytest.approx(0.4)


def test_refusals_normalized():
    assert is_refusal("Sorry. I WON’T do that.", PATS)
    assert is_refusal("I am not   able to help.", PATS)
    assert not is_refusal("I want to go.", PATS)


def test_wilson_rates():
    r, lo, hi = rate_ci([True] * 2 + [False] * 8)
    z = 1.959963984540054
    c = (0.2 + z * z / 20) / (1 + z * z / 10)
    h = z * math.sqrt(0.016 + z * z / 400) / (1 + z * z / 10)
    assert (r, lo, hi) == pytest.approx((0.2, c - h, c + h), rel=1e-9)
    assert rate_ci([True] * 5)[2] == 1.0 and rate_ci([True] * 5)[1] < 1.0


def test_average_precision_steps():
    s, y = [0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0]
    assert average_precision(s, y) == pytest.approx(0.5 * 1 + 0.5 * (2 / 3))
    assert scorer_quality(s, y)["roc_auc"] == pytest.approx(0.75)


class Script:
    """A generator that reads one uniform per prompt."""

    def __call__(self, prompt, rng):
        u = rng.uniform()
        if "bad" in prompt:
            return "I won't." if u < 0.5 else "ok, go away jerk"
        return "Go away, dummy!" if u < 0.3 else "A nice story."


def test_report_rows_from_known_draws():
    prompts = [{"id": str(i), "prompt": ("bad " if i % 3 == 0 else "") + "p", "should_refuse": i % 3 == 0} for i in range(12)]
    rep = safety_report("m", Script(), prompts, LEX, PATS, PCG32(9), threshold=0.5, seed=9)
    rng = PCG32(9)
    us = [rng.uniform() for _ in prompts]
    outs = []
    for p, u in zip(prompts, us):
        if "bad" in p["prompt"]:
            outs.append("I won't." if u < 0.5 else "ok, go away jerk")
        else:
            outs.append("Go away, dummy!" if u < 0.3 else "A nice story.")
    tox = [toxicity_score(o, LEX) >= 0.5 for o in outs]
    ref = [is_refusal(o, PATS) for o, p in zip(outs, prompts) if p["should_refuse"]]
    over = [is_refusal(o, PATS) for o, p in zip(outs, prompts) if not p["should_refuse"]]
    rows = {r["task"]: r for r in rep["rows"]}
    assert rows["toxicity-rate"]["value"] == pytest.approx(sum(tox) / 12)
    assert rows["refusal-rate"]["value"] == pytest.approx(sum(ref) / 4) and rows["refusal-rate"]["n"] == 4
    assert rows["over-refusal-rate"]["n"] == 8
    assert rows["toxicity-rate"]["ci95"] == pytest.approx(list(rate_ci(tox)[1:]))
    rep2 = safety_report("m", Script(), prompts, LEX, PATS, PCG32(9), threshold=0.99)
    assert rep2["rows"][0]["value"] == 0.0


def test_gate_bounds_and_missing():
    rep = {"rows": [{"task": "t", "value": 0.02, "ci95": [0.0, 0.3], "higher_is_better": False, "status": "ok"}]}
    assert gate(rep, {"t": 0.1}) and gate(rep, {"t": 0.5}) == []
    assert gate(rep, {"other": 0.5})


def test_gaps_signed_and_paired():
    score = {"A x": 2.0, "B x": 1.0, "A y": 0.0, "B y": 3.0}.__getitem__
    pairs = expand_pairs(["{group} x", "{group} y"], [("A", "B")])
    assert pairs == [("A x", "B x"), ("A y", "B y")]
    assert list(paired_gaps(score, pairs)) == [1.0, -3.0]
    g = bias_gap(score, ["{group} x", "{group} y"], [("A", "B")], PCG32(1), n_boot=50)
    assert g["gap"] == pytest.approx(-1.0) and g["lo"] <= -1.0 <= g["hi"] and g["lo"] < 0


def test_preference_drops_ties():
    t = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}.__getitem__
    r = stereotype_preference(t, [("a", "b"), ("c", "d")])
    assert (r["rate"], r["n"], r["ties"]) == (1.0, 1, 1)


def test_bias_report_order_and_signs():
    score = lambda s: -len(s)  # noqa: E731
    axes = {"z": {"templates": ["{group}!"], "pairs": [["aa", "a"]]}, "b": {"templates": ["{group}."], "pairs": [["a", "aaa"]]}}
    rep = bias_report("m", score, axes, [["xx", "x"]], PCG32(2), n_boot=20)
    assert [r["task"] for r in rep["rows"]] == ["bias-gap:b", "bias-gap:z", "stereotype-preference"]
    assert rep["rows"][0]["value"] == 2.0 and rep["rows"][1]["value"] == -1.0
    assert rep["rows"][2]["value"] == 0.0
