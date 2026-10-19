"""The course's reference eval suite for craft.22 (rung R8).

A model eval as a test: measure the model on the held-out text and with
seeded samples over five seeds, and fail when it is worse than the
baseline recorded from the good model, by more than a margin AND with a
one-sided sign-flip p-value below 0.05. Five seeds is the minimum: with
four, the smallest possible p-value is 1/16 > 0.05, so nothing could
ever be flagged.
"""

import json
import os
from pathlib import Path

import evals
import pytest
import tinymodel

HERE = Path(__file__).resolve().parent
FIX = Path(os.environ["TINYLLM_FIXTURES"]) / "craft.22"


@pytest.fixture(scope="module")
def current():
    model = tinymodel.load(FIX / "tinylm.npz")
    text = (FIX / "val.txt").read_text()
    vocab = set((FIX / "words.txt").read_text().split())
    return evals.measure(model, text, vocab)


@pytest.fixture(scope="module")
def baseline():
    return json.loads((HERE / "baseline.json").read_text())


def test_seeds_match_the_baseline(current, baseline):
    assert current["seeds"] == baseline["seeds"] and len(baseline["seeds"]) >= 5


def test_heldout_bpb_did_not_regress(current, baseline):
    assert not evals.regressed(baseline["bpb"], current["bpb"], evals.BPB_MARGIN), (
        f"bpb {current['bpb']} vs baseline {baseline['bpb']}"
    )


def test_sample_quality_did_not_regress(current, baseline):
    assert not evals.regressed(
        baseline["quality"], current["quality"], evals.QUALITY_MARGIN, higher_is_better=True
    ), f"quality {current['quality']} vs baseline {baseline['quality']}"
