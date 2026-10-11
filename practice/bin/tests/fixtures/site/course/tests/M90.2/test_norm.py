import json
import os
from pathlib import Path

import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.demo.norm import l1_normalize


def test_hand_example():
    # WHY: the chapter's worked example: [1, -3] has L1 norm 4, so [0.25, -0.75].
    # KIND: unit
    assert_close(l1_normalize([1.0, -3.0]), [0.25, -0.75], dtype="float32")


def test_fixture_vectors():
    # WHY: the committed fixture pins a few vectors with known answers, so the
    #      test and the chapter agree on the definition across languages.
    # KIND: golden
    rows = json.loads((Path(os.environ["TINYLLM_FIXTURES"]) / "M90.2" / "vectors.json").read_text())
    for row in rows:
        assert_close(l1_normalize(row["x"]), row["y"], dtype="float32")


def test_abs_sums_to_one():
    # WHY: the defining property of L1 normalization, over seeded random inputs.
    # KIND: property
    rng = PCG32(seed=int(os.environ.get("OL_SEED", "0")))
    for _ in range(20):
        xs = [rng.uniform() * 2 - 1 for _ in range(1 + rng.below(16))]
        assert_close(sum(abs(v) for v in l1_normalize(xs)), 1.0, dtype="float32")


def test_all_zero_raises():
    # WHY: an all-zero vector has no direction; dividing by 0 would produce nan.
    # KIND: boundary
    with pytest.raises(ValueError):
        l1_normalize([0.0, 0.0])
