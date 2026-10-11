"""M03.1 Python matrix product reference and its file-based C parity vectors."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close_bounded
from _lib.pcg32 import PCG32
from tinyllm.linalg.matmul import matmul_f32

FIX = Path(os.environ.get("TINYLLM_FIXTURES", ""))


def test_hand_example() -> None:
    # WHY: the defining product has a small, exact answer.
    # KIND: unit, smoke
    # CATCHES: s01, m001, m002, m003
    # CHAPTER: M03.1 section 3
    a = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    b = np.array([[7, 8], [9, 10], [11, 12]], np.float32)
    assert matmul_f32(a, b).tolist() == [[58.0, 64.0], [139.0, 154.0]]


def test_matches_frozen_c_parity_vectors() -> None:
    # WHY: the optional C module reads the same vectors from files, so both
    # implementations are compared to a frozen NumPy oracle.
    # KIND: differential, golden
    # CATCHES: s02, s05, s06
    # CHAPTER: M03.1 section 4
    vectors = json.loads((FIX / "parity" / "matmul.json").read_text())
    for case in vectors["cases"]:
        inp = case["input"]
        a = np.asarray(inp["a"], np.float32).reshape(inp["m"], inp["k"])
        b = np.asarray(inp["b"], np.float32).reshape(inp["k"], inp["n"])
        want = np.asarray(case["output"], np.float64).reshape(inp["m"], inp["n"])
        assert_close_bounded(matmul_f32(a, b), want, k=inp["k"], msg=case["name"])


@pytest.mark.parametrize("trans_b", [False, True])
def test_shapes_and_strided_views(trans_b: bool) -> None:
    # WHY: dimensions of one and padded row strides expose indexing mistakes.
    # KIND: differential
    # CATCHES: s02, s05, s06
    # CHAPTER: M03.1 section 4
    rng = PCG32(int(os.environ.get("OL_SEED", "0")), seq=31)
    for m, n, k in ((1, 1, 1), (2, 7, 3), (5, 1, 2), (7, 3, 33)):
        abuf = rng.uniform_array((m, k + 2), -1, 1).astype(np.float32)
        a = abuf[:, 1 : k + 1]
        shape = (n, k + 1) if trans_b else (k, n + 1)
        bbuf = rng.uniform_array(shape, -1, 1).astype(np.float32)
        b = bbuf[:, :k] if trans_b else bbuf[:, :n]
        op = b.T if trans_b else b
        want = a.astype(np.float64) @ op.astype(np.float64)
        outbuf = np.full((m, n + 2), np.nan, np.float32)
        out = outbuf[:, 1 : n + 1]
        got = matmul_f32(a, b, trans_b=trans_b, c=out)
        assert_close_bounded(got, want, k=k)
        assert np.isnan(outbuf[:, 0]).all() and np.isnan(outbuf[:, -1]).all()


def test_transpose_scaling_and_empty_inner_dimension() -> None:
    # WHY: transposed weights and beta accumulation are part of the contract;
    # an empty dot product contributes zero.
    # KIND: boundary
    # CATCHES: s02, s04, s07
    # CHAPTER: M03.1 section 2
    a = np.array([[1, 2, 3]], np.float32)
    bt = np.array([[7, 9, 11], [8, 10, 12]], np.float32)
    c = np.full((1, 2), 2, np.float32)
    assert matmul_f32(a, bt, trans_b=True, alpha=2, beta=0.5, c=c).tolist() == [
        [117, 129]
    ]
    empty_a = np.empty((2, 0), np.float32)
    empty_b = np.empty((0, 3), np.float32)
    empty = matmul_f32(empty_a, empty_b)
    assert empty.shape == (2, 3) and np.array_equal(empty, np.zeros((2, 3), np.float32))
    prior = np.full((2, 3), 3, np.float32)
    assert np.array_equal(
        matmul_f32(empty_a, empty_b, beta=0.5, c=prior),
        np.full((2, 3), 1.5, np.float32),
    )


def test_beta_zero_does_not_read_nan_c() -> None:
    # WHY: beta == 0 is an overwrite contract, including when C contains NaN.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: M03.1 section 2
    a = np.array([[2]], np.float32)
    b = np.array([[3]], np.float32)
    c = np.array([[np.nan]], np.float32)
    assert matmul_f32(a, b, beta=0, c=c).tolist() == [[6.0]]


def test_invalid_output_shape_is_rejected() -> None:
    # WHY: a mismatched output must fail at the interface boundary.
    # KIND: boundary
    # CATCHES: s08
    # CHAPTER: M03.1 section 2
    with pytest.raises(ValueError, match="shape"):
        matmul_f32(
            np.ones((2, 3), np.float32),
            np.ones((3, 4), np.float32),
            c=np.ones((2, 3), np.float32),
        )
