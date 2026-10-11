"""The batch- and chunk-invariance helpers (DESIGN 2.4): Python
(_lib/invariance.py) and C (ol_invariance.h) accept invariant kernels and
name the first differing bit pattern of a kernel whose reduction order
depends on the batch or the chunk."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

COURSE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COURSE / "tests"))
from _lib.invariance import (
    assert_batch_invariant,
    assert_bits_equal,
    assert_chunk_invariant,
)  # noqa: E402

RNG = np.random.default_rng(7)  # test data only; course tests use _lib.pcg32
X = RNG.standard_normal((9, 300)).astype(np.float32)
W = RNG.standard_normal((300, 5)).astype(np.float32)


def row_matmul(xb):
    # a fixed k order per row: invariant to the batch
    out = np.zeros((xb.shape[0], W.shape[1]), dtype=np.float32)
    for i in range(xb.shape[0]):
        acc = np.zeros(W.shape[1], dtype=np.float32)
        for k in range(xb.shape[1]):
            acc += xb[i, k] * W[k]
        out[i] = acc
    return out


def split_k_matmul(xb):
    # splits K in two when the batch is small: a different summation order
    if xb.shape[0] >= 4:
        return row_matmul(xb)
    half = xb.shape[1] // 2
    a = row_matmul_k(xb, 0, half)
    b = row_matmul_k(xb, half, xb.shape[1])
    return (a + b).astype(np.float32)


def row_matmul_k(xb, lo, hi):
    out = np.zeros((xb.shape[0], W.shape[1]), dtype=np.float32)
    for i in range(xb.shape[0]):
        for k in range(lo, hi):
            out[i] += xb[i, k] * W[k]
    return out


def test_batch_invariant_kernel_passes():
    assert_batch_invariant(row_matmul, X[:6], sizes=(1, 2, 3, 6))


def test_batch_variant_kernel_fails_with_the_row_and_ulps():
    with pytest.raises(AssertionError, match=r"row \d+ inside a batch of \d+ .* ulp"):
        assert_batch_invariant(split_k_matmul, X[:6], sizes=(4,))


def test_chunk_invariance():
    xs = X.reshape(-1)

    def sum_in_chunks(c):
        acc = np.float32(0)
        step = c or xs.size
        for i in range(0, xs.size, step):
            for v in xs[i : i + step]:
                acc = np.float32(acc + v)
        return np.array([acc], dtype=np.float32)

    assert_chunk_invariant(sum_in_chunks, [None, 1, 7, 64])  # same order, same bits

    def tree_in_chunks(c):
        step = c or xs.size
        parts = [
            np.float32(np.sum(xs[i : i + step], dtype=np.float32))
            for i in range(0, xs.size, step)
        ]
        return np.array([np.float32(sum(parts, np.float32(0)))], dtype=np.float32)

    with pytest.raises(AssertionError, match="chunk size"):
        assert_chunk_invariant(tree_in_chunks, [None, 1, 7, 64, 2])


def test_bits_equal_sees_signed_zero_and_nan_payloads():
    assert_bits_equal(
        np.array([0.0, np.nan], np.float32), np.array([0.0, np.nan], np.float32)
    )
    with pytest.raises(AssertionError):
        assert_bits_equal(np.array([0.0], np.float32), np.array([-0.0], np.float32))


C_TEST = r"""
#include "ol_test.h"
#include "ol_invariance.h"

#define K 64
#define N 3
static float w[K * N];

static void rows_fixed(void *ctx, const float *x, int64_t m, float *out) {
    (void)ctx;
    for (int64_t i = 0; i < m; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += x[i * K + k] * w[k * N + j];
            out[i * N + j] = acc;
        }
}

static void rows_split(void *ctx, const float *x, int64_t m, float *out) {
    (void)ctx;
    for (int64_t i = 0; i < m; i++)
        for (int j = 0; j < N; j++) {
            float a = 0.0f, b = 0.0f;
            int split = m > 2 ? K / 2 : K;
            for (int k = 0; k < split; k++) a += x[i * K + k] * w[k * N + j];
            for (int k = split; k < K; k++) b += x[i * K + k] * w[k * N + j];
            out[i * N + j] = a + b;
        }
}

static float xs[5 * K];

OL_TEST(fixed_order_is_invariant) { OL_BATCH_INVARIANT(rows_fixed, NULL, xs, 5, K, N); }
OL_TEST(split_k_is_not) { OL_BATCH_INVARIANT(rows_split, NULL, xs, 5, K, N); }

int main(void) {
    unsigned s = 12345u;
    for (int i = 0; i < K * N; i++) { s = s * 1103515245u + 12345u; w[i] = (float)(s >> 8) / 16777216.0f - 0.5f; }
    for (int i = 0; i < 5 * K; i++) { s = s * 1103515245u + 12345u; xs[i] = (float)(s >> 8) / 16777216.0f * 3.0f; }
    return OL_RUN_ALL();
}
"""


def test_c_batch_invariance(tmp_path):
    src = tmp_path / "t.c"
    src.write_text(C_TEST)
    exe = tmp_path / "t"
    inc = COURSE / "contracts" / "c" / "include"
    subprocess.run(
        ["cc", "-std=c11", "-O1", "-Wall", f"-I{inc}", str(src), "-o", str(exe)],
        check=True,
    )
    p = subprocess.run([str(exe)], capture_output=True, text=True)
    assert p.returncode == 1
    assert "ok   fixed_order_is_invariant" in p.stdout
    assert (
        "FAIL split_k_is_not" in p.stdout
        and "inside a batch of 3" in p.stdout
        and "ulp" in p.stdout
    )
