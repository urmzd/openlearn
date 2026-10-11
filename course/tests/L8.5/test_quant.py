"""Course tests for L8.5: weight and KV quantization.

The byte-exact oracle is course/fixtures/parity/quant_int4.json, the golden
file of the `quant.int4` parity suite (course/oracle/parity/quant_int4_golden.py,
shared with L9.5, whose kernel reads exactly these bytes); the chapter's
worked example is checked separately, by hand. Error bounds are checked against the rule's own promise
(at most half a step per element), and models are built from L0.4's Linear
with weights from the frozen PCG32 (course/tests/_lib).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32

from tinyllm.autograd.tensor import Tensor
from tinyllm.infer.quant import (
    FP8Tensor,
    KVQuant,
    MXTensor,
    Q4Tensor,
    Q8Tensor,
    QuantLinear,
    dequantize,
    export_q4,
    nbytes,
    output_error_bound,
    pack_int4,
    quantize_fp8_per_channel,
    quantize_int4_group,
    quantize_int8_per_channel,
    quantize_kv_fp8,
    quantize_model,
    quantize_mx,
    quant_ppl,
    unpack_int4,
)
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module

SEED = int(os.environ.get("OL_SEED", "0"))
FIX = Path(
    os.environ.get("TINYLLM_FIXTURES", Path(__file__).resolve().parents[2] / "fixtures")
)


def weights(g: PCG32, shape, scale=1.0) -> np.ndarray:
    """Heavy-tailed weights (a cube of uniforms), like trained matrices."""
    return (scale * g.uniform_array(shape, -1.0, 1.0) ** 3).astype(np.float32)


def f16_bits(a) -> list[int]:
    return np.asarray(a, dtype=np.float16).view(np.uint16).reshape(-1).tolist()


def test_hand_example_int4_group():
    # WHY: the chapter's worked example, one group of 4 per row. Row 0 has
    #      amax 1.75, so s = 1.75 / 7 = 0.25 (exact in float16) and
    #      w / s = 7, -2.4, 0.4, 1.2 rounds to 7, -2, 0, 1; row 1 is all ties:
    #      1.5, -0.5, 2.5 round to the even 2, 0, 2. Packing puts column 0 in
    #      the low nibble: bytes E7 10 and 02 72.
    # KIND: unit, smoke
    # CATCHES: s01, s02, s03, s04, m01
    # CHAPTER: L8.5 section 3
    w = np.array([[1.75, -0.6, 0.1, 0.3], [0.375, -0.125, 0.625, 1.75]], np.float32)
    q = quantize_int4_group(w, group=4)
    assert isinstance(q, Q4Tensor) and q.group == 4 and q.shape == (2, 4)
    assert q.packed.dtype == np.uint8 and q.packed.tolist() == [
        [0xE7, 0x10],
        [0x02, 0x72],
    ]
    assert q.scales.dtype == np.float16 and q.scales.tolist() == [[0.25], [0.25]]
    assert unpack_int4(q.packed).tolist() == [[7, -2, 0, 1], [2, 0, 2, 7]]
    assert dequantize(q).tolist() == [[1.75, -0.5, 0.0, 0.25], [0.5, 0.0, 0.5, 1.75]]


def test_hand_example_int8_per_channel():
    # WHY: int8's worked example: row 0 has amax 127, so its scale is 1 and
    #      127, -3.5, 2.5, 0.4 round half to even to 127, -4, 2, 0; a zero row
    #      gets scale 0 and codes 0 (no division by zero, no NaN); row 2 has
    #      scale 2, and 127 / 2 = 63.5 and 63 / 2 = 31.5 go to the even 64 and
    #      32. Codes stay in [-127, 127]: symmetric, so -128 is never used.
    # KIND: unit
    # CATCHES: s05, s06
    # CHAPTER: L8.5 section 3
    w = np.array([[127, -3.5, 2.5, 0.4], [0, 0, 0, 0], [-254, 127, 63, 0]], np.float32)
    q, s = quantize_int8_per_channel(w)
    assert q.dtype == np.int8 and s.dtype == np.float32
    assert q[0].tolist() == [127, -4, 2, 0]
    assert q[1].tolist() == [0, 0, 0, 0] and s[1] == 0
    assert q[2].tolist() == [-127, 64, 32, 0]
    assert s[0] == np.float32(1.0) and s[2] == np.float32(2.0)
    assert np.isfinite(dequantize((q, s))).all()
    assert dequantize(Q8Tensor(q, s))[0].tolist() == [127, -4, 2, 0]


def test_packed_bytes_match_the_golden():
    # WHY: the catalog's O test and the L9.5 byte-layout contract: every
    #      case of the parity golden file (shapes from 1 x 2 at group 2 to
    #      5 x 48 at group 16 and 2 x 64 at group 64) must give exactly its
    #      packed bytes and float16 scale bits. The same file drives
    #      `ol parity quant.int4`, where L9.5's kernel must decode them.
    # KIND: golden
    # CATCHES: s01, s02, s04, s08
    # CHAPTER: L8.5 section 4
    doc = json.loads((FIX / "parity" / "quant_int4.json").read_text())
    for case in doc["cases"]:
        i = case["input"]
        w = np.asarray(i["w"], np.float32).reshape(i["rows"], i["cols"])
        q = quantize_int4_group(w, i["group"])
        assert q.packed.reshape(-1).tolist() == case["output"]["packed"], case["name"]
        assert f16_bits(q.scales) == case["output"]["scales_f16"], case["name"]


def test_nibble_roundtrip_every_pair():
    # WHY: all 256 (even, odd) column pairs of values in [-8, 7] pack to one
    #      byte each and unpack to themselves: the low nibble is the even
    #      column and both are 4-bit two's complement (so -1 is 0xF and -8 is
    #      0x8), the rule the C kernel decodes with a shift and a sign extend.
    # KIND: property
    # CATCHES: s01, s02, s03
    # CHAPTER: L8.5 section 2
    v = np.arange(-8, 8)
    pairs = np.array([[a, b] for a in v for b in v], np.int8).reshape(1, -1)
    packed = pack_int4(pairs)
    assert packed.shape == (1, 256) and packed.dtype == np.uint8
    assert packed[0, :2].tolist() == [0x88, 0x98]  # (-8, -8), (-8, -7): -7 is 1001
    assert sorted(packed[0].tolist()) == list(range(256))
    assert unpack_int4(packed).tolist() == pairs.tolist()
    with pytest.raises(ValueError):
        pack_int4(np.array([[8, 0]]))
    with pytest.raises(ValueError):
        pack_int4(np.array([[1, 2, 3]]))


def test_int4_error_is_at_most_half_a_step():
    # WHY: the catalog's I test. Every weight is within half of its group's
    #      stored float16 scale of its dequantized value, at scales from 1e-3
    #      to 100 and groups 16, 32, 64. Quantizing against the float32 scale
    #      and then storing it as float16, or clipping at 7 with a scale of
    #      amax / 8, breaks this bound on some element.
    # KIND: property
    # CATCHES: s04, s07, s08
    # CHAPTER: L8.5 section 2
    g = PCG32(SEED, 31)
    for scale in (1e-3, 0.05, 1.0, 100.0):
        for group in (16, 32, 64):
            w = weights(g, (6, 128), scale)
            q = quantize_int4_group(w, group)
            s = np.repeat(q.scales.astype(np.float32), group, axis=1)
            err = np.abs(w - dequantize(q))
            assert (err <= s / 2 * (1 + 1e-6) + 1e-30).all(), (
                scale,
                group,
                float((err / s).max()),
            )


def test_int4_dequantizes_with_the_stored_scale():
    # WHY: L9.5 multiplies each code by the float16 scale it reads from the
    #      file, so dequantize must be exactly code * float16(scale), and the
    #      codes must have been chosen against that same float16 value. The
    #      hand-made row makes the difference visible: with amax 1 the stored
    #      scale is float16(1/7), slightly below 1/7, and a weight of exactly
    #      3.5 stored scales is a tie that rounds to the even 4, while against
    #      the float32 scale it is 3.499 and rounds to 3.
    # KIND: unit
    # CATCHES: s07
    # CHAPTER: L8.5 section 5, Pitfalls
    s16 = np.float32(np.float16(np.float32(1.0) / np.float32(7.0)))
    row = np.zeros((1, 8), np.float32)
    row[0, 0], row[0, 1] = 1.0, np.float32(3.5) * s16
    assert unpack_int4(quantize_int4_group(row, 8).packed)[0, :2].tolist() == [7, 4]
    g = PCG32(SEED, 32)
    w = weights(g, (4, 64), 3.0)
    q = quantize_int4_group(w, 32)
    s16 = q.scales.astype(np.float32)
    codes = unpack_int4(q.packed).astype(np.float32)
    assert (dequantize(q) == codes * np.repeat(s16, 32, axis=1)).all()
    expect = np.clip(np.rint(w.reshape(4, 2, 32) / s16[:, :, None]), -8, 7).reshape(
        4, 64
    )
    assert (codes == expect).all()


def test_zero_and_tiny_groups():
    # WHY: a pruned (all-zero) group and a group so small its scale
    #      underflows float16 must quantize to zeros with scale 0, never to
    #      NaN from 0 / 0; a scale that overflows float16 is an error, not
    #      an infinity in the file.
    # KIND: boundary
    # CATCHES: s09
    # CHAPTER: L8.5 section 5, Pitfalls
    w = np.zeros((2, 8), np.float32)
    w[1, :4] = 1e-9
    w[1, 4:] = [1.0, -0.5, 0.25, 0.0]
    q = quantize_int4_group(w, 4)
    assert q.scales[0].tolist() == [0.0, 0.0] and q.scales[1, 0] == 0.0
    d = dequantize(q)
    assert np.isfinite(d).all() and (d[0] == 0).all() and (d[1, :4] == 0).all()
    with pytest.raises(ValueError):
        quantize_int4_group(np.full((1, 4), 1e6, np.float32), 4)


def test_shape_rules():
    # WHY: the packed layout needs an even width and whole groups; a 1-D
    #      array or a NaN is a caller bug. Each is refused with ValueError
    #      instead of producing bytes the kernel would misread.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: L8.5 section 4
    with pytest.raises(ValueError):
        quantize_int4_group(np.ones((2, 6), np.float32), 4)  # 6 % 4
    with pytest.raises(ValueError):
        quantize_int4_group(np.ones((2, 6), np.float32), 3)  # odd group
    with pytest.raises(ValueError):
        quantize_int4_group(np.ones(8, np.float32), 4)
    with pytest.raises(ValueError):
        quantize_int8_per_channel(np.array([[1.0, np.nan]], np.float32))
    with pytest.raises(TypeError):
        dequantize(np.ones((2, 2)))


def test_int8_error_bound_and_range():
    # WHY: per-channel int8 keeps each element within half its row's step,
    #      never uses -128, and maps each row's largest magnitude to +-127
    #      exactly: the property that lets one int8 GEMV per row (L9.5's
    #      tl_matmul_q8_f32) replace the float weight.
    # KIND: property
    # CATCHES: s05, s06, s11
    # CHAPTER: L8.5 section 2
    g = PCG32(SEED, 33)
    w = weights(g, (16, 96), 2.0)
    q, s = quantize_int8_per_channel(w)
    assert q.min() >= -127 and np.abs(q).max(axis=1).tolist() == [127] * 16
    err = np.abs(w - dequantize((q, s)))
    assert (err <= s[:, None] / 2 * (1 + 1e-6)).all()


def test_fp8_per_channel_relative_error():
    # WHY: E4M3 keeps 3 mantissa bits, so any value at least 2^-6 of its
    #      row's scale lands within a relative 2^-4 of itself; the row's
    #      largest magnitude maps to 448 exactly (scale = amax / 448), so
    #      nothing saturates.
    # KIND: property
    # CATCHES: s12
    # CHAPTER: L8.5 section 2
    g = PCG32(SEED, 34)
    w = weights(g, (8, 64), 5.0)
    q = quantize_fp8_per_channel(w, "e4m3")
    assert (
        isinstance(q, FP8Tensor)
        and q.codes.dtype == np.uint8
        and q.scales.dtype == np.float32
    )
    assert_close(q.scales, np.abs(w).max(axis=1) / 448.0, rtol=1e-6, atol=0.0)
    d = dequantize(q)
    big = np.abs(w) >= q.scales[:, None] * 2.0**-6
    assert (np.abs(d - w)[big] <= np.abs(w)[big] * 2.0**-4 * (1 + 1e-6)).all()
    assert_close(np.abs(d).max(axis=1), np.abs(w).max(axis=1), rtol=1e-6, atol=0.0)


def test_mxfp4_blocks():
    # WHY: MXFP4 (M09.4's encoding) stores one power-of-two scale X per 32
    #      values and 4-bit E2M1 elements: every element is within 2X of its
    #      value (the gap from 4X to the saturated 6X), and the whole matrix
    #      costs 4.25 bits per weight.
    # KIND: property
    # CATCHES: s13, m02
    # CHAPTER: L8.5 section 2
    g = PCG32(SEED, 35)
    w = weights(g, (4, 64), 1.0)
    q = quantize_mx(w)
    assert isinstance(q, MXTensor) and q.scales.shape == (4, 2) and q.block == 32
    X = np.repeat(2.0 ** (q.scales.astype(np.float64) - 127), 32, axis=1)
    assert (np.abs(dequantize(q) - w) <= 2 * X).all()
    assert nbytes(q) == 4 * 64 // 2 + 8


def test_kv_fp8_follows_format_v2():
    # WHY: KV format v2 (formats/kv-block.md, craft.13) stores each head's
    #      slab as E4M3 codes with one float32 scale float32(amax) / 448; a
    #      head of zeros gets scale 1.0. This function is the Python oracle
    #      the craft.13 migration checks its C writer against.
    # KIND: unit
    # CATCHES: s14, s15
    # CHAPTER: L8.5 section 4
    g = PCG32(SEED, 36)
    x = weights(g, (3, 5, 8), 4.0)
    x[1] = 0.0
    kv = quantize_kv_fp8(x)
    assert (
        isinstance(kv, KVQuant)
        and kv.codes.shape == (3, 5, 8)
        and kv.scales.dtype == np.float32
    )
    amax = np.abs(x).reshape(3, -1).max(axis=1)
    assert kv.scales[0] == np.float32(amax[0]) / np.float32(448.0)
    assert kv.scales[1] == 1.0 and (kv.codes[1] == 0).all()
    d = dequantize(kv)
    big = np.abs(x) >= kv.scales[:, None, None] * 2.0**-6
    assert (np.abs(d - x)[big] <= np.abs(x)[big] * 2.0**-4 * (1 + 1e-6)).all()
    with pytest.raises(ValueError):
        quantize_kv_fp8(np.ones((2, 2), np.float32))


class TinyLM(Module):
    """embed -> up -> down -> lm_head, with L0.4 Linears."""

    def __init__(self, g: PCG32, d=64, h=128, v=40):
        super().__init__()
        self.up = Linear(d, h, bias=True)
        self.down = Linear(h, d, bias=False)
        self.lm_head = Linear(d, v, bias=False)
        self.load_state_dict(
            {
                "up.weight": weights(g, (h, d), 0.5),
                "up.bias": weights(g, (h,), 0.1),
                "down.weight": weights(g, (d, h), 0.5),
                "lm_head.weight": weights(g, (v, d), 0.5),
            }
        )

    def forward(self, x):
        return self.lm_head(self.down(self.up(x)))


def test_quant_linear_is_the_dequantized_matmul():
    # WHY: QuantLinear must compute exactly what a Linear with the
    #      dequantized weight computes: x @ W^T + b on the last axis, for a
    #      batch of rows. The q4 runner in Rust (L10.1) is checked against
    #      this function's output.
    # KIND: differential
    # CATCHES: s16, s17
    # CHAPTER: L8.5 section 4
    g = PCG32(SEED, 37)
    w, b = weights(g, (24, 64)), weights(g, (24,), 0.2)
    x = weights(g, (3, 5, 64))
    for q in (
        quantize_int4_group(w, 32),
        Q8Tensor(*quantize_int8_per_channel(w)),
        quantize_fp8_per_channel(w),
    ):
        lin = QuantLinear(q, bias=b)
        assert (lin.in_f, lin.out_f) == (64, 24)
        y = lin(Tensor(x))
        assert isinstance(y, Tensor) and y.shape == (3, 5, 24)
        want = x.astype(np.float64) @ dequantize(q).astype(np.float64).T + b
        assert_close(y.data, want, rtol=1e-5, atol=1e-5)
        assert list(lin.named_parameters()) == []


@pytest.mark.parametrize(
    "scheme,budget",
    [("int8", 0.02), ("q4_g32", 0.2), ("fp8_e4m3", 0.08), ("mxfp4", 0.35)],
)
def test_quantize_model_within_budget(scheme, budget):
    # WHY: quantize_model swaps every Linear but lm_head for a QuantLinear
    #      and leaves lm_head's parameters alone; the model's logits stay
    #      within each scheme's relative error budget (Frobenius norm of the
    #      difference over the logits'), the stand-in for the MS-L8
    #      perplexity budgets.
    # KIND: property
    # CATCHES: s16, s18, s19
    # CHAPTER: L8.5 section 2
    g = PCG32(SEED, 38)
    m = TinyLM(g)
    x = Tensor(weights(g, (8, 64)))
    ref = m(x).data.astype(np.float64)
    out = quantize_model(m, scheme)
    assert out is m
    assert isinstance(m.up, QuantLinear) and isinstance(m.down, QuantLinear)
    assert isinstance(m.lm_head, Linear) and not isinstance(m.lm_head, QuantLinear)
    assert [n for n, _ in m.named_parameters()] == ["lm_head.weight"]
    got = m(x).data.astype(np.float64)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert 0 < rel < budget, (scheme, rel)
    with pytest.raises(ValueError):
        quantize_model(TinyLM(g), "int3")


def test_export_q4_names_and_metadata():
    # WHY: the `*.q4.safetensors` file the Rust runner (L10.1) loads: each
    #      quantized weight becomes <name>.qweight (uint8) and <name>.scales
    #      (float16) with formats/safetensors.md's metadata, the bias keeps its
    #      name, and unquantized parameters (lm_head) are written as they are.
    # KIND: unit
    # CATCHES: s20
    # CHAPTER: L8.5 section 4
    g = PCG32(SEED, 39)
    m = quantize_model(TinyLM(g), "q4_g32")
    tensors, meta = export_q4(m)
    assert meta == {"format": "tinyllm", "quant": "int4-g32-sym"}
    assert sorted(tensors) == sorted(
        [
            "up.weight.qweight",
            "up.weight.scales",
            "up.bias",
            "down.weight.qweight",
            "down.weight.scales",
            "lm_head.weight",
        ]
    )
    assert tensors["up.weight.qweight"].dtype == np.uint8 and tensors[
        "up.weight.qweight"
    ].shape == (128, 32)
    assert tensors["down.weight.scales"].dtype == np.float16 and tensors[
        "down.weight.scales"
    ].shape == (64, 4)
    assert nbytes(m.up.q) == 128 * 32 + 128 * 2 * 2
    with pytest.raises(ValueError):
        export_q4(TinyLM(g))


def test_output_error_stays_within_the_budget():
    # WHY: M09.3's budget, applied to a quantized layer: the output of
    #      QuantLinear differs from the exact x @ W^T by at most
    #      |W - W'| @ |x| (the quantization step) plus gamma_k of the f32
    #      dot products. Every element of every scheme stays inside it, and
    #      the budget is not vacuous: the worst element uses more than 1%
    #      of it. Rounding alone (dA = 0) would not cover the error.
    # KIND: property
    # CATCHES: s21
    # CHAPTER: L8.5 section 2
    g = PCG32(SEED, 40)
    w = weights(g, (24, 64))
    x = weights(g, (5, 64))
    exact = x.astype(np.float64) @ w.astype(np.float64).T
    for q in (
        Q8Tensor(*quantize_int8_per_channel(w)),
        quantize_int4_group(w, 32),
        quantize_fp8_per_channel(w),
        quantize_mx(w),
    ):
        y = QuantLinear(q)(Tensor(x)).data.astype(np.float64)
        bound = output_error_bound(w, q, x, "f32")
        assert bound.shape == (5, 24) and bound.dtype == np.float64
        r = float(np.max(np.abs(y - exact) / bound))
        assert 0.01 < r <= 1.0, (type(q).__name__, r)
    with pytest.raises(ValueError):
        output_error_bound(w, quantize_int4_group(w, 32), x[:, :32])


class TokenLM(Module):
    """ids [1, T] -> logits [1, T, V]: a fixed embedding row per token, then
    TinyLM's Linears (a context-free LM, enough for perplexity)."""

    def __init__(self, g: PCG32, d=64, v=40):
        super().__init__()
        self.emb = weights(g, (v, d))
        self.body = TinyLM(g, d=d, v=v)

    def forward(self, ids):
        return self.body(Tensor(self.emb[np.asarray(ids)]))


def test_quant_ppl_reports_the_cost_and_keeps_the_model():
    # WHY: MS-L8's perplexity budgets are read from L6.7's evaluator: the
    #      strided perplexity of the float model, then of a quantized copy.
    #      int8 costs less than q4, both cost something, and the caller's
    #      model is still the float one afterwards (quantize_model works in
    #      place, so quant_ppl must quantize a copy).
    # KIND: property
    # CATCHES: s22
    # CHAPTER: L8.5 section 2
    g = PCG32(SEED, 41)
    m = TokenLM(g)
    ids = np.array([g.below(40) for _ in range(48)], dtype=np.int64)
    r8 = quant_ppl(m, "int8", ids, ctx_len=16, stride=8)
    r4 = quant_ppl(m, "q4_g32", ids, ctx_len=16, stride=8)
    assert set(r8) == {"ppl", "ppl_quant", "delta", "ratio"}
    assert r8["ppl"] == r4["ppl"] > 1.0
    assert_close(r8["delta"], r8["ppl_quant"] - r8["ppl"], rtol=1e-12, atol=0)
    assert_close(r8["ratio"], r8["ppl_quant"] / r8["ppl"], rtol=1e-12, atol=0)
    assert 0 < abs(r8["ratio"] - 1) < abs(r4["ratio"] - 1) < 0.2
    assert isinstance(m.body.up, Linear) and not isinstance(m.body.up, QuantLinear)
