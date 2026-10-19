"""My tests for L8.5 (rung R4: properties). They import only the contract."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tinyllm.autograd.tensor import Tensor
from tinyllm.infer.quant import (
    Q8Tensor,
    QuantLinear,
    dequantize,
    export_q4,
    nbytes,
    pack_int4,
    quantize_fp8_per_channel,
    quantize_int4_group,
    quantize_int8_per_channel,
    quantize_kv_fp8,
    quantize_model,
    quantize_mx,
    unpack_int4,
)
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module

# Zero, or a magnitude where float16 scales do not underflow (a group whose
# amax / 7 underflows float16 stores scale 0 by the rule, outside the bound).
finite = st.one_of(
    st.just(0.0),
    st.floats(0.0009765625, 100, width=32),
    st.floats(-100, -0.0009765625, width=32),
)


def test_by_hand():
    q = quantize_int4_group(np.array([[1.75, -0.6, 0.1, 0.3]], np.float32), 4)
    assert q.packed.tolist() == [[0xE7, 0x10]]
    q8, s = quantize_int8_per_channel(np.array([[127, -3.5, 2.5, 0.4]], np.float32))
    assert q8.tolist() == [[127, -4, 2, 0]] and s.tolist() == [1.0]


def test_codes_use_the_stored_float16_scale():
    # amax 1: the stored scale is float16(1/7); 3.5 stored scales is a tie (to 4)
    s16 = np.float32(np.float16(np.float32(1.0) / np.float32(7.0)))
    row = np.zeros((1, 8), np.float32)
    row[0, 0], row[0, 1] = 1.0, np.float32(3.5) * s16
    assert unpack_int4(quantize_int4_group(row, 8).packed)[0, :2].tolist() == [7, 4]


def test_pack_unpack_identity_and_low_nibble():
    v = np.array([[a, b] for a in range(-8, 8) for b in range(-8, 8)], np.int8).reshape(
        1, -1
    )
    assert (unpack_int4(pack_int4(v)) == v).all()
    assert pack_int4(np.array([[1, 2]])).tolist() == [[0x21]]


@settings(max_examples=60)
@given(arrays(np.float32, (3, 32), elements=finite))
def test_half_step_bound_with_the_stored_scale(w):
    q = quantize_int4_group(w, 16)
    s = np.repeat(q.scales.astype(np.float32), 16, axis=1)
    d = dequantize(q)
    assert (d == unpack_int4(q.packed) * s).all()
    assert (np.abs(w - d) <= s / 2 * (1 + 1e-6) + 1e-30).all()
    safe = np.where(s > 0, s, 1.0)
    assert (
        unpack_int4(q.packed) == np.where(s > 0, np.clip(np.rint(w / safe), -8, 7), 0)
    ).all()


@settings(max_examples=40)
@given(arrays(np.float32, (4, 8), elements=finite))
def test_int8_bound_and_range(w):
    q, s = quantize_int8_per_channel(w)
    assert q.min() >= -127
    nonzero = np.abs(w).max(axis=1) > 0
    assert (np.abs(q).max(axis=1)[nonzero] == 127).all()  # one scale per row
    assert (np.abs(w - dequantize((q, s))) <= s[:, None] / 2 * (1 + 1e-6) + 1e-30).all()


def test_shape_and_overflow_rules():
    with pytest.raises(ValueError):
        quantize_int4_group(np.ones((1, 6), np.float32), 3)
    with pytest.raises(ValueError):
        quantize_int4_group(np.full((1, 4), 1e6, np.float32), 4)


def test_fp8_mx_and_kv():
    w = np.linspace(-3, 3, 64, dtype=np.float32).reshape(2, 32)
    q = quantize_fp8_per_channel(w)
    assert (np.abs(q.scales - np.abs(w).max(axis=1) / 448) <= 1e-6 * q.scales).all()
    assert np.abs(dequantize(q) - w).max() <= 3 * 2.0**-4
    mx = quantize_mx(w)
    assert mx.scales.shape == (2, 1) and nbytes(mx) == 32 + 2
    x = np.zeros((2, 3, 4), np.float32)
    x[0] = 2.0
    kv = quantize_kv_fp8(x)
    assert kv.scales.tolist() == [np.float32(2.0) / np.float32(448.0), 1.0]
    assert (dequantize(kv) == x).all()


class Net(Module):
    def __init__(self):
        super().__init__()
        self.a = Linear(32, 16)
        self.lm_head = Linear(16, 8, bias=False)

    def forward(self, x):
        return self.lm_head(self.a(x))


def test_quant_linear_and_model():
    w = np.linspace(-1, 1, 16 * 32, dtype=np.float32).reshape(16, 32)
    b = np.arange(16, dtype=np.float32)
    x = np.ones((2, 32), np.float32)
    for q in (quantize_int4_group(w, 32), Q8Tensor(*quantize_int8_per_channel(w))):
        y = QuantLinear(q, b)(Tensor(x)).data
        assert np.abs(y - (x @ dequantize(q).T + b)).max() < 1e-4
    net = quantize_model(Net(), "q4_g32")
    assert isinstance(net.a, QuantLinear) and not isinstance(net.lm_head, QuantLinear)
    assert net.a.bias is not None and net.a.bias.shape == (16,)
    tensors, meta = export_q4(net)
    assert (
        meta["quant"] == "int4-g32-sym"
        and "a.weight.qweight" in tensors
        and "a.bias" in tensors
    )
