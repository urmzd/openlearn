"""Course tests for L14.2: WhisperConfig, whisper_sinusoids,
WhisperAttention, WhisperEncoderLayer, WhisperEncoder
(tinyllm/speech/encoder.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L14.2), and the chapter section it comes from.

The chapter's worked example (section 3) is the sinusoid table for 4
channels: inc = ln(10000) / (2 - 1), the two frequencies are 1 and 1e-4,
and row p is [sin p, sin(1e-4 p), cos p, cos(1e-4 p)]: sines first, then
cosines. And the stem's lengths: 3000 frames stay 3000 through the first
conv (k 3, pad 1) and become (3000 + 2 - 3) // 2 + 1 = 1500 through the
second (stride 2).

Golden values in $TINYLLM_FIXTURES/L14.2/encoder_hf.npz come from
course/oracle/L14.2/whisper_tiny_hf.py: two tiny random HF
WhisperForConditionalGeneration models (transformers 5.19), their encoder
weights under course names, and their encoder outputs on the L14.1 clip
en-0 and on random inputs.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.autograd.tensor import Tensor
from tinyllm.speech.encoder import (
    WhisperAttention,
    WhisperConfig,
    WhisperEncoder,
    WhisperEncoderLayer,
    whisper_sinusoids,
)

FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L14.2"
TINY = WhisperConfig(
    n_mels=80,
    n_audio_ctx=1500,
    d=32,
    n_heads=4,
    enc_layers=2,
    dec_layers=2,
    n_text_ctx=64,
    vocab=100,
    d_ff=64,
)
SMALL = WhisperConfig(
    n_mels=128,
    n_audio_ctx=8,
    d=16,
    n_heads=2,
    enc_layers=1,
    dec_layers=1,
    n_text_ctx=16,
    vocab=50,
    d_ff=48,
)


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def golden():
    return np.load(FIX / "encoder_hf.npz")


def loaded(tag: str, cfg: WhisperConfig) -> WhisperEncoder:
    z = golden()
    enc = WhisperEncoder(cfg).eval()
    pre = f"{tag}.state."
    enc.load_state_dict({k[len(pre) :]: z[k] for k in z.files if k.startswith(pre)})
    return enc


def test_hand_example_sinusoids():
    # WHY: section 3 by hand. Whisper concatenates the sines and then the
    #      cosines, with frequencies exp(-ln(10000) i / (C/2 - 1)); L5.4's
    #      transformer table interleaves sin and cos and divides by C/2. Mixing
    #      them up loads a checkpoint fine and gives garbage.
    # KIND: unit, smoke
    # CATCHES: s01, s02, s10
    # CHAPTER: L14.2 section 3
    t = whisper_sinusoids(3, 4)
    assert t.dtype == np.float32 and t.shape == (3, 4)
    want = np.array(
        [
            [math.sin(p), math.sin(1e-4 * p), math.cos(p), math.cos(1e-4 * p)]
            for p in range(3)
        ]
    )
    assert_close(t, want.astype(np.float32), rtol=0.0, atol=1e-7)


def test_sinusoids_match_hf_tables():
    # WHY: HF checkpoints store the table they were trained with. Short
    #      tables agree to float32 rounding; at 1500 positions the angles
    #      reach 1499 rad and one ulp of float32 exp moves an entry by about
    #      3e-5, which is why L14.3's loader keeps the stored copy.
    # KIND: golden
    # CATCHES: s01, s02, s10
    # CHAPTER: L14.2 section 2.2
    z = golden()
    assert_close(whisper_sinusoids(8, 16), z["sinusoids.8x16"], rtol=0.0, atol=1e-6)
    big = whisper_sinusoids(1500, 32)
    assert_close(big, z["sinusoids.1500x32"], rtol=0.0, atol=1e-4)
    assert np.abs(big - z["sinusoids.1500x32"]).mean() < 1e-6
    for bad in ((3, 5), (3, 2), (0, 4)):
        with pytest.raises(ValueError):
            whisper_sinusoids(*bad)


def test_stem_output_length():
    # WHY: section 3: a 30 s window is 3000 frames, the stride-2 conv makes
    #      1500 positions, and the sinusoid table has exactly n_audio_ctx
    #      rows. Any other length is refused, as HF does, because the
    #      positions would not line up.
    # KIND: unit
    # CATCHES: s03, m02
    # CHAPTER: L14.2 section 3
    enc = WhisperEncoder(SMALL).eval()
    out = enc(np.zeros((1, 128, 16), np.float32))
    assert out.shape == (1, 8, 16)
    for frames in (15, 17, 14, 32):
        with pytest.raises(ValueError):
            enc(np.zeros((1, 128, frames), np.float32))
    with pytest.raises(ValueError):
        enc(np.zeros((1, 80, 16), np.float32))
    with pytest.raises(ValueError):
        enc(np.zeros((128, 16), np.float32))


def test_encoder_matches_hf_tiny_whisper():
    # WHY: the whole encoder against HF's on a real clip's features: stem,
    #      GELU, positions, pre-LN blocks, k without bias, final LayerNorm.
    #      Model parity uses the stored sinusoid table (DESIGN D48), so the
    #      only differences left are float32 summation order: 1e-5.
    # KIND: golden
    # CATCHES: s03, s04, s06, s07, s08, s09
    # CHAPTER: L14.2 section 2.3
    z = golden()
    enc = loaded("tiny", TINY)
    enc.positions = z["sinusoids.1500x32"]
    out = enc(z["tiny.mel"])
    assert out.shape == (1, 1500, 32)
    assert_close(out.numpy(), z["tiny.hidden"], rtol=0.0, atol=1e-5)


def test_encoder_matches_hf_small_batch():
    # WHY: a second shape (128 mels, 2 heads, one layer) and a batch of two:
    #      every item is encoded on its own.
    # KIND: golden
    # CATCHES: s04, s05, s06, s07
    # CHAPTER: L14.2 section 2.3
    z = golden()
    enc = loaded("small", SMALL)
    out = enc(z["small.mel"]).numpy()
    assert_close(out, z["small.hidden"], rtol=0.0, atol=1e-5)
    one = enc(z["small.mel"][1:]).numpy()
    assert_close(one[0], out[1], rtol=0.0, atol=1e-6)


def test_parameter_names_are_hf_names():
    # WHY: checkpoints load by name. Whisper's key projection has no bias,
    #      so k_proj.bias must not exist, and every other name and shape must
    #      be HF's without the "model.encoder." prefix.
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: L14.2 section 2.3
    z = golden()
    want = {
        k[len("tiny.state.") :]: z[k].shape
        for k in z.files
        if k.startswith("tiny.state.")
    }
    got = {k: v.shape for k, v in WhisperEncoder(TINY).state_dict().items()}
    assert got == want
    att = WhisperAttention(8, 2)
    assert att.k_proj.bias is None and att.q_proj.bias is not None
    assert [k for k, _ in att.named_parameters()] == [
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "v_proj.weight",
        "v_proj.bias",
        "out_proj.weight",
        "out_proj.bias",
    ]


def test_config_from_hf(tmp_path):
    # WHY: the encoder's shapes come from the checkpoint's config.json in
    #      HF's names; a decoder with other heads or ffn size, another
    #      activation, or scaled embeddings is a different architecture and
    #      is refused rather than loaded wrong.
    # KIND: unit
    # CATCHES: m01, m03
    # CHAPTER: L14.2 section 4
    assert WhisperConfig.from_hf(str(FIX / "tiny-whisper")) == TINY
    assert WhisperConfig.from_hf(str(FIX / "tiny-whisper-128" / "config.json")) == SMALL
    base = json.loads((FIX / "tiny-whisper" / "config.json").read_text())
    for change in (
        {"decoder_attention_heads": 2},
        {"decoder_ffn_dim": 32},
        {"activation_function": "relu"},
        {"scale_embedding": True},
        {"model_type": "bert"},
        {"d_model": 30},
        {"d_model": 33, "encoder_attention_heads": 3, "decoder_attention_heads": 3},
        {"encoder_layers": 0},
    ):
        (tmp_path / "config.json").write_text(json.dumps({**base, **change}))
        with pytest.raises(ValueError):
            WhisperConfig.from_hf(str(tmp_path))
    assert WhisperConfig().n_audio_ctx == 1500 and WhisperConfig().d == 384


def test_blocks_are_pre_ln_residual():
    # WHY: with every residual branch zeroed (out_proj and fc2 all zero), a
    #      pre-LN block is the identity, so the encoder reduces to
    #      layer_norm(stem + positions). A post-LN block would normalize in
    #      between and break the identity.
    # KIND: property
    # CATCHES: s06, s07, s09
    # CHAPTER: L14.2 section 2.3
    z = golden()
    enc = loaded("small", SMALL)
    for layer in enc.layers:
        for lin in (layer.self_attn.out_proj, layer.fc2):
            lin.weight.data[...] = 0.0
            lin.bias.data[...] = 0.0
    x = z["small.mel"].astype(np.float64)

    def conv(x, w, b, stride):
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1)))
        L = (xp.shape[2] - 3) // stride + 1
        cols = np.stack(
            [xp[:, :, i * stride : i * stride + 3] for i in range(L)], axis=2
        )
        return np.einsum("ncLk,ock->noL", cols, w) + b[None, :, None]

    def gelu(v):
        return 0.5 * v * (1 + np.vectorize(math.erf)(v / math.sqrt(2)))

    h = gelu(conv(x, enc.conv1.weight.data, enc.conv1.bias.data, 1))
    h = gelu(conv(h, enc.conv2.weight.data, enc.conv2.bias.data, 2))
    h = h.transpose(0, 2, 1) + whisper_sinusoids(8, 16)
    mu = h.mean(-1, keepdims=True)
    var = h.var(-1, keepdims=True)
    want = (h - mu) / np.sqrt(
        var + 1e-5
    ) * enc.layer_norm.weight.data + enc.layer_norm.bias.data
    assert_close(
        enc(z["small.mel"]).numpy(), want.astype(np.float32), rtol=0.0, atol=2e-5
    )


def test_every_frame_sees_every_frame():
    # WHY: the encoder attends without a mask: changing the LAST mel frame
    #      must change the FIRST output position (a causal mask would hide
    #      it), while one layer's block maps [N, A, d] to the same shape.
    # KIND: property
    # CATCHES: s08
    # CHAPTER: L14.2 section 2.3
    enc = loaded("small", SMALL)
    x = golden()["small.mel"][:1].copy()
    a = enc(x).numpy()
    x[0, :, -1] += 1.0
    b = enc(x).numpy()
    assert np.abs(a[0, 0] - b[0, 0]).max() > 1e-4
    layer = WhisperEncoderLayer(SMALL, rng=rng(3))
    y = layer(Tensor(np.ones((2, 8, 16), np.float32)))
    assert y.shape == (2, 8, 16)
