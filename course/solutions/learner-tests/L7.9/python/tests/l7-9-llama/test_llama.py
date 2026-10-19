"""My tests for L7.9 (rung R5). Oracles: the course's golden logits of five
tiny HF checkpoints (TINYLLM_FIXTURES), the full forward as the oracle of
the cached one, M05.1's counts, and a fake Hub on 127.0.0.1 for the
downloader. They import only the contract."""

import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from tinyllm.accounting import param_count
from tinyllm.io.hf import hf_download
from tinyllm.modern.gqa import ConcatKVCache
from tinyllm.modern.llama import LlamaConfig, LlamaForCausalLM
from tinyllm.modern.mla import ConcatLatentCache
from tinyllm.modern.window import SinkWindowCache

FIX = Path(os.environ["TINYLLM_FIXTURES"]) / "L7.9"
NAMES = [
    "tiny-llama-2l",
    "tiny-mistral-swa",
    "tiny-qwen2",
    "tiny-llama3-rope",
    "tiny-yarn",
]


def cfg(**kw):
    base = dict(
        vocab_size=19,
        hidden_size=12,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=4,
        tie_word_embeddings=True,
    )
    base.update(kw)
    return LlamaConfig(**base)


def f64(m):
    for _, p in m.named_parameters():
        p.data = p.data.astype(np.float64)
    return m


@pytest.mark.parametrize("name", NAMES)
def test_golden_logits(name):
    f = np.load(FIX / "hf_logits.npz")
    m = LlamaForCausalLM.from_pretrained(str(FIX / name))
    want = f[name + ".logits"]
    got = m(f[name + ".ids"]).data
    assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max() + 1e-5


def test_counts():
    hf = json.loads((FIX / "hf_params.json").read_text())
    for name in NAMES:
        assert (
            LlamaForCausalLM.from_pretrained(str(FIX / name)).param_count() == hf[name]
        )
    smol = LlamaConfig.from_hf(str(FIX / "smollm2-135m-instruct.config.json"))
    assert smol.head_dim == 64
    assert LlamaForCausalLM(smol, init=False).param_count() == 134515008


def test_cache_decode_matches_full():
    m = LlamaForCausalLM.from_pretrained(str(FIX / "tiny-llama-2l"))
    ids = np.array([[72, 101, 108, 108, 111, 33]])
    full = m(ids).data
    c = ConcatKVCache()
    out = np.concatenate(
        [m(ids[:, :3], cache=c).data]
        + [m(ids[:, t : t + 1], cache=c).data for t in range(3, 6)],
        axis=1,
    )
    assert np.abs(out - full).max() < 1e-4


def test_variants_and_sinks():
    c = cfg(
        attention="mla",
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        num_experts=3,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        first_k_dense_replace=1,
    )
    m = f64(LlamaForCausalLM(c))
    assert [type(l.mlp).__name__ for l in m.model.layers] == ["GatedMLP", "MoE"]
    assert type(m.model.layers[0].self_attn).__name__ == "MLAttention"
    assert m.param_count() == param_count(c.model_config())["total"]
    ids = np.array([[1, 2, 3, 4]])
    full = m(ids).data
    lc = ConcatLatentCache()
    np.testing.assert_allclose(
        np.concatenate([m(ids[:, t : t + 1], cache=lc).data for t in range(4)], 1),
        full,
        atol=1e-9,
    )
    s = f64(LlamaForCausalLM(cfg(sliding_window=2, sink_tokens=1, learned_sinks=True)))
    s.model.layers[0].self_attn.sinks.data = np.array([1.0, -1.0, 0.5])
    ids = np.arange(1, 9)[None]
    full = s(ids).data
    sw = SinkWindowCache(1, 2)
    np.testing.assert_allclose(
        np.concatenate([s(ids[:, t : t + 1], cache=sw).data for t in range(8)], 1),
        full,
        atol=1e-9,
    )


def test_config_forms_and_roundtrip(tmp_path):
    v4 = {
        "model_type": "llama",
        "vocab_size": 9,
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "rope_theta": 100.0,
        "rope_scaling": {"type": "linear", "factor": 2.0},
    }
    (tmp_path / "a.json").write_text(json.dumps(v4))
    assert LlamaConfig.from_hf(str(tmp_path / "a.json")).rope_scaling == {
        "rope_type": "linear",
        "factor": 2.0,
    }
    for c in (
        cfg(sink_tokens=2, sliding_window=4),
        cfg(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=4),
    ):
        (tmp_path / "b.json").write_text(json.dumps(c.to_hf()))
        assert LlamaConfig.from_hf(str(tmp_path / "b.json")) == c
    for bad in (
        {"num_attention_heads": None},
        {"attention_bias": True},
        {"model_type": "bert"},
    ):
        d = {k: v for k, v in dict(v4, **bad).items() if v is not None}
        (tmp_path / "c.json").write_text(json.dumps(d))
        with pytest.raises(ValueError):
            LlamaConfig.from_hf(str(tmp_path / "c.json"))


def test_keys_save_and_shards(tmp_path):
    m = LlamaForCausalLM(cfg(num_hidden_layers=1))
    assert list(m.state_dict())[1:3] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
    ]
    assert list(m.state_dict())[-3:-1] == [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ]
    m.save_pretrained(str(tmp_path / "bf"), dtype="BF16")
    w = LlamaForCausalLM.from_pretrained(str(tmp_path / "bf")).model.norm.weight.data
    assert np.all(w.view(np.uint32) & 0xFFFF == 0)
    from tinyllm.io.safetensors import save_safetensors

    sd = m.state_dict()
    (tmp_path / "sh").mkdir()
    (tmp_path / "sh" / "config.json").write_text(json.dumps(m.config.to_hf()))
    keys = list(sd)
    save_safetensors(
        str(tmp_path / "sh" / "x.safetensors"), {k: sd[k] for k in keys[:3]}, {}
    )
    save_safetensors(
        str(tmp_path / "sh" / "y.safetensors"), {k: sd[k] for k in keys[3:]}, {}
    )
    wm = {k: "x.safetensors" if i < 3 else "y.safetensors" for i, k in enumerate(keys)}
    (tmp_path / "sh" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": wm})
    )
    assert (
        LlamaForCausalLM.from_pretrained(str(tmp_path / "sh")).param_count()
        == m.param_count()
    )


DATA = bytes(range(200)) * 30


class FakeHub:
    def __init__(self):
        self.log, self.ignore_range, self.cut, self.bad = [], False, None, False
        hub = self

        class Hd(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                hub.log.append((self.path, self.headers.get("Range")))
                if not self.path.startswith("/cdn"):
                    self.send_response(302)
                    self.send_header("Location", "/cdn")
                    self.send_header(
                        "X-Linked-Etag", '"%s"' % hashlib.sha256(DATA).hexdigest()
                    )
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = DATA[:-1] + b"?" if hub.bad else DATA
                r = self.headers.get("Range")
                start = 0 if (r is None or hub.ignore_range) else int(r[6:-1])
                body = data[start:]
                send = body[: hub.cut] if hub.cut else body
                hub.cut = None
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(send)
                self.close_connection = True

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Hd)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(
            target=self.srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()


@pytest.fixture
def hub():
    h = FakeHub()
    yield h
    h.srv.shutdown()
    h.srv.server_close()


def test_download_layout_and_skip(hub, tmp_path):
    d = hf_download("o/m", ["w.bin"], str(tmp_path), endpoint=hub.url)
    assert (
        Path(d) == tmp_path / "o--m" / "main"
        and (Path(d) / "w.bin").read_bytes() == DATA
    )
    hub.log.clear()
    hf_download(
        "o/m",
        ["w.bin"],
        str(tmp_path),
        endpoint=hub.url,
        sha256={"w.bin": hashlib.sha256(DATA).hexdigest()},
    )
    assert hub.log == []


def test_resume_and_ranges(hub, tmp_path):
    hub.cut = 1000
    with pytest.raises(Exception):
        hf_download("o/m", ["w.bin"], str(tmp_path), endpoint=hub.url)
    d = hf_download("o/m", ["w.bin"], str(tmp_path), endpoint=hub.url)
    assert (Path(d) / "w.bin").read_bytes() == DATA and (
        "/cdn",
        "bytes=1000-",
    ) in hub.log
    p = tmp_path / "o--m" / "main" / "v.bin.part"
    p.write_bytes(DATA[:10])
    hub.ignore_range = True
    hf_download("o/m", ["v.bin"], str(tmp_path), endpoint=hub.url)
    assert (tmp_path / "o--m" / "main" / "v.bin").read_bytes() == DATA


def test_corrupt_and_bad_names(hub, tmp_path):
    hub.bad = True
    with pytest.raises(ValueError):
        hf_download("o/m", ["w.bin"], str(tmp_path), endpoint=hub.url)
    assert not (tmp_path / "o--m" / "main" / "w.bin").exists()
    for repo, f in (("m", ["w.bin"]), ("o/m", ["../x"])):
        with pytest.raises(ValueError):
            hf_download(repo, f, str(tmp_path), endpoint=hub.url)
