"""Course tests for L0.6: atomic checkpoints (tinyllm/io/checkpoint.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L0.6), and the chapter section it comes from.

The worked example of the chapter (section 3): a Linear(2, 1) trained two
AdamW steps, saved at step 7 with the generator state (0x853c49e6748fea9b,
0xda3e39cb94b95bdb). The directory holds model.safetensors,
optimizer.safetensors (tensors weight.exp_avg, weight.exp_avg_sq,
bias.exp_avg, bias.exp_avg_sq), trainer_state.json, and MANIFEST.json, and
LATEST holds "step-000007\\n".
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pytest
from _lib.pcg32 import PCG32
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.io.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    step_name,
    verify_step_dir,
)
from tinyllm.io.safetensors import load_safetensors
from tinyllm.nn.layers import Linear
from tinyllm.optim.adamw import AdamW
from tinyllm.optim.sgd import SGD

RNG_STATE = (0x853C49E6748FEA9B, 0xDA3E39CB94B95BDB)
EXTRA = {
    "tokens_seen": 64,
    "data_cursor": {"shard": 0, "offset": 33},
    "lr": 0.01,
    "config_sha256": "9" * 64,
    "git_sha": "unknown",
}


class Rng:
    """The frozen PCG32 behind the init API Linear uses (uniform, normal)."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def next_u32(self) -> int:
        return self.g.next_u32()

    def uniform(self) -> float:
        return self.g.uniform()

    def normal(self) -> float:
        return self.g.normal()


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def model_and_opt(s: int, opt_cls=AdamW):
    m = Linear(2, 1, rng=Rng(s))
    opt = (
        opt_cls(list(m.parameters()), lr=0.01)
        if opt_cls is AdamW
        else opt_cls(list(m.parameters()), lr=0.01, momentum=0.9)
    )
    return m, opt


def train(m, opt, steps: int, start: int = 0) -> None:
    for k in range(start, start + steps):
        x = Tensor(np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float32) * (1 + k % 3))
        y = np.array([[1.0], [0.0]], dtype=np.float32)
        opt.zero_grad()
        loss = F.mean((m(x) - y) ** 2)
        loss.backward()
        opt.step()


def files(d) -> list[str]:
    return sorted(p.name for p in d.iterdir())


# --- the layout -------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    # WHY: the chapter's worked example: save at step 7, the directory and
    #      LATEST look as formats/checkpoint.md says, and loading gives back
    #      the same parameters, optimizer moments and step count, generator
    #      state, and trainer fields, bit for bit.
    # KIND: unit
    # CATCHES: s21
    # CHAPTER: L0.6 section 3, Worked example by hand
    m, opt = model_and_opt(seed())
    train(m, opt, 2)
    path = save_checkpoint(str(tmp_path), m, opt, 7, RNG_STATE, dict(EXTRA))
    assert path == str(tmp_path / "step-000007")
    assert files(tmp_path) == ["LATEST", "step-000007"]
    assert (tmp_path / "LATEST").read_text() == "step-000007\n"
    assert files(tmp_path / "step-000007") == [
        "MANIFEST.json",
        "model.safetensors",
        "optimizer.safetensors",
        "trainer_state.json",
    ]
    c = load_checkpoint(str(tmp_path))
    assert c.step == 7 and c.rng_state == RNG_STATE and c.path == path
    for name, a in m.state_dict().items():
        assert c.model[name].dtype == np.float32 and (c.model[name] == a).all(), name
    sd = opt.state_dict()
    assert c.opt["step"] == sd["step"] == 2
    for k in ("exp_avg", "exp_avg_sq"):
        for a, b in zip(c.opt[k], sd[k]):
            assert (a == b).all() and a.dtype == np.float32
    assert {k: c.extra[k] for k in EXTRA} == EXTRA


def test_layout_matches_the_formats(tmp_path):
    # WHY: other tools read these files: the durable worker (dur.09) resumes
    #      from trainer_state.json, release (dur.12) verifies MANIFEST.json.
    #      trainer_state.json has exactly the schema's keys with the generator
    #      as 16 hex digits; MANIFEST.json lists every other file, sorted, with
    #      its true sha256 and size; optimizer tensors are named
    #      "<parameter>.<state key>" with the format's metadata.
    # KIND: conformance
    # CATCHES: s13, s19, s20
    # CHAPTER: L0.6 section 2, Principles (what a checkpoint holds)
    m, opt = model_and_opt(seed())
    train(m, opt, 1)
    d = tmp_path / step_name(7)
    save_checkpoint(str(tmp_path), m, opt, 7, RNG_STATE, dict(EXTRA))
    st = json.loads((d / "trainer_state.json").read_text())
    assert set(st) == {
        "step",
        "tokens_seen",
        "data_cursor",
        "rng",
        "lr",
        "config_sha256",
        "git_sha",
    }
    assert (
        st["step"] == 7
        and st["tokens_seen"] == 64
        and st["data_cursor"] == {"shard": 0, "offset": 33}
    )
    assert st["rng"] == {"pcg_state": "853c49e6748fea9b", "pcg_inc": "da3e39cb94b95bdb"}
    man = json.loads((d / "MANIFEST.json").read_text())
    assert man["kind"] == "checkpoint"
    names = [r["name"] for r in man["files"]]
    assert (
        names
        == sorted(names)
        == ["model.safetensors", "optimizer.safetensors", "trainer_state.json"]
    )
    for r in man["files"]:
        b = (d / r["name"]).read_bytes()
        assert r["sha256"] == hashlib.sha256(b).hexdigest() and r["bytes"] == len(b), r
    arrays, meta = load_safetensors(str(d / "optimizer.safetensors"))
    assert sorted(arrays) == [
        "bias.exp_avg",
        "bias.exp_avg_sq",
        "weight.exp_avg",
        "weight.exp_avg_sq",
    ]
    assert (
        meta["format"] == "tinyllm"
        and meta["optimizer"] == "adamw"
        and meta["step"] == "7"
    )
    assert verify_step_dir(str(d)) == []
    (d / "notes.txt").write_text("not in the manifest")
    assert verify_step_dir(str(d)) != [], (
        "a file the manifest does not list makes the directory incomplete"
    )


def test_config_and_defaults(tmp_path):
    # WHY: extra may carry the run's config (written as config.json and
    #      listed in the manifest); fields it leaves out take the format's
    #      defaults, so a minimal caller still writes a schema-valid
    #      trainer_state.json: config_sha256 is the sha256 of config.json's
    #      bytes and lr comes from the optimizer.
    # KIND: unit
    # CATCHES: s23
    # CHAPTER: L0.6 section 4, The interface
    m, opt = model_and_opt(seed())
    cfg = {"tl_arch": "bigram", "vocab_size": 256}
    save_checkpoint(str(tmp_path), m, opt, 3, RNG_STATE, {"config": cfg})
    d = tmp_path / "step-000003"
    st = json.loads((d / "trainer_state.json").read_text())
    assert (
        st["config_sha256"]
        == hashlib.sha256((d / "config.json").read_bytes()).hexdigest()
    )
    assert st["lr"] == 0.01 and st["tokens_seen"] == 0 and st["git_sha"] == "unknown"
    assert st["data_cursor"] == {"shard": 0, "offset": 0}
    assert "config.json" in [
        r["name"] for r in json.loads((d / "MANIFEST.json").read_text())["files"]
    ]
    assert load_checkpoint(str(tmp_path)).extra["config"] == cfg


def test_sgd_momentum_names(tmp_path):
    # WHY: the optimizer state is stored by parameter name, whatever its
    #      shape in memory: SGD's {"state": {i: {"momentum_buffer"}}} becomes
    #      "<name>.momentum_buffer", and loads back into SGD.load_state_dict.
    # KIND: unit
    # CATCHES: s20
    # CHAPTER: L0.6 section 2, Principles (what a checkpoint holds)
    m, opt = model_and_opt(seed(), SGD)
    train(m, opt, 2)
    save_checkpoint(str(tmp_path), m, opt, 2, RNG_STATE, {})
    arrays, meta = load_safetensors(
        str(tmp_path / "step-000002" / "optimizer.safetensors")
    )
    assert (
        sorted(arrays) == ["bias.momentum_buffer", "weight.momentum_buffer"]
        and meta["optimizer"] == "sgd"
    )
    m2, opt2 = model_and_opt(seed() + 1, SGD)
    opt2.load_state_dict(load_checkpoint(str(tmp_path)).opt)
    a, b = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert all((a[i]["momentum_buffer"] == b[i]["momentum_buffer"]).all() for i in a)


def test_resume_is_bitwise(tmp_path):
    # WHY: resuming is bitwise or it is not resuming: 3 steps, save, a fresh
    #      model and optimizer loaded from the checkpoint, 3 more steps gives
    #      exactly the weights and moments of 6 uninterrupted steps. MS-L0's
    #      kill-and-resume step checks the same through your CLI.
    # KIND: property
    # CATCHES: s21
    # CHAPTER: L0.6 section 2, Principles (resume is bitwise)
    a, oa = model_and_opt(seed())
    train(a, oa, 6)
    b, ob = model_and_opt(seed())
    train(b, ob, 3)
    save_checkpoint(str(tmp_path), b, ob, 3, RNG_STATE, {})
    c, oc = model_and_opt(seed() + 50)  # different init: everything must come from disk
    got = load_checkpoint(str(tmp_path))
    c.load_state_dict(got.model)
    oc.load_state_dict(got.opt)
    train(c, oc, 3, start=got.step)
    for (n, p), (_, q) in zip(a.named_parameters(), c.named_parameters()):
        assert (p.data == q.data).all(), n
    for x, y in zip(oa.state_dict()["exp_avg_sq"], oc.state_dict()["exp_avg_sq"]):
        assert (x == y).all()


# --- crashes and corruption ---------------------------------------------------


class Crash(BaseException):
    """What a SIGKILL does to a save in progress: nothing after it runs."""


def test_crash_at_every_write_keeps_a_valid_checkpoint(tmp_path, monkeypatch):
    # WHY: the promise of formats/checkpoint.md: a crash at any instant leaves
    #      either the previous checkpoint or the new one, never a mix. The
    #      test crashes the save of step 2 at its k-th fsync or rename, for
    #      every k. After each crash, every step-<n> directory must be
    #      complete, LATEST must name one of them, and load_checkpoint must
    #      return step 1 or step 2. It also counts the fsyncs: without them,
    #      "the rename happened" says nothing about the bytes being on disk.
    # KIND: fault
    # CATCHES: s14, s15, s16
    # CHAPTER: L0.6 section 5, Pitfalls, item 6
    m, opt = model_and_opt(seed())
    train(m, opt, 1)
    save_checkpoint(str(tmp_path), m, opt, 1, RNG_STATE, {})
    w1 = m.state_dict()["weight"].copy()
    train(m, opt, 1)
    real = {"fsync": os.fsync, "rename": os.rename, "replace": os.replace}
    calls: list[str] = []

    def counting(name):
        def f(*a, **k):
            calls.append(name)
            return real[name](*a, **k)

        return f

    for name in real:
        monkeypatch.setattr(os, name, counting(name))
    save_checkpoint(str(tmp_path / "count"), m, opt, 2, RNG_STATE, {})
    total = len(calls)
    assert calls.count("fsync") >= 8, (
        f"only {calls.count('fsync')} fsync calls; formats/checkpoint.md syncs each of the 4 files, "
        "the .tmp directory, ckpt/ after the rename, LATEST.tmp, and ckpt/ again"
    )
    seen = set()
    for k in range(total):
        hits = [0]

        def crashing(name):
            def f(*a, **kw):
                hits[0] += 1
                if hits[0] == k + 1:
                    raise Crash(f"crash at call {k + 1} ({name})")
                return real[name](*a, **kw)

            return f

        for name in real:
            monkeypatch.setattr(os, name, crashing(name))
        with pytest.raises(Crash):
            save_checkpoint(str(tmp_path), m, opt, 2, RNG_STATE, {})
        for name in real:
            monkeypatch.setattr(os, name, real[name])
        # Whatever instant the crash hit: every step-<n> directory is
        # complete, and LATEST names one of them.
        steps = sorted(
            p for p in tmp_path.iterdir() if re.fullmatch(r"step-\d+", p.name)
        )
        for p in steps:
            assert verify_step_dir(str(p)) == [], (k, p.name, verify_step_dir(str(p)))
        named = (tmp_path / "LATEST").read_text().strip()
        assert named in {p.name for p in steps}, (k, named)
        c = load_checkpoint(str(tmp_path))
        assert c.step in (1, 2), c.step
        assert verify_step_dir(c.path) == []
        if c.step == 1:
            assert (c.model["weight"] == w1).all()
        seen.add(c.step)
        # Undo a completed step 2 so the next crash point starts from step 1 again.
        (tmp_path / "LATEST").write_text("step-000001\n")
    assert 1 in seen


def test_load_skips_incomplete_and_corrupt(tmp_path):
    # WHY: a directory whose manifest is missing or does not verify is
    #      skipped for the newest older one that does: a flipped byte in the
    #      newest model.safetensors, then a missing MANIFEST.json, then a
    #      LATEST naming a directory that is gone. Asking for a broken step
    #      by number is an error; nothing complete is FileNotFoundError.
    # KIND: fault
    # CATCHES: s17
    # CHAPTER: L0.6 section 5, Pitfalls, item 7
    m, opt = model_and_opt(seed())
    for s in (1, 2, 3):
        train(m, opt, 1)
        save_checkpoint(str(tmp_path), m, opt, s, RNG_STATE, {})
    f = tmp_path / "step-000003" / "model.safetensors"
    b = bytearray(f.read_bytes())
    b[-1] ^= 0x01
    f.write_bytes(bytes(b))
    assert verify_step_dir(str(tmp_path / "step-000003")) != []
    assert load_checkpoint(str(tmp_path)).step == 2
    with pytest.raises(ValueError):
        load_checkpoint(str(tmp_path), step=3)
    (tmp_path / "step-000002" / "MANIFEST.json").unlink()
    assert load_checkpoint(str(tmp_path)).step == 1
    assert load_checkpoint(str(tmp_path), step=1).step == 1
    (tmp_path / "step-000002" / "stray.txt").write_text("x")
    (tmp_path / "LATEST").write_text("step-000009\n")
    assert load_checkpoint(str(tmp_path)).step == 1
    (tmp_path / "step-000001" / "trainer_state.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path))


def test_newer_than_latest_is_ignored(tmp_path):
    # WHY: LATEST is written last, so a step directory newer than the one it
    #      names was never declared done (the run died between the two
    #      renames). Resume from what LATEST names, exactly as the format
    #      says; with no LATEST at all, the newest complete directory.
    # KIND: boundary
    # CATCHES: s18
    # CHAPTER: L0.6 section 2, Principles (writing atomically)
    m, opt = model_and_opt(seed())
    save_checkpoint(str(tmp_path), m, opt, 1, RNG_STATE, {})
    save_checkpoint(str(tmp_path), m, opt, 2, RNG_STATE, {})
    (tmp_path / "LATEST").write_text("step-000001\n")
    assert load_checkpoint(str(tmp_path)).step == 1
    (tmp_path / "LATEST").unlink()
    assert load_checkpoint(str(tmp_path)).step == 2


def test_keep_and_stale_tmp(tmp_path):
    # WHY: housekeeping runs only after the new checkpoint is durable: keep=2
    #      leaves the two newest step directories, and a stale .tmp left by
    #      an earlier crash is removed. Deleting first would leave nothing to
    #      resume from if the save then failed.
    # KIND: unit
    # CATCHES: m03
    # CHAPTER: L0.6 section 4, The interface
    m, opt = model_and_opt(seed())
    (tmp_path / "step-000000.tmp").mkdir()
    (tmp_path / "step-000000.tmp" / "half.bin").write_bytes(b"x")
    for s in (1, 2, 3):
        save_checkpoint(str(tmp_path), m, opt, s, RNG_STATE, {}, keep=2)
    assert files(tmp_path) == ["LATEST", "step-000002", "step-000003"]
    with pytest.raises(ValueError):
        save_checkpoint(str(tmp_path), m, opt, 4, RNG_STATE, {}, keep=0)


def test_rejects_bad_state(tmp_path):
    # WHY: trainer_state.json must stay schema-valid, so bad input fails at
    #      save time, not at resume time: an unknown extra key, an even PCG
    #      increment (not a PCG stream), a negative cursor, a 39-digit git
    #      sha, a negative step.
    # KIND: boundary
    # CATCHES: s22, m04
    # CHAPTER: L0.6 section 4, The interface
    m, opt = model_and_opt(seed())
    bad = [
        (RNG_STATE, {"loss": 1.0}),
        ((5, 6), {}),
        (RNG_STATE, {"data_cursor": {"shard": 0, "offset": -1}}),
        (RNG_STATE, {"git_sha": "a" * 39}),
    ]
    for rng_state, extra in bad:
        with pytest.raises(ValueError):
            save_checkpoint(str(tmp_path), m, opt, 1, rng_state, extra)
    with pytest.raises(ValueError):
        save_checkpoint(str(tmp_path), m, opt, -1, RNG_STATE, {})
    assert (
        re.fullmatch(r"step-\d{6}", step_name(5))
        and step_name(1234567) == "step-1234567"
    )
    # Checked before anything is written: no directory, no LATEST.
    assert (
        not (tmp_path / "LATEST").exists() and not (tmp_path / "step-000001").exists()
    )
