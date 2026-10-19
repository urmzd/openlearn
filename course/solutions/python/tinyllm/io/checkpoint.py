"""Atomic checkpoints (L0.6): formats/checkpoint.md.

A checkpoint is one step directory, `<dir>/step-<nnnnnn>/`, plus `<dir>/LATEST`
naming the newest complete one. Every file is written into
`step-<nnnnnn>.tmp/` and fsynced, MANIFEST.json (the sha256 and size of every
other file) goes last, and only then is the directory renamed into place and
LATEST replaced by a rename. A crash at any instant leaves the previous
checkpoint as LATEST names it, or the new one complete: never a mix. Loading
checks every file against the manifest and falls back to the newest older
step that verifies.

Contract: contracts/py/tinyllm/io/checkpoint.pyi.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from tinyllm.io.safetensors import load_safetensors, save_safetensors

_STEP_DIR = re.compile(r"^step-(\d{6,})$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT = re.compile(r"^([0-9a-f]{40}|unknown)$")
_EXTRA = {"tokens_seen", "data_cursor", "lr", "config_sha256", "git_sha", "loss_scale", "config"}
REQUIRED_FILES = ("model.safetensors", "optimizer.safetensors", "trainer_state.json")


@dataclass
class Checkpoint:
    path: str  # the step directory that was loaded
    step: int
    model: dict[str, NDArray]  # for Module.load_state_dict
    opt: dict  # for the optimizer's load_state_dict
    rng_state: tuple[int, int]  # (state, inc) of the shuffle PCG32
    extra: dict = field(default_factory=dict)


def step_name(step: int) -> str:
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    return f"step-{step:06d}"


def _fsync_path(path: Path) -> None:
    """fsync a file or a directory by path."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path: Path, data: bytes) -> None:
    """Write bytes and fsync them before returning."""
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rng_json(rng_state: Any) -> dict[str, str]:
    """(state, inc) as trainer_state.json's 16-hex-digit strings."""
    if isinstance(rng_state, dict):
        out = {str(k): str(v) for k, v in rng_state.items()}
    else:
        s, inc = rng_state
        s, inc = int(s), int(inc)
        if not (0 <= s < 1 << 64 and 0 <= inc < 1 << 64) or inc % 2 == 0:
            raise ValueError(f"rng_state must be (state, odd inc) below 2^64, got {rng_state!r}")
        out = {"pcg_state": f"{s:016x}", "pcg_inc": f"{inc:016x}"}
    if "pcg_state" not in out or "pcg_inc" not in out or not all(_HEX16.match(v) for v in out.values()):
        raise ValueError(f"rng state needs pcg_state and pcg_inc as 16 hex digits, got {out!r}")
    return out


def _flatten_opt(sd: Any, names: list[str]) -> tuple[Any, dict[str, NDArray]]:
    """The optimizer state with every array replaced by {"__tensor__": key},
    and the arrays by key. An array reached through parameter index i (a list
    position or an integer dict key) is keyed "<name of parameter i>.<the
    last string key on its path>": AdamW's exp_avg[i] is
    "<name>.exp_avg", SGD's state[i]["momentum_buffer"] is
    "<name>.momentum_buffer" (formats/checkpoint.md)."""
    arrays: dict[str, NDArray] = {}

    def key_for(path: list) -> str:
        idx = [p for p in path if isinstance(p, int)]
        strs = [p for p in path if isinstance(p, str)]
        if len(idx) == 1 and 0 <= idx[0] < len(names) and strs:
            return f"{names[idx[0]]}.{strs[-1]}"
        return "opt." + ".".join(str(p) for p in path)

    def walk(x: Any, path: list) -> Any:
        if isinstance(x, np.ndarray):
            k = key_for(path)
            if k in arrays:
                raise ValueError(f"optimizer state: two arrays map to {k!r}")
            if x.dtype != np.float32:
                raise ValueError(f"optimizer state {k!r}: dtype {x.dtype}; checkpoints hold F32")
            arrays[k] = x
            return {"__tensor__": k}
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                kk = int(k) if isinstance(k, (int, np.integer)) else str(k)
                out[str(k)] = walk(v, path + [kk])
            return out
        if isinstance(x, (list, tuple)):
            return [walk(v, path + [i]) for i, v in enumerate(x)]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        raise ValueError(f"optimizer state holds a {type(x).__name__} at {path}")

    return walk(sd, []), arrays


def _unflatten_opt(skel: Any, arrays: dict[str, NDArray]) -> Any:
    if isinstance(skel, dict):
        if set(skel) == {"__tensor__"}:
            k = skel["__tensor__"]
            if k not in arrays:
                raise ValueError(f"optimizer.safetensors has no tensor {k!r}")
            return arrays[k]
        return {k: _unflatten_opt(v, arrays) for k, v in skel.items()}
    if isinstance(skel, list):
        return [_unflatten_opt(v, arrays) for v in skel]
    return skel


def _trainer_state(step: int, rng_state: Any, extra: dict, opt: Any, config_bytes: Optional[bytes]) -> dict:
    unknown = sorted(set(extra) - _EXTRA)
    if unknown:
        raise ValueError(f"extra has keys trainer_state.json cannot hold: {unknown} (allowed: {sorted(_EXTRA)})")
    cur = extra.get("data_cursor", {"shard": 0, "offset": 0})
    if set(cur) != {"shard", "offset"} or not all(type(cur[k]) is int and cur[k] >= 0 for k in cur):
        raise ValueError(f"data_cursor must be {{shard, offset}} non-negative ints, got {cur!r}")
    sha = extra.get("config_sha256") or hashlib.sha256(config_bytes or b"").hexdigest()
    git = extra.get("git_sha", "unknown")
    if not _HEX64.match(sha) or not _GIT.match(git):
        raise ValueError(f"config_sha256 must be 64 hex digits and git_sha 40 or 'unknown': {sha!r}, {git!r}")
    tokens = extra.get("tokens_seen", 0)
    if type(tokens) is not int or tokens < 0:
        raise ValueError(f"tokens_seen must be an int >= 0, got {tokens!r}")
    lr = float(extra["lr"]) if "lr" in extra else float(getattr(opt, "lr", 0.0))
    state = {
        "step": int(step),
        "tokens_seen": tokens,
        "data_cursor": {"shard": cur["shard"], "offset": cur["offset"]},
        "rng": _rng_json(rng_state),
        "lr": lr,
        "config_sha256": sha,
        "git_sha": git,
    }
    if "loss_scale" in extra:
        state["loss_scale"] = float(extra["loss_scale"])
    return state


def save_checkpoint(
    dir: str,
    model: Any,
    opt: Any,
    step: int,
    rng_state: Any,
    extra: dict,
    keep: Optional[int] = None,
) -> str:
    root = Path(dir)
    root.mkdir(parents=True, exist_ok=True)
    name = step_name(step)
    final, tmp = root / name, root / (name + ".tmp")
    names = [n for n, _ in model.named_parameters()]
    skel, opt_arrays = _flatten_opt(opt.state_dict(), names)
    config = extra.get("config")
    config_bytes = (json.dumps(config, sort_keys=True) + "\n").encode() if config is not None else None
    state = _trainer_state(step, rng_state, extra, opt, config_bytes)
    if tmp.exists():
        shutil.rmtree(tmp)  # a crashed earlier attempt at this step
    tmp.mkdir()
    # 1. Every file into the .tmp directory, each fsynced.
    save_safetensors(str(tmp / "model.safetensors"), model.state_dict(), {"format": "tinyllm"})
    _fsync_path(tmp / "model.safetensors")
    save_safetensors(
        str(tmp / "optimizer.safetensors"),
        opt_arrays,
        {
            "format": "tinyllm",
            "optimizer": type(opt).__name__.lower(),
            "state": json.dumps(skel, sort_keys=True, separators=(",", ":")),
            "step": str(int(step)),
        },
    )
    _fsync_path(tmp / "optimizer.safetensors")
    if config_bytes is not None:
        _write(tmp / "config.json", config_bytes)
    _write(tmp / "trainer_state.json", (json.dumps(state, sort_keys=True, indent=2) + "\n").encode())
    # 2. The manifest last: its presence (and agreement) means "complete".
    files = sorted(p.name for p in tmp.iterdir())
    manifest = {
        "kind": "checkpoint",
        "files": [{"name": f, "sha256": _sha256(tmp / f), "bytes": (tmp / f).stat().st_size} for f in files],
    }
    _write(tmp / "MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode())
    _fsync_path(tmp)
    # 3. Into place by rename: a directory appears whole or not at all.
    if final.exists():
        shutil.rmtree(final)  # re-saving a step replaces it
    os.rename(tmp, final)
    _fsync_path(root)
    # 4. LATEST by write-then-rename, so a reader never sees half a name.
    _write(root / "LATEST.tmp", (name + "\n").encode())
    os.rename(root / "LATEST.tmp", root / "LATEST")
    _fsync_path(root)
    # 6. Housekeeping only after the new checkpoint is durable.
    for p in root.iterdir():
        if p.name.endswith(".tmp") and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    if keep is not None:
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        steps = sorted(_step_dirs(root), key=lambda x: -x[0])
        for _, p in steps[keep:]:
            shutil.rmtree(p, ignore_errors=True)
    return str(final)


def _step_dirs(root: Path) -> list[tuple[int, Path]]:
    """(step, path) of every step-<n> directory (no .tmp ones)."""
    out = []
    if root.is_dir():
        for p in root.iterdir():
            mm = _STEP_DIR.match(p.name)
            if mm and p.is_dir():
                out.append((int(mm.group(1)), p))
    return out


def verify_step_dir(path: str) -> list[str]:
    d = Path(path)
    man = d / "MANIFEST.json"
    if not man.is_file():
        return [f"{d.name}: no MANIFEST.json (incomplete)"]
    try:
        doc = json.loads(man.read_text())
        rows = doc["files"]
        if doc.get("kind") != "checkpoint" or not isinstance(rows, list):
            return [f"{d.name}: MANIFEST.json is not a checkpoint manifest"]
        listed = {r["name"]: r for r in rows}
    except (ValueError, KeyError, TypeError) as e:
        return [f"{d.name}: MANIFEST.json does not parse: {e}"]
    errs = []
    for f in REQUIRED_FILES:
        if f not in listed:
            errs.append(f"{d.name}: MANIFEST.json does not list {f}")
    on_disk = {p.name for p in d.iterdir() if p.name != "MANIFEST.json"}
    for f in sorted(on_disk - set(listed)):
        errs.append(f"{d.name}: {f} is not in MANIFEST.json")
    for f, r in sorted(listed.items()):
        p = d / f
        if not p.is_file():
            errs.append(f"{d.name}: {f} is missing")
        elif p.stat().st_size != r.get("bytes") or _sha256(p) != r.get("sha256"):
            errs.append(f"{d.name}: {f} does not match its sha256 or size")
    return errs


def load_checkpoint(dir: str, step: Optional[int] = None) -> Checkpoint:
    root = Path(dir)
    if step is not None:
        p = root / step_name(step)
        errs = verify_step_dir(str(p))
        if errs:
            raise ValueError("; ".join(errs))
        return _read(p)
    dirs = sorted(_step_dirs(root), key=lambda x: -x[0])
    latest = root / "LATEST"
    if latest.is_file():
        mm = _STEP_DIR.match(latest.read_text().strip())
        if mm:
            # Start from what LATEST names; newer directories were never
            # declared complete. Older ones are the fallback.
            dirs = [(n, p) for n, p in dirs if n <= int(mm.group(1))]
    for _, p in dirs:
        if not verify_step_dir(str(p)):
            return _read(p)
    raise FileNotFoundError(f"{root}: no checkpoint directory verifies")


def _read(p: Path) -> Checkpoint:
    model, _ = load_safetensors(str(p / "model.safetensors"))
    arrays, meta = load_safetensors(str(p / "optimizer.safetensors"))
    opt = _unflatten_opt(json.loads(meta["state"]), arrays)
    st = json.loads((p / "trainer_state.json").read_text())
    extra = {k: st[k] for k in ("tokens_seen", "data_cursor", "lr", "config_sha256", "git_sha") if k in st}
    if "loss_scale" in st:
        extra["loss_scale"] = st["loss_scale"]
    if (p / "config.json").is_file():
        extra["config"] = json.loads((p / "config.json").read_text())
    rng = (int(st["rng"]["pcg_state"], 16), int(st["rng"]["pcg_inc"], 16))
    return Checkpoint(str(p), int(st["step"]), model, opt, rng, extra)
