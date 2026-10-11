"""Pinned large assets (DESIGN 5.11): course/fixtures/ASSETS.tsv and the cache.

    # asset  path  url  sha256  bytes  license  revision
    smollm2-135m  config.json  https://huggingface.co/.../resolve/<rev>/config.json  <sha256>  704  Apache-2.0  HuggingFaceTB/SmolLM2-135M@<rev>

One row per file. `{asset:<asset>/<path>}` resolves to
$TINYLLM_CACHE/assets/<asset>/<path>. A download goes to a temp file next to
its target, is checked against the row's size and sha256, and only then
renamed into place, so a cached file is always a verified one. A verified
file is recorded in <asset>/.ol-verified (path -> sha256, size, mtime) so a
second fetch does not rehash hundreds of megabytes; `--verify` rehashes.

URLs: https:// and http:// (urllib), file:// (a local copy), and
build:<script> (a maintainer generator under course/oracle/ run locally with
`uv run --script <script> <out-dir>`; nothing is ever published).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import HarnessError, ctx

COLUMNS = ("asset", "path", "url", "sha256", "bytes", "license", "revision")
ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class Row:
    asset: str
    path: str
    url: str
    sha256: str
    bytes: int
    license: str
    revision: str


def manifest_path(course: Path) -> Path:
    return course / "fixtures" / "ASSETS.tsv"


def load(course: Path) -> list[Row]:
    p = manifest_path(course)
    if not p.is_file():
        return []
    rows, errs = [], []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) != len(COLUMNS):
            errs.append(
                f"ASSETS.tsv:{i}: {len(c)} columns, want {len(COLUMNS)} ({', '.join(COLUMNS)})"
            )
            continue
        try:
            n = int(c[4])
        except ValueError:
            errs.append(f"ASSETS.tsv:{i}: bytes {c[4]!r} is not an integer")
            continue
        rows.append(Row(c[0], c[1], c[2], c[3], n, c[5], c[6]))
    errs += lint(rows)
    if errs:
        raise HarnessError("\n".join(errs))
    return rows


def lint(rows: list[Row]) -> list[str]:
    errs, seen = [], set()
    for r in rows:
        where = f"ASSETS.tsv {r.asset}/{r.path}"
        if not ASSET_ID.match(r.asset):
            errs.append(
                f"{where}: asset id must be lowercase letters, digits, '.', '-'"
            )
        if r.path.startswith("/") or ".." in Path(r.path).parts:
            errs.append(f"{where}: path must be relative and stay inside the asset")
        if not SHA.match(r.sha256):
            errs.append(f"{where}: sha256 must be 64 lowercase hex digits")
        if not r.url.startswith(("https://", "http://", "file://", "build:")):
            errs.append(
                f"{where}: url must be https://, http://, file://, or build:<script>"
            )
        if not r.license or r.license == "-":
            errs.append(f"{where}: license is required (Q5)")
        if (r.asset, r.path) in seen:
            errs.append(f"{where}: listed twice")
        seen.add((r.asset, r.path))
    return errs


def cache_root() -> Path:
    return ctx.cache_dir() / "assets"


def _sidecar(asset_dir: Path) -> dict:
    try:
        return json.loads((asset_dir / ".ol-verified").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_sidecar(asset_dir: Path, data: dict) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / ".ol-verified").write_text(json.dumps(data, indent=1, sort_keys=True))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def present(r: Row, rehash: bool = False) -> bool:
    d = cache_root() / r.asset
    p = d / r.path
    if not p.is_file() or p.stat().st_size != r.bytes:
        return False
    rec = _sidecar(d).get(r.path)
    st = p.stat()
    if not rehash and rec and rec == [r.sha256, st.st_size, int(st.st_mtime)]:
        return True
    ok = sha256_file(p) == r.sha256
    if ok:
        side = _sidecar(d)
        side[r.path] = [r.sha256, st.st_size, int(st.st_mtime)]
        _save_sidecar(d, side)
    return ok


def _download(url: str, dest: Path, timeout: float) -> None:
    if url.startswith("file://"):
        shutil.copyfile(url[len("file://") :], dest)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "openlearn-ol-fetch/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f, 1 << 20)


def fetch_row(r: Row, timeout: float = 600, built: Path | None = None) -> str:
    """Download (or take from a local build) one file; returns what happened."""
    d = cache_root() / r.asset
    target = d / r.path
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(prefix=".part-", dir=target.parent)
    os.close(fd)
    tmp = Path(tmpname)
    try:
        if built is not None:
            src = built / r.path
            if not src.is_file():
                raise HarnessError(f"{r.asset}/{r.path}: the build did not produce it")
            shutil.copyfile(src, tmp)
        else:
            try:
                _download(r.url, tmp, timeout)
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                raise HarnessError(
                    f"{r.asset}/{r.path}: download failed: {e}"
                ) from None
        size = tmp.stat().st_size
        if size != r.bytes:
            raise HarnessError(
                f"{r.asset}/{r.path}: {size} bytes, ASSETS.tsv pins {r.bytes}"
            )
        got = sha256_file(tmp)
        if got != r.sha256:
            raise HarnessError(
                f"{r.asset}/{r.path}: sha256 {got}, ASSETS.tsv pins {r.sha256}"
            )
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    st = target.stat()
    side = _sidecar(d)
    side[r.path] = [r.sha256, st.st_size, int(st.st_mtime)]
    _save_sidecar(d, side)
    return "built" if built is not None else "downloaded"


def build(course: Path, script: str, out: Path) -> None:
    """Run a maintainer generator: `uv run --script <script> <out>` from the repo root."""
    root = course.parent
    p = root / script
    if not p.is_file():
        raise HarnessError(f"build:{script}: no such generator")
    out.mkdir(parents=True, exist_ok=True)
    env = ctx.base_env()
    env["OL_ASSET_OUT"] = str(out)
    rc, text = ctx.run(
        ["uv", "run", "--script", str(p), str(out)], cwd=root, env=env, timeout=3600
    )
    if rc != 0:
        raise HarnessError(f"build:{script} failed (exit {rc}):\n{ctx.tail(text, 30)}")


def fetch(
    course: Path, asset: str, rehash: bool = False, say=ctx.say
) -> tuple[int, int]:
    """Make every file of an asset present and verified; (fetched, present)."""
    rows = [r for r in load(course) if r.asset == asset]
    if not rows:
        known = sorted({r.asset for r in load(course)})
        raise HarnessError(
            f"no asset {asset!r} in ASSETS.tsv (known: {', '.join(known) or 'none'})"
        )
    fetched = have = 0
    built_dirs: dict[str, Path] = {}
    for r in rows:
        if present(r, rehash):
            have += 1
            say(f"  {ctx.DIM}ok        {r.asset}/{r.path}{ctx.RST}")
            continue
        built = None
        if r.url.startswith("build:"):
            script = r.url[len("build:") :]
            if script not in built_dirs:
                out = Path(
                    tempfile.mkdtemp(prefix=f"build-{asset}-", dir=ctx.cache_dir())
                )
                say(f"  building  {asset} with {script} (local; never published)")
                build(course, script, out)
                built_dirs[script] = out
            built = built_dirs[script]
        t0 = time.monotonic()
        what = fetch_row(r, built=built)
        fetched += 1
        say(
            f"  {ctx.GRN}{what:<9}{ctx.RST} {r.asset}/{r.path}  ({r.bytes / 1e6:.1f} MB, {time.monotonic() - t0:.1f}s, sha256 ok)"
        )
    for d in built_dirs.values():
        shutil.rmtree(d, ignore_errors=True)
    return fetched, have
