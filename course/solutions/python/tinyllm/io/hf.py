"""Download files from the Hugging Face Hub: resumable, verified, atomic (L7.9).

A 270 MB checkpoint over a flaky connection needs three things a plain
urlopen does not give: resume from where it stopped (HTTP Range), a check
that the bytes are the published ones (the sha256 the Hub sends as
X-Linked-Etag), and never leaving a half-written file under the final name
(write to .part, rename after the check).

Contract: contracts/py/tinyllm/io/hf.pyi.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Optional, Sequence

DEFAULT_ENDPOINT = "https://huggingface.co"
CHUNK = 1 << 20
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class _KeepHeaders(urllib.request.HTTPRedirectHandler):
    """Follows redirects like urllib does, remembering the first response's
    X-Linked-* headers: the CDN the Hub redirects to does not repeat them."""

    def __init__(self, seen: dict) -> None:
        super().__init__()
        self.seen = seen

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        for k in ("X-Linked-Etag", "X-Linked-Size"):
            if headers.get(k) is not None and k not in self.seen:
                self.seen[k] = headers.get(k)
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # Range is carried over


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, final: Path, want_sha: Optional[str], timeout: float) -> None:
    """One file into final, through final.part."""
    part = final.with_name(final.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    h = hashlib.sha256()
    if start:
        with open(part, "rb") as f:  # the resumed hash must cover the bytes already there
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
    seen: dict = {}
    req = urllib.request.Request(url, headers={"User-Agent": "tinyllm-hf-download/1"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if start:
        req.add_header("Range", f"bytes={start}-")
    opener = urllib.request.build_opener(_KeepHeaders(seen))
    try:
        resp = opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code != 416 or not start:
            raise
        resp = None  # 416: nothing after `start`; the .part file is complete
    total = None
    if resp is not None:
        with resp:
            for k in ("X-Linked-Etag", "X-Linked-Size"):
                if resp.headers.get(k) is not None and k not in seen:
                    seen[k] = resp.headers.get(k)
            if resp.status == 200 and start:
                # The server ignored the range and sends the whole file: start over.
                start, h = 0, hashlib.sha256()
            cr = resp.headers.get("Content-Range")
            if resp.status == 206 and cr and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                total = int(cr.rsplit("/", 1)[1])
            elif resp.headers.get("Content-Length") is not None and resp.status == 200:
                total = int(resp.headers["Content-Length"])
            with open(part, "ab" if start else "wb") as f:
                for block in iter(lambda: resp.read(CHUNK), b""):
                    f.write(block)
                    h.update(block)
    if seen.get("X-Linked-Size", "").isdigit():
        total = int(seen["X-Linked-Size"])
    etag = (seen.get("X-Linked-Etag") or "").strip('"').lower()
    expect = (want_sha or (etag if _HEX64.match(etag) else "")).lower()
    size = part.stat().st_size
    if total is not None and size < total:
        # A short body is a network failure: keep the bytes for the next resume.
        raise OSError(f"{url}: the transfer stopped after {size} of {total} bytes; call again to resume")
    if total is not None and size > total:
        part.unlink()
        raise ValueError(f"{url}: got {size} bytes, expected {total}")
    if expect and h.hexdigest() != expect:
        part.unlink()
        raise ValueError(f"{url}: sha256 {h.hexdigest()} does not match {expect}")
    os.replace(part, final)  # atomic: the final name only ever holds a checked file


def hf_download(
    repo_id: str,
    filenames: Sequence[str],
    cache_dir: str,
    revision: str = "main",
    endpoint: Optional[str] = None,
    sha256: Optional[Mapping[str, str]] = None,
    timeout: float = 60.0,
) -> str:
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise ValueError(f"repo_id must be owner/name, got {repo_id!r}")
    base = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
    dest = Path(cache_dir) / repo_id.replace("/", "--") / revision
    for name in filenames:
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"bad filename {name!r}")
        final = dest / name
        final.parent.mkdir(parents=True, exist_ok=True)
        want = (sha256 or {}).get(name)
        if final.exists() and (want is None or _sha256_file(final) == want.lower()):
            continue
        _fetch(f"{base}/{repo_id}/resolve/{revision}/{name}", final, want, timeout)
    return str(dest)
