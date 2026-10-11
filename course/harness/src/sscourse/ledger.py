"""The verdict ledger: <learner>/.ss/verdicts.jsonl (DESIGN 5.4).

One JSON object per line. Check verdicts carry `result`; events (`start`,
`spoiled`) carry `event`. Readers take the last matching line.

  {"id":"L8.3","kind":"build","tree":"sha256:..","sources":{"L8.2":"learner"},
   "result":"pass","assisted":false,"mutation":null,"ss":"a5a70b1","ts":"..."}
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import ctx
from .registry import Registry


def path(learner: Path) -> Path:
    return learner / ".ss" / "verdicts.jsonl"


def entries(learner: Path) -> list[dict]:
    p = path(learner)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append(learner: Path, entry: dict) -> dict:
    entry = dict(entry)
    entry.setdefault("ss", ctx.harness_sha())
    entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    p = path(learner)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry, sort_keys=False) + "\n")
    return entry


def event(learner: Path, mid: str, name: str, **extra) -> None:
    append(learner, {"id": mid, "event": name, **extra})


def has_event(learner: Path, mid: str, name: str) -> bool:
    return any(e.get("id") == mid and e.get("event") == name for e in entries(learner))


def latest(learner: Path, mid: str, full_only: bool = False) -> dict | None:
    """The last verdict for an id. `full_only` skips `--smoke` milestone runs,
    which never stand in for the full milestone."""
    for e in reversed(entries(learner)):
        if (
            e.get("id") == mid
            and "result" in e
            and not (full_only and e.get("mode") == "smoke")
        ):
            return e
    return None


# Directories under an artifact path that hold build output or caches, never
# the learner's work: skipped when hashing (a check's own build would
# otherwise make its pass stale).
_SKIP_DIRS = {
    ".git",
    ".ss",
    ".venv",
    "target",
    "build",
    "bin",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}


def artifact_files(learner: Path, artifacts: list[str]) -> list[Path]:
    """The files under a module's `artifacts` paths, sorted, caches skipped."""
    out: list[Path] = []
    for a in artifacts:
        base = learner / a
        if base.is_file():
            out.append(base)
        elif base.is_dir():
            for p in base.rglob("*"):
                rel = p.relative_to(base).parts
                skipped = any(x in _SKIP_DIRS for x in rel[:-1])
                if p.is_file() and not skipped and p.name != ".DS_Store":
                    out.append(p)
    return sorted(set(out))


def tree_hash(learner: Path, reg: Registry, mid: str, artifacts: bool = True) -> str:
    """Hash of the module's owned units, its artifact files, and its graded
    learner tests. A unit a later module took over (and the learner started)
    no longer counts: from then on edits to it make only the later module
    stale (DESIGN 5.2 point 4)."""
    m = reg.get(mid)
    gone = {
        u
        for u, later in reg.superseded_by(mid).items()
        if has_event(learner, later, "start")
    }
    h = hashlib.sha256()
    for u in sorted(m.owned):
        if u in gone:
            continue
        p = learner / u
        h.update(
            u.encode()
            + b"\0"
            + (p.read_bytes() if p.is_file() else b"<missing>")
            + b"\0"
        )
    if m.artifacts and artifacts:
        # Tag the scheme, so a verdict hashed before artifacts counted (the
        # hash of nothing) never matches a tree whose artifacts are missing.
        h.update(b"artifacts:v1\0" + "\0".join(m.artifacts).encode() + b"\0")
    for p in artifact_files(learner, m.artifacts) if artifacts else []:
        h.update(
            b"artifact\0"
            + p.relative_to(learner).as_posix().encode()
            + b"\0"
            + p.read_bytes()
            + b"\0"
        )
    if m.kind in ("solve", "proof"):
        sdir = learner / "solve"
        files = [sdir / f"{mid}.toml"] + (
            sorted(x for x in (sdir / mid).rglob("*") if x.is_file())
            if (sdir / mid).is_dir()
            else []
        )
        for p in files:
            h.update(
                b"solve\0"
                + p.relative_to(learner).as_posix().encode()
                + b"\0"
                + (p.read_bytes() if p.is_file() else b"<missing>")
                + b"\0"
            )
    lt = (m.learner_tests or {}).get("path")
    if lt:
        base = learner / lt
        files = (
            sorted(x for x in base.rglob("*") if x.is_file())
            if base.is_dir()
            else ([base] if base.is_file() else [])
        )
        for p in files:
            if "__pycache__" in p.parts:
                continue
            h.update(
                p.relative_to(learner).as_posix().encode()
                + b"\0"
                + p.read_bytes()
                + b"\0"
            )
    return "sha256:" + h.hexdigest()


def repo_hash(learner: Path, reg: Registry) -> str:
    """Hash of the code and graded tests every module owns, across the repo
    (`tree_hash` without artifacts: a practice check may write its own
    artifacts, such as a triage file or a lock file, as it runs). With
    `closure_hash` it tells `ss check --all` that a pass was earned on exactly
    the files there are now (`reusable`)."""
    h = hashlib.sha256(b"repo/v1\0")
    for m in reg.ordered():
        t = tree_hash(learner, reg, m.id, artifacts=False)
        h.update(m.id.encode() + b"\0" + t.encode() + b"\0")
    return "sha256:" + h.hexdigest()


def closure_hash(learner: Path, reg: Registry, mid: str) -> str:
    """Hash of the full trees (artifacts included) of a module and its deps."""
    h = hashlib.sha256(b"closure/v1\0")
    for d in [mid] + reg.closure(mid):
        h.update(d.encode() + b"\0" + tree_hash(learner, reg, d).encode() + b"\0")
    return "sha256:" + h.hexdigest()


def reusable(
    v: dict | None,
    repo: str,
    closure: str,
    course_sha: str,
    tainted: bool,
    ci: bool,
    sample_ok: bool = False,
) -> bool:
    """A pass that a new check of the same module would repeat: recorded on
    the same code (`repo_hash`), the same trees of the module and its deps
    (`closure_hash`, artifacts included), course commit, and harness, from a
    clean course tree, with no reference code substituted. Under --ci it must
    be a --ci verdict (practice checks then skip their cluster tier, and a
    sampled mutation estimate counts only when `sample_ok`,
    SS_MUTATION_SAMPLE=1); outside --ci a smoke-mode practice pass does not
    count."""
    if not v or v.get("result") != "pass" or tainted or v.get("tainted"):
        return False
    if v.get("repo") != repo or v.get("closure") != closure:
        return False
    if v.get("course_sha") != course_sha:
        return False
    if v.get("ss") != ctx.harness_sha() or v.get("assisted") or v.get("kind_filter"):
        return False
    if ci:
        if not v.get("ci"):
            return False
        if (v.get("mutation") or {}).get("sampled") and not sample_ok:
            return False
        return True
    return not is_smoke(v)


def matching(learner: Path, reg: Registry, mid: str) -> dict | None:
    """The newest verdict recorded for the files as they are now (same tree
    hash). A failed check of an edit you since reverted does not hide the
    pass the current files earned; a `blocked` verdict ran no test of the
    module and is skipped."""
    now = tree_hash(learner, reg, mid)
    for e in reversed(entries(learner)):
        if (
            e.get("id") == mid
            and e.get("result") not in (None, "blocked")
            and e.get("tree") == now
        ):
            return e
    return None


def is_smoke(v: dict | None) -> bool:
    """A practice verdict from SS_SMOKE=1: its cluster tier never ran."""
    return bool(v) and v.get("mode") == "smoke"


def fresh_pass(
    learner: Path, reg: Registry, mid: str, full_only: bool = False
) -> dict | None:
    """The pass the current files earned, if any. `full_only` refuses a
    smoke-mode pass (path stages and full milestones need the cluster tier)."""
    v = matching(learner, reg, mid)
    if v and v.get("result") == "pass" and not (full_only and is_smoke(v)):
        return v
    return None
