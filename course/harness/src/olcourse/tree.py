"""contracts/VERSION and which course tree a learner's checks run against
(DESIGN 5.4 "Which course tree runs").

The source `course/contracts/VERSION` holds only `semver`. Vendoring (init,
`ol contracts sync`) writes the learner's copy with three keys:

    semver = "0.1.0"
    sha = "<openlearn commit the contracts came from>"
    content_hash = "sha256:<hash of every vendored file except VERSION>"

Course tests, references, mutants, and fixtures come from the tree at `sha`.
When `sha` is the live checkout's HEAD the live tree is used (tagged
`tainted` when its course/ is dirty); otherwise a cached detached worktree
at $OL_CACHE/worktrees/<sha>.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import EXIT_DRIFT, HarnessError, ctx

IGNORE_NAMES = {".DS_Store", "__pycache__", "target", ".venv"}


def content_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        rel = p.relative_to(d).as_posix()
        if rel == "VERSION" or any(
            part in IGNORE_NAMES for part in p.relative_to(d).parts
        ):
            continue
        h.update(rel.encode() + b"\0" + p.read_bytes() + b"\0")
    return "sha256:" + h.hexdigest()


def read_version(p: Path) -> dict:
    if not p.is_file():
        return {}
    try:
        return tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(f"{p}: {e}") from None


def write_version(p: Path, semver: str, sha: str, chash: str) -> None:
    p.write_text(f'semver = "{semver}"\nsha = "{sha}"\ncontent_hash = "{chash}"\n')


def semver_tuple(s: str) -> tuple[int, int, int]:
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", s or "0.0.0")
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)  # type: ignore[return-value]


def vendor(course: Path, dest: Path) -> dict:
    """Copy course/contracts to dest and stamp VERSION. Returns the VERSION keys."""
    src = course / "contracts"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(*IGNORE_NAMES))
    semver = read_version(src / "VERSION").get("semver", "0.0.0")
    repo = ctx.git_toplevel(course)
    sha = (ctx.git_head(repo) if repo else None) or "live"
    v = {"semver": semver, "sha": sha, "content_hash": content_hash(dest)}
    write_version(
        dest / "VERSION", **{"semver": semver, "sha": sha, "chash": v["content_hash"]}
    )
    return v


@dataclass
class CourseTree:
    path: Path  # the course/ directory to read tests, refs, fixtures from
    sha: str
    live: bool
    tainted: bool


def _dirty(repo: Path, course: Path) -> bool:
    rc, out = ctx.git(["status", "--porcelain", "--", str(course)], repo)
    return rc == 0 and bool(out.strip())


def resolve(learner: Path | None) -> CourseTree:
    live = ctx.live_course()
    if learner is None:
        repo = ctx.git_toplevel(live)
        return CourseTree(
            live,
            (ctx.git_head(repo) if repo else None) or "live",
            True,
            bool(repo and _dirty(repo, live)),
        )
    v = read_version(learner / "contracts" / "VERSION")
    sha = v.get("sha", "live")
    repo = ctx.git_toplevel(live)
    head = ctx.git_head(repo) if repo else None
    if repo is None or sha in ("live", "") or sha == head:
        return CourseTree(live, sha, True, bool(repo and _dirty(repo, live)))
    wt = ctx.cache_dir() / "worktrees" / sha
    if not (wt / ".git").exists():
        wt.parent.mkdir(parents=True, exist_ok=True)
        rc, out = ctx.git(["worktree", "add", "--detach", str(wt), sha], repo)
        if rc != 0:
            raise HarnessError(
                f"cannot check out the course tree at {sha[:12]} (from contracts/VERSION):\n{ctx.indent(out)}\n"
                "run `ol contracts sync` to move to the current openlearn"
            )
    return CourseTree(wt / live.relative_to(repo), sha, False, False)


def precheck_contracts(learner: Path) -> list[str]:
    """Content-hash check of the vendored contracts (exit 4 on drift)."""
    vfile = learner / "contracts" / "VERSION"
    v = read_version(vfile)
    if not v:
        return ["contracts/VERSION is missing; run `ol contracts sync`"]
    want = v.get("content_hash", "")
    got = content_hash(learner / "contracts")
    if want != got:
        return ["contracts/ modified; revert, or run `ol contracts sync`"]
    return []


def advise_semver(learner: Path) -> str | None:
    have = read_version(learner / "contracts" / "VERSION").get("semver", "0.0.0")
    cur = read_version(ctx.live_course() / "contracts" / "VERSION").get(
        "semver", "0.0.0"
    )
    a, b = semver_tuple(have), semver_tuple(cur)
    if b[0] > a[0]:
        return f"contracts {cur} is a major bump over your {have}: a migration chapter is waiting; `ol contracts sync` when you start it"
    if b > a:
        return f"contracts {cur} is available (you have {have}, backward compatible): `ol contracts sync`"
    return None


def require_clean(learner: Path) -> None:
    errs = precheck_contracts(learner)
    if errs:
        raise HarnessError("; ".join(errs), EXIT_DRIFT)
