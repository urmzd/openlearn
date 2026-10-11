"""The learner's repo (DESIGN 2.15): location, init, and module state."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import HarnessError, ctx, ledger, tree, units
from .registry import Registry

GITIGNORE = """\
# written by `ol course init`
.ol/
artifacts/*
# C2's run manifest is provenance you commit (its check reads it from a
# clone); the weights beside it stay ignored.
!artifacts/c2/
artifacts/c2/*
!artifacts/c2/run.json
.venv/
__pycache__/
target/
c/build/
go/bin/
"""

SYSTEM_TOML = """\
# Harness manifest (course/DESIGN.md 2.16). You maintain this file.
# [build], [entry], [services.*], [endpoints], [ci], and [deploy] arrive as
# the passes need them; `ol milestone` tells you which entry it is missing.

[system]
name = "{name}"
version = "0.1.0"
course_version = "{semver}"
"""


def home() -> Path:
    return ctx.learner_home()


def require() -> Path:
    h = home()
    if not (h / "system.toml").is_file():
        raise HarnessError(
            f"no learner repo at {h}; run `ol course init --name <system>` (or set OL_COURSE_HOME)"
        )
    return h


def system(learner: Path) -> dict:
    try:
        return tomllib.loads((learner / "system.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise HarnessError(f"system.toml: {e}") from None


def init(name: str, at: Path | None) -> Path:
    dest = (at or home()).resolve()
    if (dest / "system.toml").exists():
        raise HarnessError(f"{dest} already holds a course repo (system.toml exists)")
    if not name or not name.replace("-", "").isalnum() or not name[0].isalpha():
        raise HarnessError(
            "--name must start with a letter and use letters, digits, and '-'"
        )
    dest.mkdir(parents=True, exist_ok=True)
    v = tree.vendor(ctx.live_course(), dest / "contracts")
    (dest / "system.toml").write_text(SYSTEM_TOML.format(name=name, semver=v["semver"]))
    (dest / ".gitignore").write_text(GITIGNORE)
    if not (dest / ".git").exists():
        rc, out = ctx.git(["init", "-q"], dest)
        if rc != 0:
            raise HarnessError(f"git init failed:\n{out}")
    return dest


# ---------------------------------------------------------------------------
# module state


def started(learner: Path, course: Path, reg: Registry, mid: str) -> bool:
    """Started: `ol start` ran for it, one of its own units differs from the
    stub, or one of its artifact files exists (a practice module has no
    units; in a fresh CI clone there is no ledger either)."""
    if ledger.has_event(learner, mid, "start"):
        return True
    m = reg.get(mid)
    if m.kind in ("solve", "proof") and (learner / "solve" / f"{mid}.toml").is_file():
        return True
    if ledger.artifact_files(learner, m.artifacts):
        return True
    for u in m.units:
        if (learner / u).is_file() and not units.is_stub(
            course, reg, learner, u, reg.unit_chain(u)[0]
        ):
            return True
    return False


def superseded(learner: Path, course: Path, reg: Registry, mid: str) -> str | None:
    """The later module that took over one of mid's units, once it is started."""
    for later in sorted(set(reg.superseded_by(mid).values())):
        if started(learner, course, reg, later):
            return later
    return None


@dataclass
class State:
    id: str
    status: str  # todo started pass smoke stale assisted spoiled self
    verdict: dict | None
    note: str = ""


def state(learner: Path, course: Path, reg: Registry, mid: str) -> State:
    v = ledger.latest(learner, mid)
    cur = ledger.matching(learner, reg, mid)
    if cur and cur.get("result") == "pass":
        v = cur  # the current files passed, whatever an edit since reverted did
    if not started(learner, course, reg, mid) and v is None:
        return State(mid, "todo", None)
    later = superseded(learner, course, reg, mid)
    if v and v.get("result") == "pass" and later:
        return State(mid, "pass", v, f"superseded by {later}")
    if not v or v.get("result") != "pass":
        return State(mid, "started", v, f"last check: {v['result']}" if v else "")
    if v.get("tree") != ledger.tree_hash(learner, reg, mid):
        return State(mid, "stale", v, "files changed since the last pass")
    if ledger.has_event(learner, mid, "spoiled"):
        return State(mid, "spoiled", v)
    if v.get("assisted"):
        refs = sorted(k for k, s in v.get("sources", {}).items() if s == "ref")
        return State(mid, "assisted", v, "ref deps: " + ",".join(refs))
    if v.get("self"):
        return State(mid, "self", v)
    if ledger.is_smoke(v):
        return State(mid, "smoke", v, "passed without its cluster tier (OL_SMOKE)")
    return State(mid, "pass", v)
