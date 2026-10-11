"""Unit versions: the reference text of a unit as one owner wrote it, its
stub, and the learner's copy (DESIGN 5.2 points 3 and 4).

The current reference for a unit lives at course/ref/<unit> and carries the
markers of the unit's current owner. Each earlier owner's version is a
snapshot at course/ref/history/<owner>/<unit>.
"""

from __future__ import annotations

from pathlib import Path

from . import HarnessError, markers
from .registry import Registry


def ref_path(course: Path, reg: Registry, unit: str, owner: str) -> Path:
    chain = reg.unit_chain(unit)
    if owner not in chain:
        raise HarnessError(f"{owner} does not own {unit}")
    if owner == chain[-1]:
        return course / "ref" / unit
    return course / "ref" / "history" / owner / unit


def ref_text(course: Path, reg: Registry, unit: str, owner: str) -> str:
    p = ref_path(course, reg, unit, owner)
    if not p.is_file():
        raise HarnessError(f"reference missing: {p}")
    return p.read_text()


def stub_text(course: Path, reg: Registry, unit: str, owner: str) -> str:
    return markers.stub(ref_text(course, reg, unit, owner), unit, want=owner)


def learner_text(learner: Path | None, unit: str) -> str | None:
    if learner is None:
        return None
    p = learner / unit
    return p.read_text() if p.is_file() else None


def is_stub(course: Path, reg: Registry, learner: Path, unit: str, owner: str) -> bool:
    """True when the learner's file is byte-equal to the stub ol would write."""
    t = learner_text(learner, unit)
    if t is None:
        return False
    try:
        return t == stub_text(course, reg, unit, owner)
    except HarnessError:
        return False
