"""The three things almost every verb needs: the learner repo, the course
tree its contracts/VERSION names, and that tree's registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import HarnessError, ids, learner, registry, tree


@dataclass
class Session:
    learner: Path
    tree: tree.CourseTree
    reg: registry.Registry

    @property
    def course(self) -> Path:
        return self.tree.path

    def module(self, mid: str) -> registry.Module:
        base = mid.removesuffix("+cuda")
        if not ids.is_course_id(mid):
            raise HarnessError(f"{mid!r} is not a course id (DESIGN 3.5)")
        if mid != base:
            raise HarnessError(
                f"{mid}: +cuda variants are local-only and arrive with the CUDA side quest"
            )
        return self.reg.get(base)


def open_session() -> Session:
    lr = learner.require()
    ct = tree.resolve(lr)
    return Session(lr, ct, registry.load(ct.path))
