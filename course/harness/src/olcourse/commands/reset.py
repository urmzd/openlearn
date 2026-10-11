"""ol reset <ID> [--force]   restore the stubs of the module's units

Refuses when a unit has uncommitted changes in your repo, unless --force."""

from __future__ import annotations

import argparse

from .. import EXIT_FAIL, ctx, ledger, units
from ..session import open_session


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ol reset")
    ap.add_argument("id")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    s = open_session()
    m = s.module(a.id)
    if not a.force:
        rc, out = ctx.git(["status", "--porcelain", "--", *m.owned], s.learner)
        dirty = (
            [line[3:] for line in out.splitlines() if line.strip()] if rc == 0 else []
        )
        if dirty:
            ctx.say(
                f"{ctx.RED}refusing{ctx.RST}: uncommitted changes in {', '.join(dirty)}. "
                f"Commit them, or `ol reset {m.id} --force` to throw them away."
            )
            return EXIT_FAIL
    for u in m.owned:
        p = s.learner / u
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(units.stub_text(s.course, s.reg, u, m.id))
        ctx.say(f"  reset {u}")
    ledger.event(s.learner, m.id, "reset")
    return 0
