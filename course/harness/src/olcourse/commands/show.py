"""ol show <ID> / ol reveal <ID>   print the reference units (spoils the module)

D34: references are an honor system. Reading them through ol records the
module as `spoiled` in the ledger, which `ol status` and `ol export` report."""

from __future__ import annotations

from .. import EXIT_HARNESS, ctx, ledger, markers, units
from ..session import open_session


def main(argv: list[str], reveal: bool = False) -> int:
    if len(argv) != 1:
        print(__doc__)
        return EXIT_HARNESS
    s = open_session()
    m = s.module(argv[0])
    ledger.event(s.learner, m.id, "spoiled", via="reveal" if reveal else "show")
    for u in m.owned:
        ctx.say(f"{ctx.BLD}{u}{ctx.RST}  {ctx.DIM}({m.id} reference){ctx.RST}\n")
        ctx.say(markers.drop_markers(units.ref_text(s.course, s.reg, u, m.id)))
    ctx.say(f"{ctx.YEL}{m.id} is now recorded as spoiled.{ctx.RST}")
    return 0
