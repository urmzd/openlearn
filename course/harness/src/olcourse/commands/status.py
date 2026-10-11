"""ol status [--graph] [--counts] [--json]   every module's state (DESIGN 5.3)

States: todo, started, pass, smoke (a practice pass under OL_SMOKE, shown as
`pass (smoke)`: its cluster tier never ran), stale, assisted, spoiled, self."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .. import ctx, learner
from ..session import open_session

COLOR = {
    "pass": ctx.GRN,
    "smoke": ctx.YEL,
    "assisted": ctx.YEL,
    "spoiled": ctx.YEL,
    "self": ctx.YEL,
    "stale": ctx.RED,
    "started": ctx.BLD,
    "todo": ctx.DIM,
}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ol status")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    s = open_session()
    states = [learner.state(s.learner, s.course, s.reg, m.id) for m in s.reg.ordered()]
    if a.json:
        print(
            json.dumps(
                [
                    {
                        "id": st.id,
                        "status": st.status,
                        "note": st.note,
                        "mutation": (st.verdict or {}).get("mutation"),
                    }
                    for st in states
                ],
                indent=1,
            )
        )
        return 0
    if a.counts:
        c = Counter(st.status for st in states)
        ctx.say("  ".join(f"{k} {c.get(k, 0)}" for k in COLOR))
        return 0
    if a.graph:
        by = {st.id: st.status for st in states}
        for m in s.reg.ordered():
            deps = " ".join(f"{d}({by.get(d, '?')})" for d in m.deps) or "-"
            ctx.say(f"  {COLOR.get(by[m.id], '')}{m.id:<10}{ctx.RST} <- {deps}")
        return 0
    cur = None
    for st in states:
        m = s.reg.get(st.id)
        if m.pass_ != cur:
            cur = m.pass_
            ctx.say(f"{ctx.BLD}pass {cur}{ctx.RST}")
        shown = "pass (smoke)" if st.status == "smoke" else st.status
        ctx.say(
            f"  {COLOR.get(st.status, '')}{shown:<9}{ctx.RST} {m.id:<10} {m.title}"
            + (f"  {ctx.DIM}{st.note}{ctx.RST}" if st.note else "")
        )
    if not states:
        ctx.say("no modules in the registry yet")
    return 0
