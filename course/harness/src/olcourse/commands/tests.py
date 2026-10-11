"""ol tests <ID>   the annotated test catalog: name, KIND, WHY (DESIGN 5.12)"""

from __future__ import annotations

from .. import EXIT_HARNESS, catalog, ctx
from ..overlay import Overlay
from ..session import open_session


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return EXIT_HARNESS
    s = open_session()
    m = s.module(argv[0])
    ov = Overlay(
        s.reg,
        s.course,
        m.id,
        {},
        s.learner,
        s.learner / ".ol" / "overlay" / m.id,
        s.learner / ".ol",
    )
    tests = catalog.collect(catalog.files_for(s.course, m.id, ov.test_dir(m.id)))
    ctx.say(f"{ctx.BLD}{m.id}{ctx.RST}  {m.title}  ({len(tests)} tests)")
    smoke = set(m.smoke)
    for t in tests:
        flag = f"  {ctx.DIM}[smoke]{ctx.RST}" if t.name in smoke else ""
        ctx.say(
            f"\n  {ctx.BLD}{t.name}{ctx.RST}  {t.lang}  {ctx.YEL}{t.fields.get('KIND', '?')}{ctx.RST}{flag}"
        )
        ctx.say(f"    why: {t.fields.get('WHY', '(missing)')}")
        if t.fields.get("CHAPTER"):
            ctx.say(f"    chapter: {t.fields['CHAPTER']}")
    if m.learner_tests:
        lt = m.learner_tests
        ctx.say(
            f"\n  your graded tests: {lt.get('path')} (rung R{lt.get('rung')}, threshold {lt.get('threshold')})"
        )
    return 0
