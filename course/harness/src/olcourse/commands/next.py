"""ol next   the next stage in content/courses/build-an-llm/paths/course/path.tsv whose deps pass (DESIGN 5.3)

A stage with a `module:`/`solve:` check is done when that module has a fresh
pass in the ledger (or the stage is marked done by `ol learn`); a manual stage
is done when `ol learn course --done <stage>` marked it."""

from __future__ import annotations

from .. import ctx, ledger, learner, paths
from ..session import open_session


def main(argv: list[str]) -> int:
    s = open_session()
    base = paths.paths_dir(s.course)
    try:
        rows = paths.expand("course", base)
    except FileNotFoundError:
        ctx.say(f"no course path yet ({base / 'course' / 'path.tsv'})")
        return 0
    for r in rows:
        if r.stage in paths.done_stages(r.owner):
            continue
        targets = paths.check_targets(r.check)
        mods = [
            t for k, t in targets if k in ("module", "solve") and t in s.reg.modules
        ]
        if mods and all(ledger.fresh_pass(s.learner, s.reg, t) for t in mods):
            continue
        waiting = [
            d
            for t in mods
            for d in s.reg.get(t).deps
            if s.reg.get(d).owned
            and learner.state(s.learner, s.course, s.reg, d).status
            not in ("pass", "assisted")
        ]
        if waiting:
            continue
        key = r.stage if r.owner == "course" else f"{r.owner}:{r.stage}"
        ctx.say(f"{ctx.BLD}next: stage {key}{ctx.RST}  {r.title}")
        ctx.say(f"  read      {r.files.replace(',', '  ')}")
        ctx.say(f"  done when {r.when}")
        for k, t in targets:
            if k in ("module", "solve"):
                ctx.say(f"  do        ol start {t} && ol check {t}")
            else:
                ctx.say(f"  check     ol {k} {t}")
        if not targets:
            ctx.say(f"  mark      ol learn course --done {key}")
        return 0
    ctx.say(f"{ctx.GRN}every stage of the course path whose deps pass is done{ctx.RST}")
    return 0
