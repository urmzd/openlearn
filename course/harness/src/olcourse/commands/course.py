"""ol course init --name <system> [--at DIR]   create the learner repo (DESIGN 2.15)
ol course where                             print the learner repo and course tree in use
ol course ci [--upstream URL]               print the learner CI recipe (DESIGN 5.13, craft.01)"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import ctx, learner, tree

UPSTREAM = "https://github.com/urmzd/openlearn.git"

RECIPE = """\
# Learner CI recipe (course/DESIGN.md 5.13, taught in craft.01). Copy these
# steps into YOUR CI workflow; openlearn ships no CI file (P2). The job needs
# git, uv, and the toolchains of the languages you have started (`ol doctor`).
# It checks out public openlearn at the sha your contracts/VERSION names, so
# the course tests that grade you are the ones your contracts came with.
set -euo pipefail
SHA="$(sed -n 's/^sha = "\\(.*\\)"$/\\1/p' contracts/VERSION)"
[ -n "$SHA" ] || {{ echo "contracts/VERSION names no sha; run ol contracts sync" >&2; exit 5; }}
if [ ! -d .ol/openlearn/.git ]; then
  git clone --quiet {upstream} .ol/openlearn
fi
git -C .ol/openlearn fetch --quiet origin
git -C .ol/openlearn checkout --quiet --detach "$SHA"
OL_COURSE_HOME="$PWD" .ol/openlearn/practice/bin/ol check --all --ci
"""


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("init", "where", "ci"):
        print(__doc__)
        return 5
    if argv[0] == "ci":
        ap = argparse.ArgumentParser(prog="ol course ci")
        ap.add_argument("--upstream", default=UPSTREAM)
        a = ap.parse_args(argv[1:])
        print(RECIPE.format(upstream=a.upstream), end="")
        return 0
    if argv[0] == "where":
        h = ctx.learner_home()
        ctx.say(
            f"learner  {h}{'' if (h / 'system.toml').is_file() else '  (not created: ol course init --name <system>)'}"
        )
        ctx.say(f"course   {ctx.live_course()}")
        if (h / "system.toml").is_file():
            ct = tree.resolve(h)
            ctx.say(
                f"tree     {ct.path}  (sha {ct.sha[:12]}{', live' if ct.live else ''}{', tainted' if ct.tainted else ''})"
            )
        return 0
    ap = argparse.ArgumentParser(prog="ol course init")
    ap.add_argument("--name", required=True)
    ap.add_argument("--at", type=Path)
    a = ap.parse_args(argv[1:])
    dest = learner.init(a.name, a.at)
    v = tree.read_version(dest / "contracts" / "VERSION")
    ctx.say(f"{ctx.GRN}created{ctx.RST} {dest}")
    ctx.say(f"  system.toml   name = {a.name}")
    ctx.say(f"  contracts/    {v.get('semver')} at {str(v.get('sha'))[:12]}")
    ctx.say("  next          ol next")
    if a.at:
        ctx.say(
            f"  {ctx.DIM}point every command at it: export OL_COURSE_HOME={dest}{ctx.RST}"
        )
    return 0
