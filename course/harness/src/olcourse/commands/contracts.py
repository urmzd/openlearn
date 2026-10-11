"""ol contracts sync [--to <tag|sha>]   re-vendor contracts/ (DESIGN 5.3)

Without --to, vendors the live openlearn's course/contracts at its HEAD.
With --to, vendors them from that commit through the worktree cache."""

from __future__ import annotations

import argparse
import difflib

from .. import HarnessError, ctx, learner, tree


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "sync":
        print(__doc__)
        return 5
    ap = argparse.ArgumentParser(prog="ol contracts sync")
    ap.add_argument("--to")
    a = ap.parse_args(argv[1:])
    lr = learner.require()
    before = tree.read_version(lr / "contracts" / "VERSION")
    course = ctx.live_course()
    if a.to:
        repo = ctx.git_toplevel(course)
        if repo is None:
            raise HarnessError("--to needs the course tree to be in a git checkout")
        rc, sha = ctx.git(["rev-parse", "--verify", a.to + "^{commit}"], repo)
        if rc != 0:
            raise HarnessError(f"unknown revision {a.to!r}")
        tmp = lr / "contracts" / "VERSION"
        tree.write_version(tmp, before.get("semver", "0.0.0"), sha.strip(), "")
        course = tree.resolve(lr).path
    old = {
        p.relative_to(lr / "contracts").as_posix()
        for p in (lr / "contracts").rglob("*")
        if p.is_file()
    }
    v = tree.vendor(course, lr / "contracts")
    new = {
        p.relative_to(lr / "contracts").as_posix()
        for p in (lr / "contracts").rglob("*")
        if p.is_file()
    }
    ctx.say(
        f"contracts {before.get('semver', '?')} ({str(before.get('sha', '?'))[:12]}) -> "
        f"{v['semver']} ({str(v['sha'])[:12]})"
    )
    for line in difflib.unified_diff(
        sorted(old), sorted(new), "before", "after", lineterm="", n=0
    ):
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            ctx.say("  " + line)
    return 0
