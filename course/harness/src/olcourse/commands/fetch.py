"""ol fetch <asset>... [--verify]
ol fetch [--list]

Pinned large assets (DESIGN 5.11): every row of course/fixtures/ASSETS.tsv is
one file with a URL, a size, a sha256, and a license. `ol fetch smollm2-135m`
downloads the missing files of that asset into
$TINYLLM_CACHE/assets/smollm2-135m/, checks each one, and only then puts it
in place. Files already there and verified are skipped (--verify rehashes
them). Milestones reach them as {asset:smollm2-135m/config.json}.

A `build:<script>` row is made locally by a maintainer generator under
course/oracle/ (nothing is ever published by ol).

Exit: 0 every file present and verified, 5 a download, build, or hash failed."""

from __future__ import annotations

import argparse

from .. import EXIT_HARNESS, EXIT_PASS, HarnessError, assets, ctx
from ..runner import open_run


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ol fetch",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("assets", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--verify", action="store_true", help="rehash files already in the cache"
    )
    a = ap.parse_args(argv)
    r = open_run(need_learner=False)
    rows = assets.load(r.course)
    if a.list or not a.assets:
        if not rows:
            ctx.say(f"no assets pinned in {assets.manifest_path(r.course)}")
            return EXIT_PASS
        by: dict[str, list] = {}
        for x in rows:
            by.setdefault(x.asset, []).append(x)
        for name, xs in sorted(by.items()):
            have = sum(assets.present(x) for x in xs)
            size = sum(x.bytes for x in xs) / 1e6
            ctx.say(
                f"  {name:<24} {have}/{len(xs)} file(s) cached  {size:8.1f} MB  {xs[0].license}  {xs[0].revision}"
            )
        ctx.say(f"  cache: {assets.cache_root()}")
        return EXIT_PASS
    failed = 0
    for name in a.assets:
        ctx.say(f"{ctx.BLD}fetch {name}{ctx.RST}")
        try:
            got, have = assets.fetch(r.course, name, rehash=a.verify)
            ctx.say(
                f"{ctx.GRN}ok{ctx.RST}   {name}: {got} fetched, {have} already cached"
            )
        except HarnessError as e:
            failed += 1
            ctx.say(f"{ctx.RED}FAIL{ctx.RST} {name}: {e}")
    return EXIT_HARNESS if failed else EXIT_PASS
