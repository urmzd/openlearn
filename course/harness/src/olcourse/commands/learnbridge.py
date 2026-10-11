"""Internal verbs for `ol learn` and the path.tsv check column (DESIGN 5.15).

ol _learn-status        stdin: one check per line; stdout: check<TAB>state
                        state: pass | assisted | self | none
ol _learn-check CHECK   0 when the check has a fresh pass, running it when
                        there is no verdict yet; nonzero otherwise
"""

from __future__ import annotations

import sys

from .. import HarnessError, ctx, ledger, learner, paths, tree
from .. import registry as registry_mod


def _session():
    h = ctx.learner_home()
    if not (h / "system.toml").is_file():
        return None
    ct = tree.resolve(h)
    return h, ct, registry_mod.load(ct.path)


def target_state(sess, kind: str, target: str) -> str:
    if sess is None:
        return "none"
    h, ct, reg = sess
    if kind in ("module", "solve"):
        if target not in reg.modules:
            return "none"
        st = learner.state(h, ct.path, reg, target)
        return (
            st.status
            if st.status in ("pass", "assisted", "self", "spoiled")
            else "none"
        )
    key = target
    if kind == "conform":
        from ..conform import parse_suite

        try:
            key = "conform:" + parse_suite(target).id
        except HarnessError:
            return "none"
    v = ledger.latest(h, key, full_only=True)
    if v and v.get("result") == "pass":
        return (
            "assisted" if v.get("assisted") else ("self" if v.get("self") else "pass")
        )
    return "none"


def check_state(sess, check: str) -> str:
    states = [target_state(sess, k, t) for k, t in paths.check_targets(check)]
    if not states:
        return "none"
    if "none" in states:
        return "none"
    if all(s == "pass" for s in states):
        return "pass"
    return "assisted"


def main(verb: str, argv: list[str]) -> int:
    try:
        sess = _session()
    except HarnessError:
        sess = None
    if verb == "_learn-status":
        for line in sys.stdin.read().splitlines():
            c = line.strip()
            if c:
                print(f"{c}\t{check_state(sess, c)}")
        return 0
    if len(argv) != 1:
        ctx.err("usage: ol _learn-check <check>")
        return 5
    check = argv[0]
    if check_state(sess, check) != "none":
        return 0
    if sess is None:
        ctx.err("no learner repo: ol course init --name <system>")
        return 5
    from .. import cli
    from ..session import Session
    from .check import check_one

    h, ct, reg = sess
    worst = 0
    for kind, target in paths.check_targets(check):
        if target_state(sess, kind, target) != "none":
            continue
        if kind in ("module", "solve"):
            code = check_one(Session(h, ct, reg), target)
        elif kind == "drill":
            code = cli.main(["drill", "status", target])
        else:
            code = cli.main([kind, target])
        worst = max(worst, code)
    return worst if worst else (0 if check_state(_session(), check) != "none" else 1)
