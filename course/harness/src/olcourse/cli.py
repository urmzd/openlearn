"""Entry point: `python -m olcourse <verb> [args]`, called by practice/bin/ol."""

from __future__ import annotations

import importlib
import signal
import sys

from . import EXIT_HARNESS, HarnessError, ctx

# verb -> module under olcourse.commands. Verbs whose module does not exist
# yet report exit 5 with the batch that builds them (DESIGN section 9).
VERBS = {
    "course": "course",
    "start": "start",
    "check": "check",
    "diff": "diff",
    "show": "show",
    "reveal": "show",
    "reset": "reset",
    "tests": "tests",
    "status": "status",
    "next": "next",
    "lint": "lint",
    "verify": "verify",
    "contracts": "contracts",
    "stub": "stub",
    "where": "where",
    "_learn-status": "learnbridge",
    "_learn-check": "learnbridge",
    # built elsewhere in the plan; dispatched here so ol routes them already
    "milestone": "milestone",
    "conform": "conform",
    "parity": "parity",
    "mutate": "mutate",
    "tdd": "tdd",
    "export": "export",
    "drill": "drill",
    "fetch": "fetch",
    "doctor": "doctor",
    "bench": "bench",
}
PENDING: dict[str, str] = {}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `ol status | head` ends quietly
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("verbs: " + " ".join(sorted(v for v in VERBS if not v.startswith("_"))))
        return 0
    verb, rest = argv[0], argv[1:]
    modname = VERBS.get(verb)
    if modname is None:
        ctx.err(f"unknown course verb {verb!r}")
        return EXIT_HARNESS
    try:
        mod = importlib.import_module(f"olcourse.commands.{modname}")
    except ModuleNotFoundError as e:
        if e.name == f"olcourse.commands.{modname}":
            ctx.err(
                f"`ol {verb}` is not built yet (arrives with {PENDING.get(verb, 'a later batch')})"
            )
            return EXIT_HARNESS
        raise
    try:
        if verb.startswith("_learn-"):
            return int(mod.main(verb, rest) or 0)
        if modname == "show":
            return int(mod.main(rest, reveal=(verb == "reveal")) or 0)
        return int(mod.main(rest) or 0)
    except HarnessError as e:
        ctx.err(str(e))
        return e.code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
