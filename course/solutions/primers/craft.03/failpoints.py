"""craft.03 kata: the failpoint spec (a primer the later modules reuse).

TL_FAILPOINTS names places in a program where a test wants something to go
wrong, and what:

    TL_FAILPOINTS="train/after-step=60*crash; data/fetch=error(timeout); x=sleep(50ms)"

The system's CLI evaluates `train/after-step` after every optimizer step (a
crash test kills a training run that way). This kata is the pure part: parse the spec and
decide, evaluation by evaluation, whether a failpoint fires. What a fired
rule does (exit 137, raise, sleep) stays with the caller.

The spec (the course testkits' grammar, without the probability form):

    spec    := entry (";" entry)*        empty entries are skipped
    entry   := name "=" [N "*"] action   whitespace around name, N, action is ignored
    action  := "crash" | "panic" | "off" | "error" ["(" msg ")"] | "sleep" "(" duration ")"
    duration:= digits ["." digits] ("ms" | "s")

`N*` (N >= 1) fires on the Nth evaluation of that name only; without it the
rule fires on every evaluation. `off` never fires and is never counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

ACTIONS = ("crash", "panic", "error", "sleep", "off")

_ENTRY = re.compile(r"^(?:(\d+)\s*\*\s*)?([a-z]+)\s*(?:\((.*)\))?$")
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)(ms|s)$")


@dataclass(frozen=True)
class Rule:
    action: str  # one of ACTIONS
    arg: str  # error's message or sleep's duration as written; "" otherwise
    nth: int  # 0: fire on every evaluation; N >= 1: on the Nth only


def sleep_seconds(duration: str) -> float:
    """"50ms" -> 0.05, "2s" -> 2.0, "1.5s" -> 1.5. ValueError otherwise."""
    m = _DURATION.match(duration.strip())
    if not m:
        raise ValueError(f"duration {duration!r}: want digits then ms or s, like 50ms or 2s")
    value = float(m.group(1))
    return value / 1000.0 if m.group(2) == "ms" else value


def parse(spec: str) -> dict[str, Rule]:
    """{name: Rule} for every entry. ValueError naming the entry for: no "=",
    an empty name, a name given twice, an unknown action, N that is not an
    integer >= 1, an argument on crash, panic, or off, or sleep without a
    valid duration."""
    table: dict[str, Rule] = {}
    for raw in spec.split(";"):
        entry = raw.strip()
        if not entry:
            continue
        name, eq, act = entry.partition("=")  # the FIRST "=": a name never holds one
        name = name.strip()
        if not eq or not name:
            raise ValueError(f"failpoint {entry!r}: want name=[N*]action")
        if name in table:
            raise ValueError(f"failpoint {name!r} is given twice")
        m = _ENTRY.match(act.strip())
        if not m or m.group(2) not in ACTIONS:
            raise ValueError(f"failpoint {entry!r}: unknown action (one of {', '.join(ACTIONS)})")
        nth = 0
        if m.group(1) is not None:
            nth = int(m.group(1))
            if nth < 1:
                raise ValueError(f"failpoint {entry!r}: N in N* counts from 1")
        action, arg = m.group(2), m.group(3)
        if arg is not None and action in ("crash", "panic", "off"):
            raise ValueError(f"failpoint {entry!r}: {action} takes no argument")
        if action == "sleep":
            if arg is None:
                raise ValueError(f"failpoint {entry!r}: sleep needs a duration, like sleep(50ms)")
            sleep_seconds(arg)
        table[name] = Rule(action, (arg or "").strip(), nth)
    return table


class Failpoints:
    """A parsed spec plus one evaluation counter per name."""

    def __init__(self, spec: str) -> None:
        self.table = parse(spec)
        self.counts: dict[str, int] = {}

    def evaluate(self, name: str) -> Optional[Rule]:
        """Count this evaluation of name and return its Rule when it fires
        now, else None. A name not in the spec, or set to off, returns None
        and is not counted."""
        rule = self.table.get(name)
        if rule is None or rule.action == "off":
            return None
        n = self.counts.get(name, 0) + 1
        self.counts[name] = n
        if rule.nth and n != rule.nth:
            return None
        return rule

    def count(self, name: str) -> int:
        """How many times name has been evaluated (and counted) so far."""
        return self.counts.get(name, 0)
