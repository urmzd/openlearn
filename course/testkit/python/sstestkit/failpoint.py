"""Named failpoints for Python units (DESIGN 4.4), the same spec as the Go,
Rust, and C kits:

    TL_FAILPOINTS="corpus/shard/after-write=crash;data/fetch=error(timeout);x=3*sleep(50ms)"

Actions: crash (os._exit(137), what a SIGKILL leaves), panic (RuntimeError),
error(msg) (inject raises FailpointError), sleep(d) with d in ms or s, off.
`N*` fires only on the Nth evaluation; `P%` with P in [0, 1] fires with that
probability from a stream seeded by OL_SEED.
"""

from __future__ import annotations

import os
import random
import re
import time


class FailpointError(Exception):
    pass


_ACT = re.compile(
    r"^(?:(\d+)\*)?(?:([0-9.]+)%)?(crash|panic|error|sleep|off)(?:\((.*)\))?$"
)
_state: dict | None = None


def load(spec: str | None = None) -> dict:
    """Parse spec (default $TL_FAILPOINTS) and make it the active set."""
    global _state
    spec = os.environ.get("TL_FAILPOINTS", "") if spec is None else spec
    table = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, eq, act = part.partition("=")
        m = _ACT.match(act.strip())
        if not eq or not m:
            raise ValueError(f"failpoint {part!r}: want name=[N*][P%]action[(arg)]")
        nth = int(m.group(1)) if m.group(1) else 0
        prob = float(m.group(2)) if m.group(2) else 0.0
        if m.group(1) and nth < 1 or not 0.0 <= prob <= 1.0:
            raise ValueError(f"failpoint {part!r}: bad count or probability")
        if m.group(3) == "sleep":
            _duration(m.group(4) or "")
        table[name.strip()] = (m.group(3), m.group(4) or "", nth, prob)
    _state = {
        "table": table,
        "counts": {},
        "rng": random.Random(int(os.environ.get("OL_SEED", "0"))),
    }
    return table


def _duration(s: str) -> float:
    m = re.fullmatch(r"([0-9.]+)(ms|s)", s.strip())
    if not m:
        raise ValueError(f"sleep({s}): want a duration like 50ms or 2s")
    return float(m.group(1)) / (1000.0 if m.group(2) == "ms" else 1.0)


def inject(name: str) -> None:
    """Evaluate a failpoint: returns normally unless it fires."""
    if _state is None:
        load()
    st = _state
    kind, arg, nth, prob = st["table"].get(name, ("off", "", 0, 0.0))
    if kind == "off":
        return
    st["counts"][name] = st["counts"].get(name, 0) + 1
    if nth and st["counts"][name] != nth:
        return
    if prob and st["rng"].random() >= prob:
        return
    if kind == "crash":
        os._exit(137)
    if kind == "panic":
        raise RuntimeError(f"failpoint {name}")
    if kind == "sleep":
        time.sleep(_duration(arg))
        return
    raise FailpointError(f"failpoint {name}: {arg}")


def count(name: str) -> int:
    return (_state or {"counts": {}})["counts"].get(name, 0)
