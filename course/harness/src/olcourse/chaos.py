"""Seeded SIGKILLs of a milestone's services while a step's command runs.

    chaos = { kill = ["durable", "worker"], every_s = "1..4", count = 20, seed = 1 }

While the step's command runs, a background thread sleeps a uniform draw from
`every_s`, picks one name of `kill`, SIGKILLs that service's whole process
group (its children too, such as the worker's Python activities), restarts it
with the same argv, ports, runtime.toml, and state directory, and waits for
its health check; `count` times, or until the command exits. The draws come
from `seed` and the run's `--seed`, so a schedule is reproducible. A service
that does not come back healthy fails the step.

`partition` (cutting links through testkit chaosproxy) and `leader:<glob>`
targets are not built (DEVIATIONS X28).
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError

KEYS = {"kill", "every_s", "count", "seed"}


@dataclass
class Spec:
    kill: list[str]
    every: tuple[float, float]
    count: int
    seed: int


def parse(raw, where: str) -> Spec:
    if not isinstance(raw, dict):
        raise HarnessError(f"{where}: chaos must be a table")
    if "partition" in raw:
        raise HarnessError(
            f"{where}: chaos.partition (testkit chaosproxy link cuts) is not built yet"
        )
    extra = sorted(set(raw) - KEYS)
    if extra:
        raise HarnessError(
            f"{where}: unknown chaos key(s) {extra} (known: {', '.join(sorted(KEYS))})"
        )
    kill = raw.get("kill")
    if (
        not isinstance(kill, list)
        or not kill
        or not all(isinstance(k, str) for k in kill)
    ):
        raise HarnessError(f"{where}: chaos.kill must be a non-empty list of services")
    for k in kill:
        if ":" in k or "*" in k:
            raise HarnessError(
                f"{where}: chaos.kill {k!r}: leader:<glob> targets are not built yet; name a service"
            )
    m = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*\.\.\s*(\d+(?:\.\d+)?)\s*", str(raw.get("every_s", ""))
    )
    if not m or float(m.group(1)) > float(m.group(2)):
        raise HarnessError(
            f'{where}: chaos.every_s {raw.get("every_s")!r} is not "a..b" seconds'
        )
    count = raw.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise HarnessError(f"{where}: chaos.count must be a positive integer")
    seed = raw.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise HarnessError(f"{where}: chaos.seed must be an integer")
    return Spec(list(kill), (float(m.group(1)), float(m.group(2))), count, seed)


@dataclass
class Monkey:
    """Runs a Spec against a services.Stack in a background thread."""

    stack: object  # services.Stack
    spec: Spec
    run_seed: int
    log: Path
    events: list[dict] = field(default_factory=list)
    error: str = ""

    def __post_init__(self) -> None:
        missing = [k for k in self.spec.kill if k not in self.stack.instances]
        if missing:
            raise HarnessError(
                f"chaos.kill names {missing}, which this step's services do not start "
                f"(started: {', '.join(self.stack.instances) or 'none'})"
            )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "Monkey":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _loop(self) -> None:
        rng = random.Random(self.spec.seed * 1_000_003 + self.run_seed)
        t0 = time.monotonic()
        lo, hi = self.spec.every
        with open(self.log, "a") as f:
            for i in range(self.spec.count):
                wait = rng.uniform(lo, hi)
                if self._stop.wait(wait):
                    return
                name = rng.choice(self.spec.kill)
                ev = {
                    "n": i + 1,
                    "wait_s": round(wait, 3),
                    "t_s": round(time.monotonic() - t0, 2),
                    "service": name,
                }
                try:
                    ev["pid"] = self.stack.kill(name)
                    self.stack.restart(name)
                    ev["restarted"] = True
                except HarnessError as e:
                    ev["restarted"] = False
                    self.error = f"after SIGKILL {i + 1}, {e}"
                self.events.append(ev)
                f.write(json.dumps(ev) + "\n")
                f.flush()
                if self.error:
                    return

    def summary(self) -> str:
        by: dict[str, int] = {}
        for e in self.events:
            by[e["service"]] = by.get(e["service"], 0) + 1
        what = ", ".join(f"{k} x{v}" for k, v in sorted(by.items())) or "none"
        return f"chaos: {len(self.events)} of {self.spec.count} SIGKILL(s) ({what})"
