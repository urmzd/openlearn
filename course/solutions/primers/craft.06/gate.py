"""primers/craft.06/gate.py: my performance regression gate (rung R7).

    python3 gate.py --base BASE.c --head HEAD.c [--include DIR] [--runs N]

Builds bench.c (next to this file) twice, once with each kernels.c, then
runs the two programs alternately (base, head, base, head, ...) `runs`
times. Each metric keeps its best (smallest) time per side: noise only ever
makes a run slower, so the minimum is the cleanest estimate. A metric
regresses when head > base * (1 + budget), the budget from budget.toml.

Exit 0 when nothing regresses, 1 when something does, 2 when the gate
cannot decide (bad arguments, a build failure, a crash, a metric missing).
The last stdout line is JSON: {"ok": bool, "change": {metric: fraction}}.
Base and head run on the same machine within seconds of each other, so the
gate needs no calibration and no stored numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def regressed(base: float, head: float, max_regression: float) -> bool:
    """True when head is more than max_regression slower than base."""
    return head > base * (1.0 + max_regression)


def load_budget(path: Path) -> tuple[int, dict[str, float]]:
    doc = tomllib.loads(path.read_text())
    budget = {k: float(v) for k, v in doc.get("budget", {}).items()}
    if not budget:
        raise ValueError(f"{path}: no [budget] entries")
    return int(doc.get("runs", 3)), budget


def run(cmd: list[str], timeout: float = 120) -> tuple[int, str]:
    """One subprocess in its own process group, killed whole on timeout."""
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return 124, out + f"\n(timed out after {timeout:.0f} s)"
    return p.returncode, out


def build(kernels: Path, include: list[str], out: Path) -> None:
    cmd = [
        "cc",
        "-std=c11",
        "-O2",
        f"-I{HERE}",
        *[f"-I{d}" for d in include],
        str(HERE / "bench.c"),
        str(kernels),
        "-o",
        str(out),
    ]
    if platform.system() != "Darwin":
        cmd.append("-lm")
    rc, msg = run(cmd)
    if rc != 0:
        raise RuntimeError(f"building {kernels} failed:\n{msg}")


def metrics(exe: Path) -> dict[str, float]:
    rc, out = run([str(exe)])
    if rc != 0:
        raise RuntimeError(f"{exe.name} exited {rc}:\n{out[-2000:]}")
    for line in reversed(out.strip().splitlines()):
        if line.startswith("{"):
            return {k: float(v) for k, v in json.loads(line).items()}
    raise RuntimeError(f"{exe.name} printed no JSON metrics line")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gate.py")
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--head", required=True, type=Path)
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--runs", type=int)
    a = ap.parse_args(argv)
    try:
        runs, budget = load_budget(HERE / "budget.toml")
        runs = a.runs or runs
        best: dict[str, dict[str, float]] = {"base": {}, "head": {}}
        with tempfile.TemporaryDirectory(prefix="gate-") as d:
            exes = {side: Path(d) / f"bench-{side}" for side in best}
            build(a.base.resolve(), a.include, exes["base"])
            build(a.head.resolve(), a.include, exes["head"])
            for _ in range(runs):
                for side in ("base", "head"):  # alternate: drift hits both sides alike
                    for k, v in metrics(exes[side]).items():
                        best[side][k] = min(v, best[side].get(k, float("inf")))
        missing = [k for k in budget if k not in best["base"] or k not in best["head"]]
        if missing:
            raise RuntimeError(f"the benchmark reports no {missing}")
    except (RuntimeError, ValueError, OSError) as e:
        print(f"gate: {e}", file=sys.stderr)
        return 2
    change, bad = {}, []
    print(f"{'metric':<16} {'base ns':>12} {'head ns':>12} {'change':>8} {'budget':>7}")
    for k, limit in budget.items():
        b, h = best["base"][k], best["head"][k]
        change[k] = round(h / b - 1.0, 4)
        slow = regressed(b, h, limit)
        bad += [k] if slow else []
        print(
            f"{k:<16} {b:>12.0f} {h:>12.0f} {change[k]:>+8.1%} {limit:>+7.0%}  {'REGRESSED' if slow else 'ok'}"
        )
    print(json.dumps({"ok": not bad, "change": change}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
