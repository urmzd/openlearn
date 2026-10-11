"""craft.06 course tests: grading YOUR benchmark and performance gate (rung R7).

Annotated exemplars (DESIGN 5.12). A gate is graded by what it catches
(DESIGN 5.6): your primers/craft.06/gate.py, with your bench.c and
budget.toml, compares a base kernels.c with a head kernels.c. Here the base
is always the course's kata and the head is, in turn, the course's kata
(no false alarm), your kata (it keeps pace), and the course's kata with one
planted slowdown from course/mutants/craft.06 (each must fail the gate).
Every build and run happens in a scratch directory, in its own process
group, with a timeout. Nothing here edits your files.

The worked example of the chapter (section 3): base 8.0 ms, head 9.6 ms is
+20%, inside a 25% budget; head 16.0 ms (twice the work) is +100% and
regresses; exactly +25% (10.0 ms) is still allowed.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/craft.06")).resolve()
COURSE = Path(os.environ.get("OL_COURSE_TREE", Path(__file__).resolve().parents[2]))
REF = COURSE / "ref" / "primers" / "craft.06"
FAULTS = COURSE / "mutants" / "craft.06"
INCLUDE = COURSE / "contracts" / "c" / "include"
UNIT = "primers/craft.06/kernels.c"
THRESHOLD = 0.90
KERNELS = ("kata_matmul", "kata_sum", "kata_rmsnorm", "kata_count_below")

sys.path.insert(0, str(COURSE / "harness" / "src"))
from olcourse import markers  # noqa: E402


def run(cmd: list[str], cwd: Path, timeout: float = 120) -> tuple[int, str]:
    """One command in its own process group; the whole group is killed on
    timeout, which counts as a failure (exit 124)."""
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
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
        return 124, (out or "") + f"\n(timed out after {timeout:.0f} s)"
    return p.returncode, out


def tail(text: str, n: int = 20) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:])


def reference() -> str:
    return markers.drop_markers((REF / "kernels.c").read_text())


def faulty(mid: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / UNIT
        dst.parent.mkdir(parents=True)
        dst.write_text(reference())
        rc, out = run(
            ["patch", "-s", "-p1", "-d", d, "-i", str(FAULTS / f"{mid}.patch")],
            Path(d),
            30,
        )
        assert rc == 0, f"fault {mid} does not apply: {out}"
        return dst.read_text()


def faults() -> list[tuple[str, bool, str]]:
    rows = []
    for line in (FAULTS / "manifest.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            c = line.split("\t")
            rows.append((c[0], c[5] == "y", c[6]))
    return rows


def gate(base: str, head: str) -> tuple[int, str]:
    """Your gate.py on two kernels.c texts (base and head)."""
    with tempfile.TemporaryDirectory(prefix="ol-craft06-") as d:
        b, h = Path(d, "base.c"), Path(d, "head.c")
        b.write_text(base)
        h.write_text(head)
        return run(
            [
                sys.executable,
                str(PRIMER / "gate.py"),
                "--base",
                str(b),
                "--head",
                str(h),
                "--include",
                str(INCLUDE),
            ],
            Path(d),
            300,
        )


def your_gate():
    path = PRIMER / "gate.py"
    assert path.is_file(), f"write your gate in {path} (chapter section 4)"
    spec = importlib.util.spec_from_file_location("craft06_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hand_example_regression_arithmetic():
    # WHY: the chapter's worked example, on your gate's `regressed`: +20% is
    #      inside a 25% budget, twice the time is a regression, and exactly
    #      the budget is still allowed (the bound is inclusive).
    # KIND: unit
    # CHAPTER: craft.06 section 3, Worked example by hand
    g = your_gate()
    assert g.regressed(8.0e6, 9.6e6, 0.25) is False
    assert g.regressed(8.0e6, 16.0e6, 0.25) is True
    assert g.regressed(8.0, 10.0, 0.25) is False
    assert g.regressed(8.0, 10.01, 0.25) is True


def test_your_artifacts_are_a_benchmark_and_a_budget():
    # WHY: rung R7 grades a benchmark and a gate, so both must be real: a
    #      bench.c that times every kata kernel with ol_bench.h and prints its
    #      metrics as one JSON line, and a budget.toml with at least two runs
    #      and one budget per kernel, each a fraction in (0, 1): a budget of
    #      1 or more cannot catch a 2x slowdown.
    # KIND: unit
    # CHAPTER: craft.06 section 4, The artifact and its check
    bench = PRIMER / "bench.c"
    assert bench.is_file(), f"write your benchmark in {bench} (chapter section 4)"
    text = bench.read_text()
    for need in (
        '#include "ol_bench.h"',
        '#include "kernels.h"',
        "ol_bench_best",
        "ol_bench_done",
    ):
        assert need in text, f"bench.c needs {need}"
    for k in KERNELS:
        assert k in text, f"bench.c never times {k}"
    assert "rand(" not in text, "fixed inputs only: no rand() (DESIGN 5.11)"
    doc = tomllib.loads((PRIMER / "budget.toml").read_text())
    assert int(doc.get("runs", 0)) >= 2, (
        "budget.toml: runs >= 2 (the best of several runs)"
    )
    budget = doc.get("budget", {})
    assert len(budget) >= len(KERNELS), (
        f"budget.toml: one [budget] entry per kernel, found {sorted(budget)}"
    )
    for k, v in budget.items():
        assert 0.0 < float(v) < 1.0, f"budget {k} = {v}: a fraction in (0, 1)"
    tree = ast.parse((PRIMER / "gate.py").read_text())
    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "regressed" for n in tree.body
    ), "gate.py defines regressed"


def test_your_kata_is_correct():
    # WHY: a benchmark of wrong code measures nothing. Your kata is compared
    #      with the plain version of each kernel (kata_check.c), on sizes that
    #      are not multiples of the tiles or of 8, with NaN in C beforehand.
    # KIND: golden
    # CHAPTER: craft.06 section 4, The artifact and its check
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "kata_check"
        rc, out = run(
            [
                "cc",
                "-std=c11",
                "-O2",
                f"-I{PRIMER}",
                str(Path(__file__).parent / "kata_check.c"),
                str(PRIMER / "kernels.c"),
                "-o",
                str(exe),
                "-lm",
            ],
            Path(d),
        )
        assert rc == 0, "your kata does not compile:\n" + tail(out)
        rc, out = run([str(exe)], Path(d))
        assert rc == 0, "your kata disagrees with the plain kernels:\n" + tail(out)


def test_your_gate_accepts_the_reference():
    # WHY: baseline A (DESIGN 5.6): with the same kata as base and head the
    #      gate must pass, twice in a row. A gate that fires on noise is
    #      switched off within a week; best-of-runs and alternating base and
    #      head are what keep it quiet.
    # KIND: conformance
    # CHAPTER: craft.06 section 5, Pitfalls
    for attempt in (1, 2):
        rc, out = gate(reference(), reference())
        assert rc == 0, (
            f"run {attempt}: your gate fails the course's kata against itself (exit {rc}):\n"
            + tail(out)
        )


def test_your_kata_keeps_pace():
    # WHY: baseline B: your kata, as head, against the course's as base,
    #      passes your own budget. A plain triple loop does not.
    # KIND: unit
    # CHAPTER: craft.06 section 4, The artifact and its check
    rc, out = gate(reference(), (PRIMER / "kernels.c").read_text())
    assert rc == 0, f"your kata is slower than the budget allows (exit {rc}):\n" + tail(
        out
    )


def test_perf_faults_trip_your_gate():
    # WHY: the grade. Each planted slowdown keeps the results correct and
    #      costs at least 2x on one kernel. Your gate must exit 1 on at least
    #      90% of them and always on s01, s02, s04, and s09; exit 2 (cannot
    #      decide) is not a catch. A fault your gate passes gets one rerun
    #      (CI runners are shared); one it passes twice survives and prints
    #      its description.
    # KIND: fault
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s08, s09, s10
    # CHAPTER: craft.06 section 5, Pitfalls
    rows = faults()
    killed, survivors = 0, []
    for mid, required, public in rows:
        head = faulty(mid)
        rc, out = gate(reference(), head)
        if rc == 0:
            # Noise only ever slows a run, so a stall on the base side can
            # hide a real slowdown once; a gate that truly misses it misses
            # it again. Exit 2 is deterministic and is not retried.
            rc, out = gate(reference(), head)
        if rc == 1:
            killed += 1
        else:
            survivors.append(
                f"{mid}{' (required)' if required else ''}: {public} (gate exit {rc})"
            )
    score = killed / len(rows)
    req_ok = not any("(required)" in s for s in survivors)
    print(f"craft.06 perf-fault score: {killed}/{len(rows)} = {score:.2f}")
    assert score >= THRESHOLD and req_ok, (
        f"score {score:.2f} (bar {THRESHOLD:.2f}); slowdowns your gate misses:\n  "
        + "\n  ".join(survivors)
    )
