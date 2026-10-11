"""craft.04 course tests: grading YOUR property tests (rung R4).

Annotated exemplars (DESIGN 5.12). A test is graded by what it catches
(DESIGN 5.6): your primers/craft.04/test_props.py runs against your kata,
against the course's reference kata, and against each planted fault in
course/mutants/craft.04, in a scratch directory with the Hypothesis profile
`ol` (derandomized, no example database), so a fault is caught or survives
the same way on every run. Nothing here edits your files.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/craft.04")).resolve()
COURSE = Path(os.environ.get("OL_COURSE_TREE", Path(__file__).resolve().parents[2]))
TESTS = PRIMER / "test_props.py"
FAULTS = COURSE / "mutants" / "craft.04"
UNIT = "primers/craft.04/textops.py"
THRESHOLD = 0.80

sys.path.insert(0, str(COURSE / "harness" / "src"))
from olcourse import markers  # noqa: E402


def reference() -> str:
    return markers.drop_markers(
        (COURSE / "ref" / "primers" / "craft.04" / "textops.py").read_text()
    )


def run_props(kata: str, timeout: float = 120) -> subprocess.CompletedProcess:
    """Your test_props.py against `kata` (the text of a textops.py)."""
    with tempfile.TemporaryDirectory(prefix="ol-craft04-") as d:
        Path(d, "textops.py").write_text(kata)
        shutil.copy2(TESTS, Path(d, "test_props.py"))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([d, str(COURSE / "tests")]))
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-x",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:randomly",
                "-p",
                "_lib.ol_hypothesis",
                f"--rootdir={d}",
                "test_props.py",
            ],
            cwd=d,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def tail(text: str, n: int = 25) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:])


def faults() -> list[tuple[str, str, bool, str]]:
    rows = []
    for line in (FAULTS / "manifest.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            c = line.split("\t")
            rows.append((c[0], c[1], c[5] == "y", c[6]))
    return rows


def faulty(mid: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / UNIT
        dst.parent.mkdir(parents=True)
        dst.write_text(reference())
        p = subprocess.run(
            ["patch", "-s", "-p1", "-d", d, "-i", str(FAULTS / f"{mid}.patch")],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, f"fault {mid} does not apply: {p.stdout}{p.stderr}"
        return dst.read_text()


def test_your_tests_are_properties():
    # WHY: rung R4 grades properties, not examples: test_props.py exists and
    #      holds at least five tests, at least four of them Hypothesis
    #      properties (@given), and draws no randomness of its own (the `ol`
    #      profile makes Hypothesis reproducible; `random` would not be).
    # KIND: unit
    # CHAPTER: craft.04 section 4
    assert TESTS.is_file(), f"write your property tests in {TESTS} (chapter section 4)"
    tree = ast.parse(TESTS.read_text(), str(TESTS))
    tests = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    given = [
        n
        for n in tests
        if any(
            isinstance(d, ast.Call)
            and getattr(d.func, "id", getattr(d.func, "attr", "")) == "given"
            for d in n.decorator_list
        )
    ]
    assert len(tests) >= 5, f"{len(tests)} test functions; write at least 5"
    assert len(given) >= 4, f"{len(given)} @given properties; write at least 4"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + (
                [node.module]
                if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
            assert not any(x.split(".")[0] == "random" for x in names), (
                "draw inputs from Hypothesis, not `random`"
            )


def test_your_kata_passes_your_tests():
    # WHY: baseline B (DESIGN 5.6): your properties hold for your own
    #      textops.py. A property that fails here is a bug in one of the two.
    # KIND: unit
    # CHAPTER: craft.04 section 4
    assert TESTS.is_file(), "no primers/craft.04/test_props.py yet"
    r = run_props((PRIMER / "textops.py").read_text())
    assert r.returncode == 0, "your tests fail against your own kata:\n" + tail(
        r.stdout + r.stderr
    )


def test_your_tests_accept_the_reference():
    # WHY: baseline A (DESIGN 5.6): a test that rejects a correct
    #      implementation is wrong, however many faults it catches. Your
    #      properties must hold for the course's textops.py.
    # KIND: conformance
    # CHAPTER: craft.04 section 5, Pitfalls
    assert TESTS.is_file(), "no primers/craft.04/test_props.py yet"
    r = run_props(reference())
    assert r.returncode == 0, "your tests reject a correct implementation:\n" + tail(
        r.stdout + r.stderr
    )


def test_planted_faults_are_caught():
    # WHY: the grade. Each planted fault is one pitfall of the chapter (an
    #      idempotence break, a tie that goes right, a decode that drops
    #      bytes, ...). Your tests must fail on at least 80% of them and on
    #      every required one; a survivor prints its public description.
    # KIND: fault
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s08, s09, s10
    # CHAPTER: craft.04 section 5, Pitfalls
    assert TESTS.is_file(), "no primers/craft.04/test_props.py yet"
    rows = faults()
    killed, survivors = 0, []
    for mid, _unit, required, public in rows:
        r = run_props(faulty(mid))
        if r.returncode != 0:
            killed += 1
        else:
            survivors.append(f"{mid}{' (required)' if required else ''}: {public}")
    score = killed / len(rows)
    req_ok = not any("(required)" in s for s in survivors)
    print(f"craft.04 mutation score: {killed}/{len(rows)} = {score:.2f}")
    assert score >= THRESHOLD and req_ok, (
        f"score {score:.2f} (bar {THRESHOLD:.2f}); planted faults your tests miss:\n  "
        + "\n  ".join(survivors)
    )
