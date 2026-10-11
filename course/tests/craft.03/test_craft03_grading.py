"""craft.03 course tests: the grade of YOUR tests, primers/craft.03/test_failpoints.py.

This is mutation testing (DESIGN 5.6) in miniature, the same procedure
`ol check` and `ol mutate` apply to your graded tests from L0.3 on:

  1. your tests pass against your own kata (green);
  2. your tests pass against the course's kata (they do not reject a correct
     implementation);
  3. for each planted fault in course/mutants/craft.03 (one at a time, a
     patch of the course's kata), your tests must FAIL: the fault is
     killed. The score is killed / total; you pass with at least 0.70 and
     the fault marked required killed (rung R3).

A surviving fault is shown by its public description, or as "a planted bug
from Pitfall N survived" for a semantic one. Each run is a fresh pytest
process in its own process group, one at a time, with a 60 s limit.
"""

from __future__ import annotations

import ast
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/craft.03"))
TESTS = PRIMER / "test_failpoints.py"
SPEC_NAMES = {"ACTIONS", "Rule", "parse", "Failpoints", "sleep_seconds"}
THRESHOLD = 0.70


def course_tree() -> Path | None:
    t = os.environ.get("OL_COURSE_TREE")
    if t and (Path(t) / "ref" / "primers" / "craft.03" / "failpoints.py").is_file():
        return Path(t)
    return None


def run_tests_against(kata_text: str) -> tuple[bool, str]:
    """Run your test file against one kata implementation in a scratch dir:
    (passed, output)."""
    with tempfile.TemporaryDirectory(prefix="craft03-") as td:
        d = Path(td)
        (d / "failpoints.py").write_text(kata_text)
        shutil.copy(TESTS, d / "test_failpoints.py")
        env = dict(os.environ, PYTHONPATH=str(d), PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.Popen(
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
                "--rootdir",
                str(d),
                "--confcutdir",
                str(d),
                str(d / "test_failpoints.py"),
            ],
            cwd=d,
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            out, _ = p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            return False, out + "\n(timed out after 60 s: counted as a failure)"
        return p.returncode == 0, out


def patched(course: Path, patch: Path) -> str:
    """The course's kata with one planted fault (markers are comments)."""
    with tempfile.TemporaryDirectory(prefix="craft03-mut-") as td:
        d = Path(td)
        unit = d / "primers" / "craft.03" / "failpoints.py"
        unit.parent.mkdir(parents=True)
        unit.write_text(_dropped(course))
        r = subprocess.run(
            ["patch", "-s", "-p1", "-d", str(d), "-i", str(patch)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"{patch.name} does not apply: {r.stdout}{r.stderr}")
        return unit.read_text()


def _dropped(course: Path) -> str:
    text = (course / "ref" / "primers" / "craft.03" / "failpoints.py").read_text()
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if "SOLUTION-BEGIN" not in line and "SOLUTION-END" not in line
    )


def manifest(course: Path) -> list[list[str]]:
    rows = (course / "mutants" / "craft.03" / "manifest.tsv").read_text().splitlines()
    return [r.split("\t") for r in rows if r.strip() and not r.startswith("#")]


def test_your_tests_import_only_the_spec():
    # WHY: graded tests are black-box (DESIGN 5.6): they may use only the
    #      names the kata's spec defines, so they grade behavior and survive
    #      any correct implementation, including the course's.
    # KIND: unit
    # CHAPTER: craft.03 section 2, Principles (black-box tests)
    tree = ast.parse(TESTS.read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "failpoints":
            bad += [a.name for a in node.names if a.name not in SPEC_NAMES]
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "failpoints"
        ):
            if node.attr not in SPEC_NAMES:
                bad.append(node.attr)
    assert not bad, (
        f"your tests use {sorted(set(bad))}, which the spec does not define (allowed: {sorted(SPEC_NAMES)})"
    )
    n = sum(
        1
        for x in ast.walk(tree)
        if isinstance(x, ast.FunctionDef) and x.name.startswith("test_")
    )
    assert n >= 3, f"{n} test functions: write at least one per rule of the spec"


def test_your_tests_pass_on_your_kata():
    # WHY: green first: a test that fails against your own finished code is
    #      either a wrong test or an unfinished kata.
    # KIND: unit
    # CHAPTER: craft.03 section 4, The artifact and its check (step 3)
    ok, out = run_tests_against((PRIMER / "failpoints.py").read_text())
    assert ok, "your tests fail against your own kata:\n" + out[-3000:]


def test_your_tests_accept_the_course_kata():
    # WHY: baseline A of DESIGN 5.6: if your tests reject a correct
    #      implementation, every fault "dies" for the wrong reason and the
    #      score means nothing. They must pass against the course's kata.
    # KIND: unit
    # CHAPTER: craft.03 section 4, The artifact and its check (step 4)
    course = course_tree()
    if course is None:
        pytest.skip(
            "the course's kata needs the openlearn course tree (OL_COURSE_TREE); `ol check craft.03` sets it"
        )
    ok, out = run_tests_against(_dropped(course))
    assert ok, (
        "your tests reject a correct implementation (the course's kata):\n"
        + out[-3000:]
    )


def test_your_tests_kill_the_planted_faults():
    # WHY: a test is graded by what it catches. Each planted fault is one
    #      plausible bug; your tests must fail on at least 70% of them and on
    #      the required one (rung R3). Survivors are listed so you can see
    #      which behavior no test of yours pins down.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, m01, m02
    # CHAPTER: craft.03 section 2, Principles (mutation testing)
    course = course_tree()
    if course is None:
        pytest.skip(
            "the planted faults need the openlearn course tree (OL_COURSE_TREE); `ol check craft.03` sets it"
        )
    ok, out = run_tests_against(_dropped(course))
    if not ok:
        pytest.fail(
            "baseline: your tests fail on the course's kata, so no fault can be graded (see the test above)"
        )
    killed, survived, required_missed = 0, [], []
    rows = manifest(course)
    for mid, _unit, tier, operator, line, required, public, _private in (
        (r + [""] * 8)[:8] for r in rows
    ):
        passed, _ = run_tests_against(
            patched(course, course / "mutants" / "craft.03" / f"{mid}.patch")
        )
        if passed:
            if tier == "semantic":
                p = operator.removeprefix("pitfall-")
                survived.append(f"{mid}: a planted bug from Pitfall {p} survived")
            else:
                survived.append(f"{mid}: {public} (line {line})")
            if required.strip().lower() in ("y", "yes", "true"):
                required_missed.append(mid)
        else:
            killed += 1
    score = killed / len(rows)
    print(
        f"mutation score {score:.2f} ({killed}/{len(rows)} killed), threshold {THRESHOLD:.2f}"
    )
    for s in survived:
        print("  survived", s)
    assert not required_missed, (
        f"required fault(s) {required_missed} survived:\n" + "\n".join(survived)
    )
    assert score >= THRESHOLD, f"score {score:.2f} < {THRESHOLD:.2f}:\n" + "\n".join(
        survived
    )
