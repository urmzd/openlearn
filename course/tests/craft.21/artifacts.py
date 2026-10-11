"""craft.21 artifact check: your resilience tests (rung R10).

Run by `ol check craft.21` in your repo. Your artifacts live in primers/craft.21/:

  taskq.go             the kata: a durable task queue and its worker (`ol start`)
  resilience_test.go   your resilience tests (package craft21_test), which
                       kill, crash, pause, and starve the kata with the
                       course's fault kit (faults/faults.go)

The check runs the course's suite (course/tests/go/craft_21) against your
kata, runs your tests against your kata and against the course's, and then
grades your tests by planted faults (course/mutants/craft.21). Rung R10: your
tests must fail on at least 0.90 of them and on every one marked required,
which includes the three resilience faults of DESIGN 5.12: the idempotency
key ignored, the lease not renewed, and fsync skipped.

Every Go run happens in a scratch module named craft21 (your kata, the
course's fault kit, and the tests), in its own process group with a timeout,
one at a time. The first run writes primers/craft.21/go.mod and
primers/craft.21/faults/faults.go when they are missing, so `go test` works
in your directory too. It never overwrites a file.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import Ctx, Fail, run  # noqa: E402

DIR = "primers/craft.21"
UNIT = "primers/craft.21/taskq.go"
TESTS = "primers/craft.21/resilience_test.go"
THRESHOLD = 0.90
HERE = Path(__file__).resolve().parent


def course_tree() -> Path:
    return Path(os.environ.get("OL_COURSE_TREE") or HERE.parents[1])


def sh(argv: list[str], cwd: Path, timeout: float = 120) -> tuple[int, str]:
    """Run argv in its own process group; kill the group on timeout (exit 124)."""
    env = dict(
        os.environ, GOWORK="off", GOTOOLCHAIN="local", GOFLAGS="-count=1", GOPROXY="off"
    )
    p = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return 124, (out or "") + f"\n(timed out after {timeout:.0f} s)"


def tail(text: str, n: int = 25) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:])


def reference_kata() -> str:
    sys.path.insert(0, str(course_tree() / "harness" / "src"))
    from olcourse import markers

    return markers.drop_markers((course_tree() / "ref" / UNIT).read_text())


def faulty(mid: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / UNIT
        dst.parent.mkdir(parents=True)
        dst.write_text(reference_kata())
        rc, out = sh(
            [
                "patch",
                "-s",
                "-p1",
                "-d",
                d,
                "-i",
                str(course_tree() / "mutants" / "craft.21" / f"{mid}.patch"),
            ],
            Path(d),
            30,
        )
        if rc != 0:
            raise Fail(f"fault {mid} does not apply: {out}")
        return dst.read_text()


def faults() -> list[tuple[str, bool, str, str]]:
    rows = []
    for line in (
        (course_tree() / "mutants" / "craft.21" / "manifest.tsv")
        .read_text()
        .splitlines()
    ):
        if line.strip() and not line.startswith("#"):
            c = line.split("\t")
            rows.append((c[0], c[5] == "y", c[6], c[2]))
    return rows


def module(c: Ctx, kata: str, tests: str) -> Path:
    """A scratch module craft21: kata, the course's fault kit, and either your
    tests ("learner") or the course's suite ("course")."""
    d = Path(tempfile.mkdtemp(prefix="ol-craft21-"))
    c.cleanups.append(lambda: shutil.rmtree(d, ignore_errors=True))
    (d / "go.mod").write_text("module craft21\n\ngo 1.22\n")
    (d / "taskq.go").write_text(kata)
    (d / "faults").mkdir()
    shutil.copy(HERE / "faults" / "faults.go", d / "faults" / "faults.go")
    if tests == "learner":
        for f in sorted(c.path(DIR).glob("*_test.go")):
            shutil.copy(f, d / f.name)
    else:
        for f in sorted(
            (course_tree() / "tests" / "go" / "craft_21").glob("*_test.go")
        ):
            shutil.copy(f, d / f.name)
    return d


def go_test(d: Path, tags: str = "") -> tuple[int, str]:
    argv = ["go", "test"] + (["-tags", tags] if tags else []) + ["./..."]
    return sh(argv, d, 120)


def learner_kata(c: Ctx) -> str:
    return c.require_file(UNIT).read_text()


# -- tests ----------------------------------------------------------------------


def test_files_present(c: Ctx) -> None:
    # WHY: the check and the chapter agree on the layout; the first run writes
    #      go.mod and the course's fault kit so `go test` works in primers/craft.21.
    # KIND: unit
    # CHAPTER: craft.21 section 4, The artifact and its check
    d = c.path(DIR)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "go.mod").exists():
        (d / "go.mod").write_text("module craft21\n\ngo 1.22\n")
        print("       wrote primers/craft.21/go.mod")
    if not (d / "faults" / "faults.go").exists():
        (d / "faults").mkdir(exist_ok=True)
        shutil.copy(HERE / "faults" / "faults.go", d / "faults" / "faults.go")
        print(
            "       wrote primers/craft.21/faults/faults.go (the course's kit; the check uses its own copy)"
        )
    missing = [p for p in (UNIT, TESTS) if not c.path(p).is_file()]
    if missing:
        raise Fail(
            f"missing: {', '.join(missing)} (run `ol start craft.21` for the kata; section 4 for the tests)"
        )


def test_your_tests_inject_faults(c: Ctx) -> None:
    # WHY: R10 tests are resilience tests: they must crash the disk, kill or
    #      pause a worker, and assert on effects, not only call the happy
    #      path. A suite with no Crash and no effect assertion cannot catch a
    #      skipped fsync or a lost idempotency key, whatever its line count.
    # KIND: unit
    # CHAPTER: craft.21 section 2.5
    src = "\n".join(f.read_text() for f in sorted(c.path(DIR).glob("*_test.go")))
    if not re.search(r"^package craft21_test\b", src, re.M):
        raise Fail("your tests must be black-box: package craft21_test (DESIGN 5.6)")
    need = {
        "a disk crash (Crash or CrashTorn)": r"\.Crash(Torn)?\(",
        "time moved past a lease (Advance)": r"\.Advance\(",
        "an assertion on effects (AssertExactlyOnce or Deliveries)": r"AssertExactlyOnce\(|\.Deliveries\(",
        "a restart (craft21.Open after a crash)": r"craft21\.Open\(",
    }
    missing = [k for k, pat in need.items() if not re.search(pat, src)]
    if missing:
        raise Fail("your tests never use: " + "; ".join(missing))


def test_your_kata_passes_the_course_suite(c: Ctx) -> None:
    # WHY: your queue must itself keep its promises before your tests are
    #      graded on the course's: the course's suite (course/tests/go/craft_21)
    #      runs against your taskq.go.
    # KIND: fault
    # CHAPTER: craft.21 section 4, What the tests check
    rc, out = go_test(module(c, learner_kata(c), "course"), "primer,coursefaults")
    if rc != 0:
        raise Fail("the course's suite fails on your kata:\n" + tail(out))


def test_your_tests_pass_on_your_kata(c: Ctx) -> None:
    # WHY: your tests and your kata agree.
    # KIND: unit
    # CHAPTER: craft.21 section 4, What the tests check
    rc, out = go_test(module(c, learner_kata(c), "learner"))
    if rc != 0:
        raise Fail("your resilience tests fail on your kata:\n" + tail(out))


def test_your_tests_pass_on_the_course_kata(c: Ctx) -> None:
    # WHY: baseline A of mutation grading (DESIGN 5.6): a test that rejects a
    #      correct queue grades nothing. Your tests must pass on the course's
    #      kata, twice (a resilience test that passes only sometimes is a
    #      flaky test, not a resilience test), before any planted fault counts.
    # KIND: unit
    # CHAPTER: craft.21 section 4, What the tests check
    for i in (1, 2):
        rc, out = go_test(module(c, reference_kata(), "learner"))
        if rc != 0:
            raise Fail(
                f"run {i}: your tests reject the course's correct queue:\n" + tail(out)
            )
    c.cache["baseline"] = True


def test_your_tests_catch_the_planted_faults(c: Ctx) -> None:
    # WHY: rung R10 grades your resilience tests by what they catch: each
    #      planted fault runs alone, and your tests must fail on at least 0.90
    #      of them and on every required one, the three resilience faults
    #      (idempotency key ignored, lease not renewed, fsync skipped)
    #      included.
    # KIND: fault
    # CHAPTER: craft.21 section 5, Pitfalls
    if not c.cache.get("baseline"):
        raise Fail("graded only after your tests pass on the course's kata")
    rows = faults()
    killed, survived, required_missed = 0, [], []
    for mid, required, public, tier in rows:
        rc, _ = go_test(module(c, faulty(mid), "learner"))
        if rc != 0:
            killed += 1
        else:
            shown = (
                public if tier == "resilience" or not required else "a planted pitfall"
            )
            survived.append(f"{mid} ({shown})")
            if required:
                required_missed.append(mid)
    score = killed / len(rows)
    print(f"       mutation score {killed}/{len(rows)} = {score:.2f}")
    if score < THRESHOLD or required_missed:
        raise Fail(
            f"score {score:.2f} (threshold {THRESHOLD}); survivors: {', '.join(survived)}"
        )


if __name__ == "__main__":
    raise SystemExit(run(globals()))
