"""lang.02 course tests: your Makefile for the two-file C program.

Annotated exemplars (DESIGN 5.12). Every test copies primers/lang.02 into a
fresh directory, so nothing here touches your repo. "Rebuilt" is measured by
modification times: the test pins every file's mtime to a known past moment,
runs make, and looks at which outputs got a new mtime.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

SOURCES = ("Makefile", "main.c", "greet.c", "greet.h")
OUTPUTS = ("main.o", "greet.o", "greet")
T0 = int(time.time()) - 10_000  # a moment safely in the past


@pytest.fixture
def work(tmp_path):
    src = Path(os.environ["OL_PRIMER_DIR"])
    for name in SOURCES:
        shutil.copy2(src / name, tmp_path / name)
        os.utime(tmp_path / name, (T0, T0))
    return tmp_path


def make(d: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("MAKEFLAGS", "MFLAGS", "CFLAGS")
    }
    return subprocess.run(
        ["make", "--no-print-directory", *args],
        cwd=d,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def build(d: Path, *args: str) -> None:
    r = make(d, *args)
    assert r.returncode == 0, (
        f"make {' '.join(args)} failed (exit {r.returncode}):\n{r.stdout}{r.stderr}"
    )


def pin(d: Path, names, t: int) -> None:
    for n in names:
        if (d / n).exists():
            os.utime(d / n, (t, t))


def rebuilt(d: Path, since: int) -> set[str]:
    return {
        n for n in OUTPUTS if (d / n).exists() and int((d / n).stat().st_mtime) > since
    }


def test_make_builds_a_program_that_runs(work):
    # WHY: the plain `make` (first target) builds ./greet from both sources,
    #      and the program behaves as section 3 says: output on stdout, exit 0;
    #      wrong usage on stderr with exit 2.
    # KIND: unit
    # CHAPTER: lang.02 section 3
    build(work)
    r = subprocess.run(
        [str(work / "greet"), "Ada"], capture_output=True, text=True, timeout=10
    )
    assert (r.returncode, r.stdout) == (0, "hello, Ada\n")
    r = subprocess.run(
        [str(work / "greet")], capture_output=True, text=True, timeout=10
    )
    assert r.returncode == 2 and r.stdout == "" and "usage" in r.stderr


def test_second_make_rebuilds_nothing(work):
    # WHY: make's whole job is to skip work whose inputs did not change. A
    #      rule whose target never exists (or a .PHONY on the program) reruns
    #      every time, which is how a 2 s build becomes a 2 min build.
    # KIND: unit
    # CHAPTER: lang.02 section 5, pitfall 1
    build(work)
    pin(work, OUTPUTS, T0 + 100)
    build(work)
    want = set()
    got = rebuilt(work, T0 + 100)
    assert got == want, (
        f"make rebuilt {sorted(got) or 'nothing'}; want {sorted(want) or 'nothing'}"
    )


def test_editing_one_source_rebuilds_only_its_object(work):
    # WHY: one object per source is what makes rebuilds incremental. A
    #      single `cc main.c greet.c -o greet` rule recompiles everything on
    #      every change.
    # KIND: unit
    # CHAPTER: lang.02 section 5, pitfall 2
    build(work)
    missing = [n for n in ("main.o", "greet.o") if not (work / n).exists()]
    assert not missing, (
        f"compile each .c to its own object: {missing} not built by `make`"
    )
    pin(work, OUTPUTS, T0 + 100)
    pin(work, ["greet.c"], T0 + 200)
    build(work)
    want = {"greet.o", "greet"}
    got = rebuilt(work, T0 + 200)
    assert got == want, (
        f"make rebuilt {sorted(got) or 'nothing'}; want {sorted(want) or 'nothing'}"
    )


def test_editing_the_header_rebuilds_both_objects(work):
    # WHY: both .c files #include greet.h, so a changed header changes what
    #      both objects should contain. make cannot see #include: without
    #      `main.o: greet.h` and `greet.o: greet.h` it keeps stale objects
    #      that disagree with the header (wrong struct layouts, wrong args).
    # KIND: boundary
    # CHAPTER: lang.02 section 5, pitfall 3
    build(work)
    pin(work, OUTPUTS, T0 + 100)
    pin(work, ["greet.h"], T0 + 200)
    build(work)
    want = {"main.o", "greet.o", "greet"}
    got = rebuilt(work, T0 + 200)
    assert got == want, (
        f"make rebuilt {sorted(got) or 'nothing'}; want {sorted(want) or 'nothing'}"
    )


def test_make_uses_the_cc_variable(work, tmp_path_factory):
    # WHY: `make CC=clang`, sanitizer builds (SANITIZE=1 in your C Makefile
    #      later), and CI all choose the compiler through $(CC). A recipe that
    #      hardcodes gcc ignores them (and on macOS `gcc` is clang anyway).
    # KIND: unit
    # CHAPTER: lang.02 section 5, pitfall 4
    log = tmp_path_factory.mktemp("cc") / "cc.log"
    fake = log.parent / "fakecc"
    fake.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexec cc "$@"\n')
    fake.chmod(0o755)
    build(work, f"CC={fake}")
    lines = log.read_text().splitlines() if log.exists() else []
    assert any("-c" in ln.split() for ln in lines), "no compile went through $(CC)"
    assert any("-o" in ln.split() and "greet" in ln.split() for ln in lines), (
        "the link did not go through $(CC)"
    )


def test_clean_works_even_when_a_file_named_clean_exists(work):
    # WHY: a target that is not a file must be declared .PHONY. Otherwise a
    #      stray file called `clean` makes make say "clean is up to date" and
    #      remove nothing.
    # KIND: boundary
    # CHAPTER: lang.02 section 5, pitfall 5
    build(work)
    (work / "clean").write_text("")
    build(work, "clean")
    assert [n for n in OUTPUTS if (work / n).exists()] == []
