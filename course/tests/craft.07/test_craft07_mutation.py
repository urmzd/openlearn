"""craft.07 course tests: mutation testing in depth, on your L4.5 suite.

Annotated exemplars (DESIGN 5.12). A test is graded by what it catches
(DESIGN 5.6). Here the subject is the course's L4.5 reference,
tinyllm/eval/seqmetrics.py, and three artifacts of yours in primers/craft.07:
your deepened suite (test_seqmetrics.py), your triage of ten survivors
(triage.toml), and mutants you wrote from L4.5's pitfalls (mutants/*.patch).
Every run copies one version of the unit into a scratch tree, runs pytest
in its own process group with a timeout, and never edits your files.
"""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/craft.07")).resolve()
COURSE = Path(os.environ.get("OL_COURSE_TREE", Path(__file__).resolve().parents[2]))
LEARNER = Path(os.environ.get("OL_LEARNER_ROOT", ".")).resolve()
SUITE = PRIMER / "test_seqmetrics.py"
UNIT = "python/tinyllm/eval/seqmetrics.py"
STATS = "python/tinyllm/prob/stats.py"
SURVIVORS = COURSE / "mutants" / "craft.07"
L45 = COURSE / "mutants" / "L4.5"
ALLOWED = {
    "tinyllm.eval.seqmetrics",
    "pytest",
    "math",
    "json",
    "itertools",
    "collections",
    "numpy",
    "pathlib",
    "os",
}

sys.path.insert(0, str(COURSE / "harness" / "src"))
from olcourse import markers  # noqa: E402


def ref(unit: str) -> str:
    return markers.drop_markers((COURSE / "ref" / unit).read_text())


def manifest(path: Path) -> list[list[str]]:
    rows = []
    for line in (path / "manifest.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append(line.split("\t"))
    return rows


def patched(patch: Path) -> str | None:
    """The reference unit with `patch` applied, or None when it does not apply."""
    with tempfile.TemporaryDirectory(prefix="ol-craft07-p-") as d:
        dst = Path(d) / UNIT
        dst.parent.mkdir(parents=True)
        dst.write_text(ref(UNIT))
        p = subprocess.run(
            ["patch", "-s", "-p1", "-d", d, "-i", str(patch)],
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=30,
        )
        return dst.read_text() if p.returncode == 0 else None


def run(
    unit_text: str, target: list[str], course_tests: bool = False, timeout: float = 90
) -> tuple[int, str]:
    """pytest over `target` with tinyllm/eval/seqmetrics.py = unit_text (and
    the reference M07.4 stats.py beside it), in its own process group."""
    with tempfile.TemporaryDirectory(prefix="ol-craft07-") as d:
        for rel, text in ((UNIT, unit_text), (STATS, ref(STATS))):
            p = Path(d) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        paths = [str(Path(d) / "python")]
        if course_tests:
            paths.append(str(COURSE / "tests"))
            root = [
                "--rootdir",
                str(COURSE / "tests"),
                "--confcutdir",
                str(COURSE / "tests"),
            ]
            args = [str(COURSE / "tests" / "L4.5")]
            cwd = d
        else:
            (Path(d) / "suite").mkdir()
            (Path(d) / "suite" / "test_seqmetrics.py").write_text(SUITE.read_text())
            root = ["--rootdir", str(Path(d) / "suite")]
            args = target
            cwd = str(Path(d) / "suite")
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(paths),
            TINYLLM_FIXTURES=str(COURSE / "fixtures"),
            PYTHONDONTWRITEBYTECODE="1",
        )
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-x",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            *root,
            *args,
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return 124, f"timed out after {timeout:.0f} s\n{out}"
        return proc.returncode, out


def tail(text: str, n: int = 20) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:])


def triage() -> dict:
    return tomllib.loads((PRIMER / "triage.toml").read_text())


def suite_tests() -> set[str]:
    tree = ast.parse(SUITE.read_text())
    return {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }


def test_artifacts_are_complete():
    # WHY: the three artifacts exist and are well formed before anything is
    #      graded: a suite of at least 12 tests that imports only the
    #      contract (and no `random`), a verdict with a reason of at least 20
    #      characters for each of the ten survivors, and at least two mutant
    #      patches, each starting with a `# pitfall:` line.
    # KIND: unit
    # CHAPTER: craft.07 section 4
    assert SUITE.is_file(), (
        f"write your deepened L4.5 suite at {SUITE} (chapter section 4)"
    )
    tree = ast.parse(SUITE.read_text(), str(SUITE))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module in ALLOWED, (
                f"test_seqmetrics.py imports {node.module}: only the contract and the standard helpers"
            )
        elif isinstance(node, ast.Import):
            for a in node.names:
                assert a.name in ALLOWED, (
                    f"test_seqmetrics.py imports {a.name}: only the contract and the standard helpers"
                )
    assert len(suite_tests()) >= 12, (
        f"{len(suite_tests())} tests; a deep suite has at least 12"
    )
    t = triage()
    for row in manifest(SURVIVORS):
        mid = row[0]
        assert mid in t, f"triage.toml has no [{mid}]"
        assert t[mid].get("verdict") in ("equivalent", "killable"), (
            f"[{mid}] verdict must be equivalent or killable"
        )
        assert len(str(t[mid].get("reason", "")).strip()) >= 20, (
            f"[{mid}] needs a reason (why it is equivalent, or the input that kills it)"
        )
    pats = (
        sorted((PRIMER / "mutants").glob("*.patch"))
        if (PRIMER / "mutants").is_dir()
        else []
    )
    assert len(pats) >= 2, (
        "write at least two mutants as primers/craft.07/mutants/*.patch (chapter section 4)"
    )
    for p in pats:
        assert p.read_text().startswith("# pitfall:"), (
            f"{p.name}: the first line names the pitfall (`# pitfall: ...`)"
        )


def test_suite_accepts_the_reference():
    # WHY: baseline A (DESIGN 5.6): a test that rejects a correct
    #      implementation is wrong however many mutants it kills, and it
    #      would make every equivalent mutant look killable.
    # KIND: conformance
    # CHAPTER: craft.07 section 2.1
    assert SUITE.is_file(), "no primers/craft.07/test_seqmetrics.py yet"
    rc, out = run(ref(UNIT), ["test_seqmetrics.py"])
    assert rc == 0, "your suite rejects the reference:\n" + tail(out)


def test_suite_accepts_your_l45():
    # WHY: baseline B: the suite also passes on your own L4.5 unit
    #      (python/tinyllm/eval/seqmetrics.py in your repo). A failure here is
    #      a bug in one of the two; finish L4.5 first.
    # KIND: unit
    # CHAPTER: craft.07 section 4
    mine = LEARNER / UNIT
    assert mine.is_file(), (
        f"{UNIT} is missing: craft.07 deepens your L4.5 suite, so do L4.5 first"
    )
    rc, out = run(mine.read_text(), ["test_seqmetrics.py"])
    assert rc == 0, "your suite fails on your own L4.5:\n" + tail(out)


def test_suite_kills_every_l45_mutant():
    # WHY: rung R5 asked for 80% of L4.5's mutants; in depth means all of
    #      them. Every committed mutant is killed by the course tests, so none
    #      is equivalent: a survivor is a test you have not written yet.
    # KIND: fault
    # CHAPTER: craft.07 section 2.2
    assert SUITE.is_file(), "no primers/craft.07/test_seqmetrics.py yet"
    survivors = []
    for row in manifest(L45):
        text = patched(L45 / f"{row[0]}.patch")
        assert text is not None, f"L4.5 mutant {row[0]} does not apply to the reference"
        rc, _ = run(text, ["test_seqmetrics.py"])
        if rc == 0:
            survivors.append(
                f"{row[0]}: {row[6] if row[6] != '(hidden until pass)' else 'a planted bug from ' + row[3]}"
            )
    assert not survivors, "L4.5 mutants your suite misses:\n  " + "\n  ".join(survivors)


def test_triage_matches_the_survivors():
    # WHY: the skill this module teaches: reading a survivor and deciding
    #      whether ANY input tells it apart from the reference (killable) or
    #      none does (equivalent). Five of the ten are each kind.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s08, s09, s10
    # CHAPTER: craft.07 section 3
    t = triage()
    wrong = []
    for row in manifest(SURVIVORS):
        want = "equivalent" if row[3] == "equivalent" else "killable"
        if t.get(row[0], {}).get("verdict") != want:
            wrong.append(
                f"{row[0]}: not {t.get(row[0], {}).get('verdict')!r} ({row[6]})"
            )
    assert not wrong, "verdicts to revisit:\n  " + "\n  ".join(wrong)


def test_killable_survivors_die_by_the_named_test():
    # WHY: a killable verdict is a claim; the named test proves it. That test
    #      alone passes on the reference and fails on the survivor, so the
    #      kill is the test's own, not a side effect of another.
    # KIND: fault
    # CATCHES: s02, s04, s06, s07, s09
    # CHAPTER: craft.07 section 4
    t = triage()
    names = suite_tests()
    bad = []
    for row in manifest(SURVIVORS):
        if row[3] == "equivalent":
            continue
        name = str(t.get(row[0], {}).get("test", ""))
        if name not in names:
            bad.append(f"{row[0]}: test {name!r} is not in test_seqmetrics.py")
            continue
        rc_ref, _ = run(ref(UNIT), [f"test_seqmetrics.py::{name}"])
        rc_mut, _ = run(
            patched(SURVIVORS / f"{row[0]}.patch"), [f"test_seqmetrics.py::{name}"]
        )
        if rc_ref != 0 or rc_mut == 0:
            bad.append(
                f"{row[0]}: {name} {'fails on the reference' if rc_ref else 'passes on the survivor'}"
            )
    assert not bad, "\n  ".join(["killable survivors not killed by their test:"] + bad)


def test_your_mutants_are_real_and_new():
    # WHY: a semantic mutant is a pitfall made concrete. It must apply to the
    #      reference, be a real bug (the course's L4.5 tests kill it, so it
    #      is not equivalent), and be new (no committed L4.5 or craft.07
    #      mutant produces the same file).
    # KIND: fault
    # CHAPTER: craft.07 section 2.3
    pats = (
        sorted((PRIMER / "mutants").glob("*.patch"))
        if (PRIMER / "mutants").is_dir()
        else []
    )
    assert pats, "no primers/craft.07/mutants/*.patch yet"
    known = {
        patched(d / f"{r[0]}.patch") for d in (L45, SURVIVORS) for r in manifest(d)
    }
    base = ref(UNIT)
    for p in pats:
        text = patched(p)
        assert text is not None, (
            f"{p.name} does not apply to tinyllm/eval/seqmetrics.py (a unified diff, paths a/{UNIT} and b/{UNIT})"
        )
        assert text != base, f"{p.name} changes nothing"
        assert text not in known, (
            f"{p.name} repeats a committed mutant: plant a pitfall of your own"
        )
        rc, out = run(text, [], course_tests=True)
        assert rc != 0, (
            f"{p.name} passes the course's L4.5 tests: it is equivalent, or the bug is outside what L4.5 defines"
        )


def test_your_suite_kills_your_mutants():
    # WHY: the mutants you wrote are the pitfalls you know best; your suite
    #      must kill each of them, the same bar it meets for the course's.
    # KIND: fault
    # CHAPTER: craft.07 section 2.3
    pats = (
        sorted((PRIMER / "mutants").glob("*.patch"))
        if (PRIMER / "mutants").is_dir()
        else []
    )
    assert pats, "no primers/craft.07/mutants/*.patch yet"
    alive = []
    for p in pats:
        text = patched(p)
        assert text is not None, f"{p.name} does not apply"
        rc, _ = run(text, ["test_seqmetrics.py"])
        if rc == 0:
            alive.append(p.name)
    assert not alive, "your suite misses your own mutants: " + ", ".join(alive)
