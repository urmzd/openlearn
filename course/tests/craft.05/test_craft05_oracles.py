"""craft.05 course tests: grading YOUR oracle tests (rung R5).

Annotated exemplars (DESIGN 5.12). A test is graded by what it catches
(DESIGN 5.6): your primers/craft.05/test_oracles.py runs against your kata,
against the course's reference kata, and against each planted fault in
course/mutants/craft.05, every run in a scratch directory and its own
process group with a timeout. Nothing here edits your files.

The worked example of the chapter (section 3): y = x W^T for x = (1, 2) and
W = [[1, 0], [3, 1]] is (1, 5); with upstream g = (1, 0) the weight
gradient is g^T x = [[1, 2], [0, 0]], and the transposed bug x^T g =
[[1, 0], [2, 0]] has the same shape. Only an oracle tells them apart.
"""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/craft.05")).resolve()
COURSE = Path(os.environ.get("OL_COURSE_TREE", Path(__file__).resolve().parents[2]))
LEARNER = Path(os.environ.get("OL_LEARNER_ROOT", ".")).resolve()
FIXTURES = Path(os.environ.get("TINYLLM_FIXTURES", COURSE / "fixtures"))
TESTS = PRIMER / "test_oracles.py"
FAULTS = COURSE / "mutants" / "craft.05"
UNIT = "primers/craft.05/kernels.py"
THRESHOLD = 0.80
ALLOWED = {
    "kernels",
    "numpy",
    "pytest",
    "math",
    "os",
    "pathlib",
    "itertools",
    "functools",
    "json",
    "tinyllm",
}

sys.path.insert(0, str(COURSE / "harness" / "src"))
from olcourse import markers  # noqa: E402


def reference() -> str:
    return markers.drop_markers(
        (COURSE / "ref" / "primers" / "craft.05" / "kernels.py").read_text()
    )


def run_oracles(kata: str, timeout: float = 120) -> tuple[int, str]:
    """Your test_oracles.py against `kata` (the text of a kernels.py), in its
    own process group; a run past `timeout` counts as a failure."""
    with tempfile.TemporaryDirectory(prefix="ol-craft05-") as d:
        Path(d, "kernels.py").write_text(kata)
        Path(d, "test_oracles.py").write_text(TESTS.read_text())
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join([d, str(LEARNER / "python")]),
            TINYLLM_FIXTURES=str(FIXTURES),
            PYTHONDONTWRITEBYTECODE="1",
        )
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
                f"--rootdir={d}",
                "test_oracles.py",
            ],
            cwd=d,
            env=env,
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


def tail(text: str, n: int = 25) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:])


def faults() -> list[tuple[str, bool, str]]:
    rows = []
    for line in (FAULTS / "manifest.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            c = line.split("\t")
            rows.append((c[0], c[5] == "y", c[6]))
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
            start_new_session=True,
            timeout=30,
        )
        assert p.returncode == 0, f"fault {mid} does not apply: {p.stdout}{p.stderr}"
        return dst.read_text()


def load_kata(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("craft05_kata", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hand_example_transposed_gradient():
    # WHY: the chapter's worked example: on a square W the transposed weight
    #      gradient has the right shape and the wrong values, so only an
    #      oracle catches it. Checked here on the course's kata.
    # KIND: unit
    # CHAPTER: craft.05 section 3, Worked example by hand
    with tempfile.TemporaryDirectory() as d:
        Path(d, "k.py").write_text(reference())
        K = load_kata(Path(d, "k.py"))
    x, W = np.array([[1.0, 2.0]]), np.array([[1.0, 0.0], [3.0, 1.0]])
    assert K.linear(x, W, np.zeros(2)).tolist() == [[1.0, 5.0]]
    gx, gW, gb = K.linear_backward(x, W, np.array([[1.0, 0.0]]))
    assert (
        gW.tolist() == [[1.0, 2.0], [0.0, 0.0]]
        and gx.tolist() == [[1.0, 0.0]]
        and gb.tolist() == [1.0, 0.0]
    )
    assert (x.T @ np.array([[1.0, 0.0]])).tolist() == [[1.0, 0.0], [2.0, 0.0]]


def test_your_tests_are_oracle_tests():
    # WHY: rung R5 grades oracles: test_oracles.py holds at least six tests,
    #      reads the golden fixture (TINYLLM_FIXTURES, oracles.npz), checks
    #      gradients numerically (a gradcheck of your own, or M04.1's
    #      tinyllm.num.gradcheck), and imports nothing that would be a second
    #      copy of the answer (no torch, no autograd, no `random`).
    # KIND: unit
    # CHAPTER: craft.05 section 4, The artifact and its check
    assert TESTS.is_file(), f"write your oracle tests in {TESTS} (chapter section 4)"
    text = TESTS.read_text()
    tree = ast.parse(text, str(TESTS))
    tests = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    assert len(tests) >= 6, f"{len(tests)} test functions; write at least 6"
    assert "oracles.npz" in text and "TINYLLM_FIXTURES" in text, (
        "read the golden fixture through TINYLLM_FIXTURES"
    )
    names = (
        {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        | {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        | {
            a.asname or a.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names
        }
    )
    assert any("gradcheck" in x or "central" in x for x in names), (
        "check every backward numerically (a gradcheck)"
    )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for mname in mods:
                top = mname.split(".")[0]
                assert top in ALLOWED, (
                    f"import {mname!r}: tests use numpy, pytest, the stdlib, kernels, and your gradcheck"
                )
                assert (
                    not mname.startswith("tinyllm") or mname == "tinyllm.num.gradcheck"
                ), f"import {mname!r}"


def test_your_kata_matches_torch():
    # WHY: your kata is the subject of your tests, so it has to be right: its
    #      forward values and hand-written gradients equal torch's autograd
    #      in course/fixtures/craft.05/oracles.npz (float64).
    # KIND: golden
    # CHAPTER: craft.05 section 2.1, Golden tests
    K = load_kata(PRIMER / "kernels.py")
    f = np.load(FIXTURES / "craft.05" / "oracles.npz")
    for p in ("linear", "linear_sq"):
        assert (
            np.abs(K.linear(f[p + ".x"], f[p + ".W"], f[p + ".b"]) - f[p + ".y"]).max()
            < 1e-10
        ), p
        for got, n in zip(
            K.linear_backward(f[p + ".x"], f[p + ".W"], f[p + ".g"]), ("gx", "gW", "gb")
        ):
            assert (
                got.shape == f[f"{p}.{n}"].shape
                and np.abs(got - f[f"{p}.{n}"]).max() < 1e-10
            ), f"{p}.{n}"
    gx, gw = K.rmsnorm_backward(f["rmsnorm.x"], f["rmsnorm.w"], f["rmsnorm.g"])
    assert (
        np.abs(K.rmsnorm(f["rmsnorm.x"], f["rmsnorm.w"]) - f["rmsnorm.y"]).max() < 1e-10
    )
    assert (
        np.abs(gx - f["rmsnorm.gx"]).max() < 1e-10
        and np.abs(gw - f["rmsnorm.gw"]).max() < 1e-10
    )
    ga, gb = K.swiglu_backward(f["swiglu.a"], f["swiglu.b"], f["swiglu.g"])
    assert np.abs(K.swiglu(f["swiglu.a"], f["swiglu.b"]) - f["swiglu.y"]).max() < 1e-10
    assert (
        np.abs(ga - f["swiglu.ga"]).max() < 1e-10
        and np.abs(gb - f["swiglu.gb"]).max() < 1e-10
    )
    for p, causal in (("attn", False), ("attn_causal", True)):
        q, k, v, g = (f[f"{p}.{n}"] for n in ("q", "k", "v", "g"))
        assert np.abs(K.attention(q, k, v, causal) - f[p + ".y"]).max() < 1e-10, p
        for got, n in zip(K.attention_backward(q, k, v, g, causal), ("gq", "gk", "gv")):
            assert (
                got.shape == f[f"{p}.{n}"].shape
                and np.abs(got - f[f"{p}.{n}"]).max() < 1e-10
            ), f"{p}.{n}"


def test_your_kata_passes_your_tests():
    # WHY: baseline B (DESIGN 5.6): your oracle tests pass on your own kata.
    #      A failure here is a bug in one of the two.
    # KIND: unit
    # CHAPTER: craft.05 section 4, The artifact and its check
    assert TESTS.is_file(), "no primers/craft.05/test_oracles.py yet"
    rc, out = run_oracles((PRIMER / "kernels.py").read_text())
    assert rc == 0, "your tests fail against your own kata:\n" + tail(out)


def test_your_tests_accept_the_reference():
    # WHY: baseline A (DESIGN 5.6): a test that rejects a correct kata is
    #      wrong, however many faults it catches (a tolerance too tight for
    #      float64 rounding is the usual cause).
    # KIND: conformance
    # CHAPTER: craft.05 section 5, Pitfalls
    assert TESTS.is_file(), "no primers/craft.05/test_oracles.py yet"
    rc, out = run_oracles(reference())
    assert rc == 0, "your tests reject a correct implementation:\n" + tail(out)


def test_planted_faults_are_caught():
    # WHY: the grade. Each planted fault is one pitfall of the chapter. Your
    #      tests must fail on at least 80% of them and always on s01, the
    #      transposed weight gradient; a survivor prints its description.
    # KIND: fault
    # CATCHES: s01, s02, s03, s04, s05, s06, s07, s08, s09, s10
    # CHAPTER: craft.05 section 5, Pitfalls
    assert TESTS.is_file(), "no primers/craft.05/test_oracles.py yet"
    rows = faults()
    killed, survivors = 0, []
    for mid, required, public in rows:
        rc, _ = run_oracles(faulty(mid))
        if rc != 0:
            killed += 1
        else:
            survivors.append(f"{mid}{' (required)' if required else ''}: {public}")
    score = killed / len(rows)
    req_ok = not any("(required)" in s for s in survivors)
    print(f"craft.05 mutation score: {killed}/{len(rows)} = {score:.2f}")
    assert score >= THRESHOLD and req_ok, (
        f"score {score:.2f} (bar {THRESHOLD:.2f}); planted faults your tests miss:\n  "
        + "\n  ".join(survivors)
    )
