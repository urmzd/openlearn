"""craft.22 course tests: grading YOUR model evals (rung R8).

Annotated exemplars (DESIGN 5.12). A model eval is graded like any test,
by what it catches (DESIGN 5.6), with two twists. The faults are model
mutants: copies of the course's tiny model (primers/craft.22/tinymodel.py)
with one planted defect (a wrong RoPE base, a dropped layer, a tokenizer one
merge short, ...), and a suite flags one only at p < 0.05 over five seeds.
And a model eval must not cry wolf: harmless variants (float64 arithmetic,
another equally good sampling stream) must pass. Every run copies your
test_model_evals.py, evals.py, and baseline.json next to one version of the
model in a scratch directory and runs pytest in its own process group with
a timeout. Nothing here edits your files.

The worked example of the chapter (section 3): five seeds whose bpb is
worse by 0.03 each give the one-sided sign-flip p-value 1/32 = 0.031 < 0.05:
a regression. With one seed better by 0.01 the mean is still worse by 0.022,
above the margin 0.02, but p = 2/32 = 0.0625: not a regression. With four
seeds the smallest p-value is 1/16: a four-seed suite can flag nothing.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PRIMER = Path(os.environ.get("OL_PRIMER_DIR", "primers/craft.22")).resolve()
COURSE = Path(os.environ.get("OL_COURSE_TREE", Path(__file__).resolve().parents[2]))
FIXTURES = Path(os.environ.get("TINYLLM_FIXTURES", COURSE / "fixtures"))
REF = COURSE / "ref" / "primers" / "craft.22"
TESTS = PRIMER / "test_model_evals.py"
FAULTS = COURSE / "mutants" / "craft.22"
BENIGN = Path(__file__).resolve().parent / "benign"
UNIT = "primers/craft.22/tinymodel.py"
THRESHOLD = 0.80
ALLOWED = {
    "evals",
    "tinymodel",
    "numpy",
    "pytest",
    "json",
    "os",
    "pathlib",
    "math",
    "itertools",
    "functools",
}

sys.path.insert(0, str(COURSE / "harness" / "src"))
from olcourse import markers  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def model_text() -> str:
    return (REF / "tinymodel.py").read_text()


def patched(patch: Path) -> str:
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / UNIT
        dst.parent.mkdir(parents=True)
        dst.write_text(model_text())
        p = subprocess.run(
            ["patch", "-s", "-p1", "-d", d, "-i", str(patch)],
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=30,
        )
        assert p.returncode == 0, f"{patch.name} does not apply: {p.stdout}{p.stderr}"
        return dst.read_text()


def run_suite(model: str, timeout: float = 120) -> tuple[int, str]:
    """Your suite against `model` (the text of a tinymodel.py), in its own
    process group; a run past `timeout` counts as a failure."""
    with tempfile.TemporaryDirectory(prefix="ol-craft22-") as d:
        Path(d, "tinymodel.py").write_text(model)
        for f in ("test_model_evals.py", "evals.py", "baseline.json"):
            Path(d, f).write_text((PRIMER / f).read_text())
        env = dict(
            os.environ,
            PYTHONPATH=d,
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
                "test_model_evals.py",
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


@pytest.fixture(scope="module")
def theirs():
    assert (PRIMER / "evals.py").is_file(), (
        f"no {PRIMER / 'evals.py'}: run ol start craft.22"
    )
    sys.path.insert(0, str(REF))  # their evals may import tinymodel
    try:
        return load_module(PRIMER / "evals.py", "craft22_your_evals")
    finally:
        sys.path.remove(str(REF))


@pytest.fixture(scope="module")
def ref():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "evals.py").write_text(
            markers.drop_markers((REF / "evals.py").read_text())
        )
        return load_module(Path(d, "evals.py"), "craft22_ref_evals")


@pytest.fixture(scope="module")
def world():
    tm = load_module(REF / "tinymodel.py", "craft22_tinymodel")
    model = tm.load(FIXTURES / "craft.22" / "tinylm.npz")
    text = (FIXTURES / "craft.22" / "val.txt").read_text()
    vocab = set((FIXTURES / "craft.22" / "words.txt").read_text().split())
    return model, text, vocab


def test_hand_example_five_seeds(theirs):
    # WHY: the chapter's worked example: the sign-flip p-value of five worse
    #      seeds is 1/32, of four 1/16; a mean above the margin with one
    #      seed going the other way is not a regression (p = 2/32).
    # KIND: unit
    # CHAPTER: craft.22 section 3, Worked example by hand
    assert theirs.sign_flip_p([0.03] * 5) == pytest.approx(1 / 32)
    assert theirs.sign_flip_p([0.03] * 4) == pytest.approx(1 / 16)
    assert theirs.sign_flip_p([0.0] * 5) == pytest.approx(1.0)
    assert theirs.sign_flip_p([0.03, 0.03, 0.03, 0.03, -0.01]) == pytest.approx(2 / 32)
    base = [1.0] * 5
    assert theirs.regressed(base, [1.03] * 5, 0.02) is True
    assert theirs.regressed(base, [1.03, 1.03, 1.03, 1.03, 0.99], 0.02) is False
    assert (
        theirs.regressed(base, [1.01] * 5, 0.02) is False
    )  # significant, but under the margin
    assert theirs.regressed([0.9] * 5, [0.85] * 5, 0.02, higher_is_better=True) is True


def test_your_evals_match_the_reference(theirs, ref, world):
    # WHY: your eval library is the instrument; a wrong instrument grades
    #      everything wrong. Its functions equal the course's on the course
    #      model and on small hand cases (the docstring of evals.py is the
    #      spec: window choice, pooled bits over bytes, the 4-gram rule).
    # KIND: golden
    # CHAPTER: craft.22 section 2, Principles
    model, text, vocab = world
    for seed in (0, 3):
        assert theirs.heldout_bpb(model, text, seed) == pytest.approx(
            ref.heldout_bpb(model, text, seed), abs=1e-9
        )
    ws = "the cat sat on the mat the cat sat on the mat".split()
    for n in (2, 4):
        assert theirs.repeat_ngram_rate(ws, n) == pytest.approx(
            ref.repeat_ngram_rate(ws, n)
        )
    assert theirs.repeat_ngram_rate(["a", "b"], 4) == 0.0
    assert theirs.words("Mia's DOG, ran!") == ["mia", "s", "dog", "ran"]
    assert theirs.valid_word_rate("Mia zzq ran", {"mia", "ran"}) == pytest.approx(2 / 3)
    assert theirs.valid_word_rate("123 !", vocab) == 0.0
    assert theirs.quality_score(model, vocab, 1) == pytest.approx(
        ref.quality_score(model, vocab, 1), abs=1e-12
    )


def test_your_suite_is_an_eval_suite():
    # WHY: R8 grades an eval suite: test_model_evals.py with at least two
    #      tests that import your evals and the course model, read the
    #      fixtures through TINYLLM_FIXTURES, and import nothing else that
    #      could compute the answer (no torch, no random); and a
    #      baseline.json with at least five seeds of bpb and quality.
    # KIND: unit
    # CHAPTER: craft.22 section 4, The artifact and its check
    assert TESTS.is_file(), f"write your eval suite in {TESTS} (chapter section 4)"
    text = TESTS.read_text()
    tree = ast.parse(text, str(TESTS))
    tests = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    assert len(tests) >= 2, (
        f"{len(tests)} test functions; write at least 2 (held-out loss and sample quality)"
    )
    assert "TINYLLM_FIXTURES" in text, "read the fixtures through TINYLLM_FIXTURES"
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            ):
                top = name.split(".")[0]
                imported.add(top)
                assert top in ALLOWED, (
                    f"import {name!r}: your suite uses evals, tinymodel, numpy, pytest, the stdlib"
                )
    assert {"evals", "tinymodel"} <= imported, "import evals and tinymodel"
    b = json.loads((PRIMER / "baseline.json").read_text())
    assert len(b.get("seeds", [])) >= 5 and len(b["bpb"]) == len(b["quality"]) == len(
        b["seeds"]
    ), "baseline.json needs seeds (at least 5), bpb, and quality, one value per seed"


def test_your_baseline_is_the_good_model(ref, world):
    # WHY: a regression test is only as good as its baseline. baseline.json
    #      must be the good model measured today (python
    #      primers/craft.22/evals.py --record), not numbers typed in or
    #      recorded from another model.
    # KIND: golden
    # CHAPTER: craft.22 section 5, Pitfalls
    model, text, vocab = world
    b = json.loads((PRIMER / "baseline.json").read_text())
    want = ref.measure(model, text, vocab, b["seeds"])
    assert b["bpb"] == pytest.approx(want["bpb"], abs=1e-6) and b[
        "quality"
    ] == pytest.approx(want["quality"], abs=1e-9), (
        "baseline.json does not match the course model: record it again"
    )


def test_your_suite_passes_on_the_good_model():
    # WHY: baseline A (DESIGN 5.6): your suite passes on the model it was
    #      recorded from. A failure here is a bug in the suite.
    # KIND: conformance
    # CHAPTER: craft.22 section 4, The artifact and its check
    assert TESTS.is_file(), "no primers/craft.22/test_model_evals.py yet"
    rc, out = run_suite(model_text())
    assert rc == 0, "your suite fails on the good model:\n" + tail(out)


@pytest.mark.parametrize("variant", ["b01", "b02"])
def test_no_false_alarms(variant):
    # WHY: a model eval that cries wolf gets switched off. b01 computes in
    #      float64 (every bpb moves by about 1e-7); b02 samples from the same
    #      distribution with another stream (quality moves by up to 0.1 per
    #      seed, 0.03 worse on average, in both directions). Both are as good
    #      as the baseline model and must pass: exact equality fails b01, a
    #      margin without a significance test fails b02.
    # KIND: conformance
    # CHAPTER: craft.22 section 5, Pitfalls
    assert TESTS.is_file(), "no primers/craft.22/test_model_evals.py yet"
    rc, out = run_suite(patched(BENIGN / f"{variant}.patch"))
    assert rc == 0, f"your suite flags the harmless variant {variant}:\n" + tail(out)


def test_model_mutants_are_flagged():
    # WHY: the grade. Each model mutant is a real defect class: three change
    #      what the model computes (s01 RoPE base, s02 a dropped layer, s03
    #      the tokenizer), one only how it decodes (s04: the held-out loss
    #      cannot see it), one makes bpb BETTER while samples get worse
    #      (s07). Your suite must fail on at least 80% of them, and always on
    #      s01 to s03; a survivor prints its description.
    # KIND: fault
    # CATCHES: s01, s02, s03, s04, s05, s06, s07
    # CHAPTER: craft.22 section 5, Pitfalls
    assert TESTS.is_file(), "no primers/craft.22/test_model_evals.py yet"
    rows = faults()
    killed, survivors = 0, []
    for mid, required, public in rows:
        rc, _ = run_suite(patched(FAULTS / f"{mid}.patch"))
        if rc != 0:
            killed += 1
        else:
            survivors.append(f"{mid}{' (required)' if required else ''}: {public}")
    score = killed / len(rows)
    print(f"craft.22 mutation score: {killed}/{len(rows)} = {score:.2f}")
    assert score >= THRESHOLD and not any("(required)" in s for s in survivors), (
        f"score {score:.2f} (bar {THRESHOLD:.2f}); model mutants your suite misses:\n  "
        + "\n  ".join(survivors)
    )
