"""Mutation grading of the learner's tests (DESIGN 5.6) and red-then-green
(5.12): ol mutate, the cache, sampling in ol check, survivors and their
reveal rule, the black-box lint, ol tdd, and verify check 6 proving the
reference learner tests reach each threshold in all four languages."""

import json

STRONG = """from tinyllm.demo.scale import scale


def test_scales_each_element():
    assert scale([1.0, -2.0, 0.5], 2.0) == [2.0, -4.0, 1.0]


def test_does_not_modify_input():
    xs = [1.0, 2.0]
    scale(xs, 3.0)
    assert xs == [1.0, 2.0]


def test_runs_under_the_ol_hypothesis_profile():
    from hypothesis import settings

    assert settings.default.derandomize and settings.default.database is None
"""
WEAK = STRONG.split("\n\n\ndef test_does_not_modify_input")[0] + "\n"


def _tests(ss, text: str) -> None:
    d = ss.learner / "python/tests/m90-1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "test_scale_mine.py").write_text(text)


def test_python_tdd_then_mutate_then_check(ss):
    ss.add_extras("mutation")
    ss.init()
    ss("start", "M90.1", rc=0)
    out = ss("mutate", "M90.1", rc=1).out
    assert "no graded tests at python/tests/m90-1 yet" in out

    _tests(ss, STRONG)
    assert "no red" in ss("tdd", "green", "M90.1", rc=1).out
    assert "red M90.1" in ss("tdd", "red", "M90.1", rc=0).out  # the stub raises
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    assert "not red" in ss("tdd", "red", "M90.1", rc=1).out
    assert "green M90.1" in ss("tdd", "green", "M90.1", rc=0).out

    out = ss("mutate", "M90.1", rc=0).out
    assert "mutation grade: 1.00 (4/4 killed, threshold 0.75, required ok)" in out
    out2 = ss("mutate", "M90.1", rc=0).out
    assert "mutation grade: 1.00" in out2  # the full grade is cached by test hash
    cache = json.loads((ss.learner / ".ol/cache/mutation.json").read_text())
    assert len(cache["mutants"]) == 4 and len(cache["grades"]) == 1

    out = ss("check", "M90.1", rc=0).out
    assert "mutation grade (full, cached): 1.00" in out and "PASS M90.1" in out
    v = ss.verdicts("M90.1")[-1]
    assert v["mutation"]["score"] == 1.0 and v["mutation"]["required_ok"] is True


def test_weak_tests_survivors_sampling_and_reveal(ss):
    ss.add_extras("mutation")
    ss.init()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    _tests(ss, WEAK)
    out = ss("mutate", "M90.1", rc=1).out
    # 3 of 4 killed meets 0.75, but the required semantic mutant survives,
    # and before a pass it shows only its pitfall, never the planted bug.
    assert "0.75 (3/4 killed, threshold 0.75, required MISSED)" in out
    assert "survived s01 (required): a planted bug from Pitfall 1 survived" in out
    assert "in place" not in out
    # ol check: the R3 journal is missing, and the cached full grade fails.
    out = ss("check", "M90.1", rc=1).out
    assert "TDD" in out and "no `ol tdd green M90.1`" in out
    assert "mutation grade (full, cached): 0.75" in out
    out = ss("mutate", "M90.1", "--reveal-survivors", rc=1).out
    assert (
        "survived s01 (required): scales the caller's list in place (python/tinyllm/demo/scale.py:6)"
        in out
    )
    events = [
        json.loads(x)
        for x in (ss.learner / ".ol/verdicts.jsonl").read_text().splitlines()
    ]
    assert {
        "id": "M90.1",
        "event": "spoiled",
        "via": "reveal-survivors",
    }.items() <= next(e for e in events if e.get("event") == "spoiled").items()

    # A changed test file has no cached grade: ol check samples (all 4 here,
    # the manifest is smaller than the sample) and reports an estimate.
    _tests(ss, WEAK + "\n\ndef test_len():\n    assert len(scale([1.0], 2.0)) == 1\n")
    out = ss("check", "M90.1", rc=1).out
    assert "mutation grade: 0.75" in out


def test_rejects_a_correct_implementation_and_black_box_rule(ss):
    ss.add_extras("mutation")
    ss.init()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    _tests(
        ss, STRONG + "\n\ndef test_wrong():\n    assert scale([1.0], 2.0) == [3.0]\n"
    )
    out = ss("mutate", "M90.1", rc=1).out
    assert "your test rejects a correct implementation: test_wrong" in out
    _tests(ss, STRONG.replace("import scale", "import scale, _helper"))
    out = ss("mutate", "M90.1", rc=1).out
    assert (
        "black-box rule" in out
        and "tinyllm.demo.scale._helper, which is not in its contract" in out
    )


def test_verify_check_6_reaches_thresholds_in_four_languages(ss):
    ss.add_extras("mutation")
    out = ss("verify", "course", "M90.1", "rt.91", "ds.90", "dur.90", timeout=600).out
    lines = [x for x in out.splitlines() if " 6 " in x]
    assert sum("mutants apply and are killed" in x for x in lines) == 4, out
    reached = [
        x for x in lines if "reference learner tests reach the threshold: 1.00" in x
    ]
    assert len(reached) == 4, out
    # The kill cache: a second run reruns no course-test mutant.
    out = ss("verify", "course", "rt.91", timeout=600).out
    assert "3 mutants apply and are killed (3 from the kill cache)" in out
    # A reference learner suite that misses the required mutant fails check 6.
    t = ss.course / "ref/learner-tests/rt.91/c/tests/rt-91/test_mine.c"
    t.write_text(
        t.read_text()
        .replace("ol_alloc_fail_after(1);", "ol_alloc_fail_after(-1);")
        .replace("TL_ENOMEM", "TL_OK);\n    tl_demo_buf_destroy(b")
    )
    out = ss("verify", "course", "rt.91", timeout=600).out
    assert (
        "reference learner tests miss the threshold" in out and "survivors: s01" in out
    )
