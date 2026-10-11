"""ol parity (DESIGN 5.8): golden mode in four languages, live fuzz, pending
implementations, mismatches, and learner-sourced runs."""

import json


def test_reference_golden_and_fuzz_in_four_languages(ss):
    ss.add_extras("parity")
    out = ss("parity", "demo.sum", rc=0, timeout=600).out
    for impl in ("python", "c", "rust", "go"):
        assert f"pass     {impl}" in out, out
    assert "pending  future" in out and "M90.9 is not in the registry yet" in out
    out = ss("parity", "demo.sum", "--fuzz", "--cases", "40", rc=0, timeout=600).out
    assert "40 fuzz cases equal python" in out and out.count("pass ") >= 4


def test_mismatch_fails_with_the_case(ss):
    ss.add_extras("parity")
    g = ss.course / "fixtures/parity/demo.sum.json"
    doc = json.loads(g.read_text())
    doc["cases"][2]["output"] = 9.5
    g.write_text(json.dumps(doc))
    out = ss("parity", "demo.sum", rc=1, timeout=600).out
    assert "fail     c" in out and "case mixed: value 0: 9.25 vs 9.5" in out


def test_learner_implementations_run_once_they_pass(ss):
    ss.add_extras("parity")
    ss.init()
    out = ss("parity", "demo.sum", rc=0).out
    assert "pending" in out and "M90.1 has no pass of yours yet (todo)" in out
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    ss("check", "M90.1", rc=0)
    out = ss("parity", "demo.sum", rc=0).out
    assert "pass     python       learner" in out and "pending  c" in out
    v = ss.verdicts("parity:demo.sum")[-1]
    assert v["result"] == "pass" and v["impls"]["python"] == "pass"
