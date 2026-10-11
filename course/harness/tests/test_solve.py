"""The solve checker (DESIGN 5.5): every answer type, input safety, parts,
canaries, and the no-leak feedback rule."""

from __future__ import annotations

import pytest

from olcourse import solve


def check(spec: dict, answer: str) -> dict:
    return solve.check_items([solve.Question("q", spec)], {"q": answer})["q"]


def ok(spec: dict, answer: str) -> bool:
    return bool(check(spec, answer)["ok"])


X = {"x": "real"}


@pytest.mark.parametrize(
    "answer, want",
    [
        ("2*x*cos(x^2)", True),
        ("2x cos(x^2)", True),  # implicit multiplication
        ("cos(x**2)*x*2", True),
        ("2*x*cos(x)", False),  # the canary
        ("x", False),
    ],
)
def test_expr_symbolic_or_numeric(answer, want):
    spec = {"type": "expr", "expect": "2*x*cos(x**2)", "vars": X}
    assert ok(spec, answer) is want


def test_expr_numeric_feedback_names_a_point_not_the_answer():
    spec = {"type": "expr", "expect": "2*x*cos(x**2)", "vars": X, "check": "numeric"}
    r = check(spec, "2*x*cos(x)")
    assert not r["ok"] and r["msg"].startswith("differs at x=")
    assert "cos(x**2)" not in r["msg"] and "x^2" not in r["msg"]


def test_expr_forbid_walks_the_tree():
    spec = {
        "type": "expr",
        "expect": "x**2/2",
        "vars": X,
        "forbid": ["Sum"],
        "check": "either",
    }
    r = check(spec, "Sum(x, (x, 0, 1)) + x^2/2 - 1/2")
    assert not r["ok"] and "Sum" in r["msg"]


def test_expr_logs_need_assumptions():
    spec = {
        "type": "expr",
        "expect": "log(x*y)",
        "vars": {"x": "positive", "y": "positive"},
    }
    assert ok(spec, "log(x) + log(y)")


def test_equation_up_to_a_nonzero_constant():
    spec = {
        "type": "equation",
        "expect": "y = 2*x + 1",
        "vars": {"x": "real", "y": "real"},
    }
    assert ok(spec, "2*y = 4*x + 2")
    assert ok(spec, "y - 2*x - 1 = 0")
    assert not ok(spec, "y = 2*x")
    assert not ok(spec, "x = x")  # an identity is not this equation


def test_number_exact_rejects_decimals():
    spec = {"type": "number", "expect": "1/3", "exact": True}
    assert ok(spec, "1/3") and ok(spec, "2/6")
    r = check(spec, "0.3333333333")
    assert not r["ok"] and "exact" in r["msg"]
    assert ok({"type": "number", "expect": "1/3", "rtol": 1e-6}, "0.3333333")


def test_interval_union_and_endpoints_as_expressions():
    spec = {"type": "interval", "expect": "[1, 3) U (5, oo)"}
    assert ok(spec, "[1, 3) U (5, oo)")
    assert ok(spec, "(5, oo) U [1, 3)")
    assert not ok(spec, "[1, 3] U (5, oo)")
    assert ok(
        {"type": "interval", "expect": "[-sqrt(2), sqrt(2)]"}, "[-2^(1/2), 2^(1/2)]"
    )
    assert ok({"type": "interval", "expect": "EmptySet"}, "{}")


def test_set_matches_elements_one_to_one():
    spec = {"type": "set", "expect": "{1, -1, sqrt(2)}"}
    assert ok(spec, "{2^(1/2), -1, 1}")
    assert not ok(spec, "{1, -1}")
    assert not ok(spec, "{1, 1, sqrt(2)}")


def test_matrix_and_vector_equivalences():
    v = {"type": "vector", "expect": "[[1],[1]]", "up_to": "scalar"}
    assert ok(v, "[[2],[2]]") and ok(v, "[-3, -3]") and not ok(v, "[[1],[0]]")
    assert not ok(v, "[[0],[0]]")  # the zero vector is not an eigenvector
    m = {"type": "matrix", "expect": "[[1, 2], [3, 4]]"}
    assert ok(m, "[[1, 2], [3, 4]]") and not ok(m, "[[1, 3], [2, 4]]")
    cs = {"type": "matrix", "expect": "[[1, 1], [1, -1]]", "up_to": "column_sign"}
    assert ok(cs, "[[-1, 1], [-1, -1]]") and not ok(cs, "[[2, 1], [2, -1]]")
    perm = {"type": "matrix", "expect": "[[1, 0], [0, 2]]", "up_to": "permutation"}
    assert ok(perm, "[[0, 1], [2, 0]]")


def test_basis_is_span_equality():
    spec = {"type": "basis", "expect": "[[1, 0, 1], [0, 1, 1]]"}
    assert ok(spec, "[[1, 1, 2], [1, -1, 0]]")
    assert not ok(spec, "[[1, 0, 0], [0, 1, 0]]")
    r = check(spec, "[[1, 0, 1], [2, 0, 2]]")
    assert not r["ok"] and "dependent" in r["msg"]


def test_bool_and_choice_are_exact_case_insensitive():
    assert ok({"type": "bool", "expect": "true"}, "True")
    assert not ok({"type": "bool", "expect": "true"}, "no")
    assert ok({"type": "choice", "expect": "b"}, " B ")


@pytest.mark.parametrize(
    "answer, why",
    [
        ("__import__('os')", "not allowed"),
        ("x.real", "not allowed"),
        ("eval(x)", "unknown name"),
        ("y + 1", "unknown name"),
        ("x;1", "not allowed"),
        ("'x'", "not allowed"),
    ],
)
def test_input_safety_rejects_before_sympy(answer, why):
    r = check({"type": "expr", "expect": "x", "vars": X}, answer)
    assert not r["ok"] and why in r["msg"] and r.get("input_error")


def test_screen_allows_numbers_with_exponents():
    solve.screen("1e-3*x + 2.5E+2", ["x"])  # E+2 parses as a number exponent
    with pytest.raises(solve.SolveInputError):
        solve.screen("2e", [])


def test_a_hanging_answer_times_out(monkeypatch):
    # A huge power towers SymPy; the per-item alarm (5 s) stops it.
    import time

    t = time.monotonic()
    r = check(
        {"type": "expr", "expect": "x", "vars": X, "check": "symbolic"},
        "(x+1)^(10^9) - x",
    )
    assert not r["ok"] and time.monotonic() - t < 20


def test_lettered_parts_and_rejects_and_problem_names(tmp_path):
    course = tmp_path / "course"
    d = course / "solve" / "S-M99a"
    d.mkdir(parents=True)
    (d / "key.toml").write_text(
        """
[q1]
type = "number"
expect = "4"
[[q1.reject]]
answer = "5"

[q2.a]
type = "expr"
expect = "x^2"
vars = { x = "real" }
[q2.b]
type = "bool"
expect = "false"
[[q2.b.reject]]
answer = "true"

[q3]
type = "proof"
rubric = ["States the claim"]
"""
    )
    (d / "problems.md").write_text(
        "q1. Two plus two.\nq2. (a) square. (b) bool.\nq3. Prove it.\n"
    )
    qs = solve.load_key(course, "S-M99a")
    assert [q.qid for q in qs] == ["q1", "q2.a", "q2.b", "q3"]
    assert solve.verify_key(course, "S-M99a") == []
    # a canary that passes is an error
    (d / "key.toml").write_text(
        (d / "key.toml").read_text().replace('answer = "5"', 'answer = "2+2"')
    )
    errs = solve.verify_key(course, "S-M99a")
    assert any("reject canary '2+2' passes" in e for e in errs)
    # answers may write parts as [q2a]
    learner = tmp_path / "learner"
    (learner / "solve").mkdir(parents=True)
    (learner / "solve" / "S-M99a.toml").write_text(
        '[q1]\nanswer="4"\n[q2a]\nanswer="x*x"\n[q2.b]\nanswer="False"\n'
    )
    res = {
        r.qid: r.status
        for r in solve.grade(course, learner, "S-M99a", say=lambda *_: None)
    }
    assert res == {"q1": "pass", "q2.a": "pass", "q2.b": "pass", "q3": "missing"}
