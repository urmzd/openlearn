"""The SymPy side of the solve checker (DESIGN 5.5).

Runs in a subprocess (`python -m olcourse.solve_worker`) so a pathological
answer can neither hang nor crash `ol check`. It reads one JSON document on
stdin:

    {"items": [{"qid": "q1", "spec": {...key table...}, "answer": "..."}, ...]}

and writes one JSON line per item:

    {"qid": "q1", "ok": true, "msg": ""}            msg never shows the expected answer

Each item gets its own 5 s alarm; the parent adds an outer timeout.
Every string is screened by `olcourse.solve.screen` in the parent first; the
worker screens again (defence in depth) before anything reaches `parse_expr`.
"""

from __future__ import annotations

import json
import math
import random
import signal
import sys

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from .solve import FUNCS, SolveInputError, screen

TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication)
ITEM_TIMEOUT_S = 5


class Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise Timeout()


# ---------------------------------------------------------------------------
# parsing


ASSUME = {
    "real": {"real": True},
    "positive": {"positive": True},
    "negative": {"negative": True},
    "nonnegative": {"nonnegative": True},
    "integer": {"integer": True},
    "natural": {"integer": True, "nonnegative": True},
    "complex": {},
}


def _symbols(spec: dict) -> dict:
    out = {}
    for name, kind in (spec.get("vars") or {}).items():
        out[name] = sp.Symbol(name, **ASSUME.get(str(kind), {"real": True}))
    return out


def _globals() -> dict:
    g = {
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
        "Symbol": sp.Symbol,
        "__builtins__": {},
    }
    table = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "E": sp.E,
        "oo": sp.oo,
        "I": sp.I,
        "Abs": sp.Abs,
        "floor": sp.floor,
        "ceiling": sp.ceiling,
        "binomial": sp.binomial,
        "factorial": sp.factorial,
        "Sum": sp.Sum,
    }
    assert set(table) == set(FUNCS)
    g.update(table)
    return g


def parse(text: str, spec: dict, extra: tuple[str, ...] = ()):
    screen(text, list((spec.get("vars") or {}).keys()) + list(extra))
    syms = _symbols(spec)
    try:
        return parse_expr(
            text,
            local_dict=dict(syms),
            global_dict=_globals(),
            transformations=TRANSFORMS,
            evaluate=True,
        )
    except Timeout:
        raise
    except Exception as e:  # SyntaxError, TypeError, TokenError, ...
        raise SolveInputError(f"cannot parse {text!r}: {type(e).__name__}") from None


# ---------------------------------------------------------------------------
# equality


def _has_float(e) -> bool:
    return any(isinstance(a, sp.Float) for a in sp.preorder_traversal(e))


def _forbidden(e, forbid: list[str]) -> str | None:
    if not forbid:
        return None
    for node in sp.preorder_traversal(e):
        names = {
            type(node).__name__,
            getattr(getattr(node, "func", None), "__name__", ""),
        }
        for name in forbid:
            if name in names:
                return name
    return None


def _symbolic_equal(a, b) -> bool:
    d = a - b
    for f in (
        lambda x: x,
        sp.expand,
        sp.simplify,
        sp.trigsimp,
        lambda x: sp.logcombine(x, force=True),
        lambda x: sp.simplify(sp.expand_log(x, force=True)),
    ):
        try:
            if f(d) == 0:
                return True
        except Timeout:
            raise
        except Exception:
            continue
    return False


def _domain(spec: dict, free) -> dict:
    dom = spec.get("domain") or {}
    out = {}
    for s in free:
        lo, hi = dom.get(s.name, [-3.0, 3.0])
        out[s] = (float(lo), float(hi))
    return out


def _value(e, subs: dict) -> complex | None:
    try:
        v = complex(sp.N(e.subs(subs), 30))
    except Timeout:
        raise
    except Exception:
        return None
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    return v


def _numeric_equal(a, b, spec: dict) -> tuple[bool, str]:
    free = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    n = int(spec.get("samples", 32))
    rtol = float(spec.get("rtol", 1e-9))
    atol = float(spec.get("atol", 1e-12))
    rng = random.Random(int(spec.get("seed", 0)))
    dom = _domain(spec, free)
    valid = 0
    for _ in range(n):
        point = {}
        for s in free:
            lo, hi = dom[s]
            x = rng.uniform(lo, hi)
            if s.is_integer:
                x = round(x)
            elif s.is_positive and x <= 0:
                x = abs(x) + 1e-3
            point[s] = x
        va, vb = _value(a, point), _value(b, point)
        if va is None or vb is None:
            continue
        valid += 1
        if not abs(va - vb) <= atol + rtol * abs(vb):
            where = ", ".join(f"{s.name}={point[s]:.2f}" for s in free)
            return False, f"differs at {where}" if where else "differs"
    if valid < 0.75 * n:
        return False, f"only {valid} of {n} sample points were valid (want 75%)"
    return True, ""


def expr_equal(a, b, spec: dict) -> tuple[bool, str]:
    mode = spec.get("check", "either")
    if mode not in ("symbolic", "numeric", "either", "both"):
        raise SolveInputError(
            f"key: check {mode!r} is not symbolic|numeric|either|both"
        )
    sym = _symbolic_equal(a, b) if mode in ("symbolic", "either", "both") else None
    if mode == "symbolic":
        return (True, "") if sym else (False, "not equivalent")
    if mode == "either" and sym:
        return True, ""
    num, why = _numeric_equal(a, b, spec)
    if mode == "both" and not sym:
        return False, "not equivalent symbolically"
    return num, why


def check_expr(spec, ans: str) -> tuple[bool, str]:
    want = parse(str(spec["expect"]), spec)
    got = parse(ans, spec)
    bad = _forbidden(got, list(spec.get("forbid", [])))
    if bad:
        return False, f"uses {bad}; give a closed form"
    return expr_equal(got, want, spec)


def _sides(text: str, spec: dict):
    if text.count("=") != 1 or any(op in text for op in ("<=", ">=", "==")):
        raise SolveInputError("an equation needs exactly one `=`")
    lhs, rhs = text.split("=")
    return parse(lhs, spec), parse(rhs, spec)


def check_equation(spec, ans: str) -> tuple[bool, str]:
    la, ra = _sides(ans, spec)
    lb, rb = _sides(str(spec["expect"]), spec)
    num, den = sp.simplify(la - ra), sp.simplify(lb - rb)
    if num == 0:
        return False, "the equation is an identity"
    ratio = sp.simplify(num / den)
    if ratio.free_symbols == set() and ratio != 0 and ratio.is_finite:
        return True, ""
    # numeric fallback: the ratio is the same nonzero constant at every sample
    free = sorted(num.free_symbols | den.free_symbols, key=lambda s: s.name)
    rng = random.Random(0)
    vals = []
    dom = _domain(spec, free)
    for _ in range(int(spec.get("samples", 32))):
        pt = {s: rng.uniform(*dom[s]) for s in free}
        v = _value(ratio, pt)
        if v is not None:
            vals.append(v)
    if (
        len(vals) >= 3
        and abs(vals[0]) > 1e-12
        and all(abs(v - vals[0]) <= 1e-9 * abs(vals[0]) for v in vals)
    ):
        return True, ""
    return False, "not the same equation (the sides are not proportional)"


def check_number(spec, ans: str) -> tuple[bool, str]:
    want = parse(str(spec["expect"]), spec)
    got = parse(ans, spec)
    if got.free_symbols:
        return False, "a number cannot contain variables"
    if spec.get("exact"):
        if _has_float(got):
            return False, "give an exact value (no decimals)"
        if sp.simplify(got - want) == 0 or sp.nsimplify(got) == sp.nsimplify(want):
            return True, ""
        return False, "wrong value"
    rtol = float(spec.get("rtol", 1e-9))
    atol = float(spec.get("atol", 1e-12))
    a, b = complex(sp.N(got, 30)), complex(sp.N(want, 30))
    return (True, "") if abs(a - b) <= atol + rtol * abs(b) else (False, "wrong value")


# -- intervals and sets ------------------------------------------------------


def _split_top(text: str, sep: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out]


def _interval(text: str, spec: dict):
    t = text.strip()
    if t in ("{}", "EmptySet", "empty"):
        return sp.S.EmptySet
    if t in ("R", "Reals"):
        return sp.S.Reals
    if t.startswith("{") and t.endswith("}"):
        return sp.FiniteSet(
            *[parse(x, spec, ("oo",)) for x in _split_top(t[1:-1], ",")]
        )
    if len(t) < 2 or t[0] not in "[(" or t[-1] not in "])":
        raise SolveInputError(f"{t!r} is not an interval like [a, b) or (a, oo)")
    parts = _split_top(t[1:-1], ",")
    if len(parts) != 2:
        raise SolveInputError(f"{t!r}: an interval has two endpoints")
    a, b = (parse(x, spec, ("oo",)) for x in parts)
    return sp.Interval(a, b, left_open=t[0] == "(", right_open=t[-1] == ")")


def _union(text: str, spec: dict):
    pieces = [p for p in _split_union(text)]
    sets = [_interval(p, spec) for p in pieces]
    return sp.Union(*sets) if len(sets) > 1 else sets[0]


def _split_union(text: str) -> list[str]:
    # " U " (or "u", "|") between top-level pieces
    out, depth, cur, i = [], 0, "", 0
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if (
            depth == 0
            and ch in "Uu|"
            and (i == 0 or not text[i - 1].isalnum())
            and (i + 1 >= len(text) or not text[i + 1].isalnum())
        ):
            out.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    out.append(cur)
    return [x.strip() for x in out if x.strip()]


def check_interval(spec, ans: str) -> tuple[bool, str]:
    screen(
        ans,
        list((spec.get("vars") or {}).keys())
        + ["oo", "U", "u", "R", "Reals", "EmptySet", "empty"],
    )
    want = _union(str(spec["expect"]), spec)
    got = _union(ans, spec)
    if got == want:
        return True, ""
    try:
        if (
            sp.simplify(sp.Complement(got, want)) == sp.S.EmptySet
            and sp.simplify(sp.Complement(want, got)) == sp.S.EmptySet
        ):
            return True, ""
    except Exception:
        pass
    return False, "not the same set of numbers"


def _elements(text: str, spec: dict) -> list:
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        raise SolveInputError("a set is written {a, b, c}")
    inner = t[1:-1].strip()
    return [parse(x, spec) for x in _split_top(inner, ",")] if inner else []


def _match(xs: list, ys: list, eq) -> bool:
    if not xs:
        return not ys
    head, rest = xs[0], xs[1:]
    for j, y in enumerate(ys):
        if eq(head, y) and _match(rest, ys[:j] + ys[j + 1 :], eq):
            return True
    return False


def check_set(spec, ans: str) -> tuple[bool, str]:
    want = _elements(str(spec["expect"]), spec)
    got = _elements(ans, spec)
    if len(got) != len(want):
        return False, f"{len(got)} element(s); the set has a different size"
    if not _match(got, want, lambda a, b: expr_equal(a, b, spec)[0]):
        return False, "the elements differ"
    return True, ""


# -- matrices, vectors, bases --------------------------------------------------


def _matrix(text: str, spec: dict):
    t = text.strip()
    if not (t.startswith("[") and t.endswith("]")):
        raise SolveInputError("a matrix is written [[a, b], [c, d]] (rows)")
    rows = _split_top(t[1:-1], ",")
    if rows and rows[0].startswith("["):
        data = []
        for r in rows:
            if not (r.startswith("[") and r.endswith("]")):
                raise SolveInputError("every row is a [ ... ] list")
            data.append([parse(x, spec) for x in _split_top(r[1:-1], ",")])
        if len({len(r) for r in data}) > 1:
            raise SolveInputError("rows have different lengths")
        return sp.Matrix(data)
    return sp.Matrix([[parse(x, spec)] for x in rows])  # a flat list is a column


def _all_equal(a, b, spec) -> bool:
    return a.shape == b.shape and all(
        expr_equal(x, y, spec)[0] for x, y in zip(list(a), list(b))
    )


def _scalar_multiple(a, b, spec, signs_only: bool) -> bool:
    for x, y in zip(list(a), list(b)):
        if sp.simplify(y) != 0:
            c = sp.simplify(x / y)
            if c == 0:
                return False
            if signs_only and c not in (1, -1):
                return False
            return _all_equal(a, b * c, spec)
    return _all_equal(a, b, spec)


def check_matrix(spec, ans: str) -> tuple[bool, str]:
    want = _matrix(str(spec["expect"]), spec)
    got = _matrix(ans, spec)
    if got.shape != want.shape:
        return False, f"shape {got.shape[0]}x{got.shape[1]} is wrong"
    up = spec.get("up_to", "none")
    if up == "none":
        ok = _all_equal(got, want, spec)
    elif up == "scalar":
        ok = _scalar_multiple(got, want, spec, signs_only=False)
    elif up == "sign":
        ok = _scalar_multiple(got, want, spec, signs_only=True)
    elif up == "column_sign":
        ok = all(
            _scalar_multiple(got[:, j], want[:, j], spec, signs_only=True)
            for j in range(want.shape[1])
        )
    elif up == "permutation":
        cols_g = [got[:, j] for j in range(got.shape[1])]
        cols_w = [want[:, j] for j in range(want.shape[1])]
        ok = _match(cols_g, cols_w, lambda a, b: _all_equal(a, b, spec))
    else:
        raise SolveInputError(
            f"key: up_to {up!r} is not none|scalar|sign|column_sign|permutation"
        )
    return (True, "") if ok else (False, "the entries differ")


def _vectors(text: str, spec: dict):
    """A basis is a list of vectors: [[1, 0, 1], [0, 1, 1]]; columns of the matrix."""
    m = _matrix(text, spec)
    t = text.strip()
    if t.startswith("[[") and _split_top(t[1:-1], ",")[0].startswith("["):
        return m.T  # each inner list is one vector
    return m


def check_basis(spec, ans: str) -> tuple[bool, str]:
    a = _vectors(ans, spec)
    b = _vectors(str(spec["expect"]), spec)
    if a.shape[0] != b.shape[0]:
        return False, "the vectors have the wrong length"
    ra, rb = a.rank(), b.rank()
    if ra != a.shape[1]:
        return False, "the vectors are linearly dependent (not a basis)"
    if ra == rb == a.row_join(b).rank():
        return True, ""
    return False, "they span a different space"


def check_bool(spec, ans: str) -> tuple[bool, str]:
    norm = {
        "true": True,
        "yes": True,
        "t": True,
        "y": True,
        "false": False,
        "no": False,
        "f": False,
        "n": False,
    }
    a, b = ans.strip().lower(), str(spec["expect"]).strip().lower()
    if a not in norm:
        return False, "answer true or false"
    return (True, "") if norm[a] == norm.get(b, b) else (False, "wrong")


def check_choice(spec, ans: str) -> tuple[bool, str]:
    return (
        (True, "")
        if ans.strip().lower() == str(spec["expect"]).strip().lower()
        else (False, "wrong choice")
    )


CHECKERS = {
    "expr": check_expr,
    "equation": check_equation,
    "number": check_number,
    "interval": check_interval,
    "set": check_set,
    "matrix": check_matrix,
    "vector": check_matrix,
    "basis": check_basis,
    "bool": check_bool,
    "choice": check_choice,
}


def check_one(spec: dict, answer: str) -> tuple[bool, str]:
    typ = spec.get("type", "expr")
    fn = CHECKERS.get(typ)
    if fn is None:
        raise SolveInputError(f"key: unknown type {typ!r}")
    return fn(spec, answer)


def main() -> int:
    doc = json.loads(sys.stdin.read())
    signal.signal(signal.SIGALRM, _alarm)
    for item in doc.get("items", []):
        qid = item.get("qid", "?")
        signal.alarm(ITEM_TIMEOUT_S)
        try:
            ok, msg = check_one(item["spec"], str(item["answer"]))
            res = {"qid": qid, "ok": bool(ok), "msg": msg}
        except Timeout:
            res = {
                "qid": qid,
                "ok": False,
                "msg": f"checking took over {ITEM_TIMEOUT_S}s",
            }
        except SolveInputError as e:
            res = {"qid": qid, "ok": False, "msg": str(e), "input_error": True}
        except Exception as e:  # never let one answer take the batch down
            res = {"qid": qid, "ok": False, "msg": f"cannot check: {type(e).__name__}"}
        finally:
            signal.alarm(0)
        print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
