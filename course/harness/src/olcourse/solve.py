"""The `solve` kind: SymPy-checked answers and self-graded proofs (DESIGN 5.5).

    course/solve/<S-ID>/problems.md     numbered problems q1..qN (in the chapter too)
    course/solve/<S-ID>/key.toml        typed expected answers, tolerances, reject canaries
    <learner>/solve/<S-ID>.toml         answers as ASCII math
    <learner>/solve/<S-ID>/q7.md        proofs

A question is a key table with a `type`. Lettered parts are sub-tables:
`[q3.a]` and `[q3.b]` are questions `q3.a` and `q3.b` (the answer file may
also write them `[q3a]`). A whole set is split into lettered sets by pass
(`S-M07a`, `S-M07b`): each is its own module and directory.

Types: expr equation number interval set matrix vector basis bool choice
proof. Feedback never shows the expected answer. Parsing and checking run in
a subprocess (`olcourse.solve_worker`) with a 5 s alarm per answer.

This module is stdlib only; SymPy is imported by the worker alone.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError

# The function whitelist of DESIGN 5.5 (also the worker's global namespace).
FUNCS = (
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "pi",
    "E",
    "oo",
    "I",
    "Abs",
    "floor",
    "ceiling",
    "binomial",
    "factorial",
    "Sum",
)
TYPES = (
    "expr",
    "equation",
    "number",
    "interval",
    "set",
    "matrix",
    "vector",
    "basis",
    "bool",
    "choice",
    "proof",
)
FREE_TEXT = ("bool", "choice")  # answers that are words, not math
ALLOWED = re.compile(r"^[A-Za-z0-9_+\-*/^().,\[\]{} =<>|]*$")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+")
MAX_LEN = 2000


class SolveInputError(Exception):
    """An answer the checker refuses to parse (shown to the learner)."""


def screen(text: str, names: list[str]) -> None:
    """Input safety (DESIGN 5.5): a fixed character set, no `__`, no `.` before
    a letter, and identifiers only from the declared vars and the whitelist."""
    if len(text) > MAX_LEN:
        raise SolveInputError(f"answer longer than {MAX_LEN} characters")
    if not ALLOWED.match(text):
        bad = sorted({c for c in text if not ALLOWED.match(c)})
        raise SolveInputError(f"character(s) not allowed: {''.join(bad)!r}")
    if "__" in text:
        raise SolveInputError("`__` is not allowed")
    if re.search(r"\.[A-Za-z]", text):
        raise SolveInputError("`.` followed by a letter is not allowed")
    allowed = set(FUNCS) | set(names)
    for m in TOKEN.finditer(text):
        tok = m.group(0)
        if tok[0].isalpha() or tok[0] == "_":
            if tok not in allowed:
                raise SolveInputError(
                    f"unknown name {tok!r} (variables: {', '.join(sorted(names)) or 'none'})"
                )


# ---------------------------------------------------------------------------
# keys and answers


@dataclass
class Question:
    qid: str  # q1, q3.a
    spec: dict
    rejects: list[str] = field(default_factory=list)

    @property
    def type(self) -> str:
        return str(self.spec.get("type", "expr"))


QUESTION_MARKERS = ("type", "expect", "rubric")
ANSWER_MARKERS = ("answer", "proof")


def _flatten(raw: dict, markers: tuple[str, ...], prefix: str = "") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        qid = f"{prefix}.{k}" if prefix else k
        if any(m in v for m in markers):
            out[qid] = v
        else:
            out.update(_flatten(v, markers, qid))
    return out


def solve_dir(course: Path, sid: str) -> Path:
    return course / "solve" / sid


def load_key(course: Path, sid: str) -> list[Question]:
    p = solve_dir(course, sid) / "key.toml"
    try:
        raw = tomllib.loads(p.read_text())
    except OSError:
        raise HarnessError(f"{sid}: no answer key at {p}") from None
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(f"{p}: {e}") from None
    qs = []
    for qid, spec in _flatten(raw, QUESTION_MARKERS).items():
        spec = dict(spec)
        rejects = [str(r.get("answer", "")) for r in spec.pop("reject", [])]
        typ = spec.get("type", "expr")
        if typ not in TYPES:
            raise HarnessError(
                f"{p} [{qid}]: type {typ!r} is not one of {', '.join(TYPES)}"
            )
        if typ != "proof" and "expect" not in spec:
            raise HarnessError(f"{p} [{qid}]: needs `expect`")
        qs.append(Question(qid, spec, rejects))
    if not qs:
        raise HarnessError(f"{p}: no questions (a question is a table with `type`)")
    return sorted(qs, key=lambda q: _natural(q.qid))


def _natural(s: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))


def answers_path(learner: Path, sid: str) -> Path:
    return learner / "solve" / f"{sid}.toml"


def load_answers(learner: Path, sid: str) -> dict[str, dict]:
    p = answers_path(learner, sid)
    if not p.is_file():
        return {}
    try:
        raw = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(f"{p}: {e}") from None
    flat = _flatten(raw, ANSWER_MARKERS)
    # accept `[q3a]` for part `q3.a`
    return {k: v for k, v in flat.items()} | {
        k.replace(".", ""): v for k, v in flat.items() if "." in k
    }


def answer_for(answers: dict[str, dict], qid: str) -> dict | None:
    return answers.get(qid) or answers.get(qid.replace(".", ""))


def template(qs: list[Question], sid: str) -> str:
    lines = [
        f"# Answers for {sid} (course/DESIGN.md 5.5). ASCII math: x^2, sqrt(x), pi, oo,",
        "# intervals [1, 3) U (5, oo), sets {1, 2}, matrices [[1, 2], [3, 4]] (rows),",
        "# bases as a list of vectors [[1, 0], [0, 1]]. Proofs go in the file named.",
        "",
    ]
    for q in qs:
        lines.append(f"[{q.qid}]")
        if q.type == "proof":
            lines.append(f'proof  = "{sid}/{q.qid}.md"')
        else:
            lines.append('answer = ""')
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the worker


def run_worker(items: list[dict], timeout: float | None = None) -> dict[str, dict]:
    """Check (qid, spec, answer) items in one subprocess; returns qid -> result."""
    if not items:
        return {}
    timeout = timeout or 10.0 + 6.0 * len(items)
    try:
        p = subprocess.run(
            [sys.executable, "-m", "olcourse.solve_worker"],
            input=json.dumps({"items": items}),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            it["qid"]: {
                "ok": False,
                "msg": f"the checker timed out after {timeout:.0f}s",
            }
            for it in items
        }
    out: dict[str, dict] = {}
    for line in p.stdout.splitlines():
        try:
            r = json.loads(line)
            out[r["qid"]] = r
        except (json.JSONDecodeError, KeyError):
            continue
    for it in items:
        if it["qid"] not in out:
            tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["no output"]
            out[it["qid"]] = {
                "ok": False,
                "msg": f"the checker crashed: {tail[0][:200]}",
            }
    return out


def check_items(qs: list[Question], answers: dict[str, str]) -> dict[str, dict]:
    """Screen in-process (cheap, catches most input errors), then run the worker."""
    items, out = [], {}
    for q in qs:
        if q.qid not in answers:
            continue
        a = answers[q.qid]
        if q.type not in FREE_TEXT:
            try:
                extra = list((q.spec.get("vars") or {}).keys())
                if q.type == "interval":
                    extra += ["oo", "U", "u", "R", "Reals", "EmptySet", "empty"]
                screen(a, extra)
            except SolveInputError as e:
                out[q.qid] = {"ok": False, "msg": str(e), "input_error": True}
                continue
        items.append({"qid": q.qid, "spec": q.spec, "answer": a})
    out.update(run_worker(items))
    return out


# ---------------------------------------------------------------------------
# grading a learner's set


@dataclass
class Result:
    qid: str
    status: str  # pass fail missing self self-skipped
    msg: str = ""


def proof_rubric(course: Path, q: Question) -> list[str]:
    from . import rubric

    lines = q.spec.get("rubric")
    if isinstance(lines, list) and lines:
        return [str(x) for x in lines]
    name = str(lines or "proof")
    return rubric.items(course, name)


def file_hash(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def grade(
    course: Path,
    learner: Path,
    sid: str,
    ask=None,
    recorded=None,
    say=print,
) -> list[Result]:
    """Grade every question. `ask(lines) -> list[bool] | None` self-grades a
    proof (None: not graded now); `recorded(qid, hash) -> bool | None` returns
    an earlier self-grade of the same proof file."""
    qs = load_key(course, sid)
    answers = load_answers(learner, sid)
    plain: dict[str, str] = {}
    results: dict[str, Result] = {}
    for q in qs:
        a = answer_for(answers, q.qid)
        if q.type == "proof":
            continue
        if not a or not str(a.get("answer", "")).strip():
            results[q.qid] = Result(q.qid, "missing", "no answer yet")
            continue
        plain[q.qid] = str(a["answer"])
    checked = check_items([q for q in qs if q.qid in plain], plain)
    for qid, r in checked.items():
        results[qid] = Result(qid, "pass" if r.get("ok") else "fail", r.get("msg", ""))
    for q in qs:
        if q.type != "proof":
            continue
        a = answer_for(answers, q.qid) or {}
        rel = str(a.get("proof") or f"{sid}/{q.qid}.md")
        p = learner / "solve" / rel
        if not p.is_file() or len(_body(p.read_text())) < 40:
            results[q.qid] = Result(q.qid, "missing", f"no proof in solve/{rel} yet")
            continue
        h = file_hash(p)
        prev = recorded(q.qid, h) if recorded else None
        if prev is not None:
            results[q.qid] = Result(
                q.qid,
                "self" if prev else "fail",
                "self-graded earlier (proof unchanged)"
                if prev
                else "your rubric said no",
            )
            continue
        lines = proof_rubric(course, q)
        say(f"  {q.qid}: self-grade solve/{rel} against the rubric (y/n each):")
        answers_yn = ask(q.qid, lines, h) if ask else None
        if answers_yn is None:
            results[q.qid] = Result(q.qid, "self-skipped", "rubric not answered here")
        elif all(answers_yn):
            results[q.qid] = Result(q.qid, "self", "all rubric lines: yes")
        else:
            no = [line for line, ok in zip(lines, answers_yn) if not ok]
            results[q.qid] = Result(
                q.qid, "fail", "rubric: no to " + "; ".join(no)[:200]
            )
    return [results[q.qid] for q in qs]


def _body(text: str) -> str:
    return "\n".join(
        x for x in text.splitlines() if not x.lstrip().startswith("#")
    ).strip()


# ---------------------------------------------------------------------------
# verify check 7


def verify_key(course: Path, sid: str) -> list[str]:
    """Every key parses, every `expect` passes against itself, every reject
    canary fails, and problems.md names every question."""
    errs: list[str] = []
    try:
        qs = load_key(course, sid)
    except HarnessError as e:
        return [str(e)]
    items, rejects = [], []
    for q in qs:
        if q.type == "proof":
            if not proof_rubric(course, q):
                errs.append(f"{sid} {q.qid}: a proof needs a rubric")
            continue
        items.append({"qid": q.qid, "spec": q.spec, "answer": str(q.spec["expect"])})
        for i, r in enumerate(q.rejects):
            rejects.append(
                {"qid": f"{q.qid}#reject{i + 1}", "spec": q.spec, "answer": r}
            )
    res = run_worker(items + rejects)
    for it in items:
        r = res[it["qid"]]
        if not r.get("ok"):
            errs.append(
                f"{sid} {it['qid']}: `expect` does not pass its own check ({r.get('msg')})"
            )
    for it in rejects:
        r = res[it["qid"]]
        if r.get("ok"):
            errs.append(f"{sid} {it['qid']}: reject canary {it['answer']!r} passes")
        elif r.get("input_error"):
            errs.append(
                f"{sid} {it['qid']}: reject canary {it['answer']!r} does not parse ({r.get('msg')}); "
                "a canary must be a plausible wrong answer"
            )
    probs = solve_dir(course, sid) / "problems.md"
    if not probs.is_file():
        errs.append(f"{sid}: {probs} is missing")
    else:
        text = probs.read_text()
        for q in qs:
            top = q.qid.split(".")[0]
            if not re.search(rf"\b{re.escape(top)}\b", text):
                errs.append(f"{sid}: problems.md never names {top}")
    return errs
