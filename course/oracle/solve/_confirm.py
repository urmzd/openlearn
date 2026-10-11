"""Shared helper for the solve-key oracles (maintainer only, DESIGN 3.2 `oracle/`).

Each `course/oracle/solve/<S-ID>.py` recomputes every auto-checked answer of
`course/solve/<S-ID>/key.toml` by an independent route (enumeration, brute
force, numpy in the stated precision, SymPy differentiation of the actual
function) and hands the results to `confirm`, which:

  1. runs each recomputed value through the same checker `ol check` uses
     (`olcourse.solve_worker.check_one`), so it must pass the key, and
  2. fails when an auto-checked question has no recomputed value, so no key
     entry rests on the author's hand arithmetic alone.

Run from the worktree root:

    uv run --project course/harness python course/oracle/solve/S-M05.py
"""

from __future__ import annotations

import sys
from pathlib import Path

COURSE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COURSE / "harness" / "src"))

from olcourse.solve import load_key  # noqa: E402
from olcourse.solve_worker import check_one  # noqa: E402


def confirm(sid: str, computed: dict[str, str]) -> int:
    qs = load_key(COURSE, sid)
    auto = {q.qid: q for q in qs if q.type != "proof"}
    bad = 0
    for qid, q in auto.items():
        if qid not in computed:
            print(f"MISSING {sid} {qid}: no independent value")
            bad += 1
            continue
        try:
            ok, msg = check_one(q.spec, computed[qid])
        except Exception as e:  # an unparseable recomputed value is a failure too
            ok, msg = False, f"{type(e).__name__}: {e}"
        if not ok:
            print(f"FAIL    {sid} {qid}: recomputed {computed[qid]!r} ({msg})")
            bad += 1
    extra = sorted(set(computed) - set(auto))
    for qid in extra:
        print(f"EXTRA   {sid} {qid}: not an auto-checked question in the key")
        bad += 1
    n = len(auto)
    print(f"{sid}: {n - bad} of {n} auto-checked answers confirmed independently")
    return 1 if bad else 0
