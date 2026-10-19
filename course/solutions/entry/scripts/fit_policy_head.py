"""Fit the usage policy's classifier head on hash16 vectors (gw.08, L6.5).

PR CI runs no embedding model, so the local gateway's policy
(config/policy.local.yaml) scores requests with hash16, the deterministic
lexical embedding the Go side computes in cmd/internal/hashembed: every term
(runs of letters, digits, and underscore, lowercased) adds +1 or -1, the sign
from the top bit of its FNV-1a 64 hash, to coordinate hash mod 16. This script
embeds a labelled prompt set the same way, fits the head with L6.5's
fit_linear_head (IRLS on unit vectors), and exports it with
export_linear_head. It fails when a labelled prompt lands on the wrong side
of the threshold, so a head that does not separate its own red-team set
never ships.

    uv run --project python python scripts/fit_policy_head.py \
        --data config/policy-train.jsonl --out artifacts/models/hash16/heads/usage.json

Final stdout line: {"out", "n", "metrics"}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from tinyllm.obj.heads import (  # noqa: E402
    export_linear_head,
    fit_linear_head,
    head_metrics,
    head_probs,
)

DIM = 16
TERM = re.compile(r"[^\W]+")


def fnv1a64(b: bytes) -> int:
    h = 0xCBF29CE484222325
    for x in b:
        h ^= x
        h = (h * 0x100000001B3) & ((1 << 64) - 1)
    return h


def hash16(text: str) -> list[float]:
    v = [0.0] * DIM
    for t in TERM.findall(text):
        h = fnv1a64(t.lower().encode())
        v[h % DIM] += -1.0 if h >> 63 else 1.0
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--threshold", type=float, default=0.9)
    a = ap.parse_args()
    rows = [
        json.loads(line)
        for line in Path(a.data).read_text().splitlines()
        if line.strip()
    ]
    X = [hash16(r["text"]) for r in rows]
    y = [int(r["label"]) for r in rows]
    head = fit_linear_head(X, y, ["safe", "unsafe"], l2=a.l2, threshold=a.threshold)
    p = head_probs(head, X)[:, 1]
    wrong = [
        r["text"] for r, pi, yi in zip(rows, p, y) if (pi >= a.threshold) != (yi == 1)
    ]
    if wrong:
        print(f"fit_policy_head: the head misclassifies {wrong}", file=sys.stderr)
        return 1
    metrics = head_metrics(head, X, y)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    export_linear_head(head, "hash16", a.out, metrics=metrics)
    print(json.dumps({"out": a.out, "n": len(rows), "metrics": metrics}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
