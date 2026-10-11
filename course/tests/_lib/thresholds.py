"""Reference thresholds for learning tests (DESIGN 5.11, 5.12, D35).

A learning test trains something small and checks a metric against a bar
set from the reference run over 5 seeds (mean + 3 sd for a loss, mean - 3 sd
for an accuracy). The bars live in $TINYLLM_FIXTURES/ref-thresholds.tsv,
written only by `ol verify course <ID> --record-thresholds`.

    from _lib.thresholds import check

    def test_nplm_learns():
        # WHY: ...   KIND: learning
        loss = train_and_eval(seed=int(os.environ.get("OL_SEED", 0)))
        check("L2.2/test_nplm_learns", "val_loss", loss, direction="max")

While recording (OL_RECORD_THRESHOLDS names a file), `check` appends the
observation there and asserts nothing.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


def _table() -> dict:
    p = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "ref-thresholds.tsv"
    out = {}
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                c = line.split("\t")
                out[(c[0], c[1], c[3])] = (
                    c[2],
                    float(c[4]),
                    float(c[5]),
                    int(c[6]),
                    float(c[7]),
                )
    return out


def check(
    key: str, metric: str, value: float, direction: str = "max", mode: str | None = None
) -> None:
    if direction not in ("max", "min"):
        raise ValueError(
            "direction is 'max' (lower is better) or 'min' (higher is better)"
        )
    rec = os.environ.get("OL_RECORD_THRESHOLDS")
    if rec:
        with open(rec, "a") as f:
            f.write(
                json.dumps(
                    {
                        "key": key,
                        "metric": metric,
                        "direction": direction,
                        "value": float(value),
                    }
                )
                + "\n"
            )
        return
    mode = mode or os.environ.get("OL_THRESHOLD_MODE", "full")
    row = _table().get((key, metric, mode))
    assert row is not None, (
        f"no reference threshold for {key} {metric} ({mode}) in ref-thresholds.tsv"
    )
    want_dir, mean, sd, n, thr = row
    assert want_dir == direction, (
        f"{key} {metric}: recorded as {want_dir}, checked as {direction}"
    )
    assert math.isfinite(value), f"{key} {metric}: {value} is not finite"
    if direction == "max":
        assert value <= thr, (
            f"{key} {metric} = {value:.6g} > {thr:.6g} (reference {mean:.6g} + 3 x {sd:.3g} over {n} seeds)"
        )
    else:
        assert value >= thr, (
            f"{key} {metric} = {value:.6g} < {thr:.6g} (reference {mean:.6g} - 3 x {sd:.3g} over {n} seeds)"
        )
