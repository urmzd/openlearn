"""Reference thresholds from 5 reference seeds (DESIGN 5.7, 5.11, 5.12).

A learned metric (a loss, a perplexity, an accuracy) has no exact expected
value: the reference itself scatters across seeds. So the bar is set from
the reference: run it over seeds 0..4, take the mean and the sample standard
deviation, and accept a learner run up to three standard deviations worse.

    course/fixtures/ref-thresholds.tsv
    # key	metric	direction	mode	mean	sd	n	threshold	values
    MS-C1/val loss	val_loss	max	smoke	2.31	0.012	5	2.346	2.30,2.31,...

`direction = max`: lower is better, threshold = mean + 3 sd (pass when
value <= threshold). `min`: higher is better, threshold = mean - 3 sd.
`mode` is `full` or `smoke` (a `--smoke` milestone run and a full run are
calibrated separately: C1's smoke config trains 200 steps).

Milestones write `metrics = { val_loss = "<= calibrated" }` in a
json-last-line step (key `<MS-ID>/<step name>`); course tests of KIND
learning call `_lib.thresholds.check(key, metric, value, direction)`.
Rows are written by `ol milestone <MS-ID> --record-thresholds` and
`ol verify course <ID> --record-thresholds`, never by hand.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import HarnessError

REL = "course/fixtures/ref-thresholds.tsv"
HEADER = "# key\tmetric\tdirection\tmode\tmean\tsd\tn\tthreshold\tvalues"
K_SD = 3.0
MIN_SEEDS = 5


@dataclass
class Row:
    key: str
    metric: str
    direction: str  # max | min
    mode: str  # full | smoke
    mean: float
    sd: float
    n: int
    threshold: float
    values: list[float]

    def line(self) -> str:
        return "\t".join(
            [
                self.key,
                self.metric,
                self.direction,
                self.mode,
                f"{self.mean:.10g}",
                f"{self.sd:.10g}",
                str(self.n),
                f"{self.threshold:.10g}",
                ",".join(f"{v:.10g}" for v in self.values),
            ]
        )


def path(course: Path) -> Path:
    return course / "fixtures" / "ref-thresholds.tsv"


def load(course: Path) -> dict[tuple[str, str, str], Row]:
    p = path(course)
    out: dict[tuple[str, str, str], Row] = {}
    if not p.is_file():
        return out
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) != 9:
            raise HarnessError(f"{p}:{i}: {len(c)} columns, want 9")
        r = Row(
            c[0],
            c[1],
            c[2],
            c[3],
            float(c[4]),
            float(c[5]),
            int(c[6]),
            float(c[7]),
            [float(x) for x in c[8].split(",") if x],
        )
        out[(r.key, r.metric, r.mode)] = r
    return out


def compute(
    key: str, metric: str, direction: str, mode: str, values: list[float]
) -> Row:
    if direction not in ("max", "min"):
        raise HarnessError(
            f"direction {direction!r}: max (lower is better) or min (higher is better)"
        )
    if len(values) < MIN_SEEDS:
        raise HarnessError(
            f"{key} {metric}: {len(values)} reference seed(s), want at least {MIN_SEEDS}"
        )
    if not all(math.isfinite(v) for v in values):
        raise HarnessError(
            f"{key} {metric}: a reference seed produced a non-finite value {values}"
        )
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    thr = mean + K_SD * sd if direction == "max" else mean - K_SD * sd
    return Row(key, metric, direction, mode, mean, sd, n, thr, list(values))


def lookup(course: Path, key: str, metric: str, mode: str = "full") -> Row:
    rows = load(course)
    r = rows.get((key, metric, mode))
    if r is None:
        raise HarnessError(
            f"no reference threshold for {key} {metric} ({mode}) in {REL}: "
            "a maintainer records it from 5 reference seeds (--record-thresholds)"
        )
    return r


def passes(r: Row, value: float) -> tuple[bool, str]:
    if not math.isfinite(value):
        return False, f"{value} is not finite"
    ok = value <= r.threshold if r.direction == "max" else value >= r.threshold
    op = ("<=" if ok else ">") if r.direction == "max" else (">=" if ok else "<")
    return (
        ok,
        f"{value:.6g} {op} {r.threshold:.6g} (reference mean {r.mean:.6g} {'+' if r.direction == 'max' else '-'} 3 x sd {r.sd:.3g}, {r.n} seeds)",
    )


def write(course: Path, new: list[Row]) -> Path:
    """Merge rows into the table (sorted, stable) and keep its MANIFEST row current."""
    rows = load(course)
    for r in new:
        rows[(r.key, r.metric, r.mode)] = r
    p = path(course)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEADER + "\n" + "".join(rows[k].line() + "\n" for k in sorted(rows)))
    man = course / "fixtures" / "MANIFEST.tsv"
    data = p.read_bytes()
    row = "\t".join(
        [
            REL,
            hashlib.sha256(data).hexdigest(),
            str(len(data)),
            "ss --record-thresholds",
            "-",
            "-",
            "Apache-2.0",
        ]
    )
    lines = (
        man.read_text().splitlines()
        if man.is_file()
        else [
            "# path\tsha256\tbytes\tgenerator\toracle_versions\tupstream@revision\tlicense"
        ]
    )
    lines = [x for x in lines if not x.startswith(REL + "\t")] + [row]
    man.write_text("\n".join(lines) + "\n")
    return p


def from_observations(lines: list[str], mode: str) -> list[Row]:
    """Course-test recording: JSON lines {key, metric, direction, value} over seeds."""
    groups: dict[tuple[str, str, str], list[float]] = {}
    for line in lines:
        if not line.strip():
            continue
        o = json.loads(line)
        groups.setdefault((o["key"], o["metric"], o["direction"]), []).append(
            float(o["value"])
        )
    return [compute(k, m, d, mode, v) for (k, m, d), v in sorted(groups.items())]
