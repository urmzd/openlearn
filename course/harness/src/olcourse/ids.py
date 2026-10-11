"""Course id grammar (DESIGN 3.5) and range expansion (DESIGN 4.0)."""

from __future__ import annotations

import re

AREAS = "lang|ds|rt|data|dur|gw|ag|load|dep|obs|ops|craft|ethics|review|field|iv"

# Kept byte-for-byte equal to COURSE_ID_RE in practice/bin/ol; a harness test
# asserts the two agree.
COURSE_ID_RE = (
    r"^(M[0-9]{2}\.[0-9]{1,2}|L[0-9]{1,2}\.[0-9]|C[12]|S-M[0-9]{2}[a-z]?|"
    r"(" + AREAS + r")\.[0-9]{2}|sq\.[a-z0-9-]+|MS-[A-Za-z0-9-]+)(\+cuda)?$"
)
_ID = re.compile(COURSE_ID_RE)

MODULE_KINDS = ("build", "solve", "proof", "practice", "drill", "side")
LANGS = ("python", "c", "rust", "go", "cuda", "ops", "docs", "none")
CODE_LANGS = ("python", "c", "rust", "go")
CI_TIERS = ("pr", "nightly", "local")
TEST_KINDS = (
    "unit",
    "boundary",
    "property",
    "statistical",
    "differential",
    "golden",
    "gradcheck",
    "learning",
    "conformance",
    "fault",
    "regression",
    "bench",
    "eval",
    "smoke",
)


def is_course_id(s: str) -> bool:
    return bool(_ID.match(s or ""))


def is_module_id(s: str) -> bool:
    """A course id that names a module (milestones are not modules)."""
    return is_course_id(s) and not s.startswith("MS-")


def underscore(mid: str) -> str:
    """Go and Rust test names: `dur.06` -> `dur_06`, `L10.2` -> `l10_2`."""
    return mid.replace(".", "_").replace("-", "_").replace("+", "_").lower()


_RANGE = re.compile(r"^\s*(\S+)\s+to\s+(\S+)\s*$")
_TAIL = re.compile(r"^(.*?)([0-9]+)$")


def expand(values: list[str]) -> list[str]:
    """Expand `X to Y` id ranges: `ds.01 to ds.03` -> ds.01, ds.02, ds.03."""
    out: list[str] = []
    for v in values:
        m = _RANGE.match(v)
        if not m:
            out.append(v)
            continue
        a, b = m.group(1), m.group(2)
        ma, mb = _TAIL.match(a), _TAIL.match(b)
        if not (ma and mb and ma.group(1) == mb.group(1)):
            raise ValueError(f"bad id range {v!r}: both ends need the same prefix")
        width = len(ma.group(2)) if ma.group(2).startswith("0") else 0
        lo, hi = int(ma.group(2)), int(mb.group(2))
        if hi < lo:
            raise ValueError(f"bad id range {v!r}: end before start")
        out.extend(f"{ma.group(1)}{str(i).zfill(width)}" for i in range(lo, hi + 1))
    return out
