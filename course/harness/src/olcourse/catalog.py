"""The annotated test catalog (DESIGN 5.12): every course test carries a
header of comment lines right after (or right before) its definition:

    # WHY: ...        # KIND: boundary      # CATCHES: s03, m011      # CHAPTER: ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFS = {
    ".py": re.compile(r"^\s*(?:async\s+)?def\s+(test_\w+)\s*\("),
    ".c": re.compile(r"^\s*OL_TEST\s*\(\s*(\w+)\s*\)"),
    ".rs": re.compile(r"^\s*(?:pub\s+)?fn\s+(\w+)\s*\("),
    ".go": re.compile(r"^\s*func\s+(Test\w+)\s*\(\s*\w+\s+\*testing\.T\s*\)"),
}
KEY = re.compile(r"^(WHY|KIND|CATCHES|CHAPTER)\s*:\s*(.*)$")


def _comment(line: str) -> str | None:
    s = line.strip()
    if s.startswith("*") and not (s == "*" or s.startswith(("* ", "*/"))):
        return None  # a pointer dereference, not a block-comment line
    for lead in ("#", "//", "/*", "*"):
        if s.startswith(lead):
            body = s[len(lead) :]
            body = body[:-2] if body.endswith("*/") else body
            return body.strip()
    return None


@dataclass
class Test:
    name: str
    file: Path
    line: int
    lang: str
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def kinds(self) -> list[str]:
        return [
            k.strip()
            for k in re.split(r"[,\s]+", self.fields.get("KIND", ""))
            if k.strip()
        ]

    @property
    def catches(self) -> list[str]:
        return [
            k.strip()
            for k in re.split(r"[,\s]+", self.fields.get("CATCHES", ""))
            if k.strip()
        ]


def _fields(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    key = None
    for raw in lines:
        c = _comment(raw)
        if c is None:
            continue
        m = KEY.match(c)
        if m:
            key = m.group(1)
            out[key] = m.group(2).strip()
        elif key and c:
            out[key] = (out[key] + " " + c).strip()
    return out


def parse_file(p: Path) -> list[Test]:
    pat = DEFS.get(p.suffix)
    if not pat:
        return []
    lines = p.read_text().splitlines()
    tests: list[Test] = []
    pending_rust_attr = False
    for i, line in enumerate(lines):
        if p.suffix == ".rs":
            if re.match(r"^\s*#\[test\]", line):
                pending_rust_attr = True
                continue
            if not pending_rust_attr:
                continue
        m = pat.match(line)
        if not m:
            continue
        pending_rust_attr = False
        after: list[str] = []
        j = i + 1
        while j < len(lines) and _comment(lines[j]) is not None:
            after.append(lines[j])
            j += 1
        before: list[str] = []
        k = i - 1
        while k >= 0 and (
            _comment(lines[k]) is not None or lines[k].strip().startswith("#[")
        ):
            before.insert(0, lines[k])
            k -= 1
        f = _fields(after)
        if "WHY" not in f:
            f = {**_fields(before), **f}
        lang = {".py": "python", ".c": "c", ".rs": "rust", ".go": "go"}[p.suffix]
        tests.append(Test(m.group(1), p, i + 1, lang, f))
    return tests


def collect(files: list[Path]) -> list[Test]:
    out: list[Test] = []
    for f in files:
        out.extend(parse_file(f))
    return out


def files_for(course: Path, mid: str, test_dir: Path) -> list[Path]:
    from .ids import underscore

    out: list[Path] = []
    if test_dir.is_dir():
        out += sorted(
            p
            for p in test_dir.iterdir()
            if p.suffix in (".py", ".c")
            and p.is_file()
            and not p.name.startswith(("conftest", "_"))
        )
    r = course / "tests" / "rust" / f"{underscore(mid)}.rs"
    if r.is_file():
        out.append(r)
    g = course / "tests" / "go" / underscore(mid)
    if g.is_dir():
        out += sorted(g.rglob("*_test.go"))
    return out
