"""craft.02 artifact checks: your architecture decision records lint.

Run by `ol check craft.02` (through ./check) from the root of your repo. Each
test reads docs/adr/ and fails with a message that says what to change.
Every test carries the course annotation (DESIGN 5.12): WHY it exists and its
KIND. This file is stdlib only and is not a pytest module: `check` runs every
`test_*` function in order and reports each one.

The format the tests enforce is the ADR template in the chapter
software-craftsmanship/06-documentation-writing/01-architecture-decision-records.md:

    docs/adr/NNNN-<kebab-case-slug>.md
    # ADR-NNNN: <title>
    ## Status            Proposed | Accepted | Deprecated | Superseded by ADR-NNNN, then (YYYY-MM-DD)
    ## Context
    ## Decision
    ## Consequences
    ## Alternatives considered
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

ADR_DIR = Path("docs") / "adr"
FILE_RE = re.compile(r"^(\d{4})-([a-z0-9]+(?:-[a-z0-9]+)*)\.md$")
TITLE_RE = re.compile(r"^# ADR-(\d{4}): (\S.*)$")
STATUS_RE = re.compile(
    r"^(Proposed|Accepted|Deprecated|Superseded by ADR-(\d{4}))\b.*?\((\d{4}-\d{2}-\d{2})\)"
)
SECTIONS = ["Status", "Context", "Decision", "Consequences", "Alternatives considered"]
MIN_WORDS = {
    "Context": 25,
    "Decision": 8,
    "Consequences": 10,
    "Alternatives considered": 6,
}
PLACEHOLDER_RE = re.compile(
    r"<[A-Za-z][^<>\n`]*\s[^<>\n`]*[A-Za-z.)]>|\b(TODO|TBD|FIXME|XXX)\b"
)


class Fail(Exception):
    pass


# -- reading -----------------------------------------------------------------


def adr_files(repo: Path) -> list[Path]:
    d = repo / ADR_DIR
    return (
        sorted(
            p for p in d.glob("*.md") if p.is_file() and p.name.lower() != "readme.md"
        )
        if d.is_dir()
        else []
    )


def numbered(repo: Path) -> dict[str, Path]:
    """NNNN -> file, for every file that matches the naming rule."""
    out: dict[str, Path] = {}
    for p in adr_files(repo):
        m = FILE_RE.match(p.name)
        if m:
            out.setdefault(m.group(1), p)
    return out


def adr0001(repo: Path) -> Path:
    p = numbered(repo).get("0001")
    if p is None:
        raise Fail(
            "no docs/adr/0001-<slug>.md: write ADR-0001 from the chapter's template"
        )
    return p


def prose(text: str) -> str:
    """The text with fenced code blocks and inline code removed."""
    text = re.sub(r"^```.*?^```[^\n]*$", "", text, flags=re.S | re.M)
    return re.sub(r"`[^`\n]*`", "", text)


def sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) for every `## ` heading outside code fences, in order."""
    out: list[tuple[str, str]] = []
    cur, buf, fence = None, [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and re.match(r"^## \S", line):
            if cur is not None:
                out.append((cur, "\n".join(buf)))
            cur, buf = line[3:].strip(), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out.append((cur, "\n".join(buf)))
    return out


def section(text: str, name: str) -> str:
    for h, body in sections(text):
        if h.lower() == name.lower():
            return body
    raise Fail(f"no `## {name}` section")


def items(body: str) -> list[str]:
    """List items (`-`, `*`, `+`, `1.`) and table data rows."""
    rows = []
    for line in body.splitlines():
        s = line.strip()
        if re.match(r"^([-*+]|\d+[.)])\s+\S", s):
            rows.append(s)
        elif s.startswith("|") and not re.match(r"^\|[\s:|-]+\|?$", s):
            rows.append(s)
    tables = [r for r in rows if r.startswith("|")]
    if tables:  # the first table row is its header
        rows.remove(tables[0])
    return rows


def status_line(text: str) -> str:
    body = section(text, "Status")
    for line in body.splitlines():
        if line.strip():
            return line.strip()
    raise Fail("`## Status` is empty")


# -- the tests -----------------------------------------------------------------


def test_adr_0001_exists_with_a_slug(repo: Path) -> None:
    # WHY: ADRs are found by number and skimmed by file name, so the name is
    #      part of the record: docs/adr/0001-<kebab-case-slug>.md, exactly one
    #      file per number.
    # KIND: unit
    # CHAPTER: craft.02 section 4, The artifact and its check
    files = adr_files(repo)
    if not files:
        raise Fail("docs/adr/ holds no .md file: write docs/adr/0001-<slug>.md")
    bad = [p.name for p in files if not FILE_RE.match(p.name)]
    if bad:
        raise Fail(
            f"file names must be NNNN-<kebab-case-slug>.md (lower case, digits, single hyphens): {', '.join(bad)}"
        )
    adr0001(repo)


def test_numbers_are_unique_and_have_no_gaps(repo: Path) -> None:
    # WHY: an ADR log is append-only: numbers are never reused and never
    #      skipped, so "ADR-0003" names one decision forever.
    # KIND: boundary
    # CHAPTER: craft.02 section 5, Pitfalls, item 3
    nums = sorted(
        FILE_RE.match(p.name).group(1) for p in adr_files(repo) if FILE_RE.match(p.name)
    )
    dupes = sorted({n for n in nums if nums.count(n) > 1})
    if dupes:
        raise Fail(
            f"two files share number(s) {', '.join(dupes)}: give the later decision the next free number"
        )
    want = [f"{i:04d}" for i in range(1, len(nums) + 1)]
    if nums != want:
        missing = sorted(set(want) - set(nums))
        raise Fail(
            f"numbers must run 0001, 0002, ... with no gap; missing {', '.join(missing) or want[0]}"
        )


def test_title_line_matches_the_file_number(repo: Path) -> None:
    # WHY: the first line is what a reader sees in a listing or a link
    #      preview; `# ADR-NNNN: <title>` with the file's own number keeps the
    #      file name and the heading from drifting apart.
    # KIND: unit
    # CHAPTER: craft.02 section 3, Worked example
    for num, p in sorted(numbered(repo).items()):
        lines = [x for x in p.read_text().splitlines() if x.strip()]
        m = TITLE_RE.match(lines[0]) if lines else None
        if not m:
            raise Fail(f"{p.name}: the first line must be `# ADR-{num}: <title>`")
        if m.group(1) != num:
            raise Fail(
                f"{p.name}: the title says ADR-{m.group(1)} but the file is number {num}"
            )


def test_sections_appear_in_template_order(repo: Path) -> None:
    # WHY: every ADR answers the same five questions in the same order (where
    #      are we, why, what did we decide, what does it cost, what else did we
    #      weigh), so a reader can jump to the same place in any record.
    # KIND: unit
    # CHAPTER: craft.02 section 2, Principles
    for p in numbered(repo).values():
        heads = [h.lower() for h, _ in sections(p.read_text())]
        for s in SECTIONS:
            n = heads.count(s.lower())
            if n != 1:
                raise Fail(f"{p.name}: needs exactly one `## {s}` heading (found {n})")
        order = [heads.index(s.lower()) for s in SECTIONS]
        if order != sorted(order):
            raise Fail(
                f"{p.name}: sections must come in the order {', '.join(SECTIONS)}"
            )


def test_status_is_known_and_dated(repo: Path) -> None:
    # WHY: the status says whether the decision still binds you, and the date
    #      says what the authors could have known; an undated "Accepted" cannot
    #      be read against later changes.
    # KIND: boundary
    # CHAPTER: craft.02 section 5, Pitfalls, item 4
    for p in numbered(repo).values():
        line = status_line(p.read_text())
        m = STATUS_RE.match(line)
        if not m:
            raise Fail(
                f"{p.name}: `## Status` must start with Proposed, Accepted, Deprecated, or "
                f"Superseded by ADR-NNNN, followed by the date as (YYYY-MM-DD); got {line!r}"
            )
        try:
            datetime.date.fromisoformat(m.group(3))
        except ValueError:
            raise Fail(f"{p.name}: {m.group(3)} is not a calendar date") from None


def test_superseded_points_at_a_later_adr(repo: Path) -> None:
    # WHY: you never edit an accepted decision; you write a new ADR and mark
    #      the old one superseded. The pointer must name an ADR that exists and
    #      came later, or the trail of reasons breaks.
    # KIND: boundary
    # CHAPTER: craft.02 section 5, Pitfalls, item 1
    files = numbered(repo)
    for num, p in sorted(files.items()):
        m = STATUS_RE.match(status_line(p.read_text()))
        if m and m.group(2):
            if m.group(2) not in files:
                raise Fail(
                    f"{p.name}: superseded by ADR-{m.group(2)}, which does not exist"
                )
            if m.group(2) <= num:
                raise Fail(
                    f"{p.name}: superseded by ADR-{m.group(2)}, which is not later than ADR-{num}"
                )


def test_adr_0001_is_accepted(repo: Path) -> None:
    # WHY: your system already runs on this decision; a record left at
    #      "Proposed" tells the next reader the question is still open.
    # KIND: unit
    # CHAPTER: craft.02 section 4, The artifact and its check
    p = adr0001(repo)
    line = status_line(p.read_text())
    if not (line.startswith("Accepted") or line.startswith("Superseded by")):
        raise Fail(
            f"{p.name}: ADR-0001 records a decision you have built on, so its status is Accepted (or "
            f"Superseded by a later ADR); got {line!r}"
        )


def test_no_template_placeholders_left(repo: Path) -> None:
    # WHY: a copied template with `<what you decided>` still in it reads as
    #      a finished record in a listing but says nothing.
    # KIND: boundary
    # CHAPTER: craft.02 section 5, Pitfalls, item 5
    for p in numbered(repo).values():
        for i, line in enumerate(prose(p.read_text()).splitlines(), 1):
            m = PLACEHOLDER_RE.search(line)
            if m:
                raise Fail(
                    f"{p.name}: template text left in: {m.group(0)!r} (replace it with your own words)"
                )


def test_every_section_has_substance(repo: Path) -> None:
    # WHY: the Context is the part that ages best and is skipped most; a
    #      one-line context cannot explain the decision to someone who was
    #      not in the room.
    # KIND: boundary
    # CHAPTER: craft.02 section 5, Pitfalls, item 2
    for p in numbered(repo).values():
        text = prose(p.read_text())
        for name, need in MIN_WORDS.items():
            words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", section(text, name)))
            if words < need:
                raise Fail(
                    f"{p.name}: `## {name}` has {words} word(s); write at least {need}"
                )


def test_consequences_name_more_than_one_effect(repo: Path) -> None:
    # WHY: a decision with one listed consequence is a sales pitch; the
    #      costs you accept are what a future reader needs to judge whether
    #      the decision still holds.
    # KIND: unit
    # CHAPTER: craft.02 section 5, Pitfalls, item 2
    for p in numbered(repo).values():
        n = len(items(section(prose(p.read_text()), "Consequences")))
        if n < 2:
            raise Fail(
                f"{p.name}: `## Consequences` lists {n} item(s); list each effect, good and bad, "
                "as its own bullet (at least two)"
            )


def test_alternatives_name_a_rejected_option(repo: Path) -> None:
    # WHY: if no other option was weighed, nothing was decided; naming the
    #      rejected options (and why) stops the team from re-litigating them.
    # KIND: unit
    # CHAPTER: craft.02 section 2, Principles
    for p in numbered(repo).values():
        if not items(section(prose(p.read_text()), "Alternatives considered")):
            raise Fail(
                f"{p.name}: `## Alternatives considered` names no option; add at least one rejected "
                "option with the reason, as a bullet or a table row"
            )


def test_adr_0001_records_the_language_boundaries(repo: Path) -> None:
    # WHY: ADR-0001 is the record of how the four languages meet, the decision
    #      every later module builds on: files for model data and HTTP between
    #      processes.
    # KIND: unit
    # CHAPTER: craft.02 section 1, Why now
    p = adr0001(repo)
    text = prose(p.read_text())
    missing = [
        w
        for w, pat in (
            ("file exchange", r"\b(file|safetensors)\b"),
            ("HTTP", r"\bHTTP\b"),
        )
        if not re.search(pat, text)
    ]
    if missing:
        raise Fail(
            f"{p.name}: ADR-0001 records where the languages meet; it never mentions {' or '.join(missing)}"
        )


TESTS = [
    test_adr_0001_exists_with_a_slug,
    test_numbers_are_unique_and_have_no_gaps,
    test_title_line_matches_the_file_number,
    test_sections_appear_in_template_order,
    test_status_is_known_and_dated,
    test_superseded_points_at_a_later_adr,
    test_adr_0001_is_accepted,
    test_no_template_placeholders_left,
    test_every_section_has_substance,
    test_consequences_name_more_than_one_effect,
    test_alternatives_name_a_rejected_option,
    test_adr_0001_records_the_language_boundaries,
]


def why(fn) -> str:
    import inspect

    lines = []
    for line in inspect.getsource(fn).splitlines()[1:]:
        s = line.strip()
        if not s.startswith("#"):
            break
        lines.append(s.lstrip("# ").strip())
    text = " ".join(lines)
    m = re.search(r"WHY:\s*(.*?)(?:\s+KIND:|$)", text)
    return m.group(1) if m else ""


def main(argv: list[str]) -> int:
    repo = Path(argv[1] if len(argv) > 1 else ".").resolve()
    failed = 0
    for fn in TESTS:
        try:
            fn(repo)
        except Fail as e:
            failed += 1
            print(f"FAIL {fn.__name__}\n     {e}\n     why: {why(fn)}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"{len(TESTS) - failed} of {len(TESTS)} ADR checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
