"""Chapter contract lint (DESIGN 6.2, 6.5)."""

from __future__ import annotations

import re
from pathlib import Path

from . import catalog, ids
from .registry import Module, Registry

EM_DASH = "\u2014"
BEATS = [
    "Why now",
    "Principles",
    "Worked example",
    "",
    "Pitfalls",
    "Where it's used next",
]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def sections(text: str) -> dict[str, str]:
    """`## heading` -> body, in file order."""
    out: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and re.match(r"^## [^#]", line):
            if cur is not None:
                out[cur] = "\n".join(buf)
            cur, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf)
    return out


def table_rows(body: str) -> list[list[str]]:
    rows = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("|") and not re.match(r"^\|[\s:|-]+\|?$", s):
            rows.append([c.strip() for c in s.strip("|").split("|")])
    return rows


def ticks(s: str) -> list[str]:
    return re.findall(r"`([^`]+)`", s)


def no_em_dash(text: str, where: str) -> list[str]:
    errs = []
    for i, line in enumerate(text.splitlines(), 1):
        if EM_DASH in line:
            errs.append(
                f"{where}:{i}: em dash (U+2014); use a colon, a comma, or a new sentence"
            )
    return errs


CODE_SPAN = re.compile(r"(`+)(?!`).*?(?<!`)\1(?!`)")


def links(text: str, src: Path, root: Path) -> list[str]:
    """Relative links that do not resolve to a file or directory in the repo.

    Fenced blocks and inline code spans are not links (`operator[](key)`), and
    a target must stay inside the repository: `../../../../../x` that only
    resolves because a sibling checkout happens to have `x` is broken.
    """
    errs = []
    fence = False
    root_r = root.resolve()
    where = src.relative_to(root) if src.is_relative_to(root) else src
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if fence:
            continue
        for target in LINK.findall(CODE_SPAN.sub("code", line)):
            if re.match(r"^[a-z]+:", target) or target.startswith(("#", "{")):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            dest = (
                (src.parent / path)
                if not path.startswith("/")
                else (root / path.lstrip("/"))
            ).resolve()
            if not dest.exists() or not dest.is_relative_to(root_r):
                errs.append(f"{where}:{i}: broken link {target}")
    return errs


def lint(
    reg: Registry, m: Module, root: Path, course: Path, test_dir: Path
) -> list[str]:
    if not m.chapter:
        return [f"{m.id}: registry has no `chapter`"]
    path = root / m.chapter
    where = m.chapter
    if not path.is_file():
        return [f"{m.id}: chapter {m.chapter} does not exist"]
    text = path.read_text()
    errs: list[str] = []
    first = text.splitlines()[0] if text else ""
    if first.strip() != f"<!-- ol:module {m.id} -->":
        errs.append(f"{where}:1: first line must be `<!-- ol:module {m.id} -->`")
    h1 = next((x[2:].strip() for x in text.splitlines() if x.startswith("# ")), "")
    if h1 != m.title:
        errs.append(
            f"{where}: the `# ` title {h1!r} must equal the registry title {m.title!r}"
        )
    errs += no_em_dash(text, where)
    sec = sections(text)

    # Module card.
    card = {
        r[0].strip("* "): r[1]
        for r in table_rows(sec.get("Overview", ""))
        if len(r) >= 2
    }
    mod = card.get("Module", "")
    if f"`{m.id}`" not in mod:
        errs.append(f"{where}: card Module row must name `{m.id}`")
    if m.kind not in mod:
        errs.append(f"{where}: card Module row must say the kind ({m.kind})")
    if f"Pass {m.pass_}" not in mod:
        errs.append(f"{where}: card Module row must say Pass {m.pass_}")
    for c in m.contract:
        row = card.get("Contract", "")
        if c not in row:
            errs.append(f"{where}: card Contract row must link {c}")
        cfile = course / c
        if not cfile.is_file():
            errs.append(f"{m.id}: contract {c} does not exist")
        elif f"chapter: {m.chapter}" not in cfile.read_text():
            errs.append(
                f"{c}: header comment must point back with `chapter: {m.chapter}`"
            )
    if m.kind == "build" and "Tests" not in card:
        errs.append(f"{where}: card needs a Tests row")
    for d in m.deps:
        if d not in card.get("Needs", ""):
            errs.append(f"{where}: card Needs row must name {d}")
    for u in m.used_by:
        if u not in card.get("Used by", ""):
            errs.append(f"{where}: card Used by row must name {u}")
    if m.milestone and m.milestone not in card.get("Milestone", ""):
        errs.append(f"{where}: card Milestone row must name {m.milestone}")
    ms = course / "milestones" / f"{m.milestone}.toml"
    if m.milestone and ms.is_file() and m.id not in ms.read_text():
        errs.append(f"{m.milestone}: milestone does not exercise {m.id}")

    # The six beats in order.
    numbered = [h for h in sec if re.match(r"^\d+\.\s", h)]
    nums = [int(h.split(".")[0]) for h in numbered]
    if nums != [1, 2, 3, 4, 5, 6]:
        errs.append(
            f"{where}: needs exactly the six headings `## 1.` to `## 6.` in order (found {nums})"
        )
    else:
        for h, want in zip(numbered, BEATS):
            if want and not h.split(".", 1)[1].strip().startswith(want):
                errs.append(f"{where}: `## {h}` should be `{want}`")
    beat = {int(h.split(".")[0]): sec[h] for h in numbered}
    for n, body in sorted(beat.items()):
        prose = re.sub(r"\s+", " ", body).strip()
        if len(prose) < 15 or re.match(r"^(TODO|TBD|FIXME)\b", prose, re.IGNORECASE):
            errs.append(f"{where}: beat {n} has no content (found {prose[:30]!r})")
    if "$" in beat.get(2, "") and not any(
        r and r[0].lower() == "symbol" for r in table_rows(beat.get(2, ""))
    ):
        errs.append(f"{where}: beat 2 uses math ($) but has no `| Symbol |` table")

    # Beat 4 tests table vs the course tests.
    tests = {t.name for t in catalog.collect(catalog.files_for(course, m.id, test_dir))}
    named = set()
    b4 = beat.get(4, "")
    for r in table_rows(b4):
        if r and r[0].lower() != "test":
            named |= {x.split("(")[0] for x in ticks(r[0])}
    # DESIGN 6.1: the worked example reappears as the first test. The first
    # row of the beat 4 test table names a test that beat 3 names, or whose
    # name says it is the hand example.
    first = next(
        (
            ticks(r[0])[0].split("(")[0]
            for r in table_rows(b4)
            if r and r[0].lower() != "test" and ticks(r[0])
        ),
        None,
    )
    if (
        m.kind in ("build", "side")
        and first
        and first not in beat.get(3, "")
        and not re.search(r"hand|worked", first, re.IGNORECASE)
    ):
        errs.append(
            f"{where}: the first test in beat 4 is `{first}`; it must be the beat 3 "
            "worked example (named in beat 3, or a *hand* / *worked* test)"
        )
    if m.kind in ("build", "side"):
        for t in sorted(named - tests):
            errs.append(
                f"{where}: beat 4 names test `{t}`, which is not in the course tests"
            )
        for t in sorted(tests - named):
            errs.append(f"{where}: course test `{t}` is not named in beat 4")

    # Pitfalls: Caught by names a real test (and mutant, once mutants exist).
    manifest = course / "mutants" / m.id / "manifest.tsv"
    mids = (
        {
            line.split("\t")[0]
            for line in manifest.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        if manifest.is_file()
        else None
    )
    for r in table_rows(beat.get(5, ""))[1:]:
        if len(r) < 3:
            continue
        caught = r[-1]
        for t in ticks(caught):
            if mids is not None and re.fullmatch(r"[ms]\d+", t):
                if t not in mids:
                    errs.append(
                        f"{where}: pitfall names mutant {t}, not in {manifest.relative_to(course)}"
                    )
            elif not re.fullmatch(r"[ms]\d+", t) and t not in tests:
                errs.append(
                    f"{where}: pitfall caught by `{t}`, which is not a course test"
                )
        for mt in re.findall(r"mutant\s+`?([ms]\d+)", caught):
            if mids is not None and mt not in mids:
                errs.append(f"{where}: pitfall names mutant {mt}, not in the manifest")

    # Beat 6: every dep is Back, every call site Forward.
    back, fwd = set(), set()
    for r in table_rows(beat.get(6, "")):
        if len(r) >= 2:
            found = [
                x for x in ticks(r[1]) + r[1].split() if ids.is_module_id(x.strip("`"))
            ]
            (
                back
                if r[0].lower() == "back"
                else fwd
                if r[0].lower() == "forward"
                else set()
            ).update(x.strip("`") for x in found)
    for d in m.deps:
        if d not in back:
            errs.append(f"{where}: beat 6 needs a Back row for {d}")
    for u in m.used_by:
        if u not in fwd:
            errs.append(f"{where}: beat 6 needs a Forward row for {u}")

    if m.kind == "build" and not table_rows(sec.get("Going further", ""))[1:]:
        errs.append(f"{where}: `## Going further` needs at least one row")
    return errs


def chapters_table(reg: Registry, topic: Path, root: Path) -> str:
    rows = ["| # | Module | Chapter | Kind | Pass |", "|---|---|---|---|---|"]
    mods = [
        m for m in reg.ordered() if m.chapter and (root / m.chapter).parent == topic
    ]
    for i, m in enumerate(sorted(mods, key=lambda x: x.chapter), 1):
        rel = (root / m.chapter).name
        rows.append(f"| {i} | `{m.id}` | [{m.title}]({rel}) | {m.kind} | {m.pass_} |")
    return "\n".join(rows)
