"""ol lint [--links] [--fix-index] [ID...]   chapter contract, registry, links, no U+2014 (DESIGN 6.5)

registry     every modules/*.toml parses, the 3.4 invariants hold, modules.tsv is current
chapters     the 6.5 rules for each module's chapter (all modules, or the ids given)
em dashes    none in chapters, course/**/*.md, or registry titles
--links      relative links resolve, inside the repository: in the chapters given, or
             with no ids in every markdown file git tracks or would track (DESIGN 8)
--fix-index  rewrite modules.tsv and the `## Chapters` table of each topic README"""

from __future__ import annotations

import argparse
import re
import subprocess

from .. import EXIT_FAIL, HarnessError, chapter, ctx, paths, registry, tree
from ..overlay import Overlay

START, END = "<!-- ol:chapters -->", "<!-- /ol:chapters -->"


def fix_index(reg, root) -> list[str]:
    changed = []
    p = registry.write_tsv(reg)
    changed.append(str(p))
    topics = sorted(
        {(root / m.chapter).parent for m in reg.modules.values() if m.chapter}
    )
    for t in topics:
        readme = t / "README.md"
        if not readme.is_file():
            continue
        table = chapter.chapters_table(reg, t, root)
        text = readme.read_text()
        block = f"{START}\n{table}\n{END}"
        if START in text and END in text:
            new = re.sub(
                re.escape(START) + r".*?" + re.escape(END),
                lambda _: block,
                text,
                flags=re.S,
            )
        else:
            new = text.rstrip("\n") + f"\n\n## Chapters\n\n{block}\n"
        if new != text:
            readme.write_text(new)
            changed.append(str(readme))
    return changed


def all_markdown(root) -> list:
    """Every markdown file git tracks or would track (untracked, not ignored)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "*.md"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\n")
    except (OSError, subprocess.CalledProcessError):
        return []
    return [root / f for f in out if f and (root / f).is_file()]


def run(
    only: list[str] | None = None,
    links: bool = False,
    fix: bool = False,
    quiet: bool = False,
) -> list[str]:
    course = tree.resolve(None).path
    root = course.parent
    try:
        reg = registry.load(course)
    except HarnessError as e:
        return [str(e)]
    errs: list[str] = []
    if fix:
        for c in fix_index(reg, root):
            if not quiet:
                ctx.say(f"  wrote {c}")
    if not only:
        errs += [f"registry: {e}" for e in registry.invariants(reg)]
        if not registry.tsv_current(reg):
            errs.append("course/modules.tsv is not current: run `ol lint --fix-index`")
    mods = [reg.get(i) for i in only] if only else reg.ordered()
    for m in mods:
        if "\u2014" in m.title:
            errs.append(f"{m.id}: em dash in title")
        ov = Overlay(reg, course, m.id, {}, None, course / ".lint", course / ".lint")
        errs += chapter.lint(reg, m, root, course, ov.test_dir(m.id))
    md = []
    if not only:
        md = [
            p
            for p in course.rglob("*.md")
            if not {"ref", "harness", ".lint"} & set(p.relative_to(course).parts)
        ]
        for p in md:
            if p.name.startswith("."):
                continue
            errs += chapter.no_em_dash(p.read_text(), str(p.relative_to(root)))
    if links:
        targets = [
            root / m.chapter for m in mods if m.chapter and (root / m.chapter).is_file()
        ]
        if not only:
            targets += md
            pdir = paths.paths_dir(course)
            targets += sorted(pdir.glob("course*/*.md")) if pdir.is_dir() else []
            targets += all_markdown(root)
        seen = set()
        for p in targets:
            if p in seen:
                continue
            seen.add(p)
            errs += chapter.links(p.read_text(), p, root)
    return errs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ol lint",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--links", action="store_true")
    ap.add_argument("--fix-index", action="store_true")
    a = ap.parse_args(argv)
    errs = run(a.ids or None, a.links, a.fix_index)
    for e in errs:
        ctx.say(f"{ctx.RED}FAIL{ctx.RST} {e}")
    if errs:
        ctx.say(f"{len(errs)} lint error(s)")
        return EXIT_FAIL
    ctx.say(f"{ctx.GRN}ok{ctx.RST}   ol lint")
    return 0
