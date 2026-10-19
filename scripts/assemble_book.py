#!/usr/bin/env python3
"""Assemble the whole supersource curriculum into one pandoc-ready markdown file.

This is the preprocessor behind ``scripts/build-book.sh``. It does three things
that a naive ``cat`` of every README cannot:

1. Orders tracks and topics into a coherent learning sequence (a track's
   README first, then, recursively, numbered chapter files and numbered topic
   and part directories in order).
2. Rewrites *relative* image paths (e.g. ``diagrams/c4-context.svg``) to
   absolute paths, and relative links to repository URLs, so both survive
   being concatenated into one document in a different directory.
3. Injects LaTeX ``\\part{...}`` dividers so the PDF has a real
   Part / Chapter hierarchy and a clean table of contents.

Mermaid fenced code blocks are left untouched -- pandoc's ``mermaid-filter``
renders them downstream (or they fall back to code blocks).

With ``--path NAME`` it assembles one path instead: the stages listed in
``content/courses/build-an-llm/paths/NAME/path.tsv``, each a Part holding that stage's files in order, with
the stage's "done when" criterion as the Part's opening line.

Usage:
    python scripts/assemble_book.py --out /tmp/book.md [--root .] [--course ID | --path NAME]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from course_catalog import catalog, publication, select

# Markdown inline image:  ![alt](target "title")
IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)([^)]*\))")

# Emoji that LaTeX text fonts (DejaVu Serif) can't render. The check/cross
# code points aren't in DejaVu Serif either, so map to plain words -- legible
# and unambiguous in the PDF's comparison tables.
GLYPH_SUBS = {"✅": "Yes", "❌": "No"}


def substitute_glyphs(text: str) -> str:
    for src, dst in GLYPH_SUBS.items():
        text = text.replace(src, dst)
    return text


def single_chapter(text: str) -> str:
    """Demote every H1 after the first to H2 so each README is ONE chapter.

    House-style READMEs use two H1s (the title, then "# Concepts & Techniques")
    plus extra H1s like "# Part 1". Left alone, each becomes its own PDF chapter,
    producing a redundant "Concepts & Techniques" chapter per topic. Keeping only
    the first H1 as the chapter and demoting the rest to sections fixes the
    ordering. Code-fence aware so `#` comments inside code blocks are untouched.
    """
    out: list[str] = []
    in_fence = False
    fence = ""
    seen_h1 = False
    for line in text.split("\n"):
        marker = line.lstrip()[:3]
        if in_fence:
            if marker == fence:
                in_fence = False
            out.append(line)
            continue
        if marker in ("```", "~~~"):
            in_fence = True
            fence = marker
            out.append(line)
            continue
        if line.startswith("# "):
            if seen_h1:
                out.append("#" + line)  # H1 -> H2
            else:
                seen_h1 = True
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def is_remote_or_absolute(target: str) -> bool:
    return (
        target.startswith(("http://", "https://", "/"))
        or target.startswith("data:")
        or target.startswith("#")
    )


def rewrite_images(text: str, base_dir: Path) -> str:
    """Rewrite relative image targets to absolute paths anchored at base_dir."""

    def repl(m: re.Match[str]) -> str:
        prefix, target, suffix = m.group(1), m.group(2), m.group(3)
        if is_remote_or_absolute(target):
            return m.group(0)
        resolved = (base_dir / target).resolve()
        return f"{prefix}{resolved}{suffix}"

    return IMAGE_RE.sub(repl, text)


REPO_URL = "https://github.com/urmzd/openlearn"
# Markdown inline link that is not an image:  [text](target)
LINK_RE = re.compile(r"(?<!!)(\[[^\]]*\]\()([^)\s]+)([^)]*\))")


def rewrite_links(text: str, base_dir: Path, root: Path) -> str:
    """Point relative links at the repository, so they work in the PDF.

    A relative link means nothing once every chapter sits in one file; this
    also renders course contract links as repository URLs (DESIGN 3.1).
    """

    def repl(m: re.Match[str]) -> str:
        prefix, target, suffix = m.group(1), m.group(2), m.group(3)
        if is_remote_or_absolute(target) or target.startswith(("mailto:", "{")):
            return m.group(0)
        path, _, frag = target.partition("#")
        resolved = (base_dir / path).resolve()
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            return m.group(0)
        kind = "tree" if resolved.is_dir() else "blob"
        url = f"{REPO_URL}/{kind}/main/{rel.as_posix()}" + (f"#{frag}" if frag else "")
        return f"{prefix}{url}{suffix}"

    return LINK_RE.sub(repl, text)


def latex_escape(title: str) -> str:
    for char, repl in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    ):
        title = title.replace(char, repl)
    return title


def part_divider(title: str) -> str:
    return f"\n```{{=latex}}\n\\part{{{latex_escape(title)}}}\n```\n\n"


def demote(text: str, levels: int = 1) -> str:
    """Push every heading down ``levels`` levels (H6 stays H6). Fence-aware."""
    out: list[str] = []
    fence = ""
    for line in text.split("\n"):
        marker = line.lstrip()[:3]
        if fence:
            if marker == fence:
                fence = ""
        elif marker in ("```", "~~~"):
            fence = marker
        elif re.match(r"#{1,6} ", line):
            hashes = len(line) - len(line.lstrip("#"))
            line = "#" * min(6, hashes + levels) + line[hashes:]
        out.append(line)
    return "\n".join(out)


def retitle(text: str, title: str, level: int) -> str:
    """Replace the first heading at ``level`` with ``title``, or add one."""
    lines = text.split("\n")
    prefix = "#" * level + " "
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = prefix + title
            return "\n".join(lines)
    return f"{prefix}{title}\n\n{text}"


def first_title(text: str) -> str:
    for line in text.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def chapter_text(root: Path, section: dict, files: list[str]) -> list[str]:
    """One numbered chapter: its opening is the PDF chapter, sections are H2s.

    "Chapter 5: Title" heads the chapter, "5.3 Title" each section; every
    other heading moves down a level so sections are the only H2s.
    """
    number, labels = section["number"], section.get("labels", {})
    word = "Appendix" if section.get("kind") == "appendix" else "Chapter"
    chunks: list[str] = []
    opened = False
    for rel in files:
        path = root / rel
        text = single_chapter(path.read_text(encoding="utf-8"))
        label = labels.get(rel)
        if label == number:
            title = first_title(text) or section["title"]
            body = demote(text, 1)
            text = retitle(body, f"{word} {number}: {title}", 2).replace(
                f"## {word} {number}: {title}", f"# {word} {number}: {title}", 1
            )
            opened = True
        else:
            if not opened:
                chunks.append(f"# {word} {number}: {section['title']}\n\n")
                opened = True
            depth = 1 if label is None or label.count(".") == 1 else 2
            title = first_title(text)
            text = demote(text, depth)
            if label:
                text = retitle(text, f"{label} {title}", 1 + depth)
        text = rewrite_links(text, path.parent, root)
        chunks.append(substitute_glyphs(rewrite_images(text, path.parent)))
        chunks.append("\n\n")
    return chunks


def assemble_course(
    root: Path, manifest: dict, seen: set[str] | None = None
) -> tuple[str, int]:
    seen = set() if seen is None else seen
    chunks = []
    count = 0
    appendices = False
    for section in publication(manifest, root)["reading"]:
        files = [f for f in section["files"] if f not in seen]
        if not files:
            continue
        seen.update(files)
        if section.get("number"):
            if section.get("kind") == "appendix" and not appendices:
                chunks.append(part_divider("Appendices"))
                appendices = True
            chunks.extend(chapter_text(root, section, files))
            count += len(files)
            continue
        chunks.append(part_divider(section["title"]))
        note = f"> **Done when:** {section['done']}" if section.get("done") else ""
        chunks.extend(stage_text(root, section["title"], files, note))
        count += len(files)
    return "".join(chunks), count


def assemble(root: Path) -> tuple[str, int]:
    chunks = []
    count = 0
    seen: set[str] = set()
    for manifest in catalog(root):
        body, n = assemble_course(root, manifest, seen)
        chunks.append(body)
        count += n
    return "".join(chunks), count


def path_title(root: Path, name: str) -> str:
    readme = (
        root / "content" / "courses" / "build-an-llm" / "paths" / name / "README.md"
    )
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return name


def read_path(
    root: Path, name: str, stack: tuple[str, ...] = ()
) -> list[tuple[str, str, str, list[str], str]]:
    """Stages of content/courses/build-an-llm/paths/<name>/path.tsv with @includes expanded, in order.

    Each row is (owner, stage, title, files, done_when): ``owner`` is the path
    the stage belongs to, which differs from ``name`` for included parts.
    """
    if name in stack:
        raise SystemExit(f"[error] include cycle: {' -> '.join((*stack, name))}")
    tsv = root / "content" / "courses" / "build-an-llm" / "paths" / name / "path.tsv"
    if not tsv.exists():
        raise SystemExit(f"[error] no path manifest at {tsv}")
    rows = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("@"):
            rows.extend(read_path(root, line[1:].split("\t")[0], (*stack, name)))
            continue
        # A fifth column (the `check` of DESIGN 5.15) is for `ol learn` only.
        stage, title, files, when = line.split("\t")[:4]
        rows.append((name, stage, title, [f for f in files.split(",") if f], when))
    return rows


def after_title(text: str, note: str) -> str:
    """Insert note right under the chapter's H1, so it prints on that page.

    Text placed before the H1 would land at the end of the previous chapter.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            return "\n".join([*lines[: i + 1], "", note, *lines[i + 1 :]])
    return f"{note}\n\n{text}"


def stage_text(root: Path, stage: str, files: list[str], note: str) -> list[str]:
    chunks = []
    for i, rel in enumerate(files):
        path = root / rel
        if not path.exists():
            raise SystemExit(f"[error] stage {stage} names missing file {rel}")
        text = single_chapter(path.read_text(encoding="utf-8"))
        if i == 0 and note:
            text = after_title(text, note)
        text = rewrite_links(text, path.parent, root)
        chunks.append(substitute_glyphs(rewrite_images(text, path.parent)))
        chunks.append("\n\n")
    return chunks


def assemble_path(root: Path, name: str) -> tuple[str, int]:
    """A flat path gets one Part per stage; a composed path one Part per part.

    Either way each stage's "done when" line sits under its first chapter
    title, so the Part/Chapter hierarchy stays two levels deep.
    """
    rows = read_path(root, name)
    composed = any(owner != name for owner, *_ in rows)
    chunks: list[str] = []
    chapters = 0
    current = None
    for owner, stage, title, files, when in rows:
        if composed:
            if owner != current:
                current = owner
                part = "Overview" if owner == name else path_title(root, owner)
                chunks.append(part_divider(part))
            label = stage if owner == name else f"{owner}:{stage}"
            note = f"> **Stage {label}: {title}.** Done when: {when}"
        else:
            chunks.append(part_divider(f"Stage {stage}: {title}"))
            note = f"> **Done when:** {when}"
        chunks.extend(stage_text(root, stage, files, note))
        chapters += len(files)
    return "".join(chunks), chapters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True, type=Path, help="combined markdown output path"
    )
    parser.add_argument(
        "--root", default=Path(__file__).resolve().parent.parent, type=Path
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--course", help="course id from the catalog")
    group.add_argument(
        "--path",
        help="assemble one path from content/courses/build-an-llm/paths/<NAME>/path.tsv",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if args.course:
        body, chapters = assemble_course(root, select(args.course, root))
    elif args.path:
        body, chapters = assemble_path(root, args.path)
    else:
        body, chapters = assemble(root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(f"[info] assembled {chapters} chapter(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
