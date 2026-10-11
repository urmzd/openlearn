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
``paths/NAME/path.tsv``, each a Part holding that stage's files in order, with
the stage's "done when" criterion as the Part's opening line.

Usage:
    python scripts/assemble_book.py --out /tmp/book.md [--root .] [--path NAME]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ordered learning sequence: (directory, Part title). Within each directory the
# chapters are discovered recursively (see collect_files): the README first,
# then numbered chapter files, then numbered topic and part directories, so the
# 01-, 02-, p00-, p01-, ... prefixes order naturally. course/ (except the
# contracts appendix) and archive/ are never part of the book (DESIGN 3.1).
TRACKS: list[tuple[str, str]] = [
    ("math", "Mathematics Foundations"),
    ("algorithms", "Algorithm Mastery"),
    ("ml", "Machine Learning & AI"),
    ("systems", "Systems & Architecture"),
    ("data-engineering", "Data Engineering"),
    ("ai-platform-engineering", "AI Platform Engineering"),
    ("software-craftsmanship", "Software Craftsmanship"),
    ("infrastructure", "Infrastructure"),
    ("responsible-ai", "Responsible AI"),
    ("competitive-programming", "Competitive Programming"),
    ("field-engineering", "Field Engineering"),
    ("case-studies", "Case Studies"),
]

# Appended after the tracks as a final Part. Of course/, only the contracts
# are published (course/DESIGN.md 5.16): the reference system, tests, and
# design notes stay out of the book.
APPENDICES: list[tuple[str, str]] = [
    ("STUDY-PLAN.md", "Appendix: Study Plans"),
    ("CS-CURRICULUM.md", "Appendix: Computer Science Curriculum"),
    ("SOURCES.md", "Appendix: Sources and Validation"),
    ("course/contracts", "Appendix: Course Contracts"),
]

# Directories whose every markdown file is a chapter (no README per topic).
ALL_MARKDOWN = {"course/contracts"}

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


# Directories a chapter walk descends into: numbered topics (`05-trees`),
# spine parts (`p08-inference`), and the capstone chapters of the spine.
CHAPTER_DIR_RE = re.compile(r"^(\d\d|p\d\d)-|^capstones$")
CHAPTER_FILE_RE = re.compile(r"^\d\d-.*\.md$")
# Never part of the book, wherever they appear.
EXCLUDED = ("course", "archive")


def chapter_files(base: Path) -> list[Path]:
    """README.md, then [0-9][0-9]-*.md, then each chapter directory, recursively."""
    files: list[Path] = []
    readme = base / "README.md"
    if readme.exists():
        files.append(readme)
    files.extend(
        sorted(
            p for p in base.iterdir() if p.is_file() and CHAPTER_FILE_RE.match(p.name)
        )
    )
    # Capstones close a spine, so they sort after every numbered part.
    subs = sorted(
        (p for p in base.iterdir() if p.is_dir()),
        key=lambda p: (p.name == "capstones", p.name),
    )
    for sub in subs:
        if CHAPTER_DIR_RE.match(sub.name):
            files.extend(chapter_files(sub))
    return files


def collect_files(root: Path, directory: str) -> list[Path]:
    base = root / directory
    if base.is_file():  # an appendix like STUDY-PLAN.md
        return [base]
    if directory in ALL_MARKDOWN:
        return sorted(p for p in base.rglob("*.md") if p.is_file())
    if directory.split("/")[0] in EXCLUDED or not base.is_dir():
        return []
    return chapter_files(base)


def assemble(root: Path) -> tuple[str, int]:
    chunks: list[str] = []
    chapters = 0
    for directory, title in [*TRACKS, *APPENDICES]:
        files = collect_files(root, directory)
        if not files:
            print(f"[warn] no content for '{directory}', skipping", file=sys.stderr)
            continue
        chunks.append(part_divider(title))
        for path in files:
            text = path.read_text(encoding="utf-8")
            text = single_chapter(text)
            text = rewrite_links(text, path.parent, root)
            text = substitute_glyphs(rewrite_images(text, path.parent))
            chunks.append(text)
            chunks.append("\n\n")
            chapters += 1
    return "".join(chunks), chapters


def path_title(root: Path, name: str) -> str:
    readme = root / "paths" / name / "README.md"
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return name


def read_path(
    root: Path, name: str, stack: tuple[str, ...] = ()
) -> list[tuple[str, str, str, list[str], str]]:
    """Stages of paths/<name>/path.tsv with @includes expanded, in order.

    Each row is (owner, stage, title, files, done_when): ``owner`` is the path
    the stage belongs to, which differs from ``name`` for included parts.
    """
    if name in stack:
        raise SystemExit(f"[error] include cycle: {' -> '.join((*stack, name))}")
    tsv = root / "paths" / name / "path.tsv"
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
        if i == 0:
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
    parser.add_argument("--path", help="assemble one path from paths/<NAME>/path.tsv")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.path:
        body, chapters = assemble_path(root, args.path)
    else:
        body, chapters = assemble(root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(f"[info] assembled {chapters} chapter(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
