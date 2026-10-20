#!/usr/bin/env python3
"""One course manifest supplies the website, PDF reading order, and portable pack.

Paths in source manifests are repository-relative. Exported packs preserve those
paths and put the selected manifest at their root for offline source access
and local practice. The installable web app saves published courses directly. Only build outputs duplicate files.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"course paths must be relative: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes course root: {relative}")
    if not path.exists():
        raise ValueError(f"missing course source: {relative}")
    return path


def catalog(root: Path = ROOT) -> list[dict]:
    sources = json.loads((root / "openlearn.json").read_text()).get("courses", [])
    result = []
    for source in sources:
        manifest = json.loads(resolve(root, source).read_text())
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", manifest["id"]):
            raise ValueError(f"invalid course id: {manifest['id']}")
        result.append(manifest)
    if len({m["id"] for m in result}) != len(result):
        raise ValueError("duplicate course ids")
    return result


def select(course_id: str, root: Path = ROOT) -> dict:
    aliases = json.loads((root / "openlearn.json").read_text()).get(
        "course_aliases", {}
    )
    course_id = aliases.get(course_id, course_id)
    for manifest in catalog(root):
        if manifest["id"] == course_id:
            return manifest
    raise ValueError(f"unknown course: {course_id}")


def chapters(root: Path, directory: str) -> list[str]:
    base = resolve(root, directory)
    if base.is_file():
        return [directory]

    # Overview first, then numbered topics, with capstones last.
    # Section files and section folders interleave by name, so 04-x/ comes
    # before 05-y.md.
    def walk(folder: Path) -> list[str]:
        out = []
        for child in sorted(
            folder.iterdir(),
            key=lambda p: (p.name != "README.md", p.name == "capstones", p.name),
        ):
            if child.is_file():
                if child.suffix == ".md" and child.name != "AGENTS.md":
                    out.append(child.relative_to(root).as_posix())
            elif not child.name.startswith(".") and child.name not in {
                "node_modules",
                "target",
                "__pycache__",
            }:
                out.extend(walk(child))
        return out

    return walk(base)


def stages(
    root: Path, directory: str, name: str, stack: tuple[str, ...] = ()
) -> list[dict]:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError(f"invalid path name: {name}")
    if name in stack:
        raise ValueError(f"path include cycle: {' -> '.join((*stack, name))}")
    rows = []
    for line in resolve(root, f"{directory}/{name}/path.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("@"):
            rows.extend(
                stages(root, directory, line[1:].split("\t")[0], (*stack, name))
            )
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            raise ValueError(f"invalid stage in {directory}/{name}: {line}")
        rows.append(
            {
                "title": cols[1],
                "files": [f.strip() for f in cols[2].split(",") if f.strip()],
                "done": cols[3],
            }
        )
    return rows


# Appendices every book may include; everything else belongs to exactly one book.
SHARED_APPENDICES = ("content/references/",)

# Folders inside a chapter that hold code or figures, not sections.
SUPPORT_DIRS = {"code", "diagrams", "assets", "images", "img"}


def section_labels(root: Path, directory: str, files: list[str], number: str) -> dict:
    """Number a chapter's files by position: README is the chapter, then 5.1, 5.2, ...

    A section can be a folder; its README takes the section number and the
    pages inside it are numbered 5.1.1, 5.1.2, ... Files in code or figure
    folders are not sections and get no number.
    """
    if (root / directory).is_file():
        return {directory: number}
    labels: dict[str, str] = {}
    sections: dict[str, int] = {}
    pages: dict[tuple[str, str], int] = {}
    for f in files:
        parts = Path(f).relative_to(directory).parts
        if parts == ("README.md",):
            labels[f] = number
            continue
        if any(p in SUPPORT_DIRS for p in parts[:-1]):
            continue
        k = sections.setdefault(parts[0], len(sections) + 1)
        if len(parts) == 1 or parts[1:] == ("README.md",):
            labels[f] = f"{number}.{k}"
        elif len(parts) == 2:
            j = pages.setdefault(
                (parts[0], parts[1]), sum(1 for s, _ in pages if s == parts[0]) + 1
            )
            labels[f] = f"{number}.{k}.{j}"
    return labels


def letters(i: int) -> str:
    return chr(ord("A") + i)


def publication(manifest: dict, root: Path = ROOT) -> dict:
    """Ordered, deduplicated sources for one book."""
    sections = [
        {"title": item["title"], "files": [item["file"]]}
        for item in manifest.get("start", [])
    ]
    if manifest.get("featured"):
        sections.extend(stages(root, manifest["paths"], manifest["featured"]))
    else:
        # Each track is one chapter: numbered 1..N in manifest order, then
        # appendices A, B, ... Unnumbered tracks (closing notes) get no label.
        chapter = appendix = 0
        for t in manifest.get("tracks", []):
            files = chapters(root, t["dir"])
            section = {"title": t["title"], "files": files}
            if t.get("part"):
                section["part"] = t["part"]
            if t.get("appendix"):
                section["kind"] = "appendix"
                section["number"] = letters(appendix)
                appendix += 1
            elif not t.get("unnumbered"):
                chapter += 1
                section["kind"] = "chapter"
                section["number"] = str(chapter)
            if "number" in section:
                section["labels"] = section_labels(
                    root, t["dir"], files, section["number"]
                )
            sections.append(section)
    sections.extend(
        {"title": t["title"], "files": chapters(root, t["dir"])}
        for t in manifest.get("reference", [])
    )
    seen = set()
    reading = []
    for section in sections:
        files = []
        for f in section["files"]:
            resolve(root, f)
            if f not in seen:
                seen.add(f)
                files.append(f)
        if files:
            reading.append({**section, "files": files})
    documents = list(dict.fromkeys(f for section in reading for f in section["files"]))
    if manifest.get("paths"):
        documents.extend(f for f in chapters(root, manifest["paths"]) if f not in seen)
    return {"manifest": manifest, "reading": reading, "documents": documents}


def source_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(
        {f for f in result.stdout.decode().split("\0") if f and (root / f).is_file()}
    )


def export(
    manifest: dict, output: Path, root: Path = ROOT, reading_only: bool = False
) -> Path:
    dest = output.resolve() / manifest["id"]
    if (
        dest == root.resolve()
        or root.resolve().is_relative_to(dest)
        or dest.is_relative_to(root / "content")
    ):
        raise ValueError("export destination must not overwrite course sources")
    if dest.exists():
        raise ValueError(f"export destination already exists: {dest}")
    pub = publication(manifest, root)
    want = set(pub["documents"])
    all_files = source_files(root)
    trees = [manifest["paths"]] if manifest.get("paths") else []
    if not reading_only:
        trees += manifest.get("runtime", [])
    for f in all_files:
        if any(f == t or f.startswith(t + "/") for t in trees):
            want.add(f)
    # Assets/code next to a selected lesson belong to that lesson. Do not pull
    # unrelated courses into a pack by recursively following reading links.
    parents = {Path(f).parent.as_posix() for f in pub["documents"]}
    for f in all_files:
        if Path(f).suffix != ".md" and any(f.startswith(p + "/") for p in parents):
            want.add(f)
    want.update(f for f in ["LICENSE", ".gitignore"] if (root / f).is_file())
    packed = {k: v for k, v in manifest.items() if k != "runtime"}
    dest.mkdir(parents=True)
    for f in sorted(want):
        src = resolve(root, f)
        target = dest / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    # Links to omitted lessons become source links, not broken local paths.
    pattern = re.compile(r"(!?\[[^\]\n]*\]\()([^\s)]+)([^\n)]*\))")
    for f in want:
        if not f.endswith(".md"):
            continue
        target = dest / f

        def rewrite(match):
            a, link, b = match.groups()
            if re.match(r"(?:[a-z]+:|#|/)", link):
                return match[0]
            rel, sep, fragment = link.partition("#")
            resolved = Path(f).parent / rel
            source = (root / resolved).resolve()
            if not source.is_relative_to(root):
                return match[0]
            relative = source.relative_to(root).as_posix()
            if (dest / relative).exists():
                return match[0]
            kind = "tree" if source.is_dir() else "blob"
            return (
                a
                + manifest["repo"].rstrip("/")
                + f"/{kind}/main/{relative}"
                + sep
                + fragment
                + b
            )

        target.write_text(pattern.sub(rewrite, target.read_text()))
    (dest / "openlearn.json").write_text(json.dumps(packed, indent=2) + "\n")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "validate", "index", "export"])
    parser.add_argument("--course")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/packs")
    parser.add_argument("--reading-only", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        manifests = [select(args.course, root)] if args.course else catalog(root)
        if args.command == "list":
            print("\n".join(m["id"] for m in manifests))
        elif args.command == "index":
            print(json.dumps([publication(m, root) for m in manifests]))
        elif args.command == "validate":
            owners: dict[str, str] = {}
            for m in manifests:
                p = publication(m, root)
                for doc in p["documents"]:
                    if doc.startswith(SHARED_APPENDICES):
                        continue
                    other = owners.setdefault(doc, m["id"])
                    if other != m["id"]:
                        raise ValueError(
                            f"{doc} is published by both {other} and {m['id']}: "
                            "a chapter belongs to one book; link it from the other"
                        )
                print(f"ok   {m['id']}: {len(p['documents'])} source documents")
        else:
            for m in manifests:
                dest = export(m, args.output, root, args.reading_only)
                if args.zip:
                    archive = args.output / f"openlearn-{m['id']}.zip"
                    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                        for f in sorted(dest.rglob("*")):
                            if f.is_file():
                                z.write(f, f.relative_to(dest).as_posix())
                print(dest)
    except ValueError as err:
        parser.error(str(err))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
