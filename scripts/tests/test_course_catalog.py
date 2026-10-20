"""Publication contracts: shared sources, portable packs, and safe destinations."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assemble_book import assemble_course
from course_catalog import ROOT, catalog, export, publication, resolve, select


def test_every_course_has_resolvable_unique_reading_sources():
    courses = catalog()
    assert len(courses) >= 2
    for manifest in courses:
        pub = publication(manifest)
        reading = [f for section in pub["reading"] for f in section["files"]]
        assert reading and len(reading) == len(set(reading))
        assert all((ROOT / f).is_file() for f in pub["documents"])
    # Each chapter belongs to one book; reading guides link to it instead.
    lesson = "content/courses/mathematics/09-linear-algebra/README.md"
    owners = [m["id"] for m in courses if lesson in publication(m)["documents"]]
    assert owners == ["mathematics"]
    shared = {}
    for manifest in courses:
        for doc in publication(manifest)["documents"]:
            if not doc.startswith("content/references/"):
                shared.setdefault(doc, []).append(manifest["id"])
    assert {d: o for d, o in shared.items() if len(o) > 1} == {}


def test_book_does_not_repeat_overview_also_present_in_track(tmp_path):
    (tmp_path / "lessons").mkdir()
    (tmp_path / "lessons/README.md").write_text("# A lesson\n\nUNIQUE_EXPLANATION\n")
    manifest = {
        "start": [{"file": "lessons/README.md", "title": "Overview"}],
        "tracks": [{"dir": "lessons", "title": "Lessons"}],
    }
    text, count = assemble_course(tmp_path, manifest)
    assert count == 1
    assert text.count("UNIQUE_EXPLANATION") == 1


def test_export_can_be_opened_without_source_tree_and_excludes_ignored_files(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("*.secret\n")
    (root / "lessons").mkdir()
    (root / "lessons/README.md").write_text(
        "# Lesson\n\n[Other](../other.md)\n![Diagram](figure.svg)\n"
    )
    (root / "lessons/figure.svg").write_text("<svg/>")
    (root / "lessons/token.secret").write_text("not for publication")
    (root / "other.md").write_text("# Another course\n")
    manifest = {
        "id": "sample",
        "title": "Sample",
        "repo": "https://example.com/source",
        "tracks": [{"dir": "lessons", "title": "Lessons"}],
    }
    pack = export(manifest, tmp_path / "packs", root)
    assert not (pack / "lessons/token.secret").exists()
    assert (pack / "lessons/figure.svg").exists()
    assert not (pack / "other.md").exists()
    assert (
        "https://example.com/source/blob/main/other.md"
        in (pack / "lessons/README.md").read_text()
    )
    root.rename(tmp_path / "removed-source")
    packed = json.loads((pack / "openlearn.json").read_text())
    assert publication(packed, pack)["documents"] == ["lessons/README.md"]
    with pytest.raises(ValueError, match="already exists"):
        export(manifest, tmp_path / "packs", tmp_path / "removed-source")


def test_publication_rejects_paths_outside_root(tmp_path):
    root = tmp_path / "course"
    root.mkdir()
    (tmp_path / "outside.md").write_text("private")
    with pytest.raises(ValueError, match="escapes course root"):
        resolve(root, "../outside.md")


def test_chapters_and_sections_are_numbered_by_position():
    pub = publication(select("interview-preparation"))
    chapters = [s for s in pub["reading"] if s.get("kind") == "chapter"]
    assert [s["number"] for s in chapters] == ["1", "2", "3", "4", "5", "6"]
    labels = chapters[1]["labels"]
    base = "content/courses/interview-preparation/02-big-tech-and-ai-labs"
    assert labels[f"{base}/README.md"] == "2"
    assert labels[f"{base}/01-anthropic/README.md"] == "2.1"
    assert labels[f"{base}/01-anthropic/02-coding-round-1.md"] == "2.1.2"
    assert labels[f"{base}/02-google/README.md"] == "2.2"


def test_appendices_are_lettered_and_code_folders_are_not_sections():
    llm = publication(select("build-an-llm"))
    appendices = [s for s in llm["reading"] if s.get("kind") == "appendix"]
    assert [(s["number"], s["title"]) for s in appendices] == [
        ("A", "Capstones"),
        ("B", "Reading Paths"),
    ]
    ml = publication(select("machine-learning"))
    llm_systems = next(s for s in ml["reading"] if s.get("number") == "20")
    code = "content/courses/ai-engineering/20-inference-engines/03-model-loading/code/README.md"
    assert code in llm_systems["files"] and code not in llm_systems["labels"]
    assert (
        llm_systems["labels"][
            "content/courses/ai-engineering/20-inference-engines/05-serving-platforms.md"
        ]
        == "20.5"
    )


def test_systems_engineering_owns_the_consolidated_lifecycle():
    courses = catalog()
    assert "infrastructure" not in {course["id"] for course in courses}
    systems = select("systems")
    assert systems["title"] == "Systems Engineering"
    chapters = [
        section
        for section in publication(systems)["reading"]
        if section.get("kind") in ("chapter", "appendix")
    ]
    assert [(section["number"], section["title"]) for section in chapters] == [
        ("1", "System Design"),
        ("2", "Software Architecture"),
        ("3", "RPC and Protocols"),
        ("4", "Streaming and SSE"),
        ("5", "Messaging and Distributed Queueing"),
        ("6", "Distributed Data and Caching"),
        ("7", "Durable Orchestration and Workers"),
        ("8", "Search and Indexing"),
        ("9", "Cloud Native Architecture"),
        ("10", "Kubernetes Workloads and Deployment"),
        ("11", "Observability"),
        ("12", "Incident Response and Recovery"),
        ("13", "Case Study: Exactly-Once Event API"),
        ("A", "The Apache Ecosystem"),
    ]
    assert all(
        file.startswith("content/courses/systems/")
        for chapter in chapters
        for file in chapter["files"]
    )


def test_six_books_and_separate_guides_have_canonical_aliases():
    courses = catalog()
    assert {m["id"] for m in courses if m["kind"] == "book"} == {
        "mathematics",
        "programming-and-algorithms",
        "software-engineering",
        "data-engineering",
        "ai-engineering",
        "systems",
    }
    assert {m["id"] for m in courses if m["kind"] == "field-guide"} == {
        "field-engineering",
        "interview-preparation",
    }
    assert {m["id"] for m in courses if m["kind"] == "reading-path"} == {
        "computer-science",
        "research",
    }
    aliases = json.loads((ROOT / "openlearn.json").read_text())["course_aliases"]
    for old, canonical in aliases.items():
        assert select(old)["id"] == canonical


def test_book_parts_survive_publication_and_assembly():
    manifest = select("ai-engineering")
    chapters = [
        s for s in publication(manifest)["reading"] if s.get("kind") == "chapter"
    ]
    parts = list(dict.fromkeys(s["part"] for s in chapters))
    assert parts == [
        "Learning foundations",
        "Build a model",
        "Train and serve models",
        "Applications and evaluation",
        "Responsible deployment",
    ]
    text, count = assemble_course(ROOT, manifest)
    assert count == len(publication(manifest)["documents"])
    for part in parts:
        assert text.count("\\part{" + part + "}") == 1
