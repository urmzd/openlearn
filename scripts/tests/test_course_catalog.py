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
    lesson = "content/courses/mathematics-for-computing/04-linear-algebra/README.md"
    owners = [m["id"] for m in courses if lesson in publication(m)["documents"]]
    assert owners == ["mathematics-for-computing"]
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
    llm_systems = next(s for s in ml["reading"] if s.get("number") == "4")
    code = "content/courses/machine-learning/04-llm-systems/03-model-loading/code/README.md"
    assert code in llm_systems["files"] and code not in llm_systems["labels"]
    assert (
        llm_systems["labels"][
            "content/courses/machine-learning/04-llm-systems/05-serving-platforms.md"
        ]
        == "4.5"
    )
