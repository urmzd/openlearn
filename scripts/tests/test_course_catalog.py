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
    # One canonical math lesson is reused by both course definitions.
    lesson = "content/tracks/math/03-linear-algebra/README.md"
    assert lesson in publication(select("mathematics"))["documents"]
    assert lesson in publication(select("computer-science"))["documents"]


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
