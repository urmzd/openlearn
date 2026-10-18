"""Cargo contract dependencies resolve to the overlay's contract tree."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from olcourse.overlay import Overlay


def test_external_contract_path_is_rebased_for_ref_and_learner(tmp_path: Path):
    course = tmp_path / "course"
    ref = course / "ref" / "rust"
    learner = tmp_path / "learner"
    overlay_contracts = learner / "contracts"
    overlay = Overlay(
        reg=None,
        course=course,
        target="L10.6",
        sources={},
        learner=learner,
        work=tmp_path / "work",
        target_dir=tmp_path / "target",
    )

    for root, origin in (
        (ref, course / "ref" / "rust" / "crates" / "tl-engine"),
        (learner / "rust", learner / "rust" / "crates" / "tl-engine"),
    ):
        manifest = {
            "dependencies": {
                "tl-proto": {"path": "../../../../contracts/rust/tl-proto"}
            }
        }
        overlay._fix_paths(manifest, origin, root)
        assert (
            Path(manifest["dependencies"]["tl-proto"]["path"])
            == overlay_contracts / "rust" / "tl-proto"
        )


@pytest.mark.parametrize("learner_lock", [None, "learner resolution"])
def test_fresh_farm_uses_reference_lock_unless_learner_supplies_one(
    tmp_path, learner_lock
):
    course = tmp_path / "course"
    ref = course / "ref/rust"
    ref.mkdir(parents=True)
    (ref / "Cargo.toml").write_text("[workspace]\nmembers = []\n")
    (ref / "Cargo.lock").write_text("reviewed reference resolution")
    learner = tmp_path / "learner" if learner_lock else None
    if learner:
        (learner / "rust").mkdir(parents=True)
        (learner / "rust/Cargo.lock").write_text(learner_lock)
    overlay = Overlay(
        reg=SimpleNamespace(all_units=lambda: []),
        course=course,
        target="L10.6",
        sources={},
        learner=learner,
        work=tmp_path / "work",
        target_dir=tmp_path / "target",
    )
    farm = overlay.build_rust([])
    assert (farm / "Cargo.lock").read_text() == (
        learner_lock or "reviewed reference resolution"
    )
    # An existing farm may contain a newer resolution for learner dependencies.
    (farm / "Cargo.lock").write_text("existing farm resolution")
    overlay.build_rust([])
    assert (farm / "Cargo.lock").read_text() == "existing farm resolution"
