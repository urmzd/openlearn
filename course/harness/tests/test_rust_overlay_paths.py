"""Cargo contract dependencies resolve to the overlay's contract tree."""

from pathlib import Path

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
