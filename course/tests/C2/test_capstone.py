"""Optional post-training capstone checks: provenance and supported claims."""

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("OL_REPO", "."))


def test_run_manifest_and_model_card():
    # WHY: a result without its starting checkpoint, dataset revision, objective,
    #      and seed cannot be reproduced or compared with the fixed held-out set.
    # KIND: conformance
    # CATCHES: none
    # CHAPTER: C2 sections 2 and 4, provenance and artifact manifest
    run = json.loads((ROOT / "artifacts/c2/run.json").read_text())
    card = (ROOT / "docs/capstone/c2/MODEL_CARD.md").read_text().lower()
    assert (
        run["base_checkpoint"]
        and run["dataset_revision"]
        and run["objective"]
        and run["seed"] is not None
    )
    assert "limitations" in card and "evaluation" in card and "base checkpoint" in card
