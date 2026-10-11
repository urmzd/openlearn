"""R9 artifact checks. Each test states why it is part of the release gate."""

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("OL_REPO", "."))


# WHY: distinct agent regressions require distinct scorers so one healthy
# dimension cannot hide a broken citation, tool call, or state transition.
# KIND: regression
def test_agent_mutants_have_separate_ci_coverage():
    """Each semantic agent regression needs a scorer that can observe it."""
    suite = ROOT / "go/tests/agent-evals/suite.json"
    data = json.loads(suite.read_text())
    names = {s["name"] for s in data["scorers"]}
    assert {"citation_resolution", "tool_success", "state_change"} <= names


# WHY: fixed seeds and a held-out split make runs comparable while preventing
# tuning against the evaluation examples.
# KIND: property, eval
def test_eval_run_is_reproducible():
    """A fixed suite must include deterministic seeds and a held-out split."""
    config = json.loads((ROOT / "go/tests/agent-evals/suite.json").read_text())
    assert config["seed"] is not None
    assert config["split"]["heldout"]
    train = set(config["split"]["train"])
    validation = set(config["split"]["validation"])
    heldout = set(config["split"]["heldout"])
    assert train and validation and heldout
    assert len(train) == len(config["split"]["train"])
    assert len(validation) == len(config["split"]["validation"])
    assert len(heldout) == len(config["split"]["heldout"])
    assert train.isdisjoint(validation | heldout)
    assert validation.isdisjoint(heldout)
    assert {row["id"] for row in config["observations"]} == train | validation | heldout
    assert config["comparison"]["design"] == "paired"
    assert config["comparison"]["confidence"] == 0.95
