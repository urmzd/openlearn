"""craft.01 course tests: your CI workflow (.github/workflows/ci.yml).

Annotated exemplars (DESIGN 5.12). These read the workflow statically: they
cannot run GitHub Actions, so they check the three gates are present and
wired the way that makes each one actually catch something. Whether the
workflow is green on your default branch is MS-P0's `ci-status` step.
"""

import os
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(os.environ["OL_LEARNER"])
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
GATES = ("commit-lint", "native-tests", "course-check")


@pytest.fixture(scope="module")
def wf() -> dict:
    assert WORKFLOW.is_file(), ".github/workflows/ci.yml is missing"
    doc = yaml.safe_load(WORKFLOW.read_text())
    assert isinstance(doc, dict), "ci.yml must be a YAML mapping"
    return doc


def steps(wf: dict, job: str) -> list[dict]:
    return [
        s
        for s in (wf.get("jobs", {}).get(job, {}) or {}).get("steps", []) or []
        if isinstance(s, dict)
    ]


def runs(wf: dict, job: str) -> str:
    return "\n".join(str(s.get("run", "")) for s in steps(wf, job))


def test_workflow_runs_on_push_and_pull_request(wf):
    # WHY: a gate that does not run on every push and every pull request
    #      gates nothing. YAML 1.1 reads the bare key `on` as the boolean
    #      true, so a YAML parser hands it back under True, not "on".
    # KIND: unit
    # CHAPTER: craft.01 section 5, pitfall 3
    on = wf.get("on", wf.get(True))
    assert on is not None, "no `on:` trigger"
    names = {on} if isinstance(on, str) else set(on)
    assert {"push", "pull_request"} <= names, (
        f"triggers are {sorted(names)}; want push and pull_request"
    )


def test_workflow_has_the_three_gate_jobs(wf):
    # WHY: section 3: three jobs with fixed ids, so a failing check names the
    #      gate that failed and branch protection can require each by name.
    # KIND: unit
    # CHAPTER: craft.01 section 3
    jobs = wf.get("jobs") or {}
    missing = [g for g in GATES if g not in jobs]
    assert not missing, f"missing jobs {missing}; have {sorted(jobs)}"
    for g in GATES:
        assert jobs[g].get("runs-on"), f"job {g} needs runs-on"
        assert steps(wf, g), f"job {g} has no steps"


def test_commit_lint_job_fetches_history_and_runs_your_hook(wf):
    # WHY: actions/checkout clones one commit by default, so a lint over
    #      "the commits in this push" sees only the last one. The job must
    #      fetch full history (fetch-depth: 0) and run the same
    #      .githooks/commit-msg you run locally, never a second copy of the rule.
    # KIND: unit
    # CHAPTER: craft.01 section 5, pitfall 4
    checkouts = [
        s
        for s in steps(wf, "commit-lint")
        if str(s.get("uses", "")).startswith("actions/checkout")
    ]
    assert checkouts, "commit-lint must check out the repo"
    assert any(
        str((s.get("with") or {}).get("fetch-depth")) == "0" for s in checkouts
    ), "commit-lint checkout needs `with: {fetch-depth: 0}`"
    text = runs(wf, "commit-lint")
    assert ".githooks/commit-msg" in text, "commit-lint must run .githooks/commit-msg"
    # Every commit the push or pull request adds, not only HEAD: `git
    # rev-list` over a range (`BASE..HEAD`, written inline or built in a
    # variable; `^BASE HEAD` and `--not` are the same range), looped over the hook.
    assert "git rev-list" in text and re.search(r"\.\.|\^\"?\$|--not\b", text), (
        'commit-lint must walk a range of commits with `git rev-list "$BASE..$HEAD"`, '
        "not lint HEAD alone"
    )


def test_native_tests_job_runs_commands(wf):
    # WHY: the native gate runs your own tests with each language's own
    #      tool (pytest, make, cargo test, go test). In Pass 0 it builds and
    #      runs your primers; every later pass adds a step.
    # KIND: unit
    assert runs(wf, "native-tests").strip(), "native-tests has no `run:` step"


def test_course_check_job_pins_openlearn_to_contracts_version(wf):
    # WHY: your repo does not contain openlearn, so CI clones it, and it
    #      must check out the commit your contracts/VERSION names: the course
    #      tests that grade you are the ones your contracts came with. The
    #      gate is `ol check --all --ci` (no --ref-deps, no prompts), aimed
    #      at your repo through OL_COURSE_HOME.
    # KIND: conformance
    # CHAPTER: craft.01 section 5, pitfall 5
    text = runs(wf, "course-check")
    for needle in ("contracts/VERSION", "ol check --all --ci", "OL_COURSE_HOME"):
        assert needle in text, (
            f"course-check must mention `{needle}` (ol course ci prints the recipe)"
        )
    assert re.search(r"\bcheckout\b[^\n]*\$\{?SHA\b", text), (
        "course-check must `git checkout` the sha read from contracts/VERSION "
        '(for example `git -C .ol/openlearn checkout --detach "$SHA"`)'
    )
    assert "--ref-deps" not in text, (
        "CI never grades with reference code (--ci forbids --ref-deps)"
    )
