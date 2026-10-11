"""dep.05 artifact check: CI for your repo (.github/workflows/platform.yml).

Run by `ol check dep.05` in your repo. Static only: GitHub Actions cannot run
here, so the check reads the workflow the way a reviewer would and checks
that each job is present and wired to catch what it exists for. Whether the
pipeline is green on your default branch is MS-prod's and MS-P0's
`ci-status` step.

Jobs (ids fixed, so branch protection can require each by name):

  lint       ruff (Python), gofmt and go vet (Go), cargo fmt and clippy (Rust)
  unit       pytest, go test -race, cargo test, and the C build
  images     docker build of every deploy/docker/*.Dockerfile
  kind-e2e   helm/kind-action with deploy/kind/cluster.yaml, `tilt ci`, and
             `ol milestone MS-prod --smoke` through the openlearn checkout
             at contracts/VERSION
  perf-gate  optional, with the optional load.02: on main only,
             `{loadgen} compare <base> <head> --metric ... --max-regress ...`,
             the head report kept as an artifact. Checked when present.

craft.01's ci.yml (commit-lint, native-tests, course-check) stays as it is.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import Ctx, Fail, run  # noqa: E402

WORKFLOW = ".github/workflows/platform.yml"
JOBS = ("lint", "unit", "images", "kind-e2e")
OPTIONAL_JOBS = ("perf-gate",)  # needs the optional load.02's `compare`
MAX_MINUTES = 60
MOVING_REFS = {
    "main",
    "master",
    "latest",
    "stable",
    "nightly",
    "head",
    "dev",
    "develop",
    "trunk",
}


def _wf(c: Ctx) -> dict:
    if "wf" not in c.cache:
        docs = c.yaml_file(WORKFLOW)
        if len(docs) != 1 or not isinstance(docs[0], dict):
            raise Fail(f"{WORKFLOW} must hold one YAML mapping")
        c.cache["wf"] = docs[0]
    return c.cache["wf"]


def _job(c: Ctx, name: str) -> dict:
    job = (_wf(c).get("jobs") or {}).get(name)
    if not isinstance(job, dict):
        raise Fail(f"{WORKFLOW} has no job {name!r} (see test_required_jobs)")
    return job


def _steps(c: Ctx, name: str) -> list[dict]:
    return [s for s in _job(c, name).get("steps") or [] if isinstance(s, dict)]


def _runs(c: Ctx, name: str) -> str:
    return "\n".join(str(s.get("run", "")) for s in _steps(c, name))


def _uses(c: Ctx, name: str) -> list[dict]:
    return [s for s in _steps(c, name) if s.get("uses")]


def test_workflow_parses_and_triggers(c: Ctx) -> None:
    # WHY: a gate that does not run on every pull request and every push to
    #      main gates nothing. YAML 1.1 reads the bare key `on` as true, so the
    #      parser hands it back under True.
    # KIND: unit
    # CHAPTER: dep.05 section 4, The interface
    on = _wf(c).get("on", _wf(c).get(True))
    names = {on} if isinstance(on, str) else set(on or [])
    if not {"push", "pull_request"} <= names:
        raise Fail(
            f"{WORKFLOW} triggers on {sorted(map(str, names))}; want push and pull_request"
        )


def test_required_jobs(c: Ctx) -> None:
    # WHY: fixed job ids are what branch protection requires by name, and a
    #      job with no timeout can hold a runner for GitHub's 6-hour default
    #      when a kind cluster hangs.
    # KIND: unit
    # CHAPTER: dep.05 section 5, Pitfall 5
    jobs = _wf(c).get("jobs") or {}
    errs = [f"missing job {j!r} (have {sorted(jobs)})" for j in JOBS if j not in jobs]
    for j in JOBS + OPTIONAL_JOBS:
        job = jobs.get(j)
        if not isinstance(job, dict):
            continue
        if not job.get("runs-on") or not job.get("steps"):
            errs.append(f"job {j}: needs runs-on and steps")
        t = job.get("timeout-minutes")
        if not isinstance(t, (int, float)) or t > MAX_MINUTES:
            errs.append(
                f"job {j}: timeout-minutes is {t}; set it, at most {MAX_MINUTES}"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_actions_are_pinned(c: Ctx) -> None:
    # WHY: `uses: owner/action@main` runs whatever that branch holds today with
    #      your token; a release tag (v4, v1.12.0) or a commit sha changes only
    #      when you change it. The same goes for downloaded tools: a pinned
    #      release URL, never a script from a moving branch piped to a shell.
    # KIND: boundary
    # CHAPTER: dep.05 section 5, Pitfall 2
    errs = []
    for name, job in (_wf(c).get("jobs") or {}).items():
        for s in (job or {}).get("steps") or []:
            run_ = str((s or {}).get("run", ""))
            if re.search(r"/(main|master)/\S*\s*\|\s*(sudo\s+)?(ba)?sh\b", run_):
                errs.append(
                    f"job {name}: `{run_.strip()[:60]}...` pipes a script from a moving branch to a shell"
                )
            u = str((s or {}).get("uses", ""))
            if not u or u.startswith(("./", "docker://")):
                continue
            ref = u.rsplit("@", 1)[1] if "@" in u else ""
            if not ref:
                errs.append(f"job {name}: `uses: {u}` has no @ref")
            elif ref.lower() in MOVING_REFS:
                errs.append(
                    f"job {name}: `uses: {u}` follows a moving branch; pin a release tag or a commit sha"
                )
            elif not re.fullmatch(r"v?\d+(\.\d+){0,2}|[0-9a-f]{40}", ref):
                errs.append(
                    f"job {name}: `uses: {u}`: ref {ref!r} is neither a version tag nor a 40-hex commit sha"
                )
    if errs:
        raise Fail("\n".join(dict.fromkeys(errs)))


def test_lint_covers_every_language(c: Ctx) -> None:
    # WHY: formatting and lint failures are the cheapest to fix and the most
    #      expensive to review; one job checks all three toolchains you write
    #      library code in.
    # KIND: unit
    text = _runs(c, "lint")
    want = {
        "ruff (Python)": r"\bruff\b",
        "gofmt (Go)": r"\bgofmt\b",
        "go vet (Go)": r"\bgo\s+vet\b",
        "cargo fmt --check (Rust)": r"\bcargo\s+fmt\b[^\n]*--check",
        "cargo clippy with -D warnings (Rust)": r"\bcargo\s+clippy\b[^\n]*-D\s+warnings",
    }
    missing = [k for k, pat in want.items() if not re.search(pat, text)]
    if missing:
        raise Fail(f"the lint job does not run {', '.join(missing)}")


def test_unit_runs_every_suite(c: Ctx) -> None:
    # WHY: your own tests, natively, on every push: pytest, go test with the
    #      race detector (the gateway and the durable engine are concurrent
    #      code), cargo test, and the C library build the engine links.
    # KIND: unit
    # CHAPTER: dep.05 section 5, Pitfall 3
    text = _runs(c, "unit")
    want = {
        "pytest": r"\bpytest\b",
        "go test -race": r"\bgo\s+test\b[^\n]*-race",
        "cargo test": r"\bcargo\s+test\b",
        "make -C c": r"\bmake\b[^\n]*-C\s+c\b",
    }
    missing = [k for k, pat in want.items() if not re.search(pat, text)]
    if missing:
        raise Fail(f"the unit job does not run {', '.join(missing)}")


def test_images_job_builds_every_dockerfile(c: Ctx) -> None:
    # WHY: a Dockerfile that breaks (a moved path, a removed COPY source) must
    #      fail the pull request that broke it, not the next deploy.
    # KIND: conformance
    files = sorted(c.path("deploy/docker").glob("*.Dockerfile"))
    if not files:
        raise Fail("deploy/docker/ has no *.Dockerfile (dep.00, dep.01)")
    text = _runs(c, "images")
    files_with = " ".join(
        str((s.get("with") or {}).get("file", "")) for s in _uses(c, "images")
    )
    errs = []
    for f in files:
        rel = str(f.relative_to(c.root))
        part = f.name.removesuffix(".Dockerfile")
        loop = re.search(r"for\s+\w+\s+in\s+([^;\n]*)", text)
        looped = (
            bool(loop)
            and part in loop.group(1).split()
            and re.search(r"deploy/docker/\$\{?\w+\}?\.Dockerfile", text)
        )
        if rel not in text and rel not in files_with and not looped:
            errs.append(f"the images job never builds {rel}")
    if "docker build" not in text and "docker/build-push-action" not in str(
        [s.get("uses") for s in _uses(c, "images")]
    ):
        errs.append(
            "the images job runs neither `docker build` nor docker/build-push-action"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_kind_e2e_job(c: Ctx) -> None:
    # WHY: the deployed system, end to end, on every pull request: a kind
    #      cluster made from YOUR deploy/kind/cluster.yaml (the NodePort
    #      mappings MS-prod uses), `tilt ci` (dep.04) to build and roll out,
    #      then the MS-prod smoke steps from the openlearn commit your
    #      contracts/VERSION names.
    # KIND: conformance
    # CHAPTER: dep.05 section 5, Pitfall 4
    errs = []
    kinds = [
        s
        for s in _uses(c, "kind-e2e")
        if str(s["uses"]).startswith("helm/kind-action@")
    ]
    if not kinds:
        errs.append("kind-e2e does not use helm/kind-action")
    elif (
        str((kinds[0].get("with") or {}).get("config", ""))
        != "deploy/kind/cluster.yaml"
    ):
        errs.append(
            "helm/kind-action must create the cluster from deploy/kind/cluster.yaml (with: {config: ...})"
        )
    text = _runs(c, "kind-e2e")
    if not re.search(r"\btilt\s+ci\b", text):
        errs.append("kind-e2e does not run `tilt ci`")
    if not re.search(r"\bmilestone\s+MS-prod\b[^\n]*--smoke", text):
        errs.append("kind-e2e does not run `ol milestone MS-prod --smoke`")
    if "contracts/VERSION" not in text or not re.search(
        r"\bcheckout\b[^\n]*\$\{?SHA\b", text
    ):
        errs.append(
            'kind-e2e must run ol from the openlearn commit contracts/VERSION names (git checkout "$SHA")'
        )
    needs = _job(c, "kind-e2e").get("needs") or []
    needs = [needs] if isinstance(needs, str) else needs
    if "images" not in needs:
        errs.append(
            "kind-e2e must need: [images] (no point deploying images that do not build)"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_perf_gate_runs_on_main_against_the_last_report(c: Ctx) -> None:
    # WHY: a perf gate compares two load reports with load.02's statistics. On
    #      main only: each green main run uploads its report, and the next one
    #      compares against it, so a regression fails the commit that made it.
    #      On pull requests it would compare against nothing stable. The job
    #      is optional (it needs the optional load.02); a job that is there
    #      must be wired right.
    # KIND: conformance
    # CHAPTER: dep.05 section 5, Pitfall 6
    if "perf-gate" not in (_wf(c).get("jobs") or {}):
        return
    job = _job(c, "perf-gate")
    cond = str(job.get("if", ""))
    errs = []
    if not re.search(
        r"refs/heads/main|ref_name\s*==\s*'main'|ref_name\s*==\s*\"main\"", cond
    ):
        errs.append(
            f"perf-gate `if:` is {cond!r}; restrict it to main (github.ref == 'refs/heads/main')"
        )
    text = _runs(c, "perf-gate")
    if not re.search(
        r"\bcompare\b[^\n]*--metric\s+\S+[^\n]*--max-regress\s+\S+|\bcompare\b[^\n]*--max-regress\s+\S+[^\n]*--metric\s+\S+",
        text,
    ):
        errs.append(
            "perf-gate does not run `compare <base> <head> --metric <m> --max-regress <x%>` (load.02)"
        )
    if not any(
        str(s["uses"]).startswith("actions/upload-artifact@")
        for s in _uses(c, "perf-gate")
    ):
        errs.append(
            "perf-gate does not upload its report (actions/upload-artifact): the next run has no baseline"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_least_privilege_and_no_echoed_secrets(c: Ctx) -> None:
    # WHY: the default GITHUB_TOKEN may write to your repo; declare
    #      `permissions:` and grant only reads. A secret echoed in a step is in
    #      the log forever (masking misses transformed values): pass secrets
    #      through env.
    # KIND: boundary
    # CHAPTER: dep.05 section 5, Pitfall 1
    errs = []
    if "permissions" not in _wf(c):
        errs.append(
            f"{WORKFLOW} sets no top-level permissions (permissions: {{contents: read}})"
        )
    for name, job in (_wf(c).get("jobs") or {}).items():
        for s in (job or {}).get("steps") or []:
            if re.search(
                r"\becho\b[^\n]*\$\{\{\s*secrets\.", str((s or {}).get("run", ""))
            ):
                errs.append(f"job {name}: a run step echoes a secret")
    if errs:
        raise Fail("\n".join(errs))


if __name__ == "__main__":
    raise SystemExit(run(globals()))
