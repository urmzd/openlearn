"""craft.08's pull request: build it, find its seeded defects, grade a review.

Shared by `check` (which builds the branch before the tests run) and
test_craft08_review.py. Stdlib only, plus the harness's own stdlib modules
(markers, scratchcopy) from the course tree.

The PR is three versions of go/gateway/limit/limit.go, all built from the
CURRENT gw.03 reference with `patch` (course/mutants/craft.08/):

  intended  reference + pr.patch           the eviction feature, written correctly
  head      intended + s01 ... s05         what you review: five seeded defects
  sNN       intended + sNN alone           one defect, for its proof

A defect's lines are the lines of the head that differ from the head built
without it, so the grade never depends on hard-coded line numbers.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
COURSE = Path(os.environ.get("OL_COURSE_TREE") or HERE.parents[1]).resolve()
MAT = COURSE / "mutants" / "craft.08"

sys.path.insert(0, str(COURSE / "harness" / "src"))
from olcourse import markers  # noqa: E402

SEVERITIES = ("blocking", "major", "minor", "nit", "question")
SERIOUS = ("blocking", "major")
VERDICTS = ("approve", "comment", "request-changes")
CATEGORIES = (
    "correctness",
    "concurrency",
    "performance",
    "security",
    "api",
    "tests",
    "docs",
    "style",
)


def spec() -> dict:
    return tomllib.loads((MAT / "pr.toml").read_text())


def manifest() -> list[dict]:
    rows = []
    for line in (MAT / "manifest.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split("\t")
            rows.append(
                {
                    "id": f[0],
                    "unit": f[1],
                    "category": f[3],
                    "required": f[5] == "y",
                    "public": f[6],
                    "private": f[7],
                }
            )
    return rows


def reference(unit: str) -> str:
    """The unit's course reference with markers dropped (what the PR starts from)."""
    return markers.drop_markers((COURSE / "ref" / unit).read_text())


def apply(patches: list[Path], files: dict[str, str]) -> dict[str, str]:
    """files after `patch -p1` of each patch in order (a patch that does not
    apply raises ValueError naming it)."""
    with tempfile.TemporaryDirectory(prefix="ol-craft08-") as d:
        for rel, text in files.items():
            p = Path(d) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        for patch in patches:
            r = subprocess.run(
                [
                    "patch",
                    "-s",
                    "-p1",
                    "--no-backup-if-mismatch",
                    "-d",
                    d,
                    "-i",
                    str(patch),
                ],
                capture_output=True,
                text=True,
                start_new_session=True,
                timeout=60,
            )
            if r.returncode != 0:
                raise ValueError(
                    f"{patch.relative_to(COURSE)} does not apply:\n{r.stdout}{r.stderr}"
                )
        out = {}
        for p in Path(d).rglob("*"):
            if p.is_file() and not p.name.endswith((".orig", ".rej")):
                out[p.relative_to(d).as_posix()] = p.read_text()
        return out


@dataclass
class PR:
    spec: dict
    reference: str
    intended: dict[str, str]  # unit and test file
    head: dict[str, str]
    single: dict[str, str]  # defect id -> unit text with that defect alone
    defects: list[dict] = field(default_factory=list)  # manifest rows + "lines"

    @property
    def unit(self) -> str:
        return self.spec["unit"]

    def head_lines(self) -> list[str]:
        return self.head[self.unit].splitlines()


def changed_lines(a: str, b: str) -> list[int]:
    """1-based lines of b that differ from a; a pure deletion counts as the
    line that now stands where the deleted block was."""
    out: list[int] = []
    sm = difflib.SequenceMatcher(a=a.splitlines(), b=b.splitlines(), autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            out += list(range(j1 + 1, j2 + 1)) if j2 > j1 else [j1 + 1]
    return out


_CACHE: dict[str, PR] = {}


def build() -> PR:
    if "pr" in _CACHE:
        return _CACHE["pr"]
    sp = spec()
    unit = sp["unit"]
    ref = reference(unit)
    rows = manifest()
    ids = [r["id"] for r in rows]
    intended = apply([MAT / "pr.patch"], {unit: ref})
    head = apply([MAT / f"{i}.patch" for i in ids], intended)
    single = {i: apply([MAT / f"{i}.patch"], {unit: intended[unit]})[unit] for i in ids}
    for r in rows:
        without = apply(
            [MAT / f"{i}.patch" for i in ids if i != r["id"]], {unit: intended[unit]}
        )[unit]
        r["lines"] = changed_lines(without, head[unit])
    pr = PR(sp, ref, intended, head, single, rows)
    _CACHE["pr"] = pr
    return pr


def blob_id(text: str) -> str:
    """git's object id of a file with this content (git hash-object)."""
    data = text.encode()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# -- the branch in the learner's repo -------------------------------------------


def git(args: list[str], cwd: Path) -> tuple[int, str]:
    r = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=120,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def branch(pr: PR) -> str:
    return f"drill/{pr.spec['name']}"


def branch_blob(learner: Path, pr: PR) -> str | None:
    rc, out = git(
        ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch(pr)}:{pr.unit}"],
        learner,
    )
    return out if rc == 0 else None


def ensure_branch(learner: Path) -> tuple[str, str]:
    """(state, message): the PR branch exists in the learner's repo with the
    current head, building it in .ol/craft.08/pr when it is missing or stale."""
    from olcourse import scratchcopy
    from olcourse.registry import load as load_registry

    pr = build()
    want = blob_id(pr.head[pr.unit])
    rc, _ = git(["rev-parse", "--verify", "--quiet", "HEAD"], learner)
    if rc != 0:
        return "error", "your repo has no commit yet: commit your work, then rerun"
    have = branch_blob(learner, pr)
    if have == want:
        return "current", f"{branch(pr)} is current"
    commits = []
    for c in pr.spec["commit"]:
        what = c["writes"]
        if what == "reference":
            commits.append(
                {
                    "kind": "write",
                    "path": pr.unit,
                    "text": pr.reference,
                    "message": c["message"],
                }
            )
        elif what == "head":
            commits.append(
                {
                    "kind": "write",
                    "path": pr.unit,
                    "text": pr.head[pr.unit],
                    "message": c["message"],
                }
            )
        elif what == "test":
            t = pr.spec["test"]
            commits.append(
                {
                    "kind": "write",
                    "path": t,
                    "text": pr.head[t],
                    "message": c["message"],
                }
            )
    if have is not None:
        git(["branch", "-q", "-D", branch(pr)], learner)
    built = scratchcopy.build(
        learner,
        learner / ".ol" / "craft.08" / "pr",
        pr.spec["name"],
        commits,
        COURSE,
        load_registry(COURSE),
    )
    scratchcopy.publish(learner, built)
    state = "rebuilt" if have is not None else "built"
    return (
        state,
        f"{branch(pr)} {state}: {len(built.commits)} commits on your HEAD (scratch copy .ol/craft.08/pr)",
    )


TEMPLATE = '''# Your review of the pull request on branch {branch} (craft.08, chapter section 4).
#
#   git log --stat {branch}~2..{branch}       the two commits under review
#   git diff {branch}~2 {branch}              the change
#   cd .ol/craft.08/pr/go && go test -race ./gateway/limit/...   your tests, against the PR
#
# `line` is a line of the file AT THE PR HEAD ({branch}). Severity: blocking,
# major, minor, nit, or question. A blocking or major finding needs a
# suggestion. Category: {categories}.

pr      = "{branch}"
blob    = "{blob}"            # the PR head's {unit}: leave it as written
verdict = ""                  # approve, comment, or request-changes
summary = """
"""

# One table per finding; copy it as often as you need.
# [[finding]]
# file       = "{unit}"       # default; or {test}
# line       = 0
# severity   = "major"
# category   = "correctness"
# comment    = """What is wrong, and the input or interleaving that shows it."""
# suggestion = """The change you ask for."""
'''


def write_template(learner: Path) -> bool:
    pr = build()
    p = learner / pr.spec["review"]
    if p.exists():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        TEMPLATE.format(
            branch=branch(pr),
            blob=blob_id(pr.head[pr.unit]),
            unit=pr.unit,
            test=pr.spec["test"],
            categories=", ".join(CATEGORIES),
        )
    )
    return True


# -- grading -------------------------------------------------------------------


@dataclass
class Grade:
    found: dict[str, dict]  # defect id -> the finding that hit it
    unmatched: list[dict]  # serious findings on the unit that hit no defect
    decoy_hits: list[tuple[dict, str]]  # (finding, decoy line text)


def findings(review: dict) -> list[dict]:
    out = []
    for f in review.get("finding", []):
        g = dict(f)
        g.setdefault("file", build().unit)
        out.append(g)
    return out


def grade(review: dict) -> Grade:
    pr = build()
    tol = int(pr.spec["grade"]["tolerance"])
    found: dict[str, dict] = {}
    unmatched: list[dict] = []
    decoy_hits: list[tuple[dict, str]] = []
    lines = pr.head_lines()
    decoys = {
        d["line"]: i + 1
        for i, x in enumerate(lines)
        for d in pr.spec.get("decoy", [])
        if x.strip() == d["line"]
    }
    for f in findings(review):
        if f["file"] != pr.unit or not isinstance(f.get("line"), int):
            continue
        ln = f["line"]
        best, dist = None, None
        for d in pr.defects:
            lo, hi = min(d["lines"]), max(d["lines"])
            gap = 0 if lo <= ln <= hi else min(abs(ln - lo), abs(ln - hi))
            if gap <= tol and (dist is None or gap < dist):
                best, dist = d, gap
        if best is not None:
            prev = found.get(best["id"])
            if prev is None or (
                prev.get("severity") not in SERIOUS and f.get("severity") in SERIOUS
            ):
                found[best["id"]] = f
            continue
        if f.get("severity") in SERIOUS:
            unmatched.append(f)
            for text, at in decoys.items():
                if abs(ln - at) <= tol:
                    decoy_hits.append((f, text))
    return Grade(found, unmatched, decoy_hits)


# -- running Go against one version of the unit --------------------------------


GO_SUM = (
    "github.com/BurntSushi/toml v1.6.0 h1:dRaEfpa2VI55EwlIW72hMRHdWouJeRF7TPYhI+AUQjk=\n"
    "github.com/BurntSushi/toml v1.6.0/go.mod h1:ukJfTF/6rtPPRCnwkur4qwRxa8vTRFBF0uk2lLoLwho=\n"
)
GO_PACKAGES = ("go/config", "go/gateway/server", "go/gateway/auth")


def go_test(
    unit_text: str, extra_test: str | None, run: str, race: bool, timeout: float = 120
) -> tuple[int, str]:
    """`go test` of tinyllm/gateway/limit in a scratch module holding the
    reference gateway packages it imports, this limit.go, the course proofs
    (proof/limit_proof_test.go), and optionally the PR's own test file. Its
    own process group, killed on timeout."""
    with tempfile.TemporaryDirectory(prefix="ol-craft08-go-") as d:
        root = Path(d)
        for pkg in GO_PACKAGES:
            for src in sorted((COURSE / "ref" / pkg).glob("*.go")):
                if src.name.endswith("_test.go"):
                    continue
                dst = root / Path(pkg).relative_to("go") / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(markers.drop_markers(src.read_text()))
        lim = root / "gateway" / "limit"
        lim.mkdir(parents=True)
        (lim / "limit.go").write_text(unit_text)
        (lim / "limit_proof_test.go").write_text(
            (HERE / "proof" / "limit_proof_test.go").read_text()
        )
        if extra_test is not None:
            (lim / "idle_evict_test.go").write_text(extra_test)
        (root / "go.mod").write_text(
            "module tinyllm\n\ngo 1.22\n\nrequire github.com/BurntSushi/toml v1.6.0\n"
        )
        (root / "go.sum").write_text(GO_SUM)
        env = dict(
            os.environ,
            GOWORK="off",
            GOFLAGS="",
            GOTOOLCHAIN="local",
            CGO_ENABLED="1" if race else "0",
        )
        cmd = (
            ["go", "test", "-count=1", "-run", run]
            + (["-race"] if race else [])
            + ["./gateway/limit/"]
        )
        p = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            return 124, f"go test timed out after {timeout:.0f} s\n{out}"
        return p.returncode, out


PROOF_OF = {
    "s01": ("TestProofDebtIsNotForgiven", False),
    "s02": ("TestProofLenIsRaceFree", True),
    "s03": ("TestProofInFlightChargeIsKept", False),
    "s04": ("TestProofRetryAfterRoundsUp", False),
    "s05": ("TestProofSweepRunsOncePerCadence", False),
}


def tail(text: str, n: int = 25) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:])
