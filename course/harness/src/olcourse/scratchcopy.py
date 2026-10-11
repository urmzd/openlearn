"""The scratch-copy builder for seeded pull requests (DESIGN 5.10 `git-branch`,
craft.08, ops.06 to ops.08).

A drill or a review exercise needs a branch whose commits change the
learner's system in a known way, without ever editing the learner's working
tree or patching code they wrote by hand. The builder clones the learner repo
into a scratch directory (`git clone --no-hardlinks`, history kept), creates
`drill/<name>`, and makes a scripted series of commits there:

    {kind = "seeded", module = "L9.1", mutant = "p01", message = "perf: ..."}
        the module's unit becomes the REFERENCE plus that committed mutant
        (a perf or seeded mutant from course/mutants/<ID>/), markers dropped
    {kind = "run", argv = ["cargo", "add", "serde@1.0.200"], cwd = "rust", message = "..."}
        a command (dependency bumps: cargo add, go get, uv add)
    {kind = "write", path = "data/ledger.tsv", text = "...", message = "..."}
        a data or ledger change

Commits carry a fixed author and dates from SOURCE_DATE_EPOCH (default the
learner's HEAD time), so the series is byte-identical across runs. The
branch is then fetched into the learner repo as `drill/<name>`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, ctx, markers, units
from .registry import Registry

AUTHOR = ("ol drill", "drill@openlearn.invalid")


@dataclass
class Built:
    path: Path
    branch: str
    commits: list[str] = field(default_factory=list)  # shas, oldest first


def _git(args: list[str], cwd: Path, env: dict | None = None) -> str:
    rc, out = ctx.run(["git", *args], cwd=cwd, env=env)
    if rc != 0:
        raise HarnessError(
            f"git {' '.join(args)} failed in {cwd}:\n{ctx.indent(out.strip())}"
        )
    return out.strip()


def mutated_unit(
    course: Path, reg: Registry, module: str, mutant: str
) -> tuple[str, str]:
    """(unit path, reference text of the unit plus the mutant patch)."""
    man = course / "mutants" / module / "manifest.tsv"
    if not man.is_file():
        raise HarnessError(
            f"no mutants for {module} (course/mutants/{module}/manifest.tsv)"
        )
    row = next(
        (
            line.split("\t")
            for line in man.read_text().splitlines()
            if line.split("\t")[0] == mutant
        ),
        None,
    )
    if row is None:
        raise HarnessError(f"{module} has no mutant {mutant!r}")
    unit = row[1]
    chain = reg.unit_chain(unit)
    owner = module if module in chain else chain[-1]
    text = markers.drop_markers(units.ref_text(course, reg, unit, owner))
    patch = course / "mutants" / module / f"{mutant}.patch"
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / unit
        dst.parent.mkdir(parents=True)
        dst.write_text(text)
        rc, out = ctx.run(["patch", "-s", "-p1", "-d", td, "-i", str(patch)])
        if rc != 0:
            raise HarnessError(f"{module} {mutant}: the patch does not apply:\n{out}")
        return unit, dst.read_text()


def build(
    learner: Path,
    dest: Path,
    name: str,
    commits: list[dict],
    course: Path,
    reg: Registry,
    base: str = "HEAD",
) -> Built:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(["clone", "-q", "--no-hardlinks", str(learner), str(dest)], learner.parent)
    branch = f"drill/{name}"
    _git(["checkout", "-q", "-b", branch, base], dest)
    epoch = os.environ.get("SOURCE_DATE_EPOCH") or _git(
        ["log", "-1", "--format=%ct", base], dest
    )
    built = Built(dest, branch)
    for i, c in enumerate(commits):
        kind = c.get("kind")
        msg = str(c.get("message") or f"chore: drill {name} step {i + 1}")
        if kind == "seeded":
            unit, text = mutated_unit(course, reg, str(c["module"]), str(c["mutant"]))
            p = dest / unit
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        elif kind == "run":
            argv = [str(a) for a in c.get("argv", [])]
            if not argv:
                raise HarnessError(f"drill {name} commit {i + 1}: `run` needs argv")
            argv = [a.replace("{scratch}", str(dest)) for a in argv]
            env = {
                **os.environ,
                **{
                    k: str(v).replace("{scratch}", str(dest))
                    for k, v in (c.get("env") or {}).items()
                },
            }
            rc, out = ctx.run(
                argv,
                cwd=dest / str(c.get("cwd", ".")),
                env=env,
                timeout=float(c.get("timeout_s", 600)),
            )
            if rc != 0:
                raise HarnessError(
                    f"drill {name} commit {i + 1}: `{' '.join(argv)}` exited {rc}:\n{ctx.indent(ctx.tail(out, 20))}"
                )
        elif kind == "write":
            p = dest / str(c["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(c.get("text", "")))
        else:
            raise HarnessError(
                f"drill {name} commit {i + 1}: kind {kind!r} is seeded, run, or write"
            )
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": AUTHOR[0],
            "GIT_AUTHOR_EMAIL": AUTHOR[1],
            "GIT_COMMITTER_NAME": AUTHOR[0],
            "GIT_COMMITTER_EMAIL": AUTHOR[1],
            "GIT_AUTHOR_DATE": f"{int(epoch) + 60 * (i + 1)} +0000",
            "GIT_COMMITTER_DATE": f"{int(epoch) + 60 * (i + 1)} +0000",
        }
        _git(["add", "-A"], dest)
        # Never signed: a signature carries its own timestamp, and the
        # series must be byte-identical across runs.
        _git(
            ["-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", msg],
            dest,
            env,
        )
        built.commits.append(_git(["rev-parse", "HEAD"], dest))
    return built


def publish(learner: Path, built: Built) -> None:
    """Fetch the branch into the learner repo (their working tree is untouched)."""
    _git(["fetch", "-q", str(built.path), f"+{built.branch}:{built.branch}"], learner)


def remove(learner: Path, built_path: Path, branch: str) -> None:
    rc, _ = ctx.git(
        ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], learner
    )
    if rc == 0:
        cur = ctx.git(["rev-parse", "--abbrev-ref", "HEAD"], learner)[1].strip()
        if cur == branch:
            raise HarnessError(
                f"you are on {branch}: check out your own branch, then `ol drill reset`"
            )
        _git(["branch", "-q", "-D", branch], learner)
    shutil.rmtree(built_path, ignore_errors=True)
