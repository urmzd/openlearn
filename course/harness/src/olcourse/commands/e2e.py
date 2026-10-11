"""ol verify course --e2e: the reference learner, end to end (DESIGN 5.14,
nightly CI job `reference-learner-e2e`).

Assembles a learner repo from the reference (`assemble`): `ol course init`,
`ol start` for every build and practice module in pass order, each unit
replaced by its current owner's reference with markers dropped,
course/ref/primers and course/ref/docs (markers dropped), then course/ref/entry
and course/ref/system.toml on top, committed as a conventional commit. Then,
through practice/bin/ol exactly as a learner runs it:

  1. ol check --all --ci                      every started module passes (practice checks without their cluster tier)
  2. ol milestone <MS> --smoke                 every `ci = "pr"` milestone (drills are not required under --smoke)
  3. ol conform openapi:<v> --target <tier>    each tier system.toml declares, each version with cases
  4. ol export <tmp> --allow-incomplete        then the vendored tests run natively through the glue

`ol verify course --e2e --keep DIR` assembles the learner at DIR and leaves it
there, so the B1 verification sequence can run `ol milestone` against it.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import secrets
import time
import shutil
import tempfile
from pathlib import Path

from .. import conform, ctx, kube, markers, milestones, paths, units


def _ss(env: dict, *args: str, timeout: float = 1800) -> tuple[int, str]:
    t0 = time.monotonic()
    rc, out = ctx.run(
        ["bash", str(ctx.root() / "practice" / "bin" / "ol"), *args],
        env=env,
        timeout=timeout,
    )
    ctx.say(
        f"  {ctx.DIM}ol {' '.join(args[:3])}: exit {rc} in {time.monotonic() - t0:.1f}s{ctx.RST}"
    )
    return rc, out


def learner_env(course: Path, lr: Path, scratch: Path) -> dict:
    env = ctx.base_env()
    env.update(
        {
            "OL_COURSE_ROOT": str(course),
            "OL_COURSE_HOME": str(lr),
            "OL_SCRATCH": str(scratch),
            "OL_PATHS_DIR": str(paths.paths_dir(course)),
            # A deployed reference (the nightly kind job) holds the caller's key
            # in its Secret; otherwise any fresh key serves the local services.
            "TL_API_KEY": os.environ.get("TL_API_KEY")
            or f"tl_e2e_{secrets.token_hex(8)}",
        }
    )
    env.pop("VIRTUAL_ENV", None)
    return env


def started_kinds(m) -> bool:
    """Modules the reference learner starts: code modules and practice
    artifacts. Drills are graded by `ol drill end` against a cluster."""
    return (m.kind in ("build", "side") and bool(m.owned)) or m.kind == "practice"


def _start_in_process(env: dict, ids_: list[str]) -> tuple[bool, str]:
    from .. import cli

    keys = ("OL_COURSE_ROOT", "OL_COURSE_HOME", "OL_SCRATCH", "OL_PATHS_DIR")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: env[k] for k in keys})
    try:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            for mid in ids_:
                if cli.main(["start", mid]) != 0:
                    return False, f"ol start {mid}\n{buf.getvalue()}"
        return True, ""
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _tdd_in_process(env: dict, phase: str, ids_: list[str]) -> list[str]:
    """`ol tdd <phase>` for each id; returns the ids whose phase did not hold."""
    from .. import cli

    keys = ("OL_COURSE_ROOT", "OL_COURSE_HOME", "OL_SCRATCH", "OL_PATHS_DIR")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: env[k] for k in keys})
    bad = []
    try:
        for mid in ids_:
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    rc = cli.main(["tdd", phase, mid])
                except Exception:  # a harness error is a failed phase here
                    rc = 5
            if rc != 0:
                bad.append(mid)
        return bad
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _copy_dropping_markers(src: Path, dest: Path) -> None:
    """Copy a tree; text files with SOLUTION markers lose the marker lines."""
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if any(
            part in ("target", "__pycache__", ".venv", "build") for part in rel.parts
        ):
            continue
        out = dest / rel
        if f.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if markers.has_markers(text):
            out.write_text(markers.drop_markers(text))


def commit_all(lr: Path, msg: str) -> tuple[int, str]:
    # Only ever commit the learner repo itself: a learner assembled inside
    # another checkout (a scratch dir under a worktree) without its own .git
    # would otherwise commit into that outer repository.
    rc, top = ctx.git(["rev-parse", "--show-toplevel"], lr)
    if rc != 0 or Path(top.strip()).resolve() != lr.resolve():
        rc, out = ctx.git(["init", "-q"], lr)
        if rc != 0:
            return rc, out
    rc, out = ctx.git(["add", "-A"], lr)
    if rc != 0:
        return rc, out
    return ctx.git(
        [
            "-c",
            "user.name=ol e2e",
            "-c",
            "user.email=e2e@localhost",
            "commit",
            "-qm",
            msg,
            "--allow-empty",
        ],
        lr,
    )


def _lock(lr: Path, env: dict) -> None:
    """Write the lockfiles the first checks would otherwise create (Cargo.lock,
    uv.lock), so they are committed with the reference system and `ol export`
    finds a clean tree. A learner commits theirs the same way. Best effort:
    a project that cannot lock offline is left for its first check."""
    for toml in [lr / "rust" / "Cargo.toml", *sorted(lr.glob("primers/*/Cargo.toml"))]:
        if toml.is_file() and not (toml.parent / "Cargo.lock").exists():
            ctx.run(
                [
                    "cargo",
                    "generate-lockfile",
                    "--offline",
                    "--manifest-path",
                    str(toml),
                ],
                env=env,
                timeout=120,
            )
    for toml in [
        lr / "python" / "pyproject.toml",
        *sorted(lr.glob("primers/*/pyproject.toml")),
    ]:
        if toml.is_file() and not (toml.parent / "uv.lock").exists():
            ctx.run(
                ["uv", "lock", "--quiet", "--project", str(toml.parent)],
                env=env,
                timeout=300,
            )


def assemble(
    reg, course: Path, lr: Path, env: dict, only: list[str] | None = None
) -> tuple[bool, str]:
    """A learner repo built from the reference: `ol course init`, `ol start`
    for every started kind (or just `only`), every unit as its current owner's
    reference with markers dropped, course/ref/primers into primers/ and
    course/ref/docs into docs/ (markers dropped), then course/ref/entry, course/ref/system.toml, and
    course/ref/solve, committed with a conventional message."""
    rc, out = _ss(env, "course", "init", "--name", "forge")
    if rc != 0:
        return False, "ol course init\n" + out
    ids_ = (
        only if only is not None else [m.id for m in reg.ordered() if started_kinds(m)]
    )
    ok, why = _start_in_process(env, ids_)
    if not ok:
        return False, why
    ref = course / "ref"
    # The reference learner's own graded tests (DESIGN 5.12), written test
    # first: `ol tdd red` runs them against the stubs before the reference
    # units land, `ol tdd green` after, as the R3+ journal requires.
    tdd_ids = []
    for mid in ids_:
        src = ref / "learner-tests" / mid
        if src.is_dir():
            _copy_dropping_markers(src, lr)
            lt = reg.get(mid).learner_tests or {}
            if int(lt.get("rung", 0)) >= 3:
                tdd_ids.append(mid)
    not_red = _tdd_in_process(env, "red", tdd_ids)
    for u, chain in reg.all_units().items():
        dest = lr / u
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(markers.drop_markers(units.ref_text(course, reg, u, chain[-1])))
    for sub in ("primers", "docs"):
        if (ref / sub).is_dir():
            _copy_dropping_markers(ref / sub, lr / sub)
    if (ref / "entry").is_dir():
        _copy_dropping_markers(ref / "entry", lr)
    # The reference crates live under course/ref/rust, one level deeper than
    # the learner's rust tree. Rebase paths into vendored contracts before
    # Cargo resolves the learner workspace during the build step.
    engine_manifest = lr / "rust" / "crates" / "tl-engine" / "Cargo.toml"
    if engine_manifest.is_file():
        manifest = engine_manifest.read_text()
        manifest = manifest.replace(
            "../../../../contracts/rust/tl-proto",
            "../../../contracts/rust/tl-proto",
        )
        engine_manifest.write_text(manifest)
    if (ref / "system.toml").is_file():
        shutil.copy2(ref / "system.toml", lr / "system.toml")
    # Reference answers and proofs for solve and proof modules (never
    # exported, D34), so milestones that require them (S-M07d for MS-P5,
    # review.01 for MS-prod) see the same passes a learner would record.
    if (ref / "solve").is_dir():
        shutil.copytree(ref / "solve", lr / "solve", dirs_exist_ok=True)
    _lock(lr, env)
    rc, out = commit_all(lr, "feat: the reference system")
    if rc != 0:
        return False, "commit the reference learner\n" + out
    not_green = _tdd_in_process(env, "green", [m for m in tdd_ids if m not in not_red])
    note = ""
    if not_red or not_green:
        note = f"; tdd not red: {not_red or 'none'}, not green: {not_green or 'none'}"
    return (
        True,
        f"{len(ids_)} module(s) started, entry points, primers, docs, system.toml, "
        f"{len(tdd_ids)} tdd journal(s){note}",
    )


def _failed_sections(out: str, per: int = 60) -> str:
    """Each non-passing module's own `check <ID>` block from a check --all
    run (a tail of the whole output shows only the last module and the
    table), at most `per` lines each."""
    lines = out.splitlines()
    starts = [i for i, x in enumerate(lines) if re.match(r"^check \S+ ", x)]
    blocks = []
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        block = lines[i:end]
        if any(re.match(r"^(FAIL|BLOCKED|DRIFT) ", x) for x in block):
            blocks.append("\n".join(block[:per]))
    return "\n".join(blocks)


def run(
    reg,
    course: Path,
    work: Path,
    rep,
    kind: bool = False,
    keep: Path | None = None,
    only: list[str] | None = None,
) -> None:
    """kind=False: milestones in --smoke mode (PR CI). kind=True: every milestone
    with a `ci = "kind"` step in full mode against the [deploy] cluster.
    keep: assemble the reference learner there and leave it for later runs.
    only: run just these milestones (the PR kind job deploys the Pass 1
    system, so it runs MS-P1; the nightly job deploys more and runs all)."""
    ctx.say(
        f"{ctx.BLD}verify course --{'kind' if kind else 'e2e'}{ctx.RST}  reference learner from {course / 'ref'}"
    )
    ref = course / "ref"
    if not (ref / "system.toml").is_file():
        rep.skip(
            "e2e",
            "no course/ref/system.toml yet: the reference entry points arrive with the tracer modules",
        )
        return
    if kind:
        import tomllib

        ready, why = kube.cluster_ready(
            tomllib.loads((ref / "system.toml").read_text()).get("deploy", {})
        )
        if not ready:
            rep.fail("kind", f"--kind needs the reference cluster: {why}")
            return
    tmp = Path(tempfile.mkdtemp(prefix="e2e-", dir=work))
    if keep is not None:
        keep = keep.resolve()
        if keep.exists() and any(keep.iterdir()):
            rep.fail("e2e", f"--keep {keep}: the directory exists and is not empty")
            return
    lr = keep or tmp / "learner"
    env = learner_env(course, lr, tmp / "scratch")
    # Sampled mutation grades everywhere in this run, MS-P0's "your CI is
    # green" step included (verify check 6 holds the full grades).
    env["OL_MUTATION_SAMPLE"] = "1"
    ok, what = assemble(reg, course, lr, env)
    if not ok:
        rep.fail("e2e", "assemble the reference learner", what)
        return
    rep.ok("e2e", f"assembled the reference learner at {lr}: {what}")

    # Every module's full mutation grade is verify check 6; here the
    # reference learner's tests run against the required mutants plus a
    # seeded sample, so the whole repo checks in CI time.
    rc, out = _ss(env, "check", "--all", "--ci", timeout=5400)
    if rc != 0:
        # The status table's non-passing rows, which a tail would cut off.
        bad = [
            x.strip()
            for x in out.splitlines()
            if re.match(r"^\s+\S+\s+(fail|blocked|error|drift)\b", x)
        ]
        rep.fail(
            "e2e",
            "ol check --all --ci: " + (", ".join(bad) or "see the output"),
            _failed_sections(out) or out,
            lines=600,
        )
    else:
        rep.ok("e2e", "ol check --all --ci")

    for msid in milestones.all_ids(course):
        if only and msid not in only:
            continue
        ms = milestones.load(course, msid)
        if kind and not any(
            s.kind for s in milestones.plan(course, msid, smoke=False).steps
        ):
            continue
        if not kind and ms.ci != "pr":
            continue
        # Drills are graded by `ol drill end` with a responder (the nightly
        # kind-e2e job, B9), not here. `--smoke` does not require them (I30);
        # a full (kind) run that requires one runs assisted for that drill.
        drills_ = (
            [
                r_
                for r_ in milestones.plan(course, msid, smoke=False).requires
                if r_ in reg.modules and reg.get(r_).kind == "drill"
            ]
            if kind
            else []
        )
        args = ["milestone", msid] + ([] if kind else ["--smoke"])
        args += ["--ref-deps"] if drills_ else []
        rc, out = _ss(env, *args)
        label = "kind" if kind else "e2e"
        what = f"ol milestone {msid}" + ("" if kind else " --smoke")
        if drills_:
            what += f" --ref-deps (drills: {', '.join(drills_)})"
        rep.fail(label, what, out) if rc != 0 else rep.ok(label, what)
    if kind:
        return

    import tomllib

    declared = tomllib.loads((lr / "system.toml").read_text()).get("services", {})
    # The versions a milestone or drill grades (`suite = "openapi:v1:..."`):
    # v2 is the api-v2-migration drill's target on a cluster, not a PR gate.
    graded = {
        m.group(1)
        for f in list((course / "milestones").glob("*.toml"))
        + list((course / "drills").glob("*/drill.toml"))
        for m in re.finditer(r"openapi:(v\d+)", f.read_text())
    }
    versions = sorted(
        {
            v
            for c in conform.load_cases(course)
            for v in c.versions
            if v in graded
            and (lr / "contracts" / "openapi" / f"openai-subset.{v}.yaml").is_file()
        }
    )
    for tier in [t for t in conform.TIERS if t in declared]:
        for v in versions:
            rc, out = _ss(env, "conform", f"openapi:{v}", "--target", tier)
            rep.fail(
                "e2e", f"ol conform openapi:{v} --target {tier}", out
            ) if rc != 0 else rep.ok("e2e", f"ol conform openapi:{v} --target {tier}")

    # The checks leave lock files (python/uv.lock, Cargo.lock) a learner commits.
    commit_all(lr, "chore: lock files")
    dest = tmp / "export"
    rc, out = _ss(env, "export", str(dest), "--allow-incomplete", "--json")
    if rc != 0:
        rep.fail("e2e", "ol export", out)
        return
    cmds = json.loads(out.strip().splitlines()[-1])["commands"]
    nenv = ctx.base_env()
    nenv.pop("VIRTUAL_ENV", None)
    for lang, lines in cmds.items():
        # The vendored Rust tests get their own target dir; practice checks run
        # the learner's [build] steps, whose entry points name rust/target.
        lenv = dict(nenv)
        if lang == "rust":
            lenv["CARGO_TARGET_DIR"] = str(tmp / "export-target")
        for c in lines:
            rc, out = ctx.run(["bash", "-c", c], cwd=dest, env=lenv, timeout=1800)
            rep.fail("e2e", f"vendored {lang} tests: {c}", out) if rc != 0 else rep.ok(
                "e2e", f"vendored {lang} tests run natively"
            )
