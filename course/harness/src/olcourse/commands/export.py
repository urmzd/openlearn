"""ol export <DIR> [--remote URL] [--allow-incomplete] [--json]

Gives you a standalone repo you own (DESIGN 5.13):
  1. refuses on uncommitted changes in your course repo;
  2. `git clone --no-hardlinks` of it into DIR, history kept;
  3. vendors the course tests and fixtures of every module you passed into
     third_party/openlearn/ with the frozen helpers, VERSION, LICENSE,
     NOTICE, ASSETS.tsv, and the test glue: a go.work, an ol-tests crate in a
     Rust workspace file, and a conftest.py (wiring, not a runner or CI);
  4. never copies references, mutants, solve keys, or .ol/;
  5. writes third_party/openlearn/STATUS.md; any module that is not pass,
     assisted, spoiled, or self is `incomplete`, which refuses the export
     unless --allow-incomplete;
  6. commits the vendored files and prints the native command for each
     language. With --remote, `origin` points there (push it yourself).
Exit 0 on success, 1 when refused."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tomllib
from pathlib import Path

from .. import (
    EXIT_FAIL,
    EXIT_PASS,
    HarnessError,
    ctx,
    ids,
    learner,
    ledger,
    markers,
    tomlw,
    tree,
)
from ..overlay import Overlay
from ..session import Session, open_session

DONE = ("pass", "smoke", "assisted", "spoiled", "self")
VENDOR = "third_party/openlearn"

CONFTEST = '''"""Test glue written by `ol export` (not a runner): lets the vendored
openlearn course tests run natively with pytest from the repo root, one
module at a time (modules share test file names):

    uv run --project python --with pytest --with hypothesis --with numpy \\
        pytest --rootdir third_party/openlearn/tests third_party/openlearn/tests/<module>
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ.setdefault("TINYLLM_FIXTURES", str(HERE.parent / "fixtures"))
for p in (str(HERE.parent / "testkit" / "python"), str(HERE), str(ROOT / "python")):
    if p not in sys.path and Path(p).is_dir():
        sys.path.insert(0, p)

try:
    from hypothesis import settings
except ImportError:
    settings = None
if settings is not None:
    settings.register_profile("ol", derandomize=True, database=None, deadline=None, max_examples=100)
    settings.load_profile("ol")
'''

NOTICE = """openlearn course tests, fixtures, and frozen helpers
Copyright the openlearn authors. Licensed under the Apache License 2.0
(see LICENSE in this directory).

Vendored by `ol export` from openlearn commit {sha}
(course contracts {semver}). These files are test wiring for your own code;
openlearn's reference implementations were never part of this repository.
"""


def _copy_tree(src: Path, dest: Path) -> int:
    n = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        # Caches (__pycache__, .pytest_cache, .ruff_cache, ...) are never vendored.
        if p.is_dir() or any(
            x == "__pycache__" or x.startswith(".") for x in rel.parts[:-1]
        ):
            continue
        q = dest / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    return n


def statuses(s: Session) -> list[tuple[str, str, str, dict | None]]:
    out = []
    for m in s.reg.ordered():
        st = learner.state(s.learner, s.course, s.reg, m.id)
        status = st.status if st.status in DONE else "incomplete"
        out.append(
            (
                m.id,
                status,
                st.note if status != "incomplete" else st.status,
                (st.verdict or {}).get("mutation"),
            )
        )
    return out


def milestones_earned(lr: Path) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for e in ledger.entries(lr):
        if e.get("kind") == "milestone" and e.get("result") == "pass":
            flags = (
                ("smoke", e.get("mode") == "smoke"),
                ("assisted", bool(e.get("assisted"))),
            )
            tag = ", ".join(t for t, on in flags if on) or "pass"
            if seen.get(e["id"]) != "pass":
                seen[e["id"]] = tag
    return sorted(seen.items())


def status_md(rows, earned, sha: str, semver: str) -> str:
    lines = [
        "# openlearn course status",
        "",
        f"Exported by `ol export` from openlearn `{sha[:12]}` (contracts {semver}).",
        "",
        "| Module | Status | Mutation score | Note |",
        "|---|---|---|---|",
    ]
    for mid, status, note, mut in rows:
        score = (
            f"{mut['score']:.2f}" if isinstance(mut, dict) and "score" in mut else "-"
        )
        lines.append(f"| {mid} | {status} | {score} | {note or ''} |")
    lines += ["", "## Milestones", ""]
    lines += [f"- {mid}: {tag}" for mid, tag in earned] or ["- none yet"]
    return "\n".join(lines) + "\n"


def _crates(lr: Path) -> list[tuple[str, str]]:
    """(dir relative to rust/, package name) for every library crate in the learner's rust/."""
    out = []
    root = lr / "rust"
    for man in sorted(root.glob("**/Cargo.toml")) if root.is_dir() else []:
        if "target" in man.relative_to(root).parts or man.parent == root:
            continue
        try:
            doc = tomllib.loads(man.read_text())
        except tomllib.TOMLDecodeError:
            continue
        name = doc.get("package", {}).get("name")
        if name and ("lib" in doc or (man.parent / "src" / "lib.rs").is_file()):
            out.append((man.parent.relative_to(root).as_posix(), name))
    return out


def vendor(s: Session, dest: Path, passing: list[str]) -> dict[str, list[str]]:
    """Copy tests, fixtures, helpers, and glue; returns the native commands."""
    v = dest / VENDOR
    v.mkdir(parents=True, exist_ok=True)
    course = s.course
    probe = Overlay(s.reg, course, passing[0] if passing else "", {}, s.learner, v, v)
    langs: dict[str, list[str]] = {}
    go_ids, rust_ids, c_ids, py_ids, practice = [], [], [], [], []
    for mid in passing:
        m = s.reg.get(mid)
        tdir = probe.test_dir(mid)
        if tdir.is_dir():
            _copy_tree(tdir, v / "tests" / mid)
            if m.kind == "practice":
                # A practice module's tests run only through its own check,
                # which sets up what they need (OL_PRIMER_DIR and friends).
                # A check marked `ol-export: course-tree` grades against the
                # course's reference katas, which are never exported: it runs
                # only through `ol check` in a openlearn checkout.
                chk = tdir / "check"
                if chk.is_file() and "ol-export: course-tree" not in chk.read_text(
                    errors="replace"
                ):
                    practice.append(mid)
            else:
                if any(tdir.glob("*.py")):
                    py_ids.append(mid)
                if any(tdir.glob("*.c")):
                    c_ids.append(mid)
        g = course / "tests" / "go" / ids.underscore(mid)
        if g.is_dir():
            _copy_tree(g, v / "tests" / "go" / ids.underscore(mid))
            go_ids.append(mid)
        rs = course / "tests" / "rust" / f"{ids.underscore(mid)}.rs"
        if rs.is_file():
            (v / "rust" / "ol-tests" / "tests").mkdir(parents=True, exist_ok=True)
            shutil.copy2(rs, v / "rust" / "ol-tests" / "tests" / rs.name)
            rust_ids.append(mid)
        # Course tests run with OL_COURSE_TREE pointed at the vendor root,
        # so preserve each passed module's declared contracts at that path
        # (the data contracts every test may read are copied once, below).
        for contract in m.contract:
            src = course / contract
            if src.is_file():
                target = v / contract
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
        fx = course / "fixtures" / mid
        if fx.is_dir():
            _copy_tree(fx, v / "fixtures" / mid)
        for f in m.fixtures.get("files", []):
            src = course.parent / f
            rel = (
                Path(f).relative_to("course/fixtures")
                if f.startswith("course/fixtures/")
                else Path(f).name
            )
            if src.is_file():
                (v / "fixtures" / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, v / "fixtures" / rel)

    # Fixtures your own [build].steps name ({fixture:...} in system.toml): the
    # vendored practice checks run those steps.
    try:
        sys_text = (s.learner / "system.toml").read_text()
    except OSError:
        sys_text = ""
    for rel in sorted(set(re.findall(r"\{fixture:([^}]+)\}", sys_text))):
        src = course / "fixtures" / rel
        if src.is_file():
            (v / "fixtures" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, v / "fixtures" / rel)

    # Frozen helpers and the pytest glue (always: your own tests may use them too).
    if (course / "tests" / "_lib").is_dir():
        _copy_tree(course / "tests" / "_lib", v / "tests" / "_lib")
    # The fault and determinism kit (4.4): Go module, Python package, Rust crate.
    if (course / "testkit").is_dir():
        _copy_tree(course / "testkit", v / "testkit")
    (v / "tests").mkdir(parents=True, exist_ok=True)
    (v / "tests" / "conftest.py").write_text(CONFTEST)
    man = course / "fixtures" / "MANIFEST.tsv"
    if man.is_file():
        keep = [
            x
            for x in man.read_text().splitlines()
            if x.startswith("#")
            or (
                x.strip()
                and (
                    v / "fixtures" / x.split("\t")[0].removeprefix("course/fixtures/")
                ).is_file()
            )
        ]
        (v / "fixtures").mkdir(parents=True, exist_ok=True)
        (v / "fixtures" / "MANIFEST.tsv").write_text("\n".join(keep) + "\n")
    # Data contracts the course tests read by path (schemas, specs, the
    # metric and API definitions), whichever module declares them.
    for sub in ("formats", "spec", "otel", "openapi", "config", "helm"):
        if (course / "contracts" / sub).is_dir():
            _copy_tree(course / "contracts" / sub, v / "contracts" / sub)
    assets = course / "fixtures" / "ASSETS.tsv"
    if assets.is_file():
        shutil.copy2(assets, v / "ASSETS.tsv")
        fixtures_assets = v / "fixtures" / "ASSETS.tsv"
        fixtures_assets.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(assets, fixtures_assets)
    # The learning tests' bars (_lib.thresholds reads them from the fixtures
    # root), recorded from the reference over 5 seeds.
    bars = course / "fixtures" / "ref-thresholds.tsv"
    if bars.is_file():
        (v / "fixtures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(bars, v / "fixtures" / bars.name)

    c_units = [
        u
        for u in s.reg.all_units()
        if markers.lang_of(u) == "c" and u.endswith(".c") and (s.learner / u).is_file()
    ]
    if py_ids:
        # One pytest per module, as `ol check` runs them: modules share test
        # file names (test_metrics.py in L1.6 and M07.7) and sibling helper
        # names, which one rootless session cannot import twice. The rootdir
        # keeps tests/conftest.py in reach of a single module's run.
        langs["python"] = [
            "uv run --project python --with pytest --with hypothesis --with numpy "
            'bash -c \'rc=0; for t in "$@"; do pytest -q -p no:cacheprovider '
            f'--rootdir={VENDOR}/tests "$t" || rc=1; done; exit $rc\' _ '
            + " ".join(f"{VENDOR}/tests/{mid}" for mid in py_ids)
        ]
    if go_ids:
        for name in ("go.mod", "go.sum"):
            if (course / "tests" / "go" / name).is_file():
                shutil.copy2(course / "tests" / "go" / name, v / "tests" / "go" / name)
        from ..overlay import _go_version, go_version_str

        gv = max(
            _go_version(m)
            for m in (
                s.learner / "go" / "go.mod",
                course / "tests" / "go" / "go.mod",
                s.learner / "contracts" / "go" / "go.mod",
            )
        )
        # The contracts module is a workspace module, as the overlay and the
        # reference [build] use it: your go.mod replaces it without a
        # `require`, which only workspace mode resolves.
        uses = ["../../go", "./tests/go"]
        if (s.learner / "contracts" / "go" / "go.mod").is_file():
            uses.append("../../contracts/go")
        if (v / "testkit" / "go" / "go.mod").is_file():
            uses.append("./testkit/go")
        work = (
            f"go {go_version_str(gv)}\n\nuse (\n"
            + "".join(f"\t{u}\n" for u in uses)
            + ")\n"
        )
        (v / "go.work").write_text(work)
        # Course tests read fixtures and contracts through these, as `ol check`
        # sets them; absolute so the `cd` below keeps them valid.
        langs["go"] = [
            f'(export OL_COURSE_TREE="$PWD/{VENDOR}" TINYLLM_FIXTURES="$PWD/{VENDOR}/fixtures"; '
            f"cd {VENDOR}/tests/go && go test ./...)"
        ]
    if rust_ids:
        deps = {
            name: {"path": f"../../../../rust/{d}"} for d, name in _crates(s.learner)
        }
        if (s.learner / "contracts" / "rust" / "tl-contracts" / "Cargo.toml").is_file():
            deps["tl-contracts"] = {"path": "../../../../contracts/rust/tl-contracts"}
        if (v / "testkit" / "rust" / "tl-testkit" / "Cargo.toml").is_file():
            deps["tl-testkit"] = {"path": "../../testkit/rust/tl-testkit"}
        (v / "rust" / "Cargo.toml").write_text(
            # `crates` (the link below) is excluded: a test that runs
            # `cargo --manifest-path .../crates/<crate>/Cargo.toml` reaches your
            # crate through this directory, and cargo must not claim it for
            # this workspace.
            tomlw.dumps(
                {
                    "workspace": {
                        "resolver": "2",
                        "members": ["ol-tests"],
                        "exclude": ["crates"],
                    }
                }
            )
        )
        (v / "rust" / "ol-tests" / "src").mkdir(parents=True, exist_ok=True)
        # Some course tests include a unit by path from the ol-tests crate
        # (`$CARGO_MANIFEST_DIR/../crates/...`), as the overlay lays it out;
        # a relative link gives the vendored crate the same view.
        link = v / "rust" / "crates"
        if not link.exists() and not link.is_symlink():
            depth = len(Path(VENDOR).parts) + 1
            link.symlink_to(Path(*([".."] * depth)) / "rust" / "crates")
        (v / "rust" / "ol-tests" / "src" / "lib.rs").write_text(
            "// test glue written by ol export: the course tests live in tests/\n"
        )
        (v / "rust" / "ol-tests" / "Cargo.toml").write_text(
            tomlw.dumps(
                {
                    "package": {
                        "name": "ol-tests",
                        "version": "0.0.0",
                        "edition": "2021",
                        "publish": False,
                    },
                    "dependencies": deps,
                }
            )
        )
        langs["rust"] = [
            f'OL_COURSE_TREE="$PWD/{VENDOR}" TINYLLM_FIXTURES="$PWD/{VENDOR}/fixtures" '
            f"cargo test --manifest-path {VENDOR}/rust/Cargo.toml"
        ]
    if c_ids:
        langs["c"] = [
            f"mkdir -p c/build && cc -std=c11 -D_POSIX_C_SOURCE=200809L -g -fsanitize=address,undefined,float-cast-overflow -DOL_COUNTING_ALLOC=1 -Icontracts/c/include "
            f"{VENDOR}/tests/{mid}/*.c {' '.join(c_units)} -o c/build/test-{mid} -lm && c/build/test-{mid}"
            for mid in c_ids
        ]
    if practice:
        # As `ol check --all --ci` runs them: no cluster tier (OL_SMOKE=1).
        langs["practice"] = [
            f"OL_SMOKE=1 OL_COURSE_TREE={VENDOR} TINYLLM_FIXTURES={VENDOR}/fixtures "
            f"{VENDOR}/tests/{mid}/check"
            for mid in practice
        ]
    return langs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ol export",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dir", type=Path)
    ap.add_argument("--remote")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    s = open_session()
    lr = s.learner
    dest = a.dir.expanduser().resolve()

    rc, out = ctx.git(["status", "--porcelain"], lr)
    if rc != 0:
        raise HarnessError(f"{lr} is not a git repo: {out.strip()}")
    if out.strip():
        ctx.say(
            f"{ctx.RED}refusing:{ctx.RST} uncommitted changes in {lr}; commit them first\n{ctx.indent(out.rstrip())}"
        )
        return EXIT_FAIL
    if ctx.git_head(lr) is None:
        ctx.say(f"{ctx.RED}refusing:{ctx.RST} {lr} has no commits yet")
        return EXIT_FAIL
    if dest.exists() and any(dest.iterdir()):
        ctx.say(f"{ctx.RED}refusing:{ctx.RST} {dest} exists and is not empty")
        return EXIT_FAIL

    rows = statuses(s)
    incomplete = [mid for mid, st, _, _ in rows if st == "incomplete"]
    if incomplete and not a.allow_incomplete:
        ctx.say(
            f"{ctx.RED}refusing:{ctx.RST} {len(incomplete)} module(s) incomplete: {', '.join(incomplete[:12])}"
            + (" ..." if len(incomplete) > 12 else "")
            + "\n  finish them, or export anyway with --allow-incomplete"
        )
        return EXIT_FAIL

    dest.parent.mkdir(parents=True, exist_ok=True)
    rc, out = ctx.run(["git", "clone", "--no-hardlinks", "-q", str(lr), str(dest)])
    if rc != 0:
        raise HarnessError(f"git clone failed:\n{out}")
    ctx.git(["remote", "remove", "origin"], dest)
    if a.remote:
        ctx.git(["remote", "add", "origin", a.remote], dest)

    passing = [mid for mid, st, _, _ in rows if st in DONE]
    langs = vendor(s, dest, passing)
    v = dest / VENDOR
    ver = tree.read_version(lr / "contracts" / "VERSION")
    sha, semver = str(ver.get("sha", s.tree.sha)), str(ver.get("semver", "0.0.0"))
    (v / "VERSION").write_text(
        f'semver = "{semver}"\nsha = "{sha}"\ncontent_hash = "{ver.get("content_hash", "")}"\n'
    )
    lic = s.course.parent / "LICENSE"
    if lic.is_file():
        shutil.copy2(lic, v / "LICENSE")
    (v / "NOTICE").write_text(NOTICE.format(sha=sha, semver=semver))
    (v / "STATUS.md").write_text(status_md(rows, milestones_earned(lr), sha, semver))

    ctx.git(["add", "-A", VENDOR], dest)
    ident = []
    if ctx.git(["config", "user.email"], dest)[0] != 0:
        ident = ["-c", "user.name=ol export", "-c", "user.email=ol-export@localhost"]
    rc, out = ctx.git(
        [
            *ident,
            "commit",
            "-q",
            "-m",
            f"chore(tests): vendor openlearn course tests at {sha[:12]}",
        ],
        dest,
    )
    if rc != 0:
        raise HarnessError(f"git commit in {dest} failed:\n{out}")

    if a.json:
        print(
            json.dumps(
                {
                    "dir": str(dest),
                    "vendored": passing,
                    "incomplete": incomplete,
                    "commands": langs,
                }
            )
        )
        return EXIT_PASS
    ctx.say(f"{ctx.GRN}exported{ctx.RST} {dest}")
    ctx.say(
        f"  vendored the tests of {len(passing)} module(s); {len(incomplete)} incomplete (see {VENDOR}/STATUS.md)"
    )
    if a.remote:
        ctx.say(f"  origin = {a.remote}   push it: git -C {dest} push -u origin HEAD")
    if langs:
        ctx.say("  run the vendored tests natively, from the repo root:")
        for lang, cmds in langs.items():
            for c in cmds:
                ctx.say(f"    {lang:<8} {c}")
    return EXIT_PASS
