"""ol verify course [ID...] [--changed REF] [--global] [--nightly] [--e2e|--kind [MS-ID...] [--keep DIR]] [--assemble DIR]
maintainer checks (DESIGN 5.14)

Per module (checks 1 to 14):
   1 reference passes its course tests       8 chapter lints (6.5)
   2 stub compiles and fails the tests       9 a build module has a module call site
   3 determinism over seeds 0..runs-1       10 tests carry WHY and KIND; frozen helpers only
   4 Python references match their .pyi     11 (global) ol learn --verify
   5 markers: ownership, ids, whole bodies  12 fixtures resolve to MANIFEST.tsv rows
   6 mutants apply and are killed           13 registry invariants (3.4)
   7 solve keys parse                       14 seams: contract-only imports, compiles vs stubbed neighbours
Global: headers compile standalone (-pedantic -Werror), contracts/go, tl-contracts, and tl-proto
build, the fully stubbed reference tree compiles in all four languages with every
course test, fixture budgets, modules.tsv is current, paths verify.

With IDs, only those modules (plus the registry rules naming them). --changed maps
changed files to modules, verifies them fully, and runs the smoke tests of their
reverse dependents. --nightly uses three determinism runs.
OL_VERIFY_MUTANT_SAMPLE=N checks only N mutants per module (rotated by
GITHUB_SHA) and leaves the learner-test threshold to the nightly run; PR CI sets
it. OL_MUTATION_CACHE names a kill cache that persists across runs.

--e2e assembles a learner repo from course/ref (+ ref/primers, ref/docs,
ref/entry, ref/system.toml) and runs it as a learner would: check --all --ci,
every `ci = "pr"` milestone with --smoke, ol conform on each declared tier, then
ol export and the vendored tests natively. --kind does the same assembly and
runs every milestone with a kind step in full (or only the milestone ids
given, e.g. `--kind MS-P1`); it needs the reference kind
cluster. A milestone that requires a drill runs with --ref-deps (drills are
graded by `ol drill end`). --keep DIR leaves that learner at DIR; --assemble DIR
only builds it there (the nightly kind job deploys it to kind).

Practice modules with an executable `check` get checks 1 and 2 too: the check
passes in a learner assembled from the whole reference, and fails in a fresh
learner where only `ol start <ID>` ran (both with OL_SMOKE=1)."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .. import (
    EXIT_FAIL,
    EXIT_HARNESS,
    HarnessError,
    catalog,
    chapter,
    ctx,
    ids,
    markers,
    paths,
    placeholders,
    precheck,
    registry,
    tree,
    units,
)
from ..overlay import Overlay

MiB = 1024 * 1024
FROZEN_PY = [
    (
        r"^\s*(from|import)\s+tinyllm\.num\.(gradcheck|tolerance|rng)\b",
        "imports the learner's gradcheck/tolerance/rng",
    ),
    (
        r"\bnp\.random\.(seed|default_rng|rand|randn|random|normal|uniform|choice|randint|shuffle|permutation)\b",
        "uses numpy's global or unseeded RNG",
    ),
    (r"^\s*(import random\b|from random import)", "uses the stdlib random module"),
    (
        r"\bnp\.testing\.|\bnp\.(allclose|isclose)\b|\bmath\.isclose\b",
        "asserts closeness without _lib.close",
    ),
]
FROZEN_OTHER = {
    ".c": [
        (r"\b(s?rand)\s*\(", "uses C rand(); construct _lib PCG32 via the contract")
    ],
    ".rs": [(r"\brand::|thread_rng", "uses the rand crate; use tl-contracts::testing")],
    ".go": [(r'"math/rand(/v2)?"', "imports math/rand; use contracts/go/testing")],
}


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def ok(self, n, msg: str) -> None:
        ctx.say(f"  {ctx.GRN}ok{ctx.RST}    {n:>2} {msg}")

    def skip(self, n, msg: str) -> None:
        ctx.say(f"  {ctx.DIM}skip  {n:>2} {msg}{ctx.RST}")

    def fail(self, n, msg: str, detail: str = "", lines: int = 40) -> None:
        self.failures += 1
        ctx.say(f"  {ctx.RED}FAIL{ctx.RST}  {n:>2} {msg}")
        if detail:
            ctx.say(ctx.indent(ctx.tail(detail, lines), 9))


def _ov(
    reg, course, mid, sources, work: Path, tag: str, patches=None, seed=0
) -> Overlay:
    return Overlay(
        reg,
        course,
        mid,
        sources,
        None,
        work / mid / tag,
        work,
        dict(patches or {}),
        seed,
    )


def _compile(ov: Overlay, langs: set[str], all_tests: bool = False) -> tuple[bool, str]:
    out = []
    ok = True
    for lang in sorted(langs):
        fn = {
            "python": ov.compile_python,
            "c": ov.compile_c,
            "rust": lambda: ov.compile_rust(all_tests),
            "go": lambda: ov.compile_go(all_tests),
        }.get(lang)
        if fn is None:
            continue
        good, msg = fn()
        if not good:
            ok = False
            out.append(f"[{lang}]\n{msg}")
    return ok, "\n".join(out)


def _unit_langs(m) -> set[str]:
    return {markers.lang_of(u) for u in m.owned} - {None}


def _owner_text(course, reg, unit, owner) -> str:
    return units.ref_text(course, reg, unit, owner)


def check_markers(course, reg, m) -> list[str]:
    errs = []
    for u in m.owned:
        try:
            text = _owner_text(course, reg, u, m.id)
        except Exception as e:  # noqa: BLE001
            errs.append(str(e))
            continue
        where = str(units.ref_path(course, reg, u, m.id).relative_to(course))
        try:
            regs = markers.regions(text)
        except markers.MarkerError as e:
            errs.append(f"{where}: {e}")
            continue
        if not regs and markers.defines_functions(text, u):
            errs.append(
                f"{where}: no SOLUTION-BEGIN {m.id} markers, so it cannot be stubbed"
            )
        for r in regs:
            if r.id != m.id:
                errs.append(
                    f"{where}:{r.begin + 1}: marker id {r.id or '(none)'}; every marker in this unit must carry {m.id}"
                )
        errs += markers.lint(text, where)
    return errs


def check_pyi(course, reg, m) -> list[str]:
    errs = []
    for u in m.owned:
        if markers.lang_of(u) != "python":
            continue
        pyi = precheck.pyi_for(course / "contracts", u)
        if pyi.is_file():
            errs += precheck.python_unit(
                markers.drop_markers(_owner_text(course, reg, u, m.id)),
                pyi.read_text(),
                u,
                partial=reg.unit_chain(u)[-1] != m.id,
            )
    return errs


def _apply_patch(
    course: Path, reg, m, mutant: str, unit: str
) -> tuple[str | None, str]:
    patch = course / "mutants" / m.id / f"{mutant}.patch"
    if not patch.is_file():
        return None, f"{patch.relative_to(course)} missing"
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / unit
        dst.parent.mkdir(parents=True)
        dst.write_text(markers.drop_markers(_owner_text(course, reg, unit, m.id)))
        rc, out = ctx.run(["patch", "-s", "-p1", "-d", td, "-i", str(patch)])
        if rc != 0:
            return None, f"{mutant}: patch does not apply\n{out}"
        return dst.read_text(), ""


def _course_tests_hash(course: Path, reg, m) -> str:
    probe = _ov(reg, course, m.id, {}, Path("/nonexistent"), "hash")
    files = catalog.files_for(course, m.id, probe.test_dir(m.id))
    d = probe.test_dir(m.id)
    if d.is_dir():
        files += [
            p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        ]
    lib = course / "tests" / "_lib"
    if lib.is_dir():
        files += [p for p in lib.rglob("*.py") if "__pycache__" not in p.parts]
    h = hashlib.sha256()
    for p in sorted(set(files)):
        h.update(str(p.relative_to(course)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def check_mutants(course, reg, m, work, rep: Report) -> None:
    """Check 6: every committed patch applies and the course tests kill it
    (cached by ref unit hash, course test hash, and patch hash, so an unchanged
    mutant is never rerun); then the reference learner tests reach the
    module's mutation threshold through the same pipeline as `ol mutate`."""
    from .. import mutation

    man = course / "mutants" / m.id / "manifest.tsv"
    if not man.is_file():
        rep.skip(6, "no mutants committed for this module")
        return
    rows = [
        line.split("\t")
        for line in man.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    sources = {x: "ref" for x in [m.id] + reg.closure(m.id)}
    # Keys are content hashes, so CI can keep one file across runs.
    shared = os.environ.get("OL_MUTATION_CACHE")
    cache = mutation.Cache(Path(shared) if shared else work / "mutation-cache.json")
    thash = _course_tests_hash(course, reg, m)
    total = len(rows)
    sample = int(os.environ.get("OL_VERIFY_MUTANT_SAMPLE") or 0)
    if 0 < sample < total:
        # PR CI: a rotating sample per commit; nightly runs every mutant.
        salt = os.environ.get("GITHUB_SHA", "")
        rows = sorted(
            rows,
            key=lambda r: hashlib.sha256(f"{salt}\0{m.id}\0{r[0]}".encode()).digest(),
        )[:sample]
    bad, hits = [], 0
    for row in rows:
        mutant, unit = row[0], row[1]
        text, why = _apply_patch(course, reg, m, mutant, unit)
        if text is None:
            bad.append(why)
            continue
        patch = course / "mutants" / m.id / f"{mutant}.patch"
        key = hashlib.sha256(
            b"course-kill/v1\0"
            + _owner_text(course, reg, unit, m.id).encode()
            + b"\0"
            + thash.encode()
            + b"\0"
            + patch.read_bytes()
        ).hexdigest()
        if cache.data["mutants"].get(key, {}).get("status") == "killed":
            hits += 1
            continue
        runs = _ov(
            reg, course, m.id, sources, work, f"mutant-{mutant}", {unit: text}
        ).run_tests(m.id)
        if all(r.ok for r in runs):
            bad.append(f"{mutant}: survives the course tests")
        else:
            cache.data["mutants"][key] = {"status": "killed"}
    cache.save()
    of = f"{len(rows)} of {total}" if len(rows) < total else f"{total}"
    if bad:
        rep.fail(6, f"mutants ({len(bad)} of {of})", "\n".join(bad))
    else:
        rep.ok(
            6,
            f"{of} mutants apply and are killed"
            + (f" ({hits} from the kill cache)" if hits else ""),
        )
    if m.learner_tests and len(rows) < total:
        rep.skip(6, "reference learner tests' threshold: nightly (sampled run)")
        return
    if m.learner_tests:
        try:
            root = mutation.scratch_tests_root(course, m, work)
        except HarnessError as e:
            rep.fail(6, "reference learner tests", str(e))
            return
        if root is None:
            rep.fail(
                6,
                f"no reference learner tests at course/ref/learner-tests/{m.id}/ to prove the threshold reachable",
            )
            return
        g = mutation.Grader(
            reg, course, m, root, work, work, cache, impl_sources=None
        ).grade()
        if g.passed:
            rep.ok(
                6,
                f"reference learner tests reach the threshold: {g.score:.2f} >= {g.threshold:.2f} ({g.killed}/{g.total})",
            )
        else:
            surv = [r.mid for r in g.results if r.status == "survived"]
            rep.fail(
                6,
                f"reference learner tests miss the threshold: {g.score:.2f} < {g.threshold:.2f}"
                + ("" if g.required_ok else " or a required mutant survives"),
                (g.reason + "\n" if g.reason else "")
                + (f"survivors: {', '.join(surv)}" if surv else ""),
            )


def check_annotations(course, reg, m, test_dir: Path) -> list[str]:
    errs = []
    files = catalog.files_for(course, m.id, test_dir)
    man = course / "mutants" / m.id / "manifest.tsv"
    mids = (
        {
            line.split("\t")[0]
            for line in man.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        if man.is_file()
        else None
    )
    for t in catalog.collect(files):
        where = f"{t.file.relative_to(course)}:{t.line} {t.name}"
        if "WHY" not in t.fields:
            errs.append(f"{where}: missing `WHY:`")
        if "KIND" not in t.fields:
            errs.append(f"{where}: missing `KIND:`")
        bad = [k for k in t.kinds if k not in ids.TEST_KINDS]
        if bad:
            errs.append(f"{where}: KIND {bad} not in the D26 vocabulary")
        if mids is not None:
            for c in t.catches:
                if c not in mids:
                    errs.append(
                        f"{where}: CATCHES {c}, which is not in the mutants manifest"
                    )
    for f in files:
        text = f.read_text()
        rules = FROZEN_PY if f.suffix == ".py" else FROZEN_OTHER.get(f.suffix, [])
        for pat, why in rules:
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(pat, line):
                    errs.append(
                        f"{f.relative_to(course)}:{i}: {why} (D35: frozen helpers only)"
                    )
    return errs


def _module_names(unit: str) -> str:
    rel = unit.split("/", 1)[1] if unit.startswith("python/") else unit
    return rel[:-3].replace("/", ".")


def check_seams_static(course, reg, m) -> list[str]:
    errs = []
    py_units = {
        _module_names(u): u for u in reg.all_units() if markers.lang_of(u) == "python"
    }
    for u in m.owned:
        lang = markers.lang_of(u)
        text = markers.drop_markers(_owner_text(course, reg, u, m.id))
        if lang == "python":
            try:
                tree_ = ast.parse(text)
            except SyntaxError as e:
                errs.append(f"{u}:{e.lineno}: {e.msg}")
                continue
            aliases: dict[str, str] = {}
            used: dict[str, set[str]] = {}
            for node in ast.walk(tree_):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module in py_units
                    and py_units[node.module] not in m.owned
                ):
                    used.setdefault(node.module, set()).update(
                        a.name for a in node.names
                    )
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name in py_units and py_units[a.name] not in m.owned:
                            aliases[a.asname or a.name] = a.name
            for node in ast.walk(tree_):
                if isinstance(node, ast.Attribute):
                    base = ast.unparse(node.value)
                    if base in aliases:
                        used.setdefault(aliases[base], set()).add(node.attr)
            for mod, names in used.items():
                pyi = course / "contracts" / "py" / (mod.replace(".", "/") + ".pyi")
                if not pyi.is_file():
                    errs.append(
                        f"{u}: imports {mod}, which has no contract ({pyi.relative_to(course)})"
                    )
                    continue
                declared = precheck.declared_names(ast.parse(pyi.read_text()))
                for n in sorted(names - declared):
                    errs.append(
                        f"{u}: uses {mod}.{n}, which {pyi.relative_to(course)} does not declare"
                    )
        elif lang == "c":
            for i, line in enumerate(text.splitlines(), 1):
                mm = re.match(r'^\s*#\s*include\s*"([^"]+)"', line)
                if mm and not (
                    mm.group(1) == "tinyllm.h" or mm.group(1).startswith("tinyllm/")
                ):
                    errs.append(
                        f'{u}:{i}: includes "{mm.group(1)}"; units reach each other only through tinyllm/*.h'
                    )
    return errs


def fixture_errors(course: Path, mods) -> list[str]:
    errs = []
    fx = course / "fixtures"
    man = fx / "MANIFEST.tsv"
    rows: dict[str, list[str]] = {}
    if man.is_file():
        for line in man.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                cols = line.split("\t")
                rows[cols[0]] = cols
    total = 0
    if fx.is_dir():
        for p in sorted(fx.rglob("*")):
            if not p.is_file() or p.name in ("MANIFEST.tsv", "ASSETS.tsv", "README.md"):
                continue
            rel = p.relative_to(course.parent).as_posix()
            size = p.stat().st_size
            total += size
            if size > 8 * MiB:
                errs.append(
                    f"{rel}: {size / MiB:.1f} MiB is over the 8 MiB per-file budget"
                )
            row = rows.get(rel)
            if row is None:
                errs.append(f"{rel}: no row in course/fixtures/MANIFEST.tsv")
            elif len(row) > 1 and row[1] != hashlib.sha256(p.read_bytes()).hexdigest():
                errs.append(
                    f"{rel}: sha256 differs from MANIFEST.tsv (regenerate with the oracle, update the row)"
                )
    if total > 56 * MiB:  # D52: 50 MiB plus B14's multimodal fixtures
        errs.append(
            f"committed fixtures total {total / MiB:.1f} MiB, over the 56 MiB budget"
        )
    for m in mods:
        for f in m.fixtures.get("files", []):
            if f not in rows:
                errs.append(f"{m.id}: fixture {f} has no MANIFEST.tsv row")
            elif not (course.parent / f).is_file():
                errs.append(f"{m.id}: fixture {f} is missing")
    return errs


def spec_errors(reg, course: Path) -> tuple[list[str], int]:
    """Milestones, drills, and conformance cases parse; every committed fixture
    a milestone names (`{fixture:...}`, `course/fixtures/...`) has a MANIFEST row;
    every `requires` id is a module or milestone."""
    from .. import conform, drills, milestones

    errs: list[str] = []
    rows = set()
    man = course / "fixtures" / "MANIFEST.tsv"
    if man.is_file():
        rows = {
            x.split("\t")[0]
            for x in man.read_text().splitlines()
            if x.strip() and not x.startswith("#")
        }
    ms_ids = milestones.all_ids(course)
    n = 0

    def known(i: str) -> bool:
        return i in reg.modules or i in ms_ids

    for msid in ms_ids:
        n += 1
        try:
            ms = milestones.load(course, msid)
        except Exception as e:  # report every broken spec, not just the first
            errs.append(str(e))
            continue
        errs += [
            f"{msid}: requires {r!r}, which is not a module or milestone"
            for r in ms.requires
            if not known(r)
        ]
        errs += [
            f"{msid}: includes {i!r}, which is not a milestone"
            for i in ms.includes
            if i not in ms_ids
        ]
        for st in ms.steps:
            if st.expect.get("match") == "tokens-equal" and not st.expect.get(
                "near_tie"
            ):
                # Without a near_tie rerun the oracle must be margin-filtered.
                for v in st.variants() or [{}]:
                    f = st.expect.get("file", "")
                    try:
                        f2 = placeholders.expand(
                            f, lambda k, v=v: str(v[k]) if k in v else None
                        )
                    except HarnessError:
                        continue
                    fp = course.parent / f2 if f2.startswith("course/") else course / f2
                    try:
                        doc = json.loads(fp.read_text())
                    except (OSError, ValueError):
                        continue
                    margins = doc.get("margins") if isinstance(doc, dict) else None
                    if isinstance(margins, list) and margins and min(margins) < 1e-3:
                        errs.append(
                            f"{msid} step {st.name!r}: {f2} holds a near tie (top-2 margin {min(margins):.3g} < 1e-3); "
                            "oracle files keep only margin-filtered prompts (5.7)"
                        )
            blob = json.dumps([st.argv, st.expect, st.http])
            refs = [
                f"course/fixtures/{x}" for x in re.findall(r"\{fixture:([^}]+)\}", blob)
            ]
            refs += re.findall(r"\b(course/fixtures/[^\"\s]+)", blob)
            for v in st.variants() or [{}]:
                for f in refs:
                    try:
                        f2 = placeholders.expand(
                            f, lambda k, v=v: str(v[k]) if k in v else None
                        )
                    except HarnessError as e:
                        errs.append(f"{msid} step {st.name!r}: fixture path {f}: {e}")
                        continue
                    if f2 not in rows:
                        errs.append(
                            f"{msid} step {st.name!r}: fixture {f2} has no MANIFEST.tsv row"
                        )
    for d in (
        (course / "drills").glob("*/drill.toml") if (course / "drills").is_dir() else []
    ):
        n += 1
        try:
            dr = drills.load_file(d)
            errs += [
                f"drill {dr.name}: requires {r!r}, which is not a module or milestone"
                for r in dr.requires
                if not known(r)
            ]
        except Exception as e:
            errs.append(str(e))
    try:
        n += len(conform.load_cases(course))
    except Exception as e:
        errs.append(str(e))
    from .. import assets, parity

    try:
        pinned = {r.asset for r in assets.load(course)}
    except HarnessError as e:
        pinned = set()
        errs.append(str(e))
    for msid in ms_ids:
        try:
            blob = milestones.path_of(course, msid).read_text()
        except OSError:
            continue
        for a in sorted(set(re.findall(r"\{asset:([^/}]+)", blob))):
            if a not in pinned:
                errs.append(
                    f"{msid}: {{asset:{a}/...}} has no row in course/fixtures/ASSETS.tsv"
                )
    try:
        suites = parity.load_all(course)
    except Exception as e:
        suites = []
        errs.append(str(e))
    for su in suites:
        n += 1
        live = [im for im in su.impls if im.module in reg.modules]
        for im in live:
            if not (parity.suite_dir(course) / im.driver).is_file():
                errs.append(
                    f"parity {su.id}: {im.module} is in the registry, but its driver {im.driver} is missing"
                )
        if live and su.golden:
            if not (course.parent / su.golden).is_file():
                errs.append(f"parity {su.id}: golden {su.golden} is missing")
            elif su.golden not in rows:
                errs.append(
                    f"parity {su.id}: golden {su.golden} has no MANIFEST.tsv row"
                )
    return errs, n


_REF_LEARNER: dict[str, Path] = {}


def _practice_env(course: Path, lr: Path, scratch: Path) -> dict:
    from . import e2e

    env = e2e.learner_env(course, lr, scratch)
    env.update(
        {
            "OL_COURSE_TREE": str(course),
            "OL_SMOKE": "1",
            "OL_SEED": "0",
            "TINYLLM_FIXTURES": str(course / "fixtures"),
            "TINYLLM_CACHE": str(ctx.cache_dir()),
            "OL_BIN": str(ctx.root() / "practice" / "bin" / "ol"),
        }
    )
    return env


def reference_learner(reg, course: Path, work: Path) -> tuple[Path | None, str]:
    """One learner assembled from the whole reference per verify run (e2e.assemble)."""
    from . import e2e

    if "lr" in _REF_LEARNER:
        return _REF_LEARNER["lr"], ""
    tmp = Path(tempfile.mkdtemp(prefix="practice-ref-", dir=work))
    lr = tmp / "learner"
    env = _practice_env(course, lr, tmp / "scratch")
    ok, why = e2e.assemble(reg, course, lr, env)
    if not ok:
        return None, why
    # Built as a learner's repo is by the time its practice checks run, using
    # the reference [build].steps (native C artifacts, engine, gateway, and
    # the served model).
    from .. import services, system

    if (lr / "system.toml").is_file():
        ok, why = services.run_build(system.load(lr), tmp / "build.log", env)
        if not ok:
            return None, why
    _REF_LEARNER["lr"] = lr
    return lr, ""


def verify_smoke(reg, course: Path, mid: str, work: Path, rep: Report) -> None:
    """Run dependent checks in the same reference context as full checks."""
    m = reg.get(mid)
    ctx.say(f"{ctx.BLD}smoke {mid}{ctx.RST}")
    if m.kind == "practice":
        lr, why = reference_learner(reg, course, work)
        if lr is None:
            rep.fail(1, "assemble the reference learner", why)
            return
        env = _practice_env(course, lr, lr.parent / "scratch")
        rc, out = ctx.run(
            [str(course / "tests" / mid / "check")],
            cwd=lr,
            env=env,
            timeout=m.timeout_s,
        )
        if rc:
            rep.fail(1, "dependent smoke tests", out)
        else:
            rep.ok(1, "reference artifact smoke check")
        return
    sources = {x: "ref" for x in [mid] + reg.closure(mid)}
    rs = _ov(reg, course, mid, sources, work, "smoke").run_tests(mid, m.smoke)
    if all(r.ok for r in rs):
        rep.ok(1, f"smoke: {', '.join(m.smoke)}")
    else:
        rep.fail(
            1, "dependent smoke tests", "\n".join(r.output for r in rs if not r.ok)
        )


def verify_practice(reg, course: Path, m, work: Path, rep: Report) -> None:
    """Checks 1 and 2 for a practice module with an executable artifact check
    (H15): it passes in a learner assembled from the reference, and fails in a
    fresh learner where only `ol start <ID>` ran. Run with OL_SMOKE=1: cluster
    tiers belong to the milestone's kind steps (the nightly kind job)."""
    from . import e2e

    check = course / "tests" / m.id / "check"
    if not check.is_file():
        rep.skip(1, "practice: no executable artifact check")
        rep.skip(2, "practice: no executable artifact check")
        rep.skip(3, "practice: no course tests to repeat")
        return
    lr, why = reference_learner(reg, course, work)
    first: tuple[int, str] | None = None
    if lr is None:
        rep.fail(1, "assemble the reference learner", why)
    else:
        env = _practice_env(course, lr, lr.parent / "scratch")
        rc, out = ctx.run([str(check)], cwd=lr, env=env, timeout=m.timeout_s)
        first = (rc, out)
        rep.ok(
            1, "reference artifacts pass the check (OL_SMOKE=1)"
        ) if rc == 0 else rep.fail(
            1, f"reference artifacts fail the check (exit {rc})", out
        )
    tmp = Path(tempfile.mkdtemp(prefix=f"practice-{m.id}-", dir=work))
    fresh = tmp / "learner"
    env = _practice_env(course, fresh, tmp / "scratch")
    rc, out = e2e._ss(env, "course", "init", "--name", "forge")
    ok, why = (rc == 0, out)
    if ok:
        ok, why = e2e._start_in_process(env, [m.id])
    if ok:
        rc, why = e2e.commit_all(fresh, f"chore: start {m.id}")
        ok = rc == 0
    if not ok:
        rep.fail(2, "set up a fresh learner", why)
    else:
        rc, out = ctx.run([str(check)], cwd=fresh, env=env, timeout=m.timeout_s)
        rep.ok(
            2, f"a fresh start fails the check (exit {rc})"
        ) if rc != 0 else rep.fail(
            2,
            "the check passes on a fresh start: it does not exercise the artifacts",
            out,
        )
    shutil.rmtree(tmp, ignore_errors=True)
    if first is None:
        rep.skip(3, "practice: no reference run to repeat")
        return
    env = _practice_env(course, lr, lr.parent / "scratch")
    rc, out = ctx.run([str(check)], cwd=lr, env=env, timeout=m.timeout_s)
    a, b = _result_lines(first[1]), _result_lines(out)
    if (rc, b) == (first[0], a):
        rep.ok(
            3, f"deterministic over 2 runs of the check ({len(a)} result line(s) equal)"
        )
    else:
        diff = "\n".join(difflib.unified_diff(a, b, "run 1", "run 2", lineterm=""))
        rep.fail(
            3, f"the check is not deterministic (exit {first[0]}, then {rc})", diff
        )


_RESULT = re.compile(
    r"^\s*(ok|pass|fail|skip|PASS|FAIL|SKIP|FAILED|ERROR)\b|\b\d+ (passed|failed|skipped)\b"
)
_TIMING = re.compile(r"\(?\b\d+(\.\d+)?\s?(s|ms)\b\)?|\bin \d+(\.\d+)?s\b")


def _result_lines(out: str) -> list[str]:
    """The per-test result lines of a check's output, timings removed."""
    return [
        " ".join(_TIMING.sub("", line).split())
        for line in out.splitlines()
        if _RESULT.search(line)
    ]


def verify_module(reg, course, mid: str, work: Path, runs: int, rep: Report) -> None:
    m = reg.get(mid)
    ctx.say(
        f"{ctx.BLD}verify {mid}{ctx.RST}  {m.title}  ({m.kind}, {', '.join(m.lang)})"
    )
    closure = reg.closure(mid)
    all_ref = {x: "ref" for x in [mid] + closure}
    code = m.kind in ("build", "side") and bool(m.owned)
    probe = _ov(reg, course, mid, all_ref, work, "ref")
    test_langs = probe.test_langs(mid)

    # 5 first: stubs are meaningless over broken markers.
    errs = check_markers(course, reg, m) if m.owned else []
    rep.fail(5, "markers", "\n".join(errs)) if errs else (
        rep.ok(5, "markers: owner ids, whole function bodies")
        if m.owned
        else rep.skip(5, "no units")
    )
    markers_ok = not errs

    if m.kind == "practice":
        verify_practice(reg, course, m, work, rep)
    elif not code:
        rep.skip(1, f"{m.kind}: no reference code")
        rep.skip(2, f"{m.kind}: no stubs")
        rep.skip(3, f"{m.kind}: no course tests to repeat")
    elif not test_langs:
        rep.fail(1, f"no course tests found for {mid}")
    else:
        r1 = probe.run_tests(mid)
        if all(r.ok for r in r1):
            rep.ok(1, f"reference passes ({', '.join(r.lang for r in r1)})")
        else:
            rep.fail(
                1,
                "reference fails its own tests",
                "\n".join(r.output for r in r1 if not r.ok),
            )
        if markers_ok:
            stubbed = dict(all_ref)
            stubbed[mid] = "stub"
            sov = _ov(reg, course, mid, stubbed, work, "stub")
            good, msg = _compile(sov, _unit_langs(m))
            if not good:
                rep.fail(2, "stubbed tree does not compile", msg)
            else:
                r2 = sov.run_tests(mid)
                passing = [r.lang for r in r2 if r.ok]
                if passing:
                    rep.fail(
                        2,
                        f"tests pass on the stub ({', '.join(passing)}): they do not exercise {mid}",
                    )
                else:
                    rep.ok(2, "stub compiles and fails the tests")
        else:
            rep.skip(2, "markers are broken")
        bad = []
        for seed in range(1, max(runs, 2)):
            r3 = _ov(
                reg, course, mid, all_ref, work, f"seed{seed}", seed=seed
            ).run_tests(mid)
            if [r.ok for r in r3] != [r.ok for r in r1]:
                bad.append(
                    f"seed {seed}: "
                    + ", ".join(f"{r.lang}={'pass' if r.ok else 'fail'}" for r in r3)
                )
        rep.fail(3, "nondeterministic", "\n".join(bad)) if bad else rep.ok(
            3, f"deterministic over {max(runs, 2)} seeded runs"
        )

    if "python" in _unit_langs(m):
        errs = check_pyi(course, reg, m)
        rep.fail(4, "reference vs .pyi", "\n".join(errs)) if errs else rep.ok(
            4, "Python references match their contracts"
        )

    if code:
        check_mutants(course, reg, m, work, rep)
    if m.kind in ("solve", "proof"):
        from .. import solve

        errs = solve.verify_key(course, m.id)
        rep.fail(7, "solve key", "\n".join(errs)) if errs else rep.ok(
            7, "solve key parses; every expect passes and every reject canary fails"
        )

    errs = chapter.lint(reg, m, course.parent, course, probe.test_dir(mid))
    rep.fail(8, "chapter", "\n".join(errs)) if errs else rep.ok(
        8, f"chapter {m.chapter}"
    )

    errs = registry.call_site_errors(reg, m)
    rep.fail(9, "call site", "\n".join(errs)) if errs else (
        rep.ok(9, "call site " + ", ".join(m.used_by)) if m.kind == "build" else None
    )

    errs = check_annotations(course, reg, m, probe.test_dir(mid))
    rep.fail(10, "test annotations", "\n".join(errs)) if errs else rep.ok(
        10, "tests annotated, frozen helpers only"
    )

    errs = fixture_errors(course, [m]) if m.fixtures.get("files") else []
    errs = [e for e in errs if e.startswith(m.id)]
    rep.fail(12, "fixtures", "\n".join(errs)) if errs else None

    errs = [
        e
        for e in registry.invariants(reg)
        if re.search(r"(^|\W)" + re.escape(mid) + r"(\W|$)", e)
    ]
    rep.fail(13, "registry", "\n".join(errs)) if errs else rep.ok(
        13, "registry invariants"
    )

    if code and markers_ok:
        errs = check_seams_static(course, reg, m)
        nb = {x: "stub" for x in reg.modules}
        nb[mid] = "ref"
        good, msg = _compile(_ov(reg, course, mid, nb, work, "seams"), _unit_langs(m))
        if not good:
            errs.append("does not compile against stubbed neighbours:\n" + msg)
        rep.fail(14, "seams", "\n".join(errs)) if errs else rep.ok(
            14, "seams: contract names only, compiles against stubbed neighbours"
        )


def verify_global(reg, course, work: Path, rep: Report) -> None:
    ctx.say(f"{ctx.BLD}verify course (global){ctx.RST}  {course}")
    errs = registry.invariants(reg)
    if not registry.tsv_current(reg):
        errs.append("modules.tsv is not current: run `ol lint --fix-index`")
    rep.fail(13, "registry", "\n".join(errs)) if errs else rep.ok(
        13, f"registry: {len(reg.modules)} modules, modules.tsv current"
    )

    inc = course / "contracts" / "c" / "include"
    bad = []
    for h in sorted(inc.rglob("*.h")) if inc.is_dir() else []:
        rel = h.relative_to(inc).as_posix()
        p = work / "hdr" / (rel.replace("/", "_") + ".c")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'#include "{rel}"\n')
        rc, out = ctx.run(
            [
                "cc",
                "-std=c11",
                "-pedantic",
                "-Wall",
                "-Werror",
                f"-I{inc}",
                "-fsyntax-only",
                str(p),
            ]
        )
        if rc != 0:
            bad.append(f"{rel}:\n{out}")
    gomod = course / "contracts" / "go"
    if (gomod / "go.mod").is_file():
        env = ctx.base_env()
        env["GOWORK"] = "off"
        rc, out = ctx.run(["go", "build", "./..."], cwd=gomod, env=env)
        if rc != 0:
            bad.append(f"contracts/go:\n{out}")
    tlc = course / "contracts" / "rust" / "tl-contracts"
    if (tlc / "Cargo.toml").is_file():
        env = ctx.base_env()
        env["CARGO_TARGET_DIR"] = str(work / "target")
        rc, out = ctx.run(
            ["cargo", "build", "-q", "--manifest-path", str(tlc / "Cargo.toml")],
            env=env,
        )
        if rc != 0:
            bad.append(f"contracts/rust/tl-contracts:\n{out}")
    tlp = course / "contracts" / "rust" / "tl-proto"
    if (tlp / "Cargo.toml").is_file():
        # Build a copy so Cargo.lock never lands in the course tree.
        cp = work / "tl-proto"
        shutil.rmtree(cp, ignore_errors=True)
        shutil.copytree(tlp, cp)
        env = ctx.base_env()
        env["CARGO_TARGET_DIR"] = str(work / "target")
        rc, out = ctx.run(
            ["cargo", "build", "-q", "--manifest-path", str(cp / "Cargo.toml")],
            env=env,
            timeout=900,
        )
        if rc != 0:
            bad.append(f"contracts/rust/tl-proto:\n{out}")
    gen = course / "oracle" / "contracts" / "gen-proto.sh"
    buf = shutil.which("buf") or str(Path.home() / "go" / "bin" / "buf")
    if gen.is_file() and Path(buf).is_file():
        env = ctx.base_env()
        env["PATH"] = str(Path(buf).parent) + os.pathsep + env.get("PATH", "")
        rc, out = ctx.run(["bash", str(gen), "--check"], env=env, timeout=900)
        if rc != 0:
            bad.append(f"generated proto code (gen-proto.sh --check):\n{out}")
    from .. import schema as jschema

    nex = 0
    fmt = course / "contracts" / "formats"
    for sp in sorted(fmt.glob("*.schema.json")) if fmt.is_dir() else []:
        try:
            sch = json.loads(sp.read_text())
        except json.JSONDecodeError as e:
            bad.append(f"{sp.relative_to(course)}: {e}")
            continue
        for i, ex in enumerate(sch.get("examples", [])):
            nex += 1
            errs = jschema.validate(ex, sch, sch)
            if errs:
                bad.append(
                    f"{sp.relative_to(course)} example {i + 1}:\n"
                    + "\n".join(errs[:10])
                )
    rep.fail(4, "contracts", "\n".join(bad)) if bad else rep.ok(
        4,
        "headers compile standalone (-pedantic -Werror); contract crates and modules build; "
        f"formats/*.schema.json validate their {nex} example(s)",
    )
    if (course / "contracts" / "openapi").is_dir() and shutil.which(
        "openapi-spec-validator"
    ) is None:
        rep.skip(4, "contracts/openapi: `openapi-spec-validator` not installed")
    if (course / "contracts" / "proto").is_dir() and not Path(buf).is_file():
        rep.skip(4, "contracts/proto: `buf` not installed, generated code not compared")

    all_units = reg.all_units()
    langs = {markers.lang_of(u) for u in all_units} - {None}
    if langs:
        stub_all = {x: "stub" for x in reg.modules}
        anyid = next(iter(reg.modules))
        good, msg = _compile(
            _ov(reg, course, anyid, stub_all, work, "stubbed-tree"),
            langs,
            all_tests=True,
        )
        rep.fail(
            2, "the fully stubbed reference tree does not compile", msg
        ) if not good else rep.ok(
            2,
            f"fully stubbed reference tree compiles with every course test ({', '.join(sorted(langs))})",
        )
        # Ownership across history snapshots: each snapshot carries its owner's id.
        snap_errs = []
        hist = course / "ref" / "history"
        for p in sorted(hist.rglob("*")) if hist.is_dir() else []:
            if p.is_file():
                owner, unit = (
                    p.relative_to(hist).parts[0],
                    "/".join(p.relative_to(hist).parts[1:]),
                )
                chain = reg.unit_chain(unit) if unit in all_units else []
                if owner not in chain[:-1]:
                    snap_errs.append(
                        f"ref/history/{owner}/{unit}: {owner} is not an earlier owner of {unit} (chain {chain})"
                    )
        for u, chain in all_units.items():
            for prev in chain[:-1]:
                if not (hist / prev / u).is_file():
                    snap_errs.append(
                        f"{u}: {chain[-1]} took it over, but ref/history/{prev}/{u} is missing"
                    )
        rep.fail(5, "history snapshots", "\n".join(snap_errs)) if snap_errs else rep.ok(
            5, "history snapshots match the owner chains"
        )

    errs = fixture_errors(course, reg.modules.values())
    rep.fail(12, "fixtures", "\n".join(errs)) if errs else rep.ok(
        12, "fixtures: manifest rows, hashes, budgets"
    )
    errs, n = spec_errors(reg, course)
    rep.fail(
        12, "milestone, drill, and conformance specs", "\n".join(errs)
    ) if errs else rep.ok(
        12,
        f"{n} milestone, drill, conformance, and parity spec(s) parse; their fixtures resolve",
    )

    ss = ctx.root() / "practice" / "bin" / "ol"
    env = ctx.base_env()
    env["OL_COURSE_ROOT"] = str(course)
    env["OL_PATHS_DIR"] = str(paths.paths_dir(course))
    rc, out = ctx.run(["bash", str(ss), "learn", "--verify"], env=env)
    rep.fail(11, "paths (ol learn --verify)", out) if rc != 0 else rep.ok(
        11, "paths: ol learn --verify"
    )
    order = stage_order_errors(reg, course)
    rep.fail(11, "path stage order", "\n".join(order)) if order else None

    errs = []
    for m in reg.ordered():
        if m.chapter and (course.parent / m.chapter).is_file():
            errs += chapter.no_em_dash(
                (course.parent / m.chapter).read_text(), m.chapter
            )
    rep.fail(8, "em dashes", "\n".join(errs)) if errs else None


def stage_order_errors(reg, course: Path) -> list[str]:
    """Check 11: in every path, a module's stage comes after its deps' stages."""
    base = paths.paths_dir(course)
    errs: list[str] = []
    for tsv in sorted(base.glob("*/path.tsv")) if base.is_dir() else []:
        name = tsv.parent.name
        try:
            rows = paths.expand(name, base)
        except (FileNotFoundError, ValueError):
            continue  # ol learn --verify reports broken includes
        first: dict[str, int] = {}
        for i, r in enumerate(rows):
            for kind, t in paths.check_targets(r.check):
                if kind in ("module", "solve"):
                    first.setdefault(t, i)
        for mid, i in first.items():
            if mid not in reg.modules:
                continue
            for d in reg.get(mid).deps:
                if d in first and first[d] > i:
                    errs.append(
                        f"paths/{name}: {mid} (row {i + 1}) comes before its dep {d} (row {first[d] + 1})"
                    )
    return errs


def changed_modules(reg, course: Path, ref: str) -> tuple[list[str], bool]:
    repo = ctx.git_toplevel(course)
    if repo is None:
        return sorted(reg.modules), True
    files: set[str] = set()
    for args in (
        ["diff", "--name-only", f"{ref}...HEAD"],
        ["diff", "--name-only", "HEAD"],
        ["ls-files", "--others", "--exclude-standard"],
    ):
        rc, out = ctx.git(args, repo)
        if rc != 0 and args[0] == "diff" and "..." in args[2]:
            raise ValueError(f"git diff {ref}...HEAD failed:\n{out}")
        files |= {x.strip() for x in out.splitlines() if x.strip()}
    croot = course.relative_to(repo).as_posix()
    under = {f[len(croot) + 1 :] for f in files if f.startswith(croot + "/")}
    harness = any(f.startswith(("course/harness/", "practice/bin/ol")) for f in files)
    by_us = {ids.underscore(m): m for m in reg.modules}
    hits: set[str] = set()
    for f in under:
        parts = f.split("/")
        if parts[0] == "modules" and f.endswith(".toml"):
            hits.add(Path(f).stem)
        elif parts[0] == "ref" and len(parts) > 2 and parts[1] == "history":
            hits.add(parts[2])
        elif parts[0] == "ref":
            unit = "/".join(parts[1:])
            hits |= set(reg.unit_chain(unit)) if unit in reg.all_units() else set()
        elif parts[0] == "tests" and len(parts) > 2 and parts[1] in ("go", "rust"):
            key = parts[2].removesuffix(".rs")
            if key in by_us:
                hits.add(by_us[key])
        elif parts[0] in ("tests", "fixtures", "mutants", "solve") and len(parts) > 1:
            hits.add(parts[1])
        elif parts[0] == "contracts":
            hits |= {
                m.id
                for m in reg.modules.values()
                if any(c.removeprefix("contracts/") in f for c in m.contract)
            }
    for f in files:
        hits |= {m.id for m in reg.modules.values() if m.chapter == f}
    return sorted(h for h in hits if h in reg.modules), harness


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "course":
        print(__doc__)
        return EXIT_HARNESS
    ap = argparse.ArgumentParser(prog="ol verify course")
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--changed")
    ap.add_argument("--global", dest="global_only", action="store_true")
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--kind", action="store_true")
    ap.add_argument(
        "--keep",
        type=Path,
        help="with --e2e: assemble the reference learner here and keep it",
    )
    ap.add_argument(
        "--assemble",
        type=Path,
        help="only assemble the reference learner here (CI deploys it to kind)",
    )
    ap.add_argument(
        "--record-thresholds",
        action="store_true",
        help="run the reference tests of the given modules over 5 seeds and write "
        "the learning tests' ref-thresholds.tsv rows (_lib.thresholds.check)",
    )
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--shard",
        help="I/N: verify every N-th target from the I-th (0-based); the global "
        "checks run in shard 0 only. CI splits a large --changed set this way.",
    )
    a = ap.parse_args(argv[1:])
    a.shard_i, a.shard_n = 0, 1
    if a.shard:
        try:
            i, n = (int(x) for x in a.shard.split("/"))
        except ValueError:
            raise HarnessError(f"--shard {a.shard}: want I/N, e.g. 0/4") from None
        if not (n >= 1 and 0 <= i < n):
            raise HarnessError(f"--shard {a.shard}: want 0 <= I < N")
        a.shard_i, a.shard_n = i, n
    if a.keep and not (a.e2e or a.kind):
        raise HarnessError("--keep goes with --e2e or --kind")
    ct = tree.resolve(None)
    course = ct.path
    reg = registry.load(course)
    work = ctx.scratch() / ".course-verify"
    work.mkdir(parents=True, exist_ok=True)
    runs = 3 if a.nightly else 2
    if a.assemble:
        from . import e2e

        dest = a.assemble.resolve()
        if dest.exists() and any(dest.iterdir()):
            raise HarnessError(
                f"--assemble {dest}: the directory exists and is not empty"
            )
        ok, what = e2e.assemble(
            reg, course, dest, e2e.learner_env(course, dest, work / "assemble-scratch")
        )
        if not ok:
            ctx.err(f"assembling the reference learner failed:\n{what}")
            return EXIT_FAIL
        ctx.say(f"{ctx.GRN}assembled{ctx.RST} the reference learner at {dest}: {what}")
        return 0
    if a.record_thresholds:
        if not a.ids:
            raise HarnessError("--record-thresholds needs module ids")
        return record_thresholds(reg, course, work, a.ids, a.seeds)
    if a.e2e or a.kind:
        from .. import milestones
        from . import e2e

        rep = Report()
        only = [x for x in a.ids if x.startswith("MS-")]
        if len(only) != len(a.ids):
            raise HarnessError(
                "--e2e and --kind take milestone ids only (MS-<name>), to run just those"
            )
        unknown = sorted(set(only) - set(milestones.all_ids(course)))
        if unknown:
            raise HarnessError(f"no such milestone: {', '.join(unknown)}")
        e2e.run(reg, course, work, rep, kind=a.kind, keep=a.keep, only=only or None)
        if rep.failures:
            ctx.say(f"{ctx.RED}{rep.failures} check(s) failed{ctx.RST}")
            return EXIT_FAIL
        ctx.say(
            f"{ctx.GRN}ol verify course --{'kind' if a.kind else 'e2e'}: ok{ctx.RST}"
        )
        return 0
    with ctx.lock(work):
        return _verify(a, reg, course, ct, work, runs)


def record_thresholds(
    reg, course: Path, work: Path, mids: list[str], seeds: int
) -> int:
    """Learning tests: run each module's reference tests with OL_SEED = 0..seeds-1
    while `_lib.thresholds.check` records instead of asserting, then write
    mean +/- 3 sd rows (DESIGN 5.11 `ref-thresholds`)."""
    import os

    from .. import thresholds

    obs = work / "thresholds.jsonl"
    all_rows = []
    for mid in mids:
        m = reg.get(mid)
        sources = {x: "ref" for x in [m.id] + reg.closure(m.id)}
        obs.unlink(missing_ok=True)
        os.environ["OL_RECORD_THRESHOLDS"] = str(obs)
        try:
            for sd in range(seeds):
                runs = _ov(
                    reg, course, mid, sources, work, f"thr{sd}", seed=sd
                ).run_tests(mid)
                if not all(r.ok for r in runs):
                    ctx.err(
                        f"{mid} seed {sd}: the reference tests fail\n"
                        + "\n".join(r.output for r in runs if not r.ok)
                    )
                    return EXIT_FAIL
        finally:
            os.environ.pop("OL_RECORD_THRESHOLDS", None)
        lines = obs.read_text().splitlines() if obs.is_file() else []
        if not lines:
            ctx.err(f"{mid}: no test called _lib.thresholds.check")
            return EXIT_FAIL
        all_rows += thresholds.from_observations(lines, "full")
    p = thresholds.write(ctx.live_course(), all_rows)
    for row in all_rows:
        ctx.say(
            f"  {row.key} {row.metric}: mean {row.mean:.6g}, sd {row.sd:.3g}, threshold {row.threshold:.6g} ({row.n} seeds)"
        )
    ctx.say(f"{ctx.GRN}wrote{ctx.RST} {len(all_rows)} row(s) to {p}")
    return 0


def _verify(a, reg, course, ct, work: Path, runs: int) -> int:
    rep = Report()
    smoke_only: list[str] = []
    if a.global_only:
        targets, glob = [], True
    elif a.changed:
        targets, harness = changed_modules(reg, course, a.changed)
        if harness:
            targets = sorted(reg.modules)
        glob = True
        smoke_only = sorted(
            {d for t in targets for d in reg.dependents(t)} - set(targets)
        )
        ctx.say(
            f"changed since {a.changed}: {', '.join(targets) or 'no modules'}"
            + (f"; dependents (smoke): {', '.join(smoke_only)}" if smoke_only else "")
        )
    elif a.ids:
        targets, glob = [reg.get(i).id for i in a.ids], False
    else:
        targets, glob = [m.id for m in reg.ordered()], True
    if a.shard_n > 1:
        targets = [t for k, t in enumerate(targets) if k % a.shard_n == a.shard_i]
        smoke_only = [t for k, t in enumerate(smoke_only) if k % a.shard_n == a.shard_i]
        glob = glob and a.shard_i == 0
        ctx.say(f"shard {a.shard_i}/{a.shard_n}: {', '.join(targets) or 'no modules'}")
    if glob:
        verify_global(reg, course, work, rep)
    for mid in targets:
        verify_module(
            reg, course, mid, work, max(runs, reg.get(mid).determinism_runs), rep
        )
    for mid in smoke_only:
        m = reg.get(mid)
        if not m.smoke:
            continue
        verify_smoke(reg, course, mid, work, rep)
    if ct.tainted:
        ctx.say(
            f"{ctx.DIM}note: course/ has uncommitted changes (verdicts would be tagged tainted){ctx.RST}"
        )
    if rep.failures:
        ctx.say(f"{ctx.RED}{rep.failures} check(s) failed{ctx.RST}")
        return EXIT_FAIL
    ctx.say(f"{ctx.GRN}ol verify course: ok{ctx.RST}")
    return 0
