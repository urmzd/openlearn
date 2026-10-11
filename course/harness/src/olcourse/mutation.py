"""Grading the learner's tests by mutation (DESIGN 5.6, 5.12).

A graded test suite runs against OUR reference with one planted fault, never
against the learner's implementation, so the grade measures only the tests:

  0. black-box lint: graded tests touch only the contract
  1. baseline A: the tests pass against the unmutated reference (deps from the
     reference); a failure is "your test rejects a correct implementation"
  2. baseline B: the tests pass against the learner's own implementation
  3. budget: baseline A within `time_budget_s`; each mutant times out at
     max(2 s, 10 x baseline A)
  4. each mutant: reference + patch, only the graded tests; killed = nonzero
     exit or timeout (reported apart). A mutant whose unit does not compile is
     `invalid` and leaves the denominator.
  5. score = killed / total; pass = score >= threshold and every required
     mutant killed

Mutant classes (manifest `tier`): `auto` and `semantic` are unit mutants;
`resilience` mutants run the suite in [learner_tests.resilience] (default the
graded path) and are killed by a failure; `perf` mutants (a 2x slowdown) must
trip the benchmark gate in [learner_tests.perf]; `model` mutants must be
flagged over 5 seeds at p < 0.05 by the eval command in [learner_tests.model]
(exact one-sided permutation test on its metric); `agent` mutants must move
the eval metric in [learner_tests.agent] so the two 95% CIs do not overlap.

Results are cached in `<learner>/.ol/cache/mutation.json` keyed by
(hash of the learner's test files, hash of the reference unit, hash of the
patch, class config); the full grade is cached by the test hash, so the
full cost is paid only when the tests change. Rust and C mutants run one at a
time over the shared warm target dir and object cache; Python and Go run in
parallel (-j).
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import HarnessError, ctx, markers, units
from .overlay import Overlay, TestRun, graded_files
from .registry import Module, Registry

CAP = 40
SAMPLE = 8
CLASSES = {
    "auto": "unit",
    "semantic": "unit",
    "resilience": "resilience",
    "perf": "perf",
    "model": "model",
    "agent": "agent",
}


def rung_threshold(rung: int) -> float:
    if rung >= 7:
        return 0.90
    if rung >= 4:
        return 0.80
    if rung == 3:
        return 0.70
    return 0.60


def _jobs(asked: int) -> int:
    """Worker threads for learner-test grading. OL_MUTATION_JOBS (for
    example 1 on a loaded machine) overrides the default of 4; an explicit
    `--jobs` wins over both."""
    if asked > 0:
        return asked
    try:
        return max(1, int(os.environ.get("OL_MUTATION_JOBS", "4")))
    except ValueError:
        return 4


@dataclass
class Mutant:
    mid: str
    unit: str
    tier: str
    operator: str
    line: str
    required: bool
    public: str
    private: str

    @property
    def klass(self) -> str:
        return CLASSES.get(self.tier, "unit")

    def survivor_text(self, revealed: bool) -> str:
        if self.tier == "semantic" and not revealed:
            p = self.operator.removeprefix("pitfall-").removeprefix("pitfall")
            which = (
                f"Pitfall {p}"
                if p.isdigit()
                else (f"a chapter pitfall ({p})" if p else "a chapter pitfall")
            )
            return f"a planted bug from {which} survived"
        where = f"{self.unit}:{self.line}" if self.line not in ("", "-") else self.unit
        return f"{self.public} ({where})"


def manifest(course: Path, module_id: str) -> list[Mutant]:
    p = course / "mutants" / module_id / "manifest.tsv"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        c = line.split("\t")
        c += [""] * (8 - len(c))
        out.append(
            Mutant(
                c[0],
                c[1],
                c[2],
                c[3],
                c[4],
                c[5].strip().lower() in ("y", "yes", "true"),
                c[6],
                c[7],
            )
        )
    return out


@dataclass
class Spec:
    """[learner_tests] (DESIGN 3.4)."""

    rung: int
    path: str
    threshold: float
    required: list[str]
    time_budget_s: float
    classes: dict

    @classmethod
    def of(cls, m: Module) -> "Spec":
        lt = m.learner_tests or {}
        if not lt.get("path"):
            raise HarnessError(f"{m.id}: [learner_tests] needs `path`")
        rung = int(lt.get("rung", 2))
        return cls(
            rung=rung,
            path=str(lt["path"]).rstrip("/"),
            threshold=float(lt.get("threshold", rung_threshold(rung))),
            required=list(lt.get("required_mutants", [])),
            time_budget_s=float(lt.get("time_budget_s", 60)),
            classes={
                k: dict(v)
                for k, v in lt.items()
                if k in ("perf", "model", "agent", "resilience") and isinstance(v, dict)
            },
        )


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for x in parts:
        h.update(x if isinstance(x, bytes) else str(x).encode())
        h.update(b"\0")
    return h.hexdigest()


def tests_hash(root: Path, rel: str, extra: list[Path] = ()) -> str:
    files = graded_files(root, rel) + list(extra)
    base = root / rel
    if base.is_dir():  # helpers (conftest.py, shared .h) count too
        files += [
            p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        ]
    h = hashlib.sha256()
    for p in sorted(set(files)):
        h.update(
            p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0"
        )
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# black-box rule (DESIGN 5.6)


def _pyi_names(contracts: Path, module: str) -> set[str] | None:
    p = contracts / "py" / Path(*module.split("."))
    pyi = p.with_suffix(".pyi")
    if not pyi.is_file():
        return None
    tree = ast.parse(pyi.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


PKGS = ("tinyllm", "corpus")
C_INCLUDES = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
C_OK = re.compile(
    r"^(tinyllm\.h|tinyllm/[\w./-]+\.h|ol_test\.h|ol_prop\.h|ol_bench\.h)$"
)


def blackbox_errors(root: Path, rel: str, contracts: Path) -> list[str]:
    files = graded_files(root, rel)
    lang = markers.lang_of(rel) or rel.split("/", 1)[0]
    errs: list[str] = []
    for f in files:
        where = f.relative_to(root).as_posix()
        text = f.read_text(errors="replace")
        if lang == "python":
            try:
                tree = ast.parse(text)
            except SyntaxError as e:
                errs.append(f"{where}:{e.lineno}: {e.msg}")
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.split(".")[0] in PKGS
                ):
                    names = _pyi_names(contracts, node.module)
                    if names is None:
                        errs.append(
                            f"{where}:{node.lineno}: imports {node.module}, which has no contract in contracts/py"
                        )
                        continue
                    for a in node.names:
                        if a.name != "*" and a.name not in names:
                            errs.append(
                                f"{where}:{node.lineno}: imports {node.module}.{a.name}, which is not in its contract"
                            )
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if (
                            a.name.split(".")[0] in PKGS
                            and _pyi_names(contracts, a.name) is None
                        ):
                            errs.append(
                                f"{where}:{node.lineno}: imports {a.name}, which has no contract in contracts/py"
                            )
        elif lang == "go":
            m = re.search(r"^package\s+(\w+)", text, re.M)
            if not m or not m.group(1).endswith("_test"):
                errs.append(
                    f"{where}: graded Go tests are external: `package <pkg>_test`"
                )
        elif lang == "rust":
            if "tests" not in f.relative_to(root).parts:
                errs.append(
                    f"{where}: graded Rust tests are integration tests under a crate's tests/"
                )
        elif lang == "c":
            for inc in C_INCLUDES.findall(text):
                if not C_OK.match(inc):
                    errs.append(
                        f'{where}: #include "{inc}": graded C tests include only tinyllm/*.h and ol_*.h'
                    )
    return errs


# ---------------------------------------------------------------------------
# statistics for model and agent mutants


def permutation_p(ref: list[float], mut: list[float], higher_is_better: bool) -> float:
    """Exact one-sided permutation test: P(a relabelling makes the mutant look
    at least as much worse as it does). Worse = lower when higher is better."""
    sign = -1.0 if higher_is_better else 1.0
    obs = sign * (sum(mut) / len(mut) - sum(ref) / len(ref))
    pool = ref + mut
    n = len(mut)
    hits = total = 0
    for idx in itertools.combinations(range(len(pool)), n):
        s = set(idx)
        a = [pool[i] for i in idx]
        b = [pool[i] for i in range(len(pool)) if i not in s]
        stat = sign * (sum(a) / len(a) - sum(b) / len(b))
        hits += stat >= obs - 1e-12
        total += 1
    return hits / total


def _last_json(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        try:
            v = json.loads(line)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# the grade


@dataclass
class MutantResult:
    mid: str
    status: str  # killed timeout survived invalid error
    klass: str
    required: bool
    detail: str = ""
    cached: bool = False

    @property
    def killed(self) -> bool:
        return self.status in ("killed", "timeout")


@dataclass
class Grade:
    module: str
    passed: bool
    score: float
    killed: int
    total: int
    required_ok: bool
    threshold: float
    sampled: bool = False
    of: int = 0  # mutants in the manifest (capped)
    tests_hash: str = ""
    baseline_s: float = 0.0
    reason: str = ""
    results: list[MutantResult] = field(default_factory=list)

    def summary(self) -> dict:
        d = {
            "score": round(self.score, 4),
            "killed": self.killed,
            "total": self.total,
            "required_ok": self.required_ok,
            "threshold": self.threshold,
            "passed": self.passed,
            "tests_hash": self.tests_hash,
        }
        if self.sampled:
            d.update(sampled=True, of=self.of)
        if self.reason:
            d["reason"] = self.reason
        return d


class Cache:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            self.data = {}
        self.data.setdefault("mutants", {})
        self.data.setdefault("grades", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        tmp.replace(self.path)


@dataclass
class Grader:
    """One module's learner tests, graded against the reference."""

    reg: Registry
    course: Path
    m: Module
    tests_root: (
        Path  # the repo the graded tests live in (the learner, or a scratch copy)
    )
    work: Path  # overlay work dirs land under here
    target_dir: Path  # CARGO_TARGET_DIR and the C object cache (shared, warm)
    cache: Cache
    impl_sources: dict | None = None  # baseline B sources; None skips it
    seed: int = 0
    jobs: int = 0  # 0: OL_MUTATION_JOBS, else 4
    say: object = None

    def __post_init__(self):
        self.spec = Spec.of(self.m)
        self.lang = markers.lang_of(self.spec.path) or self.spec.path.split("/", 1)[0]
        self.ref_sources = {x: "ref" for x in [self.m.id] + self.reg.closure(self.m.id)}
        self.thash = tests_hash(self.tests_root, self.spec.path)
        for k, v in self.spec.classes.items():
            if v.get("path"):
                self.thash = _sha(self.thash, tests_hash(self.tests_root, v["path"]))
        self._say = self.say or (lambda *_: None)
        self._ref_metrics: dict[str, list] = {}

    # -- overlays ---------------------------------------------------------------

    def overlay(self, sources: dict, tag: str, patches: dict | None = None) -> Overlay:
        return Overlay(
            self.reg,
            self.course,
            self.m.id,
            sources,
            self.tests_root,
            self.work / "mutate" / self.m.id / tag,
            self.target_dir,
            dict(patches or {}),
            self.seed,
        )

    def ref_unit_text(self, unit: str) -> str:
        owner = next(
            o for o in reversed(self.reg.unit_chain(unit)) if o in self.ref_sources
        )
        return markers.drop_markers(units.ref_text(self.course, self.reg, unit, owner))

    def mutant_text(self, mu: Mutant) -> tuple[str | None, str]:
        patch = self.course / "mutants" / self.m.id / f"{mu.mid}.patch"
        if not patch.is_file():
            return None, f"{patch.name} missing"
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / mu.unit
            dst.parent.mkdir(parents=True)
            dst.write_text(self.ref_unit_text(mu.unit))
            rc, out = ctx.run(["patch", "-s", "-p1", "-d", td, "-i", str(patch)])
            if rc != 0:
                return None, f"patch does not apply: {out.strip()[:200]}"
            return dst.read_text(), ""

    # -- one run of the graded tests ------------------------------------------------

    def _path_for(self, klass: str) -> str:
        return str(self.spec.classes.get(klass, {}).get("path") or self.spec.path)

    def run(self, ov: Overlay, klass: str, timeout: float) -> TestRun:
        conf = self.spec.classes.get(klass, {})
        if klass in ("perf", "resilience") and conf.get("cmd"):
            rc, out = ov.run_cmd(list(conf["cmd"]), timeout)
            return TestRun(self.lang, rc == 0, rc != 2, out)
        return ov.run_learner_tests(self._path_for(klass), timeout)

    def metrics(self, ov: Overlay, klass: str, timeout: float) -> tuple[list, str]:
        """model and agent classes: run the eval command over seeds."""
        conf = self.spec.classes.get(klass) or {}
        cmd = conf.get("cmd")
        if not cmd:
            raise HarnessError(f"{self.m.id}: [learner_tests.{klass}] needs `cmd`")
        metric = str(conf.get("metric", "score"))
        seeds = int(conf.get("seeds", 5)) if klass == "model" else 1
        vals = []
        for sd in range(seeds):
            argv = [str(a).replace("{seed}", str(sd)) for a in cmd]
            rc, out = ov.run_cmd(argv, timeout, {"OL_SEED": str(sd)})
            obj = _last_json(out) if rc == 0 else None
            if obj is None or metric not in obj:
                vals.append(None)
                continue
            vals.append(obj if klass == "agent" else float(obj[metric]))
        return vals, ""

    # -- baselines -------------------------------------------------------------------

    def baselines(self) -> tuple[bool, str, float]:
        errs = blackbox_errors(self.tests_root, self.spec.path, self._contracts())
        if errs:
            return (
                False,
                "graded tests break the black-box rule (DESIGN 5.6):\n"
                + "\n".join(errs),
                0.0,
            )
        t0 = time.monotonic()
        r = self.run(
            self.overlay(self.ref_sources, "baseline-ref"),
            "unit",
            max(self.spec.time_budget_s * 3, 60),
        )
        took = time.monotonic() - t0
        if not r.ok:
            names = _failed_names(r.output)
            what = ", ".join(names) if names else "(see output)"
            return (
                False,
                f"your test rejects a correct implementation: {what}\n{ctx.tail(r.output, 30)}",
                took,
            )
        if took > self.spec.time_budget_s:
            return (
                False,
                f"the graded tests took {took:.1f}s against the reference; the budget is {self.spec.time_budget_s:g}s",
                took,
            )
        classes = {mu.klass for mu in self.mutants()} & {"perf", "resilience"}
        for k in sorted(classes):
            if k in self.spec.classes:
                rk = self.run(
                    self.overlay(self.ref_sources, f"baseline-{k}"),
                    k,
                    max(self.spec.time_budget_s * 3, 60),
                )
                if not rk.ok:
                    return (
                        False,
                        f"your {k} suite rejects the reference implementation\n{ctx.tail(rk.output, 30)}",
                        took,
                    )
        if self.impl_sources is not None:
            rb = self.run(
                self.overlay(self.impl_sources, "baseline-impl"),
                "unit",
                max(self.spec.time_budget_s * 3, 60),
            )
            if not rb.ok:
                return (
                    False,
                    f"your tests fail against your own implementation\n{ctx.tail(rb.output, 30)}",
                    took,
                )
        return True, "", took

    def _contracts(self) -> Path:
        c = self.tests_root / "contracts"
        return c if c.is_dir() else self.course / "contracts"

    # -- mutants ------------------------------------------------------------------------

    def key(self, mu: Mutant) -> str:
        patch = self.course / "mutants" / self.m.id / f"{mu.mid}.patch"
        conf = json.dumps(self.spec.classes.get(mu.klass, {}), sort_keys=True)
        return _sha(
            "mutant/v2",
            self.thash,
            self.closure_hash(),
            self.ref_unit_text(mu.unit) if mu.unit in self.reg.all_units() else mu.unit,
            patch.read_bytes() if patch.is_file() else b"",
            mu.klass,
            conf,
        )

    def closure_hash(self) -> str:
        """The reference code a mutant runs beside: every unit of the module and
        its closure. Part of each mutant's cache key, so a shared cache
        (OL_MUTATION_CACHE) never answers for a changed dependency."""
        if not hasattr(self, "_closure_hash"):
            owners = set(self.ref_sources)
            us = sorted(
                u for u, chain in self.reg.all_units().items() if owners & set(chain)
            )
            self._closure_hash = _sha(
                "closure/v1", *[u + "\0" + self.ref_unit_text(u) for u in us]
            )
        return self._closure_hash

    def one(self, mu: Mutant, baseline_s: float) -> MutantResult:
        k = self.key(mu)
        hit = self.cache.data["mutants"].get(k)
        if hit:
            return MutantResult(
                mu.mid,
                hit["status"],
                mu.klass,
                mu.required,
                hit.get("detail", ""),
                True,
            )
        text, why = self.mutant_text(mu)
        if text is None:
            res = MutantResult(mu.mid, "error", mu.klass, mu.required, why)
            return res
        ov = self.overlay(self.ref_sources, f"m-{mu.mid}", {mu.unit: text})
        timeout = max(2.0, 10.0 * baseline_s)
        if mu.klass in ("model", "agent"):
            res = self._statistical(mu, ov, timeout)
        else:
            r = self.run(ov, mu.klass, timeout)
            if r.ok:
                status = "survived"
            elif not r.compiled:
                status = "invalid"
            elif "(timed out after" in r.output[-200:]:
                status = "timeout"
            else:
                status = "killed"
            res = MutantResult(
                mu.mid,
                status,
                mu.klass,
                mu.required,
                _first_failure(r.output) if status != "survived" else "",
            )
        if res.status != "error":
            self.cache.data["mutants"][k] = {
                "status": res.status,
                "detail": res.detail[:300],
                "ts": time.time(),
            }
        return res

    def _statistical(self, mu: Mutant, ov: Overlay, timeout: float) -> MutantResult:
        conf = self.spec.classes.get(mu.klass) or {}
        if mu.klass not in self._ref_metrics:
            self._ref_metrics[mu.klass], _ = self.metrics(
                self.overlay(self.ref_sources, f"ref-{mu.klass}"), mu.klass, timeout * 5
            )
        ref = self._ref_metrics[mu.klass]
        mut, _ = self.metrics(ov, mu.klass, timeout * 5)
        if any(v is None for v in ref):
            return MutantResult(
                mu.mid,
                "error",
                mu.klass,
                mu.required,
                "the eval printed no metric against the reference",
            )
        if all(v is None for v in mut):
            return MutantResult(
                mu.mid, "killed", mu.klass, mu.required, "the eval failed on every seed"
            )
        if mu.klass == "model":
            good = [v for v in mut if v is not None]
            p = permutation_p(ref, good, bool(conf.get("higher_is_better", False)))
            killed = p < float(conf.get("alpha", 0.05))
            return MutantResult(
                mu.mid,
                "killed" if killed else "survived",
                mu.klass,
                mu.required,
                f"p = {p:.4f} over {len(ref)} + {len(good)} seeds",
            )
        # agent: non-overlapping 95% CIs
        a, b = ref[0], next(v for v in mut if v is not None)
        lo_a, hi_a = (float(x) for x in a.get("ci", [math.nan, math.nan]))
        lo_b, hi_b = (float(x) for x in b.get("ci", [math.nan, math.nan]))
        if any(math.isnan(x) for x in (lo_a, hi_a, lo_b, hi_b)):
            return MutantResult(
                mu.mid,
                "error",
                mu.klass,
                mu.required,
                "the eval printed no `ci` [lo, hi]",
            )
        apart = hi_b < lo_a or hi_a < lo_b
        return MutantResult(
            mu.mid,
            "killed" if apart else "survived",
            mu.klass,
            mu.required,
            f"CIs [{lo_a:.3g}, {hi_a:.3g}] vs [{lo_b:.3g}, {hi_b:.3g}]",
        )

    def mutants(self) -> list[Mutant]:
        ms = manifest(self.course, self.m.id)[:CAP]
        req = set(self.spec.required)
        for mu in ms:
            if mu.mid in req:
                mu.required = True
        return ms

    def grade_key(self) -> str:
        return _sha("grade/v1", self.thash, *[self.key(mu) for mu in self.mutants()])

    def cached_grade(self) -> Grade | None:
        cached = self.cache.data["grades"].get(self.grade_key())
        if not cached:
            return None
        return Grade(
            **{**cached, "results": [MutantResult(**r) for r in cached["results"]]}
        )

    def grade(self, sample: bool = False, sample_seed: int = 0) -> Grade:
        ms = self.mutants()
        g = Grade(
            self.m.id,
            False,
            0.0,
            0,
            0,
            False,
            self.spec.threshold,
            of=len(ms),
            tests_hash=self.thash,
        )
        if not ms:
            g.reason = (
                f"no mutants committed for {self.m.id} (course/mutants/{self.m.id}/)"
            )
            return g
        gkey = self.grade_key()
        if not sample:
            hit = self.cached_grade()
            if hit:
                return hit
        if not graded_files(self.tests_root, self.spec.path):
            g.reason = f"no graded tests at {self.spec.path} yet: write them first (rung R{self.spec.rung})"
            return g
        ok, why, base = self.baselines()
        g.baseline_s = round(base, 3)
        if not ok:
            g.reason = why
            return g
        chosen = ms
        if sample:
            req = [mu for mu in ms if mu.required]
            rest = [mu for mu in ms if not mu.required]
            rng = random.Random(sample_seed)
            chosen = req + rng.sample(rest, min(SAMPLE, len(rest)))
            g.sampled = len(chosen) < len(ms)
        serial = [
            mu
            for mu in chosen
            if self.lang in ("c", "rust") or mu.klass in ("model", "agent")
        ]
        parallel = [mu for mu in chosen if mu not in serial]
        results: dict[str, MutantResult] = {}
        for mu in serial:
            results[mu.mid] = self.one(mu, base)
            self._say(_line(results[mu.mid]))
        if parallel:
            with ThreadPoolExecutor(max_workers=_jobs(self.jobs)) as ex:
                for mu, r in zip(
                    parallel, ex.map(lambda x: self.one(x, base), parallel)
                ):
                    results[mu.mid] = r
                    self._say(_line(r))
        self.cache.save()
        g.results = [results[mu.mid] for mu in chosen]
        counted = [r for r in g.results if r.status not in ("invalid", "error")]
        g.total = len(counted)
        g.killed = sum(r.killed for r in counted)
        g.score = g.killed / g.total if g.total else 0.0
        g.required_ok = all(r.killed for r in g.results if r.required)
        errors = [r for r in g.results if r.status == "error"]
        g.passed = (
            bool(g.total) and g.score >= g.threshold and g.required_ok and not errors
        )
        if errors:
            g.reason = "harness: " + "; ".join(f"{r.mid}: {r.detail}" for r in errors)
        if not sample and not errors:
            self.cache.data["grades"][gkey] = {
                **asdict(g),
                "results": [asdict(r) for r in g.results],
            }
            self.cache.save()
        return g


def _line(r: MutantResult) -> str:
    mark = {
        "killed": "killed  ",
        "timeout": "timeout ",
        "survived": "SURVIVED",
        "invalid": "invalid ",
        "error": "ERROR   ",
    }[r.status]
    return f"    {mark} {r.mid}" + (" (cached)" if r.cached else "")


def _failed_names(out: str) -> list[str]:
    names = re.findall(r"^FAILED \S+::(\w+)", out, re.M)  # pytest
    names += re.findall(r"^--- FAIL: (\w+)", out, re.M)  # go
    names += re.findall(r"^test (\S+) \.\.\. FAILED", out, re.M)  # rust
    names += re.findall(r"^FAIL (\w+)\b", out, re.M)  # ol_test.h
    return sorted(set(names))


def _first_failure(out: str) -> str:
    names = _failed_names(out)
    return ("caught by " + ", ".join(names[:3])) if names else ""


def find_grade_cache(learner: Path) -> Cache:
    """The learner's mutation cache, or OL_MUTATION_CACHE when set. Entries are
    keyed by content (graded tests, the reference unit, the patch), so one file
    can serve many repos and CI runs: the reference learner's e2e job keeps it
    between runs."""
    shared = os.environ.get("OL_MUTATION_CACHE")
    return Cache(
        Path(shared) if shared else learner / ".ol" / "cache" / "mutation.json"
    )


def scratch_tests_root(course: Path, m: Module, work: Path) -> Path | None:
    """Maintainer side: a scratch repo holding course/ref/learner-tests/<ID>/
    (laid out from the learner repo root, so `python/tests/...` lands at the
    module's [learner_tests].path) plus the contracts."""
    src = course / "ref" / "learner-tests" / m.id
    if not src.is_dir():
        return None
    root = work / "ref-learner-tests" / m.id
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(src, root, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(course / "contracts", root / "contracts", dirs_exist_ok=True)
    if not graded_files(root, Spec.of(m).path):
        raise HarnessError(
            f"{src}: no graded test files at {Spec.of(m).path} (lay the files out from the learner repo root)"
        )
    return root
