"""Parity suites (DESIGN 5.8): every implementation against one oracle.

    course/conformance/parity/<suite>.toml

    id       = "matmul"
    title    = "C matmul vs numpy f64"
    equality = "matmul-bound"         # exact | close | matmul-bound | bytes | ulp
    ulp      = 4                        # ulp only: floats of `ulp_keys` within this many ulps,
    ulp_keys = ["normal"]               # every other value exact (all floats when empty)
    rtol     = 1e-5                     # close only
    atol     = 1e-6
    golden   = "course/fixtures/parity/matmul.json"   # {"cases": [{"name", "input", "output"}]}

    [fuzz]                              # optional live differential mode (--fuzz)
    generator = "gen/matmul.py"         # prints N JSON inputs, one per line: gen.py <seed> <n>
    cases     = 200

    [[impl]]
    name   = "c-v0"
    module = "M03.1"                    # whose units the driver calls
    lang   = "c"                        # python | c | rust | go
    driver = "drivers/matmul.c"         # relative to course/conformance/parity/

A driver reads one JSON object per stdin line (a case input) and prints one
JSON value per line (its output). Python drivers run in the overlay's
Python environment; C drivers link the overlay's -O2 objects plus
drivers/_lib/ol_parity.h; Rust drivers are examples of the harness crate
`ol-tests` (they `include!("../parity/ol_parity.rs")`); Go drivers are
`package main` programs copied into the farm module, so they import
`tinyllm/...` packages.

Golden mode compares each implementation with the oracle outputs (so parity
is transitive); --fuzz feeds generated inputs to every implementation and
compares each with the first one (the Python specification). An
implementation whose module is not in the registry, has no driver yet, or
(for a learner) has no fresh pass is `pending`, not a failure.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, ctx
from .overlay import LIB_FLAGS, Overlay

EQUALITIES = ("exact", "close", "matmul-bound", "bytes", "ulp")
LANGS = ("python", "c", "rust", "go")
EPS32 = 2.0**-23


@dataclass
class Impl:
    name: str
    module: str
    lang: str
    driver: str


@dataclass
class Suite:
    id: str
    title: str
    equality: str
    path: Path
    golden: str = ""
    rtol: float = 1e-5
    atol: float = 1e-6
    ulp: int = 4
    ulp_keys: list = field(default_factory=list)
    fuzz: dict = field(default_factory=dict)
    impls: list[Impl] = field(default_factory=list)


def suite_dir(course: Path) -> Path:
    return course / "conformance" / "parity"


def load(p: Path) -> Suite:
    try:
        raw = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(f"{p}: {e}") from None
    s = Suite(
        id=str(raw.get("id", p.stem)),
        title=str(raw.get("title", "")),
        equality=str(raw.get("equality", "exact")),
        path=p,
        golden=str(raw.get("golden", "")),
        rtol=float(raw.get("rtol", 1e-5)),
        atol=float(raw.get("atol", 1e-6)),
        ulp=int(raw.get("ulp", 4)),
        ulp_keys=list(raw.get("ulp_keys", [])),
        fuzz=dict(raw.get("fuzz", {})),
    )
    if s.id != p.stem:
        raise HarnessError(f"{p}: id {s.id!r} must equal the file name")
    if s.equality not in EQUALITIES:
        raise HarnessError(f"{p}: equality {s.equality!r} not in {EQUALITIES}")
    for i, x in enumerate(raw.get("impl", [])):
        for k in ("name", "module", "lang", "driver"):
            if k not in x:
                raise HarnessError(f"{p}: impl {i + 1} needs `{k}`")
        if x["lang"] not in LANGS:
            raise HarnessError(
                f"{p}: impl {x['name']}: lang {x['lang']!r} not in {LANGS}"
            )
        s.impls.append(Impl(x["name"], x["module"], x["lang"], x["driver"]))
    if not s.impls:
        raise HarnessError(f"{p}: no [[impl]]")
    return s


def load_all(course: Path) -> list[Suite]:
    d = suite_dir(course)
    return [load(p) for p in sorted(d.glob("*.toml"))] if d.is_dir() else []


def golden_cases(course: Path, s: Suite) -> list[dict]:
    if not s.golden:
        return []
    p = (
        course.parent / s.golden
        if s.golden.startswith("course/")
        else course / s.golden
    )
    try:
        doc = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise HarnessError(f"suite {s.id}: golden {p}: {e}") from None
    cases = doc.get("cases") if isinstance(doc, dict) else None
    if not isinstance(cases, list) or not all(
        "input" in c and "output" in c for c in cases
    ):
        raise HarnessError(
            f'suite {s.id}: {p} must be {{"cases": [{{"input", "output"}}]}}'
        )
    return cases


def fuzz_inputs(course: Path, s: Suite, seed: int, n: int | None) -> list[dict]:
    gen = s.fuzz.get("generator")
    if not gen:
        raise HarnessError(f"suite {s.id} has no [fuzz].generator")
    script = suite_dir(course) / gen
    count = int(n or s.fuzz.get("cases", 200))
    p = subprocess.run(
        [sys.executable, "-I", str(script), str(seed), str(count)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if p.returncode != 0:
        raise HarnessError(f"suite {s.id}: generator failed:\n{p.stderr[-2000:]}")
    return [json.loads(x) for x in p.stdout.splitlines() if x.strip()]


def fuzz_oracle(course: Path, s: Suite, inputs: list[dict]) -> tuple[list | None, str]:
    """[fuzz].oracle: a stdlib Python script that answers like a driver."""
    script = suite_dir(course) / s.fuzz["oracle"]
    feed = "".join(json.dumps(x) + "\n" for x in inputs)
    p = subprocess.run(
        [sys.executable, "-I", str(script)],
        input=feed,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if p.returncode != 0:
        return None, p.stderr[-2000:]
    outs = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    return (
        (outs, "")
        if len(outs) == len(inputs)
        else (None, f"{len(outs)} answers for {len(inputs)} inputs")
    )


# ---------------------------------------------------------------------------
# running one driver


def run_driver(
    ov: Overlay, course: Path, impl: Impl, inputs: list[dict], timeout: float = 300
) -> tuple[list | None, str]:
    drv = suite_dir(course) / impl.driver
    feed = "".join(json.dumps(x, separators=(",", ":")) + "\n" for x in inputs)
    name = "ssp_" + "".join(
        c if c.isalnum() else "_" for c in Path(impl.driver).stem + "_" + impl.name
    )
    cwd, env = ov.work, None
    ov.work.mkdir(parents=True, exist_ok=True)
    if impl.lang == "python":
        prefix, env = ov.py_env()
        cmd = prefix + ["python", str(drv)]
    elif impl.lang == "c":
        objs, err = ov.c_objects(LIB_FLAGS, "lib")
        if err:
            return None, err
        exe = ov.work / "parity" / name
        exe.parent.mkdir(parents=True, exist_ok=True)
        cc = [
            "cc",
            "-std=c11",
            "-D_POSIX_C_SOURCE=200809L",
            "-O2",
            f"-I{ov.contracts / 'c' / 'include'}",
            f"-I{suite_dir(course) / 'drivers' / '_lib'}",
            str(drv),
            *map(str, objs),
            "-o",
            str(exe),
        ]
        if platform.system() != "Darwin":
            cc += ["-lm", "-lpthread"]
        rc, out = ctx.run(cc)
        if rc != 0:
            return None, f"driver does not compile:\n{out}"
        cmd, env = [str(exe)], ov.env()
    elif impl.lang == "rust":
        farm = ov.build_rust([])
        ex = farm / "ol-tests" / "examples" / f"{name}.rs"
        ex.parent.mkdir(parents=True, exist_ok=True)
        ex.write_bytes(drv.read_bytes())
        kit = farm / "ol-tests" / "parity" / "ol_parity.rs"
        kit.parent.mkdir(parents=True, exist_ok=True)
        kit.write_bytes(
            (suite_dir(course) / "drivers" / "_lib" / "ol_parity.rs").read_bytes()
        )
        env = ov._cargo_env()
        base = [
            "cargo",
            "build",
            "-q",
            "--release",
            "--manifest-path",
            str(farm / "Cargo.toml"),
            "-p",
            "ol-tests",
            "--example",
            name,
        ]
        rc, out = ctx.run(base, env=env, timeout=900)
        if rc != 0:
            return None, f"driver does not compile:\n{out}"
        cmd = [str(Path(env["CARGO_TARGET_DIR"]) / "release" / "examples" / name)]
    else:  # go
        farm, work = ov.build_go()
        dest = farm / "ssparity" / name / "main.go"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(drv.read_bytes())
        env = ov._go_env(work)
        exe = ov.work / "parity" / name
        exe.parent.mkdir(parents=True, exist_ok=True)
        rc, out = ctx.run(
            ["go", "build", "-o", str(exe), f"./ssparity/{name}"],
            cwd=farm,
            env=env,
            timeout=600,
        )
        if rc != 0:
            return None, f"driver does not compile:\n{out}"
        cmd = [str(exe)]
    try:
        p = subprocess.run(
            cmd,
            input=feed,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"driver timed out after {timeout:.0f}s"
    except FileNotFoundError as e:
        return None, str(e)
    if p.returncode != 0:
        return (
            None,
            f"driver exited {p.returncode}:\n{ctx.tail(p.stdout + p.stderr, 20)}",
        )
    outs = []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        try:
            outs.append(json.loads(line))
        except json.JSONDecodeError:
            return None, f"driver printed a line that is not JSON: {line[:120]!r}"
    if len(outs) != len(inputs):
        return (
            None,
            f"driver answered {len(outs)} of {len(inputs)} cases\n{ctx.tail(p.stderr, 10)}",
        )
    return outs, ""


# ---------------------------------------------------------------------------
# equality


def _flat(v) -> list:
    if isinstance(v, list):
        return [y for x in v for y in _flat(x)]
    if isinstance(v, dict):
        return [y for k in sorted(v) for y in _flat(v[k])]
    return [v]


def _num_close(a, b, rtol: float, atol: float) -> bool:
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
    ):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= atol + rtol * abs(b)
    return a == b


def _matmul_bound(inp: dict) -> float:
    m, k, n = int(inp["m"]), int(inp["k"]), int(inp["n"])
    a, b = inp["a"], inp["b"]
    worst = 0.0
    for i in range(m):
        for j in range(n):
            s = sum(abs(a[i * k + p]) * abs(b[p * n + j]) for p in range(k))
            worst = max(worst, s)
    return 4 * EPS32 * math.sqrt(max(k, 1)) * worst


def _ulp_close(a, b, n: int) -> bool:
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or not (isinstance(a, float) or isinstance(b, float))
    ):
        return a == b
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= n * math.ulp(max(abs(a), abs(b)))


def _compare_ulp(s: Suite, got, want) -> str:
    keys = sorted(want) if isinstance(want, dict) and isinstance(got, dict) else [None]
    if keys != [None] and sorted(got) != keys:
        return f"keys {sorted(got)}, want {keys}"
    for k in keys:
        g, w = (got, want) if k is None else (got[k], want[k])
        loose = not s.ulp_keys or k in s.ulp_keys
        ga, wa = _flat(g), _flat(w)
        if len(ga) != len(wa):
            return f"{k or 'output'}: {len(ga)} values, want {len(wa)}"
        for i, (x, y) in enumerate(zip(ga, wa)):
            ok = _ulp_close(x, y, s.ulp) if loose else x == y
            if not ok:
                rule = f"within {s.ulp} ulp" if loose else "exact"
                return f"{k or 'output'}[{i}]: {x!r} vs {y!r} ({rule})"
    return ""


def compare(s: Suite, got, want, inp: dict) -> str:
    """'' when equal under the suite's rule, else a short mismatch."""
    if s.equality == "ulp":
        return _compare_ulp(s, got, want)
    if s.equality in ("exact", "bytes"):
        return (
            ""
            if got == want
            else f"got {json.dumps(got)[:120]}, want {json.dumps(want)[:120]}"
        )
    ga, wa = _flat(got), _flat(want)
    if len(ga) != len(wa):
        return f"{len(ga)} values, want {len(wa)}"
    if s.equality == "close":
        for i, (x, y) in enumerate(zip(ga, wa)):
            if not _num_close(x, y, s.rtol, s.atol):
                return f"value {i}: {x!r} vs {y!r} (rtol {s.rtol:g}, atol {s.atol:g})"
        return ""
    bound = _matmul_bound(inp)
    for i, (x, y) in enumerate(zip(ga, wa)):
        if not abs(float(x) - float(y)) <= bound:
            return f"value {i}: |{x!r} - {y!r}| > bound {bound:.3g} (4 eps32 sqrt(K) max(|A||B|))"
    return ""


@dataclass
class ImplResult:
    impl: str
    status: str  # pass fail pending error
    detail: str = ""
    source: str = "ref"


def run_suite(
    course: Path,
    reg,
    s: Suite,
    overlay_for,
    mode: str = "golden",
    seed: int = 0,
    cases: int | None = None,
) -> list[ImplResult]:
    """overlay_for(impl) -> (Overlay | None, pending reason, source label)."""
    out: list[ImplResult] = []
    runnable = []
    for im in s.impls:
        if im.module not in reg.modules:
            out.append(
                ImplResult(
                    im.name, "pending", f"{im.module} is not in the registry yet"
                )
            )
            continue
        if not (suite_dir(course) / im.driver).is_file():
            out.append(
                ImplResult(
                    im.name, "pending", f"driver {im.driver} arrives with {im.module}"
                )
            )
            continue
        ov, why, src = overlay_for(im)
        if ov is None:
            out.append(ImplResult(im.name, "pending", why))
            continue
        runnable.append((im, ov, src))
    if not runnable:
        return out
    if mode == "golden":
        gold = golden_cases(course, s)
        inputs = [c["input"] for c in gold]
        for im, ov, src in runnable:
            got, err = run_driver(ov, course, im, inputs)
            if got is None:
                out.append(ImplResult(im.name, "error", err, src))
                continue
            bad = next(
                (
                    f"case {c.get('name', i)}: {why}"
                    for i, (c, g) in enumerate(zip(gold, got))
                    if (why := compare(s, g, c["output"], c["input"]))
                ),
                "",
            )
            out.append(
                ImplResult(
                    im.name,
                    "fail" if bad else "pass",
                    bad or f"{len(gold)} golden cases",
                    src,
                )
            )
        return out
    inputs = fuzz_inputs(course, s, seed, cases)
    if s.fuzz.get("oracle"):
        base, err = fuzz_oracle(course, s, inputs)
        if base is None:
            raise HarnessError(f"suite {s.id}: the fuzz oracle failed: {err}")
        base_name, rest = "the oracle", runnable
    else:
        base_im, base_ov, base_src = runnable[0]
        base, err = run_driver(base_ov, course, base_im, inputs)
        if base is None:
            out.append(ImplResult(base_im.name, "error", err, base_src))
            return out
        out.append(
            ImplResult(
                base_im.name,
                "pass",
                f"{len(inputs)} fuzz cases (the reference side)",
                base_src,
            )
        )
        base_name, rest = base_im.name, runnable[1:]
    for im, ov, src in rest:
        got, err = run_driver(ov, course, im, inputs)
        if got is None:
            out.append(ImplResult(im.name, "error", err, src))
            continue
        bad = next(
            (
                f"fuzz case {i} {json.dumps(inputs[i])[:100]}: {why}"
                for i, (g, b) in enumerate(zip(got, base))
                if (why := compare(s, g, b, inputs[i]))
            ),
            "",
        )
        out.append(
            ImplResult(
                im.name,
                "fail" if bad else "pass",
                bad or f"{len(inputs)} fuzz cases equal {base_name}",
                src,
            )
        )
    return out


def which(names: list[str], all_suites: list[Suite]) -> list[Suite]:
    if not names:
        return all_suites
    by = {s.id: s for s in all_suites}
    out = []
    for n in names:
        if n not in by:
            raise HarnessError(
                f"no parity suite {n!r} (known: {', '.join(sorted(by)) or 'none'})"
            )
        out.append(by[n])
    return out


def clean_farm_extras(ov: Overlay) -> None:
    """Driver files dropped into a farm are removed by its next sync anyway."""
    for d in (
        ov.target_dir / "rust-farm" / "ol-tests" / "parity",
        ov.work / "go" / "ssparity",
    ):
        shutil.rmtree(d, ignore_errors=True)
