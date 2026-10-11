"""Milestone step matchers (DESIGN 5.7 matcher table).

    expect = { match = "<name>", ... }

| match           | keys                                                                 |
|-----------------|----------------------------------------------------------------------|
| exit-code       | code (default 0)                                                     |
| exact, contains | text or file; stdout after `normalize` (the predict normalizer)      |
| regex           | pattern                                                              |
| json-last-line  | metrics = { name = ">= 0.95" } (ops >= <= > < ==)                    |
| tokens-equal    | file (JSON ids list or {"ids": [...]}), near_tie = {logits, margin, argv} |
| numeric         | values, rtol, atol                                                   |
| logits-close    | file (reference npz), output (the step's npz), atol (1e-3), top_k (5) |
| json-schema     | schema (path or inline table)                                        |
| file-produced   | glob, min_count, schema, contains, rows (parquet total, e.g. ">= 1000") |
| sse             | min_chunks; the stream is framed byte for byte (2.6)                 |
| suite           | suite (openapi:...), service (local) or base                         |
| promql          | query, base (default [deploy].prometheus), bound, hold_s, interval_s |
| trace           | service, spans, services, parents, base (default [deploy].traces), api (jaeger or tempo) |
| git-log         | n (default 1): the last n subjects are Conventional Commits          |
| ci-status       | GitHub: gh run list on the default branch; else [ci].local commands  |
| effects         | keys ("a..b" or list), prefix, exactly_once, delivered_once, min_redelivered |

Every step that runs a command must also exit `code` (default 0).
"""

from __future__ import annotations

import ast
import base64
import glob as globmod
import json
import math
import os
import re
import shutil
import struct
import time
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import HarnessError, conform, ctx, placeholders, schema, web

CONVENTIONAL = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([^()\s]+\))?!?: \S.*$"
)
FLOAT = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)\b", re.I
)


@dataclass
class StepOut:
    rc: int | None  # None for HTTP steps and command-less steps
    stdout: str = ""
    http_status: int | None = None
    out_dir: Path | None = None
    argv: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    status: str  # pass fail near-tie pending
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "near-tie", "pending")


@dataclass
class MatchCtx:
    run: object  # runner.Run
    lookup: Callable
    step_name: str
    started_at: float
    kind_mode: bool = False
    stack: object | None = None  # services.Stack when local services run
    run_role: Callable | None = None  # (role, argv) -> (rc, stdout)
    step_argv: list[str] = field(default_factory=list)
    step_role: str | None = None
    step_key: str = ""  # <MS-ID>/<step name>: the ref-thresholds key
    smoke: bool = False  # a --smoke run reads the `smoke` thresholds
    recorder: dict | None = None  # --record-thresholds: (key, metric, dir) -> value
    effects: object | None = None  # effects.Sink when the step names sink = "effects"


def normalize(text: str) -> str:
    """The predict normalizer: strip trailing spaces, then leading and trailing blank lines."""
    lines = [x.rstrip() for x in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def last_json(stdout: str) -> dict | None:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            return None
        return v if isinstance(v, dict) else None
    return None


def _text_or_file(e: dict, mc: MatchCtx) -> str:
    if "text" in e:
        return str(e["text"])
    if "file" in e:
        return mc.run.course_file(placeholders.expand(e["file"], mc.lookup)).read_text()
    raise HarnessError(f"step {mc.step_name}: expect needs `text` or `file`")


def _cmp(value: float, spec: str) -> tuple[bool, str]:
    m = re.fullmatch(r"\s*(>=|<=|==|>|<)\s*(\S+)\s*", str(spec))
    if not m:
        raise HarnessError(
            f"threshold {spec!r}: want `<op> <number>` with op in >= <= > < =="
        )
    op, rhs = m.groups()
    if rhs == "calibrated":
        raise HarnessError("`calibrated` needs a step context (json-last-line)")
    b = float(rhs)
    ok = {
        ">=": value >= b,
        "<=": value <= b,
        ">": value > b,
        "<": value < b,
        "==": value == b,
    }[op]
    return ok, f"{value:g} {op} {b:g}"


# ---------------------------------------------------------------------------


def m_exit_code(e, out: StepOut, mc) -> Verdict:
    return Verdict("pass")  # the exit code itself is checked for every command step


def m_exact(e, out, mc) -> Verdict:
    want, got = normalize(_text_or_file(e, mc)), normalize(out.stdout)
    if want == got:
        return Verdict("pass")
    return Verdict(
        "fail",
        f"stdout differs:\n--- want\n{ctx.tail(want, 15)}\n--- got\n{ctx.tail(got, 15)}",
    )


def m_contains(e, out, mc) -> Verdict:
    want = normalize(_text_or_file(e, mc))
    return (
        Verdict("pass")
        if want in normalize(out.stdout)
        else Verdict("fail", f"stdout lacks {want[:200]!r}")
    )


def m_regex(e, out, mc) -> Verdict:
    pat = e.get("pattern")
    if not pat:
        raise HarnessError(f"step {mc.step_name}: regex needs `pattern`")
    return (
        Verdict("pass")
        if re.search(pat, normalize(out.stdout), re.M)
        else Verdict("fail", f"no match for /{pat}/")
    )


def m_json_last_line(e, out, mc) -> Verdict:
    obj = last_json(out.stdout)
    if obj is None:
        return Verdict("fail", "the last stdout line is not a JSON object")
    bad, good = [], []
    for name, spec in (e.get("metrics") or {}).items():
        if "{" in str(spec):  # "== {json:<step>.<key>}"
            spec = placeholders.expand(str(spec), mc.lookup)
        if (
            name not in obj
            or not isinstance(obj[name], (int, float))
            or isinstance(obj[name], bool)
        ):
            bad.append(f"{name}: missing or not a number")
            continue
        m = re.fullmatch(r"\s*(<=|>=)\s*calibrated\s*", str(spec))
        k = re.fullmatch(r"\s*(>=|<=|==|>|<)\s*([A-Za-z_]\w*)\s*", str(spec))
        if m:
            ok, why = _calibrated(float(obj[name]), name, m.group(1), e, mc)
        elif k and k.group(2) != "calibrated":
            # `<op> <key>`: compare with another number on the same line
            # (MS-agent: ci_low = "<= ci_high").
            other = obj.get(k.group(2))
            if not isinstance(other, (int, float)) or isinstance(other, bool):
                bad.append(f"{name}: {k.group(2)} missing or not a number")
                continue
            ok, why = _cmp(float(obj[name]), f"{k.group(1)} {float(other)!r}")
            why += f" ({k.group(2)})"
        else:
            ok, why = _cmp(float(obj[name]), spec)
        (good if ok else bad).append(f"{name}: {why}")
    return Verdict("fail", "; ".join(bad)) if bad else Verdict("pass", "; ".join(good))


def _calibrated(
    value: float, metric: str, op: str, e: dict, mc: MatchCtx
) -> tuple[bool, str]:
    """`<= calibrated` (lower is better) or `>= calibrated`: the bar is the
    reference mean +/- 3 sd over 5 reference seeds (ref-thresholds.tsv)."""
    from . import thresholds

    key = str(e.get("key") or mc.step_key)
    direction = "max" if op == "<=" else "min"
    if mc.recorder is not None:
        mc.recorder[(key, metric, direction)] = value
        return True, f"{value:.6g} recorded for the reference thresholds"
    row = thresholds.lookup(mc.run.course, key, metric, "smoke" if mc.smoke else "full")
    if row.direction != direction:
        raise HarnessError(
            f"{key} {metric}: recorded as {row.direction}, the step says {op}"
        )
    return thresholds.passes(row, value)


def _ids(v) -> list[int] | None:
    if isinstance(v, dict):
        v = v.get("ids")
    if isinstance(v, list) and all(
        isinstance(x, int) and not isinstance(x, bool) for x in v
    ):
        return v
    return None


def m_tokens_equal(e, out, mc) -> Verdict:
    got = _ids(last_json(out.stdout))
    if got is None:
        return Verdict(
            "fail", 'the last stdout line is not {"ids": [...]} (spec/cli-roles.md)'
        )
    path = mc.run.course_file(placeholders.expand(e.get("file", ""), mc.lookup))
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise HarnessError(
            f"step {mc.step_name}: expected ids file {path}: {err}"
        ) from None
    want = _ids(doc)
    if want is None:
        raise HarnessError(f'{path}: want a JSON list of ids or {{"ids": [...]}}')
    # An oracle file may carry the oracle's own top-2 margin at every step.
    oracle_margins = doc.get("margins") if isinstance(doc, dict) else None
    if got == want:
        return Verdict("pass", f"{len(got)} ids equal")
    k = next(
        (i for i, (a, b) in enumerate(zip(got, want)) if a != b),
        min(len(got), len(want)),
    )
    msg = f"first divergence at step {k}: got {got[k] if k < len(got) else '<end>'}, want {want[k] if k < len(want) else '<end>'} ({len(got)} vs {len(want)} ids)"
    nt = e.get("near_tie")
    if not nt or k >= len(want):
        return Verdict("fail", msg)
    limit = float(nt.get("margin", 1e-3))
    if isinstance(oracle_margins, list) and k < len(oracle_margins):
        # The oracle recorded its margin: no rerun needed. Margin-filtered
        # oracle files (DESIGN 5.7) never fall under the limit, so this only
        # fires for unfiltered files (learner-trained models).
        om = float(oracle_margins[k])
        if om < limit:
            return Verdict(
                "near-tie",
                f"{msg}; the oracle's top-2 margin there is {om:.3g} < {limit:g}",
            )
        return Verdict(
            "fail",
            f"{msg}; the oracle's top-2 margin there is {om:.3g} >= {limit:g}, so this is a real divergence",
        )
    margin = _near_tie_margin(nt, want[:k], mc)
    if margin is None:
        return Verdict("fail", msg + "; the near-tie rerun printed no logits")
    if margin < limit:
        return Verdict(
            "near-tie",
            f"{msg}; top-2 margin {margin:.3g} < {limit:g} under teacher forcing",
        )
    return Verdict(
        "fail",
        f"{msg}; top-2 margin {margin:.3g} >= {limit:g}, so this is a real divergence",
    )


def _near_tie_margin(nt: dict, prefix: list[int], mc: MatchCtx) -> float | None:
    """Rerun the learner's `logits` verb under teacher forcing on the expected
    prefix. Default argv: the step's argv with `generate` replaced by `logits`,
    plus `--prefix-ids <csv>`. The last stdout line is {"logits": [...]}."""
    role = nt.get("logits", mc.step_role or "tinyllm")
    csv = ",".join(map(str, prefix))
    if "argv" in nt:
        lk = placeholders.chain({"prefix_ids": csv}.get, mc.lookup)
        argv = placeholders.expand_argv(
            list(nt["argv"]), lk, f"step {mc.step_name} near_tie"
        )
    else:
        argv = ["logits" if a == "generate" else a for a in mc.step_argv] + [
            "--prefix-ids",
            csv,
        ]
    if mc.run_role is None:
        return None
    rc, stdout = mc.run_role(role, argv)
    obj = last_json(stdout) or {}
    logits = obj.get("logits")
    if rc != 0 or not isinstance(logits, list) or len(logits) < 2:
        return None
    top = sorted((float(x) for x in logits), reverse=True)
    return top[0] - top[1]


def m_numeric(e, out, mc) -> Verdict:
    want = [float(x) for x in e.get("values", [])]
    got = [float(x) for x in FLOAT.findall(out.stdout)]
    if len(got) != len(want):
        return Verdict("fail", f"{len(got)} numbers in stdout, want {len(want)}")
    rtol, atol = float(e.get("rtol", 1e-9)), float(e.get("atol", 0.0))
    for i, (a, b) in enumerate(zip(got, want)):
        if not (
            abs(a - b) <= atol + rtol * abs(b) or (math.isnan(a) and math.isnan(b))
        ):
            return Verdict(
                "fail", f"value {i}: {a!r} vs {b!r} (rtol {rtol:g}, atol {atol:g})"
            )
    return Verdict("pass")


NPY_CODES = {
    ("f", 2): "e",
    ("f", 4): "f",
    ("f", 8): "d",
    ("i", 1): "b",
    ("i", 2): "h",
    ("i", 4): "i",
    ("i", 8): "q",
    ("u", 1): "B",
    ("u", 2): "H",
    ("u", 4): "I",
    ("u", 8): "Q",
}


def _npy(raw: bytes, what: str) -> list:
    """One .npy array (numeric dtype, C order), flattened to a list. The
    harness core is stdlib only, so it reads the format itself."""
    if raw[:6] != b"\x93NUMPY" or len(raw) < 10:
        raise ValueError(f"{what}: not a .npy array")
    if raw[6] == 1:
        (hlen,), start = struct.unpack("<H", raw[8:10]), 10
    else:
        (hlen,), start = struct.unpack("<I", raw[8:12]), 12
    head = ast.literal_eval(raw[start : start + hlen].decode("latin-1"))
    descr, shape = str(head.get("descr", "")), tuple(head.get("shape", ()))
    m = re.fullmatch(r"([<>|=])([fiu])(\d+)", descr)
    code = NPY_CODES.get((m.group(2), int(m.group(3)))) if m else None
    if code is None:
        raise ValueError(f"{what}: unsupported dtype {descr!r}")
    if head.get("fortran_order") and len(shape) > 1:
        raise ValueError(f"{what}: Fortran-order arrays are not supported")
    n = math.prod(shape)
    endian = ">" if m.group(1) == ">" else "<"
    body = raw[start + hlen : start + hlen + n * int(m.group(3))]
    return list(struct.unpack(f"{endian}{n}{code}", body))


def read_npz(path: Path) -> dict[str, list]:
    """Every array of an .npz (np.savez or np.savez_compressed), flattened."""
    out: dict[str, list] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(".npy"):
                out[name[:-4]] = _npy(z.read(name), f"{path}:{name}")
    return out


def _top_ids(pairs: list[tuple[int, float]], k: int) -> list[int]:
    """The k largest values' ids, ties to the lowest id."""
    return [i for i, _ in sorted(pairs, key=lambda p: (-p[1], p[0]))[:k]]


def m_logits_close(e, out, mc) -> Verdict:
    """The step's npz (`output`: logits0, logits1, ..., one [vocab] row per
    prompt) against a reference npz (`file`). A reference row is either dense
    (logits<i>) or the oracle's largest logits (top<i>.ids, top<i>.values).
    Every reference value within `atol` (default 1e-3) and the `top_k`
    (default 5) ids identical, in order."""
    ref_p = mc.run.course_file(placeholders.expand(e.get("file", ""), mc.lookup))
    got_s = placeholders.expand(str(e.get("output", "")), mc.lookup)
    if not e.get("file") or not got_s:
        raise HarnessError(
            f"step {mc.step_name}: logits-close needs `file` and `output`"
        )
    got_p = Path(got_s)
    if not got_p.is_absolute():
        got_p = (mc.run.learner or Path.cwd()) / got_p
    try:
        ref = read_npz(ref_p)
    except (OSError, ValueError, zipfile.BadZipFile, SyntaxError) as err:
        raise HarnessError(
            f"step {mc.step_name}: reference npz {ref_p}: {err}"
        ) from None
    if not got_p.is_file():
        return Verdict("fail", f"{got_p} was not written")
    try:
        got = read_npz(got_p)
    except (OSError, ValueError, zipfile.BadZipFile, SyntaxError, struct.error) as err:
        return Verdict("fail", f"{got_p}: {err}")
    rows = sorted(
        {
            int(m.group(1))
            for k in ref
            if (m := re.fullmatch(r"(?:logits|top)(\d+)(?:\.ids)?", k))
        }
    )
    if not rows:
        raise HarnessError(f"{ref_p}: holds no logits<i> or top<i>.ids arrays")
    n_got = sum(1 for k in got if re.fullmatch(r"logits\d+", k))
    if n_got != len(rows):
        return Verdict("fail", f"{got_p} holds {n_got} logits rows, want {len(rows)}")
    atol, k = float(e.get("atol", 1e-3)), int(e.get("top_k", 5))
    worst = 0.0
    for i in rows:
        row = got.get(f"logits{i}")
        if row is None:
            return Verdict("fail", f"{got_p} has no logits{i}")
        if f"logits{i}" in ref:
            want = list(enumerate(ref[f"logits{i}"]))
            if len(row) != len(want):
                return Verdict(
                    "fail", f"logits{i}: {len(row)} values, want {len(want)}"
                )
        else:
            want = list(zip(map(int, ref[f"top{i}.ids"]), ref[f"top{i}.values"]))
            if want and max(j for j, _ in want) >= len(row):
                return Verdict(
                    "fail",
                    f"logits{i}: {len(row)} values, too few for the reference ids",
                )
        for j, v in want:
            d = abs(float(row[j]) - float(v))
            if not d <= atol:
                return Verdict(
                    "fail",
                    f"logits{i}[{j}]: {row[j]!r} vs {v!r}, |diff| {d:.3g} > {atol:g}",
                )
            worst = max(worst, d)
        want_top = _top_ids(want, k)
        got_top = _top_ids(list(enumerate(map(float, row))), k)
        if got_top != want_top:
            return Verdict("fail", f"logits{i}: top-{k} ids {got_top}, want {want_top}")
    return Verdict(
        "pass",
        f"{len(rows)} rows, max |diff| {worst:.3g} <= {atol:g}, top-{k} ids equal",
    )


def _load_schema(spec, mc) -> dict:
    if isinstance(spec, dict):
        return spec
    p = placeholders.expand(str(spec), mc.lookup)
    path = (
        mc.run.course_file(p)
        if p.startswith("course/") or not mc.run.learner
        else mc.run.learner / p
    )
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise HarnessError(f"schema {path}: {err}") from None


def m_json_schema(e, out, mc) -> Verdict:
    sch = _load_schema(e.get("schema"), mc)
    try:
        inst = json.loads(out.stdout)
    except json.JSONDecodeError:
        inst = last_json(out.stdout)
        if inst is None:
            return Verdict("fail", "stdout is not JSON")
    errs = schema.validate(inst, sch)
    return Verdict("fail", "\n".join(errs[:10])) if errs else Verdict("pass")


def m_file_produced(e, out, mc) -> Verdict:
    pat = placeholders.expand(e.get("glob", ""), mc.lookup)
    if not pat:
        raise HarnessError(f"step {mc.step_name}: file-produced needs `glob`")
    if not os.path.isabs(pat):
        pat = str((mc.run.learner or Path.cwd()) / pat)
    files = sorted(
        Path(p) for p in globmod.glob(pat, recursive=True) if Path(p).is_file()
    )
    if len(files) < int(e.get("min_count", 1)):
        return Verdict(
            "fail",
            f"{len(files)} file(s) match {pat}, want at least {e.get('min_count', 1)}",
        )
    if "rows" in e:
        total, err = _parquet_rows(files, mc)
        if total is None:
            return Verdict("fail", err)
        ok, why = _cmp(float(total), str(e["rows"]))
        if not ok:
            return Verdict(
                "fail",
                f"{len(files)} parquet file(s) hold {why.split(' ')[0]} rows; want {e['rows']}",
            )
    sch = _load_schema(e["schema"], mc) if "schema" in e else None
    for f in files:
        if sch is not None:
            try:
                errs = schema.validate(json.loads(f.read_text()), sch)
            except (json.JSONDecodeError, UnicodeDecodeError) as err:
                errs = [f"not JSON: {err}"]
            if errs:
                return Verdict("fail", f"{f}: " + "; ".join(errs[:5]))
        if "contains" in e and e["contains"] not in f.read_text(errors="replace"):
            return Verdict("fail", f"{f} lacks {e['contains']!r}")
    return Verdict("pass", f"{len(files)} file(s)")


PARQUET_ROWS = """
import sys
import pyarrow.parquet as pq
print(sum(pq.ParquetFile(p).metadata.num_rows for p in sys.argv[1:]))
"""


def _parquet_rows(files: list[Path], mc) -> tuple[int | None, str]:
    """Row counts from the parquet footers, read with pyarrow in YOUR python
    environment (the corpus modules depend on it; the harness does not)."""
    learner = mc.run.learner
    proj = (
        learner / "python"
        if learner and (learner / "python" / "pyproject.toml").is_file()
        else None
    )
    cmd = ["uv", "run", "--quiet"] + (
        ["--project", str(proj)] if proj else ["--no-project", "--with", "pyarrow"]
    )
    cmd += ["python", "-c", PARQUET_ROWS, *map(str, files)]
    env = mc.run.env()
    rc, out = ctx.run(cmd, cwd=learner, env=env, timeout=300)
    lines = out.strip().splitlines()
    if rc != 0 or not lines or not lines[-1].strip().isdigit():
        return (
            None,
            f"reading parquet row counts failed (is pyarrow in python/pyproject.toml?):\n{ctx.tail(out, 10)}",
        )
    return int(lines[-1]), ""


def m_sse(e, out, mc) -> Verdict:
    payloads, errs = web.sse_events(out.stdout.encode())
    if errs:
        return Verdict("fail", "; ".join(errs))
    n = 0
    for p in payloads[:-1]:
        try:
            ch = (json.loads(p).get("choices") or [{}])[0]
        except (json.JSONDecodeError, AttributeError):
            return Verdict("fail", f"chunk is not JSON: {p[:80]!r}")
        if ch.get("text") or (ch.get("delta") or {}).get("content"):
            n += 1
    want = int(e.get("min_chunks", 1))
    return (
        Verdict("pass", f"{n} content chunks")
        if n >= want
        else Verdict("fail", f"{n} content chunks, want at least {want}")
    )


def m_suite(e, out, mc) -> Verdict:
    from .commands.conform import run_suite

    suite = conform.parse_suite(e.get("suite", ""), e.get("target"))
    if "base" in e:
        base = placeholders.expand(e["base"], mc.lookup)
        health = base
    elif mc.kind_mode:
        dep = mc.run.system.deploy if mc.run.system else {}
        base = str(dep.get("gateway_url", ""))
        health = dep.get(
            "gateway_health_url"
        )  # absent: the healthz case is pending (cli-roles: /healthz is on --health-port)
        if not base:
            raise HarnessError("a kind suite step needs [deploy].gateway_url")
    else:
        svc = e.get("service", suite.tier)
        inst = mc.stack.instances.get(svc) if mc.stack else None
        if inst is None:
            raise HarnessError(
                f"step {mc.step_name}: service {svc!r} is not started (add it to the step's `services`)"
            )
        base, health = inst.base, f"http://127.0.0.1:{inst.ports['health_port']}"
    results = run_suite(mc.run, suite, base, health, e.get("model"), e.get("params"))
    bad = [r for r in results if r.status == "fail"]
    pend = [r.case for r in results if r.status == "pending"]
    if not results:
        return Verdict("fail", f"no case of {suite.id} applies")
    if bad:
        return Verdict("fail", "\n".join(f"{r.case}: {r.detail}" for r in bad))
    return Verdict(
        "pass",
        f"{len(results) - len(pend)} cases pass"
        + (f", pending: {', '.join(pend)}" if pend else ""),
    )


def _deploy(mc, key: str, e: dict) -> str:
    if "base" in e:
        return placeholders.expand(e["base"], mc.lookup).rstrip("/")
    v = (mc.run.system.deploy if mc.run.system else {}).get(key)
    if not v:
        raise HarnessError(f"step {mc.step_name} needs [deploy].{key} in system.toml")
    return str(v).rstrip("/")


def promql_instant(base: str, query: str) -> tuple[list | None, str]:
    r = web.get(f"{base}/api/v1/query", {"query": query}, timeout=15)
    if r.status != 200:
        return None, r.error or f"HTTP {r.status}: {r.text[:200]}"
    try:
        data = r.json()["data"]
    except (ValueError, KeyError):
        return None, f"unexpected Prometheus reply: {r.text[:200]}"
    res = data.get("result")
    if data.get("resultType") == "scalar":
        res = [{"value": res}]
    return res or [], ""


def _bound_ok(samples: list, bound: str | None) -> tuple[bool, str]:
    if not samples:
        return False, "empty result"
    if not bound:
        return True, f"{len(samples)} series"
    for s in samples:
        ok, why = _cmp(float(s["value"][1]), bound)
        if not ok:
            return False, why
    return True, f"{len(samples)} series within {bound}"


def m_promql(e, out, mc) -> Verdict:
    base = _deploy(mc, "prometheus", e)
    q = placeholders.expand(e.get("query", ""), mc.lookup)
    hold = float(e.get("hold_s", 0))
    interval = float(e.get("interval_s", os.environ.get("SS_PROMQL_INTERVAL", 15)))
    deadline = time.monotonic() + hold
    while True:
        res, err = promql_instant(base, q)
        if res is None:
            return Verdict("fail", f"query failed: {err}")
        ok, why = _bound_ok(res, e.get("bound"))
        if not ok:
            return Verdict("fail", f"{q}: {why}")
        if time.monotonic() >= deadline:
            return Verdict("pass", why + (f", held {hold:g}s" if hold else ""))
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())) or 0.01)


def is_tempo(base: str, e: dict) -> bool:
    """`api = "tempo"` or `"jaeger"` on the step decides; else Tempo's ports
    (3200, the course's NodePort 30320) or a host naming tempo mean Tempo."""
    api = str(e.get("api", "")).lower()
    if api in ("tempo", "jaeger"):
        return api == "tempo"
    if api:
        raise HarnessError(f"trace api {api!r}: want jaeger or tempo")
    u = urllib.parse.urlsplit(base)
    return u.port in (3200, 30320) or "tempo" in (u.hostname or "")


def find_trace(base: str, e: dict, start: float, end: float) -> tuple[dict | None, str]:
    """A trace holding every span name in `spans`, every service in
    `services`, and every [child, parent] pair in `parents`, from the Jaeger
    or the Tempo query API (`is_tempo`)."""
    if is_tempo(base, e):
        return find_trace_tempo(base, e, start, end)
    return find_trace_jaeger(base, e, start, end)


def find_trace_jaeger(
    base: str, e: dict, start: float, end: float
) -> tuple[dict | None, str]:
    """Jaeger query API: GET /api/traces?service=&start=&end= (microseconds)."""
    service = e.get("service") or (e.get("services") or [None])[0]
    if not service:
        raise HarnessError("a trace check needs `service` (the service to query by)")
    r = web.get(
        f"{base}/api/traces",
        {
            "service": service,
            "start": int(start * 1e6),
            "end": int(end * 1e6),
            "limit": int(e.get("limit", 50)),
        },
        timeout=15,
    )
    if r.status != 200:
        return None, r.error or f"HTTP {r.status} from {base}/api/traces"
    try:
        traces = r.json().get("data") or []
    except ValueError:
        return None, "the Jaeger reply is not JSON"
    want_spans, want_svcs = set(e.get("spans", [])), set(e.get("services", []))
    for t in traces:
        spans = t.get("spans") or []
        procs = {
            k: (v or {}).get("serviceName")
            for k, v in (t.get("processes") or {}).items()
        }
        names = {s.get("operationName") for s in spans}
        svcs = {procs.get(s.get("processID")) for s in spans}
        if not (want_spans <= names and want_svcs <= svcs):
            continue
        by_id = {s.get("spanID"): s for s in spans}
        ok = True
        for child, parent in e.get("parents", []):
            if not any(
                s.get("operationName") == child
                and any(
                    by_id.get(ref.get("spanID"), {}).get("operationName") == parent
                    for ref in s.get("references") or []
                    if ref.get("refType") == "CHILD_OF"
                )
                for s in spans
            ):
                ok = False
                break
        if ok:
            return t, ""
    return (
        None,
        f"no trace among {len(traces)} for service {service!r} has spans {sorted(want_spans)} and services {sorted(want_svcs)}",
    )


def _otlp_id(v) -> str:
    """OTLP JSON ids are hex (the OTLP spec) or base64 (protobuf JSON, as
    Tempo writes them); both become lower-case hex."""
    s = str(v or "")
    if re.fullmatch(r"(?:[0-9a-fA-F]{16}|[0-9a-fA-F]{32})", s):
        return s.lower()
    try:
        return base64.b64decode(s, validate=True).hex() if s else ""
    except ValueError:
        return s


def _otlp_spans(doc: dict) -> list[dict]:
    """Flatten an OTLP JSON trace (/api/traces/<id>: `batches` or
    `resourceSpans`, optionally under `trace`) into {name, id, parent, service}."""
    if isinstance(doc.get("trace"), dict):
        doc = doc["trace"]
    out = []
    for rs in doc.get("resourceSpans") or doc.get("batches") or []:
        attrs = (rs.get("resource") or {}).get("attributes") or []
        svc = next(
            (
                (a.get("value") or {}).get("stringValue")
                for a in attrs
                if a.get("key") == "service.name"
            ),
            None,
        )
        for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []:
            for sp in ss.get("spans") or []:
                out.append(
                    {
                        "name": sp.get("name"),
                        "id": _otlp_id(sp.get("spanId")),
                        "parent": _otlp_id(sp.get("parentSpanId")),
                        "service": svc,
                    }
                )
    return out


def _span_tree_ok(spans: list[dict], e: dict) -> bool:
    names = {s["name"] for s in spans}
    svcs = {s["service"] for s in spans}
    if not (set(e.get("spans", [])) <= names and set(e.get("services", [])) <= svcs):
        return False
    by_id = {s["id"]: s for s in spans if s["id"]}
    return all(
        any(
            s["name"] == child
            and s["parent"]
            and by_id.get(s["parent"], {}).get("name") == parent
            for s in spans
        )
        for child, parent in e.get("parents", [])
    )


def find_trace_tempo(
    base: str, e: dict, start: float, end: float
) -> tuple[dict | None, str]:
    """Tempo query API: GET /api/search?tags=service.name=<svc>&start=&end=
    (seconds), then GET /api/traces/<id> for each hit (OTLP JSON)."""
    service = e.get("service") or (e.get("services") or [None])[0]
    if not service:
        raise HarnessError("a trace check needs `service` (the service to query by)")
    tag = json.dumps(service) if re.search(r'[\s="]', service) else service
    r = web.get(
        f"{base}/api/search",
        {
            "tags": f"service.name={tag}",
            "start": int(start),
            "end": int(math.ceil(end)),
            "limit": int(e.get("limit", 50)),
        },
        headers={"Accept": "application/json"},
        timeout=15,
    )
    if r.status != 200:
        return None, r.error or f"HTTP {r.status} from {base}/api/search"
    try:
        hits = r.json().get("traces") or []
    except (ValueError, AttributeError):
        return None, "the Tempo search reply is not a JSON object"
    want_spans, want_svcs = sorted(e.get("spans", [])), sorted(e.get("services", []))
    for h in hits:
        tid = str((h or {}).get("traceID") or "")
        if not tid:
            continue
        tr = web.get(
            f"{base}/api/traces/{urllib.parse.quote(tid)}",
            headers={"Accept": "application/json"},
            timeout=15,
        )
        if tr.status != 200:
            continue
        try:
            spans = _otlp_spans(tr.json())
        except (ValueError, AttributeError):
            continue
        if _span_tree_ok(spans, e):
            return {"traceID": tid, "spans": spans}, ""
    return (
        None,
        f"no trace among {len(hits)} for service {service!r} has spans {want_spans} and services {want_svcs}",
    )


def m_trace(e, out, mc) -> Verdict:
    e = placeholders.expand_obj(dict(e), mc.lookup, f"step {mc.step_name}")
    base = _deploy(mc, "traces", e)
    # Services export spans asynchronously (batches, queues), so the trace
    # of a request made a moment ago may not be queryable yet: poll for
    # `wait_s` seconds (default 30) before failing.
    deadline = time.time() + float(e.get("wait_s", 30))
    while True:
        t, why = find_trace(
            base, e, mc.started_at - float(e.get("lookback_s", 60)), time.time() + 5
        )
        if t or time.time() >= deadline:
            break
        time.sleep(2)
    return Verdict("pass", f"trace {t.get('traceID')}") if t else Verdict("fail", why)


def m_git_log(e, out, mc) -> Verdict:
    n = int(e.get("n", 1))
    rc, log = ctx.git(["log", f"-n{n}", "--format=%s"], mc.run.learner)
    subjects = [x for x in log.splitlines() if x.strip()] if rc == 0 else []
    if len(subjects) < n:
        return Verdict("fail", f"{len(subjects)} commit(s), want at least {n}")
    bad = [s for s in subjects if not CONVENTIONAL.match(s)]
    return (
        Verdict("fail", "not Conventional Commits: " + "; ".join(repr(s) for s in bad))
        if bad
        else Verdict("pass", f"last {n} commit(s) conventional")
    )


def github_remote(learner: Path) -> str | None:
    rc, out = ctx.git(["remote", "get-url", "origin"], learner)
    url = out.strip()
    return url if rc == 0 and "github.com" in url else None


def default_branch(learner: Path) -> str:
    rc, out = ctx.git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], learner)
    if rc == 0 and out.strip():
        return out.strip().split("/", 1)[-1]
    rc, out = ctx.git(["rev-parse", "--abbrev-ref", "HEAD"], learner)
    return out.strip() if rc == 0 and out.strip() not in ("", "HEAD") else "main"


def m_ci_status(e, out, mc) -> Verdict:
    learner = mc.run.learner
    remote = github_remote(learner)
    if remote:
        if shutil.which("gh") is None:
            raise HarnessError(
                "ci-status on a GitHub remote needs the `gh` CLI (see `ss doctor`)"
            )
        branch = default_branch(learner)
        rc, txt = ctx.run(
            [
                "gh",
                "run",
                "list",
                "--branch",
                branch,
                "--limit",
                "1",
                "--json",
                "conclusion,status,headSha",
            ],
            cwd=learner,
            timeout=60,
        )
        if rc != 0:
            return Verdict("fail", f"gh run list failed: {txt.strip()[:300]}")
        try:
            runs = json.loads(txt)
        except json.JSONDecodeError:
            return Verdict("fail", f"gh printed no JSON: {txt[:200]}")
        if not runs:
            return Verdict("fail", f"no CI run on {branch} yet ({remote})")
        c = runs[0].get("conclusion")
        return (
            Verdict("pass", f"latest run on {branch}: success")
            if c == "success"
            else Verdict(
                "fail",
                f"latest run on {branch}: {c or runs[0].get('status', 'unknown')}",
            )
        )
    steps = mc.run.system.ci_local if mc.run.system else []
    if not steps:
        return Verdict(
            "fail", "no GitHub remote and no [ci].local commands in system.toml"
        )
    env = mc.run.env()
    for argv in steps:
        rc, txt = ctx.run(
            argv, cwd=learner, env=env, timeout=float(e.get("timeout_s", 1800))
        )
        if rc != 0:
            # `ss check --all` puts its verdict table last; name the modules
            # that did not pass, which a tail of the table would cut off.
            bad = [
                x.strip()
                for x in txt.splitlines()
                if re.match(r"^\s+\S+\s+(fail|blocked|error|drift)\b", x)
            ]
            return Verdict(
                "fail",
                f"[ci].local `{' '.join(argv)}` exited {rc}"
                + (f" (not passing: {', '.join(bad)})" if bad else "")
                + f":\n{ctx.indent(ctx.tail(txt, 20))}",
            )
    return Verdict("pass", f"{len(steps)} [ci].local command(s) green")


def m_perf(e, out, mc) -> Verdict:
    """Throughput or latency relative to `ss bench --calibrate` (5.11):
    expect = { match = "perf", metric = "tokens_per_s", budget = ">= 0.5 * decode_tok_s" }
    against the last stdout JSON line; `in_cluster = true` uses the Job's calibration."""
    from . import bench

    obj = last_json(out.stdout)
    metric = str(e.get("metric", ""))
    if obj is None or not isinstance(obj.get(metric), (int, float)):
        return Verdict("fail", f"the last stdout line has no number {metric!r}")
    calib = bench.load_calibration(bool(e.get("in_cluster", False)))
    ok, why = bench.check_budget(float(obj[metric]), str(e.get("budget", "")), calib)
    return Verdict("pass" if ok else "fail", f"{metric}: {why}")


def m_effects(e, out, mc) -> Verdict:
    """The effects sink (`sink = "effects"`) holds prefix+k for every k of
    `keys`, and nothing else under `prefix` (exactly_once: retries are fine,
    the sink dedupes by idempotency key, as testkit AssertExactlyOnce);
    `delivered_once` also forbids redelivery; `min_redelivered` asks that at
    least that many keys were delivered more than once (the kills hit)."""
    from . import effects

    if mc.effects is None:
        raise HarnessError(
            f'step {mc.step_name}: the effects matcher needs sink = "effects"'
        )
    if e.get("linearizable"):
        raise HarnessError(
            f"step {mc.step_name}: effects.linearizable needs a recorded history, which is not built yet"
        )
    if "keys" not in e:
        raise HarnessError(f"step {mc.step_name}: effects needs `keys`")
    keys = effects.parse_keys(e["keys"], f"step {mc.step_name}")
    prefix = placeholders.expand(str(e.get("prefix", "")), mc.lookup)
    once = bool(e.get("delivered_once", False))
    if not once and not e.get("exactly_once", True):
        raise HarnessError(
            f"step {mc.step_name}: effects checks exactly_once (or delivered_once); false checks nothing"
        )
    recs = mc.effects.records()
    problems, st = effects.check(recs, keys, prefix, once)
    if out.out_dir is not None:
        (out.out_dir / "effects.json").write_text(
            json.dumps([r.__dict__ for r in recs], indent=1)
        )
    want_re = int(e.get("min_redelivered", 0))
    if st["redelivered"] < want_re:
        problems.append(
            f"only {st['redelivered']} key(s) were delivered more than once, want >= {want_re}"
        )
    summary = (
        f"{st['applied']} of {st['want']} key(s) applied under {prefix!r}, "
        f"{st['deliveries']} deliveries, {st['redelivered']} redelivered"
    )
    if problems:
        shown = "; ".join(problems[:20]) + (
            f"; ... {len(problems) - 20} more" if len(problems) > 20 else ""
        )
        return Verdict("fail", f"{summary}: {len(problems)} problem(s): {shown}")
    return Verdict("pass", summary)


MATCHERS = {
    "exit-code": m_exit_code,
    "exact": m_exact,
    "contains": m_contains,
    "regex": m_regex,
    "json-last-line": m_json_last_line,
    "tokens-equal": m_tokens_equal,
    "numeric": m_numeric,
    "logits-close": m_logits_close,
    "json-schema": m_json_schema,
    "file-produced": m_file_produced,
    "sse": m_sse,
    "suite": m_suite,
    "promql": m_promql,
    "trace": m_trace,
    "git-log": m_git_log,
    "ci-status": m_ci_status,
    "perf": m_perf,
    "effects": m_effects,
}
NO_COMMAND = {
    "suite",
    "promql",
    "trace",
    "git-log",
    "ci-status",
    "file-produced",
    "effects",
}


def evaluate(expect: dict, out: StepOut, mc: MatchCtx) -> Verdict:
    name = expect.get("match", "exit-code")
    fn = MATCHERS.get(name)
    if fn is None:
        raise HarnessError(
            f"step {mc.step_name}: unknown matcher {name!r} (known: {', '.join(sorted(MATCHERS))})"
        )
    if out.rc is not None and out.rc != int(expect.get("code", 0)):
        return Verdict(
            "fail",
            f"exit {out.rc}, want {expect.get('code', 0)}\n{ctx.indent(ctx.tail(out.stdout, 25))}",
        )
    if out.http_status is not None and out.http_status != int(
        expect.get("status", 200)
    ):
        return Verdict(
            "fail",
            f"HTTP {out.http_status}, want {expect.get('status', 200)}: {out.stdout[:300]}",
        )
    return fn(expect, out, mc)
