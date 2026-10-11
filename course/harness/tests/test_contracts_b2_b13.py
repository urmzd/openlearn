"""The contracts written for batches B2 to B13 (contract 0.2.0) parse, compile,
agree with the design, and agree with every worked example they print.

Each worked example in a format or spec page is recomputed here by an
independent implementation written from the page's rules, so a wrong hex dump
or a wrong number in a page fails a test (DESIGN 2.4 to 2.16, 3.2)."""

import json
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from sscourse import schema

COURSE = Path(__file__).resolve().parents[2]
C = COURSE / "contracts"
INC = C / "c/include"
DESIGN = (COURSE / "DESIGN.md").read_text()

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1


def _json(rel: str):
    return json.loads((C / rel).read_text())


def _yaml(rel: str):
    return yaml.safe_load((C / rel).read_text())


def _refs(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref" and isinstance(v, str):
                out.append(v)
            else:
                _refs(v, out)
    elif isinstance(node, list):
        for v in node:
            _refs(v, out)
    return out


def _hexblock(md: str, after: str) -> bytes:
    """The bytes of the first fenced block after `after`: every run of
    two-digit hex pairs at the start of a line, comments ignored."""
    block = md[md.index(after) :].split("```")[1]
    out = bytearray()
    for line in block.splitlines():
        m = re.match(r"\s*((?:[0-9a-f]{2} )*[0-9a-f]{2})(?:\s{2,}|$)", line)
        if m:
            out += bytes.fromhex(m.group(1).replace(" ", ""))
    return bytes(out)


# -- independent primitives ----------------------------------------------------


def fnv1a64(data: bytes, h: int = 0xCBF29CE484222325) -> int:
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & M64
    return h


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


# -- C ABI -------------------------------------------------------------------

UMBRELLA = [
    "tinyllm/abi.h",
    "tinyllm/arena.h",
    "tinyllm/pool.h",
    "tinyllm/kv_pool.h",
    "tinyllm/ds.h",
    "tinyllm/topk.h",
    "tinyllm/numerics.h",
    "tinyllm/matmul.h",
    "tinyllm/softmax.h",
    "tinyllm/attention.h",
    "tinyllm/qmatmul.h",
    "tinyllm/elementwise.h",
]


def test_umbrella_holds_every_abi_v1_unit_but_kv_v2():
    text = (INC / "tinyllm.h").read_text()
    assert re.findall(r'#include "([^"]+)"', text) == UMBRELLA
    on_disk = {p.relative_to(INC).as_posix() for p in (INC / "tinyllm").glob("*.h")}
    # Other headers (test-kit seams such as failpoint.h) may sit beside the units.
    assert set(UMBRELLA) | {"tinyllm/kv_pool_v2.h"} <= on_disk
    abi = (C / "c/ABI.md").read_text()
    for h in UMBRELLA + ["tinyllm/kv_pool_v2.h"]:
        assert f"[`{h}`]" in abi, f"c/ABI.md has no row for {h}"


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_every_header_compiles_alone_twice_as_c11(tmp_path):
    for h in ["tinyllm.h"] + sorted(
        p.relative_to(INC).as_posix() for p in (INC / "tinyllm").glob("*.h")
    ):
        src = tmp_path / "one.c"
        src.write_text(f'#include "{h}"\n#include "{h}"\n')
        for flags in (
            ["-std=c11", "-Wall", "-Werror"],
            ["-std=c11", "-pedantic", "-Wall", "-Wextra", "-Werror"],
        ):
            r = subprocess.run(
                ["cc", *flags, f"-I{INC}", "-fsyntax-only", str(src)],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, f"{h} {flags}:\n{r.stderr}"


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_struct_layouts_match_abi_md(tmp_path):
    rows = re.findall(
        r"^\| `(tl_\w+)` \| (\d+) \| (.+) \|$", (C / "c/ABI.md").read_text(), re.M
    )
    assert len(rows) == 8  # tl_pcg32 left with the C RNG (M06.3 is Python only)
    lines = []
    for name, size, fields in rows:
        lines.append(f'  printf("{name} %zu\\n", sizeof({name}));')
        for f, off in re.findall(r"`(\w+)` (\d+)", fields):
            lines.append(f'  printf("{name}.{f} %zu\\n", offsetof({name}, {f}));')
    src = tmp_path / "layout.c"
    src.write_text(
        '#include <stddef.h>\n#include <stdio.h>\n#include "tinyllm.h"\nint main(void) {\n'
        + "\n".join(lines)
        + "\n  return 0;\n}\n"
    )
    exe = tmp_path / "layout"
    r = subprocess.run(
        ["cc", "-std=c11", "-Wall", "-Werror", f"-I{INC}", str(src), "-o", str(exe)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    got = dict(
        line.split()
        for line in subprocess.run(
            [str(exe)], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    for name, size, fields in rows:
        assert got[name] == size, name
        for f, off in re.findall(r"`(\w+)` (\d+)", fields):
            assert got[f"{name}.{f}"] == off, f"{name}.{f}"


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_header_constants_and_c_transcriptions(tmp_path):
    """Header macros match the format pages, and an independent C
    transcription of O'Neill's pcg32_srandom_r / pcg32_random_r reproduces
    the published demo line and the committed vectors."""
    vec = _json("spec/pcg32.vectors.json")
    src = tmp_path / "x.c"
    src.write_text(
        r"""
#include <stdio.h>
#include <string.h>
#include "tinyllm.h"
#include "tinyllm/kv_pool_v2.h"
_Static_assert(TL_KV_ENVELOPE_HEADER == 28, "envelope header");
_Static_assert(TL_KV_FORMAT_V1 == 1 && TL_KV_FORMAT_V2 == 2, "formats");
_Static_assert(TL_FNV1A64_OFFSET == 0xCBF29CE484222325ull, "fnv offset");
typedef struct { unsigned long long state, inc; } rng;
static unsigned rnd(rng *r) {
  unsigned long long old = r->state;
  r->state = old * 6364136223846793005ULL + r->inc;
  unsigned xs = (unsigned)(((old >> 18u) ^ old) >> 27u);
  unsigned rot = (unsigned)(old >> 59u);
  return (xs >> rot) | (xs << ((-rot) & 31));
}
static void srnd(rng *r, unsigned long long s, unsigned long long q) {
  r->state = 0U; r->inc = (q << 1u) | 1u; rnd(r); r->state += s; rnd(r);
}
int main(void) {
  rng r; srnd(&r, 42u, 54u);
  for (int i = 0; i < 6; i++) printf("%08x ", rnd(&r));
  printf("\n%s\n", TL_KV_MAGIC);
  unsigned long long seeds[3] = {0ull, 1ull, 1ull << 63};
  for (int s = 0; s < 3; s++) { srnd(&r, seeds[s], 54u); for (int i = 0; i < 1024; i++) printf("%u ", rnd(&r)); printf("\n"); }
  return 0;
}
"""
    )
    exe = tmp_path / "x"
    r = subprocess.run(
        ["cc", "-std=c11", "-Wall", "-Werror", f"-I{INC}", str(src), "-o", str(exe)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    out = subprocess.run(
        [str(exe)], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert out[0].split() == [
        "a15c02b7",
        "7b47f409",
        "ba1d3330",
        "83d2f293",
        "bfa4784b",
        "cbed606e",
    ]
    assert out[1] == "TLKV"
    for line, s in zip(out[2:5], ["0", "1", str(1 << 63)]):
        assert [int(x) for x in line.split()] == vec["next_u32"][s]


# -- spec/pcg32 and the frozen helper -----------------------------------------


def test_pcg32_vectors_are_current_and_match_the_frozen_helper():
    gen = COURSE / "oracle/contracts/pcg32_vectors.py"
    r = subprocess.run(
        [sys.executable, str(gen), "--check"], capture_output=True, text=True
    )
    assert r.returncode == 0, "spec/pcg32.vectors.json is stale: rerun the generator"
    sys.path.insert(0, str(COURSE / "tests"))
    try:
        from _lib.pcg32 import PCG32
    finally:
        sys.path.pop(0)
    vec = _json("spec/pcg32.vectors.json")
    for s in ["0", "1", str(1 << 63)]:
        g = PCG32(int(s))
        assert [g.next_u32() for _ in range(1024)] == vec["next_u32"][s]
        g = PCG32(int(s))
        assert [g.uniform() for _ in range(8)] == vec["uniform_f64"][s]
        g = PCG32(int(s))
        assert [g.below(10) for _ in range(16)] == vec["below_10"][s]
        # The frozen helper's normal() is the cosine half of each pair (test
        # data only, D35); it equals the spec's first normal.
        assert math.isclose(PCG32(int(s)).normal(), vec["normal"][s][0], rel_tol=1e-12)
        for p, pid in vec["purposes"].items():
            assert vec["child_seed"][s][p] == mix64(
                (int(s) + pid * 0x9E3779B97F4A7C15) & M64
            )
    md = (C / "spec/pcg32.md").read_text()
    assert (
        vec["next_u32"]["0"][0] == 0x47C28B93
        and "0x47C28B93" in md
        and "0x9AE4F7499BA72696" in md
    )


# -- spec/sampling --------------------------------------------------------------


def _oracle_rng():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pcg32_vectors", COURSE / "oracle/contracts/pcg32_vectors.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sample(
    x, *, T, top_k, top_p, min_p, seed, prompt=(), out=(), r=1.0, a_p=0.0, a_f=0.0
):
    """spec/sampling.md, transcribed step by step."""
    m = _oracle_rng()
    V = len(x)
    lg = [float(v) for v in x]
    for i in sorted(set(prompt) | set(out)):
        if r != 1.0:
            lg[i] = lg[i] / r if lg[i] > 0 else lg[i] * r
    for i in sorted(set(out)):
        c = list(out).count(i)
        lg[i] = lg[i] - a_f * c - a_p
    M = max(lg)
    Z = 0.0
    for v in lg:
        Z += math.exp(v - M)
    logp = [v - M - math.log(Z) for v in lg]
    if T == 0:
        best = max(range(V), key=lambda i: (lg[i], -i))
        return best, logp[best]
    lg = [v / T for v in lg]
    order = sorted(range(V), key=lambda i: (-lg[i], i))
    keep = order[:top_k] if 0 < top_k < V else order

    def soft(ids):
        mm = max(lg[i] for i in ids)
        z = 0.0
        for i in sorted(ids):
            z += math.exp(lg[i] - mm)
        return {i: math.exp(lg[i] - mm) / z for i in ids}

    if top_p < 1:
        q, s, kept = soft(keep), 0.0, []
        for i in sorted(keep, key=lambda i: (-lg[i], i)):
            kept.append(i)
            s += q[i]
            if s >= top_p:
                break
        keep = kept
    if min_p > 0:
        q = soft(keep)
        keep = [i for i in keep if q[i] >= min_p * max(q.values())]
    q = soft(keep)
    u = m.stream(seed, "sample").uniform()
    c = 0.0
    for i in sorted(keep):
        c += q[i]
        if u < c:
            return i, logp[i]
    return max(keep), logp[max(keep)]


def test_sampling_worked_example():
    tok, lp = _sample(
        [1.0, 3.0, 2.0, 3.0, -1.0], T=1, top_k=3, top_p=0.8, min_p=0, seed=0
    )
    assert tok == 3 and round(lp, 5) == -0.92487
    md = (C / "spec/sampling.md").read_text()
    assert "**id 3**" in md and "-0.92487" in md and "0xF88BB8A8724C81EC" in md
    assert mix64((0 + 4 * 0x9E3779B97F4A7C15) & M64) == 0xF88BB8A8724C81EC
    # greedy ignores the generator and ties go to the lowest id
    assert _sample([1.0, 3.0, 2.0, 3.0], T=0, top_k=0, top_p=1, min_p=0, seed=9)[0] == 1


# -- formats: worked examples ---------------------------------------------------


def test_kv_block_hash_and_envelope_example():
    md = (C / "formats/kv-block.md").read_text()
    assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C and crc32c(b"123456789") == 0xE3069283

    def H(parent, toks):
        return (
            fnv1a64(
                struct.pack("<Q", parent) + b"".join(struct.pack("<I", t) for t in toks)
            )
            or 1
        )

    h0 = H(0, [1, 2])
    h1 = H(h0, [3, 4])
    assert f"0x{h0:016X}" in md and f"0x{h1:016X}" in md
    payload = b"".join(struct.pack("<e", v) for v in [1, 2, 3, 4, 0.5, -1, 0, 0.25])
    body = (
        b"TLKV"
        + struct.pack("<HHIIIII", 1, 1, 1, 2, 1, 1, 2)
        + struct.pack("<QI", h0, 2)
        + payload
    )
    env = body + struct.pack("<I", crc32c(body))
    assert len(env) == 28 + (12 + 16) + 4 == 60
    assert _hexblock(md, "### Worked example (v1)") == env
    assert f"0x{crc32c(body):08X}" in md


def test_bloom_example():
    md = (C / "formats/bloom.md").read_text()
    n, p = 4, 0.1
    m = math.ceil(-n * math.log(p) / math.log(2) ** 2)
    k = max(1, math.floor(m / n * math.log(2) + 0.5))
    assert (m, k) == (20, 3)

    def bits(item):
        h1 = fnv1a64(item)
        h2 = mix64(h1) | 1
        return [((h1 + i * h2) & M64) % m for i in range(k)]

    assert (
        bits(b"cat") == [11, 14, 17]
        and bits(b"dog") == [13, 0, 3]
        and bits(b"bird") == [14, 7, 16]
    )
    arr = bytearray((m + 7) // 8)
    for it in (b"cat", b"dog"):
        for g in bits(it):
            arr[g >> 3] |= 1 << (g & 7)
    blob = b"TLBF" + struct.pack("<IQIIQ", 1, m, k, 0, 2) + bytes(arr)
    assert len(blob) == 35 and _hexblock(md, "`to_bytes()` is 35 bytes") == blob


def test_tokens_bin_example():
    md = (C / "formats/tokens-bin.md").read_text()
    hdr = struct.pack("<256i", 20240520, 1, 3, 256, *([0] * 252))
    blob = hdr + struct.pack("<3H", 1, 2, 3)
    assert len(blob) == 1030
    lines = md[md.index("## Worked example") :].split("```")[1].strip().splitlines()
    for line in lines:
        m = re.match(r"offset (\d+)\s+((?:[0-9a-f]{2} )*[0-9a-f]{2})\s", line + " ")
        if m and "..." not in line:
            off, data = int(m.group(1)), bytes.fromhex(m.group(2).replace(" ", ""))
            assert blob[off : off + len(data)] == data, line


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(num: int, payload: bytes) -> bytes:
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def test_wal_example_from_hand_protobuf():
    md = (C / "formats/wal.md").read_text()
    started = _field(1, b"Echo")
    event = _varint(1 << 3 | 0) + _varint(1) + _field(10, started)
    payload = _field(1, b"w") + _field(2, b"r") + _field(3, event)
    assert len(payload) == 18
    seq = struct.pack("<Q", 1)
    rec = struct.pack("<II", len(payload), crc32c(seq + payload)) + seq + payload
    assert _hexblock(md, "As record `seq = 1` it is 34 bytes") == rec
    assert f"0x{crc32c(seq + payload):08X}" in md


def test_usage_sql_applies_twice_and_checks_rows():
    db = sqlite3.connect(":memory:")
    sql = (C / "formats/usage.v1.sql").read_text()
    db.executescript(sql)
    db.executescript(sql)
    db.execute(
        "insert into usage (request_id, ts_ms, tenant, key_id, model, route, status, stream, e2e_ms, prompt_tokens, cached_tokens)"
        " values ('r1', 1, 'acme', 'abcdefghijkl', 'm', '/v1/chat/completions', 200, 1, 12.5, 10, 4)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "insert into usage (request_id, ts_ms, tenant, key_id, model, route, status, stream, e2e_ms, prompt_tokens, cached_tokens)"
            " values ('r2', 1, 'acme', 'abcdefghijkl', 'm', '/v1/chat/completions', 200, 1, 12.5, 1, 4)"
        )
    assert db.execute("select version from schema_version").fetchall() == [(1,)]


# -- JSON schemas ----------------------------------------------------------------


def _schemas():
    return sorted(p for p in C.rglob("*.json") if p.name.endswith(".schema.json"))


def test_every_schema_parses_resolves_and_accepts_its_examples():
    assert len(_schemas()) >= 25
    for p in _schemas():
        sch = json.loads(p.read_text())
        assert "title" in sch and "description" in sch, p
        for ref in _refs(sch, []):
            assert schema.resolve_ref(ref, sch) is not None, (p, ref)
        for ex in sch.get("examples", []):
            assert schema.validate(ex, sch) == [], (p.name, schema.validate(ex, sch))


def _design_block(after: str, lang: str) -> str:
    i = DESIGN.index(after)
    return DESIGN[i:].split(f"```{lang}\n", 1)[1].split("```", 1)[0]


def test_runtime_schema_accepts_the_design_example_and_rejects_drift():
    sch = _json("config/runtime.schema.json")
    rt = tomllib.loads(_design_block("### 2.12 Runtime configuration", "toml"))
    assert schema.validate(rt, sch) == []
    bad = json.loads(json.dumps(rt))
    bad["engine"]["prefix_cache"] = "trie"
    bad["engine"]["kv_blokcs"] = 1
    bad["gateway"]["routes"][1]["cascade"][0]["accept_if"] = "mean_logprob above -1"
    assert len(schema.validate(bad, sch)) == 3
    admin = _yaml("openapi/admin.v1.yaml")
    for r in rt["gateway"]["routes"]:
        assert schema.validate(r, admin["components"]["schemas"]["Route"], admin) == []


def test_system_schema_allows_per_instance_env():
    sch = _json("config/system.schema.json")
    sysm = {
        "system": {"name": "forge", "version": "0.4.0", "course_version": "0.2.0"},
        "services": {
            "decode": {
                "entry": "engine",
                "section": "engine",
                "env": {"TL_ENGINE__ROLE": "decode"},
            }
        },
    }
    assert schema.validate(sysm, sch) == []
    sysm["services"]["decode"]["env"] = {"TL_ENGINE__ROLE": 1}
    assert schema.validate(sysm, sch)


def test_schema_rejections():
    pol = _json("formats/policy.v1.schema.json")
    assert schema.validate(
        {
            "version": 1,
            "rules": [{"id": "x", "match": {}, "action": "block", "reason": "r"}],
        },
        pol,
    )
    head = _json("formats/linear-head.schema.json")
    assert schema.validate({**head["examples"][0], "classes": ["only"]}, head)
    prog = _json("formats/progress.schema.json")
    assert schema.validate({"ts": 1, "kind": "ckpt", "step": 5}, prog), (
        "ckpt needs a path"
    )
    assert (
        schema.validate({"ts": 1, "kind": "metric", "name": "x", "value": 1.0}, prog)
        == []
    )
    ts = _json("formats/trainer-state.schema.json")
    ex = ts["examples"][0]
    assert schema.validate({**ex, "rng": {"pcg_state": 1, "pcg_inc": "00"}}, ts)
    slo = _json("otel/slo.schema.json")
    ex = json.loads(json.dumps(slo["examples"][0]))
    ex["profile"] = "drill"
    assert schema.validate(ex, slo), "drill with prod windows"
    ex = json.loads(json.dumps(slo["examples"][0]))
    del ex["alerts"]["TPOTBudgetBurnSlow"]
    assert schema.validate(ex, slo)
    ex = json.loads(json.dumps(slo["examples"][0]))
    ex["slos"]["ttft"]["objective"] = 0.9
    assert schema.validate(ex, slo), "looser than the course default"
    res = _json("formats/eval-result.schema.json")
    summ = {
        "suite": "s",
        "run_id": "r",
        "subjects": ["a", "b"],
        "n_boot": 1000,
        "seed": 0,
        "metrics": {
            "a": {
                "em": {
                    "mean": 0.5,
                    "ci_low": 0.4,
                    "ci_high": 0.6,
                    "n": 100,
                    "errored": 0,
                }
            }
        },
        "ab": {
            "base": "a",
            "exp": "b",
            "metrics": {
                "em": {
                    "delta": 0.1,
                    "ci_low": 0.0,
                    "ci_high": 0.2,
                    "p_value": 0.04,
                    "n_pairs": 100,
                }
            },
        },
    }
    assert schema.validate(summ, res["$defs"]["summary"], res) == []


TOKENIZERS = [
    *sorted(
        Path.home().glob(
            ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-*/snapshots/*/tokenizer.json"
        )
    ),
    *sorted(
        Path.home().glob(
            ".cache/huggingface/hub/models--openai-community--gpt2/snapshots/*/tokenizer.json"
        )
    ),
    *sorted(
        Path.home().glob(
            ".cache/huggingface/hub/models--*bert*/snapshots/*/tokenizer.json"
        )
    ),
]


def test_tokenizer_subset_schema():
    sch = _json("formats/tokenizer-json.schema.json")
    ex = sch["examples"][0]
    assert schema.validate({**ex, "model": {**ex["model"], "byte_fallback": True}}, sch)
    assert schema.validate(
        {**ex, "pre_tokenizer": {"type": "Split", "pattern": " "}}, sch
    )
    assert schema.validate({**ex, "model": {"type": "WordLevel", "vocab": {}}}, sch)
    for p in TOKENIZERS:  # real files when a local HF cache has them; never downloaded
        assert schema.validate(json.loads(p.read_text()), sch) == [], p


# -- OpenAPI -----------------------------------------------------------------------


def _ok(inst, spec, name):
    return schema.validate(inst, spec["components"]["schemas"][name], spec)


def test_openapi_files_parse_and_refs_resolve():
    for f in ("openai-subset.v1.yaml", "openai-subset.v2.yaml", "admin.v1.yaml"):
        spec = _yaml(f"openapi/{f}")
        assert spec["openapi"].startswith("3.1"), f
        for ref in _refs(spec, []):
            assert schema.resolve_ref(ref, spec) is not None, (f, ref)
        ops = [
            op["operationId"]
            for path in spec["paths"].values()
            for k, op in path.items()
            if isinstance(op, dict) and "operationId" in op
        ]
        assert len(ops) == len(set(ops)), f


def test_openapi_v1_surface_matches_design():
    spec = _yaml("openapi/openai-subset.v1.yaml")
    assert set(spec["paths"]) == {
        "/healthz",
        "/readyz",
        "/metrics",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/models",
        "/v1/models/{model}",
        "/v1/tokenize",
    }
    chat = spec["paths"]["/v1/chat/completions"]["post"]["responses"]
    assert {"200", "400", "401", "403", "404", "422", "429", "451", "503"} <= set(chat)
    fields = set(
        spec["components"]["schemas"]["ChatCompletionRequest"]["allOf"][1]["properties"]
    ) | set(spec["components"]["schemas"]["SamplingFields"]["properties"])
    for f in [
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "logprobs",
        "top_logprobs",
        "tools",
        "tool_choice",
        "response_format",
        "user",
    ]:
        assert f in fields, f
    codes = spec["components"]["schemas"]["Error"]["properties"]["error"]["properties"][
        "code"
    ]["enum"]
    for c in [
        "invalid_api_key",
        "insufficient_scope",
        "model_not_found",
        "unsupported_parameter",
        "rate_limit_exceeded",
        "usage_policy",
        "no_capacity",
    ]:
        assert c in codes


def test_openapi_v1_bodies():
    spec = _yaml("openapi/openai-subset.v1.yaml")
    req = {
        "model": "smol",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "42", "tool_call_id": "c1"},
        ],
        "temperature": 0,
        "stop": ["\n"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}},
            }
        ],
        "tool_choice": "auto",
    }
    assert _ok(req, spec, "ChatCompletionRequest") == []
    assert _ok({**req, "temperature": -1}, spec, "ChatCompletionRequest")
    assert _ok(
        {**req, "messages": [{"role": "tool", "content": "x"}]},
        spec,
        "ChatCompletionRequest",
    )
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    done = {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": "smol",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": usage,
    }
    assert _ok(done, spec, "ChatCompletion") == []
    first = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "smol",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    frag = {
        **first,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": '{"a"'}}]
                },
                "finish_reason": None,
            }
        ],
    }
    last = {
        **first,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    usage_chunk = {**first, "choices": [], "usage": usage}
    for ch in (first, frag, last, usage_chunk):
        assert _ok(ch, spec, "ChatCompletionChunk") == [], ch
    assert _ok(
        {**last, "choices": [{"index": 0, "delta": {}, "finish_reason": "eos"}]},
        spec,
        "ChatCompletionChunk",
    )
    emb = {
        "object": "list",
        "model": "smol",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.6, 0.8]}],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    assert _ok(emb, spec, "EmbeddingList") == []
    err = {
        "error": {
            "message": "x",
            "type": "policy_error",
            "param": None,
            "code": "usage_policy",
        }
    }
    assert _ok(err, spec, "Error") == []
    assert _ok(
        {
            "error": {
                "message": "x",
                "type": "policy_error",
                "param": None,
                "code": "nope",
            }
        },
        spec,
        "Error",
    )


def test_openapi_v2_differs_from_v1_exactly_as_designed():
    v1, v2 = (
        _yaml("openapi/openai-subset.v1.yaml"),
        _yaml("openapi/openai-subset.v2.yaml"),
    )
    assert set(v1["paths"]) == set(v2["paths"])
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert _ok(usage, v1, "Usage") == [] and _ok(usage, v2, "Usage")
    assert (
        _ok({**usage, "prompt_tokens_details": {"cached_tokens": 2}}, v2, "Usage") == []
    )
    comp = v2["paths"]["/v1/completions"]["post"]
    assert comp["deprecated"] is True
    hdrs = comp["responses"]["200"]["headers"]
    assert hdrs["Deprecation"]["required"] and hdrs["Sunset"]["required"]
    codes2 = v2["components"]["schemas"]["Error"]["properties"]["error"]["properties"][
        "code"
    ]["enum"]
    assert "rate_limit_exceeded" not in codes2
    assert {
        "requests_limit_exceeded",
        "tokens_limit_exceeded",
        "queue_full",
        "model_draining",
    } <= set(codes2)
    names = {p["name"] for p in v2["components"]["parameters"].values()}
    assert "X-TL-API-Version" in names


def test_admin_surface_matches_design():
    spec = _yaml("openapi/admin.v1.yaml")
    want = {
        ("/admin/v1/keys", "post"),
        ("/admin/v1/keys", "get"),
        ("/admin/v1/keys/{key_id}", "delete"),
        ("/admin/v1/workers", "get"),
        ("/admin/v1/routes", "get"),
        ("/admin/v1/routes", "put"),
        ("/admin/v1/models/{model}:drain", "post"),
        ("/admin/v1/usage", "get"),
        ("/admin/v1/policy", "get"),
        ("/admin/v1/policy", "put"),
        ("/admin/v1/cache:purge", "post"),
    }
    have = {(p, m) for p, ops in spec["paths"].items() for m in ops}
    assert want == have
    created = {
        "key_id": "abcdefghijkl",
        "tenant": "acme",
        "name": "n",
        "scopes": ["infer"],
        "rpm": 60,
        "tpm": 1000,
        "priority": 0,
        "models": [],
        "created_at": "2026-10-09T00:00:00Z",
        "expires_at": None,
        "revoked": False,
        "key": "tl_abcdefghijkl_" + "A" * 32,
    }
    assert _ok(created, spec, "KeyCreated") == []
    assert _ok({**created, "key": "sk-123"}, spec, "KeyCreated")
    pol = _json("formats/policy.v1.schema.json")["examples"][0]
    assert _ok(pol, spec, "PolicyDocument") == []


# -- protos and generated code ---------------------------------------------------------


PROTOS = sorted((C / "proto").rglob("*.proto"))


def _messages(text: str) -> list[str]:
    return re.findall(r"^\s*message (\w+) \{", text, re.M)


def test_protos_match_design_and_generated_code():
    assert [p.relative_to(C / "proto").as_posix() for p in PROTOS] == [
        "tl/control/v1/control.proto",
        "tl/durable/v1/durable.proto",
        "tl/engine/v1/engine.proto",
        "tl/kv/v1/kv.proto",
        "tl/raft/v1/raft.proto",
    ]
    rpcs = {
        "engine": {"Prefill", "Info", "Cancel", "Drain"},
        "kv": {"HasBlocks", "PushKv", "Release"},
        "control": {"Heartbeat"},
        "durable": {
            "StartWorkflow",
            "SignalWorkflow",
            "CancelWorkflow",
            "DescribeWorkflow",
            "GetHistory",
            "ListWorkflows",
            "ListDeadLetters",
            "RedriveDeadLetter",
            "PollWorkflowTask",
            "CompleteWorkflowTask",
            "PollActivityTask",
            "RecordHeartbeat",
            "CompleteActivityTask",
            "FailActivityTask",
        },
        "raft": {"RequestVote", "AppendEntries", "InstallSnapshot"},
    }
    for p in PROTOS:
        text = p.read_text()
        pkg = p.parent.parent.name
        assert f"package tl.{pkg}.v1;" in text
        assert (
            f'option go_package = "supersource.urmzd.com/tl/contracts/gen/tl/{pkg}/v1;{pkg}v1";'
            in text
        )
        assert set(re.findall(r"rpc (\w+)\(", text)) == rpcs[pkg], pkg
        go = (C / f"go/gen/tl/{pkg}/v1/{pkg}.pb.go").read_text()
        rs = (C / f"rust/tl-proto/src/gen/tl.{pkg}.v1.rs").read_text()
        for msg in _messages(text):
            assert re.search(rf"^type \w*{msg} struct", go, re.M), (pkg, msg)
            assert (
                re.search(rf"pub struct {msg} \{{", rs) or f"pub struct {msg} {{" in rs
            ), (pkg, msg)
        assert (C / f"go/gen/tl/{pkg}/v1/{pkg}_grpc.pb.go").is_file()
    # the design's fixed field numbers survive (2.7)
    eng = (C / "proto/tl/engine/v1/engine.proto").read_text()
    for frag in [
        "uint32 rng_draws_consumed = 8;",
        "repeated uint64 block_hashes = 4;",
        "int32 priority = 7;",
    ]:
        assert frag in eng
    dur = (C / "proto/tl/durable/v1/durable.proto").read_text()
    for frag in [
        "ActivityDeadLettered dead_lettered = 27;",
        "uint64 lease_token = 10;",
        "string idempotency_key = 7;",
        "ContinueAsNew continue_as_new = 6;",
        "message WalRecord {",
    ]:
        assert frag in dur


@pytest.mark.skipif(shutil.which("go") is None, reason="no go toolchain")
def test_contracts_go_module_builds(tmp_path):
    env = {**os.environ, "GOWORK": "off", "GOFLAGS": "-mod=mod"}
    r = subprocess.run(
        ["go", "build", "./..."],
        cwd=C / "go",
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if r.returncode != 0 and ("dial tcp" in r.stderr or "proxy.golang.org" in r.stderr):
        pytest.skip("go module cache cold and no network")
    assert r.returncode == 0, r.stderr


# -- OTel -------------------------------------------------------------------------------


def test_metrics_yaml_covers_design_and_names_translate():
    d = _yaml("otel/metrics.yaml")
    inst = {i["name"]: i for i in d["instruments"]}
    assert len(inst) == len(d["instruments"])
    for name in [
        "gen_ai.server.time_to_first_token",
        "gen_ai.server.time_per_output_token",
        "gen_ai.server.request.duration",
        "gen_ai.client.token.usage",
        "http.server.request.duration",
        "tl.engine.kv.blocks",
        "tl.engine.queue.depth",
        "tl.engine.batch.tokens",
        "tl.engine.active_sequences",
        "tl.engine.prefix_cache.hit_ratio",
        "tl.gateway.requests",
        "tl.gateway.ratelimit.rejections",
        "tl.gateway.cache.hits",
        "tl.durable.task_queue.depth",
        "tl.durable.dlq.size",
        "tl.durable.redeliveries",
        "tl.durable.task.schedule_to_start",
        "tl.train.loss",
        "tl.train.tokens_per_second",
        "tl.train.grad_norm",
        "tl.engine.spec_accept_rate",
    ]:
        assert name in inst, name
    assert (
        inst["gen_ai.server.time_to_first_token"]["prometheus"]
        == "gen_ai_server_time_to_first_token_seconds"
    )
    for i in d["instruments"]:
        base = i["name"].replace(".", "_")
        suffix = {"s": "_seconds", "By": "_bytes"}.get(i["unit"], "")
        want = (
            base
            + ("" if base.endswith(suffix) else suffix)
            + ("_total" if i["type"] == "counter" else "")
        )
        assert i["prometheus"] == want, i["name"]
        if i["type"] == "histogram":
            assert i["buckets"] == sorted(set(i["buckets"])), i["name"]
        assert i["type"] in {"histogram", "counter", "gauge"}


def test_semconv_lists_every_design_span():
    md = (C / "otel/semconv.md").read_text()
    for span in [
        "POST /v1/chat/completions",
        "gateway.auth",
        "gateway.ratelimit",
        "gateway.policy",
        "gateway.cache",
        "gateway.route",
        "gateway.proxy",
        "tl.engine.v1.EngineControl/Prefill",
        "engine.queue",
        "engine.prefill",
        "engine.decode",
        "kv.transfer",
        "workflow <type>",
        "activity <type>",
        "train.run",
        "train.step",
        "train.checkpoint",
        "corpus.stage <stage>",
        "agent.run",
        "agent.llm_call",
        "agent.tool <name>",
        "rag.retrieve",
        "eval.case",
    ]:
        assert f"`{span}`" in md, span


# -- Helm values schemas -------------------------------------------------------------------


def _values(chart: str) -> dict:
    common = {
        "image": {"repository": f"localhost:5001/forge-{chart}", "tag": "0.4.0"},
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "1", "memory": "256Mi"},
        },
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "allowPrivilegeEscalation": False,
        },
        "probes": {"liveness": {"path": "/healthz"}, "readiness": {"path": "/readyz"}},
        "pdb": {"enabled": True, "maxUnavailable": 1},
    }
    extra = {
        "gateway": {
            "replicaCount": 1,
            "service": {"type": "NodePort", "nodePort": 30080},
            "pepper": {"name": "gw", "key": "pepper"},
        },
        "engine": {
            "role": "decode",
            "model": {"dir": "/artifacts/models/smol-135m/v3"},
        },
        "durable": {
            "replicaCount": 1,
            "persistence": {"enabled": True, "size": "5Gi"},
            "walMaxBytes": 2147483648,
        },
        "worker": {
            "taskQueues": ["train"],
            "durableAddress": "forge-durable:7233",
            "autoscaling": {"enabled": True, "minReplicas": 1, "maxReplicas": 4},
        },
        "agent": {
            "replicaCount": 1,
            "durableAddress": "forge-durable:7233",
            "provider": {
                "baseUrl": "http://forge-gateway:8080/v1",
                "model": "smol",
                "apiKey": {"name": "agent", "key": "k"},
            },
        },
    }[chart]
    return {**common, **extra}


@pytest.mark.parametrize("chart", ["gateway", "engine", "durable", "worker", "agent"])
def test_helm_values_schema_policy(chart, tmp_path):
    sch = _json(f"helm/{chart}.values.schema.json")
    good = _values(chart)
    assert schema.validate(good, sch) == []
    for bad in (
        {**good, "image": {**good["image"], "tag": "latest"}},
        {**good, "securityContext": {**good["securityContext"], "runAsNonRoot": False}},
        {**good, "env": [{"name": "TL_GATEWAY_PEPPER", "value": "hunter2"}]},
        {k: v for k, v in good.items() if k != "pdb"},
        {**good, "resources": {"requests": good["resources"]["requests"]}},
    ):
        assert schema.validate(bad, sch), bad
    ok_env = {
        **good,
        "env": [
            {
                "name": "TL_API_KEY",
                "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}},
            },
            {"name": "TL_ENGINE__ROLE", "value": "decode"},
        ],
    }
    assert schema.validate(ok_env, sch) == []
    if shutil.which("helm"):
        (tmp_path / "c/templates").mkdir(parents=True)
        (tmp_path / "c/Chart.yaml").write_text(
            f"apiVersion: v2\nname: forge-{chart}\nversion: 0.1.0\n"
        )
        (tmp_path / "c/values.schema.json").write_text(
            (C / f"helm/{chart}.values.schema.json").read_text()
        )
        (tmp_path / "c/values.yaml").write_text(json.dumps(good))
        r = subprocess.run(
            ["helm", "lint", str(tmp_path / "c")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode == 0, r.stdout + r.stderr
        r = subprocess.run(
            ["helm", "template", str(tmp_path / "c"), "--set", "image.tag=latest"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode != 0


# -- the rest ---------------------------------------------------------------------------------


def test_allowed_deps_parse():
    d = tomllib.loads((C / "allowed-deps.toml").read_text())
    assert d["rust"]["tl-proto"] == {
        "prost": "0.14",
        "tonic": "0.14",
        "tonic-prost": "0.14",
    }
    cargo = tomllib.loads((C / "rust/tl-proto/Cargo.toml").read_text())
    assert cargo["dependencies"] == d["rust"]["tl-proto"]
    gomod = (C / "go/go.mod").read_text()
    assert (
        "google.golang.org/grpc v1." in gomod
        and "google.golang.org/protobuf v1." in gomod
    )
    assert {"pyarrow", "zstandard"} <= set(d["python"]["corpus"])


def test_markdown_links_in_contracts_resolve():
    bad = []
    for p in C.rglob("*.md"):
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", p.read_text()):
            if "://" in target:
                continue
            if not (p.parent / target).exists():
                bad.append(f"{p.relative_to(C)} -> {target}")
    assert bad == []


def test_version_is_a_minor_bump():
    # 0.3.0 for B2 to B13's additive contracts; 0.4.0 for craft.14's v2 probes
    # and error code. Still 0.x minor bumps: nothing removed or renamed.
    v = tomllib.loads((C / "VERSION").read_text())
    assert v["semver"] == "0.4.0"
