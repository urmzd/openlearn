"""The B1 contracts in course/contracts parse, resolve, and accept the examples
the design gives (DESIGN 2.4, 2.6, 2.9, 2.10, 2.16). They are the source the
learner vendors, so a broken contract breaks every learner at once."""

import json
import re
import shutil
import struct
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from olcourse import schema

COURSE = Path(__file__).resolve().parents[2]
C = COURSE / "contracts"
FIXTURE_SYSTEM = (
    COURSE.parent / "practice/bin/tests/fixtures/site/course/ref/system.toml"
)

DESIGN_SYSTEM_TOML = """
[system]
name    = "forge"
version = "0.4.0"
course_version = "1.0.0"
[build]
steps = [["make", "-C", "c"], ["cargo", "build", "--release", "--manifest-path", "rust/Cargo.toml"]]
[entry]
tinyllm  = ["uv", "run", "--project", "python", "python", "-m", "tinyllm"]
corpus   = ["uv", "run", "--project", "python", "python", "-m", "corpus"]
tl-tok   = ["rust/target/release/tl-tok"]
engine   = ["rust/target/release/tl-serve", "--config", "{config}"]
gateway  = ["go", "run", "./go/cmd/gateway", "--config", "{config}"]
durable  = ["go", "run", "./go/cmd/durable", "--data", "{data}", "--port", "{grpc_port}"]
worker   = ["go", "run", "./go/cmd/worker", "--queue", "{queue}", "--durable", "127.0.0.1:{durable.grpc_port}"]
ctl      = ["go", "run", "./go/cmd/forge"]
loadgen  = ["go", "run", "./go/cmd/loadgen"]
agent    = ["go", "run", "./go/cmd/agent-docsqa"]
[services.engine]
config = "deploy/runtime.dev.toml"
health = "http://127.0.0.1:{health_port}/healthz"
ready_timeout_s = 60
[services.gateway]
config = "deploy/runtime.dev.toml"
health = "http://127.0.0.1:{health_port}/readyz"
after  = ["engine"]
[endpoints]
api_base    = "http://127.0.0.1:{gateway.port}/v1"
api_key_env = "TL_API_KEY"
[ci]
local = [["just", "ci"]]
[deploy]
kube_context = "kind-forge"
namespace    = "forge"
gateway_url  = "http://127.0.0.1:30080"
prometheus   = "http://127.0.0.1:30090"
traces       = "http://127.0.0.1:30320"
services     = { gateway = "deploy/forge-gateway", decode = "deploy/forge-engine-decode", prefill = "deploy/forge-engine-prefill", durable = "statefulset/forge-durable" }
"""


def _json(rel: str) -> dict:
    return json.loads((C / rel).read_text())


def _refs(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref":
                out.append(v)
            else:
                _refs(v, out)
    elif isinstance(node, list):
        for v in node:
            _refs(v, out)
    return out


# -- C ABI -------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_headers_compile_and_declare_v0(tmp_path):
    src = tmp_path / "use.c"
    src.write_text(
        '#include "tinyllm.h"\n'
        '_Static_assert(TL_ABI_VERSION == 1, "abi v1");\n'
        '_Static_assert(sizeof(tl_status) == 4 && sizeof(tl_dtype) == 4, "int32 codes");\n'
        "int use(void) {\n"
        "  float a[4] = {1, 2, 3, 4}, b[4] = {1, 0, 0, 1}, c[4] = {0};\n"
        "  tl_allocator al = {0, 0, 0};\n"
        "  tl_set_last_error(tl_status_str(TL_EUNSUPPORTED));\n"
        "  (void)tl_set_allocator(&al); tl_free(tl_alloc(64, 64));\n"
        "  return (int)tl_abi_version() + (int)tl_last_error()[0]\n"
        "    + tl_matmul_f32(a, b, c, 2, 2, 2, 2, 2, 2, 1.0f, 0.0f, 0, (tl_pool *)0);\n"
        "}\n"
    )
    for flags in (
        ["-std=c11", "-Wall", "-Werror"],
        ["-std=c11", "-pedantic", "-Wall", "-Wextra", "-Werror"],
    ):
        r = subprocess.run(
            ["cc", *flags, f"-I{C / 'c/include'}", "-fsyntax-only", str(src)],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
    for h in ("tinyllm.h", "tinyllm/abi.h", "tinyllm/matmul.h"):
        one = tmp_path / "one.c"
        one.write_text(f'#include "{h}"\n#include "{h}"\n')
        r = subprocess.run(
            [
                "cc",
                "-std=c11",
                "-pedantic",
                "-Wall",
                "-Werror",
                f"-I{C / 'c/include'}",
                "-fsyntax-only",
                str(one),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{h} alone, twice:\n{r.stderr}"


def test_umbrella_keeps_the_v0_units():
    # Contract 0.2.0 adds every other ABI v1 unit (test_contracts_b2_b13.py).
    text = (C / "c/include/tinyllm.h").read_text()
    inc = re.findall(r'#include "([^"]+)"', text)
    assert inc[0] == "tinyllm/abi.h" and "tinyllm/matmul.h" in inc


# -- OpenAPI v0 --------------------------------------------------------------


def test_openapi_v0_parses_and_refs_resolve():
    spec = yaml.safe_load((C / "openapi/openai-subset.v0.yaml").read_text())
    assert spec["openapi"].startswith("3.1")
    assert set(spec["paths"]) == {"/healthz", "/readyz", "/v1/completions"}
    for ref in _refs(spec, []):
        assert schema.resolve_ref(ref, spec) is not None, ref
    post = spec["paths"]["/v1/completions"]["post"]
    assert post["responses"]["200"]["content"]["text/event-stream"]["x-tl-chunk"]
    req = spec["components"]["schemas"]["CompletionRequest"]
    assert set(req["properties"]) == {
        "model",
        "prompt",
        "max_tokens",
        "temperature",
        "seed",
        "stream",
    }


def test_openapi_v0_schemas_accept_and_reject():
    spec = yaml.safe_load((C / "openapi/openai-subset.v0.yaml").read_text())
    s = spec["components"]["schemas"]

    def ok(inst, name):
        return schema.validate(inst, s[name], spec) == []

    assert ok(
        {
            "model": "tracer",
            "prompt": "Once",
            "max_tokens": 8,
            "temperature": 0,
            "seed": 0,
            "stream": False,
        },
        "CompletionRequest",
    )
    assert not ok(
        {"model": "tracer", "prompt": "Once", "temperature": -1}, "CompletionRequest"
    )
    assert not ok({"model": "tracer", "prompt": ""}, "CompletionRequest")
    done = {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 1,
        "model": "tracer",
        "choices": [{"index": 0, "text": "hi", "finish_reason": "length"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    assert ok(done, "Completion")
    assert not ok(
        {**done, "choices": [{"index": 0, "text": "hi", "finish_reason": None}]},
        "Completion",
    )
    chunk = {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 1,
        "model": "tracer",
        "choices": [{"index": 0, "text": "", "finish_reason": None}],
    }
    assert ok(chunk, "CompletionChunk")
    err = {
        "error": {
            "message": "bad",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": None,
        }
    }
    assert ok(err, "Error")
    assert not ok({"error": {"message": "bad"}}, "Error")


# -- formats -----------------------------------------------------------------


SMOLLM2_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 49152,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-5,
    "rope_theta": 100000,
    "rope_scaling": None,
    "hidden_act": "silu",
    "tie_word_embeddings": True,
    "bos_token_id": 0,
    "eos_token_id": 0,
    "torch_dtype": "bfloat16",
    "pretraining_tp": 1,
    "use_cache": True,
}


def test_config_schema():
    sch = _json("formats/config.schema.json")
    bigram = {
        "tl_arch": "bigram",
        "tl_tokenizer": "bytes",
        "vocab_size": 256,
        "tl_format": 1,
    }
    assert schema.validate(bigram, sch) == []
    assert schema.validate(SMOLLM2_CONFIG, sch) == []
    assert schema.validate({**SMOLLM2_CONFIG, "tl_arch": "llama"}, sch) == []
    assert schema.validate({**bigram, "vocab_size": 300}, sch), "bytes needs 256 ids"
    assert schema.validate({**bigram, "tl_arch": "mamba"}, sch), "unknown arch"
    assert schema.validate({"vocab_size": 256}, sch), "no arch and not llama"
    no_dims = {k: v for k, v in SMOLLM2_CONFIG.items() if k != "hidden_size"}
    assert schema.validate(no_dims, sch), "llama without hidden_size"
    for ex in sch["examples"]:
        assert schema.validate(ex, sch) == []


def test_safetensors_worked_example():
    # The bytes formats/safetensors.md derives by hand.
    header = {
        "__metadata__": {"format": "tinyllm"},
        "w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
    }
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    md = (C / "formats/safetensors.md").read_text()
    assert text.decode() in md and len(text) == 93
    text += b" " * ((8 - len(text) % 8) % 8)
    blob = struct.pack("<Q", len(text)) + text + struct.pack("<4f", 1, 2, 3, 4)
    assert len(blob) == 120 and blob[:8].hex(" ") in md


def test_tokenizer_worked_examples():
    import codecs

    assert list("héllo".encode()) == [104, 195, 169, 108, 108, 111]
    assert bytes([226, 130]).decode("utf-8", "replace") == "�"
    d = codecs.getincrementaldecoder("utf-8")("replace")
    assert [d.decode(bytes([b])) for b in (195, 169, 255)] == ["", "é", "�"]


# -- harness manifest ----------------------------------------------------------


def test_system_schema_accepts_design_and_fixture_manifests():
    sch = _json("config/system.schema.json")
    for ref in _refs(sch, []):
        assert schema.resolve_ref(ref, sch) is not None, ref
    assert schema.validate(tomllib.loads(DESIGN_SYSTEM_TOML), sch) == []
    assert schema.validate(tomllib.loads(FIXTURE_SYSTEM.read_text()), sch) == []
    bad = tomllib.loads(DESIGN_SYSTEM_TOML)
    bad["entry"]["enigne"] = ["x"]
    bad["services"]["engine"]["ready_timeout_s"] = 0
    bad["system"]["name"] = "Forge"
    errs = schema.validate(bad, sch)
    assert len(errs) == 3, errs
    assert schema.validate({**bad, "entry": {"engine": "tl-serve --port 1"}}, sch), (
        "argv must be a list"
    )


def test_contracts_have_no_em_dashes():
    for p in C.rglob("*"):
        if p.is_file():
            assert "\u2014" not in p.read_text(errors="replace"), p
