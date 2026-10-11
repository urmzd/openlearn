"""Unit tests for the pieces behind ol milestone, conform, and drill that need
no learner repo: the JSON Schema subset, placeholders, runtime.toml
generation, SSE framing, matchers, milestone composition, drill grading."""

import tomllib
from pathlib import Path

import pytest

from olcourse import (
    HarnessError,
    drills,
    matchers,
    milestones,
    placeholders,
    schema,
    services,
    web,
)


def test_schema_subset():
    root = {
        "components": {
            "schemas": {
                "C": {
                    "type": "object",
                    "required": ["n"],
                    "properties": {
                        "n": {"type": "integer", "minimum": 1},
                        "s": {"type": ["string", "null"], "enum": ["a", None]},
                    },
                }
            }
        }
    }
    s = {"$ref": "#/components/schemas/C"}
    assert schema.validate({"n": 2, "s": None}, s, root) == []
    assert schema.validate({"n": 0}, s, root) == ["$.n: 0 < minimum 1"]
    assert schema.validate({"s": "b"}, s, root) == [
        "$: missing required property 'n'",
        "$.s: 'b' is not one of ['a', None]",
    ]
    assert schema.validate(True, {"type": "integer"}) == [
        "$: expected integer, got boolean"
    ]
    assert schema.validate(
        [1, "x"], {"type": "array", "items": {"type": "number"}}
    ) == ["$[1]: expected number, got string"]
    assert schema.validate({"a": 1}, {"additionalProperties": False}) == [
        "$: unexpected property 'a'"
    ]
    assert schema.validate(3, {"oneOf": [{"type": "integer"}, {"type": "number"}]}) == [
        "$: matches 2 of oneOf, want exactly 1"
    ]
    assert schema.validate(None, {"type": "string", "nullable": True}) == []
    assert schema.validate("x", {"const": "text_completion"}) == [
        "$: expected 'text_completion', got 'x'"
    ]


def test_placeholders_splice_and_errors():
    lk = placeholders.chain(
        {"port": "8000", "out": "/tmp/o"}.get,
        placeholders.from_dict({"deploy": {"ns": "forge"}}),
        lambda n: ["uv", "run", "x"] if n == "tinyllm" else None,
    )
    assert placeholders.expand_argv(
        ["{tinyllm}", "--port", "{port}", "--json", '{"a": 1}', "{deploy.ns}/x"], lk
    ) == ["uv", "run", "x", "--port", "8000", "--json", '{"a": 1}', "forge/x"]
    assert (
        placeholders.expand("run {tinyllm} > {out}/log", lk)
        == "run uv run x > /tmp/o/log"
    )
    with pytest.raises(HarnessError, match=r"unknown placeholder \{nope\} in step s"):
        placeholders.expand("{nope}", lk, "step s")
    assert placeholders.expand_obj({"a": ["{port}"], "b": 1}, lk) == {
        "a": ["8000"],
        "b": 1,
    }


def test_runtime_toml_forces_allocated_ports():
    ports = {
        "port": 101,
        "health_port": 102,
        "grpc_port": 103,
        "kv_port": 104,
        "registry_port": 105,
    }
    tmpl = '[engine]\nhttp_listen = ":8000"\nmodel_dir = "{data}/m"\n\n[gateway]\nupstream = "http://127.0.0.1:{port}"\n'
    text, env = services.runtime_toml(
        tmpl, "engine", ports, {"data": "/d", "port": "101"}.get, "t"
    )
    doc = tomllib.loads(text)
    assert (
        doc["engine"]["http_listen"] == "127.0.0.1:101"
        and doc["engine"]["grpc_listen"] == "127.0.0.1:103"
    )
    assert (
        doc["engine"]["kv_listen"] == "127.0.0.1:104"
        and doc["engine"]["health_listen"] == "127.0.0.1:102"
    )
    assert (
        doc["engine"]["model_dir"] == "/d/m"
        and doc["gateway"]["upstream"] == "http://127.0.0.1:101"
    )
    assert (
        env["TL_ENGINE__HTTP_LISTEN"] == "127.0.0.1:101"
        and env["TL_ENGINE__HEALTH_LISTEN"] == "127.0.0.1:102"
    )
    text, env = services.runtime_toml("", "gateway", ports, {}.get, "t")
    assert tomllib.loads(text)["gateway"] == {
        "listen": "127.0.0.1:101",
        "registry_listen": "127.0.0.1:105",
        "health_listen": "127.0.0.1:102",
    }
    pool = services.PortPool()
    assert len({pool.take() for _ in range(20)}) == 20


def test_sse_framing():
    ok = b'data: {"a":1}\n\n: ping\n\ndata: {"a":2}\n\ndata: [DONE]\n\n'
    assert web.sse_events(ok) == (['{"a":1}', '{"a":2}', "[DONE]"], [])
    _, errs = web.sse_events(b'data: {"a":1}\n\n')
    assert errs == ["stream does not end with `data: [DONE]`"]
    _, errs = web.sse_events(b'data: {"a":1}\ndata: x\n\ndata: [DONE]\n\n')
    assert errs and "want one `data: <json>` line" in errs[0]
    _, errs = web.sse_events(b"data: [DONE]\n")
    assert "stream does not end with a blank line" in errs


def test_matcher_helpers():
    assert matchers.normalize("\n\n a  \n\nb \n\n") == " a\n\nb"
    assert matchers.last_json('progress\n{"ids": [1, 2]}\n') == {"ids": [1, 2]}
    assert matchers.last_json("not json") is None
    for good in ("feat: x", "fix(gw.03)!: y", "chore(tests): vendor"):
        assert matchers.CONVENTIONAL.match(good)
    for bad in ("wip stuff", "feat:x", "Feat: x", "feat(): x"):
        assert not matchers.CONVENTIONAL.match(bad)
    assert matchers._cmp(0.96, ">= 0.95") == (True, "0.96 >= 0.95")
    with pytest.raises(HarnessError, match="step context"):
        matchers._cmp(1.0, "<= calibrated")


def _ms(d: Path, name: str, body: str) -> None:
    (d / "milestones").mkdir(exist_ok=True)
    (d / "milestones" / f"{name}.toml").write_text(f'id = "{name}"\n' + body)


def test_pass_gates_compose_and_rerun_earlier_smoke(tmp_path):
    _ms(
        tmp_path,
        "MS-P0",
        '[[step]]\nname = "a"\nsmoke = true\nexpect = { match = "git-log" }\n'
        '[[step]]\nname = "b"\nexpect = { match = "ci-status" }\n',
    )
    _ms(
        tmp_path,
        "MS-L1",
        'requires = ["L1.1"]\n[[step]]\nname = "c"\nrun = "tinyllm"\nsmoke = true\n'
        '[[step]]\nname = "d"\nrun = "tinyllm"\nci = "kind"\nsmoke = true\n',
    )
    _ms(
        tmp_path,
        "MS-P1",
        'includes = ["MS-L1"]\nrequires = ["rt.01"]\n[[step]]\nname = "e"\nrun = "tinyllm"\n',
    )
    _ms(tmp_path, "MS-P2", '[[step]]\nname = "f"\nrun = "tinyllm"\n')
    p = milestones.plan(tmp_path, "MS-P1", smoke=False)
    assert [s.qualified("MS-P1") for s in p.steps] == [
        "e",
        "MS-L1/c",
        "MS-L1/d",
        "MS-P0/a",
    ]
    assert p.requires == ["rt.01", "L1.1"]
    p = milestones.plan(tmp_path, "MS-P2", smoke=True)
    assert [s.qualified("MS-P2") for s in p.steps] == [
        "MS-P0/a",
        "MS-L1/c",
    ]  # no kind step in --smoke
    v = milestones.Step("s", "MS-X", matrix={"b": [1, 2], "a": ["x"]}).variants()
    assert v == [{"a": "x", "b": 1}, {"a": "x", "b": 2}]
    _ms(
        tmp_path,
        "MS-bad",
        '[[step]]\nname = "g"\nexpect = { match = "tokens-equal" }\n',
    )
    with pytest.raises(HarnessError, match="needs a command"):
        milestones.load(tmp_path, "MS-bad")
    _ms(tmp_path, "MS-loop", 'includes = ["MS-loop"]\n')
    with pytest.raises(HarnessError, match="include cycle"):
        milestones.plan(tmp_path, "MS-loop", smoke=False)


def test_drill_grading_helpers(tmp_path):
    assert drills.final_run_start([0, 15, 30, 60, 75, 90], 90, 15) == 60
    assert (
        drills.final_run_start([0, 15, 30], 90, 15) is None
    )  # stopped holding before the end
    assert drills.final_run_start([], 90, 15) is None
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "2026-10-08-x.md").write_text(
        "# PM\n## Summary\n### Root cause ##\n"
    )
    path, errs = drills.postmortem_errors(
        tmp_path,
        {"path": "docs/{date}-x.md", "sections": ["Summary", "Root cause", "Impact"]},
    )
    assert path.name == "2026-10-08-x.md" and errs == [
        "docs/2026-10-08-x.md lacks section(s): Impact"
    ]
    j = drills.Journal.open(tmp_path / "run")
    j.append({"n": 0, "kind": "a", "undo": [["x"]]})
    j.append({"n": 1, "kind": "b", "undo": []})
    j.append({"n": 2, "kind": "c", "undo": [["y"]]})
    j.append({"undone": 2})
    assert [e["n"] for e in drills.Journal.open(tmp_path / "run").pending_undos()] == [
        0
    ]
