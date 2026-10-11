"""Unit tests for the B1 integration changes (course/DEVIATIONS.md I04 to I10,
I17): dependency-first registry order, the healthz case without a health
base, placeholders in [build].steps, system.toml schema validation, drill
document labels, and the reference system manifest itself."""

import json
import sys
import tomllib
from pathlib import Path

from olcourse import conform, drills, registry, services, system

COURSE = Path(__file__).resolve().parents[2]


def test_ordered_puts_every_module_after_its_deps():
    reg = registry.load(COURSE)
    order = [m.id for m in reg.ordered()]
    assert sorted(order) == sorted(reg.modules)
    pos = {mid: i for i, mid in enumerate(order)}
    for m in reg.modules.values():
        for d in m.deps:
            if d in pos:
                assert pos[d] < pos[m.id], f"{d} must come before {m.id}"
    # and pass order still holds
    passes = [reg.get(mid).pass_ for mid in order]
    assert passes == sorted(passes)


def test_healthz_is_pending_without_a_health_base():
    case = conform.Case("v0.healthz", "health", ["v0"], ["gateway"], "healthz")
    client = conform.Client(base="http://127.0.0.1:9", tier="gateway", spec={})
    res = conform.run(client, [case], conform.Suite("v0", "gateway"))
    assert [r.status for r in res] == ["pending"]
    assert "gateway_health_url" in res[0].detail


def test_build_steps_expand_fixtures(tmp_path):
    fx = tmp_path / "fixtures" / "MS-X"
    fx.mkdir(parents=True)
    (fx / "corpus.txt").write_text("hello")
    (tmp_path / "system.toml").write_text("")
    raw = {
        "build": {
            "steps": [
                [
                    sys.executable,
                    "-c",
                    "import sys, pathlib; pathlib.Path('out.txt').write_text(open(sys.argv[1]).read())",
                    "{fixture:MS-X/corpus.txt}",
                ]
            ]
        }
    }
    sysm = system.System(tmp_path / "system.toml", raw)
    ok, why = services.run_build(
        sysm, tmp_path / "build.log", {"TINYLLM_FIXTURES": str(tmp_path / "fixtures")}
    )
    assert ok, why
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_reference_system_toml_matches_the_schema(tmp_path):
    (tmp_path / "contracts" / "config").mkdir(parents=True)
    schema_src = COURSE / "contracts" / "config" / "system.schema.json"
    (tmp_path / "contracts" / "config" / "system.schema.json").write_text(
        schema_src.read_text()
    )
    raw = tomllib.loads((COURSE / "ref" / "system.toml").read_text())
    assert system.schema_errors(tmp_path, raw) == []
    raw["deploy"]["gateway_health_url"] = "http://127.0.0.1:30464"
    assert system.schema_errors(tmp_path, raw) == []
    raw["deploy"]["no_such_key"] = 1
    assert system.schema_errors(tmp_path, raw) != []


def test_reference_entry_roles_match_cli_roles():
    raw = tomllib.loads((COURSE / "ref" / "system.toml").read_text())
    entry = raw["entry"]
    assert {"tinyllm", "engine", "gateway"} <= set(entry)
    # The engine runs the v1 server from a runtime.toml template whose listen
    # keys the harness fills with allocated ports ({config}).
    assert entry["engine"][-2:] == ["--config", "{config}"]
    template = raw["services"]["engine"]["config"]
    assert (COURSE / "ref" / "entry" / template).is_file()
    # The gateway reaches the engine at its allocated port: in the tracer
    # flags form (--upstream) or the --config form's template.
    gw = entry["gateway"]
    if gw[-2:] == ["--config", "{config}"]:
        gw_template = COURSE / "ref" / "entry" / raw["services"]["gateway"]["config"]
        assert "http://127.0.0.1:{engine.port}" in gw_template.read_text()
    else:
        assert "http://127.0.0.1:{engine.port}" in gw
    assert raw["services"]["gateway"]["after"] == ["engine"]
    trained = [s for s in raw["build"]["steps"] if "train" in s]
    assert trained and "{fixture:MS-P1/corpus.txt}" in trained[0]


def test_drill_documents_name_themselves(tmp_path):
    path, errs = drills.postmortem_errors(
        tmp_path,
        {"path": "docs/runbooks/x.md", "sections": ["Symptoms"]},
        what="runbook",
    )
    assert path is None and errs == ["no runbook at docs/runbooks/x.md"]
    (tmp_path / "docs" / "runbooks").mkdir(parents=True)
    (tmp_path / "docs" / "runbooks" / "x.md").write_text("# x\n\n## Symptoms\n\nok\n")
    path, errs = drills.postmortem_errors(
        tmp_path,
        {"path": "docs/runbooks/x.md", "sections": ["Symptoms"]},
        what="runbook",
    )
    assert errs == [] and path.name == "x.md"


def test_engine_crashloop_drill_grades_rollout_and_runbook():
    spec = tomllib.loads(
        (COURSE / "drills" / "engine-crashloop" / "drill.toml").read_text()
    )
    checks = spec["resolve"]["check"]
    assert any("rollout" in c for c in checks) and any("suite" in c for c in checks)
    assert spec["doc"][0]["label"] == "runbook"
    assert json.dumps(spec)  # plain data, no harness objects
