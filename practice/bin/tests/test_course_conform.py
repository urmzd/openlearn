"""ol conform openapi:v0: starts the tier's service, validates every response
against the vendored contract, tiers, the gateway key, pending cases, --base."""

import json


def _ledger(ss, key):
    return [
        e
        for e in map(
            json.loads, (ss.learner / ".ol/verdicts.jsonl").read_text().splitlines()
        )
        if e.get("id") == key
    ]


def test_engine_and_gateway_tiers(ss):
    ss.init()
    ss.install_system()
    out = ss("conform", "openapi:v0", "--target", "engine", rc=0).out
    for case in (
        "v0.healthz",
        "v0.completions.schema",
        "v0.stream.framing",
        "v0.stream.equals_nonstream",
        "v0.seed",
        "v0.err.400",
        "v0.completions.length",
    ):
        assert f"pass    {case}" in out, case
    assert "v0.auth.401" not in out  # gateway-only case
    assert _ledger(ss, "conform:openapi:v0:engine")[-1]["result"] == "pass"
    # Gateway tier without a key: every authenticated case gets a 401.
    out = ss("conform", "openapi:v0:gateway", rc=1).out
    assert "fail    v0.completions.schema" in out and "HTTP 401" in out
    out = ss("conform", "openapi:v0:gateway", rc=0, env={"TL_API_KEY": "tl_a_b"}).out
    assert "pass    v0.auth.401" in out and "v0.seed" not in out
    smoke = ss(
        "conform", "openapi:v0:gateway:smoke", rc=0, env={"TL_API_KEY": "tl_a_b"}
    ).out
    assert "v0.completions.greedy_stable" not in smoke and "v0.stream.framing" in smoke


def test_contract_violations_fail_with_the_field(ss):
    ss.init()
    ss.install_system()
    eng = ss.learner / "serve/engine.py"
    good = eng.read_text()
    eng.write_text(
        good.replace('"object": "text_completion"', '"object": "completion"')
    )
    out = ss("conform", "openapi:v0", rc=1).out
    assert "$.object: expected 'text_completion'" in out
    eng.write_text(good.replace('self.wfile.write(b"data: [DONE]\\n\\n")', "pass"))
    out = ss("conform", "openapi:v0", rc=1).out
    assert "does not end with `data: [DONE]`" in out
    eng.write_text(
        good.replace(
            'return self.error(400, "temperature must be in [0, 2]", "invalid_request_error", param="temperature")',
            'return self.send_json(400, {"message": "bad"})',
        )
    )
    out = ss("conform", "openapi:v0", rc=1).out
    assert "fail    v0.err.400" in out and '400 body is not {"error"' in out


def test_pending_case_and_base_url(ss):
    ss.init()
    ss.install_system()
    case = ss.course / "conformance/openapi/cases/v0.future.toml"
    case.write_text(
        'id = "v0.future"\nversions = ["v0"]\ntiers = ["engine"]\nrequires = ["M90.2"]\ncheck = "healthz"\n'
    )
    ss.commit_site("a case that waits for M90.2")
    ss("contracts", "sync", rc=0)
    ss.commit_learner("chore: sync contracts")
    out = ss("conform", "openapi:v0", rc=0).out
    assert "pending v0.future" in out and "requires M90.2" in out
    # --base tests a server you run; nothing listens here, so the cases fail cleanly.
    out = ss("conform", "openapi:v0", "--base", "http://127.0.0.1:9", rc=1).out
    assert "ConnectionRefusedError" in out or "Connection refused" in out


def test_unknown_suite_and_missing_service(ss):
    ss.init()
    assert "unknown suite" in ss("conform", "grpc:v1", rc=5).out
    assert "needs [services.engine]" in ss("conform", "openapi:v0", rc=5).out
