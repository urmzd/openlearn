"""ol milestone: [build], services on allocated ports with generated
runtime.toml files, matchers (tokens-equal with the near-tie rule, contains,
sse, suite, git-log, ci-status, trace), smoke and kind tags, pass-gate
composition, blocked and assisted verdicts."""

import json
import tomllib


def test_http_step_json_can_supply_a_later_command_environment(ss, fake_http):
    ss.init()
    ss.install_system()
    api = fake_http({"/load-key": lambda body: (201, {"key": "isolated-load-key"})})
    st = ss.learner / "system.toml"
    st.write_text(
        st.read_text().replace(
            "[entry]\n", '[entry]\nprobe = ["python3", "probe.py"]\n'
        )
    )
    (ss.learner / "probe.py").write_text(
        'import json, os\nprint(json.dumps({"key": os.environ["LOAD_KEY"]}))\n'
    )
    (ss.course / "milestones/MS-HTTP90.toml").write_text(
        """
id = "MS-HTTP90"
title = "HTTP response data reaches a later command"
requires = []
pass = 1
ci = "pr"
[[step]]
name = "mint-key"
http = { method = "POST", url = "FAKE_URL/load-key", json = { scopes = ["infer"] } }
expect = { match = "json-schema", status = 201, schema = { type = "object", required = ["key"] } }
[[step]]
name = "load"
run = "probe"
env = { LOAD_KEY = "{json:mint-key.key}" }
expect = { match = "contains", text = "isolated-load-key" }
""".replace("FAKE_URL", api.url)
    )
    out = ss("milestone", "MS-HTTP90", rc=0).out
    assert "PASS" in out
    assert api.calls == [("/load-key", {"scopes": ["infer"]})]


def _pass_m901(ss):
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    ss("check", "M90.1", rc=0)


def test_blocked_then_assisted_then_pass(ss):
    ss.init()
    ss.install_system()
    out = ss("milestone", "MS-M90", rc=3).out
    assert "BLOCKED" in out and "M90.1" in out
    out = ss("milestone", "MS-M90", "--ref-deps", rc=0).out
    assert (
        "assisted" in out
        and "greedy ids[prompt=ab]" in out
        and "greedy ids[prompt=hello]" in out
    )
    v = [
        e
        for e in map(
            json.loads, (ss.learner / ".ol/verdicts.jsonl").read_text().splitlines()
        )
        if e.get("id") == "MS-M90"
    ]
    assert v[-1]["result"] == "pass" and v[-1]["assisted"] is True
    _pass_m901(ss)
    out = ss("milestone", "MS-M90", rc=0).out
    assert "PASS" in out and "(assisted)" not in out
    logs = sorted((ss.learner / ".ol/milestones/MS-M90").iterdir())
    summary = json.loads((logs[-1] / "summary.json").read_text())
    assert summary["result"] == "pass" and len(summary["steps"]) == 3
    assert "build ok" in (logs[-1] / "build.log").read_text()
    assert "MS-M90" in ss("milestone", "list", rc=0).out


def test_tokens_equal_divergence_and_near_tie(ss):
    ss.init()
    ss.install_system()
    _pass_m901(ss)
    cli = ss.learner / "python/tinyllm/__main__.py"
    good = cli.read_text()
    # A real bug: the learner's generate drops one token's worth of state.
    cli.write_text(
        good.replace(
            "ids = bigram.generate(a.prompt, a.max_tokens",
            "ids = bigram.generate(a.prompt + 'x', a.max_tokens",
        )
    )
    out = ss("milestone", "MS-M90", "--step", "greedy", rc=1).out
    assert "first divergence at step 0" in out and "real divergence" in out
    # A near tie: same divergence, but the learner's logits show a top-2 margin under 1e-3.
    cli.write_text(
        cli.read_text().replace(
            'print(json.dumps({"logits": bigram.logits(a.prompt, prefix)}))',
            "row = bigram.logits(a.prompt, prefix); row[0] = max(row) - 1e-4; "
            'print(json.dumps({"logits": row}))',
        )
    )
    out = ss("milestone", "MS-M90", "--step", "greedy", rc=0).out
    assert "near-tie" in out and "margin" in out


def test_build_failure_aborts(ss):
    ss.init()
    ss.install_system()
    _pass_m901(ss)
    st = ss.learner / "system.toml"
    st.write_text(st.read_text().replace("print('build ok')", "raise SystemExit(3)"))
    out = ss("milestone", "MS-M90", rc=1).out
    assert "build step" in out and "exited 3" in out


def test_missing_entry_names_the_role(ss):
    ss.init()
    ss.install_system()
    _pass_m901(ss)
    st = ss.learner / "system.toml"
    st.write_text(
        st.read_text().replace(
            'tinyllm = ["python3", "python/tinyllm/__main__.py"]\n', ""
        )
    )
    assert (
        "milestone MS-M90 needs [entry].tinyllm in system.toml"
        in ss("milestone", "MS-M90", rc=5).out
    )


def test_p1_smoke_runs_services_suites_and_earlier_gates(ss):
    ss.init()
    ss.install_system()
    _pass_m901(ss)
    out = ss(
        "milestone", "MS-P1", "--smoke", rc=0, env={"TL_API_KEY": "tl_k_secret"}
    ).out
    # Own smoke steps, MS-M90's smoke step (included), and MS-P0's smoke steps (earlier gate).
    for name in (
        "engine conforms to openapi:v0",
        "gateway conforms to openapi:v0",
        "32 tokens stream through the gateway",
        "MS-M90/greedy ids[prompt=ab]",
        "MS-P0/conventional commits",
        "MS-P0/your CI is green",
    ):
        assert name in out, name
    assert (
        "kind NodePort" not in out and "one trace" not in out
    )  # kind steps never run in --smoke
    logdir = sorted((ss.learner / ".ol/milestones/MS-P1").iterdir())[-1]
    eng = tomllib.loads((logdir / "services/engine.runtime.toml").read_text())
    gw = tomllib.loads((logdir / "services/gateway.runtime.toml").read_text())
    ports = {
        eng["engine"]["http_listen"],
        eng["engine"]["health_listen"],
        gw["gateway"]["listen"],
        gw["gateway"]["health_listen"],
    }
    assert len(ports) == 4 and all(p.startswith("127.0.0.1:") for p in ports)
    # The gateway's template reached the engine through {engine.port}.
    assert gw["gateway"]["upstream"] == "http://" + eng["engine"]["http_listen"]
    # A smoke pass is recorded, but never stands in for the full milestone.
    assert "smoke pass" in ss("milestone", "list", rc=0).out


def test_p1_full_skips_kind_steps_without_a_cluster(ss):
    ss.init()
    ss.install_system()
    _pass_m901(ss)
    out = ss(
        "milestone", "MS-P1", "--step", "kind NodePort", rc=1, env={"TL_API_KEY": "k"}
    ).out
    assert "skip" in out and "kind step" in out and "INCOMPLETE" in out


def test_p1_kind_steps_against_fake_cluster(ss, fake_http):
    ss.init()
    ss.use_fakes()
    now_trace = {
        "data": [
            {
                "traceID": "abc",
                "spans": [
                    {
                        "spanID": "1",
                        "operationName": "POST /v1/completions",
                        "processID": "p1",
                        "references": [],
                    },
                    {
                        "spanID": "2",
                        "operationName": "engine.decode",
                        "processID": "p2",
                        "references": [{"refType": "CHILD_OF", "spanID": "1"}],
                    },
                ],
                "processes": {
                    "p1": {"serviceName": "forge-gateway"},
                    "p2": {"serviceName": "forge-engine"},
                },
            }
        ]
    }
    jaeger = fake_http({"/api/traces": lambda q: (200, now_trace)})
    ss.install_system({"traces": jaeger.url})
    _pass_m901(ss)
    out = ss("milestone", "MS-P1", "--step", "trace", rc=0).out
    assert "trace abc" in out
    assert jaeger.calls[0][1]["service"] == "forge-gateway"
    # The same span names without the parent edge do not count.
    now_trace["data"][0]["spans"][1]["references"] = []
    assert "no trace" in ss("milestone", "MS-P1", "--step", "trace", rc=1).out


def test_p0_git_log_and_ci_status(ss):
    ss.init()
    ss.install_system()
    ss("milestone", "MS-P0", rc=0)
    (ss.learner / "notes.txt").write_text("x")
    ss.commit_learner("wip stuff")
    assert "not Conventional Commits" in ss("milestone", "MS-P0", rc=1).out
    ss.commit_learner("docs: notes")
    st = ss.learner / "system.toml"
    st.write_text(st.read_text().replace("print('ci ok')", "raise SystemExit(1)"))
    ss.commit_learner("ci: break it")
    assert "[ci].local" in ss("milestone", "MS-P0", rc=1).out
    # With a GitHub remote, ci-status asks gh about the default branch's latest run.
    ss.use_fakes()
    from conftest import git

    git(ss.learner, "remote", "add", "origin", "git@github.com:me/forge.git")
    out = ss(
        "milestone",
        "MS-P0",
        rc=0,
        env={"FAKE_GH_JSON": '[{"conclusion": "success", "status": "completed"}]'},
    ).out
    assert "success" in out
    assert "run list --branch" in (ss.tmp / "gh.log").read_text()
    assert (
        "failure"
        in ss(
            "milestone",
            "MS-P0",
            rc=1,
            env={"FAKE_GH_JSON": '[{"conclusion": "failure"}]'},
        ).out
    )
