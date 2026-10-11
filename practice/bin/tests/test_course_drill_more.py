"""Drill framework additions (DESIGN 5.10): git-branch through the scratch-copy
builder (seeded PRs), contract-bump, the durable and tenant injectors, the
scripted responder (`ol drill run --respond`), and the SLO profile check."""

import json
import os
import subprocess
import time

from conftest import git


def _learner(ss):
    ss.add_extras("drills")
    ss.init()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    ss.commit_learner("feat: scale")


def test_seeded_pr_branch_from_a_scratch_copy(ss):
    _learner(ss)
    mine = (ss.learner / "python/tinyllm/demo/scale.py").read_text()
    out = ss("drill", "start", "seeded-pr", rc=0).out  # no cluster needed
    assert "REVIEW: a teammate opened drill/seeded-pr" in out
    log = git(
        ss.learner, "log", "--format=%an|%s", "drill/seeded-pr", "-4"
    ).splitlines()
    assert log[:3] == [
        "ol drill|chore: bump the version file",
        "ol drill|docs: note the in-place scale",
        "ol drill|perf: scale in place to save an allocation",
    ]
    seeded = git(ss.learner, "show", "drill/seeded-pr:python/tinyllm/demo/scale.py")
    assert "xs[i] = k * xs[i]" in seeded and "SOLUTION" not in seeded
    assert (
        ss.learner / "python/tinyllm/demo/scale.py"
    ).read_text() == mine  # your tree is untouched
    assert (
        git(ss.learner, "rev-parse", "--abbrev-ref", "HEAD").strip()
        != "drill/seeded-pr"
    )
    shas = git(ss.learner, "rev-list", "drill/seeded-pr", "-3").split()
    ss("drill", "end", rc=1)  # no review document yet
    ss("drill", "reset", rc=0)
    assert (
        subprocess.run(
            ["git", "rev-parse", "--verify", "drill/seeded-pr"],
            cwd=ss.learner,
            capture_output=True,
        ).returncode
        != 0
    )
    ss("drill", "start", "seeded-pr", rc=0)
    assert (
        git(ss.learner, "rev-list", "drill/seeded-pr", "-3").split() == shas
    )  # the same series, byte for byte
    d = ss.learner / "docs/reviews"
    d.mkdir(parents=True)
    (d / "seeded-pr.md").write_text(
        "# Review\n\n## Finding\n\nin place\n\n## Evidence\n\na test\n"
    )
    assert "PASS drill ops.94" in ss("drill", "end", rc=0).out
    ss("drill", "reset", rc=0)


def test_contract_bump_branch(ss):
    _learner(ss)
    (ss.course / "contracts/openapi/openai-subset.v2.yaml").write_text(
        "openapi: 3.1.0\npaths: {}\n"
    )
    v2 = ss.commit_site("contracts: api v2")
    git(ss.site, "tag", "api/v2")
    ss("drill", "start", "api-bump", rc=0)
    version = git(ss.learner, "show", "drill/api-bump:contracts/VERSION")
    assert f'sha = "{v2}"' in version
    assert "openai-subset.v2.yaml" in git(
        ss.learner, "show", "--stat", "--format=%s", "drill/api-bump"
    )
    assert (
        git(ss.learner, "log", "-1", "--format=%s", "drill/api-bump").strip()
        == "chore(contracts): sync to api/v2"
    )
    assert not (ss.learner / "contracts/openapi/openai-subset.v2.yaml").exists()
    ss("drill", "reset", rc=0)


def test_durable_and_tenant_injectors(ss, fake_http):
    _learner(ss)
    ss.use_fakes()
    dbg = fake_http({"/debug/clock": lambda q: (200, {"ok": True})})
    st = ss.kube()
    st["namespaces"]["forge"]["deployments"]["forge-durable"] = {
        "kind": "Deployment",
        "metadata": {"name": "forge-durable"},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "durable"}},
            "template": {"spec": {"containers": [{"name": "durable", "env": []}]}},
        },
    }
    (ss.tmp / "kube.json").write_text(json.dumps(st))
    ss.install_system()
    sysf = ss.learner / "system.toml"
    sysf.write_text(
        sysf.read_text()
        .replace("[deploy]\n", f'[deploy]\ndurable_debug_url = "{dbg.url}"\n')
        .replace(
            'durable = "statefulset/forge-durable"', 'durable = "deploy/forge-durable"'
        )
        .replace(
            "[services.engine]",
            'loadgen = ["python3", "-c", "import time; time.sleep(60)"]\nctl = ["python3", "-c", "print(\'burst ok\')"]\n\n[services.engine]',
        )
    )
    ss.commit_learner("feat: durable entries")
    ss("drill", "start", "durable-faults", rc=0)
    dep = ss.kube()["namespaces"]["forge"]["deployments"]
    env = {
        e["name"]: e["value"]
        for e in dep["forge-durable"]["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    }
    assert env == {"TL_DURABLE__WAL_MAX_BYTES": "4096"}
    gw = {
        e["name"]: e["value"]
        for e in dep["forge-gateway"]["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    }
    assert (
        gw["TL_FAILPOINTS"] == "dur/activity/shard-3=panic" and gw["TL_LOG"] == "info"
    )
    assert dbg.calls[-1] == ("/debug/clock", {"skew_s": 7200.0})
    run = sorted(p for p in (ss.learner / ".ol/drills").iterdir() if p.is_dir())[-1]
    journal = [json.loads(x) for x in (run / "journal.jsonl").read_text().splitlines()]
    assert "burst `" in journal[0]["did"] and "exited 0" in journal[0]["did"]
    pid = int(journal[-1]["undo"][0][1])
    os.kill(pid, 0)  # the noisy tenant is flooding
    ss("drill", "end", rc=None)
    ss("drill", "reset", rc=0)
    dep = ss.kube()["namespaces"]["forge"]["deployments"]
    assert (
        dep["forge-durable"]["spec"]["template"]["spec"]["containers"][0]["env"] == []
    )
    assert [
        e["name"]
        for e in dep["forge-gateway"]["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    ] == ["TL_LOG"]
    assert dbg.calls[-1] == ("/debug/clock", {"skew_s": 0})
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break
    else:
        raise AssertionError("reset did not stop the flood")


def test_scripted_responder_and_slo_profile(ss, fake_http):
    _learner(ss)
    ss.use_fakes()
    fired = {"t": None}

    def query(q):
        return 200, {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {}, "value": [time.time(), "1"]}],
            },
        }

    def query_range(q):
        start, end, step = float(q["start"]), float(q["end"]), float(q["step"])
        vals = (
            [[start + 1, "1"]]
            if "ALERTS" in q["query"]
            else [[end - 2 * step + i * step, "1"] for i in range(3)]
        )
        return 200, {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {}, "values": vals}],
            },
        }

    prom = fake_http({"/api/v1/query": query, "/api/v1/query_range": query_range})
    ss.install_system({"prometheus": prom.url})
    rules = ss.learner / "deploy/observability/slo-rules.yaml"
    good = rules.read_text()
    rules.write_text(good.replace("[25s]", "[30s]"))
    out = ss("drill", "start", "respond-demo", rc=1).out
    assert "drill` SLO profile" in out and "'25s'" in out
    rules.write_text(good)
    out = ss(
        "drill",
        "run",
        "respond-demo",
        "--respond",
        "--poll-s",
        "0.1",
        rc=0,
        timeout=120,
    ).out
    assert "TTFTBudgetBurnFast is firing" in out and "respond.sh exited 0" in out
    assert (
        "not graded (scripted responder)" in out
        and "slo profile" in out
        and "PASS drill ops.97" in out
    )
    assert (
        ss.kube()["namespaces"]["forge"]["deployments"]["forge-engine-prefill"]["spec"][
            "replicas"
        ]
        == 1
    )
    v = ss.verdicts("ops.97")[-1]
    assert (
        v["result"] == "pass" and v["responder"] == "scripted" and v["detected"] is True
    )
    ss("drill", "reset", rc=0)
