"""ol drill: the safety gate, seeded injections into a fake kind cluster, the
undo journal replayed in reverse, and grading at `end` (TTD from ALERTS,
resolution held over the window, trace evidence, postmortem sections)."""

import json
import os
import random
import time

DRILL = "kill-demo"


def _prom(state):
    def query_range(q):
        start, end, step = float(q["start"]), float(q["end"]), float(q["step"])
        if "ALERTS" in q["query"]:
            vals = [[start + 45, "1"]] if state["alert"] else []
            return 200, {
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [{"metric": {}, "values": vals}] if vals else [],
                },
            }
        n = int(state["held_s"] // step) + 1
        vals = (
            [[end - (n - 1 - i) * step, "0.2"] for i in range(n)]
            if state["held_s"] >= 0
            else []
        )
        return 200, {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {}, "values": vals}] if vals else [],
            },
        }

    return {"/api/v1/query_range": query_range}


TRACE = {
    "data": [
        {
            "traceID": "t1",
            "spans": [
                {
                    "spanID": "1",
                    "operationName": "POST /v1/chat/completions",
                    "processID": "a",
                    "references": [],
                },
                {
                    "spanID": "2",
                    "operationName": "engine.decode",
                    "processID": "b",
                    "references": [{"refType": "CHILD_OF", "spanID": "1"}],
                },
            ],
            "processes": {
                "a": {"serviceName": "forge-gateway"},
                "b": {"serviceName": "forge-engine"},
            },
        }
    ]
}


def _setup(ss, fake_http, prom_state=None):
    ss.init()
    ss.use_fakes()
    prom = fake_http(_prom(prom_state or {"alert": True, "held_s": 120}))
    jaeger = fake_http({"/api/traces": lambda q: (200, TRACE)})
    ss.install_system({"prometheus": prom.url, "traces": jaeger.url})
    return prom


def _postmortem(
    ss,
    sections=(
        "Summary",
        "Impact",
        "Timeline",
        "Root cause",
        "Detection",
        "Resolution",
        "Action items",
    ),
):
    d = ss.learner / "docs/postmortems"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{time.strftime('%Y-%m-%d', time.gmtime())}-kill-demo.md").write_text(
        "# Postmortem\n\n" + "".join(f"## {s}\n\ntext\n\n" for s in sections)
    )


def test_safety_gate_refuses(ss, fake_http):
    _setup(ss, fake_http)
    st = ss.kube()
    st["current_context"] = "minikube"
    (ss.tmp / "kube.json").write_text(json.dumps(st))
    out = ss("drill", "start", DRILL, rc=1).out
    assert "refusing" in out and "current context is 'minikube'" in out
    # A non-local context in [deploy] is refused even when it is current.
    sysf = ss.learner / "system.toml"
    sysf.write_text(
        sysf.read_text().replace(
            'kube_context = "kind-forge"', 'kube_context = "minikube"'
        )
    )
    assert "not a local kind- or k3d- cluster" in ss("drill", "start", DRILL, rc=1).out
    sysf.write_text(
        sysf.read_text()
        .replace('kube_context = "minikube"', 'kube_context = "kind-forge"')
        .replace('namespace    = "forge"', 'namespace    = "prod"')
    )
    st["current_context"] = "kind-forge"
    (ss.tmp / "kube.json").write_text(json.dumps(st))
    assert "namespace 'prod' does not exist" in ss("drill", "start", DRILL, rc=1).out
    # Nothing was touched by any refused run.
    assert not any(c[0] != "config" and "get" not in c for c in ss.kube_log())


def test_start_status_end_reset(ss, fake_http):
    _setup(ss, fake_http)
    before = ss.kube()["namespaces"]["forge"]
    out = ss("drill", "start", DRILL, "--seed", "7", rc=0).out
    assert (
        "PAGE: forge TTFTBudgetBurnFast is firing" in out
        and "seed 7" not in out
        and "decode-" not in out
    )
    after = ss.kube()["namespaces"]["forge"]
    killed = ss.kube()["deleted"]
    assert killed == [
        random.Random(7).choice(["decode-a", "decode-b"])
    ]  # seeded, and never the unready decode-c
    assert after["deployments"]["forge-engine-prefill"]["spec"]["replicas"] == 0
    gw = after["deployments"]["forge-gateway"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert {"name": "TL_GATEWAY__ROUTE_POLICY", "value": "weighted"} in gw["env"]
    dec = after["deployments"]["forge-engine-decode"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert dec["resources"]["limits"]["memory"] == "64Mi"
    assert (
        after["configmaps"]["forge-runtime"]["data"]["route_policy"]
        == "least_outstanding"
    )
    # Every kubectl call stayed in the gated context and namespace.
    for call in ss.kube_log():
        if call[:1] != ["config"]:
            assert call[:2] == ["--context", "kind-forge"], call
            assert call[2:4] == ["-n", "forge"] or call[2:3] == ["get"], call

    status = ss("drill", "status", rc=0).out
    assert "min elapsed" in status and "seed" not in status
    assert "still running" in ss("drill", "start", DRILL, rc=5).out

    out = ss("drill", "end", rc=1).out  # no postmortem yet
    assert "ok   detected" in out and "fired after 45s" in out
    assert "ok   resolve 1" in out and "ok   trace evidence" in out
    assert "FAIL postmortem" in out and "seed 7" in out
    assert "FAIL runbook" in out and "no runbook at docs/runbooks/kill-demo.md" in out
    v = [
        e
        for e in map(
            json.loads, (ss.learner / ".ol/verdicts.jsonl").read_text().splitlines()
        )
        if e.get("id") == "ops.90"
    ]
    assert v[-1]["result"] == "fail" and v[-1]["seed"] == 7 and v[-1]["ttd_s"] == 45.0

    out = ss("drill", "reset", rc=0).out
    undone = [x.strip() for x in out.splitlines() if x.strip().startswith("undid")]
    assert [u.split()[1] for u in undone] == [
        "config-drift",
        "resource-limit",
        "deploy-patch",
        "scale-zero",
    ]
    restored = ss.kube()["namespaces"]["forge"]
    for name in ("forge-gateway", "forge-engine-prefill", "forge-engine-decode"):
        assert (
            restored["deployments"][name]["spec"] == before["deployments"][name]["spec"]
        ), name
    assert restored["configmaps"]["forge-runtime"]["data"] == {
        "route_policy": "affinity"
    }
    assert "nothing to undo" in ss("drill", "reset", rc=0).out

    # A second run with a postmortem passes.
    ss("drill", "start", DRILL, "--seed", "3", rc=0)
    _postmortem(ss)
    rb = ss.learner / "docs/runbooks"
    rb.mkdir(parents=True, exist_ok=True)
    (rb / "kill-demo.md").write_text(
        "# Runbook\n\n## Symptoms\n\nx\n\n## Diagnosis\n\nx\n\n## Mitigation\n\nx\n"
    )
    out = ss("drill", "end", rc=0).out
    assert "PASS" in out and "TTD 45.0s" in out and "ok   runbook" in out
    ss("drill", "reset", rc=0)


def test_end_fails_without_detection_or_hold(ss, fake_http):
    _setup(ss, fake_http, {"alert": False, "held_s": 30})
    ss("drill", "start", DRILL, "--seed", "1", rc=0)
    _postmortem(ss, ("Summary", "Impact"))
    out = ss("drill", "end", rc=1).out
    assert "FAIL detected" in out and "never fired" in out
    assert "FAIL resolve 1" in out and "held 30s of 60s" in out
    assert "lacks section(s): Timeline" in out
    ss("drill", "reset", rc=0)


def test_loadgen_netem_and_aborted_start(ss, fake_http):
    _setup(ss, fake_http)
    d = ss.course / "drills/flood"
    d.mkdir()
    (d / "drill.toml").write_text(
        'id = "ops.91"\ntitle = "flood"\n[loadgen]\nargv = []\n'
        '[[inject]]\nkind = "scale-zero"\ntarget = "{deploy.services.prefill}"\nat = "loadgen+0s"\n'
    )
    n = ss.course / "drills/net"
    n.mkdir()
    (n / "drill.toml").write_text(
        'id = "ops.92"\n[[inject]]\nkind = "netem"\ntarget = "{deploy.services.decode}"\n'
    )
    ss.commit_site("two more drills")
    ss("contracts", "sync", rc=0)
    sysf = ss.learner / "system.toml"
    sysf.write_text(
        sysf.read_text().replace(
            "[services.engine]",
            'loadgen = ["python3", "-c", "import time; time.sleep(60)"]\n\n[services.engine]',
        )
    )
    ss.commit_learner("feat: loadgen entry")
    assert "flood" in ss("drill", "list", rc=0).out
    ss("drill", "start", "flood", rc=0)
    runs = sorted(p for p in (ss.learner / ".ol/drills").iterdir() if p.is_dir())
    journal = [
        json.loads(x) for x in (runs[-1] / "journal.jsonl").read_text().splitlines()
    ]
    pid = int(journal[0]["undo"][0][1])
    os.kill(pid, 0)  # loadgen is running
    ss("drill", "end", rc=0)
    ss("drill", "reset", rc=0)
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("reset did not stop the loadgen")
    # netem: tc in an ephemeral debug container on a ready pod; reset deletes the qdisc.
    ss("drill", "start", "ops.92", "--seed", "3", rc=0)
    dbg = ss.kube()["debug"]
    assert (
        dbg[-1][-10:]
        == [
            "tc",
            "qdisc",
            "add",
            "dev",
            "eth0",
            "root",
            "netem",
            "delay",
            "300ms",
            "loss",
            "5%",
        ][-10:]
    )
    assert "--profile=netadmin" in dbg[-1]
    ss("drill", "end", rc=None)
    ss("drill", "reset", rc=0)
    assert (
        ss.kube()["debug"][-1][-5:]
        == ["tc", "qdisc", "del", "dev", "eth0", "root"][-5:]
    )
    # A start that fails half way undoes what it did and leaves no active run.
    b = ss.course / "drills/broken"
    b.mkdir()
    (b / "drill.toml").write_text(
        'id = "ops.93"\n[[inject]]\nkind = "scale-zero"\ntarget = "{deploy.services.prefill}"\n'
        '[[inject]]\nkind = "clock-skew"\nurl = "http://127.0.0.1:9/debug/clock"\n'
    )
    ss.commit_site("a broken drill")
    ss("contracts", "sync", rc=0)
    out = ss("drill", "start", "broken", rc=5).out
    assert "clock-skew: POST" in out and "undid scale-zero" in out
    ss("drill", "start", "flood", rc=0)  # an aborted start left no active run behind
    ss("drill", "reset", rc=0)
