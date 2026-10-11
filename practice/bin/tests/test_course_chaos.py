"""Milestone `chaos` and `sink` step keys, the `effects` matcher, and the
`{json:<step>.<key>}` placeholder; drill `[workload]` and `[[resolve.run]]`
(DESIGN 5.7, 5.10). The services are small Python fakes written into the
learner repo: an appender that posts keyed effects to $TL_TEST_EFFECTS_URL and
resumes from a state file after a SIGKILL, and a ctl that waits for it."""

import json
import os
import time

APPENDER = r"""
import json, os, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
data, n = a["--data"], int(a["--n"])
state = os.path.join(data, "done")
if a.get("--once") == "1" and os.path.exists(os.path.join(data, "started")):
    sys.exit("refusing to start twice")
open(os.path.join(data, "started"), "a").close()

class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

srv = ThreadingHTTPServer(("127.0.0.1", int(a["--health"])), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
done = open(state).read().split() if os.path.exists(state) else []
start = len(done) + (2 if a.get("--skip") == "1" and done else 0)
url = os.environ["TL_TEST_EFFECTS_URL"]
for i in range(start, n):
    body = json.dumps({"key": f"job/{i}", "value": str(i)}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5).read()
    time.sleep(0.03)  # a kill here redelivers job/i after the restart
    with open(state, "a") as f:
        f.write(f"{i}\n")
while True:
    time.sleep(1)
"""

CTL = r"""
import json, os, sys, time
verb = sys.argv[1]
if verb == "wait":
    state, n = sys.argv[2], int(sys.argv[3])
    while not os.path.exists(state) or str(n - 1) not in open(state).read().split():
        time.sleep(0.05)
    print(json.dumps({"n": n, "sink": os.environ.get("TL_TEST_EFFECTS_URL", "")}))
elif verb == "echo":
    print(json.dumps({"n": int(sys.argv[2])}))
elif verb == "work":
    with open(sys.argv[2], "w") as f:
        f.write(os.environ.get("WL_TAG", ""))
    time.sleep(120)
elif verb == "check":
    print("checking")
    print(json.dumps({"ok": 0 if os.environ.get("FAIL") else 1}))
"""

MS = """id = "MS-chaos"
title = "kills"
requires = []
pass = 0
services = ["appender"]

[[step]]
name = "appends-under-kills"
sink = "effects"
run = "ctl"
argv = ["wait", "{appender.data}/done", "40"]
chaos = { kill = ["appender"], every_s = "0.1..0.25", count = 3, seed = 1 }
expect = { match = "effects", keys = "0..39", prefix = "job/", exactly_once = true }
timeout_s = 60

[[step]]
name = "same n"
run = "ctl"
argv = ["echo", "40"]
expect = { match = "json-last-line", metrics = { n = "== {json:appends-under-kills.n}" } }
"""


def _chaos_learner(ss, appender_args="", ms=MS):
    (ss.course / "milestones/MS-chaos.toml").write_text(ms)
    ss.commit_site("MS-chaos")
    ss.init()
    ss.install_system()
    (ss.learner / "bin").mkdir(exist_ok=True)
    (ss.learner / "bin/appender.py").write_text(APPENDER)
    (ss.learner / "bin/ctl.py").write_text(CTL)
    sysf = ss.learner / "system.toml"
    sysf.write_text(
        sysf.read_text().replace(
            "[services.engine]",
            'appender = ["python3", "bin/appender.py"]\nctl = ["python3", "bin/ctl.py"]\n\n'
            "[services.appender]\n"
            'section = "worker"\n'
            'health = "http://127.0.0.1:{health_port}/healthz"\n'
            'args = ["--health", "{health_port}", "--data", "{data}", "--n", "40"'
            + appender_args
            + "]\nready_timeout_s = 10\n\n[services.engine]",
        )
    )
    ss.commit_learner("feat: appender")


def _last_run(ss, mid="MS-chaos"):
    return sorted((ss.learner / ".ss/milestones" / mid).iterdir())[-1]


def test_chaos_kills_restart_and_effects_exactly_once(ss):
    _chaos_learner(ss)
    out = ss("milestone", "MS-chaos", rc=0).out
    assert "effects sink on http://127.0.0.1:" in out
    assert "chaos: 3 of 3 SIGKILL(s) (appender x3)" in out
    assert "40 of 40 key(s) applied under 'job/'" in out
    run = _last_run(ss)
    events = [
        json.loads(x)
        for x in (run / "out/appends-under-kills/chaos.jsonl").read_text().splitlines()
    ]
    assert [e["service"] for e in events] == ["appender"] * 3
    assert all(e["restarted"] for e in events)
    assert len({e["pid"] for e in events}) == 3  # a new process each time
    log = (run / "services/appender.log").read_text()
    assert log.count("--- SIGKILL; restarting") == 3
    recs = json.loads((run / "out/appends-under-kills/effects.json").read_text())
    assert sorted(int(r["key"].split("/")[1]) for r in recs) == list(range(40))
    # The step's command saw the sink URL too, and {json:...} read its last line.
    last = json.loads(
        (run / "out/appends-under-kills/stdout.txt").read_text().splitlines()[-1]
    )
    assert last["sink"].startswith("http://127.0.0.1:")
    assert "n: 40 == 40" in out
    # The schedule is seeded: a rerun kills at the same offsets.
    ss("milestone", "MS-chaos", rc=0)
    again = [
        json.loads(x)
        for x in (_last_run(ss) / "out/appends-under-kills/chaos.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [e["wait_s"] for e in again] == [e["wait_s"] for e in events]


def test_effects_matcher_reports_a_lost_effect(ss):
    # A buggy resume skips the next index after every restart.
    _chaos_learner(ss, ', "--skip", "1"')
    out = ss("milestone", "MS-chaos", "--step", "appends", rc=1).out
    assert "missing job/" in out and "problem(s)" in out


def test_a_service_that_does_not_come_back_fails_the_step(ss):
    _chaos_learner(ss, ', "--once", "1"')
    out = ss("milestone", "MS-chaos", "--step", "appends", rc=1).out
    assert "after SIGKILL 1" in out and "did not become healthy again" in out
    assert "refusing to start twice" in out


def test_spec_errors(ss):
    bad = {
        "partition": MS.replace(
            'kill = ["appender"]', 'kill = ["appender"], partition = ["appender"]'
        ),
        "leader:<glob>": MS.replace(
            'kill = ["appender"]', 'kill = ["leader:durable-*"]'
        ),
        'needs sink = "effects"': MS.replace('sink = "effects"\n', ""),
        "every_s": MS.replace('"0.1..0.25"', '"3..1"'),
        "history": MS.replace(
            'sink = "effects"\n', 'sink = "effects"\nhistory = "linearizability"\n'
        ),
        "do not start": MS.replace('kill = ["appender"]', 'kill = ["durable"]'),
    }
    _chaos_learner(ss)
    for want, text in bad.items():
        (ss.course / "milestones/MS-chaos.toml").write_text(text)
        ss.commit_site(f"bad: {want}")
        ss("contracts", "sync", rc=0)
        out = ss("milestone", "MS-chaos", "--step", "appends", rc=None).out
        assert want in out, (want, out)


DRILL = """id = "ops.94"
title = "a workload under a fault"

[workload]
role = "ctl"
argv = ["work", "{marker}"]
env = { WL_TAG = "tagged" }
after_inject = AFTER
wait_rollout = "{deploy.services.gateway}"

[[inject]]
kind = "scale-zero"
target = "{deploy.services.prefill}"

[[resolve.run]]
role = "ctl"
argv = ["check"]
expect = { match = "json-last-line", metrics = { ok = "== 1" } }

[[resolve.run]]
role = "ctl"
argv = ["check"]
env = { FAIL = "1" }
expect = { match = "json-last-line", metrics = { ok = "== 1" } }
"""


def _drill_learner(ss, after: bool):
    marker = ss.tmp / "workload.txt"
    d = ss.course / "drills/wl"
    d.mkdir()
    (d / "drill.toml").write_text(
        DRILL.replace("AFTER", "true" if after else "false").replace(
            "{marker}", str(marker)
        )
    )
    ss.commit_site("drill wl")
    ss.init()
    ss.use_fakes()
    ss.install_system()
    (ss.learner / "bin").mkdir(exist_ok=True)
    (ss.learner / "bin/ctl.py").write_text(CTL)
    sysf = ss.learner / "system.toml"
    sysf.write_text(
        sysf.read_text().replace(
            "[services.engine]", 'ctl = ["python3", "bin/ctl.py"]\n\n[services.engine]'
        )
    )
    ss.commit_learner("feat: ctl")
    return marker


def _journal(ss):
    runs = sorted(p for p in (ss.learner / ".ss/drills").iterdir() if p.is_dir())
    return [
        json.loads(x) for x in (runs[-1] / "journal.jsonl").read_text().splitlines()
    ]


def _gone(pid: int) -> bool:
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_drill_workload_after_inject_and_resolve_runs(ss):
    marker = _drill_learner(ss, after=True)
    ss("drill", "start", "wl", rc=0)
    j = _journal(ss)
    assert [e["kind"] for e in j] == ["scale-zero", "workload"]
    pid = int(j[1]["undo"][0][1])
    os.kill(pid, 0)  # the workload runs in the background
    deadline = time.time() + 5
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert marker.read_text() == "tagged"
    cmds = [c for c in ss.kube_log() if "rollout" in c]
    assert any("status" in c and "deploy/forge-gateway" in c for c in cmds)
    log = [
        c for c in ss.kube_log() if "scale" in c or ("rollout" in c and "status" in c)
    ]
    assert "scale" in log[0]  # the rollout wait came after the injection
    out = ss("drill", "end", rc=1).out
    assert "resolve run 1" in out and "resolve run 2" in out
    lines = {
        x.split()[1] + " " + x.split()[2] + " " + x.split()[3]: x
        for x in out.splitlines()
        if "resolve run" in x
    }
    assert "ok" in lines["resolve run 1"] and "FAIL" in lines["resolve run 2"]
    assert "ok: 0 == 1" in out
    v = ss.verdicts("ops.94")[-1]
    assert v["result"] == "fail" and v["resolved"] is False
    ss("drill", "reset", rc=0)
    assert _gone(pid), "reset did not stop the workload"


def test_drill_workload_before_inject_and_failed_rollout_aborts(ss):
    _drill_learner(ss, after=False)
    ss("drill", "start", "wl", rc=0)
    j = _journal(ss)
    assert [e["kind"] for e in j] == ["workload", "scale-zero"]
    ss("drill", "reset", rc=0)
    assert _gone(int(j[0]["undo"][0][1]))
    # A rollout that never completes aborts the start before any injection.
    st = ss.kube()
    st["rollout_failing"] = ["deploy/forge-gateway"]
    (ss.tmp / "kube.json").write_text(json.dumps(st))
    n_scale = sum("scale" in c for c in ss.kube_log())
    out = ss("drill", "start", "wl", rc=None).out
    assert "rollout status deploy/forge-gateway exited 1" in out
    assert "aborted" in ss("drill", "status", rc=0).out
    assert sum("scale" in c for c in ss.kube_log()) == n_scale
