"""Course perf budgets (DESIGN 5.11): ol bench --calibrate (host and the
in-cluster Job through the safety gate), ol bench <ID>|course [--assert] in
four languages, and the `perf` milestone matcher."""

import json
from pathlib import Path


def test_calibrate_then_bench_four_languages(ss):
    ss.add_extras("bench")
    out = ss("bench", "rt.91", rc=5).out
    assert "run `ol bench --calibrate` first" in out
    out = ss("bench", "--calibrate", rc=0).out
    assert "calibrated" in out and "GFLOP/s" in out
    cal = json.loads((Path(ss.env["OL_CACHE"]) / "calibration.json").read_text())
    assert (
        cal["metrics"]["matmul_gflops"] > 0
        and cal["metrics"]["decode_tok_s"] > 0
        and len(cal["runs"]) == 3
    )
    out = ss("bench", "course", "--assert", rc=0, timeout=900).out
    for mid, metric in (
        ("rt.91", "gelem_per_s"),
        ("M90.1", "calls_per_s"),
        ("dur.90", "ns_per_op"),
        ("ds.90", "melem_per_s"),
    ):
        assert f"ok   {mid}" in out and metric in out, out
    # A budget the machine cannot meet: reported, and the verdict under --assert.
    toml = ss.course / "modules/rt.91.toml"
    toml.write_text(
        toml.read_text().replace(">= 0.0001 * matmul_gflops", ">= 1000 * matmul_gflops")
    )
    out = ss("bench", "rt.91", rc=0).out
    assert (
        "SLOW rt.91" in out and "(1000 x matmul_gflops)" in out and "report only" in out
    )
    assert "1 budget(s) missed" in ss("bench", "rt.91", "--assert", rc=1).out


def test_learner_bench_runs_your_units(ss):
    ss.add_extras("bench")
    ss("bench", "--calibrate", rc=0)
    ss.init()
    for mid, unit in (
        ("rt.90", "c/src/runtime/abi.c"),
        ("rt.91", "c/src/runtime/demo.c"),
    ):
        ss("start", mid, rc=0)
        ss.implement(unit)
        ss("check", mid, rc=0)
    out = ss("bench", "rt.91", "--assert", rc=0).out
    assert "ok   rt.91" in out and "yours" in out
    events = [
        json.loads(x)
        for x in (ss.learner / ".ol/verdicts.jsonl").read_text().splitlines()
    ]
    assert any(
        e.get("event") == "bench" and e["id"] == "rt.91" and e["ok"] for e in events
    )


def test_perf_matcher_and_in_cluster_calibration(ss):
    ss.add_extras("bench")
    ss.init()
    ss.install_system()
    (ss.learner / "decode.py").write_text("print('{\"tokens_per_s\": 1e9}')\n")
    st = ss.learner / "system.toml"
    st.write_text(
        st.read_text().replace("[entry]\n", '[entry]\nctl = ["python3", "decode.py"]\n')
    )
    ss.commit_learner("feat: a fast decode loop")
    out = ss("milestone", "MS-B90", rc=5).out
    assert "run `ol bench --calibrate` first" in out
    ss("bench", "--calibrate", rc=0)
    out = ss("milestone", "MS-B90", rc=0).out
    assert "tokens_per_s: 1e+09 >=" in out
    # in-cluster: a Job in [deploy].namespace through the drill safety gate
    ss.use_fakes()
    st_ = ss.kube()
    st_["job_logs"] = (
        '{"matmul_gflops": 3.5, "decode_tok_s": 120.0}\n{"matmul_gflops": 4.0, "decode_tok_s": 110.0}\n'
    )
    (ss.tmp / "kube.json").write_text(json.dumps(st_))
    out = ss("bench", "--calibrate", "--in-cluster", rc=0).out
    assert "in kind-forge/forge: matmul 4.00 GFLOP/s, decode 120 steps/s" in out
    k = ss.kube()
    kinds = [x["kind"] for x in k["applied"]]
    assert kinds == ["ConfigMap", "Job"] and "job/ol-calibrate" in k["deleted_objects"]
    cal = json.loads(
        (Path(ss.env["OL_CACHE"]) / "calibration.in-cluster.json").read_text()
    )
    assert cal["metrics"] == {"matmul_gflops": 4.0, "decode_tok_s": 120.0}
    st_ = ss.kube()
    st_["current_context"] = "minikube"
    (ss.tmp / "kube.json").write_text(json.dumps(st_))
    assert "refusing" in ss("bench", "--calibrate", "--in-cluster", rc=1).out
