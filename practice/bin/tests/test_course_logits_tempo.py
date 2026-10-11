"""The logits-close matcher (an npz against a reference npz) and the trace
matcher on the Tempo query API (search, then OTLP JSON by trace id)."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from olcourse import HarnessError, matchers


def _mc(tmp: Path, **kw) -> matchers.MatchCtx:
    run = SimpleNamespace(
        learner=tmp,
        course_file=lambda rel: Path(rel) if Path(rel).is_absolute() else tmp / rel,
        system=SimpleNamespace(deploy={}),
    )
    lk = {"out": str(tmp / "out")}
    return matchers.MatchCtx(
        run=run, lookup=lk.get, step_name="s", started_at=time.time(), **kw
    )


def _logits(tmp: Path, rows: list[np.ndarray], compressed: bool = False) -> None:
    (tmp / "out").mkdir(exist_ok=True)
    save = np.savez_compressed if compressed else np.savez
    save(
        tmp / "out" / "logits.npz",
        **{f"logits{i}": r.astype(np.float32) for i, r in enumerate(rows)},
    )


def _ref_top(tmp: Path, rows: list[np.ndarray], n: int = 8) -> str:
    tops = {}
    for i, r in enumerate(rows):
        ix = np.argsort(-r, kind="stable")[:n]
        tops[f"top{i}.ids"], tops[f"top{i}.values"] = (
            ix.astype(np.int64),
            r[ix].astype(np.float32),
        )
    np.savez(tmp / "ref.npz", **tops)
    return str(tmp / "ref.npz")


EXPECT = {"match": "logits-close", "output": "{out}/logits.npz"}


def _check(tmp: Path, ref: str, **kw) -> matchers.Verdict:
    out = matchers.StepOut(rc=0)
    return matchers.evaluate({**EXPECT, "file": ref, **kw}, out, _mc(tmp))


def test_logits_close_passes_within_atol_and_same_top_k(tmp_path):
    rng = np.random.default_rng(0)
    rows = [rng.normal(size=64) * 5 for _ in range(3)]
    ref = _ref_top(tmp_path, rows)
    _logits(tmp_path, [r + 4e-4 for r in rows], compressed=True)
    v = _check(tmp_path, ref)
    assert v.status == "pass", v.detail
    assert "3 rows" in v.detail and "top-5 ids equal" in v.detail


def test_logits_close_fails_past_atol(tmp_path):
    rows = [np.linspace(-3, 3, 32)]
    ref = _ref_top(tmp_path, rows)
    bad = rows[0].copy()
    bad[31] += 2e-3  # the largest logit, so the top-k order is unchanged
    _logits(tmp_path, [bad])
    v = _check(tmp_path, ref)
    assert v.status == "fail" and "logits0[31]" in v.detail and "> 0.001" in v.detail
    assert _check(tmp_path, ref, atol=1e-2).status == "pass"


def test_logits_close_fails_on_swapped_top_k(tmp_path):
    rows = [np.array([0.0, 1.0, 1.0005, 3.0, -1.0, 2.0])]
    ref = _ref_top(tmp_path, rows, n=6)
    got = rows[0].copy()
    got[1], got[2] = 1.0005, 1.0  # within atol, but ranks 3 and 4 swap
    _logits(tmp_path, [got])
    v = _check(tmp_path, ref)
    assert v.status == "fail" and "top-5 ids [3, 5, 1, 2, 0]" in v.detail
    assert _check(tmp_path, ref, top_k=2).status == "pass"


def test_logits_close_row_count_missing_file_and_dense_reference(tmp_path):
    rows = [np.arange(10.0), np.arange(10.0)[::-1].copy()]
    ref = _ref_top(tmp_path, rows)
    v = _check(tmp_path, ref)
    assert v.status == "fail" and "was not written" in v.detail
    _logits(tmp_path, rows[:1])
    v = _check(tmp_path, ref)
    assert v.status == "fail" and "1 logits rows, want 2" in v.detail
    # A dense reference compares every value and the vector length.
    np.savez(tmp_path / "dense.npz", logits0=rows[0].astype(np.float64))
    assert _check(tmp_path, str(tmp_path / "dense.npz")).status == "pass"
    _logits(tmp_path, [rows[0][:9]])
    v = _check(tmp_path, str(tmp_path / "dense.npz"))
    assert v.status == "fail" and "9 values, want 10" in v.detail


def test_logits_close_needs_file_and_output(tmp_path):
    with pytest.raises(HarnessError, match="needs `file` and `output`"):
        matchers.evaluate(
            {"match": "logits-close", "file": "x.npz"},
            matchers.StepOut(rc=0),
            _mc(tmp_path),
        )


def test_read_npz_matches_numpy(tmp_path):
    arrs = {
        "a": np.arange(6, dtype=np.int64).reshape(2, 3),
        "b.values": np.array([1.5, -2.25], dtype=">f8"),
        "c": np.array([7], dtype=np.uint8),
        "d": np.array([0.5, 0.25], dtype=np.float16),
    }
    np.savez(tmp_path / "x.npz", **arrs)
    got = matchers.read_npz(tmp_path / "x.npz")
    assert got == {k: v.ravel().tolist() for k, v in arrs.items()}


# -- trace on Tempo -----------------------------------------------------------


def _b64(hexid: str) -> str:
    return base64.b64encode(bytes.fromhex(hexid)).decode()


def _otlp(parent_of_engine: str) -> dict:
    """Tempo's /api/traces/<id> reply: OTLP JSON with base64 ids."""

    def rs(svc: str, spans: list[dict]) -> dict:
        return {
            "resource": {
                "attributes": [{"key": "service.name", "value": {"stringValue": svc}}]
            },
            "scopeSpans": [{"scope": {"name": "x"}, "spans": spans}],
        }

    return {
        "batches": [
            rs(
                "forge-gateway",
                [{"name": "gateway.proxy", "spanId": _b64("00000000000000a1")}],
            ),
            rs(
                "forge-engine",
                [
                    {
                        "name": "POST /v1/completions",
                        "spanId": _b64("00000000000000b2"),
                        "parentSpanId": _b64(parent_of_engine),
                    }
                ],
            ),
        ]
    }


TEMPO_EXPECT = {
    "match": "trace",
    "api": "tempo",
    "service": "forge-gateway",
    "services": ["forge-gateway", "forge-engine"],
    "spans": ["gateway.proxy", "POST /v1/completions"],
    "parents": [["POST /v1/completions", "gateway.proxy"]],
    "wait_s": 0,
}


def test_trace_on_tempo_search_then_trace_by_id(tmp_path, fake_http):
    trace = {"doc": _otlp("00000000000000a1")}
    tempo = fake_http(
        {
            "/api/search": lambda q: (200, {"traces": [{"traceID": "t1"}]}),
            "/api/traces/t1": lambda q: (200, trace["doc"]),
        }
    )
    mc = _mc(tmp_path)
    v = matchers.evaluate(
        {**TEMPO_EXPECT, "base": tempo.url}, matchers.StepOut(rc=None), mc
    )
    assert v.status == "pass" and "trace t1" in v.detail, v.detail
    path, q = tempo.calls[0]
    assert path == "/api/search" and q["tags"] == "service.name=forge-gateway"
    assert int(q["start"]) <= mc.started_at <= int(q["end"])
    assert tempo.calls[1][0] == "/api/traces/t1"
    # The same spans without the parent edge do not count.
    trace["doc"] = _otlp("00000000000000ff")
    v = matchers.evaluate(
        {**TEMPO_EXPECT, "base": tempo.url}, matchers.StepOut(rc=None), mc
    )
    assert v.status == "fail" and "no trace among 1" in v.detail
    # resourceSpans under `trace` (the v2 shape) with hex ids also reads.
    doc = _otlp("00000000000000a1")
    for b in doc["batches"]:
        for sp in b["scopeSpans"][0]["spans"]:
            for k in ("spanId", "parentSpanId"):
                if k in sp:
                    sp[k] = base64.b64decode(sp[k]).hex()
    trace["doc"] = {"trace": {"resourceSpans": doc["batches"]}}
    v = matchers.evaluate(
        {**TEMPO_EXPECT, "base": tempo.url}, matchers.StepOut(rc=None), mc
    )
    assert v.status == "pass", v.detail


def test_trace_on_tempo_search_error(tmp_path, fake_http):
    tempo = fake_http({})
    v = matchers.find_trace(tempo.url, TEMPO_EXPECT, 0, 1)
    assert v[0] is None and "HTTP 404" in v[1] and "/api/search" in v[1]


def test_is_tempo_by_port_host_or_api():
    assert matchers.is_tempo("http://127.0.0.1:30320", {})
    assert matchers.is_tempo("http://localhost:3200", {})
    assert matchers.is_tempo("http://tempo.observability:80", {})
    assert not matchers.is_tempo("http://127.0.0.1:30686", {})
    assert not matchers.is_tempo("http://127.0.0.1:30320", {"api": "jaeger"})
    assert matchers.is_tempo("http://127.0.0.1:9", {"api": "Tempo"})
    with pytest.raises(HarnessError, match="want jaeger or tempo"):
        matchers.is_tempo("http://x", {"api": "zipkin"})
