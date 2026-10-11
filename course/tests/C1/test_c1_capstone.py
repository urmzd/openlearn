"""C1 course tests: the capstone's artifacts (DESIGN 4.3 C1, 6.4 practice).

Annotated exemplars (DESIGN 5.12). `ol check C1` runs these in your repo.
They cannot rerun hours of training in CI; the milestone MS-C1 runs your
entry points on your checkpoint for that. These tests check what a reviewer
of the capstone checks: that your specs are real train specs of the
capstone sizes, that the report's numbers agree with each other and with
what can be recomputed from your configs (parameter counts, KV bytes,
active parameters, the scaling fit), that every ablation is a fair paired
comparison with an honest verdict, that the zoo table and the samples are
there and pass the course's scorers, that your ADRs cite the ablations,
and that your release passes YOUR ModelRelease preflight (dur.12).

The worked example of the chapter (section 3): the full config (vocabulary
4096, hidden 320, 8 layers, 8 heads, 4 KV heads, SwiGLU 864, tied) has
10,409,280 parameters and caches 5,120 bytes of K and V per token in bf16.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(os.environ.get("OL_REPO", ".")).resolve()
COURSE = Path(os.environ.get("OL_COURSE_TREE", Path(__file__).resolve().parents[2]))
HERE = Path(__file__).resolve().parent
SMOKE = os.environ.get("OL_SMOKE", "") not in ("", "0")
REPORT = "docs/capstone/c1/report.json"


def contract(rel: str) -> Path:
    """A contract file: your vendored copy, else the course's."""
    p = REPO / "contracts" / rel
    return p if p.is_file() else COURSE / "contracts" / rel


def load_json(rel: str):
    p = REPO / rel
    assert p.is_file(), f"{rel} is missing (chapter section 4)"
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        pytest.fail(f"{rel}: not JSON: {e}")


def validate(doc, schema_path: Path, where: str) -> None:
    import jsonschema

    schema = json.loads(schema_path.read_text())
    errs = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(doc),
        key=lambda e: list(e.path),
    )
    assert not errs, f"{where}: " + "; ".join(
        f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}" for e in errs[:5]
    )


def llama_params(c: dict) -> tuple[int, int]:
    """(total, active) parameters of a Llama-family config.json, the course's
    own count (independent of your accounting module): tied or untied
    embeddings, GQA or MLA attention without biases, dense or routed SwiGLU."""
    d, V, L = c["hidden_size"], c["vocab_size"], c["num_hidden_layers"]
    H = c["num_attention_heads"]
    kv = c.get("num_key_value_heads", H)
    hd = c.get("head_dim", d // H)
    if c.get("tl_attention", "gqa") == "mla":
        r, dr = c.get("tl_mla_rank", c.get("kv_lora_rank")), c["qk_rope_head_dim"]
        dn, dv = c["qk_nope_head_dim"], c.get("v_head_dim", hd)
        attn = d * H * (dn + dr) + d * (r + dr) + r + r * H * (dn + dv) + H * dv * d
    else:
        attn = d * H * hd + 2 * d * kv * hd + H * hd * d
    E = c.get("tl_num_experts", 0) or 0
    k = c.get("tl_top_k_experts", 1)
    ff = (
        c.get("moe_intermediate_size", c["intermediate_size"])
        if E
        else c["intermediate_size"]
    )
    mlp_total = d * E + E * 3 * d * ff if E else 3 * d * ff
    mlp_active = d * E + k * 3 * d * ff if E else mlp_total
    base = (
        V * d * (1 if c.get("tie_word_embeddings", False) else 2)
        + d
        + L * (attn + 2 * d)
    )
    return base + L * mlp_total, base + L * mlp_active


def kv_bytes_per_token(c: dict) -> int:
    """bf16 KV cache bytes per token: K and V of every KV head (GQA), or the
    latent plus the shared rope key (MLA)."""
    L, H = c["num_hidden_layers"], c["num_attention_heads"]
    if c.get("tl_attention", "gqa") == "mla":
        return (
            L
            * (c.get("tl_mla_rank", c.get("kv_lora_rank")) + c["qk_rope_head_dim"])
            * 2
        )
    hd = c.get("head_dim", c["hidden_size"] // H)
    return L * 2 * c.get("num_key_value_heads", H) * hd * 2


FULL = {
    "tl_arch": "llama",
    "vocab_size": 4096,
    "hidden_size": 320,
    "intermediate_size": 864,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "tie_word_embeddings": True,
}


def test_hand_example_parameter_count():
    # WHY: the chapter's worked example: the full config's parameters and KV
    #      bytes by hand, then your report's parameter count must be the
    #      same count of YOUR config (a report that disagrees with its own
    #      config describes some other model).
    # KIND: unit
    # CHAPTER: C1 section 3, Worked example by hand
    assert llama_params(FULL) == (10_409_280, 10_409_280)
    assert kv_bytes_per_token(FULL) == 5_120
    r = load_json(REPORT)
    want = llama_params(r["model"]["config"])[0]
    assert r["model"]["params"] == want, (
        f"report says {r['model']['params']:,} parameters; its config has {want:,}"
    )


def test_specs_are_capstone_train_specs():
    # WHY: the full and short runs are defined by train specs your
    #      `{tinyllm} train --spec` and TrainRun (dur.11) execute. They must
    #      validate against formats/train-spec.schema.json and be the
    #      capstone's sizes: about 10M parameters on about 1e8 tokens (full),
    #      about 2.5M on about 2e7 (short), a Llama with a 4096-ish vocabulary.
    # KIND: unit
    # CHAPTER: C1 section 4, The artifacts and their check
    schema = contract("formats/train-spec.schema.json")
    for name, (lo, hi, min_tokens) in {
        "tinystories-10m": (8e6, 1.3e7, 5e7),
        "tinystories-short": (1.5e6, 3.5e6, 1e7),
    }.items():
        rel = f"specs/c1/{name}.json"
        s = load_json(rel)
        validate(s, schema, rel)
        m = s["model"]
        assert m.get("tl_arch") == "llama", f"{rel}: model.tl_arch must be llama"
        n = llama_params(m)[0]
        assert lo <= n <= hi, f"{rel}: {n:,} parameters, want {lo:,.0f} to {hi:,.0f}"
        b = s["batch"]
        tokens = s["steps"] * b["micro_batch"] * b.get("accum", 1) * b["seq_len"]
        assert tokens >= min_tokens, (
            f"{rel}: {tokens:,} training tokens, want at least {min_tokens:,.0f}"
        )
        assert 2048 <= m["vocab_size"] <= 16384, f"{rel}: vocab_size {m['vocab_size']}"


def test_report_is_complete():
    # WHY: the report is what the capstone measured; it must follow the
    #      report schema (course/tests/C1/capstone-report.schema.json), name a
    #      train spec that exists, and be a full or short run. A smoke-tier
    #      report (the milestone's --smoke config) is accepted only with
    #      OL_SMOKE=1, as CI runs it.
    # KIND: unit
    # CHAPTER: C1 section 4, The artifacts and their check
    r = load_json(REPORT)
    validate(r, HERE / "capstone-report.schema.json", REPORT)
    assert r["tier"] in ("full", "short") or (SMOKE and r["tier"] == "smoke"), (
        f"tier {r['tier']!r}: run the full or short capstone (smoke reports pass only under OL_SMOKE=1)"
    )
    validate(
        load_json(r["spec"]), contract("formats/train-spec.schema.json"), r["spec"]
    )
    lo, hi = r["eval"]["ci95"]
    assert lo <= r["eval"]["val_bpb"] <= hi, "the val bpb lies outside its own CI"
    assert r["train"]["tokens"] >= r["train"]["steps"], "fewer tokens than steps"
    for k in ("zoo", "samples", "release"):
        assert (REPO / r[k]).is_file(), f"report.{k} names {r[k]}, which does not exist"


def test_ablations_are_fair_and_paired():
    # WHY: the three core ablations are experiments, so each must hold one
    #      thing fixed: the tokenizer pair at one vocabulary size, GQA vs MLA
    #      at equal KV bytes per token (recomputed here, within 10%), dense vs
    #      MoE at equal active parameters (within 2%). Each reports a paired
    #      mean difference b - a with a 95% CI that contains it, and the
    #      verdict must follow the CI: a winner only when it excludes zero.
    # KIND: unit
    # CHAPTER: C1 section 5, Pitfalls
    r = load_json(REPORT)
    by = {a["id"]: a for a in r["ablations"]}
    assert set(by) == {"tokenizer", "attention", "mlp"}, (
        f"ablations {sorted(by)}: want tokenizer, attention, mlp"
    )
    min_n = 16 if r["tier"] == "smoke" else 100
    for a in by.values():
        lo, hi = a["ci95"]
        assert lo <= a["mean_diff"] <= hi, (
            f"{a['id']}: mean {a['mean_diff']} outside its CI {a['ci95']}"
        )
        want = "b" if hi < 0 else "a" if lo > 0 else "tie"
        assert a["winner"] == want, (
            f"{a['id']}: winner {a['winner']!r}, but the CI {a['ci95']} says {want!r}"
        )
        assert a["n"] >= min_n, (
            f"{a['id']}: n = {a['n']} paired units, want at least {min_n}"
        )
    t = by["tokenizer"]
    names = {t["a"]["config"].get("tokenizer"), t["b"]["config"].get("tokenizer")}
    assert names == {"bpe", "unigram"}, f"tokenizer arms {names}: want bpe and unigram"
    assert t["a"]["config"].get("vocab_size") == t["b"]["config"].get("vocab_size"), (
        "the tokenizer arms differ in vocab_size"
    )
    at = by["attention"]
    ka, kb = (
        kv_bytes_per_token(at["a"]["config"]),
        kv_bytes_per_token(at["b"]["config"]),
    )
    assert {
        at["a"]["config"].get("tl_attention", "gqa"),
        at["b"]["config"].get("tl_attention", "gqa"),
    } == {"gqa", "mla"}
    assert at.get("kv_bytes_per_token") == {"a": ka, "b": kb}, (
        f"KV bytes per token are {ka} and {kb}, the report says {at.get('kv_bytes_per_token')}"
    )
    assert abs(ka - kb) <= 0.10 * max(ka, kb), f"not equal KV bytes: {ka} vs {kb}"
    mp = by["mlp"]
    pa, pb = llama_params(mp["a"]["config"])[1], llama_params(mp["b"]["config"])[1]
    assert sorted(
        bool(c["config"].get("tl_num_experts")) for c in (mp["a"], mp["b"])
    ) == [False, True], "one dense arm and one MoE arm"
    assert mp.get("active_params") == {"a": pa, "b": pb}, (
        f"active parameters are {pa} and {pb}, the report says {mp.get('active_params')}"
    )
    assert abs(pa - pb) <= 0.02 * max(pa, pb), (
        f"not equal active parameters: {pa:,} vs {pb:,}"
    )


def test_scaling_fit_reproduces():
    # WHY: the scaling law is a least-squares fit of log bpb = log a - alpha
    #      log params over at least three sizes (M03.5's lstsq). Refit here:
    #      the reported a and alpha must be the fit of the reported points.
    # KIND: unit
    # CHAPTER: C1 section 2, Principles
    s = load_json(REPORT)["scaling"]
    pts = s["points"]
    n = [p["params"] for p in pts]
    assert len(set(n)) >= 3 and n == sorted(n), (
        f"scaling sizes {n}: want at least 3 distinct sizes in increasing order"
    )
    X = np.c_[np.ones(len(pts)), np.log(n)]
    coef, *_ = np.linalg.lstsq(X, np.log([p["bpb"] for p in pts]), rcond=None)
    a, alpha = math.exp(coef[0]), -coef[1]
    assert s["fit"]["a"] == pytest.approx(a, rel=1e-4) and s["fit"][
        "alpha"
    ] == pytest.approx(alpha, abs=1e-4), (
        f"the fit of the points is a = {a:.6g}, alpha = {alpha:.6g}; the report says {s['fit']}"
    )


def test_zoo_table():
    # WHY: the model-zoo baseline table puts the capstone among every model
    #      family you trained: KN-4, NPLM, LSTM, GPT, and your Llama, in bits
    #      per byte on the same held-out text, plus the seq2seq,
    #      classification, and word-similarity rows (scored, or skipped with
    #      a reason). Its Llama row is the report's val bpb. At the full and
    #      short tiers the capstone must beat every baseline.
    # KIND: unit
    # CHAPTER: C1 section 4, The artifacts and their check
    r = load_json(REPORT)
    z = load_json(r["zoo"])
    validate(z, contract("formats/eval-results.schema.json"), r["zoo"])
    assert z["suite"] == "zoo"
    rows = z["rows"]
    lm = [x for x in rows if x["metric"] == "bpb" and x["status"] == "ok"]
    assert len({x["task"] for x in lm}) == 1, (
        "every bpb row must be scored on the same held-out text"
    )
    arch = {x.get("tl_arch") for x in lm}
    assert {"nplm", "rnnlm", "gpt", "llama"} <= arch, (
        f"bpb rows for {sorted(a for a in arch if a)}: want nplm, rnnlm, gpt, llama"
    )
    assert any(x.get("tl_arch") is None and "kn" in x["model"].lower() for x in lm), (
        "a Kneser-Ney baseline row (model kn-4)"
    )
    for task in ("dates", "sst2-2k", "word-sim"):
        row = next((x for x in rows if x["task"] == task), None)
        assert row is not None and (row["status"] == "ok" or row.get("reason")), (
            f"a {task} row, scored or skipped with a reason"
        )
    llama = [x for x in lm if x.get("tl_arch") == "llama"]
    assert any(abs(x["value"] - r["eval"]["val_bpb"]) < 1e-6 for x in llama), (
        "the Llama row is not the report's val bpb"
    )
    if r["tier"] != "smoke":
        best = min(lm, key=lambda x: x["value"])
        assert best.get("tl_arch") == "llama", (
            f"{best['model']} beats the capstone on bpb"
        )


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def test_samples_pass_the_quality_scorers():
    # WHY: 20 seeded samples (seeds 0 to 19) of the capstone model, scored by
    #      the course's deterministic scorers: at least 20 words each, a
    #      repeated word 8-gram rate of at most 0.25 per sample and 0.10 on
    #      average (a looping model repeats itself), and at least 30% distinct
    #      words (a degenerate one repeats a few).
    # KIND: unit
    # CHAPTER: C1 section 4, The artifacts and their check
    r = load_json(REPORT)
    p = REPO / r["samples"]
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert sorted(x["seed"] for x in rows) == list(range(20)), (
        "want 20 samples with seeds 0 to 19"
    )
    rates = []
    for x in rows:
        ws = words(x["text"])
        assert len(ws) >= 20, f"seed {x['seed']}: {len(ws)} words"
        grams = [tuple(ws[i : i + 8]) for i in range(len(ws) - 7)]
        seen, rep = set(), 0
        for g in grams:
            rep += g in seen
            seen.add(g)
        rate = rep / len(grams) if grams else 0.0
        assert rate <= 0.25, f"seed {x['seed']}: repeated 8-gram rate {rate:.2f}"
        assert len(set(ws)) / len(ws) >= 0.3, (
            f"seed {x['seed']}: {len(set(ws))} distinct of {len(ws)} words"
        )
        rates.append(rate)
    assert sum(rates) / len(rates) <= 0.10, (
        f"mean repeated 8-gram rate {sum(rates) / len(rates):.3f}"
    )


def numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\w.])-?\d+\.\d+", text)]


def test_adrs_cite_the_ablations():
    # WHY: a decision record cites its evidence. One ADR decides the
    #      vocabulary (it names "vocab" or "tokenizer" in its title) and cites
    #      the tokenizer ablation's mean difference; one decides the
    #      architecture and cites the attention and MLP ablations' mean
    #      differences (to 3 decimals). Both keep the ADR template's sections.
    # KIND: unit
    # CHAPTER: C1 section 5, Pitfalls
    r = load_json(REPORT)
    by = {a["id"]: a["mean_diff"] for a in r["ablations"]}
    adrs = sorted((REPO / "docs" / "adr").glob("*.md"))
    assert adrs, "docs/adr/ has no ADRs"

    def cites(text: str, x: float) -> bool:
        return any(abs(v - x) <= 6e-4 for v in numbers(text))

    def ok(a: Path) -> bool:
        t = a.read_text()
        return all(
            h in t
            for h in ("## Status", "## Context", "## Decision", "## Consequences")
        )

    title = {a: a.read_text().splitlines()[0].lower() for a in adrs}
    vocab = [
        a
        for a in adrs
        if ("vocab" in title[a] or "tokeniz" in title[a])
        and cites(a.read_text(), by["tokenizer"])
        and ok(a)
    ]
    assert vocab, (
        f"no ADR titled for the vocabulary or tokenizer cites the tokenizer ablation ({by['tokenizer']:.3f})"
    )
    arch = [
        a
        for a in adrs
        if cites(a.read_text(), by["attention"])
        and cites(a.read_text(), by["mlp"])
        and ok(a)
    ]
    assert arch, (
        f"no ADR cites both the attention ({by['attention']:.3f}) and MLP ({by['mlp']:.3f}) ablations"
    )


PREFLIGHT = """package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"tinyllm/workflows"
)

func main() {
	b, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Println("read:", err)
		os.Exit(1)
	}
	var spec workflows.ReleaseSpec
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&spec); err != nil {
		fmt.Println("decode:", err)
		os.Exit(1)
	}
	if err := spec.Validate(); err != nil {
		fmt.Println("validate:", err)
		os.Exit(1)
	}
	res, err := workflows.Preflight(context.Background(), workflows.PreflightInput{ModelCard: spec.ModelCard, Ledger: spec.Ledger, Root: os.Args[2]})
	if err != nil {
		fmt.Println("preflight:", err)
		os.Exit(1)
	}
	out, _ := json.Marshal(map[string]any{"served": spec.Served(), "sources": res.Sources})
	fmt.Println(string(out))
}
"""


def test_release_passes_your_preflight(tmp_path):
    # WHY: the capstone ships through YOUR ModelRelease (dur.12). Its release
    #      spec, specs/c1/release.json, must decode into your ReleaseSpec,
    #      pass your Validate, and pass your preflight gates on the model card
    #      and data ledger it names (paths relative to the repo here); it must
    #      gate on a quality row and a safety row (ethics.04), and its burn
    #      query must select the served model <model_id>-<version>.
    # KIND: conformance
    # CHAPTER: C1 section 4, The artifacts and their check
    spec = load_json("specs/c1/release.json")
    suites = {g.get("suite") for g in spec.get("gates", [])}
    assert {"quality", "safety"} <= suites, (
        f"gates on {sorted(suites)}: want a quality and a safety gate"
    )
    served = f"{spec.get('model_id')}-{spec.get('version')}"
    assert served in spec.get("burn", {}).get("query", ""), (
        f"the burn query never selects {served}"
    )
    go = shutil.which("go")
    if go is None:
        pytest.fail(
            "go is not on PATH (ol doctor): the release is checked with your Go workflow"
        )
    assert (REPO / "go" / "go.mod").is_file(), (
        "go/go.mod is missing: start dur.12 first"
    )
    mod = tmp_path / "preflight"
    mod.mkdir()
    (mod / "go.mod").write_text("module c1preflight\n\ngo 1.25.0\n")
    (mod / "main.go").write_text(PREFLIGHT)
    uses = [str(REPO / "go"), str(mod)] + (
        [str(REPO / "contracts" / "go")]
        if (REPO / "contracts" / "go" / "go.mod").is_file()
        else []
    )
    (tmp_path / "go.work").write_text(
        "go 1.25.0\n\nuse (\n" + "".join(f"\t{u}\n" for u in uses) + ")\n"
    )
    env = dict(os.environ, GOWORK=str(tmp_path / "go.work"), GOFLAGS="")
    p = subprocess.Popen(
        [go, "run", ".", str(REPO / "specs" / "c1" / "release.json"), str(REPO)],
        cwd=mod,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        pytest.fail("go run timed out after 120 s")
    assert p.returncode == 0, (
        "your ModelRelease refuses the capstone release:\n" + out[-2000:]
    )
    got = json.loads(out.strip().splitlines()[-1])
    assert got["served"] == served and got["sources"] >= 1
