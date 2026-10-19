# Capstone: TinyStories, owned end to end

## Overview

| | |
|---|---|
| **Module** | `C1` · practice · Python, Go, and ops · Pass 9 · about 3 h reading; reproducing it takes 4 to 12 h of laptop training (the `short` config: under 1 h; the smoke tier: minutes) |
| **Solution** | nothing new in the library: the system **runs** on one goal and the write-up records what it measured. Reference artifacts (smoke tier): the specs [`specs/c1/`](../../../../course/solutions/entry/specs/c1/) (full, short, smoke, and release), the report and its companions in [`docs/capstone/c1/`](../../../../course/solutions/docs/capstone/c1/) (`report.json`, `zoo.json`, `samples.jsonl`, `MODEL_CARD.md`, `LEDGER.jsonl`), and two ADRs, [`0002-c1-vocabulary-size.md`](../../../../course/solutions/docs/adr/0002-c1-vocabulary-size.md) and [`0003-c1-architecture.md`](../../../../course/solutions/docs/adr/0003-c1-architecture.md). CLI glue: [`cli_c1.py`](../../../../course/solutions/entry/python/tinyllm/cli_c1.py), [`cli_c1_data.py`](../../../../course/solutions/entry/python/tinyllm/cli_c1_data.py), [`cli_c1_eval.py`](../../../../course/solutions/entry/python/tinyllm/cli_c1_eval.py) |
| **Contract** | [`formats/train-spec.schema.json`](../../../../course/contracts/formats/train-spec.schema.json), [`formats/eval-results.schema.json`](../../../../course/contracts/formats/eval-results.schema.json) (the zoo table), [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json), [`templates/ADR.md`](../../../../course/contracts/templates/ADR.md), [`templates/MODEL_CARD.md`](../../../../course/contracts/templates/MODEL_CARD.md), and the report format `tl.capstone-report.v1` (its fields are walked through in section 5) |
| **Needs** | [`dur.12`](../../../tracks/ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) (release workflow preflight), [`L11.1`](../p11-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md) (bf16, accumulation, activation recompute in the full and short train specs), [`ethics.04`](../../../tracks/responsible-ai/04-bias-and-safety-evals/01-bias-and-safety-evals.md) (safety suite in the release report); reading: every pass before this one, above all `data.07` and `data.08`, `L1.2`, `L1.4`, `L1.6`, `L2.1`, `L6.7`, `L7.4`, `L7.6`, `L7.8`, `L7.9`, `M03.5`, `M05.1`, `M07.5`, `M10.3`, `M10.4`, `dur.09`, `craft.22` |
| **Used by** | no call site (the capstone, the end-to-end result of Pass 9): `C2` post-trains its release, and the `ops.08` data-incident drill retrains it |
| **Optional depth** | Eldan and Li, [*TinyStories*](https://arxiv.org/abs/2305.07759) (2023, free); Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020, free); Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (2022, free); DeepSeek-AI, [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) (MLA, free) |

## Key Takeaways

- The capstone is the system doing one real job end to end: clean data through `CorpusBuild`, a trained tokenizer, a Llama trained as a durable `TrainRun` with mixed precision and resume, evaluation against the zoo, a gated release through `ModelRelease`, and serving through the engine and gateway.
- An **ablation is an experiment**: change one thing, hold the budget equal (the same vocabulary size, equal KV bytes per token, equal active parameters), pair the measurements by held-out document, and name a winner only when the 95% CI of the paired difference excludes zero.
- Every number in the report is checkable: the parameter count follows from the config, the scaling fit from its points, and the zoo's Llama row is the report's bpb.
- Decisions are recorded with their evidence: the vocabulary ADR cites the tokenizer ablation, the architecture ADR cites the attention and MLP ablations.
- The model ships only through the system's own release workflow and is then served like any other model.

## How to read this chapter

Sections 1 to 3 explain what the capstone measures and work the sizing arithmetic by hand. Section 4 lists the artifacts and what a correct capstone must satisfy. Section 5 walks through the reference run (the smoke tier, which anyone can reproduce in minutes): how each number in its report is derived, how each decision follows from a measurement, and why the release is allowed to ship. Before reading each step of section 5, try to predict what the report must contain.

**How the reference run is produced.** The capstone uses only verbs the system already has; `<system>` is the deployed workflow CLI and `{tinyllm}` the Python CLI:

```bash
<system> data build --dataset tinystories --version v1      # CorpusBuild (data.09)
{tinyllm} tok train --algo bpe --vocab 4096 --sample 16MB   # and --algo unigram, for the ablation
<system> train --spec specs/c1/tinystories-short.json        # TrainRun (dur.11); the full spec when there is a night to spare
<system> train --spec specs/c1/abl-mla.json                  # one short run per ablation arm, same seed
<system> eval --suite zoo                                    # EvalSuite (dur.11) over every family trained
# write docs/capstone/c1/report.json from the runs' outputs, then the two ADRs
<system> release --spec specs/c1/release.json && <system> wf signal <id> approve
```

The inputs are staged by `{tinyllm} capstone prepare` ([`cli_c1_data.py`](../../../../course/solutions/entry/python/tinyllm/cli_c1_data.py)):

```sh
# Full and short tiers: use the inputs produced by the corpus pipeline.
{tinyllm} capstone prepare --spec specs/c1/tinystories-short.json --inputs artifacts --out runs/c1-inputs --seed 0
# Smoke tier: the course's synthetic stories file.
{tinyllm} capstone prepare --spec specs/c1/smoke.json --fixture MS-corpus/sources/stories.jsonl --holdout 32 --license CC0-1.0 --out runs/c1-smoke-inputs --seed 0
```

The smoke preparation fits BPE on the first 130 documents only, scrubs the fixture's seeded redaction examples, and keeps the last 32 documents for evaluation; token manifests and separate corpus ledgers record the files, hashes, counts, and license. The full preparation instead checks and copies the completed `CorpusBuild` and tokenizer outputs under `artifacts/` into the run directory, without fitting another tokenizer or changing the split; a missing input is an error. Both write a run-local `train-spec.json` with the run's seed, so a five-seed calibration really trains five independent initializations. The smoke tier then trains the smoke config (about 0.8M parameters, 200 steps), measures held-out bpb, and scores 20 seeded samples; the durable training, release, trace, and serving steps need the full system and its cluster.

---

## 1. Why now

Every part of the system has passed its own tests. None of them has had to work with all the others for hours, on data you did not choose, with numbers nobody gave you in advance. That is where the remaining bugs live: a tokenizer whose vocabulary disagrees with the model config, a checkpoint whose data cursor resumes one window late, an eval that reads the wrong split, a release spec the gateway cannot route. The capstone runs the whole system on one goal (a model that writes small stories) and defends each choice with a measurement. There is no oracle for the result: the report is the evidence, and a correct capstone is one whose evidence agrees with itself and with the runs that produced it.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | trainable parameters of the model | `int` |
| $D$ | training tokens | `int` |
| $C \approx 6ND$ | training compute in FLOP (2 for the forward pass, 4 for the backward) | `float` |
| $\mathrm{bpb}$ | held-out bits per byte: total NLL in bits over the held-out bytes | `float` |
| $u = 1, \dots, n$ | a paired unit: one held-out document | |
| $\delta_u = \mathrm{bpb}_b(u) - \mathrm{bpb}_a(u)$ | the ablation's paired difference on document $u$ | `float` |
| $\bar\delta$, $[\ell, h]$ | the mean difference and its 95% bootstrap CI | `float` |
| $a$, $\alpha$ | the scaling fit $\mathrm{bpb} = a N^{-\alpha}$ | `float` |

### 2.1 The pipeline

| Stage | What runs | Modules |
|---|---|---|
| Data | `CorpusBuild`: fetch, filter (with the KN perplexity filter), dedup, decontamination, PII, shards; the tokenizer trained on a fixed 16 MB sample; encode to `uint16` `.bin` | `data.01` to `data.09`, `L1.2`, `L1.4`, `L1.5`, `L2.1` |
| Model | `LlamaConfig(vocab=4096, d=320, layers=8, heads=8, kv_heads=4, d_ff=864, ctx=512, tied)` | `L7.1` to `L7.9`, `M05.1` |
| Train | AdamW, WSD, clipping, bf16 emulation, accumulation, recompute; `TrainRun` segments with resume | `M10.3`, `M10.4`, `L11.1`, `L0.6`, `dur.09`, `dur.11` |
| Ablations | tokenizer, attention, MLP, one `short` run per arm | `L1.4`, `L7.6`, `L7.8`, `M07.5` |
| Evaluate | held-out bpb with a CI, seeded samples, the scaling fit, the zoo table | `L6.7`, `M03.5`, `M11.2`, `craft.22` |
| Release | export, eval gates, model card, ledger, canary | `dur.12`, `data.08` |
| Serve | the Rust engine behind the gateway | `L10.*`, `gw.*` |

### 2.2 Tiers

| Tier | Model | Tokens | Laptop time (estimate) | What it shows |
|---|---|---|---|---|
| full | about 10.4M parameters | about $10^8$ | 4 to 12 h | the capstone result: beats every zoo baseline |
| short | about 2.4M | about $2 \times 10^7$ | under 1 h | the capstone result at a looser bar |
| smoke | about 0.8M, 200 steps | about $4 \times 10^5$ | minutes | that the pipeline runs end to end; not a model result |

$C = 6ND$ sizes the run before it starts: the full tier is $6 \times 1.04 \times 10^7 \times 10^8 \approx 6.2 \times 10^{15}$ FLOP. At an effective 0.2 to 0.4 TFLOP/s for numpy with Accelerate on an M-series laptop that is 4 to 9 hours (uncertain: measure the machine's throughput on a short run first).

### 2.3 Ablations are paired experiments

One ablation changes one thing and holds the budget equal, or the comparison measures the budget:

- **tokenizer**: BPE vs Unigram at the same vocabulary size on the same sample. Compare in **bits per byte**, never per token: a tokenizer that splits text into more, easier tokens has a lower per-token loss and no better model.
- **attention**: GQA vs MLA at equal **KV bytes per token**. GQA caches $L \cdot 2 \cdot n_{kv} \cdot d_h$ values per token; MLA caches the latent $r$ plus the shared rope key $d_r$, $L \cdot (r + d_r)$. In bf16 each value is 2 bytes.
- **MLP**: dense SwiGLU vs a mixture of experts at equal **active** parameters per token (the router plus $k$ experts, not all $E$).

Measure both arms on the same held-out documents and pair by document: $\delta_u$ removes how hard each document is. The verdict follows the 95% bootstrap CI of $\bar\delta$: `b` wins when $h < 0$, `a` when $\ell > 0$, and otherwise it is a tie, which is a result, not a failure.

### 2.4 The scaling fit

Three sizes of the `short` family give three points $(N_i, \mathrm{bpb}_i)$. In log space the power law is a line, $\log \mathrm{bpb} = \log a - \alpha \log N$, fitted by least squares (`M03.5`'s `lstsq`). Three points do not prove a law; they say whether the next size is worth training.

### 2.5 The zoo table

The model-zoo suite (`L6.7`, run by `EvalSuite`) scores every family the system trained on the same held-out text: KN-4, NPLM, LSTM, GPT, and the capstone Llama in bits per byte, plus the seq2seq (EM), classification (accuracy), and word-similarity (Spearman) rows. At the full and short tiers the capstone must beat every baseline.

## 3. Worked example by hand

**Parameters** of the full config ($V = 4096$, $d = 320$, 8 layers, 8 heads of $d_h = 40$, 4 KV heads, SwiGLU $f = 864$, tied embeddings):

| Part | Count |
|---|---|
| embedding $V d$ (shared with the output) | $4096 \cdot 320 = 1{,}310{,}720$ |
| per layer: $W_q$, $W_o$ ($d \cdot 8 \cdot 40$ each) | $2 \cdot 102{,}400 = 204{,}800$ |
| per layer: $W_k$, $W_v$ ($d \cdot 4 \cdot 40$ each) | $2 \cdot 51{,}200 = 102{,}400$ |
| per layer: gate, up, down ($3 d f$) | $3 \cdot 320 \cdot 864 = 829{,}440$ |
| per layer: two RMSNorm gains | $640$ |
| 8 layers | $8 \cdot 1{,}137{,}280 = 9{,}098{,}240$ |
| final norm | $320$ |
| **total** | **10,409,280** |

**KV bytes per token** in bf16: $8 \cdot 2 \cdot 4 \cdot 40 \cdot 2 = 5{,}120$. An MLA arm at equal KV bytes needs $r + d_r = 5120 / (8 \cdot 2) = 320$, for example $r = 288$, $d_r = 32$.

**Compute**: $6 \cdot 10{,}409{,}280 \cdot 10^8 \approx 6.2 \times 10^{15}$ FLOP.

**An ablation verdict**, from the reference smoke report ([`report.json`](../../../../course/solutions/docs/capstone/c1/report.json)): Unigram minus BPE is $\bar\delta = +0.032$ bits per byte over 32 documents, CI $(0.014, 0.059)$. The CI excludes zero and $\delta = b - a > 0$, so `a` (BPE) wins. MLA minus GQA is $-0.011$ with CI $(-0.042, 0.008)$: it includes zero, a tie.

Section 5 recomputes the same quantities for the reference smoke-tier model and checks them against its report.

## 4. The artifacts and what they must satisfy

| Artifact | Content |
|---|---|
| `specs/c1/tinystories-10m.json`, `specs/c1/tinystories-short.json` | train specs (`formats/train-spec.schema.json`) of the two tiers; the reference also has `specs/c1/smoke.json` |
| `specs/c1/release.json` | the `ModelRelease` spec of the capstone model (`dur.12` section 4): suites, a quality and a safety gate, the route, the canary, the burn query of `<model_id>-<version>` |
| `docs/capstone/c1/report.json` | the report (`tl.capstone-report.v1`): tier, spec, tokenizer, model config and parameters, the training run, held-out bpb with its CI, the three ablations, the scaling points and fit, and the paths below |
| `docs/capstone/c1/zoo.json` | the zoo suite's report (`formats/eval-results.schema.json`) |
| `docs/capstone/c1/samples.jsonl` | 20 samples, seeds 0 to 19: `{"seed", "prompt", "text", ...}` |
| the model card and ledger the release spec names | `MODEL_CARD.md` from the template (`ethics.03`), the release's `LEDGER.jsonl` |
| two ADRs in `docs/adr/` | the vocabulary (citing the tokenizer ablation) and the architecture (citing the attention and MLP ablations) |

The report is written by tooling from the runs' outputs (a few lines of Python over the progress files and eval reports), never by hand. The reference report is a smoke-tier run, in [`course/solutions/docs/capstone/c1/`](../../../../course/solutions/docs/capstone/c1/).

### What a correct capstone must satisfy

Every item below can be recomputed from the artifacts alone, which is what makes the report evidence rather than a claim.

| Case | Checks |
|---|---|
| parameter count | section 3's count, and the report's parameter count recomputed from its own config |
| capstone train specs | both specs validate, are Llamas of the tier's size, and train on enough tokens |
| a complete report | the report's fields; a full or short tier (a smoke report is only a pipeline check); its spec, zoo, samples, and release exist; bpb inside its CI |
| fair, paired ablations | three ablations; equal vocabulary, KV bytes, active parameters (recomputed); mean inside its CI; the verdict follows the CI |
| a reproducible scaling fit | three or more sizes; the log-log least-squares fit of the points is the reported fit |
| the zoo table | the zoo report validates; KN-4, NPLM, LSTM, GPT, Llama on one text; the other tasks scored or skipped with a reason; the Llama row is the report's bpb; the capstone wins at full and short |
| sample quality | 20 seeds; at least 20 words; repeated 8-grams at most 0.25 per sample and 0.10 on average; at least 30% distinct words |
| ADRs cite the ablations | the two ADRs, their sections, and the cited mean differences |
| the release passes the system's own preflight | the Go `ReleaseSpec` decodes the release spec strictly, `Validate` and `Preflight` pass on its card and ledger; quality and safety gates; the burn query selects the served model |

## 5. The reference capstone, step by step

### From first principles

1. **Size the run before training it.** The goal fixes a compute budget (a laptop night), and $C \approx 6ND$ turns that budget into a model size and a token count. The full config's 10.4M parameters and $10^8$ tokens are what fits in 4 to 12 hours at 0.2 to 0.4 TFLOP/s; the smoke config is the same architecture shrunk until it trains in seconds.
2. **Make every reported number derivable.** The parameter count must follow from the config (section 3's table), bpb must lie inside its own CI, and the zoo's Llama row must equal the report's bpb. If any one of these disagrees, the report describes a different model from the one that was trained.
3. **An ablation changes one thing at an equal budget.** Otherwise the comparison measures the budget. So each arm's budget quantity (vocabulary size, KV bytes per token, active parameters) is computed and recorded next to the result.
4. **Pair by document and bootstrap the mean difference.** Documents vary in difficulty far more than the arms differ, so the per-document difference $\delta_u$ removes that variance, and resampling documents gives a CI for $\bar\delta$. The verdict is mechanical: a winner only when the CI excludes zero.
5. **Decisions follow from the verdicts.** A measured winner is adopted; a tie keeps the option that costs less (engine work, memory). The ADR records the numbers so the decision can be revisited when the full tier reruns the ablation.
6. **The scaling fit is a least-squares line in log space,** so it can be reproduced from the points; three points are a planning signal, not a law.
7. **Ship only through the release workflow.** The model's release spec must pass the same strict decoding, validation, and gates as any other release, and the canary's burn query must select this model, or a bad release would be judged by another model's error rate.

### The reference run, annotated

**The spec.** The smoke spec ([`smoke.json`](../../../../course/solutions/entry/specs/c1/smoke.json)) trains the smoke config with the same optimizer and schedule family as the full tier:

```json
{
  "name": "c1-smoke",
  "tokenizer": {"id": "c1-bpe-512", "path": "tokenizers/c1-bpe-512/tokenizer.json"},
  "optimizer": {"name": "adamw", "lr": 0.003, "betas": [0.9, 0.95], "weight_decay": 0.1},
  "schedule": {"name": "wsd", "warmup_steps": 20, "decay_frac": 0.2, "min_lr_ratio": 0.1},
  "precision": {"dtype": "fp32"},
  "batch": {"micro_batch": 16, "accum": 1, "seq_len": 128},
  "steps": 200, "ckpt_every": 100, "seed": 0
}
```

Tokens: $200 \times 16 \times 128 = 409{,}600$. Compute: $6 \times 791{,}680 \times 409{,}600 \approx 1.9 \times 10^{12}$ FLOP; the report's `wall_s` is 7.4 s, an effective 0.26 TFLOP/s, inside section 2.2's estimate, so the same machine would need roughly $6.2 \times 10^{15} / 0.26 \times 10^{12} \approx 6.6$ hours for the full tier. The short spec ([`tinystories-short.json`](../../../../course/solutions/entry/specs/c1/tinystories-short.json)) adds `"precision": {"dtype": "bf16", "recompute": true}` and `"accum": 2`, the `L11.1` features. Before any run, `check_spec` in [`cli_c1.py`](../../../../course/solutions/entry/python/tinyllm/cli_c1.py) rejects what the trainer cannot honour:

```python
def check_spec(spec: dict) -> list[str]:
    errors = []
    model = spec["model"]
    if model.get("tl_tokenizer") not in ("bytes", "file"):
        errors.append("llama requires tl_tokenizer bytes or file")
    if model.get("tl_tokenizer") == "file" and not spec.get("tokenizer", {}).get("path"):
        errors.append("file tokenization requires tokenizer.path")   # vocabulary must match a file
    if spec.get("precision", {}).get("dtype", "fp32") not in ("fp32", "bf16"):
        errors.append("capstone precision must be fp32 or bf16")
    schedule = spec["schedule"]
    # ... schedule name in constant, cosine, wsd
    warmup = schedule.get("warmup_steps", 0)
    if not isinstance(warmup, int) or not 0 <= warmup < spec["steps"]:
        errors.append("warmup_steps must be an integer in [0, steps)")   # else lr never peaks
    # ... decay_frac in (0, 1], min_lr_ratio in [0, 1], keep_ckpts and ckpt_every positive
    return errors
```

Returning a list of every problem, rather than raising on the first, means one failed submission reports everything wrong with the spec.

**Parameters.** The reference smoke model is $V = 512$, $d = 128$, 4 layers, 4 heads of $d_h = 32$, 2 KV heads, $f = 344$, tied. Following section 3's table: embedding $512 \cdot 128 = 65{,}536$; per layer $2 \cdot 128 \cdot 128 + 2 \cdot 128 \cdot 64 + 3 \cdot 128 \cdot 344 + 256 = 181{,}504$; four layers 726,016; final norm 128. Total **791,680**, which is the report's `"params"`.

**The headline number.** From [`report.json`](../../../../course/solutions/docs/capstone/c1/report.json):

```json
"eval": {"val_bpb": 0.500221, "ci95": [0.448999, 0.597273], "n": 32, "val_loss": 0.993277}
```

The CI is a bootstrap over the 32 held-out documents (not over tokens: tokens within a document are correlated, and resampling them would give a CI that is too narrow). bpb lies inside its CI, as it must.

**The ablations.** Each ablation records both arms, their configs, the paired mean difference, its CI, the verdict, and the equal-budget quantity:

```json
{
  "id": "attention",
  "question": "GQA or MLA at equal KV bytes per token",
  "metric": "bpb", "unit": "held-out document", "n": 32, "seed": 0,
  "a": {"name": "gqa", "config": {"...": "the smoke config"}, "value": 0.500221},
  "b": {"name": "mla", "config": {"...": "plus", "tl_attention": "mla", "tl_mla_rank": 112,
        "qk_rope_head_dim": 16, "qk_nope_head_dim": 32, "v_head_dim": 32}, "value": 0.499284},
  "mean_diff": -0.011246,
  "ci95": [-0.042154, 0.00765],
  "winner": "tie",
  "kv_bytes_per_token": {"a": 1024, "b": 1024}
}
```

Recompute the budgets. GQA caches $L \cdot 2 \cdot n_{kv} \cdot d_h = 4 \cdot 2 \cdot 2 \cdot 32 = 512$ values per token, 1,024 bytes in bf16. MLA caches $L \cdot (r + d_r) = 4 \cdot (112 + 16) = 512$ values, also 1,024 bytes: the rank 112 was chosen to make them equal. The MoE arm (4 experts, top-1, expert width 344) has active parameters 793,728 against 791,680; the 2,048 difference is the router, $4 \text{ layers} \times 128 \times 4$ experts. The tokenizer arms share vocabulary 512 and the same 65,840-byte sample.

Note that `mean_diff` is the mean of the per-document differences, not $\mathrm{bpb}_b - \mathrm{bpb}_a$ of the pooled values ($0.499284 - 0.500221 = -0.000937$). Pooled bpb weights documents by their byte count; the paired mean weights each document equally, which is what the bootstrap over documents resamples.

The CI comes from the course's percentile bootstrap, `bootstrap_ci` in [`prob/stats.py`](../../../../course/solutions/python/tinyllm/prob/stats.py), applied to the 32 differences with `stat = mean`:

```python
def bootstrap_ci(x, stat, n_boot, alpha, rng):
    a = _sample(x, 1)
    # ... alpha in (0, 1), n_boot a positive integer
    n = a.size
    thetas = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = np.empty(n, dtype=np.int64)
        for i in range(n):
            # One uniform per index, in order: the same seed resamples the same
            # rows in Python and in Go (ag.12).
            idx[i] = min(int(rng.uniform() * n), n - 1)
        thetas[b] = float(stat(a[idx]))     # the statistic of one resample
    return (
        float(stat(a)),                     # the point estimate on the real data
        quantile(thetas, alpha / 2.0),      # 2.5th percentile
        quantile(thetas, 1.0 - alpha / 2.0),  # 97.5th percentile
    )
```

The verdict rule of section 2.3 is then mechanical: `"b"` if the upper bound is below 0, `"a"` if the lower bound is above 0, else `"tie"`. Applied to the reference: tokenizer $(0.014, 0.059)$ excludes zero with $\delta > 0$, so BPE (`a`) wins; attention $(-0.042, 0.008)$ and MLP $(-0.0002, 0.033)$ include zero, so both are ties. The MLP interval is a useful lesson: its lower bound is almost exactly zero, and the rule still calls it a tie.

**The decisions.** The ADRs turn the verdicts into choices. [`ADR-0002`](../../../../course/solutions/docs/adr/0002-c1-vocabulary-size.md) adopts byte-level BPE because it won:

> On the paired held-out documents, Unigram's bits per byte minus BPE's is 0.032 on average, 95% CI (0.014, 0.059): BPE is better and the interval excludes zero.

[`ADR-0003`](../../../../course/solutions/docs/adr/0003-c1-architecture.md) keeps GQA and a dense MLP although neither alternative lost, because a tie is decided by cost:

> Both intervals include zero: neither alternative is measurably better at this tier, and both cost engine work.

and it lists the cost precisely: MLA needs the latent cache of `L8.2` and weight absorption in the engine, and the MoE holds four times the MLP weights for the same active compute. Both ADRs record the evidence's limits (one seed, smoke tier) as a consequence, so the full tier knows to rerun them.

**The scaling fit.** The report's three points are $(123{,}712,\ 0.50393)$, $(353{,}952,\ 0.504022)$, $(791{,}680,\ 0.500221)$, and the fit is $a = 0.527387$, $\alpha = 0.003773$. Reproduce it: regress $\log \mathrm{bpb}$ on $[1, \log N]$ by least squares; the intercept is $\log a$ and the slope is $-\alpha$. That gives exactly the reported values. $\alpha \approx 0.004$ is almost flat: at 200 steps every size is data-limited, so a bigger smoke model buys nothing. The model card says so, and that is the right reading: the smoke tier proves the pipeline, not the model.

**The zoo.** [`zoo.json`](../../../../course/solutions/docs/capstone/c1/zoo.json) scores every family on the same `ts-val` text:

| Model | Parameters | bpb (95% CI) |
|---|---|---|
| KN-4 | (counts) | 0.646 (0.599, 0.725) |
| NPLM | 230,208 | 0.602 (0.544, 0.707) |
| LSTM | 592,384 | 0.496 (0.453, 0.578) |
| GPT | 875,264 | 0.943 (0.877, 1.065) |
| c1-llama | 791,680 | 0.500 (0.449, 0.597) |

The Llama row is the report's `val_bpb`. At the smoke tier the LSTM edges it out (inside overlapping CIs), which is allowed: only the full and short tiers must beat every baseline. The seq2seq, classification, and word-similarity rows are `"skipped"` with a reason, which is how a report says "not measured here" without leaving a hole.

**The samples.** The 20 samples in [`samples.jsonl`](../../../../course/solutions/docs/capstone/c1/samples.jsonl) (prompt "Once upon a time", temperature 0.8, 64 tokens, seeds 0 to 19) are scored with `craft.22`'s word-level scorers by `quality` in [`cli_c1_eval.py`](../../../../course/solutions/entry/python/tinyllm/cli_c1_eval.py). The reference samples have at least 34 words each, no repeated 8-gram in any sample, and at least 60% distinct words (70% on average), well inside the thresholds of section 4.

**The release.** [`release.json`](../../../../course/solutions/entry/specs/c1/release.json):

```json
{
  "model_id": "c1-smoke", "version": "v1", "from": "runs/c1-smoke/ckpt",
  "dtype": "f32", "quant": "none",
  "model_card": "docs/capstone/c1/MODEL_CARD.md",
  "ledger": "docs/capstone/c1/LEDGER.jsonl",
  "suites": ["quality", "safety"], "eval_seed": 0,
  "gates": [
    {"suite": "quality", "task": "ts-val", "metric": "bpb", "op": "<=", "value": 0.6},
    {"suite": "safety", "task": "refusal", "metric": "score", "op": ">=", "value": 0.9}
  ],
  "route": "tinystories", "canary_weight": 0.1, "canary_wait_s": 600, "approve_timeout_s": 86400,
  "burn": {
    "query": "sum(rate(http_server_request_duration_seconds_count{model=\"c1-smoke-v1\",http_response_status_code=~\"5..\"}[5m])) / sum(rate(http_server_request_duration_seconds_count{model=\"c1-smoke-v1\"}[5m])) / (1 - 0.99)",
    "max": 14.4
  }
}
```

Each field earns its place. The quality gate (bpb at most 0.6) is set above the measured 0.500 and its upper CI bound 0.597, so the release passes on this evidence and a regressed retrain would not. The burn query selects `model="c1-smoke-v1"`, the `<model_id>-<version>` of this release, so the canary is judged on its own errors: the ratio is the 5xx rate over the error budget of a 99% SLO, and 14.4 is the standard fast-burn threshold (the rate that would spend a 30-day budget in about two days). The model card ([`MODEL_CARD.md`](../../../../course/solutions/docs/capstone/c1/MODEL_CARD.md)) carries the evaluation table with CIs and states the out-of-scope uses; the ledger records the one data source with its license, hash, and allowed uses (`train`, `eval`), which `Preflight` checks against the release.

### Why this design

- **Cost.** The full tier is one laptop night, plus one short run (under an hour) per ablation arm and per extra size of the scaling fit. The smoke tier reproduces the whole report in minutes, which is what makes it a usable regression check for the pipeline.
- **Paired bootstrap over a t-test on pooled numbers.** Two arms' pooled bpb differ by less than the spread between documents, so an unpaired comparison would call almost everything a tie; pairing removes the document effect. The bootstrap makes no normality assumption about per-document differences, which are skewed (a few hard documents dominate).
- **Equal budgets over "the default config of each".** An MLA arm that caches more, or an MoE that computes more per token, would win for a reason that has nothing to do with the architecture. Holding the budget equal and recording it makes the comparison mean what the question asks.
- **A tie decided by cost.** When the evidence cannot separate two options, the cheaper one to build and run is the right default; the ADR keeps the door open by naming what would change the decision.
- **Tradeoffs accepted.** One seed per arm (seed variance is not in the CI), 32 evaluation documents at the smoke tier, and a three-point scaling fit. Each is stated in the ADRs or the model card rather than hidden.

### Explain it back

1. Why compare tokenizers in bits per byte rather than per-token loss?

   <details><summary>Answer</summary>

   Per-token loss depends on how text is cut. A tokenizer that produces more, smaller tokens has easier predictions per token and a lower per-token loss without modelling the text any better. Bits per byte divides the total information by the number of bytes, which is the same for both tokenizers on the same text.

   </details>

2. The reference MLA arm uses rank 112 and a rope dimension of 16. Where do those numbers come from?

   <details><summary>Answer</summary>

   From equal KV bytes per token. GQA in the smoke config caches $4 \cdot 2 \cdot 2 \cdot 32 = 512$ values per token. MLA caches $L(r + d_r)$, so $4(r + d_r) = 512$ gives $r + d_r = 128$; with $d_r = 16$, $r = 112$.

   </details>

3. The MLP ablation's CI is $(-0.0002, 0.033)$. Why is the verdict a tie and not "dense wins"?

   <details><summary>Answer</summary>

   The rule names a winner only when the 95% CI of the paired difference excludes zero. $-0.0002 < 0 < 0.033$, so it does not, however close the bound is. Declaring a winner on a CI that includes zero would build an ADR on noise.

   </details>

4. Why is the release's quality gate 0.6 and not, say, 0.55?

   <details><summary>Answer</summary>

   The gate must pass the model that is being released, on the evidence the release evaluation produces, and fail a regression. The measured bpb is 0.500 with a CI up to 0.597; a gate of 0.55 sits inside the CI, so the same model could fail it on a different evaluation seed. 0.6 is just above the upper bound.

   </details>

5. Why must the burn query name `c1-smoke-v1` exactly?

   <details><summary>Answer</summary>

   The canary is promoted or rolled back on the burn rate of its own traffic. A query that matched another model (or every model) would let a broken release ride on a healthy model's error rate, or roll back a good one because of someone else's errors.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. an ablation at unequal budget | MLA "wins" because it caches more; the MoE "wins" because it computes more | each ablation records its budget (`kv_bytes_per_token`, `active_params`, equal vocabulary), and the arms are sized to match |
| 2. a winner declared inside the noise | an ADR built on a CI that includes zero | the verdict follows the CI; the attention and MLP ablations are recorded as ties |
| 3. comparing tokenizers per token | the tokenizer with more, easier tokens looks better | every ablation's metric is bpb |
| 4. a report that describes another model | parameters, config, and zoo row disagree | the parameter count follows from the config, and the zoo's Llama row is `val_bpb` |
| 5. a scaling fit drawn by eye or in linear space | $a$ and $\alpha$ that no least-squares fit gives | the fit is least squares on $\log \mathrm{bpb}$ against $\log N$, reproducible from the points |
| 6. baselines scored on different text | a zoo table whose rows cannot be compared | every zoo row is scored on `ts-val` |
| 7. a looping or degenerate sampler | the same 8-gram again and again; a handful of words | samples are scored for repeated 8-grams and distinct words; the reference has no repeats |
| 8. decisions without evidence | ADRs that state a choice and cite nothing | ADR-0002 and ADR-0003 quote each mean difference and CI |
| 9. a release the system's own workflow refuses | a template placeholder in the card, a non-`train` source, a burn query of another model | the card is filled from the report, the ledger allows `train` and `eval`, and the burn query names `c1-smoke-v1` |
| 10. a smoke run presented as the capstone | 200 steps of a 0.8M model reported as "the model" | the report says `"tier": "smoke"`, and the model card scopes it to a pipeline test |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.12` | the release workflow the capstone ships through |
| Back | `L11.1` | the full and short training specs use bf16, accumulation, and checkpointed activations |
| Back | `ethics.04` | the release report carries the capstone's safety suite rows |
| Forward | `C2` (optional) | post-trains the released model into an instruction follower |
| Forward | `ops.08` | revokes a source of this model's data and drives the retrain decision through `ModelRelease` |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the capstone pipeline | Karpathy's `nanoGPT` and `llm.c` | the same loop on GPUs, with careful throughput accounting | `llm.c/train_gpt2.c` |
| the ablations | the DeepSeek-V2 and Mixtral reports | MLA and MoE ablations at scale, with compute-matched baselines | the papers above |
| the scaling fit | Chinchilla's three approaches | fits over hundreds of runs, isoFLOP profiles | Hoffmann et al., section 3 |
| the zoo table | HELM, the Open LLM Leaderboard | many tasks, standardized prompts and seeds | `crfm-helm` |
