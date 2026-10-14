<!-- ol:module C1 -->
# Capstone: TinyStories, owned end to end

## Overview

| | |
|---|---|
| **Module** | `C1` · practice · Python, Go, and ops · Pass 9 · 2 to 3 days of work plus 4 to 12 h of laptop training (the `short` config: under 1 h; the smoke tier: minutes) |
| **You build** | nothing new in the library: you **run** your system on one goal and write up what it measured. Artifacts: `specs/c1/` (the full, short, and release specs), `docs/capstone/c1/` (`report.json`, `zoo.json`, `samples.jsonl`, the model card and data ledger the release names), and two ADRs in `docs/adr/` |
| **Contract** | [`formats/train-spec.schema.json`](../../../course/contracts/formats/train-spec.schema.json), [`formats/eval-results.schema.json`](../../../course/contracts/formats/eval-results.schema.json) (the zoo table), [`formats/ledger.schema.json`](../../../course/contracts/formats/ledger.schema.json), [`templates/ADR.md`](../../../course/contracts/templates/ADR.md), [`templates/MODEL_CARD.md`](../../../course/contracts/templates/MODEL_CARD.md), and the report schema `course/tests/C1/capstone-report.schema.json` |
| **Tests** | `course/tests/C1/`: the artifact check (section 4), which recomputes what can be recomputed and runs your `dur.12` preflight; the run itself is MS-C1's |
| **Needs** | [`dur.12`](../../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) (release workflow preflight), [`L11.1`](../p11-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md) (bf16, accumulation, activation recompute in the full and short train specs), [`ethics.04`](../../../responsible-ai/04-bias-and-safety-evals/01-bias-and-safety-evals.md) (safety suite in the release report); reading: every pass before this one, above all `data.07` and `data.08`, `L1.2`, `L1.4`, `L1.6`, `L2.1`, `L6.7`, `L7.4`, `L7.6`, `L7.8`, `L7.9`, `M03.5`, `M05.1`, `M07.5`, `M10.3`, `M10.4`, `dur.09`, `craft.22` |
| **Used by** | no call site (the capstone): `C2` post-trains its release, and the `ops.08` data-incident drill retrains it |
| **Milestone** | `MS-C1` (part of MS-P9) |
| **Optional depth** | Eldan and Li, [*TinyStories*](https://arxiv.org/abs/2305.07759) (2023, free); Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020, free); Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (2022, free); DeepSeek-AI, [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) (MLA, free) |

## Key Takeaways

- The capstone is your system doing one real job end to end: clean data through `CorpusBuild`, a trained tokenizer, a Llama trained as a durable `TrainRun` with mixed precision and resume, evaluation against the zoo, a gated release through `ModelRelease`, and serving through your engine and gateway.
- An **ablation is an experiment**: change one thing, hold the budget equal (the same vocabulary size, equal KV bytes per token, equal active parameters), pair the measurements by held-out document, and name a winner only when the 95% CI of the paired difference excludes zero (`test_ablations_are_fair_and_paired`).
- Every number in the report is checkable: the parameter count follows from the config (`test_hand_example_parameter_count`), the scaling fit from its points (`test_scaling_fit_reproduces`), the zoo's Llama row is the report's bpb (`test_zoo_table`).
- Decisions are recorded with their evidence: the vocabulary ADR cites the tokenizer ablation, the architecture ADR cites the attention and MLP ablations (`test_adrs_cite_the_ablations`).
- The model ships only through **your** release workflow (`test_release_passes_your_preflight`) and is then served like any other model.

## How to work this chapter

```bash
ol check C1                            # tells you which artifact is missing
<system> data build --dataset tinystories --version v1      # CorpusBuild (data.09)
{tinyllm} tok train --algo bpe --vocab 4096 --sample 16MB   # and --algo unigram, for the ablation
<system> train --spec specs/c1/tinystories-short.json        # TrainRun (dur.11); the full spec when you can spare a night
<system> train --spec specs/c1/abl-mla.json                  # one short run per ablation arm, same seed
<system> eval --suite zoo                                    # EvalSuite (dur.11) over every family you trained
# write docs/capstone/c1/report.json from the runs' outputs, then the two ADRs
ol check C1                            # the artifact check
<system> release --spec specs/c1/release.json && <system> wf signal <id> approve
ol milestone MS-C1                     # val loss at the calibrated bar, samples, conformance, artifacts, a trace
```

`ol milestone MS-C1 --smoke` prepares the committed synthetic stories, trains the smoke config (about 1M parameters, 200 steps), measures held-out bpb, and checks 20 seeded samples. The preparation step fits BPE on the first 130 documents only, scrubs the fixture's seeded redaction examples, and keeps the last 32 documents for evaluation. Token manifests and separate corpus ledgers record the files, hashes, counts, and license. The durable training, release, trace, and serving steps require the full milestone and its cluster.

The full milestone expects your completed `CorpusBuild` and tokenizer outputs under `artifacts/`, at the paths named by the training spec. Its preparation step checks and copies those inputs into the run directory without fitting another tokenizer or changing the split. A missing input is an error. Both tiers write a run-local `train-spec.json` with the milestone seed, so five-seed calibration actually trains five independent initializations.

The preparation entry points are:

```sh
# Full tier: use the inputs produced by your corpus pipeline.
{tinyllm} capstone prepare --spec specs/c1/tinystories-short.json --inputs artifacts --out runs/c1-inputs --seed 0
# Smoke tier: TINYLLM_FIXTURES is supplied by the course harness.
{tinyllm} capstone prepare --spec specs/c1/smoke.json --fixture MS-corpus/sources/stories.jsonl --holdout 32 --license CC0-1.0 --out runs/c1-smoke-inputs --seed 0
```

---

## 1. Why now

Every part of your system has passed its own tests. None of them has had to work with all the others for hours, on data you did not choose, with numbers nobody gave you in advance. That is where the remaining bugs live: a tokenizer whose vocabulary disagrees with the model config, a checkpoint whose data cursor resumes one window late, an eval that reads the wrong split, a release spec your gateway cannot route. The capstone runs the whole system on one goal (a model that writes small stories) and asks you to defend each choice with a measurement. It is also the first time the course cannot hand you an oracle: the report is the evidence, and the check verifies that the evidence agrees with itself.

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

| Stage | What runs | Your modules |
|---|---|---|
| Data | `CorpusBuild`: fetch, filter (with the KN perplexity filter), dedup, decontamination, PII, shards; the tokenizer trained on a fixed 16 MB sample; encode to `uint16` `.bin` | `data.01` to `data.09`, `L1.2`, `L1.4`, `L1.5`, `L2.1` |
| Model | `LlamaConfig(vocab=4096, d=320, layers=8, heads=8, kv_heads=4, d_ff=864, ctx=512, tied)` | `L7.1` to `L7.9`, `M05.1` |
| Train | AdamW, WSD, clipping, bf16 emulation, accumulation, recompute; `TrainRun` segments with resume | `M10.3`, `M10.4`, `L11.1`, `L0.6`, `dur.09`, `dur.11` |
| Ablations | tokenizer, attention, MLP, one `short` run per arm | `L1.4`, `L7.6`, `L7.8`, `M07.5` |
| Evaluate | held-out bpb with a CI, seeded samples, the scaling fit, the zoo table | `L6.7`, `M03.5`, `M11.2`, `craft.22` |
| Release | export, eval gates, model card, ledger, canary | `dur.12`, `data.08` |
| Serve | the Rust engine behind your gateway | `L10.*`, `gw.*` |

### 2.2 Tiers

| Tier | Model | Tokens | Laptop time (estimate) | Accepted by |
|---|---|---|---|---|
| full | about 10.4M parameters | about $10^8$ | 4 to 12 h | `ol check C1`, MS-C1 |
| short | about 2.4M | about $2 \times 10^7$ | under 1 h | `ol check C1`, MS-C1 at a looser bar |
| smoke | about 0.8M, 200 steps | about $4 \times 10^5$ | minutes | CI only (`OL_SMOKE=1`, MS-C1 `--smoke`) |

$C = 6ND$ sizes the run before you start it: the full tier is $6 \times 1.04 \times 10^7 \times 10^8 \approx 6.2 \times 10^{15}$ FLOP. At an effective 0.2 to 0.4 TFLOP/s for numpy with Accelerate on an M-series laptop that is 4 to 9 hours (uncertain: measure your own throughput with `ol bench` first).

### 2.3 Ablations are paired experiments

One ablation changes one thing and holds the budget equal, or the comparison measures the budget:

- **tokenizer**: BPE vs Unigram at the same vocabulary size on the same sample. Compare in **bits per byte**, never per token: a tokenizer that splits text into more, easier tokens has a lower per-token loss and no better model.
- **attention**: GQA vs MLA at equal **KV bytes per token**. GQA caches $L \cdot 2 \cdot n_{kv} \cdot d_h$ values per token; MLA caches the latent $r$ plus the shared rope key $d_r$, $L \cdot (r + d_r)$. In bf16 each value is 2 bytes.
- **MLP**: dense SwiGLU vs a mixture of experts at equal **active** parameters per token (the router plus $k$ experts, not all $E$).

Measure both arms on the same held-out documents and pair by document: $\delta_u$ removes how hard each document is. The verdict follows the 95% bootstrap CI of $\bar\delta$: `b` wins when $h < 0$, `a` when $\ell > 0$, and otherwise it is a tie, which is a result, not a failure.

### 2.4 The scaling fit

Three sizes of the `short` family give three points $(N_i, \mathrm{bpb}_i)$. In log space the power law is a line, $\log \mathrm{bpb} = \log a - \alpha \log N$, fitted by least squares (`M03.5`'s `lstsq`). Three points do not prove a law; they tell you whether the next size is worth training.

### 2.5 The zoo table

The model-zoo suite (`L6.7`, run by `EvalSuite`) scores every family you trained on the same held-out text: KN-4, NPLM, LSTM, GPT, and the capstone Llama in bits per byte, plus the seq2seq (EM), classification (accuracy), and word-similarity (Spearman) rows. At the full and short tiers the capstone must beat every baseline.

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

**An ablation verdict**, from the reference smoke report: Unigram minus BPE is $\bar\delta = +0.032$ bits per byte over 32 documents, CI $(0.014, 0.059)$. The CI excludes zero and $\delta = b - a > 0$, so `a` (BPE) wins. MLA minus GQA is $-0.011$ with CI $(-0.042, 0.008)$: it includes zero, a tie.

The first two blocks are `test_hand_example_parameter_count`; the verdict rule is `test_ablations_are_fair_and_paired`.

## 4. The artifacts and their check

| Artifact | Content |
|---|---|
| `specs/c1/tinystories-10m.json`, `specs/c1/tinystories-short.json` | train specs (`formats/train-spec.schema.json`) of the two tiers |
| `specs/c1/release.json` | the `ModelRelease` spec of your capstone model (`dur.12` section 4): suites, a quality and a safety gate, the route, the canary, the burn query of `<model_id>-<version>` |
| `docs/capstone/c1/report.json` | the report (`course/tests/C1/capstone-report.schema.json`): tier, spec, tokenizer, model config and parameters, the training run, held-out bpb with its CI, the three ablations, the scaling points and fit, and the paths below |
| `docs/capstone/c1/zoo.json` | the zoo suite's report (`formats/eval-results.schema.json`) |
| `docs/capstone/c1/samples.jsonl` | 20 samples, seeds 0 to 19: `{"seed", "prompt", "text", ...}` |
| the model card and ledger your release spec names | `MODEL_CARD.md` from the template (`ethics.03`), the release's `LEDGER.jsonl` |
| two ADRs in `docs/adr/` | the vocabulary (citing the tokenizer ablation) and the architecture (citing the attention and MLP ablations) |

The report is written by your tooling from the runs' outputs (a few lines of Python over the progress files and eval reports), never by hand. The course's reference report is a smoke-tier run, `course/oracle/C1/smoke_capstone.py`.

**The check** (`ol check C1`, `course/tests/C1/check`):

| Test | KIND | Checks |
|---|---|---|
| `test_hand_example_parameter_count` | unit | section 3, then your report's parameter count against its own config |
| `test_specs_are_capstone_train_specs` | unit | both specs validate, are Llamas of the tier's size, and train on enough tokens |
| `test_report_is_complete` | unit | the report schema; a full or short tier (smoke only with `OL_SMOKE=1`); its spec, zoo, samples, and release exist; bpb inside its CI |
| `test_ablations_are_fair_and_paired` | unit | three ablations; equal vocab, KV bytes, active parameters (recomputed); mean inside its CI; the verdict follows the CI |
| `test_scaling_fit_reproduces` | unit | three or more sizes; the log-log least-squares fit of the points is the reported fit |
| `test_zoo_table` | unit | the zoo report validates; KN-4, NPLM, LSTM, GPT, Llama on one text; the other tasks scored or skipped with a reason; the Llama row is the report's bpb; the capstone wins at full and short |
| `test_samples_pass_the_quality_scorers` | unit | 20 seeds; at least 20 words; repeated 8-grams at most 0.25 per sample and 0.10 on average; at least 30% distinct words |
| `test_adrs_cite_the_ablations` | unit | the two ADRs, their sections, and the cited mean differences |
| `test_release_passes_your_preflight` | conformance | your Go `ReleaseSpec` decodes the release spec strictly, `Validate` and `Preflight` pass on its card and ledger; quality and safety gates; the burn query selects the served model |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. an ablation at unequal budget | MLA "wins" because it caches more; the MoE "wins" because it computes more | `test_ablations_are_fair_and_paired` |
| 2. a winner declared inside the noise | an ADR built on a CI that includes zero | `test_ablations_are_fair_and_paired` |
| 3. comparing tokenizers per token | the tokenizer with more, easier tokens looks better | `test_ablations_are_fair_and_paired` (bits per byte only) |
| 4. a report that describes another model | parameters, config, and zoo row disagree | `test_hand_example_parameter_count`, `test_zoo_table` |
| 5. a scaling fit drawn by eye or in linear space | $a$ and $\alpha$ that no least-squares fit gives | `test_scaling_fit_reproduces` |
| 6. baselines scored on different text | a zoo table whose rows cannot be compared | `test_zoo_table` |
| 7. a looping or degenerate sampler | the same 8-gram again and again; a handful of words | `test_samples_pass_the_quality_scorers` |
| 8. decisions without evidence | ADRs that state a choice and cite nothing | `test_adrs_cite_the_ablations` |
| 9. a release your own workflow refuses | a template placeholder in the card, a non-`train` source, a burn query of another model | `test_release_passes_your_preflight` |
| 10. a smoke run presented as the capstone | 200 steps of a 0.8M model reported as "the model" | `test_report_is_complete` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.12` | the release workflow the capstone ships through |
| Back | `L11.1` | the full and short training specs use bf16, accumulation, and checkpointed activations |
| Back | `ethics.04` | the release report carries the capstone's safety suite rows |
| Forward | `C2` (optional) | post-trains the released model into an instruction follower |
| Forward | `ops.08` | revokes a source of this model's data and drives the retrain decision through `ModelRelease` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the capstone pipeline | Karpathy's `nanoGPT` and `llm.c` | the same loop on GPUs, with careful throughput accounting | `llm.c/train_gpt2.c` |
| the ablations | the DeepSeek-V2 and Mixtral reports | MLA and MoE ablations at scale, with compute-matched baselines | the papers above |
| the scaling fit | Chinchilla's three approaches | fits over hundreds of runs, isoFLOP profiles | Hoffmann et al., section 3 |
| the zoo table | HELM, the Open LLM Leaderboard | many tasks, standardized prompts and seeds | `crfm-helm` |
