# What the System Does After Pass 9

Pass 9 trains at scale and runs the capstone. Mixed precision, gradient accumulation, and activation checkpointing train like fp32 in less memory. Then a Llama of about 10M parameters is trained on TinyStories, with core ablations and the zoo table, evaluated with safety evals, released through the ModelRelease workflow, and served. Everything from earlier passes keeps working: the engine still streams the Pass 1 bigram, and the earlier models still train.

The training techniques are [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](../../12-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md). The capstone is [Capstone: TinyStories, owned end to end](../../capstones/01-tinystories.md), with [ModelRelease](../../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md), [model evals as tests](../../../software-craftsmanship/03-testing-mentality/08-model-evals-as-tests.md), [the datasheet and model card](../../../responsible-ai/03-model-and-data-cards/01-datasheet-and-model-card.md), [bias and safety evals](../../../responsible-ai/04-bias-and-safety-evals/01-bias-and-safety-evals.md), and [Hessian-vector products and the recompute-versus-memory schedule](../../../mathematics-for-computing/09-matrix-calculus-and-autodiff/04-hessian-vector-products-and-recompute.md). The optional [collectives](../../12-training-at-scale/02-collectives-over-processes.md) and [data parallelism and ZeRO](../../12-training-at-scale/03-ddp-and-zero.md) sections, and the optional curvature diagnostics (the Newton step, the top Hessian eigenvalue, and the edge of stability) and Muon optimizer outlined in [Optimization](../../../mathematics-for-computing/11-optimization/README.md) in *Mathematics for Computing*, are depth the capstone does not need.

## Mixed precision, accumulation, and recompute train like fp32 in less memory

The training techniques run through the system's own entry point on a real model: a byte-level Llama of about 1.6M parameters (a `llama-2m` config) trains on Tiny Shakespeare twice, with the same windows in the same order. The first run is fp32 with one batch of 32 per step. The second uses 4 micro-batches of 8 (gradient accumulation), emulated bf16 matmuls, and checkpointing of every decoder layer but the last. The second run reaches the first one's held-out loss within 2% and uses less memory. A 60-step run takes about 40 seconds; the full comparison trains 300 steps.

Why this matters: accumulation decouples the effective batch from what fits in memory, bf16 halves activation memory and doubles matmul throughput on real hardware, and activation checkpointing trades recomputation for memory. Showing that the combination matches fp32 within 2% is what justifies using it for the capstone.

The `train llama` verb of the `tinyllm` role is specified under "Verbs of later passes" in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md). It keeps the rules of that page: exit 2 on a usage error, and the last stdout line is one JSON object.

```text
{tinyllm} train llama --cfg <config.json> --data <text file> [--steps S]
              [--seq-len T] [--micro-batch B] [--accum K] [--bf16]
              [--checkpoint-activations] [--lr LR] [--seed S]
              [--baseline <run.json>] [--out <run.json>]
    A Llama from config.json (tl_tokenizer "bytes": ids are the file's
    bytes); the last tenth of the file is held out. Each step draws
    K x B windows of T + 1 bytes from the first nine tenths with PCG32
    (seed S), in micro-batch order, and makes one optimizer step with
    train_step_mixed: AdamW, clipping at 1.0, precision bf16 with --bf16,
    else fp32; --checkpoint-activations runs the decoder layers through
    checkpoint_sequential. Python's tracemalloc measures the peak of the
    training loop. Final line:
    {"loss": the fp32 cross-entropy on 16 seeded held-out windows,
     "train_loss", "first_loss", "steps", "tokens", "precision",
     "micro_batch", "accum", "recompute", "peak_mem" (bytes), "params",
     and with --baseline <run.json>: "loss_rel_diff" =
     |loss - base loss| / base loss, "peak_mem_ratio" = peak_mem / base
     peak_mem}; --out writes the same object to a file.
```

Drawing the same windows in the same order for both runs is what makes the comparison fair: the only differences are precision, accumulation, and recompute.

## The capstone: trained, evaluated, released, and served

The capstone model runs end to end through the system's own entry points. It trains under the subprocess activity contract of [subprocess activities](../../../ai-platform-engineering/05-durable-orchestration-and-workers/09-subprocess-activities.md), reaches the held-out bits per byte calibrated for its size, and writes samples that pass its own seeded quality suite. It trains again as a durable TrainRun whose trace reaches `train.step`, ships through the ModelRelease workflow with its gates and canary, and is served by the gateway. The artifacts (specs, report, zoo table, samples, model card, ledger, and ADRs) are under [`docs/capstone`](../../../../../course/solutions/docs/capstone/) and [`docs/MODEL_CARD.md`](../../../../../course/solutions/docs/MODEL_CARD.md). The model-card section covers the model card and datasheet, and the safety-evals section covers the safety rows the release gates on.

The verbs are specified in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md):

```text
{tinyllm} train --spec <file> --progress <file>
    The subprocess activity verb (spec/subprocess-activity.md): the run
    goes under TL_ARTIFACTS/runs/<TL_IDEMPOTENCY_KEY or spec name>/, its
    checkpoints in ckpt/ with LATEST (formats/checkpoint.md). Final line:
    {"steps", "final_loss", "ckpt"}.

{tinyllm} eval ppl --model <ckpt dir or model dir> --data <tokens.bin>
    Pass 5's verb; a run's ckpt/ directory means its LATEST step. The
    final line holds "bpb" (and "ppl", "tokens").

{tinyllm} eval --suite quality --model <dir> --seed S --samples N
    N seeded samples (seeds S to S + N - 1) from "Once upon a time",
    scored with the model-evals-as-tests suite. Final line: {"samples": N,
    "repeat8": mean repeated word 8-gram rate, "distinct1": mean distinct
    word fraction}.

{ctl} release --spec <release.json> --id <workflow id> [--approve] [--wait] [--json]
    Starts ModelRelease on a release spec; --approve sends the approve
    signal right after the start (unattended runs); with --wait the final
    line is {"workflow_id", "status", "outcome", "served"}.

{ctl} train --spec ... --id ... --wait --json   as in Pass 8.
```

The training specs are in [`specs/c1`](../../../../../course/solutions/entry/specs/c1/tinystories-10m.json): `tinystories-10m.json` (the capstone), `tinystories-short.json` (a shorter run of the same model), `smoke.json` (about 1M parameters, for a quick end-to-end check), and `release.json` (the release spec). The held-out bar for each size is the mean plus three standard deviations of five seeded runs. The release uses the durable services of Pass 8 (durable and worker) for the TrainRun, and the kind deploy of Pass 7 for release, serving, and traces.

Why this matters: this is the first model the course trains that is worth serving. Training it as a durable workflow makes the long run restartable, the quality suite and safety evals are what a release must pass, and ModelRelease with a canary is how a new model reaches users without blindly replacing a working one. [Pass 10](../course-p10-agents/README.md) puts an agent on top of it.
