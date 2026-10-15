# Pass 9 milestones

**Pass result**: about 10M model trained on TinyStories (mixed precision, accumulation, resume) with core ablations and the zoo table, safety evals, released through ModelRelease, and served.

The path includes the milestone stages below. Run each component milestone after its modules pass, then run the pass gate. The component gates run before the pass gate, which also reruns the smoke steps of earlier passes.

| Gate | Specification | What it covers |
|---|---|---|
| `MS-L11` | [`MS-L11.toml`](../../../../../course/milestones/MS-L11.toml) | Mixed precision, accumulation, and recompute train like fp32 in less memory. Requires `L11.1`. |
| `MS-C1` | [`MS-C1.toml`](../../../../../course/milestones/MS-C1.toml) | Your capstone model, trained, evaluated, released through ModelRelease, and served. Requires `C1`, `dur.12`, `craft.22`, `L11.1`, `ethics.03`, `ethics.04`. |
| `MS-P9` | [`MS-P9.toml`](../../../../../course/milestones/MS-P9.toml) | Training at scale and the capstone: a model you trained, evaluated, released, and serve. Requires `M08.4`, `L11.1`, `ethics.03`, `ethics.04`, `dur.12`, `craft.22`, `C1`. |


## Component gate details

## MS-L11: Mixed precision, accumulation, and recompute train like fp32 in less memory

includes it. Page: content/courses/build-an-llm/paths/course-p09-capstone-training/milestone.md (owned
by the pass gate's author).

Your L11.1 training tricks, run through your own entry point on a real
model: a byte-level Llama of about 1.6M parameters (fixture
configs/llama-2m.json) trains on tinyshakespeare twice with the same
windows in the same order: once in fp32 with one batch of 32 per step, once
with 4 micro-batches of 8 (gradient accumulation), emulated bf16 matmuls,
and every decoder layer but the last checkpointed. The second run must
reach the first one's held-out loss within 2% and use less memory.

This file fixes the Pass 9 `train llama` verb of your `tinyllm` role
(spec/cli-roles.md, "Verbs of later passes"). Every verb keeps the rules of
that page: exit 2 on a usage error, the last stdout line is one JSON object.

  {tinyllm} train llama --cfg <config.json> --data <text file> [--steps S]
                [--seq-len T] [--micro-batch B] [--accum K] [--bf16]
                [--checkpoint-activations] [--lr LR] [--seed S]
                [--baseline <run.json>] [--out <run.json>]
      A Llama (L7.9) from config.json (tl_tokenizer "bytes": ids are the
      file's bytes); the last tenth of the file is held out. Each step draws
      K x B windows of T + 1 bytes from the first nine tenths with your
      PCG32 (seed S), in micro-batch order, and makes one optimizer step
      with your train_step_mixed (L11.1): AdamW (M10.3), clipping at 1.0,
      precision bf16 with --bf16, else fp32; --checkpoint-activations runs
      the decoder layers through your checkpoint_sequential. Python's
      tracemalloc measures the peak of the training loop. Final line:
      {"loss": the fp32 cross-entropy on 16 seeded held-out windows,
       "train_loss", "first_loss", "steps", "tokens", "precision",
       "micro_batch", "accum", "recompute", "peak_mem" (bytes), "params",
       and with --baseline <run.json>: "loss_rel_diff" =
       |loss - base loss| / base loss, "peak_mem_ratio" = peak_mem / base
       peak_mem}; --out writes the same object to a file.

`ol milestone MS-L11 --smoke` runs 60 steps per run (about 40 s); the full
run trains 300.

## MS-C1: Your capstone model, trained, evaluated, released through ModelRelease, and served

includes it). Page: content/courses/build-an-llm/paths/course-p09-capstone-training/milestone.md (owned
by the pass gate's author).

Your capstone model, end to end through your own entry points: it trains
under the subprocess activity contract (dur.09), reaches the calibrated
held-out bar for its tier, writes samples your own seeded quality suite
(craft.22) passes, trains again as a durable TrainRun whose trace reaches
train.step, ships through your ModelRelease workflow (dur.12) with its
gates and canary, and is served by your gateway. The artifacts (specs,
report, zoo table, samples, model card, ledger, ADRs) are C1's check, so
C1 is in `requires`; so are dur.12, craft.22, L11.1, ethics.03 (the model
card and datasheet), and ethics.04 (the safety rows the release gates on).

This file fixes these verbs (spec/cli-roles.md, "Verbs of later passes"):

  {tinyllm} train --spec <file> --progress <file>
      The subprocess activity verb (spec/subprocess-activity.md): the run
      goes under TL_ARTIFACTS/runs/<TL_IDEMPOTENCY_KEY or spec name>/, its
      checkpoints in ckpt/ with LATEST (formats/checkpoint.md). Final line:
      {"steps", "final_loss", "ckpt"}.
  {tinyllm} eval ppl --model <ckpt dir or model dir> --data <tokens.bin>
      MS-L6's verb; a run's ckpt/ directory means its LATEST step. The
      final line holds "bpb" (and "ppl", "tokens").
  {tinyllm} eval --suite quality --model <dir> --seed S --samples N
      N seeded samples (seeds S to S + N - 1) from "Once upon a time",
      scored with your craft.22 evals. Final line: {"samples": N,
      "repeat8": mean repeated word 8-gram rate, "distinct1": mean distinct
      word fraction}.
  {ctl} release --spec <release.json> --id <workflow id> [--approve] [--wait] [--json]
      Starts ModelRelease (dur.12) on a release spec; --approve sends the
      approve signal right after the start (unattended runs); with --wait
      the final line is {"workflow_id", "status", "outcome", "served"}.
  {ctl} train --spec ... --id ... --wait --json   as fixed by MS-durable.

Tiers: the full run uses `vars` (specs/c1/tinystories-short.json, or
change it to tinystories-10m.json); `ol milestone MS-C1 --smoke` (the
nightly train-smoke job) uses `smoke_vars`, the about-1M-parameter smoke
spec. The `<= calibrated` bars come from the reference's 5-seed runs per
tier in course/fixtures/ref-thresholds.tsv (key MS-C1/<step>), recorded by
`ol milestone MS-C1 --record-thresholds` on the author's machine
(DESIGN 9, B11; DEVIATIONS B112-10).

Your system.toml must declare [entry].tinyllm and [entry].ctl, the
services of MS-durable (durable, worker) for the TrainRun steps, and
[deploy] for the kind steps (release, serving, trace).

## MS-P9: Training at scale and the capstone: a model you trained, evaluated, released, and serve

Pass 9 trains at scale and runs the capstone: mixed precision, gradient
accumulation, and activation checkpointing that match fp32 in less memory
(MS-L11), then a roughly 10M-parameter Llama on TinyStories, evaluated,
released through your ModelRelease workflow, and served (MS-C1). The gate
is those two plus the spiral invariant: the smoke steps of every earlier
pass gate rerun, so your engine still streams the Pass 1 bigram and your
earlier models still train.

`requires` is every core Pass 9 stage of DESIGN 7.4 (the optional L11.2,
L11.3, M10.5, and M10.6 are not required). craft.22 grades your model
evals (rung R8) with its own artifact check, C1 its capstone artifacts.

ci is "nightly", not "pr": MS-C1's steps train the capstone (smoke tier in
the nightly train-smoke job, the full tier on your machine), so a PR run
of this gate would train it too. MS-L11 alone stays a PR milestone
(DEVIATIONS B112-11).

Run a gate with `practice/bin/ol milestone <ID> --smoke`; omit `--smoke` for its full local and cluster steps.
