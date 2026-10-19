# What the system does after Pass 9

**Pass result**: about 10M model trained on TinyStories (mixed precision, accumulation, resume) with core ablations and the zoo table, safety evals, released through ModelRelease, and served.

Pass 9 trains at scale and runs the capstone: mixed precision, gradient accumulation, and activation checkpointing that match fp32 in less memory, then a roughly 10M-parameter Llama on TinyStories, evaluated, released through the ModelRelease workflow, and served. Everything from earlier passes keeps working: the engine still streams the Pass 1 bigram and the earlier models still train.

| What the system does | Built from |
|---|---|
| Mixed precision, accumulation, and recompute train like fp32 in less memory. | `L11.1` |
| The capstone model, trained, evaluated, released through ModelRelease, and served. | `C1`, `dur.12`, `craft.22`, `L11.1`, `ethics.03`, `ethics.04`, `M08.4` |

The optional `L11.2`, `L11.3`, `M10.5`, and `M10.6` are depth that the capstone does not need.

## Details

## Mixed precision, accumulation, and recompute train like fp32 in less memory

The L11.1 training techniques, run through the system's own entry point on a
real model: a byte-level Llama of about 1.6M parameters (a `llama-2m`
config) trains on Tiny Shakespeare twice with the same windows in the same
order: once in fp32 with one batch of 32 per step, once with 4 micro-batches
of 8 (gradient accumulation), emulated bf16 matmuls, and every decoder layer
but the last checkpointed. The second run reaches the first one's held-out
loss within 2% and uses less memory. A 60-step run takes about 40 s; the
full comparison trains 300 steps.

Why it is needed: accumulation decouples the effective batch from what fits
in memory, bf16 halves activation memory and doubles matmul throughput on
real hardware, and activation checkpointing trades recomputation for
memory. Showing that the combination matches fp32 within 2% is what
justifies using it for the capstone.

The `train llama` verb of the `tinyllm` role
([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes"). Every verb keeps the rules of
that page: exit 2 on a usage error, the last stdout line is one JSON object.

  {tinyllm} train llama --cfg <config.json> --data <text file> [--steps S]
                [--seq-len T] [--micro-batch B] [--accum K] [--bf16]
                [--checkpoint-activations] [--lr LR] [--seed S]
                [--baseline <run.json>] [--out <run.json>]
      A Llama (L7.9) from config.json (tl_tokenizer "bytes": ids are the
      file's bytes); the last tenth of the file is held out. Each step draws
      K x B windows of T + 1 bytes from the first nine tenths with PCG32
      (seed S), in micro-batch order, and makes one optimizer step with
      train_step_mixed (L11.1): AdamW (M10.3), clipping at 1.0, precision
      bf16 with --bf16, else fp32; --checkpoint-activations runs the decoder
      layers through checkpoint_sequential. Python's tracemalloc measures
      the peak of the training loop. Final line:
      {"loss": the fp32 cross-entropy on 16 seeded held-out windows,
       "train_loss", "first_loss", "steps", "tokens", "precision",
       "micro_batch", "accum", "recompute", "peak_mem" (bytes), "params",
       and with --baseline <run.json>: "loss_rel_diff" =
       |loss - base loss| / base loss, "peak_mem_ratio" = peak_mem / base
       peak_mem}; --out writes the same object to a file.

Drawing the same windows in the same order for both runs is what makes the
comparison fair: the only differences are precision, accumulation, and
recompute.

## The capstone model, trained, evaluated, released through ModelRelease, and served

The capstone model, end to end through the system's own entry points: it
trains under the subprocess activity contract (dur.09), reaches the
held-out bits per byte calibrated for its size, writes samples that pass
its own seeded quality suite (craft.22), trains again as a durable TrainRun
whose trace reaches `train.step`, ships through the ModelRelease workflow
(dur.12) with its gates and canary, and is served by the gateway. The
artifacts (specs, report, zoo table, samples, model card, ledger, ADRs) are
under [`docs/capstone`](../../../../../course/solutions/docs/capstone/) and
[`docs/MODEL_CARD.md`](../../../../../course/solutions/docs/MODEL_CARD.md);
ethics.03 covers the model card and datasheet, and ethics.04 the safety
rows the release gates on.

The verbs ([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes"):

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
      scored with the craft.22 evals. Final line: {"samples": N,
      "repeat8": mean repeated word 8-gram rate, "distinct1": mean distinct
      word fraction}.
  {ctl} release --spec <release.json> --id <workflow id> [--approve] [--wait] [--json]
      Starts ModelRelease (dur.12) on a release spec; --approve sends the
      approve signal right after the start (unattended runs); with --wait
      the final line is {"workflow_id", "status", "outcome", "served"}.
  {ctl} train --spec ... --id ... --wait --json   as in Pass 8.

Sizes: the training specs are in
[`specs/c1`](../../../../../course/solutions/entry/specs/c1/tinystories-10m.json):
`tinystories-10m.json` (the capstone), `tinystories-short.json` (a shorter
run of the same model), `smoke.json` (about 1M parameters, for a quick end
to end check), and `release.json` (the release spec). The held-out bars for
each size are the mean plus three standard deviations of five seeded runs.
The release uses the durable services of Pass 8 (durable, worker) for the
TrainRun and the kind deploy of Pass 7 for release, serving, and traces.

Why it is needed: this is the first model the course trains that is worth
serving. Training it as a durable workflow makes the long run restartable,
the quality suite and safety evals are what a release must pass, and
ModelRelease with a canary is how a new model reaches users without
replacing a working one blindly.
