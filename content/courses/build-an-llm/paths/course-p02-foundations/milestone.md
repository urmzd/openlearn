# Pass 2 milestones

**Pass result**: the bigram retrained by their autograd (L0.5 takes over bigram.py), gradcheck everywhere, the token-stream reader; the engine unchanged (same checkpoint contract).

The path includes the milestone stages below. Run each component milestone after its modules pass, then run the pass gate. The component gates run before the pass gate, which also reruns the smoke steps of earlier passes.

| Gate | Specification | What it covers |
|---|---|---|
| `MS-L0` | [`MS-L0.toml`](../../../../../course/milestones/MS-L0.toml) | Your autograd retrains the tracer and survives a kill. Requires `L0.1`, `L0.2`, `L0.3`, `L0.4`, `L0.5`, `L0.6`. |
| `MS-P2` | [`MS-P2.toml`](../../../../../course/milestones/MS-P2.toml) | Foundations: your autograd retrains the model your engine serves. Requires `S-M00`, `M00.1`, `M00.2`, `M00.3`, `M00.4`, `M01.1`, `M01.2`, `M02.1`, `M01.3`, `S-M01`, `M02.2`, `S-M02`, `M04.1`, `M04.2`, `S-M04`, `S-M05`, `M06.1`, `S-M06a`, `M06.3`, `M03.2`, `M03.3`, `M03.4`, `S-M03a`, `M07.0`, `S-M07a`, `M09.1`, `M09.2`, `S-M09a`, `M11.1`, `S-M11a`, `M08.1`, `M08.2`, `M08.3`, `S-M08`, `M10.1`, `M10.2`, `M10.3`, `M10.4`, `S-M10a`, `M07.3`, `craft.03`, `L0.1`, `L0.2`, `L0.3`, `L0.4`, `L0.5`, `L0.6`. |


## Component gate details

## MS-L0: Your autograd retrains the tracer and survives a kill

Your autograd (L0.1 to L0.4) checks its own gradients, retrains the tracer
bigram to the count MLE in a model directory the unchanged tracer engine
serves (L0.5), trains an MLP on the digits, and survives a kill: a resumed
token-stream run ends bitwise equal to the uninterrupted one, cursor and
generator included (L0.6).

This file fixes the Pass 2 verbs of your `tinyllm` role (spec/cli-roles.md,
"Verbs of later passes"). Every verb keeps the rules of that page: exit 2 on
a usage error, the last stdout line is one JSON object.

  {tinyllm} gradcheck --suite all
      Runs F.gradcheck_all() (L0.2 over M04.1) in float64. Exit 0 only when
      every check passes. Final line:
      {"suite": "all", "checks": <int>, "failed": <int>, "max_rel_err": <float>, "worst": "<op>"}

  {tinyllm} train bigram --method autograd --data <raw file> --out <dir> [--seed S]
      Trains BigramLogits by gradient descent on the mean cross-entropy of
      every (byte, next byte) pair of the file (steps, learning rate, and
      optimizer are yours) and writes the Pass 1 model directory into <dir>:
      config.json and model.safetensors with bigram.weight F32 [256, 256].
      Final line: the Pass 1 keys {"out", "tokens", "nll"} (nll of the
      trained table on the file, nats per byte); more keys are allowed.
      `--method counts` (the default) is the Pass 1 verb, unchanged.

  {tinyllm} train bigram --method autograd --data <shard.bin> --out <dir>
                --max-steps N --batch B --seq-len T --ckpt-every K [--seed S] [--resume]
      A .bin --data is a formats/tokens-bin.md shard read by your TokenStream
      (random windows from a PCG32 seeded with S). Every K steps, and at step
      N, write an atomic checkpoint <dir>/ckpt/step-<nnnnnn>/ and
      <dir>/ckpt/LATEST (formats/checkpoint.md). --resume continues from the
      newest checkpoint under <dir>/ckpt/ that verifies. At the end write the
      model directory into <dir> and the final line, also as <dir>/final.json:
      {"step": N, "tokens_seen": <int>, "loss": <float>, "loss_hex": "<float.hex(loss)>",
       "data_cursor": {"shard": <int>, "offset": <int>}, "params_sha256": "<hex>"}
      where loss is the training loss of step N and params_sha256 hashes every
      parameter in named_parameters() order (name bytes, then float32
      little-endian C-order data). No path or time may appear in the line:
      two runs that agree bit for bit print the same bytes.

  {tinyllm} train mlp --data <digits.npz> --hidden H --epochs E --ckpt <dir> [--seed S]
      A one-hidden-layer ReLU MLP with H hidden units trained for E epochs
      (the npz holds x_train uint8 [3823, 64] in 0..16, y_train, x_test,
      y_test; preprocessing is yours), checkpoints under <dir>/ckpt/. Final
      line: {"step": <int>, "epochs": E, "train_loss": <float>, "test_acc": <float>}.

  Failpoint (the TL_FAILPOINTS spec of the course testkits):
      train/after-step   evaluated once after every optimizer step, after
                         that step's checkpoint (if any) is complete.
                         `train/after-step=N*crash` exits 137 on the Nth
                         evaluation of this process, as a SIGKILL would.

`ol milestone MS-L0 --smoke` runs the smoke steps (all of them: this
milestone needs no services). Every later pass gate reruns them.

## MS-P2: Foundations: your autograd retrains the model your engine serves

Pass 2 rebuilds the tracer's model on foundations you wrote: the math
modules, your autograd, your loader and checkpoints. The gate is MS-L0 plus
the spiral invariant: the smoke steps of MS-P0 and MS-P1 rerun, so your CI
stays green and your engine still streams the bigram through your gateway.

From this pass on, train the model your engine serves with your autograd:
in system.toml, the [build] step that writes artifacts/models/bigram becomes
  ["uv", "run", "--project", "python", "python", "python/tinyllm/__main__.py",
   "train", "bigram", "--method", "autograd", "--data", "{fixture:MS-P1/corpus.txt}",
   "--out", "artifacts/models/bigram"]
The engine is unchanged: same model directory, same tensor, same config.

`requires` is every Pass 2 stage of DESIGN 7.4 (math, solve sets, craft.03,
L0.1 to L0.6); each needs a fresh pass of your own.

Run a gate with `practice/bin/ol milestone <ID> --smoke`; omit `--smoke` for its full local and cluster steps.
