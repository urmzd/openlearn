# What the system does after Pass 2

**Pass result**: the bigram retrained by the course's own autograd (L0.5 takes over bigram.py), gradcheck everywhere, the token-stream reader; the engine unchanged (same checkpoint contract).

Pass 2 rebuilds the tracer's model on foundations written in the course: the math modules, the autograd, the loader, and checkpoints. Everything from Pass 1 keeps working: CI stays green and the engine still streams the bigram through the gateway, now from a checkpoint trained by gradient descent.

| Capability | What it covers | Built from |
|---|---|---|
| Autograd retrains the tracer and survives a kill | The autograd checks its own gradients, retrains the tracer bigram, trains an MLP on the digits, and resumes bitwise from a checkpoint | `L0.1`, `L0.2`, `L0.3`, `L0.4`, `L0.5`, `L0.6` |
| Foundations | The math, numerics, probability, optimization, and testing modules the autograd rests on | `S-M00`, `M00.1`, `M00.2`, `M00.3`, `M00.4`, `M01.1`, `M01.2`, `M02.1`, `M01.3`, `S-M01`, `M02.2`, `S-M02`, `M04.1`, `M04.2`, `S-M04`, `S-M05`, `M06.1`, `S-M06a`, `M06.3`, `M03.2`, `M03.3`, `M03.4`, `S-M03a`, `M07.0`, `S-M07a`, `M09.1`, `M09.2`, `S-M09a`, `M11.1`, `S-M11a`, `M08.1`, `M08.2`, `M08.3`, `S-M08`, `M10.1`, `M10.2`, `M10.3`, `M10.4`, `S-M10a`, `M07.3`, `craft.03` |

## The autograd retrains the tracer and survives a kill

The autograd (L0.1 to L0.4) checks its own gradients, retrains the tracer
bigram to the count MLE in a model directory the unchanged tracer engine
serves (L0.5), trains an MLP on the digits, and survives a kill: a resumed
token-stream run ends bitwise equal to the uninterrupted one, cursor and
generator included (L0.6).

Why each piece is needed: gradcheck is the only independent evidence that
every backward is right before any model depends on it; retraining the
bigram by gradient descent and getting the count MLE back proves the whole
training stack on a problem whose answer is known; and bitwise resume is
what makes long runs in later passes restartable without changing results.

The solution's `tinyllm` entry point
([`__main__.py`](../../../../../course/solutions/entry/python/tinyllm/__main__.py))
adds these Pass 2 verbs ([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md),
"Verbs of later passes"). Every verb keeps the rules of that page: exit 2 on
a usage error, the last stdout line is one JSON object.

  {tinyllm} gradcheck --suite all
      Runs F.gradcheck_all() (L0.2 over M04.1) in float64. Exit 0 only when
      every check passes. Final line:
      {"suite": "all", "checks": <int>, "failed": <int>, "max_rel_err": <float>, "worst": "<op>"}

  {tinyllm} train bigram --method autograd --data <raw file> --out <dir> [--seed S]
      Trains BigramLogits by gradient descent on the mean cross-entropy of
      every (byte, next byte) pair of the file and writes the Pass 1 model
      directory into <dir>: config.json and model.safetensors with
      bigram.weight F32 [256, 256].
      Final line: the Pass 1 keys {"out", "tokens", "nll"} (nll of the
      trained table on the file, nats per byte); more keys are allowed.
      `--method counts` (the default) is the Pass 1 verb, unchanged.

  {tinyllm} train bigram --method autograd --data <shard.bin> --out <dir>
                --max-steps N --batch B --seq-len T --ckpt-every K [--seed S] [--resume]
      A .bin --data is a formats/tokens-bin.md shard read by the TokenStream
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
      y_test), checkpoints under <dir>/ckpt/. Final line:
      {"step": <int>, "epochs": E, "train_loss": <float>, "test_acc": <float>}.

  Failpoint (TL_FAILPOINTS):
      train/after-step   evaluated once after every optimizer step, after
                         that step's checkpoint (if any) is complete.
                         `train/after-step=N*crash` exits 137 on the Nth
                         evaluation of this process, as a SIGKILL would.

The failpoint is how the kill is simulated: crash after step N, rerun with
`--resume`, and compare `final.json` byte for byte with an uninterrupted
run. `loss_hex` and `params_sha256` make that comparison exact rather than
approximate.

## The engine is unchanged

From this pass on, the model the engine serves is trained with the
autograd: the build step that writes artifacts/models/bigram becomes

```toml
["uv", "run", "--project", "python", "python", "python/tinyllm/__main__.py",
 "train", "bigram", "--method", "autograd", "--data", "corpus.txt",
 "--out", "artifacts/models/bigram"]
```

The engine is unchanged: same model directory, same tensor, same config.
That is the point of the checkpoint contract fixed in Pass 1: the training
side can be replaced without touching serving.
