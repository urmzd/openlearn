# What the System Does After Pass 2

Pass 2 rebuilds the tracer's model on foundations written in the course: the mathematics, the autograd, the loader, and checkpoints. The bigram is now retrained by the course's own autograd, which takes over `bigram.py`; gradcheck covers every op; and a token-stream reader feeds training. Everything from Pass 1 keeps working. CI stays green, and the engine still streams the bigram through the gateway, now from a checkpoint trained by gradient descent, with no change to the engine because the checkpoint contract has not changed.

The capability rests on the autograd sections of *Build Your Own LLM System*, from [Tensor, broadcasting backward, and no_grad](../../p00-foundations/01-tensor-and-broadcasting-backward.md), [The op library and gradcheck_all](../../p00-foundations/02-op-library-and-gradcheck.md), [Losses with a fused backward](../../p00-foundations/03-losses-with-fused-backward.md), and [Module system and basic layers](../../p00-foundations/04-module-system-and-layers.md) to [Training loop and the autograd bigram](../../p00-foundations/05-training-loop-and-the-autograd-bigram.md) and [Safetensors, atomic checkpoints, and the token stream](../../p00-foundations/06-safetensors-checkpoints-and-token-streams.md). Those in turn rest on the mathematics, numerics, probability, optimization, and testing sections listed in the [pass opening](README.md).

## The autograd retrains the tracer and survives a kill

The autograd checks its own gradients, retrains the tracer bigram to the count MLE in a model directory the unchanged tracer engine serves, trains an MLP on the digits, and survives a kill: a resumed token-stream run ends bitwise equal to the uninterrupted one, its data cursor and random generator included.

Each piece earns its place. Gradcheck is the only independent evidence that every backward pass is right before any model depends on it. Retraining the bigram by gradient descent and getting the count MLE back proves the whole training stack on a problem whose answer is known. And bitwise resume is what makes the long runs of later passes restartable without changing their results.

## The new verbs

The solution's `tinyllm` entry point, [`__main__.py`](../../../../../course/solutions/entry/python/tinyllm/__main__.py), adds four Pass 2 verbs, specified under "Verbs of later passes" in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md). Every verb keeps the rules of that page: exit 2 on a usage error, and the last stdout line is one JSON object.

The first runs every gradient check. It calls `F.gradcheck_all()` from [The op library and gradcheck_all](../../p00-foundations/02-op-library-and-gradcheck.md), which applies the method of [Partial derivatives, gradients, gradcheck](../../../mathematics-for-computing/04-calculus-3/01-gradients-and-gradcheck.md) in float64, and exits 0 only when every check passes:

```text
{tinyllm} gradcheck --suite all
    Final line:
    {"suite": "all", "checks": <int>, "failed": <int>, "max_rel_err": <float>, "worst": "<op>"}
```

The second trains the bigram by gradient descent on a raw file. `--method counts`, the default, is the Pass 1 verb, unchanged:

```text
{tinyllm} train bigram --method autograd --data <raw file> --out <dir> [--seed S]
    Trains BigramLogits by gradient descent on the mean cross-entropy of
    every (byte, next byte) pair of the file and writes the Pass 1 model
    directory into <dir>: config.json and model.safetensors with
    bigram.weight F32 [256, 256].
    Final line: the Pass 1 keys {"out", "tokens", "nll"} (nll of the
    trained table on the file, nats per byte); more keys are allowed.
```

The same verb on a token shard trains from the token stream with checkpoints and resume. This is the form that demonstrates bitwise recovery:

```text
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
```

The fourth trains a small classifier, which exercises the layers and optimizer on a problem other than language:

```text
{tinyllm} train mlp --data <digits.npz> --hidden H --epochs E --ckpt <dir> [--seed S]
    A one-hidden-layer ReLU MLP with H hidden units trained for E epochs
    (the npz holds x_train uint8 [3823, 64] in 0..16, y_train, x_test,
    y_test), checkpoints under <dir>/ckpt/.
    Final line: {"step": <int>, "epochs": E, "train_loss": <float>, "test_acc": <float>}
```

A kill is simulated with a failpoint, set through `TL_FAILPOINTS`:

```text
train/after-step   evaluated once after every optimizer step, after that
                   step's checkpoint (if any) is complete.
                   train/after-step=N*crash exits 137 on the Nth evaluation
                   of this process, as a SIGKILL would.
```

To check recovery, crash after step N, rerun with `--resume`, and compare `final.json` byte for byte with an uninterrupted run. The `loss_hex` and `params_sha256` fields make that comparison exact rather than approximate.

## The engine is unchanged

From this pass on, the model the engine serves is trained with the autograd. The build step that writes `artifacts/models/bigram` becomes:

```toml
["uv", "run", "--project", "python", "python", "python/tinyllm/__main__.py",
 "train", "bigram", "--method", "autograd", "--data", "corpus.txt",
 "--out", "artifacts/models/bigram"]
```

The engine itself does not change: same model directory, same tensor, same config. That is the point of the checkpoint contract fixed in Pass 1. The training side can be replaced without touching serving, and [Pass 3](../course-p03-tokens-and-data/README.md) relies on the same separation as it adds tokenizers and a corpus.
