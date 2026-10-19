"""The `tinyllm` CLI, Pass 2 verbs (MS-L0 fixes their flags and final lines).

    gradcheck --suite all
    train bigram --method autograd --data <file> --out <dir> [--seed S] [--steps N] [--lr X]
    train bigram --method autograd --data <shard.bin> --out <dir> --max-steps N
                 --batch B --seq-len T --ckpt-every K [--seed S] [--resume]
    train mlp --data <digits.npz> --hidden H --epochs E --ckpt <dir> [--seed S]

Entry-point territory (D16): this file is yours. It is glue over L0.1 to L0.6
(Tensor, F, cross_entropy, Module and Linear, the loader, BigramLogits,
checkpoints, TokenStream), M06.3 (PCG32), and M10.3 (AdamW). The final JSON
line of each verb, the failpoint `train/after-step`, and the files a run
leaves are fixed by course/milestones/MS-L0.toml.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

VOCAB = 256  # the byte tokenizer (formats/tokenizer.md)
TENSOR = "bigram.weight"  # the tensor the tracer engine serves (spec/cli-roles.md)
BIGRAM_CONFIG = {
    "tl_arch": "bigram",
    "tl_tokenizer": "bytes",
    "vocab_size": VOCAB,
    "tl_format": 1,
}
TOKENS_MAGIC = 20240520  # formats/tokens-bin.md


class UsageError(Exception):
    exit_code = 2  # __main__ maps it to exit 2 (spec/cli-roles.md)


# -- failpoints (the TL_FAILPOINTS spec of the course testkits) ----------------


class Failpoints:
    """The subset of TL_FAILPOINTS this CLI evaluates: `name=[N*]crash` and
    `name=off`. `crash` exits 137 at once, as a SIGKILL would leave the run:
    no cleanup, no final line. `N*` fires on the Nth evaluation only."""

    def __init__(self, spec: str) -> None:
        self.table: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            name, eq, act = part.partition("=")
            act = act.strip()
            nth = 0
            if "*" in act:
                n, _, act = act.partition("*")
                if not n.isdigit() or int(n) < 1:
                    raise UsageError(f"TL_FAILPOINTS: bad count in {part!r}")
                nth = int(n)
            if not eq or act not in ("crash", "off"):
                raise UsageError(f"TL_FAILPOINTS: want name=[N*]crash, got {part!r}")
            if act == "crash":
                self.table[name.strip()] = nth

    def inject(self, name: str) -> None:
        if name not in self.table:
            return
        self.counts[name] = self.counts.get(name, 0) + 1
        nth = self.table[name]
        if nth == 0 or self.counts[name] == nth:
            os._exit(137)


# -- helpers --------------------------------------------------------------------


def read_tokens_bin(path: str) -> int:
    """Validate a formats/tokens-bin.md header for the byte tokenizer and
    return n_tokens. TokenStream maps the ids; this only rejects a wrong file
    before training starts."""
    head = np.fromfile(path, dtype="<i4", count=256)
    if head.size != 256 or int(head[0]) != TOKENS_MAGIC:
        raise ValueError(f"{path}: not a tokens .bin file (magic {TOKENS_MAGIC})")
    if int(head[1]) not in (1, 2):
        raise ValueError(f"{path}: unknown tokens .bin version {int(head[1])}")
    if int(head[3]) not in (0, VOCAB):
        raise ValueError(
            f"{path}: vocab_size {int(head[3])}, the bigram reads bytes (256)"
        )
    return int(head[2])


def params_sha256(model) -> str:
    """sha256 over every parameter in named_parameters() order, as float32
    little-endian C-order bytes: equal digests mean bitwise-equal weights."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.data, dtype="<f4").tobytes())
    return h.hexdigest()


def lr_or(a, default: float) -> float:
    """--lr when given, else the verb's default (each run kind has its own)."""
    return float(a.lr) if a.lr is not None else default


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def git_sha() -> str:
    head = Path(".git/HEAD")
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            sha = (Path(".git") / ref[5:]).read_text().strip()
        else:
            sha = ref
        if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return sha
    except OSError:
        pass
    return "unknown"


def rng_hex(state) -> dict:
    """A PCG32 state as trainer_state.json stores it: 16 hex digits each."""
    if isinstance(state, dict):
        return {k: str(v) for k, v in state.items()}
    s, inc = state
    return {"pcg_state": f"{int(s):016x}", "pcg_inc": f"{int(inc):016x}"}


def rng_tuple(state) -> tuple[int, int]:
    if isinstance(state, dict):
        return int(state["pcg_state"], 16), int(state["pcg_inc"], 16)
    s, inc = state
    return int(s), int(inc)


def save_model_dir(out: Path, weight: np.ndarray) -> None:
    """The model directory the tracer engine serves: unchanged since Pass 1."""
    from tinyllm.io.safetensors import save_safetensors

    out.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        str(out / "model.safetensors"),
        {TENSOR: np.ascontiguousarray(weight, dtype=np.float32)},
        {"format": "tinyllm"},
    )
    (out / "config.json").write_text(json.dumps(BIGRAM_CONFIG) + "\n")


def bigram_weight(model) -> np.ndarray:
    """BigramLogits holds one [V, V] parameter: row i = logits after byte i."""
    ps = list(model.parameters())
    if len(ps) != 1 or tuple(ps[0].data.shape) != (VOCAB, VOCAB):
        raise ValueError(
            f"BigramLogits must hold one [{VOCAB}, {VOCAB}] parameter, got "
            f"{[tuple(p.data.shape) for p in ps]}"
        )
    return np.asarray(ps[0].data, dtype=np.float32)


def nll_of(weight: np.ndarray, ids: np.ndarray) -> float:
    """Mean NLL in nats per byte of a [V, V] logits table on ids, float64."""
    z = weight[ids[:-1]].astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(ids.size - 1), ids[1:]].mean())


# -- gradcheck --------------------------------------------------------------------


def cmd_gradcheck(a) -> dict:
    if a.suite != "all":
        raise UsageError(f"gradcheck: unknown suite {a.suite!r} (Pass 2 has `all`)")
    from tinyllm.autograd import functional as F

    reports = F.gradcheck_all(rtol=1e-5)
    failed = sorted(name for name, r in reports.items() if not r.ok)
    worst = max(reports, key=lambda n: reports[n].max_rel_err) if reports else ""
    for name in sorted(reports):
        r = reports[name]
        print(
            f"{'ok  ' if r.ok else 'FAIL'} {name:<16} max_rel_err {r.max_rel_err:.3e}",
            flush=True,
        )
    result = {
        "suite": a.suite,
        "checks": len(reports),
        "failed": len(failed),
        "max_rel_err": float(
            max((r.max_rel_err for r in reports.values()), default=0.0)
        ),
        "worst": worst,
    }
    if failed:
        raise CheckFailed(result, f"gradcheck failed for {', '.join(failed)}")
    return result


class CheckFailed(Exception):
    """A verb that ran but did not pass: print the final line, then exit 1."""

    exit_code = 1

    def __init__(self, result: dict, why: str) -> None:
        super().__init__(why)
        self.result = result


# -- train bigram (autograd) ---------------------------------------------------------


def train_bigram_full_batch(a) -> dict:
    """Full-batch AdamW on every (byte, next byte) pair of a raw file: the
    count-MLE is the unique minimizer of this loss, so the trained table
    reaches its NLL (MS-L0 step autograd-bigram)."""
    from tinyllm.autograd import functional as F
    from tinyllm.autograd.losses import cross_entropy
    from tinyllm.lm.bigram import BigramLogits
    from tinyllm.optim.adamw import AdamW

    ids = np.frombuffer(Path(a.data).read_bytes(), dtype=np.uint8).astype(np.int64)
    if ids.size < 2:
        raise ValueError(f"{a.data}: need at least two bytes to fit a bigram")
    x, y = ids[:-1], ids[1:]
    model = BigramLogits(VOCAB)
    opt = AdamW(list(model.parameters()), lr=lr_or(a, 0.5), weight_decay=0.0)
    loss_v = float("nan")
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        loss = cross_entropy(F.reshape(model(x), (x.size, VOCAB)), y)
        loss.backward()
        opt.step()
        loss_v = float(loss.data)
        if step % 100 == 0 or step == a.steps:
            print(f"step {step:>5}  loss {loss_v:.6f}", flush=True)
    w = bigram_weight(model)
    out = Path(a.out)
    save_model_dir(out, w)
    return {
        "out": str(out),
        "tokens": int(ids.size),
        "nll": nll_of(w, ids),
        "steps": a.steps,
        "loss": loss_v,
    }


def train_bigram_stream(a, fp: Failpoints) -> dict:
    """Random-window minibatches from a tokens .bin through TokenStream, with
    atomic checkpoints every --ckpt-every steps under <out>/ckpt/ and --resume
    from the newest one that verifies (formats/checkpoint.md)."""
    from tinyllm.autograd import functional as F
    from tinyllm.autograd.losses import cross_entropy
    from tinyllm.io.checkpoint import load_checkpoint, save_checkpoint
    from tinyllm.io.tokens import TokenStream
    from tinyllm.lm.bigram import BigramLogits
    from tinyllm.num.rng import PCG32
    from tinyllm.optim.adamw import AdamW

    for flag in ("max_steps", "batch", "seq_len", "ckpt_every"):
        if getattr(a, flag) is None or getattr(a, flag) < 1:
            raise UsageError(
                f"train bigram on a .bin needs --{flag.replace('_', '-')} >= 1"
            )
    read_tokens_bin(a.data)
    out = Path(a.out)
    ckpt = out / "ckpt"
    cfg = {
        **BIGRAM_CONFIG,
        "data": Path(a.data).name,
        "batch": a.batch,
        "seq_len": a.seq_len,
        "lr": lr_or(a, 0.1),
        "seed": a.seed,
        "max_steps": a.max_steps,
    }
    model = BigramLogits(VOCAB)
    opt = AdamW(list(model.parameters()), lr=cfg["lr"], weight_decay=0.0)
    rng = PCG32(a.seed)
    stream = TokenStream([a.data], a.seq_len, a.batch, rng)
    step, tokens_seen = 0, 0
    if a.resume:
        ck = load_checkpoint(str(ckpt))
        model.load_state_dict(ck.model)
        opt.load_state_dict(ck.opt)
        step = int(ck.step)
        tokens_seen = int(ck.extra["tokens_seen"])
        cur = ck.extra["data_cursor"]
        stream.restore(
            {
                "shard": cur["shard"],
                "offset": cur["offset"],
                "rng": rng_tuple(ck.rng_state),
            }
        )
        print(f"resumed from step {step}", flush=True)
    loss_v = float("nan")
    while step < a.max_steps:
        xb, yb = stream.next_batch()
        opt.zero_grad()
        # BigramLogits reads 1-D ids (row t = logits after byte t); a bigram
        # has no context across positions, so the [B, T] window flattens.
        n = xb.size
        loss = cross_entropy(
            F.reshape(model(np.asarray(xb).reshape(n)), (n, VOCAB)),
            np.asarray(yb).reshape(n),
        )
        loss.backward()
        opt.step()
        step += 1
        tokens_seen += n
        loss_v = float(loss.data)
        if step % a.ckpt_every == 0 or step == a.max_steps:
            c = stream.cursor()
            save_checkpoint(
                str(ckpt),
                model,
                opt,
                step,
                rng_tuple(c["rng"]),
                {
                    "tokens_seen": tokens_seen,
                    "data_cursor": {
                        "shard": int(c["shard"]),
                        "offset": int(c["offset"]),
                    },
                    "lr": float(cfg["lr"]),
                    "config_sha256": config_sha256(cfg),
                    "git_sha": git_sha(),
                    "config": BIGRAM_CONFIG,
                },
            )
            print(f"step {step:>5}  loss {loss_v:.6f}  checkpoint", flush=True)
        fp.inject("train/after-step")
    w = bigram_weight(model)
    save_model_dir(out, w)
    c = stream.cursor()
    final = {
        "step": step,
        "tokens_seen": tokens_seen,
        "loss": loss_v,
        "loss_hex": float(loss_v).hex(),
        "data_cursor": {"shard": int(c["shard"]), "offset": int(c["offset"])},
        "params_sha256": params_sha256(model),
    }
    line = json.dumps(final)
    (out / "final.json").write_text(line + "\n")
    return final


# -- train mlp -------------------------------------------------------------------------


def train_mlp(a, fp: Failpoints) -> dict:
    """A one-hidden-layer ReLU MLP on the 8x8 digits, AdamW, minibatches from
    the seeded loader, one checkpoint per epoch. Each pixel is standardized
    with the training set's mean and standard deviation (a pixel that never
    varies becomes 0): about 0.967 test accuracy against 0.953 for x / 16."""
    from tinyllm.autograd import functional as F
    from tinyllm.autograd.losses import cross_entropy
    from tinyllm.autograd.mode import no_grad
    from tinyllm.autograd.tensor import Tensor
    from tinyllm.io.checkpoint import save_checkpoint
    from tinyllm.nn.layers import Linear
    from tinyllm.nn.module import Module
    from tinyllm.num.rng import PCG32
    from tinyllm.optim.adamw import AdamW
    from tinyllm.train.loop import DataLoader

    if a.ckpt is None:
        raise UsageError("train mlp needs --ckpt <dir>")
    with np.load(a.data) as z:
        xtr, ytr = z["x_train"], z["y_train"]
        xte, yte = z["x_test"], z["y_test"]
    xtr, xte = xtr.astype(np.float64), xte.astype(np.float64)
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    xtr = ((xtr - mu) / sd).astype(np.float32)
    xte = ((xte - mu) / sd).astype(np.float32)
    ytr, yte = ytr.astype(np.int64), yte.astype(np.int64)
    n_cls = int(max(ytr.max(), yte.max())) + 1

    class MLP(Module):
        def __init__(self, d_in: int, hidden: int, d_out: int, rng) -> None:
            super().__init__()
            self.fc1 = Linear(d_in, hidden, rng=rng)
            self.fc2 = Linear(hidden, d_out, rng=rng)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    rng = PCG32(a.seed)
    model = MLP(xtr.shape[1], a.hidden, n_cls, rng.substream("init"))
    lr, batch = lr_or(a, 1e-3), a.batch or 32
    opt = AdamW(list(model.parameters()), lr=lr, weight_decay=0.0)
    shuffle = rng.substream("shuffle")
    loader = DataLoader(
        {"x": xtr, "y": ytr}, batch_size=batch, shuffle=True, rng=shuffle
    )
    cfg = {
        "arch": "mlp",
        "hidden": a.hidden,
        "epochs": a.epochs,
        "lr": lr,
        "batch": batch,
        "seed": a.seed,
    }
    step, seen, loss_v = 0, 0, float("nan")
    for epoch in range(1, a.epochs + 1):
        total, nb = 0.0, 0
        for b in loader:
            opt.zero_grad()
            loss = cross_entropy(model(Tensor(b["x"])), b["y"])
            loss.backward()
            opt.step()
            step += 1
            seen += int(len(b["y"]))
            total += float(loss.data)
            nb += 1
            fp.inject("train/after-step")
        loss_v = total / max(nb, 1)
        save_checkpoint(
            str(Path(a.ckpt) / "ckpt"),
            model,
            opt,
            step,
            rng_tuple(shuffle.state()),
            {
                "tokens_seen": seen,
                "data_cursor": {"shard": 0, "offset": 0},
                "lr": float(lr),
                "config_sha256": config_sha256(cfg),
                "git_sha": git_sha(),
                "config": cfg,
            },
        )
        print(f"epoch {epoch:>3}  train_loss {loss_v:.4f}", flush=True)
    with no_grad():
        pred = np.asarray(model(Tensor(xte)).data).argmax(axis=1)
    return {
        "step": step,
        "epochs": a.epochs,
        "train_loss": loss_v,
        "test_acc": float((pred == yte).mean()),
    }


def cmd_train_p2(a) -> dict:
    fp = Failpoints(os.environ.get("TL_FAILPOINTS", ""))
    if a.what == "mlp":
        return train_mlp(a, fp)
    if a.what == "bigram" and a.method == "autograd":
        if a.out is None:
            raise UsageError("train bigram needs --out <dir>")
        if a.data.endswith(".bin"):
            return train_bigram_stream(a, fp)
        if a.resume:
            raise UsageError(
                "--resume needs a tokens .bin (checkpoints are written only there)"
            )
        return train_bigram_full_batch(a)
    raise UsageError(f"train: unknown model {a.what!r} (bigram, mlp)")
