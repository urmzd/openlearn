"""train_step_jax.py -- the same next-token training step in JAX + Optax (README §3, §7).

Math -> code is identical to train_step_torch.py; the differences are the programming model:
    params are an explicit pytree, the step is a pure function, jax.value_and_grad gives
    (loss, grads), optax returns updates you apply, and jax.jit compiles the whole step.
Data parallelism is declared, not coded: put the batch on a Mesh with
NamedSharding(mesh, P("data")) and XLA inserts the gradient all-reduce.

Run:
    uv run --with jax --with optax python train_step_jax.py
    # pretend to have 4 devices on CPU to see the batch sharded:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run --with jax --with optax python train_step_jax.py
"""

from __future__ import annotations

import sys

try:
    import jax
    import jax.numpy as jnp
    import optax
    from jax.sharding import AxisType, NamedSharding
    from jax.sharding import PartitionSpec as P
except ImportError:
    sys.exit(
        "jax/optax not installed. Run: uv run --with jax --with optax python train_step_jax.py"
    )

TEXT = "the quick brown fox jumps over the lazy dog. " * 40
VOCAB = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(VOCAB)}
SEQ, D, HEADS = 32, 64, 4


def init_params(key: jax.Array, vocab: int) -> dict:
    ks = jax.random.split(key, 5)

    def dense(k, i, o):
        return jax.random.normal(k, (i, o)) / jnp.sqrt(i)

    return {
        "tok": jax.random.normal(ks[0], (vocab, D)) * 0.02,
        "pos": jnp.zeros((SEQ, D)),
        "qkv": dense(ks[1], D, 3 * D),
        "o": dense(ks[2], D, D),
        "gate_up": dense(ks[3], D, 8 * D),
        "down": dense(ks[4], 4 * D, D),
        "g1": jnp.ones(D),
        "g2": jnp.ones(D),
        "gf": jnp.ones(D),
    }


def rmsnorm(x, g):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * g


def forward(p: dict, idx: jax.Array) -> jax.Array:
    b, t = idx.shape
    x = p["tok"][idx] + p["pos"][:t]
    q, k, v = jnp.split(rmsnorm(x, p["g1"]) @ p["qkv"], 3, axis=-1)
    q, k, v = (a.reshape(b, t, HEADS, D // HEADS) for a in (q, k, v))
    att = jax.nn.dot_product_attention(q, k, v, is_causal=True)  # (b, t, h, d)
    x = x + att.reshape(b, t, D) @ p["o"]
    g, u = jnp.split(rmsnorm(x, p["g2"]) @ p["gate_up"], 2, axis=-1)
    x = x + (jax.nn.silu(g) * u) @ p["down"]
    return rmsnorm(x, p["gf"]) @ p["tok"].T


def loss_fn(
    p: dict, inputs: jax.Array, targets: jax.Array, mask: jax.Array
) -> jax.Array:
    logits = forward(p, inputs)
    nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.sum(nll * mask) / jnp.maximum(jnp.sum(mask), 1.0)  # mask = SFT loss mask


OPT = optax.chain(
    optax.clip_by_global_norm(1.0), optax.adamw(3e-3, b1=0.9, b2=0.95, weight_decay=0.1)
)


@jax.jit
def train_step(p, opt_state, inputs, targets, mask):
    loss, grads = jax.value_and_grad(loss_fn)(p, inputs, targets, mask)
    updates, opt_state = OPT.update(grads, opt_state, p)
    return optax.apply_updates(p, updates), opt_state, loss


def main() -> None:
    n_dev = len(jax.devices())
    # Auto axes: GSPMD propagates shardings and inserts collectives (the classic jit model).
    # JAX now defaults to Explicit axes, where shardings become part of each array's type.
    mesh = jax.make_mesh((n_dev,), ("data",), axis_types=(AxisType.Auto,))
    batch_sharding = NamedSharding(
        mesh, P("data")
    )  # shard batch dim; params replicated
    print(f"jax {jax.__version__}, devices: {n_dev} x {jax.devices()[0].platform}")

    key = jax.random.key(0)
    params = init_params(key, len(VOCAB))
    opt_state = OPT.init(params)
    data = jnp.array([STOI[c] for c in TEXT])
    bs = 32 if 32 % n_dev == 0 else n_dev * 8

    with jax.set_mesh(
        mesh
    ):  # mesh in context for any sharding-aware op inside the step
        losses, inputs = run(params, opt_state, data, bs, key, batch_sharding)
    print(f"final loss {losses[-1]:.4f}, batch sharding: {inputs.sharding.spec}")
    assert losses[-1] < 0.25 * losses[0], (
        "next-token loss should fall sharply on repetitive text"
    )
    print("OK")


def run(params, opt_state, data, bs, key, batch_sharding):
    losses = []
    for step in range(300):
        key, sub = jax.random.split(key)
        starts = jax.random.randint(sub, (bs,), 0, len(data) - SEQ - 1)
        x = jax.vmap(lambda s: jax.lax.dynamic_slice(data, (s,), (SEQ + 1,)))(starts)
        inputs = jax.device_put(x[:, :-1], batch_sharding)
        targets = jax.device_put(x[:, 1:], batch_sharding)
        mask = jax.device_put(jnp.ones(targets.shape), batch_sharding)
        params, opt_state, loss = train_step(params, opt_state, inputs, targets, mask)
        losses.append(float(loss))
        if step % 50 == 0:
            print(f"step {step:3d}  loss {losses[-1]:.4f}")
    return losses, inputs


if __name__ == "__main__":
    main()
