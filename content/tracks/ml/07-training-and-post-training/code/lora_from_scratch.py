"""lora_from_scratch.py -- LoRA on a tiny frozen MLP, numpy only (README §5).

Math -> code:
    h      = W0 x + (alpha / r) * B A x          (W0 frozen, A in R^{r x k}, B in R^{d x r})
    init   : A ~ N(0, 1/k), B = 0  ->  the adapted model starts exactly at the base model
    grads  : G = dL/dW_eff  ->  dL/dB = s * G A^T,  dL/dA = s * B^T G,   s = alpha / r
    merge  : W_merged = W0 + s * B A               (zero extra inference latency)

Setup: a "pretrained" 2-layer MLP (frozen) must adapt to a new task whose true
weights differ from the base by a low-rank delta. That is the LoRA hypothesis
(Hu et al., 2021, arXiv:2106.09685): the fine-tuning update has low intrinsic rank.

Run:
    uv run --with numpy python lora_from_scratch.py
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(0)
D_IN, D_HID, D_OUT = 64, 128, 32
RANK, ALPHA = 4, 8.0
SCALE = ALPHA / RANK


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


# ---- the frozen "pretrained" base and the shifted target task -----------------
W1 = RNG.normal(0, 1 / np.sqrt(D_IN), (D_HID, D_IN))
W2 = RNG.normal(0, 1 / np.sqrt(D_HID), (D_OUT, D_HID))


def low_rank_delta(d: int, k: int, r: int, mag: float) -> np.ndarray:
    return (
        mag
        * RNG.normal(0, 1 / np.sqrt(k), (d, r))
        @ RNG.normal(0, 1 / np.sqrt(r), (r, k))
    )


W1_TASK = W1 + low_rank_delta(D_HID, D_IN, 2, 0.5)
W2_TASK = W2 + low_rank_delta(D_OUT, D_HID, 2, 0.5)

X = RNG.normal(size=(1024, D_IN))
Y = relu(X @ W1_TASK.T) @ W2_TASK.T


# ---- LoRA parameters ------------------------------------------------------------
def init_lora() -> dict[str, np.ndarray]:
    return {
        "A1": RNG.normal(0, 1 / np.sqrt(D_IN), (RANK, D_IN)),
        "B1": np.zeros((D_HID, RANK)),
        "A2": RNG.normal(0, 1 / np.sqrt(D_HID), (RANK, D_HID)),
        "B2": np.zeros((D_OUT, RANK)),
    }


def effective(p: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return W1 + SCALE * p["B1"] @ p["A1"], W2 + SCALE * p["B2"] @ p["A2"]


def loss_and_grads(p: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray):
    w1, w2 = effective(p)
    z = x @ w1.T
    h = relu(z)
    pred = h @ w2.T
    diff = pred - y
    n = x.shape[0]
    loss = float(np.mean(np.sum(diff**2, axis=1)))
    # backprop through the effective weights (the frozen W0 gets no update)
    d_pred = 2.0 * diff / n
    g_w2 = d_pred.T @ h
    d_z = (d_pred @ w2) * (z > 0)
    g_w1 = d_z.T @ x
    grads = {
        "B1": SCALE * g_w1 @ p["A1"].T,
        "A1": SCALE * p["B1"].T @ g_w1,
        "B2": SCALE * g_w2 @ p["A2"].T,
        "A2": SCALE * p["B2"].T @ g_w2,
    }
    return loss, grads


def finite_diff_check(p: dict[str, np.ndarray]) -> float:
    """Max relative error between analytic and central-difference grads on a few entries."""
    q = {k: v.copy() for k, v in p.items()}
    q["B1"] += RNG.normal(0, 0.1, q["B1"].shape)  # move off B=0 so dA is non-zero
    q["B2"] += RNG.normal(0, 0.1, q["B2"].shape)
    xb, yb = X[:64], Y[:64]
    _, g = loss_and_grads(q, xb, yb)
    eps, worst = 1e-6, 0.0
    for name in ("A1", "B1", "A2", "B2"):
        for _ in range(5):
            idx = tuple(RNG.integers(0, s) for s in q[name].shape)
            old = q[name][idx]
            q[name][idx] = old + eps
            lp, _ = loss_and_grads(q, xb, yb)
            q[name][idx] = old - eps
            lm, _ = loss_and_grads(q, xb, yb)
            q[name][idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(
                worst, abs(num - g[name][idx]) / max(1e-8, abs(num) + abs(g[name][idx]))
            )
    return worst


def train(
    steps: int = 1500, lr: float = 3e-3, batch: int = 128
) -> tuple[dict, list[float]]:
    p = init_lora()
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    for t in range(1, steps + 1):
        idx = RNG.integers(0, X.shape[0], batch)
        loss, g = loss_and_grads(p, X[idx], Y[idx])
        for k in p:  # Adam on the adapter params only
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            p[k] -= lr * (m[k] / (1 - b1**t)) / (np.sqrt(v[k] / (1 - b2**t)) + eps)
        if t % 250 == 0 or t == 1:
            history.append(loss)
            print(f"step {t:5d}  train loss {loss:.5f}")
    return p, history


def main() -> None:
    full = W1.size + W2.size
    lora = RANK * (D_IN + D_HID) + RANK * (D_HID + D_OUT)
    print(f"full fine-tune trainable params : {full:,}")
    print(
        f"LoRA r={RANK} trainable params     : {lora:,}  ({100 * lora / full:.2f}% of full)"
    )

    p0 = init_lora()
    base_loss, _ = loss_and_grads(p0, X, Y)
    w1, w2 = effective(p0)
    assert np.allclose(w1, W1) and np.allclose(w2, W2), (
        "B=0 init must reproduce the base model"
    )
    print(f"loss of frozen base on new task  : {base_loss:.5f}")

    err = finite_diff_check(p0)
    print(f"grad check max rel error         : {err:.2e}")
    assert err < 1e-5, err

    p, hist = train()
    final_loss, _ = loss_and_grads(p, X, Y)
    print(f"final full-data loss             : {final_loss:.2e}")
    assert final_loss < 0.1 * base_loss, (
        "LoRA should recover most of the low-rank task shift"
    )

    # merge: fold the adapter into the dense weights, outputs must be identical
    w1m, w2m = effective(p)
    merged = relu(X @ w1m.T) @ w2m.T
    unmerged = relu(X @ W1.T + SCALE * (X @ p["A1"].T) @ p["B1"].T)
    unmerged = unmerged @ W2.T + SCALE * (unmerged @ p["A2"].T) @ p["B2"].T
    assert np.allclose(merged, unmerged), "merged and unmerged LoRA must agree"
    print("merged == unmerged forward       : OK")
    print(
        f"loss {base_loss:.4f} -> {final_loss:.2e} training {lora:,} of {full:,} params"
    )


if __name__ == "__main__":
    main()
