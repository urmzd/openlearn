"""dpo_loss.py -- DPO loss and its gradient, checked against finite differences (README §4).

Math -> code (Rafailov et al., 2023, arXiv:2305.18290):
    h(x, y_w, y_l) = beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ]
    L_DPO          = -E[ log sigmoid(h) ]
    dL/d log pi(y_w) = -beta * sigmoid(-h)        dL/d log pi(y_l) = +beta * sigmoid(-h)

The sigmoid(-h) weight is the whole story: pairs the policy already ranks
correctly (h >> 0) stop contributing; pairs it ranks wrongly get full weight.

To exercise the chain rule through a real "policy", each response's log-prob is
a sum of per-token log-softmax terms over free logits theta[t, :], so
    d log pi(y) / d theta[t] = onehot(y_t) - softmax(theta[t]).

Run:
    uv run --with numpy python dpo_loss.py
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(0)
VOCAB, T, PAIRS, BETA = 11, 6, 8, 0.1


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(log_sigmoid(x))


def seq_logprob(theta: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """theta: (P, T, V) logits; tokens: (P, T) -> (P,) summed log-prob per sequence."""
    lp = log_softmax(theta)
    return np.take_along_axis(lp, tokens[..., None], axis=-1)[..., 0].sum(axis=-1)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta: float = BETA):
    h = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    loss = float(-np.mean(log_sigmoid(h)))
    w = sigmoid(-h) / h.shape[0]
    return loss, -beta * w, beta * w, h  # loss, dL/dpi_w, dL/dpi_l, margins


def dpo_from_logits(theta_w, theta_l, y_w, y_l, ref_w, ref_l):
    pi_w, pi_l = seq_logprob(theta_w, y_w), seq_logprob(theta_l, y_l)
    loss, g_pw, g_pl, h = dpo_loss(pi_w, pi_l, ref_w, ref_l)
    onehot_w, onehot_l = np.eye(VOCAB)[y_w], np.eye(VOCAB)[y_l]
    d_w = onehot_w - np.exp(log_softmax(theta_w))  # d log pi / d theta
    d_l = onehot_l - np.exp(log_softmax(theta_l))
    grad_w = g_pw[:, None, None] * d_w
    grad_l = g_pl[:, None, None] * d_l
    return loss, grad_w, grad_l, h


def main() -> None:
    y_w = RNG.integers(0, VOCAB, (PAIRS, T))
    y_l = RNG.integers(0, VOCAB, (PAIRS, T))
    ref_theta_w = RNG.normal(size=(PAIRS, T, VOCAB))
    ref_theta_l = RNG.normal(size=(PAIRS, T, VOCAB))
    ref_w, ref_l = seq_logprob(ref_theta_w, y_w), seq_logprob(ref_theta_l, y_l)

    # 1) policy == reference -> every margin is 0 -> loss is exactly log 2
    loss0, *_ = dpo_from_logits(ref_theta_w, ref_theta_l, y_w, y_l, ref_w, ref_l)
    print(f"loss at policy == reference : {loss0:.6f}  (log 2 = {np.log(2):.6f})")
    assert abs(loss0 - np.log(2)) < 1e-12

    # 2) analytic gradient vs central finite differences, perturbed policy
    theta_w = ref_theta_w + RNG.normal(0, 0.5, ref_theta_w.shape)
    theta_l = ref_theta_l + RNG.normal(0, 0.5, ref_theta_l.shape)
    loss, gw, gl, _ = dpo_from_logits(theta_w, theta_l, y_w, y_l, ref_w, ref_l)
    eps, worst = 1e-6, 0.0
    for which, theta, grad in (("w", theta_w, gw), ("l", theta_l, gl)):
        for _ in range(40):
            idx = tuple(RNG.integers(0, s) for s in theta.shape)
            old = theta[idx]
            theta[idx] = old + eps
            lp = dpo_from_logits(theta_w, theta_l, y_w, y_l, ref_w, ref_l)[0]
            theta[idx] = old - eps
            lm = dpo_from_logits(theta_w, theta_l, y_w, y_l, ref_w, ref_l)[0]
            theta[idx] = old
            num = (lp - lm) / (2 * eps)
            denom = max(1e-10, abs(num) + abs(grad[idx]))
            worst = max(worst, abs(num - grad[idx]) / denom)
    print(f"loss at perturbed policy    : {loss:.6f}")
    print(f"grad check max rel error    : {worst:.2e}")
    assert worst < 1e-6, worst

    # 3) gradient descent on the logits drives the implicit reward margin up
    lr = 50.0
    for step in range(301):
        loss, gw, gl, h = dpo_from_logits(theta_w, theta_l, y_w, y_l, ref_w, ref_l)
        if step % 100 == 0:
            acc = float(np.mean(h > 0))
            print(
                f"step {step:3d}  loss {loss:.4f}  mean margin {h.mean():+.3f}  pref acc {acc:.2f}"
            )
        theta_w -= lr * gw
        theta_l -= lr * gl
    assert loss < loss0 and np.all(h > 0), "DPO should rank every chosen above rejected"
    print(
        "OK: analytic DPO gradient matches finite differences and training separates pairs"
    )


if __name__ == "__main__":
    main()
