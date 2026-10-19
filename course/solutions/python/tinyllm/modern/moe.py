"""Mixture of experts: routing, sorted dispatch, the Switch load-balancing
loss, and DeepSeek-V3's aux-loss-free bias (L7.8).

A dense MLP spends the same FLOPs on every token. An MoE layer holds E
experts and lets a router send each token to k of them, so the parameters
grow with E while the compute per token grows with k. The engineering is in
moving tokens to their experts in contiguous batches (sorted dispatch) and
in keeping the router from sending everything to the same few experts.

Contract: contracts/py/tinyllm/modern/moe.pyi.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.mlp import GatedMLP
from tinyllm.nn.layers import Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32

ROUTERS = ("softmax_topk", "topk_softmax", "sigmoid")


def topk_ids(scores: ArrayLike, k: int) -> NDArray:
    s = np.asarray(scores)
    if not 1 <= k <= s.shape[-1]:
        raise ValueError(f"k must lie in [1, {s.shape[-1]}], got {k}")
    # A stable sort of -s keeps equal scores in id order: ties go to the lowest id.
    return np.argsort(-s, axis=-1, kind="stable")[..., :k].astype(np.int64)


def _ids(topk_idx: ArrayLike, n_experts: int) -> NDArray:
    idx = np.asarray(topk_idx)
    if idx.ndim != 2 or idx.dtype.kind not in "iu":
        raise ValueError(f"topk_idx must be an integer [N, k] array, got {idx.dtype} {idx.shape}")
    if idx.size and (idx.min() < 0 or idx.max() >= n_experts):
        raise ValueError(f"expert ids must lie in [0, {n_experts}), got {idx.min()}..{idx.max()}")
    return idx.astype(np.int64)


def load_balance_loss(router_probs: Tensor, topk_idx: ArrayLike, n_experts: int) -> Tensor:
    idx = _ids(topk_idx, n_experts)
    if router_probs.ndim != 2 or router_probs.shape != (idx.shape[0], n_experts):
        raise ValueError(f"router_probs must be [N, E] = {(idx.shape[0], n_experts)}, got {router_probs.shape}")
    N = idx.shape[0]
    # f_e counts assignments: a constant for autograd. P_e carries the gradient.
    f = np.bincount(idx.ravel(), minlength=n_experts).astype(router_probs.dtype) / N
    P = F.mean(router_probs, axis=0)
    return F.sum(P * f) * float(n_experts)


def dispatch(x: Tensor, topk_idx: ArrayLike, n_experts: int) -> tuple[Tensor, NDArray, NDArray]:
    idx = _ids(topk_idx, n_experts)
    if x.ndim != 2 or x.shape[0] != idx.shape[0]:
        raise ValueError(f"x must be [N, d] with N = {idx.shape[0]}, got {x.shape}")
    k = idx.shape[1]
    perm = np.argsort(idx.ravel(), kind="stable")  # assignment n*k + j, grouped by expert
    counts = np.bincount(idx.ravel(), minlength=n_experts)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size)
    # Assignment a belongs to token a // k.
    return x[perm // k], offsets, inv_perm.astype(np.int64)


def combine(y_sorted: Tensor, topk_w: Tensor, inv_perm: ArrayLike) -> Tensor:
    N, k = topk_w.shape
    y = y_sorted[np.asarray(inv_perm)]  # back to assignment order n*k + j
    y = F.reshape(y, (N, k, y_sorted.shape[-1]))
    return F.sum(y * F.reshape(topk_w, (N, k, 1)), axis=1)


class MoE(Module):
    def __init__(
        self, d: int, d_ff_expert: int, n_experts: int, top_k: int, n_shared: int = 0,
        router: Literal["softmax_topk", "topk_softmax", "sigmoid"] = "softmax_topk", norm_topk: bool = True,
        aux_loss_coef: float = 0.01, bias_update_rate: float = 0.0, routed_scaling: float = 1.0,
        act: Literal["silu", "gelu_tanh"] = "silu", rng: Any = None,
    ) -> None:
        super().__init__()
        if min(d, d_ff_expert, n_experts) < 1:
            raise ValueError(f"sizes must be >= 1, got d={d}, f={d_ff_expert}, E={n_experts}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must lie in [1, {n_experts}], got {top_k}")
        if n_shared < 0:
            raise ValueError(f"n_shared must be >= 0, got {n_shared}")
        if router not in ROUTERS:
            raise ValueError(f"router must be one of {ROUTERS}, got {router!r}")
        if aux_loss_coef < 0 or bias_update_rate < 0 or not routed_scaling > 0:
            raise ValueError("aux_loss_coef and bias_update_rate must be >= 0, routed_scaling > 0")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.n_experts, self.top_k, self.router = n_experts, top_k, router
        self.norm_topk, self.aux_loss_coef = bool(norm_topk), float(aux_loss_coef)
        self.bias_update_rate, self.routed_scaling = float(bias_update_rate), float(routed_scaling)
        self.gate = Linear(d, n_experts, bias=False, rng=r)
        self.experts = ModuleList([GatedMLP(d, d_ff_expert, act=act, rng=r) for _ in range(n_experts)])
        self.shared_experts = GatedMLP(d, n_shared * d_ff_expert, act=act, rng=r) if n_shared else None
        # Plain state (no requires_grad): the aux-free bias is updated by a rule,
        # not by the optimizer.
        self.e_score_correction_bias = np.zeros(n_experts, dtype=np.float32)
        # What the last forward measured. Kept in a dict: a Tensor that
        # requires grad assigned as an attribute would register as a parameter.
        self._last: dict[str, Any] = {}

    @property
    def aux_loss(self) -> Optional[Tensor]:
        return self._last.get("aux")

    @property
    def expert_load(self) -> Optional[NDArray]:
        return self._last.get("load")

    def route(self, x: Tensor) -> tuple[NDArray, Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError(f"route takes x [N, d], got {x.shape}")
        z = self.gate(x)
        k = self.top_k
        if self.router == "sigmoid":
            s = F.sigmoid(z)
            idx = topk_ids(s.data + self.e_score_correction_bias, k)  # the bias chooses ...
            w = F.gather(s, idx, axis=-1)  # ... the unbiased score weighs
            if self.norm_topk:
                w = w / (F.sum(w, axis=-1, keepdims=True) + 1e-20)
            probs = s / F.sum(s, axis=-1, keepdims=True)
        elif self.router == "softmax_topk":
            probs = F.softmax(z, axis=-1)
            idx = topk_ids(probs.data, k)
            w = F.gather(probs, idx, axis=-1)
            if self.norm_topk:
                w = w / F.sum(w, axis=-1, keepdims=True)
        else:  # topk_softmax
            probs = F.softmax(z, axis=-1)
            idx = topk_ids(z.data, k)
            w = F.softmax(F.gather(z, idx, axis=-1), axis=-1)
        if self.routed_scaling != 1.0:
            w = w * self.routed_scaling
        aux = load_balance_loss(probs, idx, self.n_experts) * self.aux_loss_coef
        return idx, w, aux

    def forward(self, x: Tensor) -> Tensor:
        lead, d = x.shape[:-1], x.shape[-1]
        xf = F.reshape(x, (-1, d))
        idx, w, aux = self.route(xf)
        xs, offsets, inv_perm = dispatch(xf, idx, self.n_experts)
        parts = []
        for e in range(self.n_experts):
            a, b = int(offsets[e]), int(offsets[e + 1])
            if b > a:  # an expert with no token does no work
                parts.append(self.experts[e](xs[a:b]))
        y = combine(F.concat(parts, axis=0), w, inv_perm)
        if self.shared_experts is not None:
            y = y + self.shared_experts(xf)
        self._last = {"aux": aux, "load": np.bincount(idx.ravel(), minlength=self.n_experts).astype(np.int64)}
        return F.reshape(y, lead + (d,))

    def update_bias(self, expert_load: ArrayLike) -> None:
        load = np.asarray(expert_load, dtype=np.float64)
        if load.shape != (self.n_experts,):
            raise ValueError(f"expert_load must be [E] = [{self.n_experts}], got {load.shape}")
        # Overloaded experts (load above the mean) get a lower bias, so the
        # router picks them less next step; underloaded ones a higher bias.
        step = self.bias_update_rate * np.sign(load.mean() - load)
        self.e_score_correction_bias = (self.e_score_correction_bias + step).astype(np.float32)
