"""LoRA: low-rank adapters, PiSSA initialization, and merging (L6.6).

Fine-tuning every weight of a model costs one optimizer state per weight.
LoRA freezes a Linear's weight W [out, in] and learns a low-rank update
instead: W x becomes W x + s B A x with A [r, in], B [out, r], r small, and
s = alpha / r. With B = 0 at the start the adapted model IS the base model.
PiSSA starts from the top-r singular directions of W instead (M03.5): the
adapter holds the principal part of W and the frozen residual holds the rest,
so the first steps move the directions that matter most. Merging folds
s B A back into W, so a fine-tuned model serves with no extra cost.

Contract: contracts/py/tinyllm/obj/lora.pyi.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.linalg.svd import low_rank
from tinyllm.nn.layers import Dropout, Linear
from tinyllm.nn.module import Module

PEFT_PREFIX = "base_model.model."


def _register_frozen(owner: Module, name: str, t: Tensor | None) -> None:
    """Register t under name, then freeze it: L0.4 registers a parameter when
    it is assigned with requires_grad True, and keeps it registered after."""
    if t is None:
        setattr(owner, name, None)
        return
    t.requires_grad = True
    setattr(owner, name, t)
    t.requires_grad = False
    t.grad = None


class LoRALinear(Module):
    def __init__(
        self,
        base: Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        init: Literal["default", "pissa"] = "default",
        rng: Any = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, Linear):
            raise TypeError(f"LoRALinear wraps a Linear, got {type(base).__name__}")
        out_f, in_f = base.weight.shape
        if not 1 <= r <= min(in_f, out_f):
            raise ValueError(f"r must be in 1..{min(in_f, out_f)} for a {out_f} x {in_f} weight, got {r}")
        if not alpha > 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if init not in ("default", "pissa"):
            raise ValueError(f"init must be 'default' or 'pissa', got {init!r}")
        self.in_f, self.out_f, self.r = in_f, out_f, int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        # The base's own Tensors, registered here under the base's names and
        # frozen: the checkpoint keys stay "<name>.weight" and "<name>.bias".
        object.__setattr__(self, "_base", base)
        _register_frozen(self, "weight", base.weight)
        _register_frozen(self, "bias", base.bias)
        self.lora_A = Linear(in_f, self.r, bias=False, rng=rng)
        self.lora_B = Linear(self.r, out_f, bias=False, rng=rng)
        self.lora_B.weight.data[...] = 0.0  # B = 0: the adapted model starts as the base
        self.lora_dropout = Dropout(dropout, rng=rng)
        if init == "pissa":
            # W = U S V^T. low_rank gives balanced factors P = U_r sqrt(S_r),
            # Q = sqrt(S_r) V_r^T with P Q = W_r. Divide each by sqrt(s) so
            # s B A = W_r exactly, and keep only the residual W - W_r frozen.
            W = self.weight.data.astype(np.float64)
            P, Q = low_rank(W, self.r)
            root = math.sqrt(self.scaling)
            self.lora_B.weight.data[...] = (P / root).astype(np.float32)
            self.lora_A.weight.data[...] = (Q / root).astype(np.float32)
            self.weight.data[...] = (W - P @ Q).astype(np.float32)

    def delta_weight(self) -> NDArray:
        A = self.lora_A.weight.data.astype(np.float64)
        B = self.lora_B.weight.data.astype(np.float64)
        return (self.scaling * (B @ A)).astype(np.float32)

    def forward(self, x: Tensor) -> Tensor:
        y = F.matmul(x, F.transpose(self.weight, 0, 1))
        if self.bias is not None:
            y = y + self.bias
        # Dropout on the adapter's input only: the frozen path sees x as it is.
        h = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return y + h * self.scaling


def _resolve(model: Module, name: str) -> tuple[Module, str]:
    """(parent module, attribute name) of the dotted child name."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _lora_modules(model: Module) -> list[tuple[str, LoRALinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def inject_lora(
    model: Module,
    target: Callable[[str, Module], bool],
    r: int,
    alpha: float,
    dropout: float = 0.0,
    init: Literal["default", "pissa"] = "default",
    rng: Any = None,
) -> list[str]:
    # A LoRALinear's own lora_A and lora_B are never adapted again.
    inside = [n + "." for n, _ in _lora_modules(model)]
    chosen = [
        n
        for n, m in list(model.named_modules())
        if n
        and isinstance(m, Linear)
        and not any(n.startswith(p) for p in inside)
        and target(n, m)
    ]
    if not chosen:
        raise ValueError("inject_lora: target matched no Linear")
    # Freeze everything first; the adapters built below are the only
    # trainable parameters (a caller may unfreeze a head afterwards).
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    for n in chosen:
        parent, attr = _resolve(model, n)
        setattr(parent, attr, LoRALinear(getattr(parent, attr), r, alpha, dropout, init, rng))
    return chosen


def merge_lora(model: Module) -> list[str]:
    merged = []
    for n, m in _lora_modules(model):
        base = m._base
        W = m.weight.data.astype(np.float64)
        base.weight.data[...] = (W + m.delta_weight().astype(np.float64)).astype(np.float32)
        parent, attr = _resolve(model, n)
        setattr(parent, attr, base)  # the plain Linear again: no adapter left
        merged.append(n)
    return merged


def lora_state_dict(model: Module) -> dict[str, NDArray]:
    out: dict[str, NDArray] = {}
    for n, m in _lora_modules(model):
        out[f"{PEFT_PREFIX}{n}.lora_A.weight"] = m.lora_A.weight.data.copy()
        out[f"{PEFT_PREFIX}{n}.lora_B.weight"] = m.lora_B.weight.data.copy()
    return out


def load_lora_state_dict(model: Module, sd: Mapping[str, Any]) -> None:
    own = lora_state_dict(model)
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    if missing or unexpected:
        raise KeyError(f"adapter mismatch: missing {missing}, unexpected {unexpected}")
    mods = dict(_lora_modules(model))
    for key in own:
        n, which = key[len(PEFT_PREFIX) :].rsplit(".", 2)[0], key.rsplit(".", 2)[1]
        lin = getattr(mods[n], which)
        a = np.asarray(sd[key])
        if a.shape != lin.weight.shape:
            raise ValueError(f"{key}: shape {a.shape}, the adapter has {lin.weight.shape}")
        lin.weight.data[...] = a.astype(np.float32)


def trainable_fraction(model: Module) -> float:
    total = trainable = 0
    for _, p in model.named_parameters():
        total += p.data.size
        trainable += p.data.size if p.requires_grad else 0
    if total == 0:
        raise ValueError("trainable_fraction: the model has no parameters")
    return trainable / total
