"""The module system: parameters and children found by attribute (L0.4).

Assigning a Tensor that requires grad, or a Module, as an attribute
registers it under that name. Registration order is assignment order, and
the dotted names (`0.weight`, `fc.bias`) are the safetensors key contract
every checkpoint in the course uses (formats/safetensors.md).

Contract: contracts/py/tinyllm/nn/module.pyi.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray

from tinyllm.autograd.tensor import Tensor


class Module:
    def __init__(self) -> None:
        # object.__setattr__: our own __setattr__ needs these dicts to exist.
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if "_params" not in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__}: call super().__init__() before assigning attributes"
            )
        params, children = self.__dict__["_params"], self.__dict__["_children"]
        if isinstance(value, Tensor) and value.requires_grad:
            children.pop(name, None)
            params[name] = value
        elif isinstance(value, Module):
            params.pop(name, None)
            children[name] = value
        else:
            # Re-assigning a name to plain state (None, a constant Tensor)
            # unregisters it: a Linear built with bias=False has no "bias".
            params.pop(name, None)
            children.pop(name, None)
        object.__setattr__(self, name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not define forward")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        seen: set[int] = set()
        out: list[tuple[str, Tensor]] = []

        def walk(pre: str, m: Module) -> None:
            for name, p in m._params.items():
                if id(p) not in seen:  # a tied weight is listed once, first name wins
                    seen.add(id(p))
                    out.append((pre + name, p))
            for name, c in m._children.items():
                walk(pre + name + ".", c)

        # Depth first, own parameters before children, in registration order.
        walk(prefix, self)
        return iter(out)

    def parameters(self) -> Iterator[Tensor]:
        return (p for _, p in self.named_parameters())

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        out = [(prefix, self)]
        for name, c in self._children.items():
            out.extend(c.named_modules(prefix + name if not prefix else f"{prefix}.{name}"))
        return iter(out)

    def train(self, mode: bool = True) -> "Module":
        for _, m in self.named_modules():
            object.__setattr__(m, "training", bool(mode))
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def state_dict(self) -> dict[str, NDArray]:
        # Copies: a saved state must not change when training continues.
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True) -> None:
        own = dict(self.named_parameters())
        if strict:
            missing = sorted(set(own) - set(sd))
            unexpected = sorted(set(sd) - set(own))
            if missing or unexpected:
                raise KeyError(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
        arrays = {}
        for name, p in own.items():
            if name not in sd:
                continue
            a = np.asarray(sd[name])
            if a.shape != p.shape:
                raise ValueError(f"{name}: shape {a.shape} in the state dict, {p.shape} in the model")
            arrays[name] = a
        for name, a in arrays.items():
            # In place: the optimizer holds these Tensors and their arrays.
            own[name].data[...] = a.astype(own[name].data.dtype, copy=False)
