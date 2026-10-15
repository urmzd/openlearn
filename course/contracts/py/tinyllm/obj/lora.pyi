# contracts/py/tinyllm/obj/lora.pyi (L6.6): LoRA with PiSSA init and merge
# chapter: content/courses/build-an-llm/p06-objectives/06-lora-pissa-and-merge.md
#
# Hu et al. (2021), LoRA; Meng et al. (2024), PiSSA. A LoRALinear keeps a
# Linear's weight W [out, in] and bias frozen and adds a trainable low-rank
# update:
#
#   y = x W^T + b + s * B (A (dropout(x)))      s = alpha / r
#   A = lora_A.weight [r, in],  B = lora_B.weight [out, r]
#
# Parameters of a LoRALinear, in registration order (= state_dict order):
#   weight [out, in]   the base Linear's own Tensor, frozen (requires_grad False)
#   bias   [out]       the base's bias, frozen (absent when the base has none)
#   lora_A.weight      L0.4 Linear(in, r, bias=False), built from rng
#   lora_B.weight      L0.4 Linear(r, out, bias=False), built from rng after A
# so a checkpoint of the base loads into the adapted model by name, and the
# keys of a merged model are the base's keys. Freezing is requires_grad False
# on a registered parameter (L0.4 registers at assignment): it stays in
# named_parameters and state_dict, and gets no gradient.
#
# init "default": A as L0.4's Linear init, B = 0, so the adapted model's
# output equals the base's bit for bit.
# init "pissa":  with M03.5's low_rank(W, r) = (P, Q), P Q = W_r the best
# rank-r approximation: B = P / sqrt(s), A = Q / sqrt(s) (so s B A = W_r) and
# weight = W - W_r (the frozen residual), all float32 (computed in float64).
#
# Adapter files use PEFT's key names: "base_model.model.<module name>.lora_A.weight"
# and ".lora_B.weight" (adapter_model.safetensors).
from typing import Any, Callable, Literal, Mapping, Optional

from numpy.typing import NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Dropout, Linear
from tinyllm.nn.module import Module

PEFT_PREFIX: str  # "base_model.model."

class LoRALinear(Module):
    in_f: int
    out_f: int
    r: int
    alpha: float
    scaling: float  # alpha / r
    weight: Tensor
    bias: Optional[Tensor]
    lora_A: Linear
    lora_B: Linear
    lora_dropout: Dropout

    def __init__(
        self,
        base: Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        init: Literal["default", "pissa"] = "default",
        rng: Any = None,
    ) -> None:
        """Wrap base (its Tensors are reused, not copied; pissa rewrites the
        weight's data in place to the residual). TypeError when base is not a
        Linear; ValueError unless 1 <= r <= min(in, out), alpha > 0, init is
        "default" or "pissa", and 0 <= dropout <= 1. rng (a PCG32) builds
        lora_A, then lora_B, then is the Dropout's generator; None as L0.4."""

    def delta_weight(self) -> NDArray:
        """float32 [out, in]: s * B @ A (computed in float64)."""

    def forward(self, x: Tensor) -> Tensor:
        """[..., in] -> [..., out] as above. Dropout (L0.4, active in training
        mode only) applies to the adapter's input, never to the frozen path."""

def inject_lora(
    model: Module,
    target: Callable[[str, Module], bool],
    r: int,
    alpha: float,
    dropout: float = 0.0,
    init: Literal["default", "pissa"] = "default",
    rng: Any = None,
) -> list[str]:
    """Freeze every parameter of model, then replace each Linear child whose
    dotted name n (from named_modules, never "" and never inside a
    LoRALinear) has target(n, module) True by a LoRALinear(module, r, alpha,
    dropout, init, rng), in named_modules order. Returns the replaced names.
    After it the adapters are the only trainable parameters. ValueError when
    target matches nothing."""

def merge_lora(model: Module) -> list[str]:
    """Fold every adapter into its weight, W <- W + s B A (in place, float64
    then float32), and put the original Linear back in its place: the model
    has no LoRALinear left, the same state_dict keys as before inject_lora,
    and the same outputs (within float32 rounding). Parameters stay frozen.
    Returns the merged names (empty when there is nothing to merge)."""

def lora_state_dict(model: Module) -> dict[str, NDArray]:
    """Copies of every adapter's A and B under PEFT names, in named_modules
    order, A before B."""

def load_lora_state_dict(model: Module, sd: Mapping[str, Any]) -> None:
    """Copy adapter weights from a lora_state_dict in place. KeyError naming
    the missing and unexpected keys before anything is copied; ValueError
    for a shape mismatch."""

def trainable_fraction(model: Module) -> float:
    """(elements of parameters with requires_grad) / (elements of all
    parameters). ValueError for a model with no parameters."""
