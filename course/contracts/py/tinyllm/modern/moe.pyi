# contracts/py/tinyllm/modern/moe.pyi (L7.8): mixture of experts
# chapter: content/courses/ai-engineering/13-decoder-block/08-mixture-of-experts.md
#
# A router scores E experts (L7.2 gated MLPs) for every token and sends the
# token to its top k; the output is the weighted sum of those k experts'
# outputs (plus shared experts every token uses). Parameters grow with E,
# compute per token with k.
#
# Symbols: N tokens (all leading axes flattened), d width, E n_experts,
# k top_k, f d_ff_expert, S n_shared, z = router logits x W_g^T [N, E].
#
# Routers (topk: the k largest, ties to the LOWEST expert id):
#   "softmax_topk"  p = softmax(z); idx = topk(p); w = p[idx];
#                   norm_topk: w /= sum(w)          (Mixtral, Qwen-MoE)
#   "topk_softmax"  idx = topk(z); w = softmax(z[idx]) (always sums to 1;
#                   norm_topk is ignored)          (Switch-style)
#   "sigmoid"       s = sigmoid(z); idx = topk(s + b), b the aux-free bias;
#                   w = s[idx] (the bias chooses, it never weighs);
#                   norm_topk: w /= sum(w) + 1e-20 (DeepSeek-V3)
#   then w *= routed_scaling. The router logits and the softmax/sigmoid are
#   computed in float32 at least.
#
# Load-balancing loss (Switch Transformer eq. 4 to 6 as Hugging Face's
# load_balancing_loss_func computes it, one layer, no padding mask):
#     f_e = (number of the N k assignments that went to e) / N
#     P_e = mean over tokens of router_probs[:, e]
#     L   = E * sum_e f_e P_e          (k at perfect balance, up to E k when
#                                       every token picks the same experts)
# f is a count and has no gradient; the gradient reaches the router through P.
# router_probs is softmax(z) for the softmax routers and s / sum(s) for
# "sigmoid".
#
# Aux-loss-free balancing (DeepSeek-V3): after each step,
#     b_e += bias_update_rate * sign(mean(load) - load_e)
# so an overloaded expert's bias falls and it is chosen less; b is not a
# trained parameter (it is plain state, not in state_dict).
#
# Sorted dispatch: the N k (token, slot) assignments, flattened as n k + j,
# are sorted by expert (stable), so each expert reads one contiguous slice:
#     perm      = stable argsort of topk_idx.ravel()       [N k]
#     x_sorted  = x[perm // k]                              [N k, d]
#     offsets   = [0, c_0, c_0 + c_1, ..., N k]             [E + 1], c_e = count of e
#     inv_perm  = the inverse permutation of perm           [N k]
# combine puts each output back at its assignment (y_sorted[inv_perm]) and
# sums the k slots of every token weighted by topk_w.
#
# Parameters, float32, in registration order (state_dict keys are DeepSeek's
# and Qwen-MoE's on the Hub), drawn from one rng (PCG32, M06.3; None means
# PCG32(0).substream("init")) in this order:
#   gate.weight                    [E, d]   (a Linear without bias)
#   experts.<e>.{gate,up,down}_proj.weight   GatedMLP(d, f) for e = 0 .. E - 1
#   shared_experts.{gate,up,down}_proj.weight GatedMLP(d, S f), only when S > 0
from typing import Any, Literal, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.mlp import GatedMLP
from tinyllm.nn.layers import Linear, ModuleList
from tinyllm.nn.module import Module

def topk_ids(scores: ArrayLike, k: int) -> NDArray:
    """int64 [N, k]: per row the ids of the k largest scores, largest first,
    ties to the lowest id. ValueError unless 1 <= k <= scores.shape[-1]."""

def load_balance_loss(router_probs: Tensor, topk_idx: ArrayLike, n_experts: int) -> Tensor:
    """The scalar L above. router_probs [N, E], topk_idx [N, k] integers.
    ValueError for shapes that disagree or an id outside [0, E)."""

def dispatch(x: Tensor, topk_idx: ArrayLike, n_experts: int) -> tuple[Tensor, NDArray, NDArray]:
    """(x_sorted [N k, d], offsets int64 [E + 1], inv_perm int64 [N k]) as
    above. Tokens keep their order within each expert's slice (stable).
    Gradients of x_sorted flow back to x (summed over a token's k copies).
    ValueError for an id outside [0, E) or shapes that disagree."""

def combine(y_sorted: Tensor, topk_w: Tensor, inv_perm: ArrayLike) -> Tensor:
    """[N, d]: y[n] = sum_j topk_w[n, j] * y_sorted[inv_perm[n k + j]].
    Gradients reach y_sorted and topk_w."""

class MoE(Module):
    gate: Linear
    experts: ModuleList
    shared_experts: Optional[GatedMLP]
    e_score_correction_bias: NDArray  # [E] float32, zeros at construction
    n_experts: int
    top_k: int
    router: str
    @property
    def aux_loss(self) -> Optional[Tensor]:
        """aux_loss_coef * load_balance_loss of the last forward (None before
        one). Read-only, and never a registered parameter: it is not in
        named_parameters() or state_dict()."""
    @property
    def expert_load(self) -> Optional[NDArray]:
        """int64 [E]: assignments per expert in the last forward."""

    def __init__(
        self,
        d: int,
        d_ff_expert: int,
        n_experts: int,
        top_k: int,
        n_shared: int = 0,
        router: Literal["softmax_topk", "topk_softmax", "sigmoid"] = "softmax_topk",
        norm_topk: bool = True,
        aux_loss_coef: float = 0.01,
        bias_update_rate: float = 0.0,
        routed_scaling: float = 1.0,
        act: Literal["silu", "gelu_tanh"] = "silu",
        rng: Any = None,
    ) -> None:
        """ValueError when d, d_ff_expert, or n_experts < 1, top_k outside
        [1, n_experts], n_shared < 0, an unknown router, a negative
        coefficient or rate, or routed_scaling <= 0."""

    def route(self, x: Tensor) -> tuple[NDArray, Tensor, Tensor]:
        """x [N, d] -> (topk_idx int64 [N, k], topk_w [N, k],
        aux = aux_loss_coef * load_balance_loss(router_probs, topk_idx, E)),
        the router above. topk_w and aux carry gradients to gate.weight
        (and topk_w to x)."""

    def forward(self, x: Tensor) -> Tensor:
        """[..., d] -> [..., d]: route the flattened tokens, run each expert
        on its slice of dispatch(...), combine, and add shared_experts(x)
        when S > 0. Sets aux_loss and expert_load. Backward reaches x, the
        router, and every expert that received a token."""

    def update_bias(self, expert_load: ArrayLike) -> None:
        """b += bias_update_rate * sign(mean(load) - load) (float32).
        ValueError for a load that is not [E]."""
