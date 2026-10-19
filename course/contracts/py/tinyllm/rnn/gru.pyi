# contracts/py/tinyllm/rnn/gru.pyi (L3.3): the GRU, in torch's gate order
# chapter: content/courses/build-an-llm/04-recurrent/03-gru.md
#
# One step, with x [B, D], state h [B, H], and the three gates stacked in
# torch's order r, z, n (rows 0:H, H:2H, 2H:3H of the weights):
#
#     a = x @ w_ih^T + b_ih                 [B, 3H]   (input part)
#     b = h @ w_hh^T + b_hh                 [B, 3H]   (recurrent part)
#     r = sigmoid(a[:, 0:H] + b[:, 0:H])         reset gate
#     z = sigmoid(a[:, H:2H] + b[:, H:2H])       update gate
#     n = tanh(a[:, 2H:3H] + r * b[:, 2H:3H])    candidate: r scales the
#                                                recurrent part AFTER its bias
#     h' = (1 - z) * n + z * h                   z = 1 keeps the old state
#
# (Cho et al. 2014 apply r to h before the matmul; torch and cuDNN apply it
# after, to h @ w_hn^T + b_hn. Weights trained one way do not load the other.)
#
# Every op is from the op library (L0.2), so autograd does the backward.
# Parameter names, shapes, and registration order are torch.nn.GRU's and
# torch.nn.GRUCell's, so torch state_dicts load by name. Parameters are
# float32.
#
# Initialization (one `rng`, a PCG32 of M06.3; None means
# PCG32(0).substream("init")), in registration order, for each layer: the
# three gate blocks of w_ih, each xavier_uniform((H, D_in), gain=1) (M07.3),
# then the three blocks of w_hh, each orthogonal_init((H, H), gain=1)
# (M03.3), stacked in the order r, z, n. Both biases are zero.
#
# Bidirectional runs are tinyllm.rnn.bi.bidirectional (L3.4) over two GRUs.
from typing import Any, Optional

from numpy.typing import ArrayLike

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

def gru_cell(
    x: Tensor,
    h: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> Tensor:
    """One step as above: h'. x [B, D], h [B, H], w_ih [3H, D], w_hh [3H, H],
    b_ih and b_hh [3H]. Arrays and numbers are constants."""

class GRUCell(Module):
    weight_ih: Tensor  # [3H, D]
    weight_hh: Tensor  # [3H, H]
    bias_ih: Tensor  # [3H]
    bias_hh: Tensor  # [3H]

    def __init__(self, d_in: int, d_h: int, rng: Any = None) -> None:
        """ValueError unless d_in >= 1 and d_h >= 1."""
    def forward(self, x: Tensor, h: Optional[Any] = None) -> Tensor:
        """gru_cell with this cell's parameters; h None is zeros of x's dtype."""

class GRU(Module):
    # Registered per layer k, in this order: weight_ih_l{k} [3H, D_k],
    # weight_hh_l{k} [3H, H], bias_ih_l{k} [3H], bias_hh_l{k} [3H], with
    # D_0 = d_in and D_k = H above. (torch.nn.GRU's names.)
    num_layers: int
    hidden_size: int

    def __init__(
        self,
        d_in: int,
        d_h: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        rng: Any = None,
    ) -> None:
        """Dropout with probability `dropout` (L0.4 Dropout, its own default
        stream) on each layer's output sequence except the last layer's, in
        training mode only, as torch. ValueError unless d_in, d_h,
        num_layers >= 1 and 0 <= dropout <= 1."""

    def forward(
        self,
        x: Tensor,
        state: Optional[Any] = None,
        lengths: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Tensor]:
        """x [T, B, d_in] (time first). state h0 [num_layers, B, H] (a Tensor
        or an array), or None for zeros. Returns (out [T, B, H], h_n
        [num_layers, B, H]), out being the last layer's h at every step.

        lengths [B] (integers in 1..T) marks the valid prefix of each
        sequence, as torch's pack_padded_sequence: at a step t >= lengths[b]
        the state of sequence b is carried unchanged and out[t, b] is 0, so
        h_n holds each sequence's state after its own last step. None means
        every sequence has length T. ValueError for a wrong x or state shape,
        or lengths outside 1..T."""
