# contracts/py/tinyllm/rnn/lstm.pyi (L3.2): the LSTM, in torch's gate order
# chapter: content/courses/build-an-llm/p03-recurrent/02-lstm.md
#
# One step, with x [B, D], state h, c [B, H], and the four gates stacked in
# torch's order i, f, g, o (rows 0:H, H:2H, 2H:3H, 3H:4H of the weights):
#
#     z = x @ w_ih^T + b_ih + h @ w_hh^T + b_hh          [B, 4H]
#     i = sigmoid(z[:, 0:H])      input gate
#     f = sigmoid(z[:, H:2H])     forget gate
#     g = tanh(z[:, 2H:3H])       candidate
#     o = sigmoid(z[:, 3H:4H])    output gate
#     c' = f * c + i * g
#     h' = o * tanh(c')
#
# Every op is from the op library (tinyllm.autograd.functional, L0.2), so the
# backward comes from autograd. Parameter names, shapes, and registration
# order are torch.nn.LSTM's and torch.nn.LSTMCell's, so torch state_dicts
# load by name (L0.4 load_state_dict). Parameters are float32.
#
# Initialization (one `rng`, a PCG32 of M06.3; None means
# PCG32(0).substream("init")), in registration order, for each layer: the
# four gate blocks of w_ih, each xavier_uniform((H, D_in), gain=1) (M07.3),
# then the four blocks of w_hh, each orthogonal_init((H, H), gain=1)
# (M03.3), stacked in the order i, f, g, o. Biases are zero except the
# forget block of b_ih, which is 1 (so f starts near sigmoid(1) = 0.73 and
# the cell remembers by default; torch's own init leaves it at random).
#
# Bidirectional runs are tinyllm.rnn.bi.bidirectional (L3.4) over two LSTMs.
from typing import Any, Optional

from numpy.typing import ArrayLike

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

def lstm_cell(
    x: Tensor,
    h: Tensor,
    c: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    """One step as above: (h', c'). x [B, D], h and c [B, H], w_ih [4H, D],
    w_hh [4H, H], b_ih and b_hh [4H]. Arrays and numbers are constants."""

class LSTMCell(Module):
    weight_ih: Tensor  # [4H, D]
    weight_hh: Tensor  # [4H, H]
    bias_ih: Tensor  # [4H]
    bias_hh: Tensor  # [4H]

    def __init__(self, d_in: int, d_h: int, rng: Any = None) -> None:
        """ValueError unless d_in >= 1 and d_h >= 1."""
    def forward(
        self, x: Tensor, state: Optional[tuple[Any, Any]] = None
    ) -> tuple[Tensor, Tensor]:
        """lstm_cell with this cell's parameters; state None is zeros of x's dtype."""

class LSTM(Module):
    # Registered per layer k, in this order: weight_ih_l{k} [4H, D_k],
    # weight_hh_l{k} [4H, H], bias_ih_l{k} [4H], bias_hh_l{k} [4H], with
    # D_0 = d_in and D_k = H above. (torch.nn.LSTM's names.)
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
        state: Optional[tuple[Any, Any]] = None,
        lengths: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """x [T, B, d_in] (time first). state (h0, c0), each
        [num_layers, B, H] (Tensors or arrays), or None for zeros. Returns
        (out [T, B, H], (h_n, c_n) each [num_layers, B, H]), out being the
        last layer's h at every step.

        lengths [B] (integers in 1..T) marks the valid prefix of each
        sequence, as torch's pack_padded_sequence: at a step t >= lengths[b]
        the state of sequence b is carried unchanged and out[t, b] is 0, so
        h_n and c_n hold each sequence's state after its own last step and
        padding never leaks into it. None means every sequence has length T.
        ValueError for a wrong x or state shape, or lengths outside 1..T."""
