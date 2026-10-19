# contracts/py/tinyllm/rnn/bi.pyi (L3.4): bidirectional RNNs with length-aware reversal
# chapter: content/courses/build-an-llm/04-recurrent/04-bidirectional-rnn.md
#
# A batch of sequences of different lengths is padded to T steps, time first:
# x [T, B, ...], with sequence b real at t < lengths[b] and padding after.
# The backward direction must read each sequence from ITS last real step to
# its first, so reversing the whole padded tensor (x[::-1]) is wrong: it puts
# the padding first and the backward RNN reads it before the real tokens.
#
# `fwd` and `bwd` are any single-direction recurrent modules with
# forward(x, state=None, lengths=None) -> (out [T, B, H], state) that output 0
# past each length and carry their state unchanged there (L3.2 LSTM, L3.3 GRU).
# This is torch's bidirectional=True: the forward module plays the *_l0
# weights and the backward module the *_l0_reverse weights.
from typing import Any, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

def reversal_index(lengths: ArrayLike, T: int) -> NDArray:
    """int64 [T, B]: idx[t, b] = lengths[b] - 1 - t for t < lengths[b], else t.
    Gathering x[idx[t, b], b] reverses each sequence inside its own length and
    leaves its padding where it was. Applying it twice is the identity.
    ValueError unless lengths is a 1-D integer array with every value in 1..T."""

def reverse_padded(x: Any, lengths: Optional[ArrayLike]) -> Any:
    """x [T, B, ...] (a Tensor, gradient flowing back through the gather, or
    an ndarray) with each sequence reversed inside its length:
    out[t, b] = x[reversal_index(lengths, T)[t, b], b]. lengths None means
    every sequence has length T (a plain flip of the time axis). Returns the
    same kind as x. ValueError as reversal_index, or for x with fewer than 2
    dimensions."""

def bidirectional(
    fwd: Module, bwd: Module, x: Tensor, lengths: Optional[ArrayLike] = None
) -> Tensor:
    """[T, B, H_f + H_b]: the forward module's outputs, then the backward
    module's, concatenated on the last axis (torch's order):
        out_f, _ = fwd(x, lengths=lengths)
        out_r, _ = bwd(reverse_padded(x, lengths), lengths=lengths)
        out = concat([out_f, reverse_padded(out_r, lengths)], axis=-1)
    so out[t, b, H_f:] is the backward state after reading sequence b from its
    last real step down to t. Padded positions are 0 in both halves. Each
    direction's final state is in `out`: out[lengths[b] - 1, b, :H_f] and
    out[0, b, H_f:]. lengths None means every sequence has length T.
    ValueError as reverse_padded."""
