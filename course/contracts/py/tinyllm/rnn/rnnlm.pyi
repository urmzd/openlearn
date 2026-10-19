# contracts/py/tinyllm/rnn/rnnlm.pyi (L3.6): an RNN language model trained with stateful TBPTT
# chapter: content/courses/build-an-llm/04-recurrent/06-rnn-language-model-stateful-tbptt.md
#
# A recurrent language model reads token ids one at a time and predicts the
# next one from its hidden state:
#
#     e_t      = E[ids_t]                                [d_emb]
#     h_t      = cell(e_t, h_{t-1})                      [d_h] (last layer)
#     logits_t = h_t @ W_out^T + b_out                   [V]
#     loss     = mean_t cross_entropy(logits_t, ids_{t+1})
#
# cell is "lstm" (L3.2's LSTM), "gru" (L3.3's GRU), or "rnn": ElmanRNN below,
# whose whole forward and backward over a sequence is ONE autograd op built on
# L3.1's rnn_forward and rnn_backward (the way cuDNN fuses a recurrent layer).
#
# Token arrays are batch first: ids int [B, T]; logits float32 [B, T, V].
# The recurrent state has the shape of the cell's: rnn and gru h
# [n_layers, B, d_h]; lstm (h, c), each [n_layers, B, d_h].
#
# Parameters, in registration order (= state_dict order = the safetensors keys
# of the model directory), float32:
#   emb.weight                          [V, d_emb]     L0.4 Embedding
#   rnn.<the cell's own names>          per layer k: weight_ih_l{k}, weight_hh_l{k},
#                                       bias_ih_l{k}, bias_hh_l{k} (torch's names)
#   out.weight [V, d_h], out.bias [V]                  L0.4 Linear
# All drawn from one rng (a PCG32, M06.3; None means PCG32(0).substream("init"))
# in that order.
#
# Stateful truncated BPTT (Williams and Peng; Zaremba et al. 2014): the
# token stream is cut into `batch` contiguous lanes, lane b = stream[b * L :
# (b + 1) * L] with L = len(stream) // batch (the tail is dropped). Step j
# reads the window [p, p + k) of every lane (inputs) and [p + 1, p + k + 1)
# (targets), starting at p = 0 and advancing p by k. The hidden state left by
# one window starts the next one, DETACHED (the gradient is cut at the window
# boundary, the state is not). When a lane has fewer than k + 1 tokens left,
# p returns to 0 and the state to zeros (a new epoch).
#
# The model directory (save_rnnlm, load_rnnlm) is what `{tinyllm} train
# rnnlm` writes and the L6.7 zoo loads: config.json (formats/config.schema.json)
#   {"tl_arch": "rnnlm", "tl_tokenizer": <tokenizer>, "vocab_size": V,
#    "tl_format": 1, "tl_cell": cell, "tl_d_emb": d_emb, "hidden_size": d_h,
#    "num_hidden_layers": n_layers}
# and model.safetensors: state_dict() as F32 (L0.6's save_safetensors) with
# metadata {"format": "tinyllm", "tl_arch": "rnnlm"}.
from typing import Any, Literal, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Embedding, Linear
from tinyllm.nn.module import Module

class ElmanRNN(Module):
    # Registered per layer k: weight_ih_l{k} [H, D_k], weight_hh_l{k} [H, H],
    # bias_ih_l{k} [H], bias_hh_l{k} [H] (torch.nn.RNN's names; tanh). Init
    # from rng in that order: weight_ih xavier_uniform((H, D_k), 1) (M07.3),
    # weight_hh orthogonal_init((H, H), 1) (M03.3), biases 0.
    num_layers: int
    hidden_size: int

    def __init__(
        self, d_in: int, d_h: int, num_layers: int = 1, rng: Any = None
    ) -> None:
        """ValueError unless d_in, d_h, num_layers >= 1."""

    def forward(
        self,
        x: Tensor,
        state: Optional[Any] = None,
        lengths: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Tensor]:
        """x [T, B, D] time first, state h0 [num_layers, B, H] (Tensor or array,
        None for zeros). Each layer is one op: forward rnn_forward(x, h0,
        weight_ih^T, weight_hh^T, bias_ih + bias_hh) in float64 (L3.1), output
        cast to x's dtype; backward rnn_backward on the saved cache, giving
        gradients for x, h0, both weights, and both biases (each bias gets the
        same dbh). Returns (out [T, B, H], h_n [num_layers, B, H]).
        ValueError for wrong shapes, or lengths other than None (an LM stream
        has no padding)."""

class RNNLM(Module):
    vocab: int
    d_emb: int
    d_h: int
    cell: str  # "rnn" | "lstm" | "gru"
    n_layers: int
    emb: Embedding
    rnn: Module  # ElmanRNN, L3.2 LSTM, or L3.3 GRU
    out: Linear

    def __init__(
        self,
        vocab: int,
        d_emb: int,
        d_h: int,
        cell: Literal["rnn", "lstm", "gru"],
        n_layers: int = 1,
        rng: Any = None,
    ) -> None:
        """The parameters above. ValueError for a size below 1 or another cell."""

    def init_state(self, batch: int) -> Any:
        """Zeros of the cell's state shape, float32."""

    def forward(
        self, ids: ArrayLike, state: Optional[Any] = None
    ) -> tuple[Tensor, Any]:
        """(logits [B, T, V], new state) for ids int [B, T], starting from
        state (None: zeros). ValueError for ids not [B, T >= 1] integers or
        an id outside [0, V)."""

    def detach_state(self, state: Any) -> Any:
        """The same state with every Tensor detached (the TBPTT cut)."""

    def nll(self, ids: ArrayLike, chunk: int = 256) -> NDArray:
        """float64 [len(ids) - 1]: -log P(ids[t] | ids[:t]) for t = 1 ..
        len(ids) - 1, one lane, run statefully chunk by chunk under no_grad
        (the state carries across chunks, so the result does not depend on
        chunk); log-softmax in float64 of the float32 logits. ValueError for
        ids not 1-D, len(ids) < 2, or chunk < 1."""

    def generate(
        self, prefix: Sequence[int], n: int, temperature: float, seed: int
    ) -> list[int]:
        """Continue prefix (at least one id) with exactly n new ids, returning
        only those. temperature == 0 is greedy (ties to the lowest id) and
        draws nothing; otherwise one PCG32(seed) per call, one uniform() per
        token, and the draw of L0.5's BigramLM.sample on logits / temperature.
        ValueError for an empty prefix, n < 0, or temperature < 0."""

def tbptt_batches(
    stream: ArrayLike, k: int, batch: int
) -> list[tuple[NDArray, NDArray]]:
    """One epoch of windows as above: [(inputs int64 [batch, k], targets int64
    [batch, k]), ...], in order. ValueError unless stream is 1-D integer,
    k >= 1, batch >= 1, and every lane has at least k + 1 tokens."""

def train_tbptt(
    model: RNNLM,
    stream: ArrayLike,
    k: int,
    batch: int,
    opt: Any,
    clip: Optional[float],
    steps: int,
) -> list[float]:
    """Exactly `steps` optimizer steps of stateful TBPTT over tbptt_batches
    (epoch after epoch, the state reset to zeros at each epoch start), each
    L0.5's train_step on the mean cross-entropy (L0.3) of the window with
    `clip` (M10.4 global-norm clipping). Returns each step's loss. The state
    carried into window j + 1 is the detached state after window j.
    ValueError as tbptt_batches, or steps < 0."""

def save_rnnlm(model: RNNLM, dir: str, tokenizer: str = "bytes") -> None:
    """Write the model directory above into dir (created if missing).
    tokenizer is config.json's tl_tokenizer ("bytes" or "file"). ValueError
    for another tokenizer."""

def load_rnnlm(dir: str) -> RNNLM:
    """The RNNLM a save_rnnlm directory holds, its parameters equal to the
    saved ones. ValueError when tl_arch is not "rnnlm" or a key is missing;
    KeyError (strict load_state_dict) when the tensors do not match."""
