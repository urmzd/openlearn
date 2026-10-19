# contracts/py/tinyllm/lm/nplm.pyi (L2.2): Bengio's neural probabilistic language model
# chapter: content/courses/build-an-llm/03-statistical-lm/02-bengio-nplm.md
#
# Bengio, Ducharme, Vincent, and Jauvin (JMLR 2003). The next token is
# predicted from a fixed window of the `context` = n - 1 tokens before it.
# Each is looked up in one shared embedding table C, the rows are
# concatenated oldest first, and one tanh hidden layer plus an optional
# direct linear path give the logits:
#
#     x = [C[w_{t-n+1}], ..., C[w_{t-1}]]            float32 [B, context * d_emb]
#     y = b + W x + U tanh(d + H x)                   float32 [B, V]
#     P(w_t = v | window) = softmax(y)[v]
#
# Parameters, in registration order (= state_dict order = the safetensors
# keys of the model directory), all float32, built from L0.4's layers:
#
#   emb.weight     C  [V, d_emb]                  Embedding(V, d_emb)
#   hidden.weight  H  [d_hidden, context*d_emb]   Linear(context*d_emb, d_hidden)
#   hidden.bias    d  [d_hidden]
#   out.weight     U  [V, d_hidden]               Linear(d_hidden, V)
#   out.bias       b  [V]
#   direct.weight  W  [V, context*d_emb]          Linear(context*d_emb, V, bias=False),
#                                                  only when direct
#
# Every initializer draws from ONE generator `rng` (a PCG32, M06.3), passed to
# the layers in that order; None means PCG32(0).substream("init")
# (spec/pcg32.md). A window is a row of `context` token ids, oldest first;
# ids are integers in [0, V).
#
# The model directory (`save_nplm`, `load_nplm`) is what `{tinyllm} lm train
# nplm` writes and the L6.7 zoo loads: config.json (formats/config.schema.json)
#   {"tl_arch": "nplm", "tl_tokenizer": <tokenizer>, "vocab_size": V,
#    "tl_format": 1, "tl_context": context, "tl_d_emb": d_emb,
#    "hidden_size": d_hidden, "tl_direct": direct}
# and model.safetensors: state_dict() as F32 (L0.6's save_safetensors), with
# the metadata {"format": "tinyllm", "tl_arch": "nplm"}.
from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Embedding, Linear
from tinyllm.nn.module import Module

def windows(ids: ArrayLike, context: int) -> tuple[NDArray, NDArray]:
    """Every training example of one token sequence: (ctx int64
    [N, context], targets int64 [N]) with N = len(ids) - context, row i =
    ids[i : i + context] and target ids[i + context]. The first `context`
    tokens are never targets (they have no full window). ValueError when
    ids is not 1-D integer, context < 1, or len(ids) <= context."""

class NPLM(Module):
    vocab: int
    context: int
    d_emb: int
    d_hidden: int
    use_direct: bool
    emb: Embedding
    hidden: Linear
    out: Linear
    direct: Optional[Linear]  # None when direct is False

    def __init__(
        self,
        vocab: int,
        context: int,
        d_emb: int,
        d_hidden: int,
        direct: bool = True,
        rng: Any = None,
    ) -> None:
        """The parameters above, drawn from rng in registration order.
        ValueError when vocab, context, d_emb, or d_hidden is below 1."""

    def forward(self, ctx_ids: ArrayLike) -> Tensor:
        """Logits float32 [B, V] for windows ctx_ids [B, context], built from
        the op library (F.embedding, F.reshape, F.tanh, the Linears) so
        backward reaches every parameter, C included. ValueError for a
        ctx_ids that is not [B, context] or an id outside [0, V)."""

    def nll(self, ids: ArrayLike, batch_size: int = 1024) -> NDArray:
        """float64 [len(ids) - context]: -log P(ids[t] | ids[t-context:t])
        for t = context .. len(ids) - 1, under no_grad, in batches of
        batch_size windows; log-softmax in float64 of the float32 logits.
        ValueError as windows."""

    def perplexity(self, ids: ArrayLike) -> float:
        """exp(mean nll(ids)), summed with M11.2's NLLAccumulator."""

    def generate(
        self, prefix: Sequence[int], n: int, temperature: float, seed: int
    ) -> list[int]:
        """Continue prefix with exactly n new ids and return only the new ids;
        each step conditions on the last `context` ids. temperature == 0 is
        greedy (highest logit, ties to the lowest id) and draws nothing.
        Otherwise one PCG32(seed) (stream 54) per call, one uniform() per
        token, and the draw of L0.5's BigramLM.sample: weights
        w_j = exp(z_j - max z) of z = logits / temperature in float64, summed
        in id order, and the next id is the first j whose running sum exceeds
        u * sum(w). The same seed gives the same ids. ValueError when
        len(prefix) < context, n < 0, or temperature < 0."""

def train_nplm(
    model: NPLM,
    ids: ArrayLike,
    steps: int,
    batch_size: int,
    lr: float,
    rng: Any,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    clip: Optional[float] = None,
) -> list[float]:
    """Minimize the mean cross-entropy (L0.3) of the windows of ids
    (windows(ids, model.context)) for exactly `steps` optimizer steps and
    return the loss of each step. Batches come from L0.5's
    DataLoader({"x": ctx, "y": targets}, batch_size, shuffle=True, rng=rng,
    drop_last=True), epoch after epoch (a new permutation each epoch); each
    step is L0.5's train_step with one M10.2 SGD(model.parameters(), lr,
    momentum, weight_decay=weight_decay) created before the first step, and
    the given clip. ValueError when steps < 0 or there are fewer windows
    than batch_size."""

def save_nplm(model: NPLM, dir: str, tokenizer: str = "bytes") -> None:
    """Write the model directory above into dir (created if missing):
    config.json then model.safetensors. tokenizer is config.json's
    tl_tokenizer ("bytes" or "file"). ValueError for another tokenizer."""

def load_nplm(dir: str) -> NPLM:
    """The NPLM a save_nplm directory holds, in training mode, its
    parameters equal to the saved ones. ValueError when config.json's
    tl_arch is not "nplm" or a key is missing; KeyError (strict
    load_state_dict) when the tensors do not match the config."""
