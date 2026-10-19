# contracts/py/tinyllm/infer/beam.pyi (L4.4): beam search, generic over a step function
# chapter: content/courses/build-an-llm/05-attention-origins/04-beam-search.md
#
# Beam search keeps the beam_size best prefixes (by summed log-probability)
# and extends them one token at a time. It knows nothing about the model:
# the model is a step function
#
#     logits, new_state = step_fn(state, y_prev)
#
#   state      the model's decoder state for k live hypotheses: any nesting of
#              tuples (NamedTuples too), lists, and dicts whose array leaves
#              (anything with a `shape` of at least one dimension: numpy
#              arrays, autograd Tensors) have the hypothesis on axis 0;
#              other leaves (None, numbers, strings) are shared by all rows
#   y_prev     int64 [k]: the last token of each live hypothesis (bos at the
#              first step, where k = 1 and state is init_state)
#   logits     [k, V]: unnormalized scores of the next token (any float
#              dtype, or a Tensor); -inf marks a token that may not follow
#
# Op order of one step (the tests pin every choice):
#
#   1. lp = log_softmax(logits) in float64, per row (M09.2)
#   2. cand[b, v] = logprob(live[b]) + lp[b, v]                 [k, V]
#   3. rank all k * V candidates by cand, highest first; ties go to the
#      lower flat index b * V + v (lower beam, then lower token id)
#   4. keep the first min(beam_size - finished so far, number of finite
#      candidates)
#   5. a kept candidate whose token is eos is finished (it leaves the beam
#      and keeps its slot, so the beam narrows by one); the others are the
#      new live hypotheses, in rank order
#   6. the state rows of the new live hypotheses are their parents' rows:
#      select_state(new_state, parents)
#
# The beam shrinks as hypotheses finish. The search stops when no live
# hypothesis is left (beam_size have finished, or nothing finite remains),
# or after max_len generated tokens: live hypotheses of
# length max_len are then returned unfinished. A finished hypothesis's score
# is logprob / len(tokens) ** length_penalty (0 gives the raw log-probability,
# 1 the mean per token). Ranking during the search uses logprob only.
from dataclasses import dataclass
from typing import Any, Callable

from numpy.typing import ArrayLike, NDArray

@dataclass
class Hypothesis:
    tokens: list[int]  # generated ids after bos; the last is eos when finished
    logprob: float  # sum of log p(token | prefix) over tokens, float64
    score: float  # logprob / len(tokens) ** length_penalty
    finished: bool  # ended with eos (False: cut at max_len)

def select_state(state: Any, idx: ArrayLike) -> Any:
    """The rows idx (int [k']) of every array leaf of state, axis 0
    (leaf[idx]), keeping the nesting: a tuple stays a tuple (a NamedTuple
    the same NamedTuple), a list a list, a dict a dict with the same keys.
    Leaves without a shape, or with shape (), are returned unchanged. Rows may
    repeat (two children of one parent). TypeError for a set or another
    container whose order is undefined."""

def beam_search(
    step_fn: Callable[[Any, NDArray], tuple[Any, Any]],
    init_state: Any,
    bos: int,
    eos: int,
    beam_size: int,
    max_len: int,
    length_penalty: float = 1.0,
) -> list[Hypothesis]:
    """At most beam_size hypotheses, best score first (ties keep the order in
    which they were finalized: by step, then by rank). Calls step_fn at most
    max_len times, never after the last live hypothesis finished. With
    beam_size = 1 it is greedy decoding (argmax, ties to the lowest id);
    with beam_size >= V ** max_len it returns the exact best sequences
    (every sequence is kept). ValueError when beam_size < 1, max_len < 1,
    length_penalty < 0, or logits is not [k, V] for the k live hypotheses."""

def greedy_decode(
    step_fn: Callable[[Any, NDArray], tuple[Any, Any]],
    init_state: Any,
    bos: int,
    eos: int,
    max_len: int,
) -> Hypothesis:
    """beam_search(..., beam_size=1, length_penalty=0.0)[0]: the argmax
    token at every step (ties to the lowest id) until eos or max_len.
    ValueError as beam_search, and when every token at the first step is -inf."""
