# contracts/py/tinyllm/eval/lm.pyi (L6.7): the language-model evaluation harness
# chapter: content/courses/ai-engineering/12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md
#
# A model here is anything callable as model(ids int64 [1, T]) that returns
# logits [1, T, V] (a Tensor or an ndarray), or a tuple whose first item is
# those logits (L6.1's GPT returns (logits, loss)); position t predicts token
# t + 1. log_probs runs it under no_grad and takes a float64 log-softmax.
#
# Strided perplexity of ids [n] with window ctx_len and stride s
# (1 <= s < ctx_len): windows start at 0, s, 2 s, ...; window [b, e) with
# e = min(b + ctx_len, n) scores the tokens t in [max(e_prev, 1), e), each
# from the window's own log-probabilities at position t - 1 - b; it stops at
# the first window with e = n. Every token 1..n-1 is scored exactly once and
# sees at least ctx_len - s tokens of context (fewer only near the start).
# s = ctx_len - 1 is the cheapest valid setting; s = 1 gives every token the
# full window.
#
# Multiple choice (the lm-evaluation-harness recipe): for each choice c, the
# log-likelihood sum_j log p(c_j | context, c_<j) of its tokens after the
# context's tokens; "acc" takes the argmax of the sums, "acc_norm" of the sums
# divided by the choice's length in UTF-8 bytes. Ties go to the first choice.
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

@dataclass
class Result:
    value: float  # the metric
    lo: float  # 95% confidence interval
    hi: float
    n: int  # items
    per_item: NDArray  # float64 [n], one score per item (0 or 1 for accuracy)

class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        """The UTF-8 bytes of text (the byte tokenizer, D32)."""

def log_probs(model: Any, ids: ArrayLike) -> NDArray:
    """float64 [T, V]: log-softmax of the model's logits for ids [T].
    ValueError when the model returns another number of positions."""

def token_nlls(model: Any, ids: ArrayLike, ctx_len: int, stride: int) -> NDArray:
    """float64 [n - 1]: -log p(ids[t] | its window's context) for t = 1 ..
    n - 1, as above. ValueError for ids not 1-D with n >= 2, or unless
    1 <= stride < ctx_len."""

def eval_ppl(model: Any, ids: ArrayLike, ctx_len: int, stride: int, n_bytes: int = 0) -> dict[str, float]:
    """token_nlls summed with M11.2's NLLAccumulator (n_bytes as given):
    {"nll_sum", "n_tokens", "nll_mean", "ppl", "bits_per_token"}, plus "bpb"
    when n_bytes > 0; with at least 2 tokens, the 95% interval of the mean NLL
    from M07.4's mean_ci turned into "ppl_lo", "ppl_hi" (exp of its ends) and,
    with n_bytes, "bpb_lo", "bpb_hi" (its ends times n_tokens / (n_bytes ln 2))."""

def score_choices(
    model: Any, tok: Any, context: str, choices: Sequence[str], normalize: bool = True, max_len: Optional[int] = None
) -> NDArray:
    """float64 [len(choices)]: each choice's log-likelihood as above, divided
    by its UTF-8 byte length when normalize. tok.encode(str) -> list[int]
    encodes the context and each choice separately; with max_len, a longer
    sequence keeps its last max_len tokens. ValueError for an empty context,
    a choice with no tokens, or a choice that leaves no context in max_len."""

def run_task(
    model: Any, tok: Any, task_jsonl: str, metric: str, rng: Any, n_boot: int = 1000, max_len: Optional[int] = None
) -> Result:
    """One item per JSONL line {"context": str, "choices": [str, ...],
    "label": int}; metric "acc" or "acc_norm". per_item is 1.0 when the
    argmax is the label; value, lo, hi = M07.4's bootstrap_ci(per_item, mean,
    n_boot, 0.05, rng). ValueError for another metric or an empty file."""

def compare(a: Result, b: Result, n_perm: int = 10000, rng: Any = None) -> float:
    """The p-value of M07.5's paired_permutation_test(a.per_item,
    b.per_item, n_perm, rng): small means the two models differ on these
    items. rng None means PCG32(0).substream("sample"). ValueError when the
    two results have different item counts."""
