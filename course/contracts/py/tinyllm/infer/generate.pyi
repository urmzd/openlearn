# contracts/py/tinyllm/infer/generate.pyi (L8.2)
# chapter: content/courses/build-an-llm/09-inference/02-kv-cache-and-generate.md
#
# Autoregressive generation: encode the prompt, run the model once over it
# (prefill), then repeatedly sample one token (L8.1) and run the model on
# that token alone against the KV cache (decode), streaming text through an
# incremental UTF-8 detokenizer and stopping at EOS, a stop string, or the
# token budget.
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

from numpy.typing import NDArray

from tinyllm.infer.sample import SamplingParams
from tinyllm.tok.base import Tokenizer

class CausalLM(Protocol):
    """What generate needs from a model (L7.9's LlamaForCausalLM has it):
    forward(ids [B, T] int, positions [T] int64 or None, cache or None)
    returns logits [B, T, V] (a Tensor with .data, or an ndarray). With a
    cache, the model calls cache.update(layer, k, v) once per layer per call
    and attends to what it returns."""

    def forward(self, ids: Any, positions: Optional[Any] = None, cache: Optional[Any] = None) -> Any: ...

def cache_dims(model: Any) -> tuple[int, int, int, int]:
    """(n_layers, n_kv_heads, d_head, max_len) of a model: its attributes of
    those names, or else its `config` (L7.9's LlamaForCausalLM.config) or
    `cfg`, read with the LlamaConfig names (num_hidden_layers,
    num_key_value_heads, head_dim, max_position_embeddings). ValueError
    when none is there."""

class IncrementalDecoder:
    """Streams text from token ids without ever emitting half a character.
    A token may end in the middle of a UTF-8 sequence (byte-level BPE splits
    multi-byte characters), and decoding such a prefix gives U+FFFD. The
    decoder keeps a window [prefix, read) of already-emitted ids as context,
    decodes window + new ids, and emits only the new text, and only when it
    does not end in U+FFFD. Invariant: the concatenation of every push() and
    the final flush() equals tok.decode(all ids)."""

    def __init__(self, tok: Tokenizer) -> None: ...
    def push(self, token_id: int) -> str:
        """Add one id; return the text that is now final ("" while a
        character is incomplete)."""
    def flush(self) -> str:
        """Return whatever is still held back (an incomplete sequence at the
        end decodes to U+FFFD) and reset the window."""

@dataclass
class Generation:
    text: str  # the completion, cut before the first stop string
    ids: list[int]  # every sampled id, EOS excluded
    logprobs: list[float]  # L8.1 logprob of each id in ids
    timings: dict[str, float]  # prefill_s, decode_s, total_s
    stats: dict[str, float]  # prompt_tokens, completion_tokens, mean_entropy, tokens_per_s
    finish_reason: str = "length"  # "stop" (EOS or a stop string) or "length"

def generate(
    model: Any,
    tok: Tokenizer,
    prompt: str,
    p: SamplingParams,
    cache: Any = "contiguous",
    kv_dtype: Any = ...,
    on_text: Optional[Callable[[str], None]] = None,
    eos_ids: Sequence[int] = (),
) -> Generation:
    """Generate up to p.max_tokens tokens after tok.encode(prompt).

    cache: "contiguous" builds a KVCache(cache_dims(model), batch 1,
    kv_dtype, max_len = min(model max_len, prompt + max_tokens)); "none"
    recomputes the full sequence every step (the oracle the cache is checked
    against); any object with update/seq_len()/positions/mask (L8.3's paged
    cache) is used as given and must be empty. "paged" as a string is a
    ValueError: L8.3 passes its cache object.

    Each step samples with L8.1's sample(logits of the last position, p,
    history = ids generated so far, rng, prompt = prompt ids); rng is
    request_rng(p.seed), seed 0 when p.seed is None. An id in eos_ids stops
    generation and is not part of ids or text. A stop string stops it too,
    and text is cut before its first occurrence. on_text receives the text
    as it becomes final, never a character fragment and never a prefix that
    could still turn into a stop string; the pieces concatenate to text.
    Generation stops with finish_reason "length" at max_tokens or when the
    context reaches the model's max_len. ValueError for an empty prompt
    encoding, a prompt longer than max_len, or an unknown cache mode."""
