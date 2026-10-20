# contracts/py/tinyllm/eval/zoo.pyi (L6.7): the model zoo (D36)
# chapter: content/courses/ai-engineering/12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md
#
# The manifest (zoo.json): {"suite": "zoo", "models": [entry, ...]}; paths in
# an entry are relative to the manifest's directory. Every entry has "id",
# "dir" (a model directory, or the n-gram's .safetensors file), and "task"
# (a name for the report). The rest depends on the family's kind:
#
#   kind        tl_arch                          entry fields      metric (95% CI)
#   lm          bigram ngram nplm rnnlm gpt      data: tokens .bin  bpb (M07.4 mean_ci of the per-token NLL)
#                                                (ctx_len, stride for gpt; n_bytes unless the tokenizer is
#                                                bytes: entry "tokenizer", else tl_tokenizer, else
#                                                bytes when vocab_size is 256, as for a bare n-gram file)
#   seq2seq     seq2seq transformer              data: JSONL {"src": [ids], "tgt": [ids]},
#                                                bos, eos, beam (4), max_len, pad_id (0)
#                                                                   em (Wilson)
#   classifier  any config with "tl_head" (L6.5) data: split<TAB>sentence<TAB>label, split ("val"),
#                                                cls_id (1), sep_id (2), pad_id (0)
#                                                                   accuracy (Wilson)
#   mlm         bert                             as classifier, mask_id (3), p (0.15)
#                                                                   accuracy of masked tokens (Wilson)
#   rtd         electra                          as classifier, p (0.15); the mask id and
#                                                specials from config.json
#                                                                   replaced-token detection accuracy (Wilson)
#   embeddings  word2vec                         data: w1<TAB>w2<TAB>score lines
#                                                                   spearman (Fisher z)
#
# Text rows are encoded with the byte tokenizer: [cls] + UTF-8 bytes + [sep],
# cut to the model's max length, right-padded with pad_id (D32: the control
# bytes 0 to 4 are free for specials). Sequence-to-sequence EM compares the
# best hypothesis without its eos with "tgt" through L4.5's exact_match on
# space-joined ids; beam search is L4.4's (seq2seq) or L5.5's translate.
#
# The report (formats/eval-results.schema.json): {"format":
# "tl.eval-results.v1", "suite", "seed", "manifest", "rows"}, one row per
# entry in manifest order. A row that cannot be scored has status "error" (an
# exception, its type and message in "reason") or "skipped" (no scorer for
# its tl_arch) and value null; the other rows are still scored. The random
# draws (bert's masking, electra's corruption) come from rng in row order.
# tl_arch is left out of a row when the schema's enum lacks it (the n-gram).
#
# A word2vec directory (save_word2vec): config.json {"tl_arch": "word2vec",
# "tl_tokenizer": "file", "tl_format": 1, "vocab_size", "hidden_size"},
# vocab.json (the words, in row order), model.safetensors {"embeddings": F32 [V, d]}.
from typing import Any, Callable, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

FORMAT: str  # "tl.eval-results.v1"
FAMILIES: dict[str, tuple[str, str, bool]]  # tl_arch -> (kind, metric, higher_is_better)
SCHEMA_ARCHS: set[str]

def read_config(dir: str) -> dict[str, Any]:
    """config.json of the directory, or the metadata of the n-gram's
    safetensors file (dir itself or dir/model.safetensors). ValueError when
    neither names a tl_arch."""

def kind_of(cfg: dict[str, Any]) -> str:
    """"classifier" when cfg has "tl_head", else FAMILIES[tl_arch][0].
    KeyError ("no zoo scorer for tl_arch ...") for an unknown family."""

def save_word2vec(emb: ArrayLike, vocab: Sequence[str], dir: str) -> None:
    """Write the word2vec directory above. ValueError when the rows and the
    vocabulary disagree."""

def load_model(dir: str) -> Any:
    """The family's own loader by tl_arch: L6.5's load_classifier (with a
    tl_head), BigramLM(weight=bigram.weight), NGramLM.load, load_nplm,
    load_rnnlm, load_gpt, load_seq2seq, load_transformer, load_bert,
    load_electra's model; for word2vec the pair (embeddings, vocab)."""

def lm_token_nlls(model: Any, arch: str, cfg: dict[str, Any], ids: ArrayLike, entry: dict[str, Any]) -> NDArray:
    """Per-token NLLs in nats of a language model on ids [n]: the bigram's
    log-softmax table for tokens 1..n-1; NGramLM.nll (every token, the first
    from <s>); NPLM.nll and RNNLM.nll; for gpt, L6.7's token_nlls with
    ctx_len (default n_ctx) and stride (default ctx_len // 2)."""

def encode_batch(
    texts: Sequence[str], max_len: int, cls_id: int = 1, sep_id: int = 2, pad_id: int = 0
) -> tuple[NDArray, NDArray]:
    """(ids int64 [n, T], real bool [n, T]) as above, T the longest row.
    ValueError for max_len < 3."""

def score_entry(entry: dict[str, Any], base: str, rng: Any) -> dict[str, Any]:
    """One report row (status "ok") for one manifest entry, under no_grad.
    Raises what loading or scoring raises; run_zoo turns that into a row."""

def run_zoo(manifest: str, rng: Any, seed: int = 0) -> dict[str, Any]:
    """The report above for every entry of the manifest file."""

def write_report(report: dict[str, Any], path: str) -> None:
    """The report as JSON at path (parents created)."""
