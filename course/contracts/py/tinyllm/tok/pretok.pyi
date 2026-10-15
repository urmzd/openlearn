# contracts/py/tinyllm/tok/pretok.pyi (L1.2)
# chapter: content/courses/build-an-llm/p01-tokenizers/02-byte-level-bpe.md
#
# The GPT-2 pre-tokenizer written over unicodedata (no regex module), and the
# Hugging Face Digits pre-tokenizer. For every str x, "".join(f(x)) == x and
# no piece is empty.
def is_space(ch: str) -> bool:
    """ch has the Unicode White_Space property (what \\s matches):
    U+0009..U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000..U+200A, U+2028,
    U+2029, U+202F, U+205F, U+3000. Not str.isspace (which adds U+001C..U+001F)."""
def is_letter(ch: str) -> bool:
    """\\p{L}: general category Lu, Ll, Lt, Lm, or Lo."""
def is_number(ch: str) -> bool:
    """\\p{N}: general category Nd, Nl, or No."""
def pretokenize_gpt2(text: str) -> list[str]:
    """Exactly the matches of
    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+
    scanned left to right, alternatives tried in that order."""
def split_digits(text: str, individual: bool = True) -> list[str]:
    """Each \\p{N} code point alone (individual) or each run of them as one
    piece; the text between digits stays whole."""
