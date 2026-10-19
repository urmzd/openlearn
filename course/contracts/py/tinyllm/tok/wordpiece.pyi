# contracts/py/tinyllm/tok/wordpiece.pyi (L1.3)
# chapter: content/courses/build-an-llm/02-tokenizers/03-wordpiece.md
#
# BERT's tokenizer: the basic tokenizer (Hugging Face BertNormalizer and
# BertPreTokenizer) and greedy longest-match WordPiece. Ids equal
# bert-base-uncased's (formats/tokenizer.md, WordPiece). Uses M06.2
# (tinyllm.tok.trie) for the longest-prefix search.
from typing import Mapping, Optional, Sequence

from tinyllm.tok.base import Tokenizer

def is_bert_whitespace(ch: str) -> bool:
    """Unicode White_Space (tab, newline, and carriage return included)."""
def is_bert_control(ch: str) -> bool:
    """General category C* (Cc, Cf, Cs, Co, Cn), except tab, newline, carriage return."""
def is_bert_punctuation(ch: str) -> bool:
    """ASCII punctuation (33-47, 58-64, 91-96, 123-126) or category P*."""
def is_cjk(ch: str) -> bool:
    """In a CJK ideograph block: 4E00-9FFF, 3400-4DBF, 20000-2A6DF,
    2A700-2B73F, 2B740-2B81F, 2B920-2CEAF, F900-FAFF, 2F800-2FA1F."""

class WordPieceTokenizer(Tokenizer):
    vocab: dict[str, int]
    def __init__(
        self,
        vocab: Mapping[str, int],
        unk_token: str = "[UNK]",
        prefix: str = "##",
        max_input_chars_per_word: int = 100,
        lowercase: bool = True,
        strip_accents: Optional[bool] = None,
    ) -> None:
        """special_ids = whichever of [PAD] [UNK] [CLS] [SEP] [MASK] are in
        vocab. ValueError when unk_token is missing or ids have gaps."""
    @classmethod
    def from_vocab(
        cls, vocab_txt: str, lowercase: bool = True, strip_accents: Optional[bool] = None
    ) -> "WordPieceTokenizer":
        """vocab.txt: one token per line, id = line number from 0."""
    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "WordPieceTokenizer":
        """A BERT tokenizer.json (BertNormalizer with clean_text and
        handle_chinese_chars, BertPreTokenizer, WordPiece; model.type may be
        absent). Special added tokens become special_ids."""
    def normalize(self, text: str) -> str:
        """Clean (drop U+0000, U+FFFD, controls; white space -> U+0020), space
        around CJK ideographs, strip accents (NFD then drop category Mn) when
        strip_accents, or when it is None and lowercase; then lowercase each
        code point."""
    def basic_tokenize(self, text: str) -> list[str]:
        """normalize, split on white space, every punctuation character alone."""
    def wordpiece(self, word: str) -> list[int]:
        """Greedy longest match: the longest vocab entry starting the word,
        then the longest "##" entry continuing it, ...; [UNK] alone when some
        position has no match or the word exceeds max_input_chars_per_word."""
    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Special tokens in the raw text ([MASK], leftmost longest, case
        sensitive) are one id each; the rest is basic_tokenize + wordpiece.
        add_special wraps the ids as [CLS] ids [SEP]."""
    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """The Hugging Face WordPiece decoder with cleanup: tokens after the
        first get a leading space unless they start with "##" (dropped); then
        " ." " ?" " !" " ," " ' " " n't" " 'm" " do not" " 's" " 've" " 're"
        are replaced inside each token; the tokens are joined."""
    def token_to_id(self, s: str) -> Optional[int]: ...
    def id_to_token(self, i: int) -> str: ...
    def save(self, dir: str) -> None:
        """dir/tokenizer.json (BertNormalizer, BertPreTokenizer, WordPiece,
        TemplateProcessing [CLS] $A [SEP], WordPiece decoder)."""
    @classmethod
    def load(cls, dir: str) -> "WordPieceTokenizer": ...
