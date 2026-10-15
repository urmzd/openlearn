# contracts/py/tinyllm/tok/trie.pyi (M06.2)
# chapter: content/tracks/math/06-discrete-math-2/02-trees-and-tries-longest-prefix-match.md
#
# A trie is a rooted tree whose edges are labelled with characters. The path
# from the root to a node spells a string; a node is terminal when that
# string is a key, and then it holds the key's value. Matching walks one edge
# per character of the text, so finding every key that is a prefix of
# s[start:] costs O(length of the longest match), whatever the number of keys.
# WordPiece (L1.3) segments by repeated longest_prefix; the Unigram lattice
# (L1.4) adds one edge per match from prefixes; L1.2 matches added tokens.
from collections.abc import Iterator

class Trie:
    def __init__(self) -> None:
        """An empty trie: one root node, no keys."""

    def insert(self, key: str, value: int) -> None:
        """Make key a key with this value. Inserting an existing key replaces
        its value and does not change len(). A key may be a prefix of another
        key in either insertion order. ValueError for an empty key."""

    def get(self, key: str) -> int | None:
        """The value of key, or None when key is not a key (also when it is
        only a prefix of one)."""

    def __contains__(self, key: str) -> bool:
        """True exactly when get(key) is not None."""

    def __len__(self) -> int:
        """The number of distinct keys."""

    def node_count(self) -> int:
        """The number of nodes, the root included: 1 + the number of distinct
        non-empty prefixes of the keys."""

    def longest_prefix(self, s: str, start: int = 0) -> tuple[int, int | None]:
        """(length, value) of the longest key k with s[start:start + len(k)] == k,
        or (0, None) when no key is a prefix of s[start:]. Characters are
        compared as code points (no normalization, no case folding).
        ValueError unless 0 <= start <= len(s)."""

    def prefixes(self, s: str, start: int = 0) -> Iterator[tuple[int, int]]:
        """Every key that is a prefix of s[start:], as (length, value), shortest
        first. ValueError unless 0 <= start <= len(s) (raised on the first
        next())."""

    def items(self) -> Iterator[tuple[str, int]]:
        """Every (key, value) in increasing code-point order of the key, by a
        preorder walk that visits children in increasing character order.
        Iterative: a key of 100 000 characters does not hit the recursion limit."""
