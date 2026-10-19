"""A character trie with longest-prefix match (M06.2).

Nodes live in two parallel lists (an arena): _children[i] maps a character to
the index of the child node, and _value[i] is the value of the key that ends
at node i, or None. Node 0 is the root and spells the empty string. Every
walk is a loop, never recursion, so a key as long as a document is fine.

Contract: contracts/py/tinyllm/tok/trie.pyi.
"""

from __future__ import annotations

from collections.abc import Iterator


def _check_start(s: str, start: int) -> None:
    """ValueError unless 0 <= start <= len(s)."""
    if not 0 <= start <= len(s):
        raise ValueError(f"start must lie in [0, {len(s)}], got {start}")


class Trie:
    """Keys are non-empty strings, values are ints."""

    def __init__(self) -> None:
        self._children: list[dict[str, int]] = [{}]
        self._value: list[int | None] = [None]
        self._size = 0

    def insert(self, key: str, value: int) -> None:
        if not key:
            raise ValueError("a trie key must be a non-empty string")
        node = 0
        for ch in key:
            nxt = self._children[node].get(ch)
            if nxt is None:
                nxt = len(self._children)
                self._children.append({})
                self._value.append(None)
                self._children[node][ch] = nxt
            node = nxt
        if self._value[node] is None:
            self._size += 1
        self._value[node] = value

    def _find(self, key: str) -> int | None:
        """The node that spells key, or None when no key starts with it."""
        node = 0
        for ch in key:
            nxt = self._children[node].get(ch)
            if nxt is None:
                return None
            node = nxt
        return node

    def get(self, key: str) -> int | None:
        node = self._find(key)
        return None if node is None else self._value[node]

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return self._size

    def node_count(self) -> int:
        return len(self._children)

    def longest_prefix(self, s: str, start: int = 0) -> tuple[int, int | None]:
        _check_start(s, start)
        best_len, best_val = 0, None
        node = 0
        i = start
        while i < len(s):
            nxt = self._children[node].get(s[i])
            if nxt is None:
                break
            node = nxt
            i += 1
            # Remember the deepest node that ends a key, not the deepest node.
            if self._value[node] is not None:
                best_len, best_val = i - start, self._value[node]
        return best_len, best_val

    def prefixes(self, s: str, start: int = 0) -> Iterator[tuple[int, int]]:
        _check_start(s, start)
        node = 0
        i = start
        while i < len(s):
            nxt = self._children[node].get(s[i])
            if nxt is None:
                return
            node = nxt
            i += 1
            v = self._value[node]
            if v is not None:
                yield i - start, v

    def items(self) -> Iterator[tuple[str, int]]:
        # Preorder with children in increasing character order: a key comes
        # before its extensions and "ab..." before "b...", which is exactly
        # code-point string order. The stack holds (node, depth, edge char);
        # path holds the characters from the root to the current node, so a
        # long chain is never copied once per node.
        path: list[str] = []
        stack: list[tuple[int, int, str]] = [(0, 0, "")]
        while stack:
            node, depth, ch = stack.pop()
            if depth:
                del path[depth - 1 :]
                path.append(ch)
            v = self._value[node]
            if v is not None:
                yield "".join(path), v
            kids = self._children[node]
            for c in sorted(kids, reverse=True):
                stack.append((kids[c], depth + 1, c))
