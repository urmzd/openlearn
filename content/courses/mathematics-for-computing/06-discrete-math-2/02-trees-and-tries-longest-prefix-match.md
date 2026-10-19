# Trees and tries, longest-prefix match

## Overview

| | |
|---|---|
| **Module** | `M06.2` · build · Python · Pass 3 · about 1 h reading |
| **Solution** | [`trie.py`](../../../../course/solutions/python/tinyllm/tok/trie.py): `Trie` (`insert`, `get`, `__contains__`, `__len__`, `node_count`, `longest_prefix`, `prefixes`, `items`) |
| **Contract** | [`course/contracts/py/tinyllm/tok/trie.pyi`](../../../../course/contracts/py/tinyllm/tok/trie.pyi) |
| **Needs** | no code from earlier modules · reading: `M06.1` graphs, DAGs, and iterative traversal |
| **Used by** | `L1.2` added-token matching in byte-level BPE · `L1.3` WordPiece greedy longest match · `L1.4` the Unigram lattice · later: `ds.07` and `L8.4` generalize it to a radix tree over token ids in Rust |
| **Optional depth** | Sedgewick and Wayne, *Algorithms* (4th ed.), section 5.2 "Tries"; Knuth, *TAOCP* vol. 3, section 6.3 "Digital Searching" |

## Key Takeaways

- A tree with $n$ nodes has $n - 1$ edges and one path from the root to each node; a trie labels those edges with characters, so each node spells the string on its path and a lookup costs one step per character, not one comparison per key.
- Longest-prefix match remembers the deepest node that ends a key, not the deepest node reached.
- A trie stores each shared prefix once: it has exactly $1 + $ (number of distinct non-empty prefixes) nodes.
- A preorder walk that visits children in increasing character order lists the keys in sorted order, and as a loop it survives a key of 100 000 characters.

## How to read this chapter

Sections 1 to 3 build trees and tries from first principles and run longest-prefix match by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Pass 3 gives the system real tokenizers, and three of them ask the same question thousands of times per sentence: which vocabulary entries start at this position of the text? WordPiece (`L1.3`, BERT's tokenizer) takes the longest one and jumps past it. The Unigram tokenizer (`L1.4`) needs every one of them to build its lattice. Byte-level BPE (`L1.2`) must find added tokens such as `<|im_start|>` before it splits anything. The obvious code, `[w for w in vocab if text.startswith(w, i)]`, compares against all 30 000 entries at every position: tokenizing a 1 MB corpus that way is $10^6 \times 3 \cdot 10^4$ string comparisons, hours of Python. A trie answers the same question in a handful of dictionary lookups, however large the vocabulary. This module derives it, on top of the tree vocabulary from `M06.1`.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T$ | a tree: nodes and edges, connected, with no cycle | `Trie` |
| $n$ | the number of nodes of $T$ | `int` |
| $r$ | the root: the one node with no parent | node index 0 |
| $\mathrm{depth}(v)$ | edges on the path from $r$ to $v$ | `int` |
| $K$ | the set of keys (vocabulary entries), non-empty strings | `set[str]` |
| $\mathrm{val}(k)$ | the value stored for key $k$ (a token id) | `int` |
| $s$, $i$ | the text being matched and the start position in it | `str`, `int` |
| $s[i:j]$ | the characters of $s$ at positions $i, \dots, j - 1$ | `str` |
| $P(K)$ | the set of distinct non-empty prefixes of the keys | `set[str]` |
| $m$ | the length of the longest key | `int` |

**Trees.** A tree is a graph that is connected (there is a path between any two nodes) and has no cycle. Two facts follow and are worth proving once (`S-M06b` q6 does): between any two nodes there is exactly one path, and a tree with $n$ nodes has exactly $n - 1$ edges. Pick one node as the root $r$ and every edge gets a direction, away from the root. Each node other than $r$ then has exactly one parent, the next node on its unique path to $r$, and its children are the nodes it is parent of. A node with no children is a leaf. $\mathrm{depth}(v)$ is the number of edges from $r$ to $v$. This is the same parent-and-children structure as the autograd graph in `M06.1`, minus the sharing: in a tree no node has two parents.

**Tries.** A trie (from re*trie*val) for a set of strings $K$ is a rooted tree whose edges carry characters, with two rules: the edges out of one node carry distinct characters, and the path from $r$ to a node $v$ spells a string $\mathrm{str}(v)$ (the root spells the empty string). A node is **terminal** when $\mathrm{str}(v) \in K$, and then it stores $\mathrm{val}(\mathrm{str}(v))$. Because edges out of a node are distinct, each string has at most one node, and the node for $k$ is found by following $k$'s characters one at a time from the root: $\lvert k \rvert$ steps, each a dictionary lookup on the node's children.

**Counting nodes.** Every non-root node spells a non-empty prefix of some key (the path to it is the start of some key's path), and every non-empty prefix of a key is spelled by exactly one node. So $n = 1 + \lvert P(K) \rvert$. Keys that share a start share nodes: `un`, `unbe`, and `unbeliev` together need 8 non-root nodes, not $2 + 4 + 8 = 14$. In the worst case (no sharing) $n = 1 + \sum_{k \in K} \lvert k \rvert$.

**Longest-prefix match.** To find the longest key that is a prefix of $s[i:]$, walk from the root along $s_i, s_{i+1}, \dots$ until a character has no edge or the text ends. Every terminal node the walk passes is a key that is a prefix of $s[i:]$, shortest first, so `prefixes` yields them in the order met, and `longest_prefix` returns the last one. The walk is at most $\min(m, \lvert s \rvert - i)$ steps, independent of $\lvert K \rvert$. The node where the walk stops need not be terminal: the keys `unbe` and `unbeliev` put the walk for `unbelie` on a pass-through node seven edges down, and the answer is the last terminal node, `unbe`. Lengths are counted from $i$, and $i = \lvert s \rvert$ is the empty suffix, which matches nothing.

**Greedy segmentation.** WordPiece splits a word by repeating longest-prefix match: take the longest key at position $i$, emit it, move $i$ forward by its length, repeat; if nothing matches, the word is unknown. This is "maximal munch", and it is not always the segmentation with the fewest pieces, which is one reason `L1.4` uses a lattice and dynamic programming instead.

**Walks without recursion.** A preorder walk visits a node, then each child subtree in turn. Visiting children in increasing character order lists keys in sorted (code point) order, because a key sorts before each of its extensions (`ab` < `abc`) and everything under the edge `a` sorts before everything under `b`. Written recursively, the walk needs one stack frame per level, and Python stops at about 1000 frames; a long run of one byte in a corpus is a deep chain. So the walk keeps its own stack of (node, depth, character) and a path list, as `M06.1`'s toposort keeps its own stack.

**Characters are code points.** Edges compare Python `str` characters, which are Unicode code points. `"é"` (U+00E9) and `"e" + "́"` (e plus a combining accent) look the same and are different keys. Normalization (NFC, NFKC) is the tokenizer's job (`L1.1`), done once before matching, never inside the trie.

## 3. Worked example by hand

Keys: `un` = 1, `unbe` = 2, `unbeliev` = 3, `able` = 4, `a` = 5. The trie (`*` marks a terminal node and its value):

```
(root)
├─ a *5
│  └─ b ─ l ─ e *4
└─ u ─ n *1
       └─ b ─ e *2
              └─ l ─ i ─ e ─ v *3
```

**Nodes.** The distinct non-empty prefixes are `u un unb unbe unbel unbeli unbelie unbeliev` (8) and `a ab abl able` (4), so $n = 1 + 12 = 13$ nodes and 12 edges, as a tree must have.

**Longest prefix of `unbelievable` from position 0.** Walk one character per step and remember the last terminal node:

| step | char | node spells | terminal? | best so far |
|---|---|---|---|---|
| 1 | `u` | `u` | no | (0, None) |
| 2 | `n` | `un` | yes, 1 | (2, 1) |
| 3 | `b` | `unb` | no | (2, 1) |
| 4 | `e` | `unbe` | yes, 2 | (4, 2) |
| 5 to 7 | `l i e` | `unbelie` | no | (4, 2) |
| 8 | `v` | `unbeliev` | yes, 3 | (8, 3) |
| 9 | `a` | no edge `a` under `unbeliev`: stop | | **(8, 3)** |

`prefixes` yields the terminal rows in order: `(2, 1), (4, 2), (8, 3)`.

**From position 8**, the rest is `able`: `a` is terminal (1, 5), then `b`, `l`, `e` reach `able` (4, 4). Longest: (4, 4). The greedy segmentation of `unbelievable` is `unbeliev` + `able`.

**Deepest node is not the answer.** For `unbelie`, the walk ends on the node `unbelie`, seven edges deep, which is not terminal. The answer is the last terminal passed: (4, 2), the key `unbe`.

**Preorder.** Children in character order (`a` before `u`): `a`, `able`, `un`, `unbe`, `unbeliev`, which is `sorted(keys)`.

These numbers are the first two cases in section 4.

## 4. The interface

```python
# python/tinyllm/tok/trie.py
class Trie:
    def __init__(self) -> None
    def insert(self, key: str, value: int) -> None                       # ValueError for ""
    def get(self, key: str) -> int | None
    def __contains__(self, key: str) -> bool
    def __len__(self) -> int                                             # distinct keys
    def node_count(self) -> int                                          # 1 + distinct prefixes
    def longest_prefix(self, s: str, start: int = 0) -> tuple[int, int | None]   # (0, None) if none
    def prefixes(self, s: str, start: int = 0) -> Iterator[tuple[int, int]]      # shortest first
    def items(self) -> Iterator[tuple[str, int]]                         # sorted, iterative
```

The tree can be stored any way; the solution keeps two parallel lists, a children dict and a value per node, with node 0 as the root. `start` outside `[0, len(s)]` is a `ValueError`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example, longest | unit | section 3: (8, 3), (4, 4), 13 nodes, greedy split | pins the definition |
| Hand example, all matches | unit | all matches, shortest first | lattice edges for `L1.4` |
| Deepest key, not deepest node | boundary | `unbelie` gives `unbe` | WordPiece emits only vocabulary pieces |
| Longest, not first | boundary | `a`, `ab`, `abc` against `abcd` | greedy means longest |
| No match | boundary | (0, None) for no match and for an empty trie | callers test the length |
| Start offset and bounds | boundary | lengths from `start`, `start == len(s)`, out of range raises | matching from every position of one string |
| Value 0 | boundary | token id 0 is found everywhere | id 0 is a real token |
| Code points, not normalized | unit | `é` and `e` + U+0301 are different keys | normalization belongs to `L1.1` |
| Brute force | differential | random keys and texts against a scan of every key | the trie answers the obvious question |
| A prefix inserted later | boundary | `ab` inserted after `abc` | vocabularies arrive in any order |
| Re-insert | unit | same `len`, same nodes, new value | an added token overrides a learned one |
| A prefix is not a key | boundary | `unbel` exists as a node, is not a key | no pieces outside the vocabulary |
| Empty key | boundary | `insert("")` raises | a zero-length match loops a segmenter forever |
| Node count | property | $n = 1 + \lvert P(K) \rvert$ on random keys | shared prefixes are stored once |
| Sorted items | property | `items()` equals `sorted(keys.items())` | deterministic vocabulary files |
| Very deep key | boundary | a 100 000 character key | no recursion limit |
| Greedy segmentation | golden | greedy splits of 26 strings from a brute-force oracle | WordPiece in `L1.3` |
| Lattice | golden | every match at every position | the Unigram lattice in `L1.4` |

## 5. The solution, step by step

### From first principles

1. A lookup must cost one step per character of the text, not one comparison per key (section 1). So the structure is a tree whose edges carry characters and whose paths spell strings: a trie (2, "Tries").
2. Each node needs a map from character to child. A Python `dict` per node gives an $O(1)$ step with any alphabet, including all of Unicode.
3. A node must also say whether its string is a key and, if so, with which value. The value is an `int` and 0 is a real token id, so "not a key" must be a separate sentinel, `None`, and every test is `is not None`.
4. Nodes are stored in two parallel lists indexed by integers (an arena) instead of as objects pointing at objects. A node is then just an index, `node_count` is the length of a list, and no node ever needs a parent pointer, because every walk goes downward from the root.
5. Longest-prefix match walks the text from `start` and remembers the last terminal node passed (the worked example's "best so far" column), because the node where the walk stops need not be a key. `prefixes` is the same walk yielding each terminal node as it is met, so matches come out shortest first without collecting or sorting.
6. `items` must be iterative (keys can be 100 000 characters long) and sorted. A preorder walk with children taken in increasing character order produces sorted keys (2, "Walks without recursion"), so the explicit stack must pop children in that order.

### The code, annotated

The arena starts with the root, which spells the empty string and is never a key:

```python
class Trie:
    def __init__(self) -> None:
        self._children: list[dict[str, int]] = [{}]   # node i -> {char: child index}
        self._value: list[int | None] = [None]         # value of the key ending at node i
        self._size = 0                                 # number of distinct keys
```

`insert` follows the key's characters and creates nodes only where the path does not exist yet, which is why shared prefixes are stored once and $n = 1 + \lvert P(K) \rvert$:

```python
    def insert(self, key: str, value: int) -> None:
        if not key:   # the empty key would match length 0 everywhere: a segmenter loops
            raise ValueError("a trie key must be a non-empty string")
        node = 0
        for ch in key:
            nxt = self._children[node].get(ch)
            if nxt is None:                    # first key through this prefix: new node
                nxt = len(self._children)
                self._children.append({})
                self._value.append(None)
                self._children[node][ch] = nxt
            node = nxt
        # The value is stored whether or not the node was just created: inserting
        # "ab" after "abc" reuses an existing pass-through node and makes it terminal.
        if self._value[node] is None:          # a new key, not a re-insert
            self._size += 1
        self._value[node] = value              # a re-insert replaces the value
```

Lookups follow the key and then ask whether the node is terminal. A node that exists only because a longer key passes through it (`unbel`) returns `None`:

```python
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
        return self.get(key) is not None       # not `bool(...)`: id 0 is a key
```

`longest_prefix` is the table of section 3 as a loop. Lengths are measured from `start`, and `start == len(s)` is allowed (the loop runs zero times and returns `(0, None)`):

```python
    def longest_prefix(self, s: str, start: int = 0) -> tuple[int, int | None]:
        _check_start(s, start)                 # ValueError unless 0 <= start <= len(s)
        best_len, best_val = 0, None
        node = 0
        i = start
        while i < len(s):
            nxt = self._children[node].get(s[i])
            if nxt is None:                    # no edge: no longer key can match
                break
            node = nxt
            i += 1
            # Remember the deepest node that ends a key, not the deepest node.
            if self._value[node] is not None:
                best_len, best_val = i - start, self._value[node]
        return best_len, best_val
```

`prefixes` is the same walk as a generator; it yields at each terminal node, so the order is shortest first by construction:

```python
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
```

`items` keeps its own stack. Each entry carries the node, its depth, and the character on the edge into it, and one shared `path` list holds the characters from the root to the current node, so a deep chain is not copied once per level:

```python
    def items(self) -> Iterator[tuple[str, int]]:
        path: list[str] = []
        stack: list[tuple[int, int, str]] = [(0, 0, "")]
        while stack:
            node, depth, ch = stack.pop()
            if depth:
                del path[depth - 1 :]          # back up to this node's parent...
                path.append(ch)                # ...and step along its edge
            v = self._value[node]
            if v is not None:
                yield "".join(path), v         # a key before its extensions
            kids = self._children[node]
            # Push in reverse order so the smallest character is popped first:
            # the walk is a preorder in increasing code-point order, i.e. sorted keys.
            for c in sorted(kids, reverse=True):
                stack.append((kids[c], depth + 1, c))
```

The truncation `del path[depth - 1:]` is what lets one list serve the whole walk: when the walk jumps from the end of `unbeliev` back to a sibling higher up, the depth of the popped entry says how much of the path is still valid.

Full solution: [`trie.py`](../../../../course/solutions/python/tinyllm/tok/trie.py).

### Why this solution

- **Cost.** `insert`, `get`, and `__contains__` take $O(\lvert k \rvert)$ dictionary steps; `longest_prefix` and `prefixes` take $O(\min(m, \lvert s \rvert - \text{start}))$, independent of the vocabulary size. `items` touches every node once and sorts each node's children, $O(n \log \sigma)$ for at most $\sigma$ children per node, plus the cost of joining each key. Memory is one dict and one slot per node.
- **The scan of every key**, `[w for w in vocab if s.startswith(w, i)]`, is correct, and it is the oracle of the brute-force case in section 4, but it costs $O(\lvert K \rvert)$ comparisons per position: hours on a 1 MB corpus with 30 000 entries.
- **A sorted list with binary search** finds whether one string is a key in $O(\log \lvert K \rvert)$, but "all keys that are prefixes of $s[i:]$" is not one contiguous range of a sorted list, so it needs one search per candidate length.
- **Node objects with a parent and a children dict** work too, but cost an object per node and make `node_count` a traversal; the arena is two lists and an integer.
- **A double-array trie or a radix tree** is smaller and faster (see "Going further"), but harder to build incrementally. For a vocabulary of tens of thousands of entries, built once per tokenizer, a dict per node is the clear and adequate choice.
- **Tradeoff accepted.** Characters are code points and nothing is normalized; that keeps the trie a pure data structure and puts normalization in one place (`L1.1`).

### Explain it back

1. Why does `longest_prefix` keep `best_len, best_val` instead of returning the node where the walk stops?

   <details><summary>Answer</summary>

   The walk stops where the text leaves the trie, and that node may be a pass-through node that is not a key. For `unbelie` it stops seven edges down, but the longest key is `unbe`, the last terminal node passed. Only terminal nodes are vocabulary pieces.

   </details>

2. Why is the test `self._value[node] is not None` and never `if self._value[node]:`?

   <details><summary>Answer</summary>

   Values are token ids, and 0 is a real id. A truthiness test would treat the key with id 0 as absent in `get`, `__contains__`, `longest_prefix`, `prefixes`, and `items`.

   </details>

3. `insert("abc", 1)` and then `insert("ab", 2)`. Which lines make `ab` a key, and why must they run even though no node is created?

   <details><summary>Answer</summary>

   The walk reaches the existing node for `ab`, then `if self._value[node] is None: self._size += 1` and `self._value[node] = value` make it terminal. Storing the value only for newly created nodes, or only at leaves, would lose every key that is a prefix of an earlier one.

   </details>

4. Why does `items` push children in reverse sorted order?

   <details><summary>Answer</summary>

   A stack pops the last item pushed. Pushing in decreasing order means the smallest character is popped first, so the walk visits subtrees in increasing order, which yields the keys in sorted order.

   </details>

5. Why does `prefixes` yield as it walks instead of building a list and returning it?

   <details><summary>Answer</summary>

   The walk meets terminal nodes shortest first, which is the order the Unigram lattice wants, so yielding preserves it with no extra work. A caller that only needs the first few matches can stop early, and no intermediate list is built per position.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. returning the deepest node reached | `unbelie` matches 7 characters that are not a key | updates `best_len, best_val` only at terminal nodes; see step 5 |
| 2. returning the first key on the path | `unbelievable` splits into `un` + ... | keeps walking after a match and overwrites the best |
| 3. ignoring `start`, or rejecting `start == len(s)` | matches the beginning of the text again; the last position of a string crashes | starts the walk at `i = start`, returns `i - start`, and accepts `0 <= start <= len(s)` |
| 4. `if value:` instead of `if value is not None:` | token id 0 disappears | every test is `is not None`; see step 3 |
| 5. storing a value only when a node was created or is a leaf | `ab` inserted after `abc` is lost | stores the value after the walk, whether or not a node was created |
| 6. "a node exists" read as "a key exists" | `"unbel" in trie` is true | `get` returns the node's value, `None` for a pass-through node |
| 7. a recursive walk | `RecursionError` on a long run of one character | an explicit stack and one shared `path` list |
| 8. children in insertion order | `items()` order depends on how the vocabulary was loaded | pushes `sorted(kids, reverse=True)` so the smallest character pops first |
| 9. collecting matches, then reversing | the lattice gets longest-first edges | yields at each terminal node as the walk meets it |
| 10. counting a re-inserted key twice | `len` drifts above the vocabulary size | increments `_size` only when the node had no value |
| 11. accepting the empty key | every position matches length 0 | `insert` raises `ValueError` for `""` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M06.1` | rooted graphs, parents and children, and why every walk keeps its own stack |
| Forward | `L1.2` | byte-level BPE finds added and special tokens (`<|im_start|>`) with `longest_prefix` before splitting the rest |
| Forward | `L1.3` | WordPiece is `longest_prefix` in a loop, with `##` continuation keys |
| Forward | `L1.4` | the Unigram lattice adds one edge per `prefixes` match at every position, then runs Viterbi |
| Forward | `ds.07`, `L8.4` | the same idea over token ids instead of characters, compressed into a radix tree, finds the longest cached prompt prefix in the Rust engine (they read this chapter; they do not call this code) |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Trie` | Hugging Face `tokenizers` | a byte trie for added tokens and an Aho-Corasick automaton that finds all of them in one pass | `tokenizers/src/tokenizer/added_vocabulary.rs` |
| `prefixes` | SentencePiece | a double-array trie (Darts): two flat integer arrays, cache friendly, built once per model | `third_party/darts_clone/darts.h`, `src/unigram_model.cc` |
| `longest_prefix` | SGLang RadixAttention | a radix tree over token ids whose nodes own KV-cache blocks, with LRU eviction of unlocked leaves | `python/sglang/srt/mem_cache/radix_cache.py` |
