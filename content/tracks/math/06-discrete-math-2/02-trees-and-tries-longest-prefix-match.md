<!-- ol:module M06.2 -->
# Trees and tries, longest-prefix match

## Overview

| | |
|---|---|
| **Module** | `M06.2` · build · Python · Pass 3 · 2 to 3 h |
| **You build** | `python/tinyllm/tok/trie.py`: `Trie` (`insert`, `get`, `__contains__`, `__len__`, `node_count`, `longest_prefix`, `prefixes`, `items`) |
| **Contract** | [`course/contracts/py/tinyllm/tok/trie.pyi`](../../../../course/contracts/py/tinyllm/tok/trie.pyi) |
| **Tests** | `course/tests/M06.2/` (what they check: section 4) |
| **Needs** | no code from earlier modules · reading: `M06.1` graphs, DAGs, and iterative traversal |
| **Used by** | `L1.2` added-token matching in byte-level BPE · `L1.3` WordPiece greedy longest match · `L1.4` the Unigram lattice · later: `ds.07` and `L8.4` generalize it to a radix tree over token ids in Rust |
| **Milestone** | `MS-P3` (tokens and data) |
| **Optional depth** | Sedgewick and Wayne, *Algorithms* (4th ed.), section 5.2 "Tries"; Knuth, *TAOCP* vol. 3, section 6.3 "Digital Searching" |

## Key Takeaways

- A tree with $n$ nodes has $n - 1$ edges and one path from the root to each node; a trie labels those edges with characters, so each node spells the string on its path and a lookup costs one step per character, not one comparison per key (`test_hand_example_longest_prefix`).
- Longest-prefix match remembers the deepest node that ends a key, not the deepest node reached (`test_deepest_key_not_deepest_node`).
- A trie stores each shared prefix once: it has exactly $1 + $ (number of distinct non-empty prefixes) nodes (`test_node_count_is_distinct_prefixes`).
- A preorder walk that visits children in increasing character order lists the keys in sorted order, and as a loop it survives a key of 100 000 characters (`test_items_in_code_point_order`, `test_items_survives_a_very_deep_key`).

## How to work this chapter

```bash
ol start M06.2              # stubs trie.py into your repo
ol tests M06.2              # read the test catalog first
ol check M06.2              # exit code is the verdict
ol diff  M06.2              # after passing: your code against the reference
```

---

## 1. Why now

Pass 3 gives your system real tokenizers, and three of them ask the same question thousands of times per sentence: which vocabulary entries start at this position of the text? WordPiece (`L1.3`, BERT's tokenizer) takes the longest one and jumps past it. The Unigram tokenizer (`L1.4`) needs every one of them to build its lattice. Byte-level BPE (`L1.2`) must find added tokens such as `<|im_start|>` before it splits anything. The obvious code, `[w for w in vocab if text.startswith(w, i)]`, compares against all 30 000 entries at every position: tokenizing a 1 MB corpus that way is $10^6 \times 3 \cdot 10^4$ string comparisons, hours of Python. A trie answers the same question in a handful of dictionary lookups, however large the vocabulary. This module builds it, on top of the tree vocabulary from `M06.1`.

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

**Trees.** A tree is a graph that is connected (there is a path between any two nodes) and has no cycle. Two facts follow and are worth proving once (`S-M06b` asks you to): between any two nodes there is exactly one path, and a tree with $n$ nodes has exactly $n - 1$ edges. Pick one node as the root $r$ and every edge gets a direction, away from the root. Each node other than $r$ then has exactly one parent, the next node on its unique path to $r$, and its children are the nodes it is parent of. A node with no children is a leaf. $\mathrm{depth}(v)$ is the number of edges from $r$ to $v$. This is the same parent-and-children structure as the autograd graph in `M06.1`, minus the sharing: in a tree no node has two parents.

**Tries.** A trie (from re*trie*val) for a set of strings $K$ is a rooted tree whose edges carry characters, with two rules: the edges out of one node carry distinct characters, and the path from $r$ to a node $v$ spells a string $\mathrm{str}(v)$ (the root spells the empty string). A node is **terminal** when $\mathrm{str}(v) \in K$, and then it stores $\mathrm{val}(\mathrm{str}(v))$. Because edges out of a node are distinct, each string has at most one node, and the node for $k$ is found by following $k$'s characters one at a time from the root: $\lvert k \rvert$ steps, each a dictionary lookup on the node's children.

**Counting nodes.** Every non-root node spells a non-empty prefix of some key (the path to it is the start of some key's path), and every non-empty prefix of a key is spelled by exactly one node. So $n = 1 + \lvert P(K) \rvert$. Keys that share a start share nodes: `un`, `unbe`, and `unbeliev` together need 8 non-root nodes, not $2 + 4 + 8 = 14$. In the worst case (no sharing) $n = 1 + \sum_{k \in K} \lvert k \rvert$.

**Longest-prefix match.** To find the longest key that is a prefix of $s[i:]$, walk from the root along $s_i, s_{i+1}, \dots$ until a character has no edge or the text ends. Every terminal node you pass is a key that is a prefix of $s[i:]$, shortest first, so `prefixes` yields them in the order met, and `longest_prefix` returns the last one. The walk is at most $\min(m, \lvert s \rvert - i)$ steps, independent of $\lvert K \rvert$. The node where the walk stops need not be terminal: the keys `unbe` and `unbeliev` put the walk for `unbelie` on a pass-through node seven edges down, and the answer is the last terminal node, `unbe`. Lengths are counted from $i$, and $i = \lvert s \rvert$ is the empty suffix, which matches nothing.

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

These numbers are the first cases in section 4: `test_hand_example_longest_prefix` and `test_hand_example_prefixes`.

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

Store the tree any way you like; the reference keeps two parallel lists, a children dict and a value per node, with node 0 as the root. `start` outside `[0, len(s)]` is a `ValueError`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_longest_prefix` | unit | section 3: (8, 3), (4, 4), 13 nodes, greedy split | you and the test agree on the definition |
| `test_hand_example_prefixes` | unit | all matches, shortest first | lattice edges for `L1.4` |
| `test_deepest_key_not_deepest_node` | boundary | `unbelie` gives `unbe` | WordPiece emits only vocabulary pieces |
| `test_longest_not_first_match` | boundary | `a`, `ab`, `abc` against `abcd` | greedy means longest |
| `test_no_match_is_zero_none` | boundary | (0, None) for no match and for an empty trie | callers test the length |
| `test_start_offset_and_bounds` | boundary | lengths from `start`, `start == len(s)`, out of range raises | matching from every position of one string |
| `test_value_zero_is_a_value` | boundary | token id 0 is found everywhere | id 0 is a real token |
| `test_code_points_are_not_normalized` | unit | `é` and `e` + U+0301 are different keys | normalization belongs to `L1.1` |
| `test_matches_brute_force` | differential | random keys and texts against a scan of every key | the trie answers the obvious question |
| `test_insert_a_prefix_of_an_existing_key` | boundary | `ab` inserted after `abc` | vocabularies arrive in any order |
| `test_reinsert_replaces_the_value` | unit | same `len`, same nodes, new value | an added token overrides a learned one |
| `test_get_and_contains_need_a_key_not_a_prefix` | boundary | `unbel` exists as a node, is not a key | no pieces outside the vocabulary |
| `test_empty_key_rejected` | boundary | `insert("")` raises | a zero-length match loops a segmenter forever |
| `test_node_count_is_distinct_prefixes` | property | $n = 1 + \lvert P(K) \rvert$ on random keys | shared prefixes are stored once |
| `test_items_in_code_point_order` | property | `items()` equals `sorted(keys.items())` | deterministic vocabulary files |
| `test_items_survives_a_very_deep_key` | boundary | a 100 000 character key | no recursion limit |
| `test_golden_greedy_segmentation` | golden | greedy splits of 26 strings from a brute-force oracle | WordPiece in `L1.3` |
| `test_golden_lattice` | golden | every match at every position | the Unigram lattice in `L1.4` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. returning the deepest node reached | `unbelie` matches 7 characters that are not a key | `test_deepest_key_not_deepest_node` (mutant `s01`) |
| 2. returning the first key on the path | `unbelievable` splits into `un` + ... | `test_longest_not_first_match` (mutant `s02`) |
| 3. ignoring `start`, or rejecting `start == len(s)` | matches the beginning of the text again; the last position of a string crashes | `test_start_offset_and_bounds` (mutants `s03`, `s11`) |
| 4. `if value:` instead of `if value is not None:` | token id 0 disappears | `test_value_zero_is_a_value` (mutant `s06`) |
| 5. storing a value only when a node was created or is a leaf | `ab` inserted after `abc` is lost | `test_insert_a_prefix_of_an_existing_key` (mutant `s04`) |
| 6. "a node exists" read as "a key exists" | `"unbel" in trie` is true | `test_get_and_contains_need_a_key_not_a_prefix` (mutant `s12`) |
| 7. a recursive walk | `RecursionError` on a long run of one character | `test_items_survives_a_very_deep_key` (mutant `s07`) |
| 8. children in insertion order | `items()` order depends on how the vocabulary was loaded | `test_items_in_code_point_order` (mutant `s08`) |
| 9. collecting matches, then reversing | the lattice gets longest-first edges | `test_hand_example_prefixes` (mutant `s09`) |
| 10. counting a re-inserted key twice | `len` drifts above the vocabulary size | `test_reinsert_replaces_the_value` (mutant `s05`) |
| 11. accepting the empty key | every position matches length 0 | `test_empty_key_rejected` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M06.1` | rooted graphs, parents and children, and why every walk keeps its own stack |
| Forward | `L1.2` | byte-level BPE finds added and special tokens (`<|im_start|>`) with `longest_prefix` before splitting the rest |
| Forward | `L1.3` | WordPiece is `longest_prefix` in a loop, with `##` continuation keys |
| Forward | `L1.4` | the Unigram lattice adds one edge per `prefixes` match at every position, then runs Viterbi |
| Forward | `ds.07`, `L8.4` | the same idea over token ids instead of characters, compressed into a radix tree, finds the longest cached prompt prefix in your Rust engine (they read this chapter; they do not call this code) |

If you skip this module, `ol check L1.3` stops with `L1.3 needs M06.2`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Trie` | Hugging Face `tokenizers` | a byte trie for added tokens and an Aho-Corasick automaton that finds all of them in one pass | `tokenizers/src/tokenizer/added_vocabulary.rs` |
| `prefixes` | SentencePiece | a double-array trie (Darts): two flat integer arrays, cache friendly, built once per model | `third_party/darts_clone/darts.h`, `src/unigram_model.cc` |
| `longest_prefix` | SGLang RadixAttention | a radix tree over token ids whose nodes own KV-cache blocks, with LRU eviction of unlocked leaves | `python/sglang/srt/mem_cache/radix_cache.py` |
