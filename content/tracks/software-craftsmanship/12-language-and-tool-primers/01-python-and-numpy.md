<!-- ol:module lang.01 -->
# Python and numpy: arrays, dtypes, broadcasting, views vs copies, uv projects

## Overview

| | |
|---|---|
| **Module** | `lang.01` · practice · Python · Pass 0 · 3 to 4 h |
| **You build** | `primers/lang.01/`: a uv project (`pyproject.toml`, `uv.lock`); `worksheet.py` with `broadcast_shape`, `unbroadcast`, `as_c_float32`; `bigram.py` with `bigram_counts`, `row_normalize` |
| **Contract** | none: a primer exercise, not part of the system. The signatures are in the stubs `ol start lang.01` writes |
| **Tests** | `course/tests/lang.01/`, run by its `check` script (what they check: section 4) |
| **Needs** | nothing: the course starts here. Python 3.11 or later and `uv` on your PATH (`ol doctor`) |
| **Used by** | no code call site (a primer). It is the concept prerequisite of `L0.0` (byte bigram from counts), optional C exercises such as `M03.1` (row-major matmul), and `L0.1` (broadcasting backward) |
| **Milestone** | MS-P0 (page: `content/courses/build-an-llm/paths/course-p00-setup/milestone.md`) |
| **Optional depth** | numpy user guide, "Broadcasting", "Copies and views", "Data type objects" (numpy.org/doc, free); Harris et al., "Array programming with NumPy", *Nature* 585, 2020 (free); uv docs, "Working on projects" (docs.astral.sh/uv, free) |

## Key Takeaways

- A numpy array is **one flat buffer plus metadata** (dtype, shape, strides). Transposes and slices change only the metadata, so they are views that share the buffer.
- **Broadcasting aligns shapes on the right**, pads with 1s on the left, and stretches size-1 axes. You can predict any broadcast's shape: the tests check your rule against numpy on all 1600 small pairs.
- **Every broadcast forward is a sum backward.** `unbroadcast` is the adjoint of broadcasting, the identity L0.1's autograd uses on every elementwise op.
- **The dtype is part of the value.** A `uint8` count wraps at 256; an `int64` count does not.
- **A uv project pins your environment.** `pyproject.toml` declares numpy, `uv.lock` pins the exact build, and `uv run` reproduces it on any machine, including CI.

## How to work this chapter

```bash
ol start lang.01                             # writes stubs: primers/lang.01/{worksheet,bigram}.py
uv init --bare primers/lang.01               # step 1: your uv project
uv add --project primers/lang.01 numpy       #         declares numpy and writes uv.lock
ol tests lang.01                             # read the test catalog first
ol check lang.01                             # exit code is the verdict
```

---

## 1. Why now

Your system is an empty repository. In Pass 1 you train its first model, L0.0's byte bigram, and that training is nothing more than counting which byte follows which in a text file and turning the counts into probabilities. A Python loop over a 1 MB file does that count in about 0.12 s on a laptop; one numpy call does it in about 0.005 s, and the gap grows with every pass over the data. Next, rt.01 hands numpy arrays to C through a raw pointer, and if the array's bytes are not laid out the way C reads them, C computes on the wrong numbers with no error at all. Then L0.1's autograd has to undo broadcasting in every backward pass. All three rest on one picture of what an array is, and on knowing exactly which environment your Python runs in. This primer builds that picture with five small functions.

## 2. Principles

### 2.1 An array is a buffer plus metadata

A numpy array is a block of memory (the **buffer**) plus four facts about how to read it.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $b$ | bytes per element (`itemsize`), fixed by the dtype: 1 for `uint8`, 4 for `float32`, 8 for `int64` and `float64` | integer |
| $d$ | number of axes (`ndim`) | integer |
| $n_k$ | size of axis $k$, for $k = 0, \dots, d-1$; the tuple $(n_0, \dots, n_{d-1})$ is the `shape` | integer |
| $i_k$ | an index on axis $k$, with $0 \le i_k < n_k$ | integer |
| $s_k$ | **stride** of axis $k$: how many bytes to move to go from index $i_k$ to $i_k + 1$ | integer (bytes) |
| $o$ | byte offset of element $(0, \dots, 0)$ inside the buffer | integer (bytes) |

The element at index $(i_0, \dots, i_{d-1})$ lives at byte

$$\text{addr}(i) = o + \sum_{k=0}^{d-1} i_k \, s_k .$$

An array is **C-contiguous** (row-major) when its last axis is packed and every earlier axis steps over a whole row of the next: $s_{d-1} = b$ and $s_k = s_{k+1}\, n_{k+1}$. A `float32` array of shape (3, 4) has strides (16, 4): one step along a row moves 4 bytes, one step down a column moves a whole row of 4 elements, 16 bytes. This layout is what a C function reading `const float *a` with explicit dimensions assumes.

Because the formula is the whole story, many operations need no copying at all. They make a **view**: a new array object with new metadata over the **same** buffer.

| Operation on `x` (float32, shape (3, 4), strides (16, 4)) | Result shape | Result strides | Shares the buffer? |
|---|---|---|---|
| `x.T` (transpose) | (4, 3) | (4, 16) | yes, a view |
| `x[:, ::2]` (every other column) | (3, 2) | (16, 8) | yes, a view |
| `x.reshape(12)` | (12,) | (4,) | yes, a view (x is contiguous) |
| `x.astype(np.float64)` | (3, 4) | (32, 8) | no, a copy |
| `np.broadcast_to(np.ones(4, np.float32), (3, 4))` | (3, 4) | (0, 4) | yes: stride 0 repeats a row without copying it |

A **copy** has its own buffer; writing to it never changes the original. `np.shares_memory(a, b)` answers whether two arrays overlap. `y.flags.c_contiguous` answers whether `y` is laid out row-major.

### 2.2 The dtype is part of the value

The dtype says how to interpret each run of $b$ bytes. `uint8` holds the integers 0 to 255, and arithmetic on it is done **modulo 256**: 200 + 100 stored in `uint8` is 44, silently. `int64` holds about $\pm 9.2 \times 10^{18}$, enough for any count in this course. `float32` keeps 24 significant bits, so `np.float32(0.1)` is really 0.100000001490116..., the nearest value it can hold. numpy 2 raises `OverflowError` when you mix a `uint8` array with a Python integer that does not fit (`x * 256`), but `uint8` arithmetic between arrays, or with a constant that fits (`x * 255 + y`), still wraps without a word.

### 2.3 Broadcasting

An elementwise operation such as `x + y` needs both operands to have the same shape. Broadcasting is numpy's rule for when it may **pretend** they do, by stretching size-1 axes with stride 0 (2.1) instead of copying. For shapes $a$ (with $p$ axes) and $b$ (with $q$ axes):

1. Let $n = \max(p, q)$. Pad the shorter shape with 1s **on the left** until both have $n$ axes. Equivalently, align the shapes on their **last** axis.
2. For each axis $k$: if $a_k = b_k$, the result has $a_k$. If one of them is 1, the result has the other. Otherwise the shapes do not broadcast, and numpy raises `ValueError`.

So a bias vector of shape (3,) added to a batch of shape (2, 3) acts as (1, 3), stretched to (2, 3): the same 3 numbers are added to every row. The rule never creates data; it only reuses elements.

### 2.4 Broadcasting backward: unbroadcast

Write $B$ for the operation "broadcast $x$ from shape $s$ to shape $T$". Every element $x_j$ appears at several positions of $B x$. If some later number $L$ depends on $y = Bx$, the chain rule says the sensitivity of $L$ to $x_j$ is the sum of its sensitivities to every copy of $x_j$:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | the operand before broadcasting | shape $s$ |
| $y = Bx$ | the broadcast result | shape $T$ |
| $g$ | the gradient of $L$ with respect to $y$, one number per element of $y$ | shape $T$ |
| $\langle u, v \rangle$ | $\sum_m u_m v_m$, the sum of elementwise products of two same-shaped arrays | scalar |
| $B^\top g$ | `unbroadcast(g, s)`: $g$ summed over every position that holds a copy of the same $x_j$ | shape $s$ |

$$\frac{\partial L}{\partial x_j} = \sum_{\text{positions } m \text{ that copy } x_j} g_m , \qquad\text{equivalently}\qquad \langle Bx, g \rangle = \langle x, B^\top g \rangle .$$

Concretely, $B^\top$ sums $g$ over the leading axes broadcasting added, then over every axis where $s$ has size 1, **keeping** those axes so the result has shape exactly $s$. The right-hand identity is the test: it must hold for every $x$ and $g$.

### 2.5 Vectorization and the pair code

A Python loop runs the interpreter once per element. A numpy call runs one compiled loop over the whole buffer. To count byte pairs without a Python loop, turn each adjacent pair $(a, b)$ into one integer

$$k = 256\,a + b, \qquad 0 \le k < 65536 ,$$

count how many times each $k$ occurs with `np.bincount(k, minlength=65536)`, and reshape the 65536 counts to (256, 256). The reshape is free and exact because $256a + b$ is precisely the row-major position of $(a, b)$ in a 256 by 256 table (2.1 with $s = (256, 1)$ in elements). Compute $k$ in `int64`: in `uint8`, $256a$ does not fit.

Why not `C[a, b] += 1` with index arrays? Indexing with arrays is **buffered**: numpy reads `C[a, b]` for all pairs, adds 1, then writes back, so a pair that occurs 3 times is written 3 times with the same value and counts once. `np.add.at(C, (a, b), 1)` is the unbuffered version; `bincount` is the fast one.

### 2.6 From counts to probabilities

For a count table $C$ with rows $a$, the probability that $b$ follows $a$ is the row-normalized count $P_{ab} = C_{ab} / \sum_{b'} C_{ab'}$. In numpy that is `C / C.sum(axis=1, keepdims=True)`: the row sums keep shape (256, 1), so broadcasting divides each row by its own total. A row with no counts has total 0, and $0/0$ is NaN; this exercise keeps such rows at 0 (L0.0 adds smoothing so that no row is empty).

### 2.7 uv projects

A **uv project** is a directory with a `pyproject.toml`. Its `[project].dependencies` list says what the code needs (`numpy>=2.0`). `uv add` edits that list and writes **`uv.lock`**, which pins the exact version and file hash of every package. `uv run <cmd>` first makes the project's own environment (`.venv/` in the project directory) match the lock, then runs `<cmd>` inside it. So `uv run` on your laptop and in CI runs the same numpy. Commit `pyproject.toml` and `uv.lock`; never commit `.venv/` (the `.gitignore` that `ol course init` wrote already ignores it). Installing numpy by hand into some other Python is invisible to `uv run`.

## 3. Worked example by hand

These numbers are the first test of each function in section 4.

**Broadcast shapes.** Pad on the left, then compare axis by axis.

| a | b | a, b padded | per axis | result |
|---|---|---|---|---|
| (3, 1) | (1, 4) | (3, 1), (1, 4) | 3 vs 1 gives 3; 1 vs 4 gives 4 | (3, 4) |
| (2, 3) | (3,) | (2, 3), (1, 3) | 2 vs 1 gives 2; 3 vs 3 gives 3 | (2, 3) |
| (8, 1, 6, 1) | (7, 1, 5) | (8, 1, 6, 1), (1, 7, 1, 5) | 8; 7; 6; 5 | (8, 7, 6, 5) |
| () | (5,) | (1,), (5,) | 1 vs 5 gives 5 | (5,) |
| (4, 1) | (0,) | (4, 1), (1, 0) | 4; 1 vs 0 gives 0 | (4, 0) |
| (3,) | (4,) | (3,), (4,) | 3 vs 4: neither is 1 | `ValueError` |

**Unbroadcast.** Take $g$ = a (2, 3) array of ones. Each result entry counts how many copies its element had.

| target shape | what to sum | result |
|---|---|---|
| (3,) | the added leading axis 0 | [2, 2, 2] |
| (2, 1) | axis 1, kept | [[3], [3]] |
| (1, 3) | axis 0, kept | [[2, 2, 2]] |
| () | both axes | 6 |
| (2, 3) | nothing | g itself |

**Layout.** `x = np.arange(12, dtype=np.float32).reshape(3, 4)` has strides (16, 4). Its transpose `x.T` has shape (4, 3) and strides (4, 16), so `x.T[1, 0]` sits at byte $1 \cdot 4 + 0 \cdot 16 = 4$, which is `x[0, 1]`: correct as an array. But a C function given `x.T`'s buffer pointer walks bytes 0, 4, 8, ... and reads 0, 1, 2, 3, ..., the elements of `x` in row order, while the row-major (4, 3) array it was promised starts 0, 4, 8. `as_c_float32(x.T)` must therefore copy into a fresh buffer that reads 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11. And `as_c_float32` of the `float64` array [[0.1, 0.2, 0.3], [1, 2, 3]] (strides (24, 8)) is a `float32` copy with strides (12, 4) whose first element is 0.100000001490116.

**Bigram counts of "banana".** The bytes are 98 97 110 97 110 97 (b a n a n a). The 5 adjacent pairs and their codes $k = 256a + b$:

| pair | a, b | k |
|---|---|---|
| ba | 98, 97 | 25088 + 97 = 25185 |
| an | 97, 110 | 24832 + 110 = 24942 |
| na | 110, 97 | 28160 + 97 = 28257 |
| an | 97, 110 | 24942 |
| na | 110, 97 | 28257 |

`bincount` gives 1 at 25185, 2 at 24942, 2 at 28257. Reshaped: $C_{98,97} = 1$, $C_{97,110} = 2$, $C_{110,97} = 2$, every other entry 0, total 5 = 6 - 1.

**Row normalization.** For $C = [[1, 3], [1, 1]]$ the row sums are 4 and 2, so $P = [[1/4, 3/4], [1/2, 1/2]]$. The wrong version `C / C.sum(axis=1)` divides by [4, 2] laid along the **columns**: $[[1/4, 3/2], [1/4, 1/2]]$, whose first row sums to 1.75. No error is raised: the table is square, so the shapes broadcast.

## 4. The artifact and its check

**Step 1, the uv project.** In your repo:

```bash
uv init --bare primers/lang.01                 # writes primers/lang.01/pyproject.toml
uv add --project primers/lang.01 numpy         # adds numpy to [project].dependencies, writes uv.lock
git add primers/lang.01/pyproject.toml primers/lang.01/uv.lock
```

**Step 2, the stubs.** `ol start lang.01` writes `primers/lang.01/worksheet.py` and `primers/lang.01/bigram.py`. Every function body is `raise NotImplementedError("lang.01")`; you replace the bodies, never the signatures.

**Step 3, the functions.**

```python
# primers/lang.01/worksheet.py
def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]: ...
    # the shape of x + y for x.shape == a, y.shape == b; ValueError when they do not broadcast.
    # Implement the rule of 2.3 yourself; do not call np.broadcast_shapes.
def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray: ...
    # grad summed back down to `shape` (2.4); ValueError when `shape` could not broadcast to grad.shape
def as_c_float32(x) -> np.ndarray: ...
    # x as a C-contiguous float32 array; shares memory with x when x already is one

# primers/lang.01/bigram.py
def bigram_counts(data: bytes) -> np.ndarray: ...
    # int64 (256, 256): C[a, b] = number of i with data[i] == a and data[i + 1] == b; no Python loop
def row_normalize(counts: np.ndarray) -> np.ndarray: ...
    # float64, each nonzero row divided by its sum; zero rows stay zero
```

**Step 4, the check.** `ol check lang.01` runs `course/tests/lang.01/check` in your repo. It refuses early, with the reason, if a file is missing, then runs `uv run --project primers/lang.01 --with pytest pytest course/tests/lang.01` with `primers/lang.01` on `PYTHONPATH`. Your project provides numpy; the check adds only pytest.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_broadcast_shape_hand_examples` | unit | the section 3 table | you and the test agree on the rule |
| `test_broadcast_shape_is_your_own_rule` | unit | passes with numpy's broadcasting helpers switched off | the exercise is the rule |
| `test_broadcast_shape_aligns_on_the_right` | boundary | (3,) with (3, 1) gives (3, 3) | the bias-add shape in every layer |
| `test_broadcast_shape_rejects_incompatible_shapes` | boundary | (3,) with (4,) raises | a shape bug must fail loudly |
| `test_broadcast_shape_matches_numpy_on_every_small_pair` | differential | all 1600 pairs of shapes with up to 3 axes of sizes 1 to 3 | you can predict any broadcast |
| `test_unbroadcast_hand_example` | unit | the section 3 table | the L0.1 backward of an add |
| `test_unbroadcast_keeps_size_one_axes` | boundary | target (2, 1) stays (2, 1) | a bias gradient keeps the bias's shape |
| `test_unbroadcast_sums_leading_axes` | boundary | (4, 2, 3) to (3,) and (1, 3) | batch axes are summed away |
| `test_unbroadcast_rejects_a_shape_that_did_not_broadcast` | boundary | (2,) is not a source of (2, 3) | no invented gradients |
| `test_unbroadcast_is_the_adjoint_of_broadcasting` | property | $\langle Bx, g\rangle = \langle x, B^\top g\rangle$ on every small pair, seeded PCG32 values | the law autograd relies on |
| `test_as_c_float32_hand_example` | unit | float64 to float32, row-major bytes | rt.01's argument conversion |
| `test_as_c_float32_does_not_copy_when_it_need_not` | unit | an already-right array is shared | no copy per C call |
| `test_as_c_float32_copies_a_transposed_view` | boundary | `x.T` becomes a fresh row-major copy | C reads the right elements |
| `test_as_c_float32_copies_a_strided_slice` | boundary | `x[:, ::2]` becomes dense | C has no stride argument for this |
| `test_as_c_float32_keeps_a_scalar_0d` | boundary | a 0-d input stays shape `()` | scalar tensors in L0.0's safetensors files |
| `test_bigram_hand_example` | unit | "banana" from section 3 | the L0.0 training count |
| `test_bigram_counts_every_repeated_pair` | boundary | "aaaa" counts aa three times | real text repeats pairs constantly |
| `test_bigram_counts_do_not_wrap_at_256` | boundary | 999 repeats read as 999, dtype int64 | counts over megabytes |
| `test_bigram_counts_of_short_inputs_are_all_zero` | boundary | empty and one-byte inputs | edge of every corpus shard |
| `test_bigram_counts_use_all_256_byte_values` | boundary | bytes 128 and 255 get their own rows | UTF-8 text uses high bytes (D32) |
| `test_bigram_row_sums_count_each_leading_byte` | property | row $a$ sums to the count of $a$ in `data[:-1]` | a law, not one example |
| `test_bigram_counts_match_a_python_loop` | differential | equal to the obvious loop on random bytes | the loop is the definition |
| `test_bigram_counts_has_no_python_loop` | unit | no for, while, or comprehension in `bigram_counts` or any helper in `bigram.py` it calls | the exercise is vectorizing |
| `test_row_normalize_hand_example` | unit | [[1, 3], [1, 1]] from section 3 | the bigram's probabilities |
| `test_row_normalize_divides_rows_not_columns` | boundary | rows sum to 1 on an uneven table | the silent square-table bug |
| `test_row_normalize_leaves_empty_rows_zero` | boundary | no NaN for unseen bytes | one NaN poisons every sum |
| `test_project_declares_numpy` | unit | `[project].dependencies` lists numpy | the environment is reproducible |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Aligning shapes on the left (padding on the right) | (3,) with (3, 1) gives (3, 1) instead of (3, 3) | `test_broadcast_shape_aligns_on_the_right` |
| 2. Taking the larger size when two sizes differ and neither is 1 | (3,) with (4,) "broadcasts" to (4,) | `test_broadcast_shape_rejects_incompatible_shapes` |
| 3. Summing without `keepdims=True` in `unbroadcast` | a (2, 1) bias gets a (2,) gradient; the update `b -= lr * g` then broadcasts `b` up to (2, 2) | `test_unbroadcast_keeps_size_one_axes` |
| 4. `np.array(x, dtype=np.float32)` or `x.astype(np.float32)` | correct values, but a full copy on every call even when `x` is already right | `test_as_c_float32_does_not_copy_when_it_need_not` |
| 5. `np.asarray(x, dtype=np.float32)` | a transposed view passes through unchanged; C reads `x`'s rows instead | `test_as_c_float32_copies_a_transposed_view` |
| 6. `np.ascontiguousarray(x, dtype=np.float32)` | right for every array of 1 or more dimensions, but it returns at least 1-D: a 0-d scalar comes back with shape `(1,)` | `test_as_c_float32_keeps_a_scalar_0d` |
| 7. `C[a, b] += 1` with index arrays | each distinct pair counts once ("aaaa" gives 1, not 3) | `test_bigram_counts_every_repeated_pair` |
| 8. A `uint8` count table, or `uint8` pair codes | 999 repeats read as 231; codes `a * 255 + b` wrap | `test_bigram_counts_do_not_wrap_at_256` |
| 9. `C / C.sum(axis=1)` without `keepdims` | no error on a square table, rows do not sum to 1 | `test_row_normalize_divides_rows_not_columns` |
| 10. Dividing an empty row by its zero total | NaN rows that spread into every later sum | `test_row_normalize_leaves_empty_rows_zero` |
| 11. `pip install numpy` into a global Python instead of the project | works on your laptop, fails in CI and for `uv run` | `test_project_declares_numpy` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `L0.0` | the byte bigram's training is `bigram_counts` over a corpus, then row normalization with add-one smoothing |
| Forward | `M03.1` | row-major layout and strides help explain the C exercise's matrix buffers; Python and C are checked separately through fixture files |
| Forward | `M03.1` | `tl_matmul_f32` reads row-major float32 with an explicit leading dimension: the strides of 2.1, counted in elements |
| Forward | `L0.1` | the backward of every broadcasting op is `unbroadcast`, and its gradient test is the adjoint identity of 2.4 |
| Forward | `lang.02` | the next primer: the shell, exit codes, signals, and make |

A primer has no code call site, so no module's `ol check` blocks on it. MS-P0 does: it requires a fresh pass of `lang.01`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `unbroadcast` | PyTorch `Tensor.sum_to_size`, ATen `at::sum_to` | the same reduction, run on every broadcasting op's gradient | `aten/src/ATen/ExpandUtils.h` |
| `unbroadcast` | HIPS autograd `unbroadcast` | the same idea in a numpy-only autograd, close to what you build in L0.1 | `autograd/numpy/numpy_vjps.py` |
| `as_c_float32` | PyTorch `Tensor.contiguous()`, DLPack | zero-copy exchange of strided buffers between frameworks | dmlc/dlpack `include/dlpack/dlpack.h` |
| `bigram_counts` | `np.unique(..., return_counts=True)`, `collections.Counter` | counting arbitrary keys, not just dense integer codes | numpy docs, `numpy.bincount` |
| `primers/lang.01/uv.lock` | uv workspaces | one lock for several packages (your `python/` project in Pass 1) | docs.astral.sh/uv, "Workspaces" |
