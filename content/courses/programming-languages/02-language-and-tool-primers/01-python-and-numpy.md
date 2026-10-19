# Python and numpy: arrays, dtypes, broadcasting, views vs copies, uv projects

Every larger system in these books starts as an empty repository, and the first model it trains is a byte bigram: count which byte follows which in a text file, then turn the counts into probabilities. That training is nothing more than counting, and how you count matters. A Python loop over a 1 MB file does the count in about 0.12 s on a laptop; one numpy call does it in about 0.005 s, and the gap grows with every pass over the data. Soon after, numpy arrays are handed to C through a raw pointer, and if the array's bytes are not laid out the way C reads them, C computes on the wrong numbers with no error at all. Then an autograd engine has to undo broadcasting in every backward pass.

All three rest on one picture of what an array is, and on knowing exactly which environment your Python runs in. This section builds that picture with five small functions: `broadcast_shape`, `unbroadcast`, and `as_c_float32` in [`worksheet.py`](../../../../course/solutions/primers/lang.01/worksheet.py), and `bigram_counts` and `row_normalize` in [`bigram.py`](../../../../course/solutions/primers/lang.01/bigram.py), all inside the uv project described by [`pyproject.toml`](../../../../course/solutions/primers/lang.01/pyproject.toml). It is a primer exercise rather than part of any system, so there is no contract file; the signatures appear below.

You need nothing beyond Python 3.11 or later and `uv` on your PATH. This is the first section of the primers, and it assumes no earlier chapter.

> **Learning objectives**
> - Explain a numpy array as a buffer plus dtype, shape, and strides, and predict whether an operation returns a view or a copy.
> - Apply the broadcasting rule by hand to any pair of shapes, and implement its adjoint, `unbroadcast`.
> - Convert an arbitrary array to a C-contiguous `float32` buffer without copying when no copy is needed.
> - Count byte bigrams without a Python loop and row-normalize them without producing NaN.
> - Set up a uv project whose `pyproject.toml` and `uv.lock` make the environment reproducible.

## An array is a buffer plus metadata

A numpy array is a block of memory (the **buffer**) plus a few facts about how to read it. The following table names them.

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

Because the address formula is the whole story, many operations need no copying at all. They make a **view**: a new array object with new metadata over the **same** buffer. Here is what common operations do to `x`, a `float32` array of shape (3, 4) and strides (16, 4).

| Operation on `x` | Result shape | Result strides | Shares the buffer? |
|---|---|---|---|
| `x.T` (transpose) | (4, 3) | (4, 16) | yes, a view |
| `x[:, ::2]` (every other column) | (3, 2) | (16, 8) | yes, a view |
| `x.reshape(12)` | (12,) | (4,) | yes, a view (x is contiguous) |
| `x.astype(np.float64)` | (3, 4) | (32, 8) | no, a copy |
| `np.broadcast_to(np.ones(4, np.float32), (3, 4))` | (3, 4) | (0, 4) | yes: stride 0 repeats a row without copying it |

A **copy** has its own buffer; writing to it never changes the original. `np.shares_memory(a, b)` answers whether two arrays overlap, and `y.flags.c_contiguous` answers whether `y` is laid out row-major.

## The dtype is part of the value

The shape and strides say where each element is; the dtype says how to interpret each run of $b$ bytes. `uint8` holds the integers 0 to 255, and arithmetic on it is done **modulo 256**: 200 + 100 stored in `uint8` is 44, silently. `int64` holds about $\pm 9.2 \times 10^{18}$, enough for any count you will meet in these books. `float32` keeps 24 significant bits, so `np.float32(0.1)` is really 0.100000001490116..., the nearest value it can hold.

numpy 2 raises `OverflowError` when you mix a `uint8` array with a Python integer that does not fit (`x * 256`), but `uint8` arithmetic between arrays, or with a constant that fits (`x * 255 + y`), still wraps without a word. A `uint8` count wraps at 256; an `int64` count does not. Keep that in mind when you build the bigram counter below.

## Broadcasting

With the layout in hand, broadcasting becomes easy to state. An elementwise operation such as `x + y` needs both operands to have the same shape. Broadcasting is numpy's rule for when it may **pretend** they do, by stretching size-1 axes with stride 0 instead of copying. For shapes $a$ (with $p$ axes) and $b$ (with $q$ axes):

1. Let $n = \max(p, q)$. Pad the shorter shape with 1s **on the left** until both have $n$ axes. Equivalently, align the shapes on their **last** axis.
2. For each axis $k$: if $a_k = b_k$, the result has $a_k$. If one of them is 1, the result has the other. Otherwise the shapes do not broadcast, and numpy raises `ValueError`.

So a bias vector of shape (3,) added to a batch of shape (2, 3) acts as (1, 3), stretched to (2, 3): the same 3 numbers are added to every row. The rule never creates data; it only reuses elements. Applied by hand, it agrees with numpy on all 1600 pairs of small shapes, so you can predict any broadcast's shape.

## Broadcasting backward: unbroadcast

Reusing an element forward has a consequence backward. Write $B$ for the operation "broadcast $x$ from shape $s$ to shape $T$". Every element $x_j$ appears at several positions of $B x$. If some later number $L$ depends on $y = Bx$, the chain rule says the sensitivity of $L$ to $x_j$ is the sum of its sensitivities to every copy of $x_j$. The symbols are:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | the operand before broadcasting | shape $s$ |
| $y = Bx$ | the broadcast result | shape $T$ |
| $g$ | the gradient of $L$ with respect to $y$, one number per element of $y$ | shape $T$ |
| $\langle u, v \rangle$ | $\sum_m u_m v_m$, the sum of elementwise products of two same-shaped arrays | scalar |
| $B^\top g$ | `unbroadcast(g, s)`: $g$ summed over every position that holds a copy of the same $x_j$ | shape $s$ |

$$\frac{\partial L}{\partial x_j} = \sum_{\text{positions } m \text{ that copy } x_j} g_m , \qquad\text{equivalently}\qquad \langle Bx, g \rangle = \langle x, B^\top g \rangle .$$

Concretely, $B^\top$ sums $g$ over the leading axes broadcasting added, then over every axis where $s$ has size 1, **keeping** those axes so the result has shape exactly $s$. The right-hand identity is the test: it must hold for every $x$ and $g$. The slogan to remember is that every broadcast forward is a sum backward; `unbroadcast` is the identity an autograd engine applies on every elementwise op.

## Vectorization and the pair code

The bigram counter is where layout and dtype meet speed. A Python loop runs the interpreter once per element. A numpy call runs one compiled loop over the whole buffer. To count byte pairs without a Python loop, turn each adjacent pair $(a, b)$ into one integer

$$k = 256\,a + b, \qquad 0 \le k < 65536 ,$$

count how many times each $k$ occurs with `np.bincount(k, minlength=65536)`, and reshape the 65536 counts to (256, 256). The reshape is free and exact because $256a + b$ is precisely the row-major position of $(a, b)$ in a 256 by 256 table: the address formula with strides $(256, 1)$ counted in elements. Compute $k$ in `int64`, because in `uint8` the product $256a$ does not fit.

Why not `C[a, b] += 1` with index arrays? Indexing with arrays is **buffered**: numpy reads `C[a, b]` for all pairs, adds 1, then writes back, so a pair that occurs 3 times is written 3 times with the same value and counts once. `np.add.at(C, (a, b), 1)` is the unbuffered version; `bincount` is the fast one.

## From counts to probabilities

For a count table $C$ with rows $a$, the probability that $b$ follows $a$ is the row-normalized count $P_{ab} = C_{ab} / \sum_{b'} C_{ab'}$. In numpy that is `C / C.sum(axis=1, keepdims=True)`: the row sums keep shape (256, 1), so broadcasting divides each row by its own total. A row with no counts has total 0, and $0/0$ is NaN; this exercise keeps such rows at 0. (The real byte bigram adds smoothing so that no row is empty.)

## uv projects

The last piece is the environment that runs all of this. A **uv project** is a directory with a `pyproject.toml`. Its `[project].dependencies` list says what the code needs (`numpy>=2.0`). `uv add` edits that list and writes **`uv.lock`**, which pins the exact version and file hash of every package. `uv run <cmd>` first makes the project's own environment (`.venv/` in the project directory) match the lock, then runs `<cmd>` inside it. So `uv run` on your laptop and in CI runs the same numpy.

Commit `pyproject.toml` and `uv.lock`; never commit `.venv/` (list it in `.gitignore`). Installing numpy by hand into some other Python is invisible to `uv run`.

## Working it by hand

Before writing any code, work each function on small inputs. These numbers are also the first test case of each function.

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

**Layout.** `x = np.arange(12, dtype=np.float32).reshape(3, 4)` has strides (16, 4). Its transpose `x.T` has shape (4, 3) and strides (4, 16), so `x.T[1, 0]` sits at byte $1 \cdot 4 + 0 \cdot 16 = 4$, which is `x[0, 1]`: correct as an array. But a C function given `x.T`'s buffer pointer walks bytes 0, 4, 8, ... and reads 0, 1, 2, 3, ..., the elements of `x` in row order, while the row-major (4, 3) array it was promised starts 0, 4, 8. `as_c_float32(x.T)` therefore copies into a fresh buffer that reads 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11. And `as_c_float32` of the `float64` array [[0.1, 0.2, 0.3], [1, 2, 3]] (strides (24, 8)) is a `float32` copy with strides (12, 4) whose first element is 0.100000001490116.

**Bigram counts of "banana".** The bytes are 98 97 110 97 110 97 (b a n a n a). The 5 adjacent pairs and their codes $k = 256a + b$ are:

| pair | a, b | k |
|---|---|---|
| ba | 98, 97 | 25088 + 97 = 25185 |
| an | 97, 110 | 24832 + 110 = 24942 |
| na | 110, 97 | 28160 + 97 = 28257 |
| an | 97, 110 | 24942 |
| na | 110, 97 | 28257 |

`bincount` gives 1 at 25185, 2 at 24942, 2 at 28257. Reshaped: $C_{98,97} = 1$, $C_{97,110} = 2$, $C_{110,97} = 2$, every other entry 0, total 5 = 6 - 1.

**Row normalization.** For $C = [[1, 3], [1, 1]]$ the row sums are 4 and 2, so $P = [[1/4, 3/4], [1/2, 1/2]]$. The wrong version `C / C.sum(axis=1)` divides by [4, 2] laid along the **columns**: $[[1/4, 3/2], [1/4, 1/2]]$, whose first row sums to 1.75. No error is raised: the table is square, so the shapes broadcast.

## Designing the project and the API

The solution's uv project is made with two commands, which write `pyproject.toml` and `uv.lock`:

```bash
uv init --bare primers/lang.01                 # writes primers/lang.01/pyproject.toml
uv add --project primers/lang.01 numpy         # adds numpy to [project].dependencies, writes uv.lock
```

The result is [`pyproject.toml`](../../../../course/solutions/primers/lang.01/pyproject.toml): `requires-python = ">=3.11"` and `dependencies = ["numpy>=2.0"]`. Running the functions under that environment is `uv run --project primers/lang.01 python -c ...` (with `primers/lang.01` on `PYTHONPATH`). Adding pytest for a test run is `uv run --project primers/lang.01 --with pytest pytest ...`, so the project provides numpy and the run adds only pytest.

The five functions have these signatures:

```python
# primers/lang.01/worksheet.py
def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]: ...
    # the shape of x + y for x.shape == a, y.shape == b; ValueError when they do not broadcast.
    # The broadcasting rule by hand; it does not call np.broadcast_shapes.
def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray: ...
    # grad summed back down to `shape`; ValueError when `shape` could not broadcast to grad.shape
def as_c_float32(x) -> np.ndarray: ...
    # x as a C-contiguous float32 array; shares memory with x when x already is one

# primers/lang.01/bigram.py
def bigram_counts(data: bytes) -> np.ndarray: ...
    # int64 (256, 256): C[a, b] = number of i with data[i] == a and data[i + 1] == b; no Python loop
def row_normalize(counts: np.ndarray) -> np.ndarray: ...
    # float64, each nonzero row divided by its sum; zero rows stay zero
```

`broadcast_shape` must implement the rule itself; the tests run it with numpy's broadcasting helpers switched off, because the exercise is the rule. `bigram_counts` must contain no `for`, `while`, or comprehension, nor may any helper in `bigram.py` it calls, because vectorizing is the point.

## Building it step by step

Start from the principles and each function body follows almost by itself.

1. Broadcasting is defined axis by axis from the right. So `broadcast_shape` first makes both shapes the same length by padding on the left, then decides each axis independently: equal sizes stay, a 1 yields to the other size, anything else is an error. No numpy call is needed, and none is used, because the rule is the point.
2. `unbroadcast` is the adjoint $B^\top$. Broadcasting did two things: it prepended axes, and it stretched size-1 axes. The adjoint undoes them in reverse order of how they look: sum away the prepended axes (they did not exist in $x$), then sum each stretched axis while keeping it as size 1. Whatever remains must be exactly `shape`, or `shape` could not have broadcast to `grad.shape`.
3. `as_c_float32` must return something a C function can read through `const float *`: dtype float32 and C-contiguous strides. numpy can already decide whether `x` satisfies both and copy only if it does not; the only job is to ask for both conditions at once and not use a helper that changes the shape.
4. `bigram_counts` must count without a Python loop. The pair code $k = 256a + b$ turns a 2-D count into a 1-D histogram, which `np.bincount` computes in one compiled pass. The codes must be built in a dtype wide enough for 65535, so the bytes are widened to `int64` first.
5. `row_normalize` divides each row by its own total, which needs the totals as a column, shape (256, 1), for broadcasting to pair row $a$ with total $a$. A zero total must leave its row at zero, so the division is masked rather than repaired afterwards.

Try to predict each body before reading it.

### broadcast_shape

```python
def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    n = max(len(a), len(b))
    pa = (1,) * (n - len(a)) + tuple(a)          # pad on the LEFT: align on the last axis
    pb = (1,) * (n - len(b)) + tuple(b)
    out = []
    for axis, (x, y) in enumerate(zip(pa, pb)):
        if x == y or y == 1:                     # equal, or b stretches: keep a's size
            out.append(x)
        elif x == 1:                             # a stretches: take b's size
            out.append(y)
        else:
            raise ValueError(f"shapes {tuple(a)} and {tuple(b)} do not broadcast: "
                             f"axis {axis - n} has sizes {x} and {y}")
    return tuple(out)
```

The loop is over axes (at most a handful), not over data, so it is not a vectorization concern. The ordering of the tests matters for size 0: `(1, 0)` takes the `x == 1` branch and gives 0, matching the hand example of `(4, 1)` with `(0,)`. The error names the axis counted from the right (`axis - n` is negative), which is how numpy reports it and how a reader thinks about alignment.

### unbroadcast

```python
def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    g = np.asarray(grad)
    shape = tuple(shape)
    extra = g.ndim - len(shape)
    if extra < 0:                                # broadcasting never removes axes
        raise ValueError(f"grad has {g.ndim} axes, fewer than the target shape {shape}")
    if extra:
        g = g.sum(axis=tuple(range(extra)))      # the leading axes broadcasting added
    # axes stretched from 1: sum them, but keep each as size 1 so the result has `shape`
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1)
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    if g.shape != shape:                         # e.g. (2,) is not a source of (2, 3)
        raise ValueError(f"{shape} does not broadcast to {np.shape(grad)}")
    return g
```

Trace the hand example's `(2, 3)` ones to `(2, 1)`: `extra = 0`; axis 1 has target size 1 and grad size 3, so `axes = (1,)`; the sum with `keepdims` gives `[[3], [3]]`. To `(3,)`: `extra = 1`, summing axis 0 gives `[2, 2, 2]`, no size-1 axes, shape matches. To `(2,)`: `extra = 1`, summing axis 0 gives shape `(3,)`, which is not `(2,)`, so it raises. The final shape check is what rejects shapes that never broadcast, without a separate validation pass. When the target is `()`, `extra = 2` and the sum over both axes gives a numpy scalar of shape `()` holding 6.

### as_c_float32

```python
def as_c_float32(x) -> np.ndarray:
    # Not np.ascontiguousarray: it returns at least 1-D, so a 0-d scalar
    # would come back with shape (1,).
    return np.asarray(x, dtype=np.float32, order="C")
```

`np.asarray` returns its argument unchanged when it already has the requested dtype and order, and makes a new array otherwise. Both conditions are needed: `dtype` alone passes a float32 transposed view through untouched, and `order` alone leaves float64. On the hand example's `x.T`, `order="C"` forces a copy laid out as 0, 4, 8, 1, 5, 9, ...; on an already-right array, `np.shares_memory(x, as_c_float32(x))` is true.

### bigram_counts

```python
def bigram_counts(data: bytes) -> np.ndarray:
    x = np.frombuffer(data, dtype=np.uint8).astype(np.int64)   # view the bytes, then widen
    pairs = x[:-1] * 256 + x[1:]                 # k = 256a + b for every adjacent pair
    return np.bincount(pairs, minlength=256 * 256).astype(np.int64, copy=False).reshape(256, 256)
```

`np.frombuffer` reads the `bytes` object without copying; `.astype(np.int64)` is the one copy, and it is required, because in `uint8` the product `x * 256` would wrap. `x[:-1]` and `x[1:]` are two views of the same buffer offset by one element, so the pairs are formed without any loop. `minlength=65536` makes the histogram full length even when the highest code never occurs, so the reshape always succeeds; for empty or one-byte input, `pairs` is empty and the result is all zeros. `astype(..., copy=False)` guarantees the documented dtype at no cost when `bincount` already returned `int64`. The reshape is free and exact because $256a + b$ is the row-major position of $(a, b)$. On "banana" this produces the three non-zero entries you computed by hand.

### row_normalize

```python
def row_normalize(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64)
    totals = c.sum(axis=1, keepdims=True)        # shape (256, 1): one total per row, as a column
    # divide only where the total is positive; elsewhere keep the zeros of `out`
    return np.divide(c, totals, out=np.zeros_like(c), where=totals > 0)
```

`keepdims=True` is what makes the division pair each row with its own total; without it, you get the square-table bug from the hand example. The `where` mask broadcasts like `totals`, so a row with total 0 is never divided and keeps the zeros of `out`: no `0/0`, no NaN, and no warning. Converting to `float64` first makes the result's dtype independent of the input's.

The full solution is in [`worksheet.py`](../../../../course/solutions/primers/lang.01/worksheet.py) and [`bigram.py`](../../../../course/solutions/primers/lang.01/bigram.py).

### Why this design

In cost, `bigram_counts` is $O(N)$ in the input length with two passes over it (the widened copy and `bincount`) and a fixed 512 KB table; `row_normalize` is $O(256^2)$; `broadcast_shape` and `unbroadcast` are linear in the number of axes, plus the sums.

`bincount` beats `np.add.at`: both count correctly, but `add.at` performs an unbuffered scatter per element and is many times slower, and `C[a, b] += 1` is simply wrong. `np.unique(..., return_counts=True)` would also count, but it sorts ($O(N \log N)$) and returns only the codes present, which then need scattering into the table. The dense code space of 65536 makes a histogram the natural fit.

Calling `np.asarray` with both arguments is better than a hand-written "if contiguous and float32 then return x else copy": numpy already applies exactly that test, including for subclasses and non-array inputs such as lists.

One tradeoff is accepted deliberately: the `int64` copy of the input costs 8 bytes per input byte. For the corpora in these books this is small; for gigabytes, you would process in chunks and add the tables.

## What can go wrong

A correct solution handles a specific set of cases. Broadcasting must align on the right, so (3,) with (3, 1) gives (3, 3), the bias-add shape in every layer; incompatible shapes such as (3,) with (4,) must raise, because a shape bug must fail loudly; and the hand rule must agree with numpy on all 1600 pairs of shapes with up to 3 axes of sizes 1 to 3. `unbroadcast` must keep size-1 axes (a target of (2, 1) stays (2, 1), so a bias gradient keeps the bias's shape), must sum leading batch axes away ((4, 2, 3) to (3,) and to (1, 3)), must reject a shape that never broadcast ((2,) is not a source of (2, 3), so no gradients are invented), and must satisfy the adjoint identity $\langle Bx, g\rangle = \langle x, B^\top g\rangle$ on every small pair of shapes with seeded PCG32 values, the law autograd relies on.

`as_c_float32` must convert float64 to float32 with row-major bytes, share an already-right array so that a C call costs no copy, copy a transposed view `x.T` into fresh row-major order, densify a strided slice `x[:, ::2]` (C has no stride argument for it), and keep a 0-d input at shape `()`, which matters for scalar tensors in safetensors files. `bigram_counts` must count every repeated pair ("aaaa" counts aa three times), must not wrap at 256 (999 repeats read as 999, dtype `int64`), must return all zeros for empty and one-byte inputs, must give bytes 128 and 255 their own rows because UTF-8 text uses high bytes, must have row $a$ sum to the count of $a$ in `data[:-1]`, and must equal the obvious Python loop on random bytes, since the loop is the definition. `row_normalize` must divide rows, not columns, so that rows sum to 1 on an uneven table, and must leave empty rows at zero. Finally, the project's `[project].dependencies` must list numpy.

Most of the bugs these cases catch are silent. The following table collects the common ones.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Aligning shapes on the left (padding on the right) | (3,) with (3, 1) gives (3, 1) instead of (3, 3) | pads the shorter shape with 1s on the left before comparing |
| Taking the larger size when two sizes differ and neither is 1 | (3,) with (4,) "broadcasts" to (4,) | raises `ValueError` unless the sizes are equal or one is 1 |
| Summing without `keepdims=True` in `unbroadcast` | a (2, 1) bias gets a (2,) gradient; the update `b -= lr * g` then broadcasts `b` up to (2, 2) | sums the size-1 axes with `keepdims=True` and then checks the result shape |
| `np.array(x, dtype=np.float32)` or `x.astype(np.float32)` | correct values, but a full copy on every call even when `x` is already right | `np.asarray`, which returns `x` itself when it already is C-contiguous float32 |
| `np.asarray(x, dtype=np.float32)` | a transposed view passes through unchanged; C reads `x`'s rows instead | passes `order="C"`, so a non-contiguous view is copied |
| `np.ascontiguousarray(x, dtype=np.float32)` | right for every array of 1 or more dimensions, but it returns at least 1-D: a 0-d scalar comes back with shape `(1,)` | uses `np.asarray(..., order="C")`, which keeps 0-d as 0-d |
| `C[a, b] += 1` with index arrays | each distinct pair counts once ("aaaa" gives 1, not 3) | counts with `np.bincount`, which counts every occurrence |
| A `uint8` count table, or `uint8` pair codes | 999 repeats read as 231; codes `a * 255 + b` wrap | widens the bytes to `int64` before forming codes; `bincount` returns integer counts |
| `C / C.sum(axis=1)` without `keepdims` | no error on a square table, rows do not sum to 1 | row totals with `keepdims=True`, shape (256, 1) |
| Dividing an empty row by its zero total | NaN rows that spread into every later sum | `np.divide(..., out=zeros, where=totals > 0)` never divides by zero |
| `pip install numpy` into a global Python instead of the project | works on your laptop, fails in CI and for `uv run` | `pyproject.toml` declares `numpy>=2.0` and `uv.lock` pins it |

> **Warning:** A square table hides the missing `keepdims`. Test row normalization on an uneven table, where wrong-axis division shows up immediately.

## Where this leads

This primer has no code call site of its own, but the ideas recur at once. In *Build Your Own LLM System*, the [byte bigram](../../build-an-llm/01-foundations/01-byte-bigram.md) is trained with `bigram_counts` over a corpus followed by row normalization with add-one smoothing, and the [tensor and broadcasting backward](../../build-an-llm/01-foundations/02-tensor-and-broadcasting-backward.md) section makes `unbroadcast` the backward of every broadcasting op, testing it with the adjoint identity above. In *Mathematics for Computing*, [Vectors, matrices, row-major layout, and matmul](../../mathematics-for-computing/04-linear-algebra/01-vectors-matrices-and-matmul.md) uses row-major layout and strides to explain the C exercise's matrix buffers, where `tl_matmul_f32` reads row-major float32 with an explicit leading dimension: the strides from this section, counted in elements. Python and C are checked separately there through fixture files. When the system's runtime hands arrays to C, `as_c_float32` is the argument conversion it relies on.

The setup step imports these primer modules as a toolchain check, so a broken Python environment shows up at the start rather than later. The next section, [Shell, git, make, and processes](02-shell-git-make.md), covers the shell, exit codes, signals, and make.

### Further reading

- numpy user guide, "Broadcasting", "Copies and views", and "Data type objects" (numpy.org/doc).
- Harris et al., "Array programming with NumPy", *Nature* 585, 2020.
- uv docs, "Working on projects" and "Workspaces" (docs.astral.sh/uv). Workspaces give one lock for several packages, as the system's `python/` project uses later.
- PyTorch `Tensor.sum_to_size` and ATen `at::sum_to` (`aten/src/ATen/ExpandUtils.h`): the same reduction as `unbroadcast`, run on every broadcasting op's gradient.
- HIPS autograd `unbroadcast` (`autograd/numpy/numpy_vjps.py`): the same idea in a numpy-only autograd.
- PyTorch `Tensor.contiguous()` and DLPack (dmlc/dlpack `include/dlpack/dlpack.h`): zero-copy exchange of strided buffers between frameworks.
- numpy docs, `numpy.bincount`; and `np.unique(..., return_counts=True)` or `collections.Counter` for counting arbitrary keys rather than dense integer codes.

## Key terms

- **buffer**: the flat block of memory that holds an array's elements.
- **stride**: the number of bytes to move along an axis to reach the next index on it.
- **C-contiguous**: row-major layout, with the last axis packed and each earlier stride equal to the next stride times the next size.
- **view**: a new array object with its own metadata over an existing buffer.
- **copy**: an array with its own buffer, independent of the original.
- **broadcasting**: numpy's rule for treating differently shaped operands as one shape by stretching size-1 axes with stride 0.
- **unbroadcast**: the adjoint of broadcasting, which sums a gradient back to the operand's shape.
- **uv.lock**: the file that pins the exact version and hash of every package in a uv project.

## Summary

- A numpy array is one flat buffer plus dtype, shape, and strides; transposes and slices change only the metadata, so they are views that share the buffer.
- Broadcasting aligns shapes on the right, pads with 1s on the left, and stretches size-1 axes; you can predict any broadcast's shape by hand.
- Every broadcast forward is a sum backward: `unbroadcast` is the adjoint, checked by $\langle Bx, g\rangle = \langle x, B^\top g\rangle$.
- The dtype is part of the value: a `uint8` count wraps at 256, an `int64` count does not.
- The pair code $256a + b$ plus `np.bincount` counts bigrams in one compiled pass; masked `np.divide` normalizes rows without NaN.
- A uv project pins the environment: `pyproject.toml` declares numpy, `uv.lock` pins the build, and `uv run` reproduces it anywhere, including CI.

## Review questions

1. Why does `broadcast_shape` pad on the left rather than the right?

   <details><summary>Answer</summary>

   Broadcasting aligns shapes on their last axis: a (3,) bias added to a (2, 3) batch must line up with the 3 features, not the 2 rows. Padding the shorter shape on the left with 1s is the same as right alignment. Padding on the right would turn (3,) with (3, 1) into (3, 1) instead of (3, 3).

   </details>

2. In `unbroadcast`, why are the leading axes summed without `keepdims`, but the size-1 axes with it?

   <details><summary>Answer</summary>

   The leading axes did not exist in the operand; broadcasting created them, so they must disappear. The size-1 axes did exist, with size 1, so the gradient must keep them at size 1 for its shape to equal the operand's shape.

   </details>

3. Why does `bigram_counts` widen to `int64` before multiplying, instead of after?

   <details><summary>Answer</summary>

   The multiplication is where the overflow happens. In `uint8`, `98 * 256` wraps to 0, so the codes are wrong before anything else runs. Widening first gives every product room up to 65535.

   </details>

4. What would `row_normalize` return for an all-zero row if it used plain `c / totals`?

   <details><summary>Answer</summary>

   NaN in every entry of that row (0/0), with a runtime warning. One NaN spreads into every later sum that touches it. The masked `np.divide` with a zero-filled `out` leaves the row at zero instead.

   </details>

5. When does `as_c_float32` return the very same object it was given?

   <details><summary>Answer</summary>

   When `x` is already a numpy array with dtype float32 and C-contiguous layout. `np.asarray` then has nothing to change and returns `x` itself, so a C call on an array that is already right costs no copy.

   </details>
