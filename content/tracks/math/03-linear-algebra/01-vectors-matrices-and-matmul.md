# Vectors, matrices, row-major layout, and matmul in Python

## Overview

| | |
|---|---|
| **Module** | `M03.1` · build · Python · Pass 1 |
| **Solution** | [`matmul.py`](../../../../course/solutions/python/tinyllm/linalg/matmul.py): `matmul_f32`, a readable row-major float32 matrix product with transpose, scaling, and strided views |
| **Contract** | Shapes must compose; output shape and dtype are checked, and `beta == 0` overwrites without reading the old output |
| **Needs** | Python arrays and indexing from `lang.01` |
| **Used by** | `L0.0` uses it for bigram logits; `L0.5` uses it for the same row-selection operation during training |

## How to read this chapter

Sections 1 to 3 define the product and work it by hand. Section 4 lists the cases a correct implementation must handle. Section 5 walks through the solution and explains each choice. Try to predict each line of the loop before reading it.

## 1. Why now

A byte bigram model has a 256 by 256 weight table. For each input byte, its logits are one row of that table. Writing that operation as `one_hot(ids) @ weight` turns a lookup into a matrix product and gives us a small, testable example of the operation used throughout neural networks. The solution implements the operation in Python and NumPy, with the array layout rules explicit so callers can pass slices without silently reading padding.

The implementation is also a reference for optional standalone C kernels in `L9.1`. Both implementations consume the same JSON parity vectors, but communicate only through files. There is no native binding in this module.

## 2. Principles

For `A` with shape `(m, k)` and `B` with shape `(k, n)`, the output has shape `(m, n)` and

\[
C_{ij}=\sum_{r=0}^{k-1} A_{ir}B_{rj}.
\]

A row-major matrix stores each row contiguously. In a NumPy view, the byte stride between rows can exceed `k * itemsize`; the implementation must use the view's strides rather than assume the array is packed. A transpose changes the logical axes while retaining the underlying storage, so the result must agree whether the right operand is supplied directly or transposed.

For the scaled operation, `C <- alpha * A @ op(B) + beta * C`, `beta == 0` is an overwrite rule. The old contents of `C` must not be read: multiplying a NaN by zero still produces NaN. Floating point reductions can differ in their final bits when accumulation order changes, so comparisons against the solution use a float64 reference and a tolerance rather than bit equality.

## 3. Worked example

Let `A = [[1, 2, 3], [4, 5, 6]]` and `B = [[7, 8], [9, 10], [11, 12]]`. Then

| Output | Calculation | Value |
|---|---|---:|
| `C[0, 0]` | `1*7 + 2*9 + 3*11` | 58 |
| `C[0, 1]` | `1*8 + 2*10 + 3*12` | 64 |
| `C[1, 0]` | `4*7 + 5*9 + 6*11` | 139 |
| `C[1, 1]` | `4*8 + 5*10 + 6*12` | 154 |

Thus the result is `[[58, 64], [139, 154]]`. The same values result when `B.T` is provided with the transpose option. Shared parity vectors in JSON hold additional rectangular cases, including non-contiguous views, so implementations in Python and optional C can be compared without linking their runtimes.

## 4. What a correct solution must handle

| Case | Why it matters |
|---|---|
| The hand example above | Pins the definition to the four exact values. |
| Shared C parity vectors | Keeps the Python reference aligned with the optional C kernel, which reads the same JSON files. |
| Shapes and strided views | Catches assumptions that arrays are packed or contiguous. |
| Transpose, scaling, and an empty inner dimension | Exercises transpose interpretation, alpha/beta behavior, and the empty sum (which must be 0). |
| `beta == 0` with NaN in the old `C` | Prevents stale NaNs in a caller's output buffer from contaminating fresh results. |
| An output of the wrong shape | Makes shape mistakes fail at the boundary with a useful error. |

## 5. The solution, step by step

### From first principles

1. The definition $C_{ij}=\sum_r A_{ir}B_{rj}$ is a sum over the shared axis, so the only shape constraint is that `A`'s column count equals `op(B)`'s row count. Everything else (`m`, `n`) is read off the operands.
2. `op(B)` is either `B` (shape `(k, n)`) or `B` stored transposed (shape `(n, k)`). The transpose does not need a copy: element $(q, j)$ of `op(B)` is `B[j, q]` when `trans_b` is set. Swapping two indices is the whole transpose.
3. Strided views are handled by never computing addresses by hand. Indexing a NumPy view with `aa[i, q]` uses that view's own strides, so a slice with padding between rows reads the right elements.
4. The scaled form `alpha * A @ op(B) + beta * C` makes `beta == 0` mean "overwrite". Because $0 \cdot \text{NaN} = \text{NaN}$, the only way to honour that rule is to not evaluate `beta * C` at all when `beta` is zero.
5. The result is float32, so every intermediate is rounded to float32 the same way a C kernel with a float accumulator would. That keeps the Python version usable as the semantic reference for the C one, up to accumulation-order rounding.

### The code, annotated

Validation comes first, so that every error is raised before any output is written:

```python
aa = np.asarray(a)
bb = np.asarray(b)
if aa.dtype != np.float32 or bb.dtype != np.float32:
    raise TypeError("A and B must have dtype float32")   # no silent upcasts
if aa.ndim != 2 or bb.ndim != 2:
    raise ValueError("A and B must be two-dimensional")
m, k = aa.shape
if trans_b:
    n, bk = bb.shape          # B is stored as [N, K]
else:
    bk, n = bb.shape          # B is stored as [K, N]
if k != bk:
    raise ValueError(f"inner dimensions differ: A has {k}, op(B) has {bk}")
if c is None:
    cc = np.empty((m, n), dtype=np.float32)   # garbage is fine: beta is 0 by default
else:
    cc = np.asarray(c)
    # a caller's buffer must match exactly; broadcasting or truncation would hide bugs
    if cc.dtype != np.float32 or cc.shape != (m, n) or not cc.flags.writeable:
        raise ValueError(f"C must be a writable float32 array of shape {(m, n)}")
```

Then the triple loop, which is the definition written out:

```python
al, be = np.float32(alpha), np.float32(beta)
for i in range(m):
    for j in range(n):
        acc = np.float32(0.0)                  # empty inner dimension gives 0
        for q in range(k):
            # the transpose is an index swap, not a copy
            rhs = bb[j, q] if trans_b else bb[q, j]
            # round the product and the running sum to float32 each step
            acc = np.float32(acc + np.float32(aa[i, q] * rhs))
        # beta == 0 means overwrite: never read the old C, it may hold NaN
        prior = np.float32(0.0) if be == 0 else np.float32(be * cc[i, j])
        cc[i, j] = np.float32(al * acc + prior)
return cc
```

Full solution: [`matmul.py`](../../../../course/solutions/python/tinyllm/linalg/matmul.py).

### Why this solution

The loop costs $O(mnk)$ multiply-adds and $O(1)$ extra space beyond the output. It is deliberately slow: its job is to define the semantics (index order, float32 rounding, the `beta == 0` rule) that faster code must match. Calling `aa @ bb` would be faster but would accumulate in an order and precision NumPy chooses, and it would not show how the transpose and overwrite rules work. A blocked or vectorised kernel belongs in the optional C module, where it is compared against this reference with a tolerance.

### Explain it back

1. Why does the solution test `be == 0` instead of always computing `be * cc[i, j]`?

   <details><summary>Answer</summary>

   Because `0 * NaN` is NaN. A freshly allocated or reused output buffer can contain NaN, and an overwrite must not depend on what was there before.

   </details>

2. With `trans_b=True` and `B` of shape `(n, k)`, why is the element read `bb[j, q]`?

   <details><summary>Answer</summary>

   `op(B) = B^T`, so $\text{op}(B)_{qj} = B_{jq}$. The stored row index is the output column `j`; the stored column index is the shared axis `q`.

   </details>

3. Why does a padded row slice still give the right answer without any stride arithmetic in the code?

   <details><summary>Answer</summary>

   NumPy indexing on a view applies that view's strides. The solution never assumes a packed layout, so the row stride can exceed `k * 4` bytes without changing which elements are read.

   </details>

4. What should `matmul_f32` return when `k == 0`?

   <details><summary>Answer</summary>

   An `(m, n)` matrix of `beta * C` (or zeros when `beta == 0`), because the empty sum is 0. The accumulator starts at 0 and the inner loop never runs.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Treating the last two dimensions as packed | Sliced arrays produce wrong values or modify neighboring storage. | Reads every element through NumPy indexing, which honours the view's strides; see step 3. |
| Reversing the transpose convention | Square cases may pass while rectangular weights are indexed incorrectly. | Reads shape as `(n, k)` and the element as `bb[j, q]` when `trans_b` is set; see step 2. |
| Evaluating `beta * C` when `beta` is zero | NaNs in the old output survive an intended overwrite. | Uses `prior = 0` without touching `cc[i, j]` when `be == 0`; see step 4. |
| Comparing float32 results bit for bit | A valid reduction order fails due only to rounding. | Is compared against a float64 reference with a tolerance, not bit for bit. |
| Accepting an output with the wrong shape | Results are truncated or broadcast into an unintended shape. | Raises `ValueError` unless `C` is a writable float32 array of exactly `(m, n)`. |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.01` | NumPy arrays, shape inspection, indexing, and views. |
| Forward | `L0.0` | Converts each byte ID into its bigram-logit row through the weight matrix. |
| Forward | `L0.5` | Reuses matrix multiplication in the training path. |
| Forward | `L9.1` | Implements an optional standalone C kernel and compares it with this reference through shared JSON vectors. |
| Forward | `M03.2` | Uses matrix storage and products when solving linear systems. |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| triple-loop matmul | BLAS GEMM | blocking and vector instructions keep matrix tiles in cache | Netlib BLAS `dgemm` |
| shared parity vectors | standalone `L9.1` C kernel | checks a second implementation through files | [`matmul.c`](../../../../course/solutions/c/src/kernels/matmul.c) |
