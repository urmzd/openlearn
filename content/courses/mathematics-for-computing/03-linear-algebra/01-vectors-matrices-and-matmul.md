# Vectors, matrices, row-major layout, and matmul

A byte bigram model has a 256 by 256 weight table. For each input byte, its logits are one row of that table. Writing that lookup as `one_hot(ids) @ weight` turns it into a matrix product, and that gives you a small, testable example of the operation that runs through every neural network in this book and the ones that follow it. This section builds that product in Python and NumPy, with the array layout rules made explicit so that callers can pass slices without silently reading padding.

The implementation you build here is deliberately slow. Its job is to be the reference: it pins down index order, float32 rounding, how a transpose is read, and what happens when the output buffer already holds garbage. Faster code, including the optional C kernel in [Cache-blocked, packed, batch-invariant matmul in C](../../build-an-llm/p09-kernels/04-tiled-batch-invariant-matmul.md) in *Build Your Own LLM System*, is checked against it. The two implementations share JSON parity vectors and communicate only through files; there is no native binding.

You need NumPy arrays, shapes, indexing, and views, covered in [Python and NumPy](../../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md) in *Programming Languages*. The finished solution is [`matmul.py`](../../../../course/solutions/python/tinyllm/linalg/matmul.py), whose `matmul_f32` is a readable row-major float32 matrix product with transpose, scaling, and strided views. Its contract is short: shapes must compose, the output shape and dtype are checked, and `beta == 0` overwrites the output without reading it.

> **Learning objectives**
> - Compute a matrix product by hand from the definition $C_{ij}=\sum_r A_{ir}B_{rj}$ and state its output shape.
> - Explain how row-major storage, strides, and a transpose flag change which stored element is read, without copying data.
> - Explain why `beta == 0` must skip reading the old output, and why float32 results are compared with a tolerance rather than bit for bit.
> - Implement and validate a scaled float32 product `C <- alpha * A @ op(B) + beta * C`.

## The product and its shape

For `A` with shape `(m, k)` and `B` with shape `(k, n)`, the output has shape `(m, n)` and

$$
C_{ij}=\sum_{r=0}^{k-1} A_{ir}B_{rj}.
$$

Each output entry is a sum over the shared axis, the one of length `k`. That is the only shape constraint: `A`'s column count must equal `B`'s row count. Everything else about the output is read off the operands.

How the numbers sit in memory is a separate question. A row-major matrix stores each row contiguously. In a NumPy view, though, the byte stride between rows can exceed `k * itemsize`: slice a few columns out of a wider array and the rows are still spaced as far apart as the original rows were. An implementation must use the view's strides rather than assume the array is packed. A transpose is a similar trick in the other direction: it changes the logical axes while keeping the underlying storage, so the result must agree whether the right operand is supplied directly or supplied transposed with a flag saying so.

Real kernels compute a scaled version of the product, `C <- alpha * A @ op(B) + beta * C`, where `op(B)` is `B` or its transpose. The special value `beta == 0` is an overwrite rule, and it has a subtle consequence: the old contents of `C` must not be read at all. Multiplying a NaN by zero still produces NaN, so an output buffer that happened to contain NaN would poison a result that was supposed to replace it. Finally, floating-point reductions can differ in their last bits when the accumulation order changes, so comparisons against this reference use a float64 reference and a tolerance rather than bit equality.

## Working it by hand

Let `A = [[1, 2, 3], [4, 5, 6]]` and `B = [[7, 8], [9, 10], [11, 12]]`. Each of the four outputs is a row of `A` dotted with a column of `B`:

| Output | Calculation | Value |
|---|---|---:|
| `C[0, 0]` | `1*7 + 2*9 + 3*11` | 58 |
| `C[0, 1]` | `1*8 + 2*10 + 3*12` | 64 |
| `C[1, 0]` | `4*7 + 5*9 + 6*11` | 139 |
| `C[1, 1]` | `4*8 + 5*10 + 6*12` | 154 |

The result is `[[58, 64], [139, 154]]`. You get the same values when `B.T` is supplied with the transpose option set. The shared JSON parity vectors hold more rectangular cases, including non-contiguous views, so the Python and optional C implementations can be compared without linking their runtimes.

## Building it step by step

Start from the definition and let each requirement fall out of it.

The definition $C_{ij}=\sum_r A_{ir}B_{rj}$ is a sum over the shared axis, so the only shape check is that `A`'s column count equals `op(B)`'s row count; `m` and `n` are read off the operands. `op(B)` is either `B`, of shape `(k, n)`, or `B` stored transposed, of shape `(n, k)`. The transpose needs no copy: element $(q, j)$ of `op(B)` is `B[j, q]` when `trans_b` is set. Swapping two indices is the whole transpose.

Strided views are handled by never computing addresses by hand. Indexing a NumPy view with `aa[i, q]` uses that view's own strides, so a slice with padding between rows reads the right elements. The scaled form makes `beta == 0` mean "overwrite", and because $0 \cdot \text{NaN} = \text{NaN}$, the only way to honour that is to not evaluate `beta * C` at all when `beta` is zero. And because the result is float32, every intermediate is rounded to float32 the same way a C kernel with a float accumulator would round it. That keeps the Python version usable as the semantic reference for the C one, up to accumulation-order rounding.

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

Then comes the triple loop, which is the definition written out:

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

The full solution is [`matmul.py`](../../../../course/solutions/python/tinyllm/linalg/matmul.py).

### Why this design

The loop costs $O(mnk)$ multiply-adds and $O(1)$ extra space beyond the output. It is slow on purpose: its job is to define the semantics (index order, float32 rounding, the `beta == 0` rule) that faster code must match. Calling `aa @ bb` would be faster, but it would accumulate in an order and precision NumPy chooses, and it would hide how the transpose and overwrite rules work. A blocked or vectorised kernel belongs in the optional C kernel, where it is compared against this reference with a tolerance.

## What can go wrong

A correct implementation has to survive a handful of specific cases. The hand example pins the definition to four exact values. The shared C parity vectors keep the Python reference aligned with the optional C kernel, which reads the same JSON files. Rectangular shapes and strided views catch the assumption that arrays are packed or contiguous. Transpose, scaling, and an empty inner dimension exercise the transpose interpretation, the alpha and beta behaviour, and the empty sum, which must be 0: with `k == 0` the function returns `beta * C`, or zeros when `beta == 0`. Calling with `beta == 0` and NaN in the old `C` makes sure stale NaNs in a caller's buffer cannot contaminate fresh results. And an output of the wrong shape must fail at the boundary with a useful error.

Each of those cases corresponds to a mistake that is easy to make. The table lists them with the symptom you would see and the line of the solution that prevents it.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Treating the last two dimensions as packed | Sliced arrays produce wrong values or modify neighbouring storage. | Reads every element through NumPy indexing, which honours the view's strides. |
| Reversing the transpose convention | Square cases pass while rectangular weights are indexed incorrectly. | Reads the shape as `(n, k)` and the element as `bb[j, q]` when `trans_b` is set. |
| Evaluating `beta * C` when `beta` is zero | NaNs in the old output survive an intended overwrite. | Uses `prior = 0` without touching `cc[i, j]` when `be == 0`. |
| Comparing float32 results bit for bit | A valid reduction order fails only because of rounding. | Is compared against a float64 reference with a tolerance. |
| Accepting an output with the wrong shape | Results are truncated or broadcast into an unintended shape. | Raises `ValueError` unless `C` is a writable float32 array of exactly `(m, n)`. |

> **Warning:** Square test matrices hide transpose bugs. If `m == n == k`, reading `bb[q, j]` where you meant `bb[j, q]` still produces a matrix of the right shape. Always test with rectangular operands.

## Where this leads

The product shows up immediately. [Byte bigram: counts, logits, safetensors v0](../../build-an-llm/p00-foundations/00-byte-bigram.md) in *Build Your Own LLM System* uses it to turn each byte id into its bigram-logit row through the weight matrix, and [Training loop and the autograd bigram](../../build-an-llm/p00-foundations/05-training-loop-and-the-autograd-bigram.md) reuses it in the training path. [Cache-blocked, packed, batch-invariant matmul in C](../../build-an-llm/p09-kernels/04-tiled-batch-invariant-matmul.md) builds the optional C kernel and compares it with this reference through the shared JSON vectors. Within this chapter, the next section, [Gaussian elimination and LU with partial pivoting](02-gaussian-elimination-and-lu.md), uses matrix storage and products to solve linear systems.

### Further reading

- BLAS GEMM, for example Netlib BLAS `dgemm`: production matrix multiply, where blocking and vector instructions keep matrix tiles in cache.
- The standalone C kernel, [`matmul.c`](../../../../course/solutions/c/src/kernels/matmul.c), which checks a second implementation of the same semantics through files.

## Key terms

- **Row-major layout**: storage in which each row of a matrix is contiguous in memory.
- **Stride**: the number of bytes between consecutive elements along one axis of an array view; the row stride can exceed `k * itemsize`.
- **View**: an array that shares storage with another, reinterpreted through its own shape and strides.
- **`op(B)`**: the right operand as the product sees it, either `B` or $B^\top$, selected by a flag rather than a copy.
- **Overwrite rule**: the convention that `beta == 0` means the old output is never read.

## Summary

- $C_{ij}=\sum_r A_{ir}B_{rj}$; the only shape constraint is that the inner dimensions agree, and the output is `(m, n)`.
- A transpose is an index swap (`bb[j, q]`), and strided views are handled by letting NumPy indexing apply the view's strides.
- `beta == 0` must skip reading `C`, because $0 \cdot \text{NaN} = \text{NaN}$.
- Rounding each step to float32 makes the Python loop the semantic reference for faster kernels, compared with a tolerance against a float64 reference.
- The empty inner dimension gives an empty sum of 0.

## Review questions

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

5. Compute `C[1, 0]` for the hand example, and say which row of `A` and column of `B` it uses.

   <details><summary>Answer</summary>

   Row 1 of `A`, `(4, 5, 6)`, dotted with column 0 of `B`, `(7, 9, 11)`: $28 + 45 + 66 = 139$.

   </details>
