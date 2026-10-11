<!-- ol:module M03.1 -->
# Vectors, matrices, row-major layout, and matmul in Python

## Overview

| | |
|---|---|
| **Module** | `M03.1` · build · Python · Pass 1 |
| **You build** | `python/tinyllm/linalg/matmul.py`: a readable row-major float32 matrix product with transpose, scaling, and strided views |
| **Contract** | Shapes must compose; output shape and dtype are checked, and `beta == 0` overwrites without reading the old output |
| **Tests** | `course/tests/M03.1/test_matmul_numpy.py`: hand calculation, frozen file parity vectors, shape and stride cases, transpose/scaling, NaN overwrite, invalid output |
| **Needs** | Python arrays and indexing from `lang.01` |
| **Used by** | `L0.0` uses it for bigram logits; `L0.5` uses it for the same row-selection operation during training |
| **Milestone** | `MS-P1` |

## 1. Why now

A byte bigram model has a 256 by 256 weight table. For each input byte, its logits are one row of that table. Writing that operation as `one_hot(ids) @ weight` turns a lookup into a matrix product and gives us a small, testable example of the operation used throughout neural networks. This module implements the operation in Python and NumPy, with the array layout rules explicit so callers can pass slices without silently reading padding.

The implementation is also a reference for optional standalone C kernels in `L9.1`. Both implementations consume the same JSON parity vectors, but communicate only through files. There is no native binding in this module.

## 2. Principles

For `A` with shape `(m, k)` and `B` with shape `(k, n)`, the output has shape `(m, n)` and

\[
C_{ij}=\sum_{r=0}^{k-1} A_{ir}B_{rj}.
\]

A row-major matrix stores each row contiguously. In a NumPy view, the byte stride between rows can exceed `k * itemsize`; the implementation must use the view's strides rather than assume the array is packed. A transpose changes the logical axes while retaining the underlying storage, so the result must agree whether the right operand is supplied directly or transposed.

For the scaled operation, `C <- alpha * A @ op(B) + beta * C`, `beta == 0` is an overwrite rule. The old contents of `C` must not be read: multiplying a NaN by zero still produces NaN. Floating point reductions can differ in their final bits when accumulation order changes, so comparisons use a float64 reference and the tolerances in the course helper.

## 3. Worked example

Let `A = [[1, 2, 3], [4, 5, 6]]` and `B = [[7, 8], [9, 10], [11, 12]]`. Then

| Output | Calculation | Value |
|---|---|---:|
| `C[0, 0]` | `1*7 + 2*9 + 3*11` | 58 |
| `C[0, 1]` | `1*8 + 2*10 + 3*12` | 64 |
| `C[1, 0]` | `4*7 + 5*9 + 6*11` | 139 |
| `C[1, 1]` | `4*8 + 5*10 + 6*12` | 154 |

Thus the result is `[[58, 64], [139, 154]]`. The same values result when `B.T` is provided with the transpose option. The frozen parity JSON stores additional rectangular cases, including non-contiguous views, so implementations in Python and optional C can be compared without linking their runtimes.

## 4. What the tests establish

| Test | Why it matters |
|---|---|
| `test_hand_example` | Pins the definition to the four exact values above. |
| `test_matches_frozen_c_parity_vectors` | Keeps the Python reference aligned with the optional C file-based fixture contract. |
| `test_shapes_and_strided_views` | Catches assumptions that arrays are packed or contiguous. |
| `test_transpose_scaling_and_empty_inner_dimension` | Exercises transpose interpretation, alpha/beta behavior, and the empty sum. |
| `test_beta_zero_does_not_read_nan_c` | Prevents stale NaNs in a caller's output buffer from contaminating fresh results. |
| `test_invalid_output_shape_is_rejected` | Makes shape mistakes fail at the boundary with a useful error. |

Run `ol start M03.1`, inspect the test catalog with `ol tests M03.1`, then use `ol check M03.1` for the verdict. `ol diff M03.1` shows your implementation against the reference.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Treating the last two dimensions as packed | Sliced arrays produce wrong values or modify neighboring storage. | `test_shapes_and_strided_views` |
| Reversing the transpose convention | Square cases may pass while rectangular weights are indexed incorrectly. | `test_transpose_scaling_and_empty_inner_dimension` |
| Evaluating `beta * C` when `beta` is zero | NaNs in the old output survive an intended overwrite. | `test_beta_zero_does_not_read_nan_c` |
| Comparing float32 results bit for bit | A valid reduction order fails due only to rounding. | `test_matches_frozen_c_parity_vectors` |
| Accepting an output with the wrong shape | Results are truncated or broadcast into an unintended shape. | `test_invalid_output_shape_is_rejected` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.01` | NumPy arrays, shape inspection, indexing, and views. |
| Forward | `L0.0` | Converts each byte ID into its bigram-logit row through the weight matrix. |
| Forward | `L0.5` | Reuses matrix multiplication in the training path. |
| Forward | `L9.1` | Implements an optional standalone C kernel and checks it against shared file fixtures. |
| Forward | `M03.2` | Uses matrix storage and products when solving linear systems. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| triple-loop matmul | BLAS GEMM | blocking and vector instructions keep matrix tiles in cache | Netlib BLAS `dgemm` |
| shared parity vectors | standalone `L9.1` C kernel | checks a second implementation through files | `course/fixtures/M03.1` |
