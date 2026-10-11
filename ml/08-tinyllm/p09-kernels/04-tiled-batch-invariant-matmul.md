<!-- ol:module L9.1 -->
# Cache-blocked, packed, batch-invariant matmul in C (optional)

## Overview

| | |
|---|---|
| **Module** | `L9.1` · side · C · Pass 6 · 5 to 7 h |
| **You build** | `c/src/kernels/matmul.c`, taken over from `M03.1`: the same `tl_matmul_f32` signature and contract, now with Goto's loop nest (column strips, k-blocks, packed panels, a 4 x 16 register micro-kernel), threads from your `rt.03` pool, and a fixed reduction order that makes every row's bits independent of the batch |
| **Contract** | [`course/contracts/c/include/tinyllm/matmul.h`](../../../course/contracts/c/include/tinyllm/matmul.h) · rules: [`c/ABI.md`](../../../course/contracts/c/ABI.md) (rule 10, batch invariance) |
| **Tests** | `course/tests/L9.1/`: `test_matmul_tiled.c` (C, under ASan and UBSan, and ThreadSanitizer for the pool case) and shared file fixtures (what they check: section 4); `M03.1`'s tests run as your regression. Bench: `course/tests/L9.1/bench/matmul_512.c` |
| **Needs** | `rt.02` the error slot and the loader · `rt.03` `tl_parallel_for` (or `--ref-deps`). Reading: `M03.1` (your v0 of this file) · `M09.3` (the $\sqrt{K}$ error bound) · `M05.1` (FLOP counting) · `rt.02` (why this kernel does not use an arena) |
| **Used by** | `L9.5` measures its quantized kernel against this optional C baseline; Python and Rust have independent implementations |
| **Milestone** | `MS-L9` (the C backend generates the same tokens as numpy, and faster) |
| **Optional depth** | Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication" (2008); Van Zee and van de Geijn, "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" (2015); Williams, Waterman, and Patterson, "Roofline" (2009); He et al., "Defeating Nondeterminism in LLM Inference" (Thinking Machines, 2025) |

## Key Takeaways

- **The triple loop is slow because of memory, not arithmetic.** Blocking reuses each loaded value many times from cache and registers; your packed version is 10 to 20 times faster at $512^3$ with the same $2MNK$ operations (`ol bench L9.1`).
- **Packing copies a block into the exact order the micro-kernel reads it,** padded with zeros to whole micro-tiles, so the inner loop is unit-stride and every output element goes through the same code path (`shapes_across_tile_edges`).
- **Batch invariance is a property of the reduction order.** Each $C_{ij}$ is a sum over k-blocks of fixed size $K_C$, each a sum from 0 in increasing $k$: nothing depends on $M$, on the row's position, or on the threads, so a row has the same bits alone or in a batch of 37 (`batch_invariant_rows`, `batch_invariant_at_1_7_37`).
- **$\beta$ applies once, in the first k-block,** and later blocks add; $\beta = 0$ still never reads $C$ (`beta_applies_once_across_k_blocks`).
- **Threads split columns, never a sum,** so 1 and 4 threads give identical bits (`pool_result_equals_serial_bitwise`).

## How to work this chapter

```bash
ol start L9.1              # prints the contract diff: you edit your own c/src/kernels/matmul.c
ol tests L9.1              # read the test catalog first
ol check L9.1              # exit code is the verdict (M03.1's tests run as the regression)
ol check L9.1 --ref-deps   # only if your rt.03 is not passing yet
ol bench L9.1 --assert     # the speed budget: at least 10x M03.1's loop at 512^3 (local only)
ol parity matmul           # both of your versions against the float64 golden
ol diff  L9.1              # after passing: your code against the reference
```

---

## 1. Why now

The naive triple loop is easy to understand but slow on large matrices because it repeatedly fetches values far apart in memory. Blocked multiplication reuses a small tile in cache, and a fixed reduction order makes each row's result independent of the batch size. This optional kernel demonstrates those optimizations; the Python and Rust engine paths keep their own implementations.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $A, B, C$ | $M \times K$, $K \times N$ (or $N \times K$ with `trans_b`), $M \times N$, row-major with leading dimensions | `float*` |
| $M_R \times N_R$ | the micro-tile held in registers: $4 \times 16$ | constants `MR`, `NR` |
| $K_C$ | k values per block, the fixed reduction step: 128 | `KC` |
| $M_C$ | rows of $A$ packed at once: 64 | `MC` |
| $N_C$ | columns per strip, one thread's unit of work: 128 | `NC` |
| $P_b(i, j)$ | partial sum of block $b$: $\sum_{k = bK_C}^{\min((b+1)K_C, K) - 1} A_{ik}\,\mathrm{op}(B)_{kj}$, from 0, in increasing $k$ | float |
| $I$ | arithmetic intensity: FLOPs per byte moved from memory | FLOP/byte |

### 2.1 Why the triple loop is slow: the roofline

A core can do $F$ floating-point operations per second only if operands arrive fast enough. Memory delivers $W$ bytes per second, so a loop that does $I$ FLOPs per byte it loads runs at most at $\min(F, I \cdot W)$: the **roofline**. In `M03.1`'s loop, each inner step loads one float of $A$ and one of $B$ (8 bytes) for 2 FLOPs, $I = 0.25$, and walking down a column of $B$ touches a new 64-byte cache line on every step. A product of $N \times N$ matrices does $2N^3$ FLOPs on only $3N^2$ distinct numbers, so the data *could* be reused $N$ times; the naive loop reloads it instead. Blocking is the art of keeping a piece of data in a fast level (registers, L1, L2) while it is used many times.

### 2.2 Goto's loop nest

Five loops, each sized for one level of the memory hierarchy:

1. **Column strips** of $N_C$ columns of $C$ (and $B$). One strip is one unit of work for a thread.
2. **k-blocks** of $K_C$: pack the $K_C \times N_C$ block of $\mathrm{op}(B)$ into a buffer `bp` (it stays in L2 while every row block of $A$ passes it).
3. **Row blocks** of $M_C$: pack the $M_C \times K_C$ block of $A$ into `ap` (it stays in L1/L2 across the strip).
4. **Micro-tiles**: for each $N_R$-wide panel of `bp` and $M_R$-tall panel of `ap`,
5. **the micro-kernel** computes an $M_R \times N_R$ block of partial sums in 64 accumulators, reading one column of $M_R$ values of $A$ and one row of $N_R$ values of $B$ per $k$: $M_R + N_R = 20$ loads for $2 M_R N_R = 128$ FLOPs.

The micro-kernel's intensity is what makes it fast: 6.4 FLOPs per float loaded instead of 1, and every load from the packed buffers is sequential. The compiler turns `acc[r][c] += a[r] * b[c]` with constant $M_R, N_R$ into vector registers (16 floats of a row is four 128-bit NEON or two 256-bit AVX registers).

### 2.3 Packing

**Packing** copies a block into the order the micro-kernel will read it. `pack_b` writes panel $p$ (columns $p N_R$ to $p N_R + N_R - 1$) as $K_C$ consecutive rows of $N_R$ floats; `pack_a` writes panel $p$ (rows $p M_R$ to $p M_R + M_R - 1$) as $K_C$ consecutive columns of $M_R$ floats. Two more jobs happen during the copy:

- **`trans_b` and the leading dimensions disappear.** The packer reads $\mathrm{op}(B)_{kj}$ at `B[k*ldb + j]` or `B[j*ldb + k]` and $A_{ik}$ at `A[i*lda + k]`; the micro-kernel only ever sees contiguous panels.
- **Edges are padded with zeros.** When $M$ is not a multiple of 4 or $N$ not a multiple of 16, the last panel is filled with zeros, so the micro-kernel always computes a full $4 \times 16$ tile. A zero row contributes nothing, and the **store** writes only the real $m_r \times n_r$ corner back into $C$ through `ldc`. One code path for every tile is what makes the next section possible.

The buffers are on the stack: $K_C N_C + M_C K_C = 128 \cdot 128 + 64 \cdot 128$ floats, 96 KiB. `c/ABI.md` rule 1 forbids a kernel to allocate, and this signature has no arena to borrow from.

### 2.4 The reduction order, and batch invariance

Floating-point addition is not associative: $(a + b) + c$ and $a + (b + c)$ can differ in the last bit (`M03.1` 2.6). So "the same sum" is really "the same sum *in the same order*". In this kernel, element $(i, j)$ is

$$C_{ij} = \Big(\big(\alpha P_0 + \beta C^{\text{old}}_{ij}\big) + \alpha P_1\Big) + \alpha P_2 + \cdots, \qquad P_b = \Big(\big(0 + A_{i,bK_C}\,B_{bK_C, j}\big) + A_{i, bK_C + 1}\,B_{bK_C+1, j}\Big) + \cdots$$

Every quantity in that expression depends only on row $i$ of $A$, column $j$ of $\mathrm{op}(B)$, $K$, $\alpha$, $\beta$, and the constant $K_C$. It does not depend on $M$, on which micro-tile row $i$ landed in, on the zero padding, or on which thread computed the strip. That is **batch invariance** (`c/ABI.md` rule 10): row $i$ of a product computed with $M = 1$ equals row $i$ computed with $M = 37$, bit for bit.

Three ways to lose it, all of which real libraries do for speed:

- **A block size that depends on the shape**, such as a "GEMV path" for $M = 1$ that sums all of $K$ in one block. The decode step ($M = 1$) and the prefill ($M = T$) then round differently (section 5, mutant `s10`).
- **Splitting $K$ across threads** ("split-K") and adding the thread partials in completion order.
- **Different instructions for edge tiles**, for example a fused multiply-add in the vectorized body and separate multiply and add in a scalar remainder. A fused multiply-add rounds once where the pair rounds twice. Zero padding gives every element the same code path, and `#pragma STDC FP_CONTRACT OFF` forbids the compiler to fuse, so the arithmetic is the same on every path and every machine.

### 2.5 Threads

`tl_parallel_for(tp, n_strips, 1, strips, &args)` (`rt.03`) runs each column strip exactly once, on some worker. Strips write disjoint columns of $C$ and only read $A$ and $B$, so there is no race, and a strip's arithmetic does not depend on the worker, so the result is the same bits with 1 or 8 threads. Splitting columns rather than $K$ keeps every sum inside one thread. With `tp == NULL` the strips run serially on the caller.

### 2.6 What it costs

The packed kernel still does $2MNK$ FLOPs. Packing $B$ costs $KN$ copies per call and packing $A$ costs $MK$ copies per strip, small next to $MNK$ when the matrices are large. For $M = 1$ (decode) the micro-kernel computes a $4 \times 16$ tile of which one row is real, so a quarter of the arithmetic is wasted on padding; that is the price of invariance here, and `L9.5`'s quantized GEMV is where decode gets its own kernel.

## 3. Worked example by hand

**The product of `M03.1`, through the tiles.** $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$, $B = \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}$: $M = 2, N = 2, K = 3$. One strip (columns 0 and 1), one k-block ($K = 3 < 128$), one row block.

`pack_b` writes one 16-wide panel, 3 rows: `{7, 8, 0, ..., 0 | 9, 10, 0, ..., 0 | 11, 12, 0, ..., 0}` (14 zeros of padding per row). `pack_a` writes one 4-tall panel, 3 columns: `{1, 4, 0, 0 | 2, 5, 0, 0 | 3, 6, 0, 0}`. The micro-kernel's accumulator after each $k$:

| $k$ | adds $a_r b_c$ for $r \in \{0, 1\}$, $c \in \{0, 1\}$ | acc row 0 | acc row 1 |
|---|---|---|---|
| 0 | $1 \cdot (7, 8)$, $4 \cdot (7, 8)$ | (7, 8) | (28, 32) |
| 1 | $2 \cdot (9, 10)$, $5 \cdot (9, 10)$ | (25, 28) | (73, 82) |
| 2 | $3 \cdot (11, 12)$, $6 \cdot (11, 12)$ | (58, 64) | (139, 154) |

Rows 2 and 3 of the accumulator and columns 2 to 15 are zero (padding). The store writes only the $2 \times 2$ corner: $C = \{58, 64, 139, 154\}$, the first test, `hand_example`, and `test_hand_example`.

**Why the block size must not depend on $M$.** Take one output whose four products are, in order, $1, 10^8, 3, 3$ (float32). With the sum in one block, $((0 + 1) + 10^8) + 3 + 3$: float32 spacing near $10^8$ is 8, so adding 1 or 3 rounds back to $10^8$ each time, and the result is $100{,}000{,}000$. With blocks of 2, $(1 + 10^8) + (3 + 3) = 10^8 + 6$, which rounds up to $100{,}000{,}008$. Same numbers, different grouping, different bits. If the kernel used one block for $M = 1$ and blocks of 2 otherwise, a decode step and a prefill would disagree in exactly this way, which is what `batch_invariant_at_1_7_37` and mutant `s10` are about.

**$\beta$ across blocks.** With $K = 300$ and $K_C = 128$, element $(i, j)$ goes through three stores: $c \leftarrow \alpha P_0 + \beta c$, then $c \leftarrow c + \alpha P_1$, then $c \leftarrow c + \alpha P_2$. $\beta$ appears once (`beta_applies_once_across_k_blocks`).

## 4. The interface

The signature and the contract are `M03.1`'s, unchanged; what is new is the promise of rule 10:

```c
/* tinyllm/matmul.h */
tl_status tl_matmul_f32(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K,
                        int64_t lda, int64_t ldb, int64_t ldc, float alpha, float beta,
                        int trans_b, tl_pool *tp);
/* From L9.1 the k order is fixed and independent of M and of the row's position:
   row i computed with M = 1 equals row i computed with any M, bit for bit.
   tp may be NULL (serial); with a pool, column strips run in parallel. */
```

`ol start L9.1` does not overwrite your file: it prints the contract diff, and you rewrite the body of your own `c/src/kernels/matmul.c` in place.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3 through one padded micro-tile | the definition, and the edge store |
| `shapes_across_tile_edges` | differential | $M, N, K$ around 4, 16, 64, 128, up to 257, both `trans_b`, against float64 within $\sqrt{K}$ | every tile edge |
| `beta_applies_once_across_k_blocks` | unit | $K = 300$, $\alpha = 2$, $\beta = 0.5$; then $\beta = 0$ over NaN | accumulation across k-blocks |
| `views_and_padding_untouched` | unit | `lda`, `ldb`, `ldc` larger than the rows; $C$'s other columns unchanged | views into fused projections |
| `empty_and_invalid_like_v0` | boundary | `M03.1`'s rules: empty dims, $K = 0$ gives $\beta C$, `TL_EINVAL` leaves $C$ untouched | the contract carries over |
| `batch_invariant_rows` | property | 16 rows of a Linear layer ($K = 300$): each alone vs inside every batch at every offset, bitwise | batched decode (`L10.2`) |
| `batch_invariant_at_1_7_37` | property | the catalog's $M = 1, 7, 37$, bitwise | prefill vs decode |
| `pool_result_equals_serial_bitwise` | property | 4 threads vs serial, three strips (the last partial), under TSan too | threads never change bits |
| `baseline_row_major`, `trans_b_reads_rows_of_b` | unit | original row-major and transposed-B contract cases | the v0 interface remains compatible |
| `alpha_and_beta_scale_and_accumulate`, `beta_zero_ignores_garbage_in_c` | unit | alpha/beta behavior, including no read of C when beta is zero | preserves the original arithmetic contract |
| `empty_dimensions`, `bad_arguments_are_einval`, `leading_dimensions_skip_padding` | boundary | empty shapes, invalid arguments, and padded leading dimensions | preserves the original shape and view rules |

The bench (`ol bench L9.1`, a B test, never part of `ol check`) times your kernel against `M03.1`'s loop at $512^3$ and reports `speedup_vs_naive` (budget $\ge 10$) and GFLOP/s. The reference reaches about 19x and 50 GFLOP/s on one Apple M-series core; Apple's Accelerate (`cblas_sgemm`, AMX units) is several times faster again, which is the ceiling to compare against if you are curious, not a budget.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| applying $\beta$ in every k-block | $\beta^3 C$ for $K = 300$; with $\beta = 0$ only the last block survives | `beta_applies_once_across_k_blocks` (mutant `s01`) |
| storing the whole $4 \times 16$ micro-tile at an edge | writes past $C$ or into its padding (ASan, or a neighbour's columns) | `hand_example`, `views_and_padding_untouched` (mutant `s02`) |
| packing $B$ without `trans_b` | Linear layers multiply by the wrong matrix | `shapes_across_tile_edges` (mutant `s03`) |
| packing $A$ with $K$ instead of `lda` | right on packed matrices, wrong on every view | `views_and_padding_untouched` (mutant `s04`) |
| later k-blocks overwriting instead of adding | only the last $K \bmod K_C$ products survive | `beta_applies_once_across_k_blocks` (mutant `s05`) |
| storing with $N$ instead of `ldc` | views of $C$ are scrambled | `views_and_padding_untouched` (mutant `s06`) |
| reading $C$ when $\beta = 0$ | NaN in a fresh buffer reaches the logits | `empty_and_invalid_like_v0` (mutant `s07`) |
| returning early for $K = 0$ | $C$ keeps old values instead of $\beta C$ | `empty_and_invalid_like_v0` (mutant `s08`) |
| dropping `M03.1`'s argument checks while rewriting | a short `lda` reads past the buffer | `empty_and_invalid_like_v0` (mutant `s09`) |
| a "GEMV fast path" that sums all of $K$ at once when $M = 1$ | decode and prefill differ in the last bit; greedy tokens flip on near-ties | `batch_invariant_at_1_7_37`, `batch_invariant_rows` (mutant `s10`) |
| a partition that drops or repeats a strip | wrong only with a pool | `pool_result_equals_serial_bitwise` (mutant `s11`) |
| packing buffers sized for the full matrix on the stack | stack overflow on worker threads (512 KiB on macOS); size them by the block constants | the pool test under ASan and TSan |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the error slot behind `TL_EINVAL` |
| Back | `rt.03` | `tl_parallel_for` runs the column strips |
| Forward | `L9.5` | its benchmark and its float32 baseline are this kernel on the dequantized weight |

If you skip this module, `ol check L9.5` stops with `BLOCKED ... needs L9.1`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| 4 x 16 C micro-kernel | BLIS and OpenBLAS assembly micro-kernels | hand-scheduled FMA pipelines, prefetching, per-microarchitecture block sizes | BLIS `kernels/armv8a/`, `kernels/haswell/` |
| column-strip threads | BLIS multithreading | parallelism in several loops at once (the $j_c$ and $i_c$ loops), sized by the cache topology | Smith et al., "Anatomy of High-Performance Many-Threaded Matrix Multiplication" (2014) |
| fixed $K_C$ for invariance | batch-invariant kernels for LLM serving | invariant matmul, RMSNorm, and attention as a PyTorch library | Thinking Machines `batch_invariant_ops` |
| float32 CPU GEMM | Apple Accelerate / AMX, cuBLAS | dedicated matrix units, several times the vector throughput | `cblas_sgemm`; cuBLASLt |
| one kernel for every dtype | llama.cpp `ggml_mul_mat` | dispatch to quantized dot-product kernels per weight format | `ggml/src/ggml-cpu/ggml-cpu.c` |
