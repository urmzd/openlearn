# Cache-blocked, packed, batch-invariant matmul in C

The naive triple-loop matmul from [Vectors, matrices, row-major layout, and matmul](../../mathematics-for-computing/04-linear-algebra/01-vectors-matrices-and-matmul.md) in *Mathematics for Computing* is easy to understand and slow on large matrices, because it keeps fetching values that sit far apart in memory. This section rewrites the same function, `tl_matmul_f32`, with the same signature and contract, as a high-performance kernel: Goto's loop nest, packed operands, a $4 \times 16$ register micro-kernel, and threads from the [thread pool](02-thread-pool-and-parallel-for.md). At $512^3$ it is 10 to 20 times faster than the triple loop while doing exactly the same $2MNK$ floating-point operations.

Speed is only half of the story. The kernel also makes a promise most fast libraries do not: **batch invariance**. Row $i$ of a product has the same bits whether it was computed alone ($M = 1$, a decode step) or inside a batch of 37 (a prefill), at any position, on any number of threads. That promise is what lets the C backend generate the same tokens as numpy, only faster, and it comes entirely from fixing the order of the reduction. This optional kernel demonstrates both optimizations; the Python and Rust engine paths keep their own implementations.

You will use the error slot from [C runtime support and an arena allocator with marks](01-arena-allocator-with-marks.md) and `tl_parallel_for` from the [thread pool](02-thread-pool-and-parallel-for.md). Helpful background from *Mathematics for Computing*: the naive version of this very file in the matmul section above, the $\sqrt{K}$ error bound in [Error analysis, condition numbers, and tolerance budgets](../../mathematics-for-computing/10-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md), and FLOP counting in [Counting: parameters, FLOPs, KV bytes, and memory plans](../../mathematics-for-computing/06-discrete-math-1/01-counting-params-flops-kv-bytes.md). The contract is [`matmul.h`](../../../../course/contracts/c/include/tinyllm/matmul.h) under the rules of [`c/ABI.md`](../../../../course/contracts/c/ABI.md) (rule 10 is batch invariance), and the reference solution is [`matmul.c`](../../../../course/solutions/c/src/kernels/matmul.c), with the helpers `column_strip`, `pack_b`, `pack_a`, `micro_kernel`, and `store_tile`. As you read, try to predict each step before you see it.

> **Learning objectives**
> - Use the roofline model to explain why the triple loop is memory-bound and how blocking raises arithmetic intensity.
> - Describe the five loops of Goto's nest and what each keeps in which level of the memory hierarchy.
> - Pack operands into micro-kernel order with zero padding, absorbing `trans_b` and leading dimensions.
> - Explain why a constant k-block size, zero padding, and `FP_CONTRACT OFF` make each row's bits independent of the batch.
> - Apply $\beta$ exactly once across k-blocks and parallelize over column strips without changing a bit.

## Why the triple loop is slow: the roofline

This section uses the following quantities throughout:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $A, B, C$ | $M \times K$, $K \times N$ (or $N \times K$ with `trans_b`), $M \times N$, row-major with leading dimensions | `float*` |
| $M_R \times N_R$ | the micro-tile held in registers: $4 \times 16$ | constants `MR`, `NR` |
| $K_C$ | k values per block, the fixed reduction step: 128 | `KC` |
| $M_C$ | rows of $A$ packed at once: 64 | `MC` |
| $N_C$ | columns per strip, one thread's unit of work: 128 | `NC` |
| $P_b(i, j)$ | partial sum of block $b$: $\sum_{k = bK_C}^{\min((b+1)K_C, K) - 1} A_{ik}\,\mathrm{op}(B)_{kj}$, from 0, in increasing $k$ | float |
| $I$ | arithmetic intensity: FLOPs per byte moved from memory | FLOP/byte |

A core can do $F$ floating-point operations per second only if its operands arrive fast enough. Memory delivers $W$ bytes per second, so a loop that does $I$ FLOPs per byte it loads runs at most at $\min(F, I \cdot W)$. That ceiling is the **roofline**.

In the naive loop, each inner step loads one float of $A$ and one of $B$ (8 bytes) for 2 FLOPs, so $I = 0.25$, and walking down a column of $B$ touches a new 64-byte cache line on every step. Yet a product of $N \times N$ matrices does $2N^3$ FLOPs on only $3N^2$ distinct numbers, so the data *could* be reused $N$ times; the naive loop reloads it instead. Blocking is the art of keeping a piece of data in a fast level (registers, L1, L2) while it is used many times.

## Goto's loop nest

Goto's design uses five loops, each sized for one level of the memory hierarchy. The outermost loop walks **column strips** of $N_C$ columns of $C$ (and of $B$); one strip is one unit of work for a thread. Inside a strip, the second loop walks **k-blocks** of $K_C$ and packs the $K_C \times N_C$ block of $\mathrm{op}(B)$ into a buffer `bp`, which stays in L2 while every row block of $A$ passes it. The third loop walks **row blocks** of $M_C$ and packs the $M_C \times K_C$ block of $A$ into `ap`, which stays in L1/L2 across the strip. The fourth loop walks **micro-tiles**: each $N_R$-wide panel of `bp` against each $M_R$-tall panel of `ap`. And the fifth, the **micro-kernel**, computes an $M_R \times N_R$ block of partial sums in 64 accumulators, reading one column of $M_R$ values of $A$ and one row of $N_R$ values of $B$ per $k$: $M_R + N_R = 20$ loads for $2 M_R N_R = 128$ FLOPs.

The micro-kernel's intensity is what makes the whole thing fast: 6.4 FLOPs per float loaded instead of 1, with every load from the packed buffers sequential. The compiler turns `acc[r][c] += a[r] * b[c]` with constant $M_R, N_R$ into vector registers (16 floats of a row is four 128-bit NEON or two 256-bit AVX registers).

## Packing

The micro-kernel's sequential loads only exist because of **packing**: copying a block into the order the micro-kernel will read it. `pack_b` writes panel $p$ (columns $p N_R$ to $p N_R + N_R - 1$) as $K_C$ consecutive rows of $N_R$ floats; `pack_a` writes panel $p$ (rows $p M_R$ to $p M_R + M_R - 1$) as $K_C$ consecutive columns of $M_R$ floats.

Two more jobs happen during the copy. First, `trans_b` and the leading dimensions disappear. The packer reads $\mathrm{op}(B)_{kj}$ at `B[k*ldb + j]` or `B[j*ldb + k]` and $A_{ik}$ at `A[i*lda + k]`; the micro-kernel only ever sees contiguous panels. Second, edges are padded with zeros. When $M$ is not a multiple of 4 or $N$ not a multiple of 16, the last panel is filled with zeros, so the micro-kernel always computes a full $4 \times 16$ tile. A zero row contributes nothing, and the store writes only the real $m_r \times n_r$ corner back into $C$ through `ldc`. One code path for every tile is what makes batch invariance possible, as the next part shows.

The buffers live on the stack: $K_C N_C + M_C K_C = 128 \cdot 128 + 64 \cdot 128$ floats, 96 KiB. Rule 1 of the ABI forbids a kernel to allocate, and this signature has no arena to borrow from.

## The reduction order and batch invariance

Floating-point addition is not associative: $(a + b) + c$ and $a + (b + c)$ can differ in the last bit, as the matmul section in *Mathematics for Computing* shows. So "the same sum" really means "the same sum *in the same order*". In this kernel, element $(i, j)$ is

$$C_{ij} = \Big(\big(\alpha P_0 + \beta C^{\text{old}}_{ij}\big) + \alpha P_1\Big) + \alpha P_2 + \cdots, \qquad P_b = \Big(\big(0 + A_{i,bK_C}\,B_{bK_C, j}\big) + A_{i, bK_C + 1}\,B_{bK_C+1, j}\Big) + \cdots$$

Every quantity in that expression depends only on row $i$ of $A$, column $j$ of $\mathrm{op}(B)$, $K$, $\alpha$, $\beta$, and the constant $K_C$. It does not depend on $M$, on which micro-tile row $i$ landed in, on the zero padding, or on which thread computed the strip. That is **batch invariance** (ABI rule 10): row $i$ of a product computed with $M = 1$ equals row $i$ computed with $M = 37$, bit for bit.

Real libraries lose this property in three ways, all for speed. One is a block size that depends on the shape, such as a "GEMV path" for $M = 1$ that sums all of $K$ in one block; the decode step ($M = 1$) and the prefill ($M = T$) then round differently, and the worked example below shows exactly how. Another is splitting $K$ across threads ("split-K") and adding the thread partials in completion order. The third is using different instructions for edge tiles, for example a fused multiply-add in the vectorized body and a separate multiply and add in a scalar remainder. A fused multiply-add rounds once where the pair rounds twice. Zero padding gives every element the same code path, and `#pragma STDC FP_CONTRACT OFF` forbids the compiler to fuse, so the arithmetic is the same on every path and every machine.

## Threads

Parallelism follows the same rule: never split a sum. `tl_parallel_for(tp, n_strips, 1, strips, &args)` runs each column strip exactly once, on some worker. Strips write disjoint columns of $C$ and only read $A$ and $B$, so there is no race, and a strip's arithmetic does not depend on the worker, so the result has the same bits with 1 or 8 threads. With `tp == NULL` the strips run serially on the caller.

## What it costs

The packed kernel still does $2MNK$ FLOPs. Packing $B$ costs $KN$ copies per call and packing $A$ costs $MK$ copies per strip, small next to $MNK$ when the matrices are large. For $M = 1$ (decode) the micro-kernel computes a $4 \times 16$ tile of which one row is real, so three quarters of the arithmetic is wasted on padding. That is the price of invariance here; the quantized GEMV in [Fused int4 and int8 dequantize-matmul in C](07-fused-quantized-matmul.md) is where decode gets its own kernel.

## Working it by hand

Start with the small product from the matmul section in *Mathematics for Computing*, pushed through the tiles: $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$, $B = \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}$, so $M = 2, N = 2, K = 3$. There is one strip (columns 0 and 1), one k-block ($K = 3 < 128$), and one row block.

`pack_b` writes one 16-wide panel of 3 rows: `{7, 8, 0, ..., 0 | 9, 10, 0, ..., 0 | 11, 12, 0, ..., 0}`, with 14 zeros of padding per row. `pack_a` writes one 4-tall panel of 3 columns: `{1, 4, 0, 0 | 2, 5, 0, 0 | 3, 6, 0, 0}`. The micro-kernel's accumulator after each $k$ is:

| $k$ | adds $a_r b_c$ for $r \in \{0, 1\}$, $c \in \{0, 1\}$ | acc row 0 | acc row 1 |
|---|---|---|---|
| 0 | $1 \cdot (7, 8)$, $4 \cdot (7, 8)$ | (7, 8) | (28, 32) |
| 1 | $2 \cdot (9, 10)$, $5 \cdot (9, 10)$ | (25, 28) | (73, 82) |
| 2 | $3 \cdot (11, 12)$, $6 \cdot (11, 12)$ | (58, 64) | (139, 154) |

Rows 2 and 3 of the accumulator and columns 2 to 15 are zero (padding). The store writes only the $2 \times 2$ corner: $C = \{58, 64, 139, 154\}$, the same result as the triple loop.

Next, see why the block size must not depend on $M$. Take one output whose four products are, in order, $1, 10^8, 3, 3$ (float32). With the sum in one block, $((0 + 1) + 10^8) + 3 + 3$: float32 spacing near $10^8$ is 8, so adding 1 or 3 rounds back to $10^8$ each time, and the result is $100{,}000{,}000$. With blocks of 2, $(1 + 10^8) + (3 + 3) = 10^8 + 6$, which rounds up to $100{,}000{,}008$. Same numbers, different grouping, different bits. If the kernel used one block for $M = 1$ and blocks of 2 otherwise, a decode step and a prefill would disagree in exactly this way.

Finally, $\beta$ across blocks. With $K = 300$ and $K_C = 128$, element $(i, j)$ goes through three stores: $c \leftarrow \alpha P_0 + \beta c$, then $c \leftarrow c + \alpha P_1$, then $c \leftarrow c + \alpha P_2$. $\beta$ appears once.

## The interface

The signature and contract are unchanged from the naive version; what is new is the promise of rule 10:

```c
/* tinyllm/matmul.h */
tl_status tl_matmul_f32(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K,
                        int64_t lda, int64_t ldb, int64_t ldc, float alpha, float beta,
                        int trans_b, tl_pool *tp);
/* From L9.1 the k order is fixed and independent of M and of the row's position:
   row i computed with M = 1 equals row i computed with any M, bit for bit.
   tp may be NULL (serial); with a pool, column strips run in parallel. */
```

The solution takes over the naive version's file, `c/src/kernels/matmul.c`: same signature, new body. Every contract case of the naive version still holds.

## Building it step by step

The arithmetic is fixed at $2MNK$; the cost is memory traffic. So the design question is how many times each loaded value is reused before it leaves a fast level of the hierarchy. Registers give the most reuse per load: an $M_R \times N_R$ block of accumulators reuses each loaded $a$ value $N_R$ times and each $b$ value $M_R$ times, and $4 \times 16 = 64$ accumulators fit the vector registers of NEON and AVX2 and give 6.4 FLOPs per float loaded. The micro-kernel needs its operands contiguous, so you pack: copy a $K_C \times N_C$ block of $\mathrm{op}(B)$ and an $M_C \times K_C$ block of $A$ into micro-kernel order. Packing is also the one place that has to know about `trans_b` and leading dimensions.

Batch invariance needs one code path and a fixed grouping of the sum. Zero padding to whole tiles gives every element the same path, a constant $K_C$ fixes the grouping, and `FP_CONTRACT OFF` stops the compiler fusing on some paths and not others. $\beta$ belongs to the first k-block only, because $C$ is overwritten by the first block's store and accumulated into afterwards. Threads must split independent outputs, never a sum, so the unit of work is a column strip of $C$, scheduled by the pool. And there is no allocation in a kernel: packing buffers sized by the block constants live on the stack (96 KiB, below the 512 KiB of a macOS worker thread).

### Block constants and the arithmetic rule

```c
/* A fused multiply-add rounds once where a multiply then an add rounds
 * twice. If the compiler fused on one code path and not on another, an
 * element's bits would depend on where it sits in a tile. */
#pragma STDC FP_CONTRACT OFF

#define MR 4   /* rows of a micro-tile */
#define NR 16  /* columns of a micro-tile */
#define KC 128 /* k values per block: the fixed reduction step */
#define MC 64  /* rows per packed block of A (a multiple of MR) */
#define NC 128 /* columns per strip (a multiple of NR) */
```

### Packing

Both packers pad with zeros past the real edge, and both resolve the strides so the micro-kernel never sees them:

```c
/* panel p holds bp[p * kc * NR + k * NR + c] = op(B)[k0 + k][j0 + p * NR + c] */
static void pack_b(const mm_args *a, int64_t k0, int64_t kc, int64_t j0, int64_t nc, float *bp) {
    for (int64_t jp = 0; jp < nc; jp += NR) {
        for (int64_t k = 0; k < kc; k++) {
            for (int64_t c = 0; c < NR; c++) {
                int64_t j = j0 + jp + c;
                float v = 0.0f;                         /* zero past column nc */
                if (jp + c < nc)
                    v = a->trans_b ? a->B[j * a->ldb + (k0 + k)]     /* B is N x K */
                                   : a->B[(k0 + k) * a->ldb + j];    /* B is K x N */
                *bp++ = v;
            }
        }
    }
}

/* panel p holds ap[p * kc * MR + k * MR + r] = A[i0 + p * MR + r][k0 + k] */
static void pack_a(const mm_args *a, int64_t i0, int64_t mc, int64_t k0, int64_t kc, float *ap) {
    for (int64_t ip = 0; ip < mc; ip += MR)
        for (int64_t k = 0; k < kc; k++)
            for (int64_t r = 0; r < MR; r++)
                *ap++ = ip + r < mc ? a->A[(i0 + ip + r) * a->lda + (k0 + k)] : 0.0f;  /* lda, not K */
}
```

### The micro-kernel

Constant bounds let the compiler keep `acc` in vector registers; each partial sum starts at 0 and adds in increasing $k$:

```c
static void micro_kernel(int64_t kc, const float *ap, const float *bp, float acc[MR][NR]) {
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++) acc[r][c] = 0.0f;        /* P_b starts at 0 */
    for (int64_t k = 0; k < kc; k++) {
        const float *av = ap + k * MR, *bv = bp + k * NR;      /* sequential reads */
        for (int r = 0; r < MR; r++)
            for (int c = 0; c < NR; c++) acc[r][c] += av[r] * bv[c];   /* 20 loads, 128 FLOPs */
    }
}
```

### The store

Only the real corner is written, through `ldc`, and $\beta$ is applied only in the first block:

```c
static void store_tile(const mm_args *a, int64_t i, int64_t j, int64_t mr, int64_t nr, float acc[MR][NR],
                       int first) {
    for (int64_t r = 0; r < mr; r++) {                    /* mr <= MR: no write into padding */
        float *c = a->C + (i + r) * a->ldc + j;
        for (int64_t q = 0; q < nr; q++) {
            float v = a->alpha * acc[r][q];
            if (!first)
                c[q] = c[q] + v;                              /* later k-blocks add */
            else
                c[q] = a->beta == 0.0f ? v : v + a->beta * c[q];  /* beta once; never read C if 0 */
        }
    }
}
```

### The loop nest for one strip

```c
static void column_strip(const mm_args *a, int64_t j0, int64_t nc) {
    float bp[KC * NC], ap[MC * KC];                        /* 96 KiB, on the stack */
    for (int64_t k0 = 0; k0 < a->K; k0 += KC) {            /* k-blocks, increasing */
        int64_t kc = a->K - k0 < KC ? a->K - k0 : KC;
        pack_b(a, k0, kc, j0, nc, bp);                     /* stays in L2 for every row block */
        for (int64_t i0 = 0; i0 < a->M; i0 += MC) {
            int64_t mc = a->M - i0 < MC ? a->M - i0 : MC;
            pack_a(a, i0, mc, k0, kc, ap);
            for (int64_t jr = 0; jr < nc; jr += NR)
                for (int64_t ir = 0; ir < mc; ir += MR) {
                    float acc[MR][NR];
                    micro_kernel(kc, ap + ir * kc, bp + jr * kc, acc);
                    store_tile(a, i0 + ir, j0 + jr, mc - ir < MR ? mc - ir : MR,
                               nc - jr < NR ? nc - jr : NR, acc, k0 == 0);
                }
        }
    }
}
```

Notice what the block boundaries depend on. `k0` steps by the constant `KC`, never by anything derived from `M`; `i0` and `ir` decide only *where* a row is packed, and the padded rows contribute nothing to any real row.

### The entry point

The entry point keeps the naive version's contract and adds the strips:

```c
tl_status tl_matmul_f32(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K,
                        int64_t lda, int64_t ldb, int64_t ldc, float alpha, float beta,
                        int trans_b, tl_pool *tp) {
    /* 1. Validate everything before touching C (the M03.1 contract):
     *    negative dims, ldc < N, C NULL; and when K > 0, lda < K,
     *    ldb < (trans_b ? K : N), A or B NULL: TL_EINVAL. M or N == 0: TL_OK. */
    // ...
    /* 2. K == 0: no k-block runs, and the empty sum is 0, so C = beta * C. */
    if (K == 0) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++) C[i * ldc + j] = beta == 0.0f ? 0.0f : beta * C[i * ldc + j];
        return TL_OK;
    }
    /* 3. One range per column strip, on whichever worker. */
    mm_args a = {A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, trans_b};
    int64_t n_strips = (N + NC - 1) / NC;
    if (tp == NULL) {
        strips(&a, 0, n_strips, 0);
        return TL_OK;
    }
    return tl_parallel_for(tp, n_strips, 1, strips, &a);
}
```

`strips` is the pool's range function: for each strip index $s$ in its range it calls `column_strip` on columns $[s N_C, \min((s+1) N_C, N))$. On the worked example, `column_strip` runs once with `kc = 3`, `mc = 2`, `nc = 2`, one micro-tile, and `store_tile` writes `mr = 2`, `nr = 2`: $\{58, 64, 139, 154\}$.

The full solution is [`matmul.c`](../../../../course/solutions/c/src/kernels/matmul.c), against the contract [`matmul.h`](../../../../course/contracts/c/include/tinyllm/matmul.h).

### Why this design

The cost is $2MNK$ FLOPs as before, plus $KN$ copies per call to pack $B$ and $MK$ copies per strip to pack $A$, small next to $MNK$ for large matrices. Measured against the triple loop at $512^3$, the solution is about 19 times faster (about 50 GFLOP/s on one Apple M-series core); Apple's Accelerate (`cblas_sgemm`, on the AMX units) is several times faster again.

Each rejected alternative trades away something this kernel needs. A GEMV fast path for $M = 1$ would avoid the three wasted rows of a $4 \times 16$ tile during decode, but it sums $K$ in a different grouping, so decode and prefill disagree in the last bit and greedy decoding can flip tokens on near-ties. Here invariance wins, and the quantized kernel gives decode its own path. Split-K threading parallelizes small-$N$ products better but adds partial sums in completion order, which changes bits with timing; column strips never split a sum. Hand-written intrinsics or assembly would be faster per core but tie the kernel to one ISA, while constant-bound C loops vectorize well on both NEON and AVX2 and keep the arithmetic auditable. The tradeoff accepted is that fixed block sizes are not tuned per machine, and `FP_CONTRACT OFF` gives up the speed of fused multiply-adds, in exchange for identical bits on every path and every machine.

## What can go wrong

A correct solution has to handle every tile edge and every carried-over rule of the naive contract:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example | unit, smoke | the worked example through one padded micro-tile | the definition, and the edge store |
| Shapes across tile edges | differential | $M, N, K$ around 4, 16, 64, 128, up to 257, both `trans_b`, against float64 within $\sqrt{K}$ | every tile edge |
| $\beta$ applies once across k-blocks | unit | $K = 300$, $\alpha = 2$, $\beta = 0.5$; then $\beta = 0$ over NaN | accumulation across k-blocks |
| Views, and padding untouched | unit | `lda`, `ldb`, `ldc` larger than the rows; $C$'s other columns unchanged | views into fused projections |
| Empty and invalid inputs, as in the naive version | boundary | empty dims, $K = 0$ gives $\beta C$, `TL_EINVAL` leaves $C$ untouched | the contract carries over |
| Batch-invariant rows | property | 16 rows of a Linear layer ($K = 300$): each alone vs inside every batch at every offset, bitwise | batched decode in [continuous batching](../11-serving/03-continuous-batching.md) |
| Batch invariance at $M = 1, 7, 37$ | property | the catalog's $M = 1, 7, 37$, bitwise | prefill vs decode |
| Pool result equals serial, bitwise | property | 4 threads vs serial, three strips (the last partial), under TSan too | threads never change bits |
| Row-major and transposed $B$ | unit | original row-major and transposed-B contract cases | the original interface remains compatible |
| $\alpha$ and $\beta$ | unit | alpha/beta behavior, including no read of C when beta is zero | preserves the original arithmetic contract |
| Empty shapes, bad arguments, leading dimensions | boundary | empty shapes, invalid arguments, and padded leading dimensions | preserves the original shape and view rules |

Most bugs in a blocked matmul come from the store and the packers. Applying $\beta$ in every k-block gives $\beta^3 C$ for $K = 300$, and with $\beta = 0$ only the last block survives; `store_tile` applies $\beta$ only when `first` (`k0 == 0`). Later k-blocks that overwrite instead of adding leave only the last $K \bmod K_C$ products; the solution writes `c[q] = c[q] + v` when not `first`. Storing the whole $4 \times 16$ micro-tile at an edge writes past $C$ or into its padding (ASan, or a neighbour's columns); the store writes only the real `mr x nr` corner. Storing with $N$ instead of `ldc` scrambles views of $C$; rows of $C$ are addressed with `ldc`.

The packers have their own traps. Packing $B$ without honoring `trans_b` makes Linear layers multiply by the wrong matrix; `pack_b` reads `B[j * ldb + k]` when `trans_b`. Packing $A$ with $K$ instead of `lda` is right on packed matrices and wrong on every view; `pack_a` indexes with `lda`. Packing buffers sized for the full matrix on the stack overflow worker threads (512 KiB on macOS); size them by the block constants, 96 KiB.

> **Warning:** Reading $C$ when $\beta = 0$ lets a NaN in a fresh buffer reach the logits. Use `beta == 0.0f ? v : ...`, and do the same in the $K = 0$ path.

Rewrites tend to lose details of the old contract. Returning early for $K = 0$ leaves $C$ with old values instead of $\beta C$; an explicit $K = 0$ path writes $\beta C$. Dropping the naive version's argument checks lets a short `lda` read past the buffer; validate every argument before touching $C$.

> **Warning:** A "GEMV fast path" that sums all of $K$ at once when $M = 1$ makes decode and prefill differ in the last bit, and greedy tokens flip on near-ties. Keep one code path for every $M$, with constant `KC`.

Finally, a partition that drops or repeats a strip is wrong only with a pool, which makes it easy to miss; use $\lceil N / N_C \rceil$ strips, one pool range each.

## Where this leads

This kernel relies on the error slot behind `TL_EINVAL` from [the runtime support layer](01-arena-allocator-with-marks.md) and on `tl_parallel_for` from [the thread pool](02-thread-pool-and-parallel-for.md) to run the column strips. Looking forward, [Fused int4 and int8 dequantize-matmul in C](07-fused-quantized-matmul.md) uses it as both its benchmark and its float32 baseline, running this kernel on the dequantized weight.

Production libraries push each idea further:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| 4 x 16 C micro-kernel | BLIS and OpenBLAS assembly micro-kernels | hand-scheduled FMA pipelines, prefetching, per-microarchitecture block sizes | BLIS `kernels/armv8a/`, `kernels/haswell/` |
| column-strip threads | BLIS multithreading | parallelism in several loops at once (the $j_c$ and $i_c$ loops), sized by the cache topology | Smith et al., "Anatomy of High-Performance Many-Threaded Matrix Multiplication" (2014) |
| fixed $K_C$ for invariance | batch-invariant kernels for LLM serving | invariant matmul, RMSNorm, and attention as a PyTorch library | Thinking Machines `batch_invariant_ops` |
| float32 CPU GEMM | Apple Accelerate / AMX, cuBLAS | dedicated matrix units, several times the vector throughput | `cblas_sgemm`; cuBLASLt |
| one kernel for every dtype | llama.cpp `ggml_mul_mat` | dispatch to quantized dot-product kernels per weight format | `ggml/src/ggml-cpu/ggml-cpu.c` |

### Further reading

- Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication" (2008).
- Van Zee and van de Geijn, "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" (2015).
- Williams, Waterman, and Patterson, "Roofline" (2009).
- He et al., "Defeating Nondeterminism in LLM Inference" (Thinking Machines, 2025).

## Key terms

- **roofline**: the bound $\min(F, I \cdot W)$ on a loop's speed from peak compute $F$, memory bandwidth $W$, and arithmetic intensity $I$.
- **arithmetic intensity**: FLOPs performed per byte moved from memory.
- **micro-kernel**: the innermost loop that computes an $M_R \times N_R$ tile of partial sums in registers.
- **packing**: copying a block of an operand into the exact, contiguous order the micro-kernel reads, padded with zeros to whole tiles.
- **k-block**: a run of $K_C$ consecutive k values whose products are summed from 0 before being added to $C$.
- **batch invariance**: the property that a row's output bits do not depend on the batch size, its position in the batch, or the threads.
- **split-K**: parallelizing a matmul by splitting the reduction dimension across threads, which breaks bitwise reproducibility.

## Summary

- The triple loop is slow because of memory, not arithmetic. Blocking reuses each loaded value many times from cache and registers; the packed version is 10 to 20 times faster at $512^3$ with the same $2MNK$ operations.
- Packing copies a block into the exact order the micro-kernel reads it, padded with zeros to whole micro-tiles, so the inner loop is unit-stride and every output element goes through the same code path.
- Batch invariance is a property of the reduction order. Each $C_{ij}$ is a sum over k-blocks of fixed size $K_C$, each a sum from 0 in increasing $k$: nothing depends on $M$, on the row's position, or on the threads, so a row has the same bits alone or in a batch of 37.
- $\beta$ applies once, in the first k-block, and later blocks add; $\beta = 0$ still never reads $C$.
- Threads split columns, never a sum, so 1 and 4 threads give identical bits.

## Review questions

1. Why does zero padding make the result independent of where a row sits in a micro-tile?
   <details><summary>Answer</summary>Every row, real or padding, goes through the same full $4 \times 16$ micro-kernel and the same sequence of multiplies and adds. A real row's products and their order depend only on its own row of $A$ and the column of $B$; padding rows are zero and are never stored. So the code path, and the bits, are the same at offset 0 or offset 3 of a tile.</details>
2. Why must $K_C$ be a constant rather than chosen from the shape?
   <details><summary>Answer</summary>The k-block boundaries decide how the sum is grouped, and float addition is not associative. A shape-dependent block size (for example all of $K$ when $M = 1$) groups the same products differently and changes the last bits, as the $1, 10^8, 3, 3$ example shows.</details>
3. With $K = 300$ and $\beta = 0.5$, how many times is $\beta$ applied, and where?
   <details><summary>Answer</summary>Once, in `store_tile` with `first` true, which happens for `k0 == 0`. The second and third k-blocks add $\alpha P_1$ and $\alpha P_2$ to the stored value. Applying $\beta$ in every block would give $\beta^3 C$.</details>
4. Why is it safe to run strips on different threads without locks?
   <details><summary>Answer</summary>Each strip writes a disjoint set of columns of $C$ and only reads $A$ and $B$; packing buffers are local to the strip's stack frame. There is no shared mutable state, so no race, and the strip's arithmetic does not depend on the worker.</details>
5. Why are the packing buffers on the stack and not from an arena or `malloc`?
   <details><summary>Answer</summary>ABI rule 1 forbids a kernel to allocate, and this signature carries no arena. Their size depends only on the block constants (96 KiB), not on the matrix, so the stack is safe even on worker threads with 512 KiB stacks.</details>
6. What is the arithmetic intensity of the $4 \times 16$ micro-kernel, measured in FLOPs per float loaded, and how does it compare with the naive loop?
   <details><summary>Answer</summary>Each $k$ step loads $M_R + N_R = 20$ floats and does $2 M_R N_R = 128$ FLOPs, which is 6.4 FLOPs per float. The naive loop loads two floats for 2 FLOPs, 1 FLOP per float (0.25 FLOPs per byte).</details>
