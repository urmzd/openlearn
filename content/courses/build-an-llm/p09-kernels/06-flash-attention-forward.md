# FlashAttention forward in C

## Overview

| | |
|---|---|
| **Module** | `L9.3` · side · C · Pass 6 · 6 to 8 h |
| **Solution** | [`c/src/kernels/flash_attn.c`](../../../../course/solutions/c/src/kernels/flash_attn.c): `tl_flash_attn_fwd_f32` (with the helpers `visible_range`, `query_tile`, `items`), attention for a whole prefill (or a chunk of one) that never stores the score matrix: GQA head sharing, causal masking at absolute positions (`q_offset`), a sliding window, learned sinks, and the log-sum-exp of every row, with scratch from the arena and threads from the pool |
| **Contract** | [`course/contracts/c/include/tinyllm/attention.h`](../../../../course/contracts/c/include/tinyllm/attention.h) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) (rule 10, invariance) |
| **Needs** | `rt.02` the loader · `rt.02` the arena · `rt.03` the pool · `M09.6` `tl_expf` · `L9.2` `tl_softmax_f32` (the naive reference) · `L7.7` `windowed_attention` · `L5.1` `sdpa_forward`. Reading: `L5.2` (mask flags), `L7.5` (GQA) |
| **Used by** | `L9.4` must match this kernel bit for bit in paged decode · `L10.3` relies on the chunk invariance proved here · together with `L9.4` to `L9.6`, this completes the C kernel set of Pass 6 |
| **Optional depth** | Dao et al., "FlashAttention" (2022) and "FlashAttention-2" (2023); Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018); Rabe and Staats, "Self-attention Does Not Need $O(n^2)$ Memory" (2021) |

## Key Takeaways

- **Attention can be computed one tile of keys at a time** with the online softmax of `L9.2`, extended to the output: each row keeps a running maximum $m$, denominator $\ell$, and output $a$, and rescales $\ell$ and $a$ by $e^{m_{\text{old}} - m_{\text{new}}}$ when the maximum grows.
- **Memory stops growing with the context:** scratch is $B_r$ rows of state and one tile of $B_c$ scores per worker, so the arena's high-water mark is the same for 16 keys and 1000.
- **Visibility is per row and per absolute position:** query $i$ sits at $p = q_{\text{offset}} + i$ and sees key $j$ when $j \le p$ (causal) and $j > p - w$ (window). Rows of one query tile may see different keys.
- **Key tiles are aligned to absolute positions** ($[tB_c, (t+1)B_c)$), so a row's arithmetic is the same whether the prompt arrives in one call or in chunks, in any batch, with any query tile size: bitwise.
- **A sink is one extra score with no value:** it joins the denominator and the lse, and the weights then sum to $1 - p_{\text{sink}}$.

## How to read this chapter

Sections 1 to 3 build the tiled online softmax from first principles and work one row by hand. Section 4 fixes the C interface and lists what a correct kernel must handle. Section 5 walks through the solution, [`flash_attn.c`](../../../../course/solutions/c/src/kernels/flash_attn.c), block by block and explains why each choice (absolute tile alignment, per-row visibility, a mark-and-rewind arena) is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The numpy attention (`L5.1`, `L7.5`, `L7.7`) forms the score matrix $S = QK^\top$ for every head: $T_q \times T_k$ floats, then a softmax over each row, then $PV$. For a 2048-token prompt that is 4 million floats (16 MB) per head per layer, written once and read twice, and most of the prefill time goes to moving it rather than computing it. This standalone C kernel makes memory use independent of prompt length. `L10.3` additionally splits long prompts into chunks so prefill does not stall decode; chunk invariance ensures every query row's output is stable across split choices.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $q_i, k_j, v_j \in \mathbb{R}^D$ | query row $i$, key and value at position $j$ | `float[D]` |
| $B, H, H_{kv}$ | batch, query heads, KV heads ($H \bmod H_{kv} = 0$) | `int64_t` |
| $T_q, T_k$ | query rows in this call, keys in the cache | `int64_t` |
| $p_i = q_{\text{offset}} + i$ | the absolute position of query row $i$ | `int64_t` |
| $\sigma$ | `scale`, usually $D^{-1/2}$ | `float` |
| $s_{ij} = \sigma\, q_i \cdot k_j$ | a score | `float` |
| $V_i$ | the keys row $i$ may see | set |
| $w$ | the window ($w \le 0$: none) | `int64_t` |
| $z_h$ | the sink logit of head $h$ | `float` |
| $B_r, B_c$ | query and key tile sizes | `int64_t` |
| $m_i, \ell_i, a_i$ | running maximum, denominator, and output of row $i$ | `float`, `float`, `float[D]` |
| $\mathrm{lse}_i$ | $\log\big(\sum_{j \in V_i} e^{s_{ij}} + e^{z_h}\big)$ | `float` |

### 2.1 What is computed

For every sequence $b$, head $h$, and query row $i$:

$$o_i = \frac{\sum_{j \in V_i} e^{s_{ij}}\, v_j}{\sum_{j \in V_i} e^{s_{ij}} + e^{z_h}}, \qquad V_i = \{\, j < T_k : (\text{causal} = 0 \text{ or } j \le p_i) \text{ and } (w \le 0 \text{ or } j > p_i - w) \,\}.$$

The $e^{z_h}$ term is present only with sinks. **GQA**: head $h$ reads KV head $\lfloor h / (H / H_{kv}) \rfloor$, so the $H/H_{kv}$ query heads of a group share one KV head (`L7.5`). **Positions are absolute**: row $i$ of a chunk that starts at token 24 is at $p_i = 24 + i$, and key $j$ is at position $j$, so the same flags serve a prefill ($q_{\text{offset}} = 0$, $T_q = T_k$), a chunk of a prefill, and a decode step ($T_q = 1$, $q_{\text{offset}} = T_k - 1$). A row with no visible key and no sink has no distribution: the contract says $o_i = 0$ and $\mathrm{lse}_i = -\infty$.

### 2.2 The tiled online softmax

`L9.2` showed that a running maximum $m$ and a running sum $\ell = \sum e^{s - m}$ can absorb a row's entries one block at a time: when a block raises the maximum to $m'$, multiply the old sum by $e^{m - m'}$. The same factor fixes the running output $a = \sum e^{s_j - m} v_j$, because each of its terms was weighted against the old maximum. For one key tile with visible scores $s_j$:

$$m' = \max\big(m, \max_j s_j\big), \quad \alpha = e^{m - m'}, \quad \ell' = \alpha\,\ell + \sum_j e^{s_j - m'}, \quad a' = \alpha\, a + \sum_j e^{s_j - m'}\, v_j .$$

After the last tile, $o = a / \ell$ and $\mathrm{lse} = m + \log \ell$. A sink is the cheapest possible tile: start each row at $m = z_h$, $\ell = e^0 = 1$, $a = 0$. Without a sink, start at $m = -\infty$, $\ell = 0$; the first visible tile then has $\alpha = e^{-\infty} = 0$, which multiplies a zero accumulator. A tile in which the row sees nothing must be skipped for that row, not processed: with $m = -\infty$ and no finite score, $m - m'$ is $-\infty - (-\infty) = \mathrm{NaN}$.

### 2.3 Tiles, scratch, and why memory stops growing

The kernel loops over work items (sequence, head, query tile of $B_r$ rows). For each item it keeps $m, \ell, a$ for $B_r$ rows, $B_r (D + 2)$ floats, and one tile of $B_c$ scores, and walks the key tiles that any of its rows can see. Each key tile is read from memory once per query tile and used by all $B_r$ rows while it is in cache, which is FlashAttention's whole point on a GPU (where the tiles live in shared memory) and helps on a CPU too. The scratch per worker is $B_r(D + 2) + B_c$ floats, from the caller's `rt.02` arena (one slice per worker, taken before the threads start, because the arena is not thread-safe). None of it depends on $T_k$. The kernel takes an arena mark on entry and rewinds to it before returning, so the caller's arena is exactly as it was; with `scratch == NULL` it uses a private arena and destroys it.

### 2.4 Invariance: key tiles at absolute positions

Key tile $t$ always covers positions $[tB_c, (t + 1)B_c)$, whatever the query tile, the chunk, or the batch. Within the walk, each row:

1. computes its own $V_i$ (its own `lo` and `hi`),
2. skips every tile that holds none of its keys,
3. in a tile that holds some, adds its visible keys in increasing $j$.

So the sequence of floating-point operations that produces row $i$ depends only on $q_i$, the keys and values, $p_i$, the flags, and $B_c$. Not on which other rows share its query tile ($B_r$), not on how the prompt was chunked ($q_{\text{offset}}$ moves $p_i$ and the row together), not on the batch, and not on the worker. That is the chunk invariance `L10.3` needs: query rows computed in one call or in chunks with `q_offset` are bitwise equal. Two tempting shortcuts break it: starting the tiles at a query tile's first visible key (the tile boundaries then move with the chunk), and deciding visibility once per query tile instead of per row.

### 2.5 Threads

Work items are independent: `tl_parallel_for` runs them on `rt.03` workers, each using the scratch slice of its worker index. Every row is computed by one worker with the arithmetic above, so 4 threads give the serial bits.

## 3. Worked example by hand

One query $q = [1, 0]$, three keys and values, no mask, $\sigma = 1$, $B_c = 2$:

| $j$ | $k_j$ | $v_j$ | $s_j = q \cdot k_j$ |
|---|---|---|---|
| 0 | $[1, 0]$ | $[1, 2]$ | 1 |
| 1 | $[0, 1]$ | $[3, 4]$ | 0 |
| 2 | $[1, 1]$ | $[5, 6]$ | 1 |

**Tile 0** (keys 0, 1). Start $m = -\infty$, $\ell = 0$, $a = [0, 0]$. Tile maximum 1, so $m' = 1$ and $\alpha = e^{-\infty} = 0$. Weights $e^{1-1} = 1$ and $e^{0-1} = 0.3678794$:

$$\ell = 0 + 1 + 0.3678794 = 1.3678794, \qquad a = 1 \cdot [1, 2] + 0.3678794 \cdot [3, 4] = [2.1036383, 3.4715177].$$

**Tile 1** (key 2). Tile maximum 1, $m' = 1$, $\alpha = e^0 = 1$ (no rescale). Weight $e^0 = 1$:

$$\ell = 1.3678794 + 1 = 2.3678794, \qquad a = [2.1036383 + 5, 3.4715177 + 6] = [7.1036383, 9.4715177].$$

**Normalize**: $o = a / \ell = [3, 4]$ exactly (because $(6 + 3e^{-1})/(2 + e^{-1}) = 3$ and $(8 + 4e^{-1})/(2 + e^{-1}) = 4$), and $\mathrm{lse} = m + \log \ell = 1 + \log 2.3678794 = 1.8619948$.

A correct kernel must reproduce these numbers to float32 precision. With a causal mask and $q_{\text{offset}} = 1$ the row would see only keys 0 and 1 (tile 0), giving $o = [2.1036383, 3.4715177]/1.3678794 = [1.5378828, 2.5378828]$.

## 4. The interface

```c
/* tinyllm/attention.h: q, o [B, H, Tq, D]; k, v [B, Hkv, Tk, D]; lse [B, H, Tq] or NULL */
tl_status tl_flash_attn_fwd_f32(const float *q, const float *k, const float *v, float *o, float *lse,
                                int64_t B, int64_t H, int64_t Hkv, int64_t Tq, int64_t Tk, int64_t D,
                                float scale, int64_t q_offset, int causal, int64_t window,
                                const float *sink_logits /* [H] or NULL */, int64_t Br, int64_t Bc,
                                tl_arena *scratch /* or NULL */, tl_pool *tp /* or NULL */);
/* Br, Bc = 0 pick 64. TL_EINVAL: negative sizes or q_offset, NULL tensors.
   TL_ESHAPE: H % Hkv != 0 or Hkv > H. TL_ENOMEM: the arena cannot grow. */
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3: $o = [3, 4]$, lse $= 1.8619948$ | the definition and the lse |
| every mask flag against naive attention | differential | GQA 6:2 and MQA, causal prefill, a chunk at `q_offset` 24, window 6, window without causal, plain; odd tiles and the default 64 | every flag of a Llama-family model |
| sinks join the denominator | unit | sinks against the naive kernel; all-ones values give $1 - p_{\text{sink}} = 2/3$ and lse $= \ln 3$ | gpt-oss style learned sinks |
| a row with no visible key | boundary | a query whose window excludes every key: zeros and $-\infty$; with a sink, zeros and lse = the sink | no NaN in the residual stream |
| chunk invariance with `q_offset` | property | 37 rows in one call vs chunks of 1, 3, 5, 16 with `q_offset`, bitwise | chunked prefill (`L10.3`) |
| query tile and batch invariance | property | $B_r = 1, 3, 64$ bitwise; a sequence alone vs in a batch of 3 | batching (`L10.2`) |
| arena rewound, scratch independent of $T_k$ | property | the arena is back at its mark; high water equal for $T_k = 16$ and 1000 | memory flat in the context length |
| pooled equals serial | property | 4 threads vs serial, bitwise and race-free | threads never change bits |
| shape and argument errors | boundary | `TL_ESHAPE` for 6 heads on 4 KV heads and $H_{kv} > H$; `TL_EINVAL` cases leave $o$ alone; $T_k = 0$ gives zeros | errors before any write |

## 5. The solution, step by step

### 5.1 From first principles

1. **The output of a row is a ratio of two sums over the visible keys** (section 2.1). Both sums can be accumulated in any grouping, so nothing forces us to hold all $T_k$ scores at once.
2. **Exponentials overflow, so each sum must be kept relative to a running maximum** (`L9.2`). When the maximum rises from $m$ to $m'$, every term already added was weighted by $e^{s - m}$ and must become $e^{s - m'}$: multiply both $\ell$ and $a$ by $\alpha = e^{m - m'}$. That is the whole tile update of section 2.2; in the worked example $\alpha$ was $0$ on tile 0 and $1$ on tile 1.
3. **The state per row is therefore $D + 2$ floats** ($m$, $\ell$, $a$), and a tile of scores is $B_c$ floats. Scratch per worker is $B_r(D + 2) + B_c$, independent of $T_k$. That is the memory claim, and it fixes the scratch layout.
4. **A sink behaves like a key with score $z_h$ and value 0.** Processing that key first gives the initial state $m = z_h$, $\ell = 1$, $a = 0$, so the sink costs nothing inside the loop.
5. **Bitwise invariance needs the float operations for a row to depend on that row alone.** Float addition is not associative, so the grouping of keys into tiles must be the same in every call. The only grouping that does not depend on the call (its chunk, its query tile, its batch) is one fixed by absolute key position: tile $t = [tB_c, (t+1)B_c)$.
6. **Visibility must be per row.** Rows of one query tile have different $p_i$, so different `lo` and `hi`. The query tile only decides the *union* of tiles to walk; each row then skips tiles that hold none of its keys, which also avoids the $-\infty - (-\infty)$ NaN.
7. **Work items (sequence, head, query tile) share nothing but read-only inputs**, so they parallelize without locks. Each worker needs private scratch, and since the arena is not thread-safe, the slices are allocated once before the threads start.

### 5.2 The code, annotated

The visible range of a row is a closed interval of key indices. Causal clips the top to $p$; a window of $w$ keys clips the bottom to $p - w + 1$ (so exactly $w$ keys, including the row's own position).

```c
/* The keys row p may see: [*lo, *hi] (empty when *lo > *hi). */
static void visible_range(const fa_args *a, int64_t p, int64_t *lo, int64_t *hi) {
    *hi = a->Tk - 1;
    if (a->causal && p < *hi) *hi = p;            /* causal: j <= p, a token sees itself */
    *lo = 0;
    if (a->window > 0 && p - a->window + 1 > *lo) *lo = p - a->window + 1; /* w keys: j > p - w */
}
```

One work item processes $B_r$ query rows of one head. The first block resolves the GQA head and initializes the per-row state, including the sink, while computing the union of visible ranges over the tile's rows.

```c
static void query_tile(const fa_args *a, int64_t b, int64_t h, int64_t qt, float *state, float *s) {
    const int64_t D = a->D, Bc = a->Bc;
    const int64_t kvh = h / (a->H / a->Hkv);   /* GQA: consecutive query heads share a KV head */
    const int64_t i0 = qt * a->Br;
    const int64_t p0 = a->q_offset + i0;       /* absolute position of the tile's first query */
    const int64_t nr = a->Tq - i0 < a->Br ? a->Tq - i0 : a->Br; /* last tile may be short */
    const float *qh = a->q + ((b * a->H + h) * a->Tq) * D;
    /* k and v may be NULL when Tk == 0, and NULL + 0 is undefined behavior. */
    const float *kh = a->Tk ? a->k + ((b * a->Hkv + kvh) * a->Tk) * D : NULL;
    const float *vh = a->Tk ? a->v + ((b * a->Hkv + kvh) * a->Tk) * D : NULL;
    float *m = state, *l = state + a->Br, *acc = state + 2 * a->Br; /* acc: [Br][D] */

    int64_t lo_min = INT64_MAX, hi_max = -1;   /* the keys ANY row of the tile sees */
    for (int64_t r = 0; r < nr; r++) {
        m[r] = a->sinks ? a->sinks[h] : -INFINITY; /* the sink is the first "key" */
        l[r] = a->sinks ? 1.0f : 0.0f;             /* e^(z - z) = 1 */
        for (int64_t d = 0; d < D; d++) acc[r * D + d] = 0.0f; /* the sink has no value */
        int64_t lo, hi;
        visible_range(a, p0 + r, &lo, &hi);
        if (lo <= hi) {
            if (lo < lo_min) lo_min = lo;
            if (hi > hi_max) hi_max = hi;
        }
    }
    // ...
```

The key loop is where invariance is decided. `j0` starts at `lo_min / Bc * Bc`, the start of the absolute tile that holds `lo_min`, not at `lo_min` itself. Inside the tile, each row intersects the tile with its own range and skips it if the intersection is empty.

```c
    for (int64_t j0 = hi_max < 0 ? 0 : lo_min / Bc * Bc; j0 <= hi_max; j0 += Bc) { /* aligned tiles */
        const int64_t j1 = j0 + Bc < a->Tk ? j0 + Bc : a->Tk;
        for (int64_t r = 0; r < nr; r++) {
            int64_t lo, hi;
            visible_range(a, p0 + r, &lo, &hi);       /* per row, never per tile */
            const int64_t ja = lo > j0 ? lo : j0, jb = hi < j1 - 1 ? hi : j1 - 1;
            if (ja > jb) continue;                    /* nothing visible: state untouched, no NaN */
            const float *qr = qh + (i0 + r) * D;
            float mt = -INFINITY;
            for (int64_t j = ja; j <= jb; j++) {      /* scores of this tile only: Bc floats */
                float dot = 0.0f;
                for (int64_t d = 0; d < D; d++) dot += qr[d] * kh[j * D + d];
                s[j - ja] = a->scale * dot;
                if (s[j - ja] > mt) mt = s[j - ja];
            }
            const float mn = mt > m[r] ? mt : m[r];
            const float alpha = tl_expf(m[r] - mn);   /* m = -inf, mn finite: alpha = 0 */
            float *ar = acc + r * D;
            float lt = 0.0f;
            for (int64_t d = 0; d < D; d++) ar[d] *= alpha;  /* rescale the OUTPUT too */
            for (int64_t j = ja; j <= jb; j++) {             /* increasing j: fixed order */
                float p = tl_expf(s[j - ja] - mn);
                lt += p;
                for (int64_t d = 0; d < D; d++) ar[d] += p * vh[j * D + d];
            }
            l[r] = l[r] * alpha + lt;                 /* rescale the denominator, then add */
            m[r] = mn;
        }
    }
```

Normalization guards the empty row: with no visible key and no sink, $\ell = 0$, and the contract asks for zeros and $-\infty$ rather than $0/0$. The lse includes the maximum, because $\ell$ was accumulated relative to it.

```c
    for (int64_t r = 0; r < nr; r++) {
        float *orow = a->o + ((b * a->H + h) * a->Tq + i0 + r) * D;
        float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;      /* empty row: zeros, not NaN */
        for (int64_t d = 0; d < D; d++) orow[d] = acc[r * D + d] * inv;
        if (a->lse) a->lse[(b * a->H + h) * a->Tq + i0 + r] = l[r] > 0.0f ? m[r] + logf(l[r]) : -INFINITY;
    }
}
```

The entry point validates everything before writing, takes an arena mark, allocates one slice per worker *before* starting the pool, runs the work items, and rewinds.

```c
    if (B == 0 || H == 0 || Tq == 0) return TL_OK; /* no output row */
    if (Hkv == 0 || Hkv > H || H % Hkv != 0) {     /* uneven groups would read past the KV heads */
        tl_set_last_error("tl_flash_attn_fwd_f32: need H % Hkv == 0 and 0 < Hkv <= H");
        return TL_ESHAPE;
    }
    // ... NULL checks, Br = Bc = 64 by default ...
    tl_arena_mark mark = tl_arena_mark_get(scratch);
    const int workers = tl_pool_threads(tp);
    const int64_t slice = Br * (D + 2) + Bc;       /* no Tk anywhere */
    float *buf = tl_arena_alloc(scratch, sizeof(float) * (size_t)(slice * workers), 64);
    // ... TL_ENOMEM path rewinds and returns ...
    const int64_t n_items = B * H * a.n_qtiles;   /* one item per (b, h, query tile) */
    if (tp == NULL)
        items(&a, 0, n_items, 0);
    else
        st = tl_parallel_for(tp, n_items, 1, items, &a); /* each worker uses slice[worker] */
    tl_arena_reset_to(scratch, mark);              /* the caller's arena is exactly as found */
    tl_arena_destroy(own);                         /* NULL when the caller supplied one */
```

The file also sets `#pragma STDC FP_CONTRACT OFF` so the compiler cannot fuse `dot += q * k` into an FMA in one build and not in another, which would change bits between builds.

Full solution: [`flash_attn.c`](../../../../course/solutions/c/src/kernels/flash_attn.c).

### 5.3 Why this solution

- **Cost.** Time is the same $O(B H T_q T_k D)$ as naive attention (fewer under a window or causal mask, since invisible tiles are skipped). Memory beyond inputs and outputs is $O(\text{workers} \cdot (B_r D + B_c))$ instead of $O(T_q T_k)$ per head.
- **Why not materialize scores and call `tl_softmax_f32`?** It is simpler and is what the naive reference does, but scratch grows with $T_k$ and every score is written and read back: the cost FlashAttention removes.
- **Why not start tiles at each query tile's first visible key?** It saves a partial tile of work, but tile boundaries then move with the chunk, the summation grouping changes, and chunked prefill (`L10.3`) stops being bitwise equal to a single call.
- **Why not split $T_k$ across workers (split-KV)?** It parallelizes a single long row better, but merging partial $(m, \ell, a)$ in an order that depends on the worker count gives up the "threads never change bits" guarantee. Parallelism here comes from (sequence, head, query tile), which is plentiful in prefill.
- **Tradeoff accepted:** `visible_range` is recomputed per row per tile, a few integer operations against $B_c D$ multiply-adds, in exchange for correct per-row visibility.

### 5.4 Explain it back

1. Why must the output accumulator be multiplied by $\alpha$, not only the denominator?
   <details><summary>Answer</summary>Each term of $a$ was added as $e^{s_j - m} v_j$. After the maximum rises to $m'$, the final ratio needs every term as $e^{s_j - m'} v_j$, a factor $e^{m - m'}$ smaller. Rescaling only $\ell$ would overweight earlier tiles by $e^{m' - m}$.</details>
2. Why does the key loop start at `lo_min / Bc * Bc` and not at `lo_min`?
   <details><summary>Answer</summary>So that tile boundaries are fixed absolute positions. A row then sums its keys in the same groups whatever chunk or query tile it is in, which makes outputs bitwise invariant to chunking and $B_r$.</details>
3. What goes wrong if a row with $m = -\infty$ processes a tile in which it sees no key?
   <details><summary>Answer</summary>The tile maximum is also $-\infty$, so $m - m' = -\infty - (-\infty) = \mathrm{NaN}$, and $\alpha = \mathrm{NaN}$ poisons $\ell$ and $a$ for good. The `if (ja > jb) continue;` skip prevents it.</details>
4. Why is the scratch allocated once for all workers before `tl_parallel_for`, rather than by each worker?
   <details><summary>Answer</summary>The arena is not thread-safe; concurrent `tl_arena_alloc` calls would race. One allocation of `workers` slices, indexed by the worker id the pool passes in, needs no lock.</details>
5. Why is lse $m + \log \ell$ and why does it matter that it is exact?
   <details><summary>Answer</summary>$\ell$ is $\sum e^{s_j - m}$, so $\log \sum e^{s_j} = m + \log \ell$. Any later merge of partial attention results (split-KV, or comparing against another kernel) uses lse to reweight, so omitting $m$ breaks those merges.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| not rescaling the output accumulator when the maximum grows | earlier tiles are overweighted by $e^{m' - m}$ | `ar[d] *= alpha` before adding the tile (5.2) |
| not rescaling the denominator | outputs do not average the values | `l[r] = l[r] * alpha + lt` |
| mapping query heads to KV heads round-robin ($h \bmod H_{kv}$) | GQA models attend with the wrong keys; MHA still passes | `kvh = h / (H / Hkv)`: consecutive heads share a KV head |
| causal as $j < p$ | a token cannot see itself | `visible_range` clips `hi` to `p`, inclusive |
| a window of $w + 1$ keys | drifts from the model's training | `lo = p - w + 1`, so exactly $w$ keys |
| ignoring `q_offset` | a chunk at position 24 behaves like the start of a prompt | `p0 = q_offset + i0`; every row uses its absolute position |
| leaving the sink out of the denominator | weights sum to 1; the sink does nothing | the sink initializes $m = z_h$, $\ell = 1$ |
| lse without the maximum ($\log \ell$ instead of $m + \log \ell$) | lse off by $m$; any later merge of partial results breaks | writes `m[r] + logf(l[r])` |
| dividing by $\ell = 0$ | a row that sees nothing becomes NaN | `inv = l > 0 ? 1/l : 0`, lse $-\infty$ |
| key tiles starting at a query tile's first visible key | the bits depend on the chunking | `j0` starts at `lo_min / Bc * Bc` (step 5) |
| not rewinding the caller's arena | the engine's step arena grows every call | mark on entry, `tl_arena_reset_to` on every exit path |
| a score buffer sized by $T_k$ | scratch grows with the context: the memory FlashAttention exists to save | the score buffer is $B_c$ floats per worker |
| a pooled partition that drops a work item | wrong only with threads | one flat index over `B * H * n_qtiles`, decoded in `items` |
| deciding visibility once per query tile | rows in one tile see the wrong keys; results depend on $B_r$ | `visible_range` per row inside the tile loop (step 6) |
| accepting $H \bmod H_{kv} \ne 0$ | head groups read past the KV heads | `TL_ESHAPE` before any write |
| pointer arithmetic on a NULL `k` when $T_k = 0$ | UBSan: "applying zero offset to null pointer" | `kh`, `vh` are NULL when `Tk == 0`, and no tile is walked |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the error slot and allocator support |
| Back | `rt.02` | the arena that holds the tiles, with a mark taken and rewound around each call |
| Back | `rt.03` | `tl_parallel_for` over (sequence, head, query tile) |
| Back | `M09.6` | `tl_expf` for every weight and rescale factor |
| Back | `L9.2` | the online softmax update this kernel applies to tiles; `tl_softmax_f32` is the naive reference the kernel is compared against |
| Back | `L7.7` | `windowed_attention`, the specification of every flag |
| Back | `L5.1` | `sdpa_forward`, the specification without a causal mask |
| Forward | `L9.4` | paged attention for decode performs the same tile update on blocks; it must match this kernel bit for bit |
| Forward | the standalone Rust engine | (Pass 7) the Rust forward's prefill; `L10.3`'s chunked prefill relies on the chunk invariance |

The optional C paged-attention chapter builds directly on this standalone C kernel.

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| CPU tiles in an arena | FlashAttention-2 (CUDA) | tiles in shared memory, warps split over queries, the backward pass by recomputation | `Dao-AILab/flash-attention`, `csrc/flash_attn/src/` |
| one kernel for all flags | FlashInfer | variants generated per mask, layout, and positional encoding; paged KV inputs | `flashinfer/include/flashinfer/attention/` |
| float32 scores | FlashAttention-3 | fp8 inputs with incoherent processing, asynchronous tiles on Hopper | Shah et al. (2024) |
| fixed tile order for invariance | split-KV (FlashDecoding) | splits long contexts across thread blocks and merges partial (m, l, a) with the same rescale, trading invariance for parallelism | Dao et al., "Flash-Decoding for long-context inference" (2023) |
| float32 KV only | llama.cpp `ggml_flash_attn_ext` | CPU and GPU flash attention over quantized KV | `ggml/src/ggml-cpu/ops.cpp` |
