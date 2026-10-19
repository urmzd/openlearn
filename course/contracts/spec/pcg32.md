# PCG32: the one random generator

<!-- modules: M06.3 (Python tinyllm/num/rng.py), L10.1 (Rust port), load.01 (Go port go/ds/rng)
     vectors: spec/pcg32.vectors.json -->

Every random draw in the system, in every language, comes from this generator so a seed means the same thing in Python training, the Rust sampler, and the Go load generator.

Symbols: `u64` arithmetic is modulo 2^64, `u32` modulo 2^32; `>>` is a logical shift; `rotr32(x, r)` rotates a 32-bit value right by `r`.

## Generator

PCG-XSH-RR with 64-bit state and 32-bit output (O'Neill 2014), exactly as the reference `pcg32_random_r`:

```
state: u64, inc: u64 (always odd)

next_u32():
    old   = state
    state = old * 6364136223846793005 + inc
    xs    = u32( ((old >> 18) XOR old) >> 27 )
    rot   = old >> 59
    return rotr32(xs, rot)
```

**Seeding** is the reference `pcg32_srandom_r(seed, seq)`: `inc = (seq << 1) | 1`, `state = 0`, `next_u32()`, `state = state + seed`, `next_u32()`. The default stream is `seq = 54`, so `pcg32(seed)` means `pcg32_srandom_r(seed, 54)`. Check: `pcg32(42)` yields `0xa15c02b7 0x7b47f409 0xba1d3330 0x83d2f293 0xbfa4784b 0xcbed606e`, the first line of O'Neill's `pcg32-demo`.

## Derived draws

| Draw | Definition | Draws used |
|---|---|---|
| `uniform_f64()` | `a = next_u32(), b = next_u32()`, then `((a >> 5) * 2^26 + (b >> 6)) * 2^-53`: 53 random bits, uniform on `[0, 1)` | 2 |
| `normal()` | Box-Muller. Without a spare: `u1 = uniform_f64(), u2 = uniform_f64()`, `r = sqrt(-2 ln(1 - u1))` (`1 - u1` is never 0), return `r cos(2 pi u2)` and keep `r sin(2 pi u2)` as the spare. With a spare: return it and clear it. | 4 per pair |
| `below(n)`, `1 <= n <= 2^32` | `t = (2^32 - n) mod n`; draw `r = next_u32()` until `r >= t`; return `r mod n` (unbiased) | 1 or more |
| `shuffle(xs)` | Fisher-Yates from the end: for `i = len - 1` down to `1`: `j = below(i + 1)`, swap `xs[i]` and `xs[j]` | `len - 1` or more |
| arrays | `uniform_array`, `normal_array`, `below_array` fill in C (row-major) order, one scalar draw per element; the normal spare carries across calls. M07.0's `tinyllm.prob.rv.normal(rng, n)` is not `normal_array`: it draws `ceil(n / 2)` fresh pairs, keeps no spare, and drops the last sine when `n` is odd. A port of that function (L10.1, load.01) follows the same rule. Both agree for even `n` from a fresh generator | |

## Sub-streams

Independent purposes use independent generators derived from one user seed, so adding dropout does not change the initialization:

```
child_seed(seed, p) = mix64(seed + p * 0x9E3779B97F4A7C15)      (the p-th output of SplitMix64 seeded with seed)
stream(seed, purpose) = pcg32_srandom_r(child_seed(seed, p), p)  with p the purpose id below

mix64(z): z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
          z = (z XOR (z >> 27)) * 0x94D049BB133111EB
          return z XOR (z >> 31)
```

| Purpose | id | Used for |
|---|---|---|
| `init` | 1 | parameter initialization, in parameter registration order |
| `dropout` | 2 | dropout masks, in forward order |
| `shuffle` | 3 | data order: window starts of the training loader (saved in `trainer_state.json`) |
| `sample` | 4 | token sampling: one generator per request, `stream(request.seed, sample)` (spec/sampling.md) |
| `mutation` | 5 | test-data mutation and property tests |

The Pass 1 tracer predates this page: L0.0 samples with numpy and L10.0 with `pcg32(seed)`. From L8.1 (Python) and L10.1 (Rust) every sampler uses `stream(seed, sample)`.

## Reference vectors

[`pcg32.vectors.json`](pcg32.vectors.json), generated from this page alone:

| Key | Content |
|---|---|
| `next_u32` | the first 1024 outputs of `pcg32(s)` for `s` in `0, 1, 2^63` |
| `uniform_f64`, `normal`, `below_10` | the first 8, 8, and 16 draws of a fresh `pcg32(s)` |
| `shuffle_10` | `shuffle([0..9])` on a fresh `pcg32(s)` |
| `child_seed`, `stream_next_u32` | per seed and purpose: the derived seed, and the first 4 outputs of `stream(s, purpose)` |

Integer vectors must match exactly, and `uniform_f64` bit for bit. `normal` goes through the platform's `log`, `sqrt`, `sin`, `cos`, so it is compared within 4 ulp.

## Worked example: the first output of `pcg32(0)`

1. Seeding: `inc = (54 << 1) | 1 = 109`; `state = 0`; one step gives `state = 0 * 6364136223846793005 + 109 = 109`; add the seed 0; one more step gives `state = 109 * 6364136223846793005 + 109 = 0x9AE4F7499BA72696` (mod 2^64).
2. `next_u32()` with `old = 0x9AE4F7499BA72696`: `xs = u32(((old >> 18) XOR old) >> 27) = 0x5C9A3E14`, `rot = old >> 59 = 19`.
3. `rotr32(0x5C9A3E14, 19) = 0x47C28B93`, the first entry of `next_u32["0"]` (1203932051).
