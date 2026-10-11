# Part 9: Optional standalone C kernels

This part contains independent C exercises for data structures, numerical
routines, and neural-network kernels. The C units compile into standalone test
programs with shared C support from `rt.02`; Python and Rust implementations
remain independent. Cross-language parity uses checked-in files and process
protocols.

**Course passes**: Pass 6 optional practice, with runtime support in `rt.02`
and `rt.03`. These modules are not required by the Python or Rust engine paths.

**Build contract**: `c/Makefile` produces a static archive for local C use.
The harness compiles the declared C units into standalone test executables;
`SANITIZE=1` enables ASan and UBSan. The rules are in
[`c/ABI.md`](../../../course/contracts/c/ABI.md).

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `rt.02` | [C runtime support and arena allocator with marks (optional)](02-arena-allocator-with-marks.md) | side | 6 |
| 2 | `rt.03` | [Thread pool and tl_parallel_for (optional C)](03-thread-pool-and-parallel-for.md) | side | 6 |
| 3 | `L9.1` | [Cache-blocked, packed, batch-invariant matmul in C (optional)](04-tiled-batch-invariant-matmul.md) | side | 6 |
| 4 | `L9.2` | [Softmax in C: three-pass and online two-pass](05-softmax-three-pass-and-online.md) | side | 6 |
| 5 | `L9.3` | [FlashAttention forward in C](06-flash-attention-forward.md) | side | 6 |
| 6 | `L9.4` | [Paged attention for decode in C](07-paged-attention-decode.md) | side | 6 |
| 7 | `L9.5` | [Fused int4 and int8 dequantize-matmul in C](08-fused-quantized-matmul.md) | side | 6 |
| 8 | `L9.6` | [Elementwise kernels in C: RMSNorm, RoPE, SiLU-mul, embedding, add, argmax](09-elementwise-kernels.md) | side | 6 |
<!-- /ol:chapters -->
