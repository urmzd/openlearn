# Kernels and Numerical Performance

Up to this point every forward pass in the book has run through numpy, and the arithmetic has been someone else's problem. This chapter takes it back. You will write, in plain C11, the kernels a transformer actually spends its time in: matrix multiplication, softmax, attention for prefill and for paged decode, quantized matmul, and the small elementwise operations between them. Along the way you will see why fast kernels are mostly about memory rather than arithmetic, and how to make them fast without letting the answer change with the batch size, the chunking of a prompt, or the number of threads.

The chapter is optional. Its C units compile into standalone test programs, and nothing here is loaded by Python or linked into the Rust engine, which keeps its own independent implementations; cross-language agreement is checked through checked-in files and process protocols. The arc runs from infrastructure to kernels. It opens with the support layer every C unit reports through and an arena allocator for scratch memory, then a thread pool whose fixed partition makes parallel results bitwise reproducible. With those in hand, the tiled matmul introduces the roofline, packing, and batch invariance; the softmax section derives the online update; FlashAttention and paged attention build on that update; and the quantized matmul and elementwise kernels complete the set.

> **Learning objectives**
> - Build allocation and threading infrastructure for kernels that never allocate in the steady state and never change bits with the thread count.
> - Explain kernel performance with the roofline model, and use blocking, packing, and quantization to move a kernel toward its bound.
> - Derive the online softmax and apply it to tiled and paged attention without ever storing the score matrix.
> - Make every kernel batch-invariant by fixing its reduction order, and explain which common optimizations break that property.
> - Handle the edge cases (masked rows, NaN, empty contexts, bad tables, misaligned views) that separate a correct kernel from a plausible one.

## Before you begin

You should be comfortable with C pointers, structs, headers, and the C11 toolchain, as covered in [C: memory, pointers, structs, headers, and the C11 toolchain](../../programming-and-algorithms/01-language-primers/03-c.md) in *Programming and Algorithms*. The kernels implement operations you have already met in Python: [scaled dot-product attention](../11-transformer/01-scaled-dot-product-attention.md), the modern block's [RMSNorm](../13-decoder-block/01-pre-ln-rmsnorm.md), [RoPE](../13-decoder-block/03-rope.md), and [GQA](../13-decoder-block/05-gqa-attention.md), the [paged KV cache](../21-inference/03-paged-kv-cache.md), and [quantization](../21-inference/05-quantization.md). Several numerical helpers come from *Mathematics*, notably [stable softmax](../../mathematics/15-numerical-methods-and-floating-point/02-stable-numerics.md), [`tl_expf`](../../mathematics/15-numerical-methods-and-floating-point/06-polynomial-approximation-and-expf.md), and the [low-precision conversions](../../mathematics/15-numerical-methods-and-floating-point/07-low-precision-c-conversions.md).

The build rules are in [`c/ABI.md`](../../../../course/contracts/c/ABI.md). `c/Makefile` produces a static archive for local C use, the test harness compiles each C unit into a standalone test executable, and `SANITIZE=1` enables AddressSanitizer and UndefinedBehaviorSanitizer. The concurrency tests are also meant to run under ThreadSanitizer.

## In this chapter

1. [C runtime support and an arena allocator with marks](01-arena-allocator-with-marks.md). The status, error, and allocator-hook layer every C unit uses, and a bump allocator with nested marks that makes zero allocator calls once warmed up.
2. [Thread pool and tl_parallel_for](02-thread-pool-and-parallel-for.md). A sleeping pool with one atomic counter and a fixed partition, so parallel kernels give the same bits on any number of threads, and nested calls, concurrent callers, and shutdown are all safe.
3. [Cache-blocked, packed, batch-invariant matmul in C](03-tiled-batch-invariant-matmul.md). Goto's loop nest with packing and a register micro-kernel, 10 to 20 times faster than the triple loop, with a fixed reduction order that makes each row independent of the batch.
4. [Softmax in C: three-pass and online two-pass](04-softmax-three-pass-and-online.md). Overflow-free softmax, the online update with its invariant, and the exact handling of masked rows and NaN.
5. [FlashAttention forward in C](05-flash-attention-forward.md). Tiled attention that never stores the score matrix, with GQA, causal and window masks at absolute positions, sinks, and bitwise chunk invariance.
6. [Paged attention for decode in C](06-paged-attention-decode.md). One decode step over a block-table cache in f16, allocating nothing and matching FlashAttention bit for bit.
7. [Fused int4 and int8 dequantize-matmul in C](07-fused-quantized-matmul.md). Matmul straight from packed 4-bit and 8-bit weights, about 3.9 times faster than float32 at decode, with eight lanes that stay batch-invariant.
8. [Elementwise kernels in C: RMSNorm, RoPE, SiLU-mul, embedding, add, argmax](08-elementwise-kernels.md). Six small operations and the one detail in each that silently changes a model's output.
