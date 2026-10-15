# Part 0: Foundations

The model side of the tracer and, from Pass 2, the autograd engine every later model trains with. Pass 1 has one chapter: a byte-level bigram fitted by counting, whose logits use NumPy matmul and whose weights you write as a safetensors checkpoint the Rust engine serves. Pass 2 retrains the same bigram with your autograd (L0.5 takes over `bigram.py`) behind the same checkpoint contract.

**Course passes**: 1 (L0.0, gate [MS-P1](../paths/course-p01-tracer/milestone.md)), 2 (L0.1 to L0.6, gate MS-P2).

**Before you start**: [Python and NumPy](../../../tracks/software-craftsmanship/12-language-and-tool-primers/01-python-and-numpy.md) and [matrix multiplication in Python](../../../tracks/math/03-linear-algebra/01-vectors-matrices-and-matmul.md).

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L0.0` | [Byte bigram: counts, logits, safetensors v0](00-byte-bigram.md) | build | 1 |
| 2 | `L0.1` | [Tensor, broadcasting backward, and no_grad](01-tensor-and-broadcasting-backward.md) | build | 2 |
| 3 | `L0.2` | [The op library and gradcheck_all](02-op-library-and-gradcheck.md) | build | 2 |
| 4 | `L0.3` | [Losses with a fused backward](03-losses-with-fused-backward.md) | build | 2 |
| 5 | `L0.4` | [Module system and basic layers](04-module-system-and-layers.md) | build | 2 |
| 6 | `L0.5` | [Training loop and the autograd bigram](05-training-loop-and-the-autograd-bigram.md) | build | 2 |
| 7 | `L0.6` | [Safetensors for every dtype, atomic checkpoints, and the token stream](06-safetensors-checkpoints-and-token-streams.md) | build | 2 |
<!-- /ol:chapters -->
