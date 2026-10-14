<!-- ol:module L14.3 -->
# Whisper decoder and checkpoint weights

## Overview

| | |
|---|---|
| **Module** | `L14.3` · build · Python · Pass 12 · 4 to 6 h |
| **You build** | `python/tinyllm/speech/decoder.py` and `python/tinyllm/io/hf_whisper.py` |
| **Contract** | `course/contracts/py/tinyllm/speech/decoder.pyi`, `course/contracts/py/tinyllm/io/hf_whisper.pyi` |
| **Tests** | `course/tests/L14.3/test_decoder.py` |
| **Needs** | `L14.1`; `L14.2`, `L8.2`, `L0.6`, `L0.4`, `L0.2`, `L0.1`, `L5.1`, `L5.2`, `L5.4`, `M06.3`. Reading: `L7.9` |
| **Used by** | `ethics.08`, `L14.4` |
| **Milestone** | `MS-L14` |
| **Optional depth** | Compare token-at-a-time and chunked decoding, including a batch with different prompts. |

## Key Takeaways

- Self-attention sees only the decoder prefix; cross-attention sees the complete encoded audio window.
- Cross keys and values are projected once per window. Self keys and values grow at each decoding step.
- Output logits use the token embedding matrix transposed. A separate untied output weight is an invalid checkpoint.

## How to work this chapter

```bash
ol start L14.3
ol tests L14.3
ol check L14.3 --ref-deps
ol diff L14.3
```

## 1. Why now

L14.2 turns audio features into vectors, but cannot generate a transcript. The decoder compares its text prefix with those vectors and predicts the next token. This module supplies logits and checkpoint loading. L14.4 will supply language, timestamp and search policies.

## 2. Principles

At absolute decoder position `o + t`, add the learned positional row to the token embedding. Each block applies pre-normalized causal self-attention, pre-normalized cross-attention, and a pre-normalized GELU feed-forward network, with a residual addition after each operation. Apply the final layer norm, then multiply by the transpose of the token embedding matrix.

Cross keys and values depend on the audio window and the decoder layer, so compute them once and retain one pair per layer. Self-attention caches retain the prefix. Read the common committed cache length before any layer appends; every layer uses that same position offset and causal mask. Validate every cache shape before the first append so a rejected request cannot partially mutate the cache.

The loader removes `model.` from parameter names, verifies an optional tied output projection, and separates the encoder's frozen sinusoid table from trainable parameters. Strict state loading rejects missing, unexpected or wrongly shaped tensors. For sharded files, each tensor must occur in the file named by the index, exactly once.

## 3. Worked example

`test_hf_logits_and_parameter_count` extends the following prefix calculation to the independent HF model.

For a two-token prefix `[sot, language]`, a fresh cache has offset zero. The self-attention permission matrix is `[[True, False], [True, True]]`. Both tokens may attend to every encoded audio position. Every decoder layer appends two self-key and self-value rows, making the committed length two.

The next one-token call uses positional row two and a `1 x 3` all-true self-attention mask. It reuses the unchanged cross keys and values. Its last logits must agree with recomputing all three prefix tokens in one uncached call. If the checkpoint stores an output projection, each element must equal the corresponding token-embedding element before the redundant tensor is discarded.

## 4. The interface and tests

| Test | Kind | Contract |
|---|---|---|
| `test_hf_logits_and_parameter_count` | golden | Independent HF logits within absolute tolerance `1e-5`; parameter count includes the frozen encoder position table. |
| `test_cached_chunks_equal_full_and_reuse_cross_kv` | property | Chunked and full decoding agree; cross keys and values remain unchanged. |
| `test_causality_and_one_dimensional_ids` | property | A changed suffix cannot affect prefix logits; one-dimensional ids preserve the documented output shape. |
| `test_bad_tokens_and_positions_are_rejected` | boundary | Invalid token ids, ranks and positions leave the cache untouched. |
| `test_cross_kv_and_cache_shapes_are_validated` | boundary | Batch, layer, head and feature widths are checked before decoding. |
| `test_loader_rejects_untied_weights_and_wrong_sinusoids` | boundary | Incompatible weight names, positions and output ties fail explicitly. |
| `test_sharded_loader_and_index_integrity` | conformance | Sharded and unsharded states agree; paths cannot escape the checkpoint directory. |
| `test_loader_is_strict_about_missing_and_wrong_shapes` | boundary | Incomplete checkpoints cannot leave random parameters in the model. |

| `test_wav_entry_point_matches_explicit_frontend` | WAV channel mixing, padding and mel features produce the same cross-attention keys and logits as the explicit pipeline. |

## 5. Pitfalls

| Mistake | Symptom | Test |
|---|---|---|
| Reset position offset at every incremental call | Full and cached logits disagree | `test_cached_chunks_equal_full_and_reuse_cross_kv` |
| Omit the causal mask | Future tokens change earlier predictions | `test_causality_and_one_dimensional_ids` |
| Load an untied output matrix silently | Plausible but incorrect transcript scores | `test_loader_rejects_untied_weights_and_wrong_sinusoids` |
| Advance the committed length once per layer | Later layers use different positions | `test_cached_chunks_equal_full_and_reuse_cross_kv` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `L14.2` | Encoder output and Whisper attention projections. |
| Back | `L8.2` | Self-attention cache and offset masks. |
| Back | `L0.6` | Safetensors reader. |
| Back | `L0.4` | Registered modules, embeddings, normalization and strict state loading. |
| Back | `L0.2` | Differentiable tensor operations. |
| Back | `L0.1` | Tensor storage and shapes. |
| Back | `L5.1` | Scaled dot-product attention. |
| Back | `L5.2` | Causal mask construction. |
| Back | `L5.4` | Learned positional embeddings. |
| Back | `M06.3` | Reproducible parameter initialization. |
| Back | `L7.9` | Download the checkpoint before calling the local loader. |
| Forward | `L14.4` | Transcription policy consumes these logits and reuses cross-KV per window. |

| Back | `L14.1` | `encode_wav` decodes, mixes channels, resamples and computes model-sized log-mel features. |

| Forward | `ethics.08` | The consent and model-card evidence practice calls this implementation. |

## Going further

| Resource | Exercise |
|---|---|
| [Whisper contract](../../../course/contracts/py/tinyllm/speech/decoder.pyi) | Derive the parameter count, then compare one-token and multi-token cache updates. |
