<!-- ol:module M09.4 -->
# FP8 E4M3/E5M2, MXFP4/MXFP8 with E8M0 scales

## Overview

| | |
|---|---|
| **Module** | `M09.4` · build · Python · Pass 6 |
| **You build** | `python/tinyllm/num/lowp.py`: FP8 E4M3/E5M2 conversion, FP4 E2M1 values, block scales, and MX quantize/dequantize |
| **Contract** | [`course/contracts/py/tinyllm/num/lowp.pyi`](../../../../course/contracts/py/tinyllm/num/lowp.pyi); expected bytes are in `course/fixtures/M09.4/lowp_golden.npz` |
| **Tests** | `course/tests/M09.4/test_lowp.py`: exact code tables, tie behavior, saturation, signed zero, block layouts, and round trips |
| **Needs** | `M09.1` for IEEE-754 bit layouts; `S-M09a` for rounding by hand |
| **Used by** | `L8.5` quantizes weights; `L10.1` converts Rust model weights; `craft.13` stores fp8 KV data |
| **Milestone** | `MS-L8` |

## 1. Why now

Float32 weights and activations move four bytes per value. FP8 stores each value in one byte; MX formats group values into blocks and share a scale, while FP4 stores four bits per value. These formats reduce memory traffic, but every conversion has edge cases: ties, tiny magnitudes, signed zero, infinities, NaNs, and values beyond the finite range. This module writes the conversions from the format definitions and tests the resulting bit patterns against checked-in vectors.

An optional standalone C conversion exercise lives in `M09.7`. It consumes the same fixture files and does not load this Python code.

## 2. Principles

An IEEE-like binary format uses a sign bit, an exponent field, and a fraction field. For a normalized value with fraction width `m` and bias `b`, the decoded value is

\[
(-1)^s (1 + f/2^m) 2^{e-b}.
\]

Subnormals omit the implicit leading one and use the minimum exponent. E4M3 uses four exponent bits and three fraction bits, giving more precision near one than E5M2. E5M2 uses two fraction bits and therefore covers a wider range. The project defines saturation and special-value behavior explicitly, so callers do not depend on a library default.

Rounding is round-to-nearest, ties-to-even. A tie at `2.5` selects `2`; a tie at `3.5` selects `4`. FP8 encode/decode uses exact powers of two to map between values and codes. Microscaling formats apply one E8M0 power-of-two scale to a fixed-size block, then encode each scaled value in FP8 or E2M1.

## 3. Worked example

In E4M3, `1.0625` lies halfway between adjacent representable values `1.0` and `1.125`. Their final significand bits are even and odd respectively, so ties-to-even encodes `1.0`. A value just above the midpoint encodes `1.125`. The test `test_ties_go_to_even` checks both sides of these boundaries as well as the exact midpoint.

For an MX block, first find the largest absolute element. Choose the shared power-of-two scale that moves that maximum into the element format's top range, then divide each value by the scale and round once into the element format. Decoding multiplies each value by the same scale. The fixture's block example lets the tests check the scale code and every packed element independently.

## 4. What the tests establish

| Test | Why it matters |
|---|---|
| `test_hand_example` and `test_hand_example_mx` | Pin scalar and block calculations to hand-worked values. |
| `test_decode_every_code_golden` and `test_encode_golden` | Check the full 8-bit code space against stable expected values. |
| `test_saturation_and_specials` | Defines behavior for overflow, infinities, and NaNs. |
| `test_tiny_values_and_signed_zero` | Protects underflow and sign preservation at zero. |
| `test_encode_inverts_decode` | Every finite code remains stable through decode then encode. |
| `test_rounding_is_nearest_and_monotone` and `test_ties_go_to_even` | Catches incorrect midpoint handling and non-monotone encoders. |
| `test_scaled_quantization_roundtrip` and `test_quantize_rounds_once` | Ensures scaling does not introduce an extra rounding step. |
| `test_fp4_values_and_ties`, `test_e8m0_scale_code`, `test_mx_golden`, `test_mx_block_law`, `test_mx_shapes_and_errors` | Check element codes, scale choices, block layout, and input validation. |

Start with `ol start M09.4`, inspect the catalog with `ol tests M09.4`, then run `ol check M09.4`. The file fixture makes results reproducible without optional dependencies at runtime.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Rounding half away from zero | Exact midpoints choose the wrong code. | `test_ties_go_to_even` |
| Treating subnormals like normalized values | Tiny values jump to zero or the minimum normal. | `test_tiny_values_and_signed_zero` |
| Producing infinity on finite overflow | Codes disagree with the course's saturation rule. | `test_saturation_and_specials` |
| Applying the block scale after rounding | Quantized values differ by one code near boundaries. | `test_quantize_rounds_once` |
| Choosing scale from an average instead of the maximum magnitude | A large block member clips unexpectedly. | `test_mx_block_law` |
| Accepting malformed block shapes | Values are silently dropped or grouped incorrectly. | `test_mx_shapes_and_errors` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.1` | Provides the float bit layouts and exact powers-of-two reasoning. |
| Forward | `L8.5` | Quantizes model weights for lower-bandwidth inference. |
| Forward | `L10.1` | Converts quantized model weights in the Rust engine. |
| Forward | `craft.13` | Stores fp8 KV values and relies on the defined saturation and decoding rules. |
| Optional parallel | `M09.7` | Implements selected conversions in standalone C against the shared golden files. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| FP8 E4M3/E5M2 | accelerator matrix formats | trades exponent range against significand precision | IEEE 754 and vendor format guides |
| microscaling | block-scaled MX formats | compact values share a power-of-two scale per block | `course/fixtures/M09.4/lowp_golden.npz` |
