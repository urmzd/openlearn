<!-- ol:module M09.7 -->
# Low-precision conversions in standalone C (optional)

## Overview

| | |
|---|---|
| **Module** | `M09.7` · side · C · Pass 6 · 2 to 3 h |
| **You build** | `c/src/numerics/lowp.c`: float32 conversions to and from E4M3, E5M2, bfloat16, and binary16 |
| **Contract** | [`tinyllm/numerics.h`](../../course/contracts/c/include/tinyllm/numerics.h) |
| **Tests** | `course/tests/M09.7/test_lowp.c`, compiled into a standalone sanitized C test binary |
| **Needs** | `M09.4` for the Python value and rounding rules |
| **Used by** | Optional C modules `L9.4` and `L9.5`, through their C headers only |
| **Milestone** | `MS-L9`, the optional C module group |
| **Optional depth** | Compare the OCP FP8 rules with IEEE binary16's infinity and NaN encodings |

## Key Takeaways

- Encoding a value means rounding its significand in units of the destination format's quantum; `hand_example` checks the exact E4M3 code for 0.3.
- E4M3 finite overflow saturates, while E5M2 reserves infinity and NaN codes; `fp8_saturation_and_specials` catches mixing these policies.
- Ties to even depend on the low bit of the retained significand, not the sign; `fp8_ties_to_even` checks both sides of that decision.
- Every finite encoding should decode and re-encode to the same bits; exhaustive fixed point tests cover both FP8 formats, bfloat16, and binary16.

## How to work this chapter

```bash
ol start M09.7
ol tests M09.7
ol check M09.7
ol diff M09.7
```

---

## 1. Why now

The Python implementation in `M09.4` describes how values map to compact formats. The optional C kernels in `L9.4` and `L9.5` need the same conversions when reading half precision cache values and quantization scales. This exercise keeps those C routines independently testable. It uses ordinary C calls inside one process; data used for cross-language comparison is stored in fixture files.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $M$ | Number of retained fraction bits | integer: 3 for E4M3, 2 for E5M2, 7 for bfloat16, 10 for binary16 |
| $B$ | Exponent bias | integer: 7 for E4M3, 15 for E5M2 and binary16 |
| $e_{min}$ | Smallest normal exponent, $1-B$ | integer |
| $q_e$ | Exponent of the spacing between adjacent representable values | integer: $\max(\lfloor\log_2 v\rfloor,e_{min})-M$ |

A positive normal float32 is $v = s 2^{e_2}$ with an integer significand $s$ of at most 24 bits. A destination format with $M$ fraction bits and bias $B$ has smallest normal exponent $e_{min}=1-B$. In the binade containing $v$, adjacent destination values are separated by $2^{q_e}$ where

$$q_e = \max(\lfloor \log_2 v \rfloor, e_{min}) - M.$$

The conversion rounds $v / 2^{q_e}$ to an integer using round-to-nearest, ties-to-even. If the discarded bits are greater than half a unit, increment the retained integer. If they equal half a unit, increment only when the retained integer is odd. A carry can increment the exponent. Values below the smallest normal use the fixed subnormal quantum $2^{e_{min}-M}$.

Special values are format rules applied around that rounding core. In this course E4M3 saturates finite magnitudes above 448. E5M2 uses the all-one exponent for infinity and NaN. Signed zero preserves its sign bit. bfloat16 rounds the high half of float32 with the low half deciding ties. Binary16 uses 5 exponent bits and 10 fraction bits.

## 3. Worked example by hand

For E4M3, $M=3$, $B=7$, and $e_{min}=-6$. The float32 value 0.3 lies between $2^{-2}=0.25$ and $2^{-1}=0.5$, so its binade exponent is $-2$. Thus $q_e=-2-3=-5$ and one destination step is $2^{-5}=0.03125$. Dividing gives $0.3 / 0.03125 = 9.6$, which rounds to 10. The retained significand is 10, or binary `1010`. The low 3 bits are the fraction, `010`; the carry into the exponent makes the final code `0x2A`, whose decoded value is 0.3125. `hand_example` checks both the code and decoded value.

## 4. The interface

The header declares scalar encode and decode functions. Encoders receive one `float` and return the exact stored bit pattern in an unsigned integer. Decoders receive that integer and return a float. Array conversion functions apply the scalar rule element by element, with input and output lengths supplied by the caller. No function allocates memory or retains a pointer.

### What the tests check

| Test | KIND | Checks | Why it matters |
|---|---|---|---|
| `hand_example` | unit | E4M3 code `0x2A` and decoded value 0.3125 for 0.3 | Locks the section 3 arithmetic to the interface |
| `fp8_saturation_and_specials` | boundary | E4M3 saturation, E5M2 infinity and NaN behavior | A shared overflow branch gives one format the wrong policy |
| `fp8_tiny_and_signed_zero` | boundary | subnormal ties and signed zero | Small values otherwise disappear or lose their sign |
| `fp8_every_code_is_a_fixed_point` | exhaustive | finite E4M3 and E5M2 codes survive decode then encode | Finds errors across exponent transitions |
| `fp8_ties_to_even` | boundary | midpoint values choose the even code | Truncation and ties-away-from-zero both fail |
| `bf16_hand_values` | unit | representative bfloat16 values and tie behavior | The high-half shortcut still needs correct rounding |
| `bf16_every_code_round_trips` | exhaustive | all bfloat16 codes preserve finite values and specials | Checks the full 16-bit format |
| `f16_hand_values` | unit | binary16 normal and subnormal examples | Subnormal scaling differs from normal scaling |
| `f16_every_code_round_trips` | exhaustive | all finite binary16 codes are fixed points | Exercises all exponent and fraction combinations |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Always incrementing on a half-way remainder | Alternating midpoint values round away from even | `fp8_ties_to_even` |
| Applying E5M2 infinity rules to E4M3 | Large finite values become infinity instead of 448 | `fp8_saturation_and_specials` |
| Using the normal quantum below $e_{min}$ | Smallest subnormal codes decode at the wrong scale | `fp8_tiny_and_signed_zero`, `f16_every_code_round_trips` |
| Dropping the sign before handling zero | `-0.0` returns as positive zero | `fp8_tiny_and_signed_zero` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.4` | Defines the Python-side format rules and produces comparison vectors |
| Forward | `L9.4` | Decodes half precision values stored in the optional paged-attention C exercise |
| Forward | `L9.5` | Decodes half precision quantization scales in the optional fused matmul C exercise |

Skipping this exercise leaves the C modules that use these conversion routines without their declared prerequisite. The optional milestone remains separate from the production engine milestones.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `lowp.c` | Device conversion instructions | Hardware support can round or flush subnormals differently | OCP FP8 and IEEE 754 specifications |
| exhaustive format tests | Compiler and device conformance suites | Runs the same vectors on additional architectures | Add a standalone C target and compare output files |
