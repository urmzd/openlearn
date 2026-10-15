<!-- ol:module L12.1 -->
# Supervised fine-tuning: chat rendering, assistant-only loss, packing

## Overview

| | |
|---|---|
| **Module** | `L12.1` · optional build · Python · Pass 10 · 2 to 3 h |
| **You build** | `tinyllm/post/sft.py`: rendering, assistant masks, and document packing |
| **Contract** | [`sft.pyi`](../../../../course/contracts/py/tinyllm/post/sft.pyi) |
| **Tests** | `course/tests/L12.1/` (why: renderer parity, loss masks, and isolation between packed examples) |
| **Needs** | `L6.6` LoRA, `L0.3` cross-entropy, `L10.5` chat template |
| **Used by** | `C2` post-trained capstone |
| **Milestone** | [MS-C2](../../../../course/milestones/MS-C2.toml) |
| **Optional depth** | Hugging Face TRL supervised fine-tuning |

## Key Takeaways

- A chat template is a byte-level protocol and must match the serving engine.
- Only assistant response tokens contribute to the supervised objective.
- Packed documents need an attention boundary as well as a loss mask.

## How to work this chapter

```bash
ol start L12.1
ol tests L12.1
ol check L12.1
```

## 1. Why now

C1 gives you a pretrained TinyStories model. Demonstrations teach its response format and task behavior while preserving the serving template already used by L10.5.

## 2. Principles

For token ids `x_0...x_(T-1)`, causal cross entropy is evaluated only where the boolean assistant mask `m_t` is true: `L = -sum_t m_t log p(x_(t+1)|x_<=t) / max(1, sum_t m_t)`. System and user tokens provide context but receive no target loss. A document boundary also prevents attention from one packed example into another.

| Symbol | Meaning |
|---|---|
| `x_t` | token id at position `t` |
| `m_t` | 1 for an assistant target, otherwise 0 |
| `T` | packed sequence length |

## 3. Worked example by hand

For messages `user: hi`, `assistant: hello`, rendering yields the template's user prefix, `hi`, assistant prefix, `hello`, and end marker. If these occupy token positions 0 through 5, the user and role markers have mask 0, while the `hello` target positions have mask 1. The test checks exact rendered ids and the resulting mask.

## 4. The interface

Implement `render_chat`, `assistant_loss_mask`, and `pack` to the stub contract. Packing returns token ids, loss mask, and document ids. `test_hand_rendered_chat` checks exact rendering, `test_assistant_only_mask` checks loss targets, and `test_packed_documents_do_not_attend_across_boundaries` checks packed document ids.

| Test | Why it exists | Expected result |
|---|---|---|
| `test_hand_rendered_chat` | Pins the serving byte protocol | Exact token ids for the worked conversation |
| `test_assistant_only_mask` | Ensures prompts are context, not targets | Only assistant response positions contribute loss |
| `test_packed_documents_do_not_attend_across_boundaries` | Prevents leakage between packed examples | Attention cannot cross document ids |

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Training on user tokens | `test_assistant_only_mask`; mutant `s01` |
| Adding an EOS to every packed sample without a mask | `test_packed_documents_do_not_attend_across_boundaries`; mutant `s02` |
| Replacing each packed document id with the first id | `test_packed_documents_do_not_attend_across_boundaries`; mutant `s02` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L6.6` | Provides the adapted model parameters for the SFT run. |
| Back | `L0.3` | Supplies the token-level cross-entropy objective. |
| Forward | `C2` | Starts from this SFT checkpoint before preference optimization and preserves the serving chat contract. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `pack` | length bucketing and packing in TRL | Higher device utilization, with careful document masks | Hugging Face TRL SFT trainer |
| assistant mask | dataset mixture and loss weighting | Changes which tokens define the training distribution | `content/tracks/ml/07-training-and-post-training` |
