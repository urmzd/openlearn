# Supervised fine-tuning: chat rendering, assistant-only loss, packing

## Overview

| | |
|---|---|
| **Module** | `L12.1` · optional build · Python · Pass 10 |
| **Solution** | [`sft.py`](../../../../course/solutions/python/tinyllm/post/sft.py): `render_chat`, `assistant_loss_mask`, `pack` |
| **Contract** | [`sft.pyi`](../../../../course/contracts/py/tinyllm/post/sft.pyi) |
| **Needs** | `L6.6` LoRA, `L0.3` cross-entropy, `L10.5` chat template |
| **Used by** | `C2` post-trained capstone (SFT, then preference optimization, RL, and distillation on the C1 model) |
| **Optional depth** | Hugging Face TRL supervised fine-tuning |

## Key Takeaways

- A chat template is a byte-level protocol and must match the serving engine.
- Only assistant response tokens contribute to the supervised objective.
- Packed documents need an attention boundary as well as a loss mask.

## How to read this chapter

Sections 1 to 3 state the objective and work a conversation and a packed batch by hand. Section 4 fixes the interface. Section 5 walks through `sft.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

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

**Rendering and the mask.** For messages `user: hi`, `assistant: ok`, rendering yields the template's user prefix, `hi`, assistant prefix, `ok`, and end marker. Take a toy tokenizer that renders this conversation as the ids `[1, 2, 3, 4]` and reports the assistant span as positions 2 to 4 (half open). The user prefix and `hi` (positions 0 and 1) are context, so the mask is `[False, False, True, True]`: two target positions, so the loss divides by 2.

**Packing.** Pack two examples into `seq_len = 4`: `([11, 12], [False, True])` and `([21], [True])`. They fill three slots; the fourth is padding:

| position | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| ids | 11 | 12 | 21 | 0 |
| loss mask | F | T | T | F |
| document id | 0 | 0 | 1 | -1 |

The attention mask allows position $j$ to attend to position $i \le j$ only when both have the same document id, so token 21 cannot see 11 or 12, and the padding (id -1) is neither a target nor a context.

## 4. The interface

The contract fixes three functions. `render_chat(messages, tokenizer)` returns the serving template's token ids. `assistant_loss_mask(messages, tokenizer)` returns `(ids, mask)` with one boolean per id. `pack(examples, seq_len)` takes `(tokens, mask)` pairs and returns three length-`seq_len` arrays: token ids (`int64`), loss mask (`bool`), and document ids (`int64`, -1 for padding).

### What a correct solution must handle

| Case | Why it exists | Expected result |
|---|---|---|
| the hand-rendered chat | Pins the serving byte protocol | Exact token ids and mask for the worked conversation |
| assistant-only mask | Ensures prompts are context, not targets | Only assistant response positions contribute loss |
| packed documents stay apart | Prevents leakage between packed examples | Each token carries its own document id; padding is -1 |

## 5. The solution, step by step

### From first principles

1. **Training and serving must see the same bytes.** If the model is fine-tuned on one rendering and served with another (a missing newline, a different role marker), it is asked to continue sequences it never saw. So the solution never builds its own template: it delegates to the tokenizer's canonical one.
2. **The mask must be computed on the same rendering.** A mask built from a separate rendering could be shifted by one token and silently train on role markers. So `assistant_loss_mask` renders once and checks that the mask has exactly as many entries as the ids.
3. **Packing must carry both boundaries.** The loss mask says which positions are targets; the document id says which positions may attend to which. An EOS token alone does neither: the next document would still attend across it, and the EOS itself would be a target unless masked.
4. **Padding must be inert.** Padded positions get mask `False` and document id -1, so they are never targets and never share a document with a real token.

### The code, annotated

Rendering prefers the Hugging Face API, so a real serving tokenizer works unchanged, and falls back to the course tokenizer's own `render_chat`:

```python
def render_chat(messages, tokenizer):
    """Render messages with the serving tokenizer's canonical template."""
    if hasattr(tokenizer, "apply_chat_template"):
        # add_generation_prompt=False: a training example already contains the assistant turn
        return list(tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=False))
    return list(tokenizer.render_chat(list(messages)))
```

The mask comes either straight from the tokenizer or from its assistant spans, and is validated against the rendered ids:

```python
def assistant_loss_mask(messages, tokenizer):
    ids = render_chat(messages, tokenizer)
    if hasattr(tokenizer, "assistant_mask"):
        mask = list(tokenizer.assistant_mask(list(messages)))
    else:
        mask = [False] * len(ids)                 # default: context, not target
        spans = tokenizer.assistant_spans(list(messages))
        for start, end in spans:                  # half-open spans of assistant tokens
            if start < 0 or end < start or end > len(ids):
                raise ValueError("assistant span outside rendered token sequence")
            mask[start:end] = [True] * (end - start)
    if len(mask) != len(ids):                     # a misaligned mask trains on the wrong tokens
        raise ValueError("token and loss mask lengths differ")
    return ids, [bool(x) for x in mask]
```

Packing appends whole examples in order and records each token's document:

```python
def pack(examples, seq_len):
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    ids, loss, docs = [], [], []
    for doc_id, (tokens, mask) in enumerate(examples):
        if len(tokens) != len(mask):
            raise ValueError("token and mask lengths differ")
        if len(tokens) > seq_len:                 # an over-long example is truncated to one row
            tokens, mask = tokens[:seq_len], mask[:seq_len]
        if len(ids) + len(tokens) > seq_len:      # never split an example across rows: stop here
            break
        ids.extend(tokens); loss.extend(mask); docs.extend([doc_id] * len(tokens))
    pad = seq_len - len(ids)
    # padding: token 0, never a target, document -1 (shared with no real token)
    ids.extend([0] * pad); loss.extend([False] * pad); docs.extend([-1] * pad)
    return np.asarray(ids, dtype=np.int64), np.asarray(loss, dtype=bool), np.asarray(docs, dtype=np.int64)
```

`pack` fills one row and stops at the first example that does not fit; the caller packs the remaining examples into the next row. The trainer turns `docs` into a block-diagonal causal attention mask (`docs[i] == docs[j] and i <= j`) and uses `loss` to select the cross-entropy terms.

Full solution: [`sft.py`](../../../../course/solutions/python/tinyllm/post/sft.py)

### Why this solution

- **Cost.** All three functions are linear in the number of tokens. Packing removes most padding, so a batch of short demonstrations costs close to its real token count instead of `rows * longest`.
- **Delegating the template versus writing one.** A hand-written template is one more copy of the protocol that can drift from the serving engine. Delegation keeps exactly one source of truth.
- **Document ids versus EOS separators.** Separators keep the loss honest only if masked, and never stop attention. Document ids give the attention mask directly.
- **Stopping versus splitting.** Splitting an example across rows would make its second half attend to nothing of its first half, a target without its context. Stopping wastes a few padding slots instead.

### Explain it back

**Why does the loss divide by `max(1, sum_t m_t)` rather than by `T`?**

<details><summary>Answer</summary>

Dividing by `T` would make the loss depend on how much of the row is prompt or padding, so a long prompt would quietly shrink the learning rate for its answer. Dividing by the number of targets makes it a mean over assistant tokens; the `max(1, ...)` keeps an all-context row from dividing by zero.

</details>

**Why is a loss mask not enough for a packed row?**

<details><summary>Answer</summary>

The mask decides which positions are targets, not what they may read. Without document ids, the first token of example 1 would attend to the whole of example 0 and learn to depend on an unrelated conversation.

</details>

**What does the length check in `assistant_loss_mask` protect against?**

<details><summary>Answer</summary>

A mask that is one token shorter or longer than the ids would shift every target by that offset, training on role markers and skipping the last answer token. Raising is better than training on the wrong tokens silently.

</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Training on user tokens | the mask starts all `False` and only assistant spans are set |
| Adding an EOS to every packed sample without a mask | separation is by document id, and the mask comes with each example |
| Replacing each packed document id with the first id | each example's tokens get their own `doc_id` from `enumerate` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L6.6` | Provides the adapted model parameters for the SFT run. |
| Back | `L0.3` | Supplies the token-level cross-entropy objective. |
| Forward | `C2` | Starts from this SFT checkpoint before preference optimization and preserves the serving chat contract. |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `pack` | length bucketing and packing in TRL | Higher device utilization, with careful document masks | Hugging Face TRL SFT trainer |
| assistant mask | dataset mixture and loss weighting | Changes which tokens define the training distribution | `content/courses/machine-learning/07-training-and-post-training` |
