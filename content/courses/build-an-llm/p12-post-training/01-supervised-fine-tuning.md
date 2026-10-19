# Supervised fine-tuning: chat rendering, assistant-only loss, packing

The [TinyStories capstone](../capstones/01-tinystories.md) gives you a pretrained model that continues text well but has never seen a conversation. Supervised fine-tuning (SFT) on demonstrations teaches it a response format and task behavior. The work is less about the optimizer, which is the same causal cross-entropy you already have, than about three pieces of plumbing that are easy to get subtly wrong: rendering each conversation into exactly the bytes the serving engine will send, deciding which tokens the model is trained to predict, and packing many short examples into one row without letting them see each other.

This section builds those three functions. It uses the cross-entropy of [Losses with a fused backward](../p00-foundations/03-losses-with-fused-backward.md), typically trains the LoRA adapters of [LoRA: low-rank adaptation with PiSSA initialization and merging](../p06-objectives/06-lora-pissa-and-merge.md), and must agree with the chat template served by [tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper](../p10-serving/05-openai-server-on-tokio.md). The solution is [`sft.py`](../../../../course/solutions/python/tinyllm/post/sft.py) (`render_chat`, `assistant_loss_mask`, `pack`), and the contract is [`sft.pyi`](../../../../course/contracts/py/tinyllm/post/sft.pyi).

> **Learning objectives**
> - Render a conversation with the serving tokenizer's canonical chat template.
> - Build an assistant-only loss mask aligned with the rendered ids, and explain why prompts are context, not targets.
> - Pack several examples into one row with both a loss mask and document ids, and turn the ids into an attention mask.

## The objective

For token ids $x_0, \dots, x_{T-1}$ and a boolean assistant mask $m_t$, causal cross-entropy is evaluated only where $m_t$ is true:

$$L = -\frac{\sum_t m_t \log p(x_{t+1} \mid x_{\le t})}{\max\big(1, \sum_t m_t\big)}.$$

Here $x_t$ is the token id at position $t$, $m_t$ is 1 for an assistant target and 0 otherwise, and $T$ is the (packed) sequence length. System and user tokens provide context but receive no target loss. When several documents share a row, a document boundary must also prevent attention from one packed example into another; the mask alone does not do that.

## Working it by hand

**Rendering and the mask.** For the messages `user: hi`, `assistant: ok`, rendering yields the template's user prefix, `hi`, the assistant prefix, `ok`, and an end marker. Take a toy tokenizer that renders this conversation as the ids `[1, 2, 3, 4]` and reports the assistant span as positions 2 to 4 (half open). The user prefix and `hi` (positions 0 and 1) are context, so the mask is `[False, False, True, True]`: two target positions, so the loss divides by 2.

**Packing.** Now pack two examples into `seq_len = 4`: `([11, 12], [False, True])` and `([21], [True])`. They fill three slots, and the fourth is padding:

| position | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| ids | 11 | 12 | 21 | 0 |
| loss mask | F | T | T | F |
| document id | 0 | 0 | 1 | -1 |

The attention mask allows position $j$ to attend to position $i \le j$ only when both have the same document id, so token 21 cannot see 11 or 12, and the padding (id -1) is neither a target nor context.

## Designing the API

The contract fixes three functions. `render_chat(messages, tokenizer)` returns the serving template's token ids. `assistant_loss_mask(messages, tokenizer)` returns `(ids, mask)` with one boolean per id. `pack(examples, seq_len)` takes `(tokens, mask)` pairs and returns three length-`seq_len` arrays: token ids (`int64`), loss mask (`bool`), and document ids (`int64`, -1 for padding).

## Building it step by step

Four principles shape the code.

1. **Training and serving must see the same bytes.** If the model is fine-tuned on one rendering and served with another (a missing newline, a different role marker), it is asked to continue sequences it never saw. So the solution never builds its own template: it delegates to the tokenizer's canonical one.
2. **The mask must be computed on the same rendering.** A mask built from a separate rendering could be shifted by one token and silently train on role markers. So `assistant_loss_mask` renders once and checks that the mask has exactly as many entries as the ids.
3. **Packing must carry both boundaries.** The loss mask says which positions are targets; the document id says which positions may attend to which. An EOS token alone does neither: the next document would still attend across it, and the EOS itself would be a target unless masked.
4. **Padding must be inert.** Padded positions get mask `False` and document id -1, so they are never targets and never share a document with a real token.

### Rendering

Rendering prefers the Hugging Face API, so a real serving tokenizer works unchanged, and falls back to the book's tokenizer's own `render_chat`:

```python
def render_chat(messages, tokenizer):
    """Render messages with the serving tokenizer's canonical template."""
    if hasattr(tokenizer, "apply_chat_template"):
        # add_generation_prompt=False: a training example already contains the assistant turn
        return list(tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=False))
    return list(tokenizer.render_chat(list(messages)))
```

### The assistant mask

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

### Packing

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

`pack` fills one row and stops at the first example that does not fit; the caller packs the remaining examples into the next row. The trainer turns `docs` into a block-diagonal causal attention mask (`docs[i] == docs[j] and i <= j`) and uses `loss` to select the cross-entropy terms. The full solution is [`sft.py`](../../../../course/solutions/python/tinyllm/post/sft.py).

### Why this design

All three functions are linear in the number of tokens. Packing removes most padding, so a batch of short demonstrations costs close to its real token count instead of `rows * longest`.

Delegating the template is better than writing one because a hand-written template is one more copy of the protocol that can drift from the serving engine; delegation keeps exactly one source of truth. Document ids are better than EOS separators because separators keep the loss honest only if masked, and never stop attention, while document ids give the attention mask directly. And stopping a row at the first example that does not fit is better than splitting it: splitting would make the example's second half attend to nothing of its first half, a target without its context. Stopping wastes a few padding slots instead.

## What can go wrong

Three checks pin a correct solution. The hand-rendered chat must produce the exact token ids and mask of the worked conversation, which pins the serving byte protocol. The mask must be assistant-only, so that only assistant response positions contribute loss and prompts stay context. And packed documents must stay apart: each token carries its own document id, and padding is -1.

The common bugs are the mirror images of those checks. Training on user tokens happens when the mask defaults to `True` or is built from a different rendering; here the mask starts all `False` and only assistant spans are set. Adding an EOS to every packed sample without a mask neither stops attention nor keeps the EOS out of the loss; here separation is by document id, and the mask comes with each example. Replacing each packed document id with the first id merges every example into one document; here each example's tokens get their own `doc_id` from `enumerate`.

> **Warning:** A chat template that differs from the serving engine's by a single byte still trains a model that looks fine in offline evaluation and then misbehaves when served. Always render with the tokenizer the server uses.

## Where this leads

The SFT checkpoint is the starting point for the rest of this chapter: [Direct preference optimization and IPO](02-direct-preference-optimization.md) uses it as the reference policy, and the [post-trained capstone](../capstones/02-post-trained.md) runs SFT, then preference optimization, RL, and distillation on the TinyStories model while preserving the serving chat contract.

In production, Hugging Face TRL's SFT trainer adds length bucketing and packing for higher device utilization, with careful document masks. Dataset mixtures and loss weighting change which tokens define the training distribution; see the [Training and Post-Training](../../machine-learning/07-training-and-post-training/README.md) chapter of *Machine Learning and AI*.

### Further reading

- Hugging Face TRL, the supervised fine-tuning trainer documentation.

## Key terms

- **chat template**: the byte-level protocol that renders a list of role-tagged messages into one token sequence.
- **assistant-only loss**: cross-entropy computed only at positions whose target is an assistant response token.
- **packing**: concatenating several short examples into one fixed-length row to reduce padding.
- **document id**: a per-token label naming which packed example a token belongs to (-1 for padding), used to build a block-diagonal attention mask.

## Summary

- A chat template is a byte-level protocol and must match the serving engine, so rendering delegates to the serving tokenizer.
- Only assistant response tokens contribute to the supervised objective, and the loss is their mean.
- Packed documents need an attention boundary (document ids) as well as a loss mask; padding is neither target nor context.
- Packing stops a row at the first example that does not fit rather than splitting it.

## Review questions

1. Why does the loss divide by $\max(1, \sum_t m_t)$ rather than by $T$?

   <details><summary>Answer</summary>

   Dividing by $T$ would make the loss depend on how much of the row is prompt or padding, so a long prompt would quietly shrink the learning rate for its answer. Dividing by the number of targets makes it a mean over assistant tokens; the $\max(1, \cdot)$ keeps an all-context row from dividing by zero.

   </details>

2. Why is a loss mask not enough for a packed row?

   <details><summary>Answer</summary>

   The mask decides which positions are targets, not what they may read. Without document ids, the first token of example 1 would attend to the whole of example 0 and learn to depend on an unrelated conversation.

   </details>

3. What does the length check in `assistant_loss_mask` protect against?

   <details><summary>Answer</summary>

   A mask that is one token shorter or longer than the ids would shift every target by that offset, training on role markers and skipping the last answer token. Raising is better than training on the wrong tokens silently.

   </details>
