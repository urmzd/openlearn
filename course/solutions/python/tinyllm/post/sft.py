"""Small SFT data transforms. See chapter L12.1."""
from __future__ import annotations

import numpy as np


def render_chat(messages, tokenizer):
    """Render messages with the serving tokenizer's canonical template."""
    if hasattr(tokenizer, "apply_chat_template"):
        return list(tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=False))
    return list(tokenizer.render_chat(list(messages)))


def assistant_loss_mask(messages, tokenizer):
    ids = render_chat(messages, tokenizer)
    if hasattr(tokenizer, "assistant_mask"):
        mask = list(tokenizer.assistant_mask(list(messages)))
    else:
        mask = [False] * len(ids)
        spans = tokenizer.assistant_spans(list(messages))
        for start, end in spans:
            if start < 0 or end < start or end > len(ids):
                raise ValueError("assistant span outside rendered token sequence")
            mask[start:end] = [True] * (end - start)
    if len(mask) != len(ids):
        raise ValueError("token and loss mask lengths differ")
    return ids, [bool(x) for x in mask]


def pack(examples, seq_len):
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    ids, loss, docs = [], [], []
    for doc_id, (tokens, mask) in enumerate(examples):
        if len(tokens) != len(mask):
            raise ValueError("token and mask lengths differ")
        if len(tokens) > seq_len:
            tokens, mask = tokens[:seq_len], mask[:seq_len]
        if len(ids) + len(tokens) > seq_len:
            break
        ids.extend(tokens); loss.extend(mask); docs.extend([doc_id] * len(tokens))
    pad = seq_len - len(ids)
    ids.extend([0] * pad); loss.extend([False] * pad); docs.extend([-1] * pad)
    return np.asarray(ids, dtype=np.int64), np.asarray(loss, dtype=bool), np.asarray(docs, dtype=np.int64)
