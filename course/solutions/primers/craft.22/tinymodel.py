"""The model under test in craft.22.

A two-layer Llama-style decoder trained on synthetic stories
(weights in course/solutions/fixtures/craft.22/tinylm.npz; the training
script is not included), in numpy float32:

    ids     = encode(text)                    character BPE, 96 merges by rank
    x       = embed[ids]
    per layer l:
      h     = x + attn_l(rmsnorm(x; g_attn))  4 heads, RoPE theta 10000 (half layout), causal
      x     = h + mlp_l(rmsnorm(h; g_mlp))    SwiGLU: down(silu(gate h) * up h)
    logits  = rmsnorm(x; g_final) @ embed^T   tied embeddings

A good eval suite passes on this file, flags copies with one real defect
each (a wrong RoPE base, a layer whose output is dropped, a tokenizer one
merge short, ...), and does NOT flag harmless variants (float64
arithmetic, another equally good sampling stream).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

DTYPE = np.float32


def fixture(name: str) -> Path:
    """A craft.22 fixture under TINYLLM_FIXTURES (defaults to course/solutions/fixtures)."""
    return Path(os.environ.get("TINYLLM_FIXTURES", "course/solutions/fixtures")) / "craft.22" / name


class TinyLM:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.meta = json.loads(bytes(arrays["__meta__"]).decode())
        self.w = {k: v.astype(DTYPE) for k, v in arrays.items() if k not in ("__meta__", "chars", "merges")}
        self.chars = [chr(int(c)) for c in arrays["chars"]]
        self.cid = {c: i for i, c in enumerate(self.chars)}
        self.merges = [tuple(int(x) for x in m) for m in arrays["merges"]]
        self.rank = {m: r for r, m in enumerate(self.merges)}
        self.vocab = [*self.chars]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.layers = int(self.meta["layers"])
        self.heads = int(self.meta["heads"])
        self.theta = float(self.meta["theta"])
        self.eps = float(self.meta["eps"])
        self.ctx = int(self.meta["ctx"])

    # -- tokenizer ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Characters to ids, then merge the lowest-ranked adjacent pair until none is left."""
        ids = [self.cid[c] for c in text if c in self.cid]
        while len(ids) > 1:
            r = min((self.rank.get(p, len(self.merges)) for p in zip(ids, ids[1:])))
            if r >= len(self.merges):
                break
            a, b = self.merges[r]
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    out.append(len(self.chars) + r)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    # -- forward -----------------------------------------------------------

    def _rmsnorm(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps) * g

    def _rope(self, x: np.ndarray) -> np.ndarray:
        T, hd = x.shape[-2], x.shape[-1]
        inv = 1.0 / self.theta ** (np.arange(0, hd, 2, dtype=DTYPE) / hd)
        ang = np.arange(T, dtype=DTYPE)[:, None] * inv[None, :]
        cos = np.concatenate([np.cos(ang)] * 2, -1).astype(DTYPE)
        sin = np.concatenate([np.sin(ang)] * 2, -1).astype(DTYPE)
        x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
        return x * cos + np.concatenate([-x2, x1], -1) * sin

    def _softmax(self, s: np.ndarray) -> np.ndarray:
        e = np.exp(s - s.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    def logits(self, ids: list[int]) -> np.ndarray:
        """Next-token logits at every position: float32 [len(ids), vocab]."""
        # numpy 2.2 with Apple Accelerate raises spurious floating-point
        # flags inside matmul; the values are correct, so the flags are muted.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return self._logits(ids)

    def _logits(self, ids: list[int]) -> np.ndarray:
        w = self.w
        T = len(ids)
        D = w["embed"].shape[1]
        hd = D // self.heads
        x = w["embed"][np.asarray(ids)]
        mask = np.triu(np.full((T, T), -np.inf, dtype=DTYPE), 1)
        for layer in range(self.layers):
            p = f"l{layer}."
            h = self._rmsnorm(x, w[p + "g_attn"])
            q = (h @ w[p + "wq"].T).reshape(T, self.heads, hd).transpose(1, 0, 2)
            k = (h @ w[p + "wk"].T).reshape(T, self.heads, hd).transpose(1, 0, 2)
            v = (h @ w[p + "wv"].T).reshape(T, self.heads, hd).transpose(1, 0, 2)
            q, k = self._rope(q), self._rope(k)
            att = self._softmax(q @ k.transpose(0, 2, 1) / np.sqrt(DTYPE(hd)) + mask)
            o = (att @ v).transpose(1, 0, 2).reshape(T, D)
            x = x + o @ w[p + "wo"].T
            h = self._rmsnorm(x, w[p + "g_mlp"])
            gate = h @ w[p + "w_gate"].T
            x = x + ((gate / (1 + np.exp(-gate))) * (h @ w[p + "w_up"].T)) @ w[p + "w_down"].T
        return self._rmsnorm(x, w["g_final"]) @ w["embed"].T

    def log_probs(self, ids: list[int]) -> np.ndarray:
        """log p(next token) at every position: float [len(ids), vocab]."""
        z = self.logits(ids).astype(np.float64)
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    # -- decoding ----------------------------------------------------------

    def sample(self, prompt: str, tokens: int, rng: np.random.Generator, temperature: float = 0.8) -> str:
        """Continue `prompt` by `tokens` sampled tokens: softmax(logits / T) of
        the last position, one rng.random() per token, inverse CDF. Returns
        the continuation only."""
        ids = self.encode(prompt)
        n = len(ids)
        for _ in range(tokens):
            z = self.logits(ids[-self.ctx :])[-1].astype(np.float64) / temperature
            p = np.exp(z - z.max())
            c = np.cumsum(p)
            ids.append(int(min(np.searchsorted(c, rng.random() * c[-1], side="right"), len(c) - 1)))
        return self.decode(ids[n:])


def load(path: str | os.PathLike | None = None) -> TinyLM:
    """The course model (TINYLLM_FIXTURES/craft.22/tinylm.npz by default)."""
    with np.load(path or fixture("tinylm.npz"), allow_pickle=False) as z:
        return TinyLM({k: z[k] for k in z.files})
