"""train_step_torch.py -- a minimal, correct next-token training step in PyTorch (README §7).

Math -> code:
    L = -(1/|M|) * sum_{t in M} log p_theta(x_{t+1} | x_{<=t})
    inputs = x[:, :-1], targets = x[:, 1:]  (shift by one), M = positions not masked
    SFT loss masking: prompt positions get label -100 (cross_entropy ignore_index), so only
    the response tokens contribute to the gradient.

Tiny char-level decoder: embedding + learned positions -> one pre-norm causal attention block
(F.scaled_dot_product_attention, is_causal=True) -> SwiGLU MLP -> tied LM head.
Trains with AdamW + grad clipping, BF16 autocast on CUDA when available.

Run:
    uv run --with torch python train_step_torch.py
"""

from __future__ import annotations

import sys

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:
    sys.exit(
        "torch is not installed. Run: uv run --with torch python train_step_torch.py"
    )

TEXT = "the quick brown fox jumps over the lazy dog. " * 40
VOCAB = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(VOCAB)}
SEQ, D, HEADS = 32, 64, 4
IGNORE = -100


class TinyLM(nn.Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, D)
        nn.init.normal_(
            self.tok.weight, std=0.02
        )  # tied head: N(0,1) init gives huge logits
        self.pos = nn.Parameter(torch.zeros(SEQ, D))
        self.norm1, self.norm2, self.norm_f = (
            nn.RMSNorm(D),
            nn.RMSNorm(D),
            nn.RMSNorm(D),
        )
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.gate_up = nn.Linear(D, 2 * 4 * D, bias=False)
        self.down = nn.Linear(4 * D, D, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.tok(idx) + self.pos[:t]
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, HEADS, D // HEADS).unbind(2)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + self.o(att.transpose(1, 2).reshape(b, t, D))
        g, u = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        x = x + self.down(F.silu(g) * u)
        return self.norm_f(x) @ self.tok.weight.T  # tied unembedding


def batch(bs: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor([STOI[c] for c in TEXT])
    starts = torch.randint(0, len(data) - SEQ - 1, (bs,), generator=gen)
    x = torch.stack([data[s : s + SEQ + 1] for s in starts])
    return x[:, :-1], x[:, 1:].clone()  # inputs, next-token targets


def sft_labels(prompt_len: int, targets: torch.Tensor) -> torch.Tensor:
    """Mask prompt positions so only response tokens are trained on."""
    labels = targets.clone()
    labels[:, : prompt_len - 1] = IGNORE  # targets are shifted by one
    return labels


def train_step(model, opt, inputs, labels, device: str) -> float:
    model.train()
    with torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
    ):
        logits = model(inputs)
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE,
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def main() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyLM(len(VOCAB)).to(device)
    # model = torch.compile(model)  # opt-in compiler; same step, fused kernels
    opt = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95)
    )
    print(
        f"torch {torch.__version__} on {device}, {sum(p.numel() for p in model.parameters()):,} params"
    )

    losses = []
    for step in range(300):
        x, y = batch(32, gen)
        losses.append(train_step(model, opt, x.to(device), y.to(device), device))
        if step % 50 == 0:
            print(f"step {step:3d}  loss {losses[-1]:.4f}")
    print(
        f"final loss {losses[-1]:.4f} (uniform baseline ln V = {torch.log(torch.tensor(len(VOCAB))):.4f})"
    )
    assert losses[-1] < 0.25 * losses[0], (
        "next-token loss should fall sharply on repetitive text"
    )

    # SFT-style masking: the first 16 positions are "prompt"; their loss must not count
    x, y = batch(4, gen)
    labels = sft_labels(16, y)
    assert (labels[:, :15] == IGNORE).all() and (labels[:, 15:] != IGNORE).all()
    print(
        f"masked SFT step loss {train_step(model, opt, x.to(device), labels.to(device), device):.4f}"
    )
    print("OK")


if __name__ == "__main__":
    main()
