# contracts/py/tinyllm/accounting.pyi (M05.1): counting parameters, FLOPs, KV bytes, memory
# chapter: content/tracks/math/05-discrete-math-1/01-counting-params-flops-kv-bytes.md
#
# Exact integer counts for a Llama-family decoder (L7.9): token embedding,
# n_layers blocks of [RMSNorm, attention, RMSNorm, gated MLP], a final
# RMSNorm, and an lm_head that is either its own [vocab, d_model] matrix or
# the embedding itself (tied). Every result is a Python int, never a float,
# so a count of 134,515,008 is that number and not 1.345e8.
#
# Symbols (one block):
#   d   = d_model            H   = n_heads        Hkv = n_kv_heads
#   dh  = d_head             f   = d_ff           V   = vocab
#   E   = n_experts          k   = top_k          S   = n_shared
#   fe  = d_ff_expert (0 means d_ff)              dv  = v_head_dim (0 means dh)
#   r   = kv_lora_rank       dr  = qk_rope_dim    rq  = q_lora_rank
#
# Attention parameters per layer (biases only with qkv_bias, on q, k, v):
#   mha, gqa   q [H dh, d], k and v [Hkv dh, d] each, o [d, H dh]
#              (mha requires Hkv == H; gqa requires H % Hkv == 0)
#   mla        (DeepSeek-V2/V3, HF DeepseekV3Attention; dh is the no-rope
#              query and key width, the rope part dr is extra)
#              q: rq == 0: q_proj [H (dh + dr), d]
#                 rq > 0:  q_a_proj [rq, d], q_a_layernorm [rq], q_b_proj [H (dh + dr), rq]
#              kv_a_proj_with_mqa [r + dr, d], kv_a_layernorm [r],
#              kv_b_proj [H (dh + dv), r], o_proj [d, H dv]
# MLP parameters per layer (gated, no bias, L7.2): gate, up [f, d], down [d, f]
#   = 3 d f. With E > 0, layers n_dense_layers .. n_layers - 1 are MoE:
#   router [E, d] + E experts of 3 d fe + S shared experts of 3 d fe; the
#   first n_dense_layers layers keep the dense MLP of width f.
# Norm parameters: 2 d per layer + d final (+ rq + r per MLA layer).
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int
    tie_embeddings: bool
    attn: Literal["mha", "gqa", "mla"] = "gqa"
    kv_lora_rank: int = 0
    qk_rope_dim: int = 0
    n_experts: int = 0
    top_k: int = 0
    n_shared: int = 0
    # beyond DESIGN 4.1's row (DEVIATIONS B74-02): what HF's MLA and MoE
    # configs need for an exact count
    q_lora_rank: int = 0
    v_head_dim: int = 0
    d_ff_expert: int = 0
    n_dense_layers: int = 0
    qkv_bias: bool = False

def param_count(cfg: ModelConfig) -> dict[str, int]:
    """Parameters per component and in total, with these keys:
        embed    V d
        attn     attention parameters, all layers
        mlp      MLP parameters, all layers (dense, router, experts, shared)
        norm     every RMSNorm weight, final norm included
        lm_head  V d, or 0 when tie_embeddings (the embedding is reused)
        total    the sum of the five above: what HF's num_parameters() reports
        active   parameters one token touches: total minus the (E - k) experts
                 it is not routed to in every MoE layer (equals total for a
                 dense model)
    ValueError for an inconsistent config: a size below 1 (d_model, n_layers,
    n_heads, n_kv_heads, d_head, d_ff, vocab), mha with Hkv != H, gqa with
    H % Hkv != 0, mla with kv_lora_rank < 1, E > 0 with k outside [1, E],
    k or S > 0 with E == 0, n_dense_layers outside [0, n_layers]."""

def flops_per_token(cfg: ModelConfig, seq_len: int, training: bool) -> int:
    """Floating-point operations per token, the PaLM convention (Chowdhery
    et al. 2022, appendix B): a multiply-add is 2 FLOPs, every parameter a
    token touches in a matrix product costs one multiply-add, and every
    token attends to all seq_len positions (an upper bound; a causal average
    is about half). With
        N    = active matmul parameters = attn + active MLP + V d
               (lm_head is a matrix product even when tied; the embedding
               lookup and the norm weights are not)
        a    = 2 n_layers H seq_len (d_qk + d_v)
               (d_qk = dh, d_v = dh for mha/gqa; d_qk = dh + dr, d_v = dv for mla)
    the forward pass costs 2 N + a and a training step 3 (2 N + a): the
    backward is twice the forward. ValueError for seq_len < 1 or a bad cfg."""

def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int) -> int:
    """Bytes of KV cache one token adds, all layers:
        mha, gqa   n_layers * 2 * Hkv * dh * dtype_bytes   (a key and a value per kv head)
        mla        n_layers * (r + dr) * dtype_bytes        (the latent and the shared rope key)
    ValueError for dtype_bytes < 1 or a bad cfg."""

def memory_plan(
    cfg: ModelConfig, batch: int, seq: int, dtype_bytes: int, optimizer: Literal["sgd", "adamw"]
) -> dict[str, int]:
    """Bytes of one training step without recomputation, P = total params:
        weights      P dtype_bytes
        grads        P dtype_bytes
        master       4 P when dtype_bytes < 4 (a float32 copy the optimizer
                     updates, mixed precision), else 0
        optimizer    4 P per state, float32: sgd keeps 1 (momentum), adamw 2
        activations  batch * seq * (sum over layers of A_l * dtype_bytes + 4 V)
        total        the sum of the five above
    Every block keeps A_l values per token for its backward, and the float32
    logits keep V per token. A_l = attention + MLP saved values of layer l:
        attention  2 d (norm input and output) + q + k + v widths
                   + H seq (one softmax row per head) + H d_v (attention output)
                   q, k, v widths: H dh, Hkv dh, Hkv dh (mha/gqa);
                   H (dh + dr), H (dh + dr), H dv plus r + dr + rq (mla)
        MLP        2 d + 4 f_tok (gate, up, activation, product), where
                   f_tok = f for a dense layer and (k + S) fe + E for an MoE
                   layer (the E router logits); A uses the dense width for
                   the first n_dense_layers layers and the MoE width after
    So bf16 AdamW is 2 + 2 + 4 + 8 = 16 bytes per parameter before
    activations, the same as float32 AdamW (4 + 4 + 0 + 8).
    ValueError for batch < 1, seq < 1, dtype_bytes not in (1, 2, 4, 8), an
    unknown optimizer, or a bad cfg."""
