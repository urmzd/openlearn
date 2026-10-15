# contracts/py/tinyllm/modern/llama.pyi (L7.9): the Llama-family decoder, HF configs and weights
# chapter: content/courses/build-an-llm/p07-modern-block/09-llama-family-model.md
#
# One decoder that loads Llama, Mistral (sliding window), and Qwen2 (q/k/v
# biases) checkpoints from Hugging Face, and the course's own variants
# (MLA attention, mixture of experts, sink tokens, learned sinks) through
# tl_* keys of config.json (formats/config.schema.json):
#
#   logits = lm_head(norm(h_L)),   h_0 = embed_tokens[ids]
#   h_l    = h' + mlp(post_attention_layernorm(h'))          (L7.1 Pre-LN)
#   h'     = h_{l-1} + self_attn(input_layernorm(h_{l-1}))
#
#   self_attn  L7.5 GQAttention (L7.6 MLAttention when attention == "mla"),
#              RoPE from L7.3 with frequencies from L7.4 (rope_scaling),
#              the window and sink-token mask from L7.7 (attention_mask)
#   mlp        L7.2 GatedMLP, or L7.8 MoE on layers >= first_k_dense_replace
#              when num_experts > 0
#   norms      L7.1 RMSNorm(hidden_size, rms_norm_eps)
#   lm_head    a Linear without bias, or the embedding itself when
#              tie_word_embeddings (logits = h @ embed_tokens.weight^T)
#
# state_dict keys are Hugging Face's: model.embed_tokens.weight,
# model.layers.<i>.self_attn.*, model.layers.<i>.mlp.*,
# model.layers.<i>.input_layernorm.weight,
# model.layers.<i>.post_attention_layernorm.weight, model.norm.weight,
# lm_head.weight (absent when tied), in HF's registration order.
#
# config.json keys read by from_hf (HF name, then the tl_* override):
#   vocab_size hidden_size intermediate_size num_hidden_layers
#   num_attention_heads num_key_value_heads (default: heads)
#   head_dim (default hidden_size / heads) rms_norm_eps (1e-6)
#   tie_word_embeddings (false) max_position_embeddings (2048)
#   rope_theta and rope_scaling (transformers 4.x) or rope_parameters (5.x:
#     rope_theta inside; rope_type default, linear, yarn, llama3, and tl's
#     ntk; others are a ValueError); partial_rotary_factor (1.0);
#     rope_interleaved / rope_interleave (false: HF's half layout)
#   hidden_act: silu, or gelu_pytorch_tanh (GeGLU)
#   attention_bias: q, k, v biases (Qwen2 always; tl_qkv_bias; a Llama
#     config with attention_bias true also has an o_proj bias, which L7.5
#     does not model: ValueError); mlp_bias
#   sliding_window (Mistral; Qwen2 only with use_sliding_window),
#     tl_sliding_window; tl_sink_tokens (0); tl_learned_sinks (false)
#   tl_attention "gqa" (also "mha", "mqa") or "mla" with tl_mla_rank (or
#     kv_lora_rank), q_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
#     v_head_dim (head_dim, when absent)
#   tl_num_experts (or num_local_experts, n_routed_experts, num_experts),
#     tl_top_k_experts (or num_experts_per_tok), n_shared_experts,
#     moe_intermediate_size (default intermediate_size),
#     first_k_dense_replace, norm_topk_prob (true), tl_router
#     ("softmax_topk"), routed_scaling_factor (1.0), router_aux_loss_coef
#     (0.01)
#   tl_tokenizer ("file")
from dataclasses import dataclass
from typing import Any, Optional

from numpy.typing import ArrayLike

from tinyllm.accounting import ModelConfig
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.norm import RMSNorm
from tinyllm.modern.rope import RopeSpec
from tinyllm.nn.layers import Embedding, Linear, ModuleList
from tinyllm.nn.module import Module

@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None  # {"rope_type": ..., "factor": ..., ...}; None = default
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 2048
    qkv_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    rope_interleaved: bool = False
    partial_rotary_factor: float = 1.0
    attention: str = "gqa"  # "gqa" or "mla"
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    sliding_window: Optional[int] = None
    sink_tokens: int = 0
    learned_sinks: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    n_shared_experts: int = 0
    moe_intermediate_size: int = 0
    first_k_dense_replace: int = 0
    norm_topk_prob: bool = True
    router: str = "softmax_topk"
    routed_scaling_factor: float = 1.0
    router_aux_loss_coef: float = 0.01
    tokenizer: str = "file"

    @classmethod
    def from_hf(cls, config_json: str) -> "LlamaConfig":
        """Read a config.json (a file path, or a model directory holding
        one) by the table above. ValueError for a missing size, an
        unsupported rope_type or model_type (llama, mistral, qwen2, or any
        config with tl_arch "llama"), or inconsistent sizes."""

    def to_hf(self) -> dict:
        """The config.json dict save_pretrained writes: HF LlamaConfig keys
        (rope_theta, rope_scaling as in transformers 4.x), architectures
        ["LlamaForCausalLM"], model_type "llama", tl_arch "llama",
        tl_tokenizer, and the tl_* keys of the extensions in use.
        from_hf(to_hf(c)) == c."""

    def model_config(self) -> ModelConfig:
        """The same architecture for M05.1's counters (attn "mha" when
        Hkv == H, "gqa" otherwise, "mla"; d_head = qk_nope_head_dim for
        mla)."""

    def rope_spec(self) -> RopeSpec:
        """RopeSpec of every attention layer: L7.4's rope_inv_freq_scaled
        for the rope_scaling kind over rotary_dim = head_dim *
        partial_rotary_factor (qk_rope_head_dim for mla), base rope_theta;
        layout "interleaved" when rope_interleaved, else "half"."""

class LlamaDecoderLayer(Module):
    self_attn: Module  # GQAttention or MLAttention
    mlp: Module  # GatedMLP or MoE
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm

    def __init__(self, cfg: LlamaConfig, layer_idx: int, rng: Any = None) -> None: ...
    def forward(self, h: Tensor, positions: Any, cache: Any = None, layer: int = 0) -> Tensor:
        """The two Pre-LN residual steps above; the attention mask is L7.7's
        attention_mask(cache, layer, T, sliding_window, sink_tokens)."""

class LlamaModel(Module):
    """The decoder stack without the head; LlamaForCausalLM runs it."""

    embed_tokens: Embedding
    layers: ModuleList
    norm: RMSNorm

    def __init__(self, cfg: LlamaConfig, rng: Any = None, init: bool = True) -> None: ...

class LlamaForCausalLM(Module):
    config: LlamaConfig
    model: LlamaModel
    lm_head: Optional[Linear]  # None when tie_word_embeddings

    def __init__(self, cfg: LlamaConfig, rng: Any = None, init: bool = True) -> None:
        """init=True: random parameters from rng (PCG32, M06.3; None means
        PCG32(0).substream("init")), drawn in registration order through
        L0.4's initializers. init=False: the parameters are allocated
        without random draws (their values are unspecified); from_pretrained
        uses it and then loads every parameter."""

    def forward(self, ids: ArrayLike, positions: Optional[ArrayLike] = None, cache: Any = None) -> Tensor:
        """ids int [B, T] (or [T]) -> float32 logits [B, T, vocab] (or
        [T, vocab]). positions default to s .. s + T - 1, where
        s = cache.seq_len(0) for a cache that has seq_len (0 without a
        cache). The cache is handed to every layer (layer = its index): a
        KV cache hook for gqa (L7.5), a latent cache for mla (L7.6).
        ValueError for an id outside [0, vocab) or ids that are not 1-D or
        2-D integers."""

    def aux_loss(self) -> Optional[Tensor]:
        """The sum of the MoE layers' aux_loss from the last forward (None
        for a dense model)."""

    def param_count(self) -> int:
        """Parameters, each counted once (a tied embedding once): HF's
        num_parameters(), and M05.1's param_count(config.model_config())
        total for every config M05.1 counts."""

    @classmethod
    def from_pretrained(cls, dir: str) -> "LlamaForCausalLM":
        """config.json plus model.safetensors, or the shards named by
        model.safetensors.index.json, read with L0.6's load_safetensors
        (BF16 and F16 decoded to float32). Keys ending in rotary_emb.inv_freq
        are ignored, and lm_head.weight when tied. KeyError (strict
        load_state_dict) for a missing or unexpected key; ValueError for a
        shape mismatch."""

    def save_pretrained(self, dir: str, dtype: str = "F32") -> None:
        """Write dir/config.json (config.to_hf()) and dir/model.safetensors
        (state_dict, metadata {"format": "pt"}) with every tensor stored as
        dtype ("F32", "BF16", or "F16"). from_pretrained(dir) gives back the
        same parameters (rounded to dtype)."""
