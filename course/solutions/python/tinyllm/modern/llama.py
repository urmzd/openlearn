"""The Llama-family decoder, Hugging Face configs, and weight loading (L7.9).

Everything Part 7 built meets here: RMSNorm and the Pre-LN step, gated MLPs
or experts, RoPE with scaled frequencies, GQA or latent attention, sliding
windows and sinks. The checkpoint contract is Hugging Face's: the same
config.json keys and the same tensor names, so SmolLM2-135M loads as it is
published and the course's own models load in transformers.

Contract: contracts/py/tinyllm/modern/llama.pyi.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.accounting import ModelConfig
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.modern.ctxext import rope_inv_freq_scaled
from tinyllm.modern.gqa import GQAttention
from tinyllm.modern.mla import MLAttention
from tinyllm.modern.mlp import GatedMLP
from tinyllm.modern.moe import MoE
from tinyllm.modern.norm import RMSNorm, pre_norm_residual
from tinyllm.modern.rope import RopeSpec
from tinyllm.modern.window import attention_mask
from tinyllm.nn.layers import Embedding, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32

ROPE_KINDS = ("default", "linear", "ntk", "yarn", "llama3")
ACTS = {"silu": "silu", "gelu_pytorch_tanh": "gelu_tanh", "gelu_tanh": "gelu_tanh"}


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
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 2048
    qkv_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    rope_interleaved: bool = False
    partial_rotary_factor: float = 1.0
    attention: str = "gqa"
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
        p = Path(config_json)
        c = json.loads((p / "config.json" if p.is_dir() else p).read_text())
        mt = c.get("model_type", "llama")
        if c.get("tl_arch", "llama") != "llama" or (mt not in ("llama", "mistral", "qwen2") and "tl_arch" not in c):
            raise ValueError(f"not a Llama-family config: model_type {mt!r}, tl_arch {c.get('tl_arch')!r}")
        need = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads")
        missing = [k for k in need if k not in c]
        if missing:
            raise ValueError(f"config.json lacks {missing}")
        H = int(c["num_attention_heads"])
        # transformers 5.x nests theta and scaling in rope_parameters; 4.x has
        # rope_theta and rope_scaling at the top.
        rp = dict(c.get("rope_parameters") or {})
        theta = float(rp.pop("rope_theta", c.get("rope_theta", 10000.0)))
        scaling = rp or c.get("rope_scaling")
        if scaling:
            scaling = dict(scaling)
            kind = scaling.pop("rope_type", scaling.pop("type", "default"))
            if kind not in ROPE_KINDS:
                raise ValueError(f"rope_type {kind!r} is not supported (one of {ROPE_KINDS})")
            scaling = None if kind == "default" else {"rope_type": kind, **scaling}
        bias = bool(c.get("attention_bias", False))
        if mt == "llama" and bias:
            raise ValueError("attention_bias on a Llama config also biases o_proj, which GQAttention has not")
        window = c.get("tl_sliding_window", c.get("sliding_window") if mt != "qwen2" or c.get("use_sliding_window") else None)
        attn = c.get("tl_attention", "gqa")
        if attn not in ("gqa", "mha", "mqa", "mla"):
            raise ValueError(f"tl_attention must be gqa, mha, mqa, or mla, got {attn!r}")
        n_exp = int(c.get("tl_num_experts", c.get("num_local_experts", c.get("n_routed_experts", c.get("num_experts", 0)))) or 0)
        act = c.get("hidden_act", "silu")
        if act not in ACTS:
            raise ValueError(f"hidden_act {act!r} is not supported")
        head_dim = int(c.get("head_dim") or c["hidden_size"] // H)
        cfg = cls(
            vocab_size=int(c["vocab_size"]),
            hidden_size=int(c["hidden_size"]),
            intermediate_size=int(c["intermediate_size"]),
            num_hidden_layers=int(c["num_hidden_layers"]),
            num_attention_heads=H,
            num_key_value_heads=int(c.get("num_key_value_heads") or H),
            head_dim=head_dim,
            rms_norm_eps=float(c.get("rms_norm_eps", 1e-6)),
            rope_theta=theta,
            rope_scaling=scaling or None,
            tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
            max_position_embeddings=int(c.get("max_position_embeddings", 2048)),
            qkv_bias=bias or mt == "qwen2" or bool(c.get("tl_qkv_bias", False)),
            mlp_bias=bool(c.get("mlp_bias", False)),
            hidden_act=ACTS[act],
            rope_interleaved=bool(c.get("rope_interleaved", c.get("rope_interleave", False))),
            partial_rotary_factor=float(c.get("partial_rotary_factor", 1.0)),
            attention="mla" if attn == "mla" else "gqa",
            q_lora_rank=c.get("q_lora_rank"),
            kv_lora_rank=int(c.get("tl_mla_rank", c.get("kv_lora_rank", 0)) or 0),
            qk_nope_head_dim=int(c.get("qk_nope_head_dim", 0) or 0),
            qk_rope_head_dim=int(c.get("qk_rope_head_dim", 0) or 0),
            v_head_dim=int(c.get("v_head_dim", 0) or (head_dim if attn == "mla" else 0)),
            sliding_window=None if window is None else int(window),
            sink_tokens=int(c.get("tl_sink_tokens", 0)),
            learned_sinks=bool(c.get("tl_learned_sinks", False)),
            num_experts=n_exp,
            num_experts_per_tok=int(c.get("tl_top_k_experts", c.get("num_experts_per_tok", 0)) or 0) if n_exp else 0,
            n_shared_experts=int(c.get("n_shared_experts", 0) or 0) if n_exp else 0,
            moe_intermediate_size=int(c.get("moe_intermediate_size") or c["intermediate_size"]) if n_exp else 0,
            first_k_dense_replace=int(c.get("first_k_dense_replace", 0) or 0) if n_exp else 0,
            norm_topk_prob=bool(c.get("norm_topk_prob", True)),
            router=str(c.get("tl_router", "softmax_topk")),
            routed_scaling_factor=float(c.get("routed_scaling_factor", 1.0)),
            router_aux_loss_coef=float(c.get("router_aux_loss_coef", 0.01)),
            tokenizer=str(c.get("tl_tokenizer", "file")),
        )
        cfg.model_config()  # M05.1 rejects inconsistent sizes
        return cfg

    def to_hf(self) -> dict:
        c: dict[str, Any] = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "tl_arch": "llama",
            "tl_tokenizer": self.tokenizer,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "tie_word_embeddings": self.tie_word_embeddings,
            "max_position_embeddings": self.max_position_embeddings,
            "attention_bias": False,
            "mlp_bias": self.mlp_bias,
            "hidden_act": {"silu": "silu", "gelu_tanh": "gelu_pytorch_tanh"}[self.hidden_act],
            "rope_interleaved": self.rope_interleaved,
            "partial_rotary_factor": self.partial_rotary_factor,
            "tl_attention": self.attention,
        }
        if self.qkv_bias:
            c["tl_qkv_bias"] = True
        if self.attention == "mla":
            c.update(tl_mla_rank=self.kv_lora_rank, q_lora_rank=self.q_lora_rank, qk_nope_head_dim=self.qk_nope_head_dim,
                     qk_rope_head_dim=self.qk_rope_head_dim, v_head_dim=self.v_head_dim)
        if self.sliding_window is not None:
            c["tl_sliding_window"] = self.sliding_window
        if self.sink_tokens:
            c["tl_sink_tokens"] = self.sink_tokens
        if self.learned_sinks:
            c["tl_learned_sinks"] = True
        if self.num_experts:
            c.update(tl_num_experts=self.num_experts, tl_top_k_experts=self.num_experts_per_tok,
                     n_shared_experts=self.n_shared_experts, moe_intermediate_size=self.moe_intermediate_size,
                     first_k_dense_replace=self.first_k_dense_replace, norm_topk_prob=self.norm_topk_prob,
                     tl_router=self.router, routed_scaling_factor=self.routed_scaling_factor,
                     router_aux_loss_coef=self.router_aux_loss_coef)
        return c

    def model_config(self) -> ModelConfig:
        mla = self.attention == "mla"
        H, Hkv = self.num_attention_heads, self.num_key_value_heads
        return ModelConfig(
            vocab=self.vocab_size, d_model=self.hidden_size, n_layers=self.num_hidden_layers, n_heads=H,
            n_kv_heads=H if mla else Hkv, d_head=self.qk_nope_head_dim if mla else self.head_dim,
            d_ff=self.intermediate_size, tie_embeddings=self.tie_word_embeddings,
            attn="mla" if mla else ("mha" if Hkv == H else "gqa"),
            kv_lora_rank=self.kv_lora_rank if mla else 0, qk_rope_dim=self.qk_rope_head_dim if mla else 0,
            n_experts=self.num_experts, top_k=self.num_experts_per_tok, n_shared=self.n_shared_experts,
            q_lora_rank=(self.q_lora_rank or 0) if mla else 0, v_head_dim=self.v_head_dim if mla else 0,
            d_ff_expert=self.moe_intermediate_size if self.num_experts else 0,
            n_dense_layers=self.first_k_dense_replace if self.num_experts else 0, qkv_bias=self.qkv_bias,
        )

    def rope_spec(self) -> RopeSpec:
        if self.attention == "mla":
            r = self.qk_rope_head_dim
        else:
            r = int(self.head_dim * self.partial_rotary_factor)
        s = dict(self.rope_scaling or {})
        kind = s.get("rope_type", "default")
        inv, scale = rope_inv_freq_scaled(
            r, self.rope_theta, kind, factor=float(s.get("factor", 1.0)),
            original_max_pos=int(s.get("original_max_position_embeddings") or self.max_position_embeddings),
            beta_fast=float(s.get("beta_fast", 32)), beta_slow=float(s.get("beta_slow", 1)),
            low_freq_factor=float(s.get("low_freq_factor", 1)), high_freq_factor=float(s.get("high_freq_factor", 4)),
        )
        if s.get("attention_factor") is not None:
            scale = float(s["attention_factor"])
        return RopeSpec(inv_freq=inv, attention_scaling=scale,
                        layout="interleaved" if self.rope_interleaved else "half", rotary_dim=r)


class _FlatRNG:
    """A PCG32 stand-in that draws nothing random: init=False builds the
    parameter shapes fast, and from_pretrained overwrites every value."""

    def next_u32(self) -> int:
        return 1 << 31

    def uniform(self) -> float:
        return 0.5

    def uniforms(self, n: int) -> NDArray:
        return np.full(n, 0.5)

    def below(self, n: int) -> int:
        return 0

    def shuffle(self, xs: Any) -> None:
        return None

    def substream(self, purpose: str) -> "_FlatRNG":
        return self

    def state(self) -> tuple[int, int]:
        return (0, 0)

    def set_state(self, s: tuple[int, int]) -> None:
        return None


class LlamaDecoderLayer(Module):
    def __init__(self, cfg: LlamaConfig, layer_idx: int, rng: Any = None) -> None:
        super().__init__()
        r = rng if rng is not None else PCG32(0).substream("init")
        d, H = cfg.hidden_size, cfg.num_attention_heads
        spec = cfg.rope_spec()
        self.window, self.n_sink = cfg.sliding_window, cfg.sink_tokens
        # Registration order = HF LlamaDecoderLayer: self_attn, mlp, then the norms.
        if cfg.attention == "mla":
            self.self_attn = MLAttention(d, H, cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_nope_head_dim,
                                         cfg.qk_rope_head_dim, cfg.v_head_dim, spec, rng=r)
        else:
            # The window lives in the mask (L7.7), so sink tokens can be exempt from it.
            self.self_attn = GQAttention(d, H, cfg.num_key_value_heads, cfg.head_dim, spec, qkv_bias=cfg.qkv_bias,
                                         window=None, sinks=cfg.learned_sinks, rng=r)
        if cfg.num_experts and layer_idx >= cfg.first_k_dense_replace:
            self.mlp = MoE(d, cfg.moe_intermediate_size, cfg.num_experts, cfg.num_experts_per_tok,
                           n_shared=cfg.n_shared_experts, router=cfg.router, norm_topk=cfg.norm_topk_prob,
                           aux_loss_coef=cfg.router_aux_loss_coef, routed_scaling=cfg.routed_scaling_factor,
                           act=cfg.hidden_act, rng=r)
        else:
            self.mlp = GatedMLP(d, cfg.intermediate_size, act=cfg.hidden_act, bias=cfg.mlp_bias, rng=r)
        self.input_layernorm = RMSNorm(d, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(d, cfg.rms_norm_eps)

    def forward(self, h: Tensor, positions: Any, cache: Any = None, layer: int = 0) -> Tensor:
        mask = attention_mask(cache, layer, h.shape[1], self.window, self.n_sink)
        h = pre_norm_residual(h, self.input_layernorm, lambda u: self.self_attn(u, positions, mask, cache, layer))
        return pre_norm_residual(h, self.post_attention_layernorm, self.mlp)


class LlamaModel(Module):
    def __init__(self, cfg: LlamaConfig, rng: Any = None, init: bool = True) -> None:
        super().__init__()
        d, V = cfg.hidden_size, cfg.vocab_size
        if init:
            r = rng if rng is not None else PCG32(0).substream("init")
            self.embed_tokens = Embedding(V, d, rng=r)
            self.layers = ModuleList([LlamaDecoderLayer(cfg, i, r) for i in range(cfg.num_hidden_layers)])
        else:
            # Shapes without draws: a one-row table widened in place, and one
            # layer of each kind copied (every layer of a kind has the same shapes).
            flat = _FlatRNG()
            self.embed_tokens = Embedding(1, d, rng=flat)
            self.embed_tokens.weight = Tensor(np.zeros((V, d), np.float32), requires_grad=True)
            protos: dict[bool, LlamaDecoderLayer] = {}
            layers = []
            for i in range(cfg.num_hidden_layers):
                kind = bool(cfg.num_experts) and i >= cfg.first_k_dense_replace
                if kind not in protos:
                    protos[kind] = LlamaDecoderLayer(cfg, i, flat)
                    layers.append(protos[kind])
                else:
                    layers.append(copy.deepcopy(protos[kind]))
            self.layers = ModuleList(layers)
        self.norm = RMSNorm(d, cfg.rms_norm_eps)


def _weights(d: Path) -> dict[str, NDArray]:
    """Every tensor of a checkpoint directory, sharded or not."""
    index = d / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    out: dict[str, NDArray] = {}
    for f in files:
        tensors, _ = load_safetensors(str(d / f))
        out.update(tensors)
    return out


class LlamaForCausalLM(Module):
    def __init__(self, cfg: LlamaConfig, rng: Any = None, init: bool = True) -> None:
        super().__init__()
        r = rng if rng is not None else PCG32(0).substream("init")
        self.config = cfg
        self.model = LlamaModel(cfg, r, init)
        if cfg.tie_word_embeddings:
            self.lm_head = None  # logits reuse model.embed_tokens.weight: one tensor, one name
        elif init:
            self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size, bias=False, rng=r)
        else:
            self.lm_head = Linear(cfg.hidden_size, 1, bias=False, rng=_FlatRNG())
            self.lm_head.weight = Tensor(np.zeros((cfg.vocab_size, cfg.hidden_size), np.float32), requires_grad=True)

    def forward(self, ids: ArrayLike, positions: Optional[ArrayLike] = None, cache: Any = None) -> Tensor:
        a = np.asarray(ids)
        if a.ndim not in (1, 2) or a.dtype.kind not in "iu":
            raise ValueError(f"ids must be integer [B, T] or [T], got {a.dtype} {a.shape}")
        single = a.ndim == 1
        a = a[None] if single else a
        T = a.shape[1]
        if positions is None:
            s = int(cache.seq_len(0)) if cache is not None and hasattr(cache, "seq_len") else 0
            positions = np.arange(s, s + T)
        h = self.model.embed_tokens(a)  # ValueError for an id outside [0, vocab)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, positions, cache, i)
        h = self.model.norm(h)
        if self.lm_head is None:
            logits = F.matmul(h, F.transpose(self.model.embed_tokens.weight, 0, 1))
        else:
            logits = self.lm_head(h)
        return logits[0] if single else logits

    def aux_loss(self) -> Optional[Tensor]:
        total = None
        for layer in self.model.layers:
            aux = getattr(layer.mlp, "aux_loss", None)
            if aux is not None:
                total = aux if total is None else total + aux
        return total

    def param_count(self) -> int:
        return int(sum(p.data.size for p in self.parameters()))

    @classmethod
    def from_pretrained(cls, dir: str) -> "LlamaForCausalLM":
        d = Path(dir)
        cfg = LlamaConfig.from_hf(str(d))
        model = cls(cfg, init=False)
        sd = {k: v for k, v in _weights(d).items() if not k.endswith("rotary_emb.inv_freq")}
        if cfg.tie_word_embeddings:
            sd.pop("lm_head.weight", None)  # some tied checkpoints store the copy too
        model.load_state_dict(sd, strict=True)
        return model

    def save_pretrained(self, dir: str, dtype: str = "F32") -> None:
        if dtype not in ("F32", "BF16", "F16"):
            raise ValueError(f"dtype must be F32, BF16, or F16, got {dtype!r}")
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        cfg = self.config.to_hf()
        cfg["torch_dtype"] = {"F32": "float32", "BF16": "bfloat16", "F16": "float16"}[dtype]
        (d / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        sd = {k: np.asarray(v, dtype=np.float32) for k, v in self.state_dict().items()}
        save_safetensors(str(d / "model.safetensors"), sd, {"format": "pt"}, dtypes={k: dtype for k in sd})

