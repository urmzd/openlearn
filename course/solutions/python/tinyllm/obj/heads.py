"""Fine-tuning heads and the linear-head export (L6.5).

A pretrained encoder or decoder turns tokens into vectors; a head turns those
vectors into the answer a task wants. A sequence classifier pools one vector
per sequence (BERT's [CLS], the mean of the real tokens, or a decoder's last
token) and applies one Linear. A token classifier applies the Linear at every
position. A reward head maps the last token to one scalar and is trained so
the preferred answer scores higher (Bradley-Terry). The gateway's usage
policy (D33) is the smallest head of all: a logistic regression over an
embedding endpoint's vectors, fitted by IRLS (M07.7) and exported as JSON
that Go evaluates with one dot product.

Contract: contracts/py/tinyllm/obj/heads.pyi.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.losses import bce_with_logits, cross_entropy
from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.nn.layers import Dropout, Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.obj.bert import BertConfig, BertEncoder
from tinyllm.obj.gpt import GPT, GPTConfig
from tinyllm.obj.lora import LoRALinear, inject_lora
from tinyllm.optim.adamw import AdamW
from tinyllm.prob.metrics import ece, logistic_regression_fit, roc_auc

POOLS = ("cls", "mean", "last")
ARCHS = ("bert", "electra", "gpt")


def _rng(rng: Any) -> Any:
    return rng if rng is not None else PCG32(0).substream("init")


def _real(ids: NDArray, attn_mask: Optional[ArrayLike]) -> NDArray:
    if attn_mask is None:
        return np.ones(ids.shape, dtype=bool)
    m = np.asarray(attn_mask).astype(bool)
    if m.shape != ids.shape:
        raise ValueError(f"attn_mask {m.shape} != ids {ids.shape}")
    if not m.any(axis=1).all():
        raise ValueError("every row needs at least one real token")
    return m


def hidden_states(
    backbone: Module, ids: ArrayLike, token_type_ids: Optional[ArrayLike] = None, attn_mask: Optional[ArrayLike] = None
) -> Tensor:
    if hasattr(backbone, "hidden"):  # a decoder (L6.1's GPT): causal, no padding mask needed
        return backbone.hidden(ids)
    return backbone(ids, token_type_ids, attn_mask)  # an encoder (L6.2's BertEncoder)


def pool(h: Tensor, attn_mask: Optional[ArrayLike], how: str) -> Tensor:
    if how not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {how!r}")
    B, T = h.shape[0], h.shape[1]
    real = _real(np.zeros((B, T), dtype=np.int64), attn_mask)
    if how == "cls":
        return h[:, 0]
    if how == "mean":
        w = real.astype(h.data.dtype)[:, :, None]
        return F.sum(h * w, axis=1) / w.sum(axis=1)
    # last: the last REAL token of each row (right padding), not position T - 1.
    last = T - 1 - np.argmax(real[:, ::-1], axis=1)
    return h[np.arange(B), last]


class SequenceClassifier(Module):
    def __init__(
        self,
        backbone: Module,
        d_model: int,
        n_classes: int,
        pool: Literal["cls", "mean", "last"] = "cls",
        dropout: float = 0.0,
        rng: Any = None,
    ) -> None:
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        if n_classes < 2 or d_model < 1:
            raise ValueError(f"need n_classes >= 2 and d_model >= 1, got {n_classes}, {d_model}")
        r = _rng(rng)
        self.pool, self.d_model, self.n_classes = pool, int(d_model), int(n_classes)
        self.backbone = backbone
        self.dropout = Dropout(dropout, rng=r)
        self.classifier = Linear(d_model, n_classes, rng=r)

    def forward(
        self,
        ids: ArrayLike,
        token_type_ids: Optional[ArrayLike] = None,
        attn_mask: Optional[ArrayLike] = None,
        labels: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        h = hidden_states(self.backbone, ids, token_type_ids, attn_mask)
        logits = self.classifier(self.dropout(pool(h, attn_mask, self.pool)))
        if labels is None:
            return logits, None
        return logits, cross_entropy(logits, np.asarray(labels).astype(np.int64))


class TokenClassifier(Module):
    def __init__(self, backbone: Module, d_model: int, n_labels: int, dropout: float = 0.0, rng: Any = None) -> None:
        super().__init__()
        if n_labels < 2 or d_model < 1:
            raise ValueError(f"need n_labels >= 2 and d_model >= 1, got {n_labels}, {d_model}")
        r = _rng(rng)
        self.d_model, self.n_labels = int(d_model), int(n_labels)
        self.backbone = backbone
        self.dropout = Dropout(dropout, rng=r)
        self.classifier = Linear(d_model, n_labels, rng=r)

    def forward(
        self,
        ids: ArrayLike,
        token_type_ids: Optional[ArrayLike] = None,
        attn_mask: Optional[ArrayLike] = None,
        labels: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = np.asarray(ids)
        real = _real(x, attn_mask)
        logits = self.classifier(self.dropout(hidden_states(self.backbone, x, token_type_ids, attn_mask)))
        if labels is None:
            return logits, None
        lab = np.asarray(labels).astype(np.int64)
        # Padding is never a target, whatever label it carries.
        return logits, cross_entropy(logits, np.where(real, lab, -100), ignore_index=-100)


class RewardHead(Module):
    def __init__(self, backbone: Module, d_model: int, rng: Any = None) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.backbone = backbone
        self.score = Linear(d_model, 1, bias=False, rng=_rng(rng))

    def forward(self, ids: ArrayLike, attn_mask: Optional[ArrayLike] = None) -> Tensor:
        h = hidden_states(self.backbone, ids, None, attn_mask)
        r = self.score(pool(h, attn_mask, "last"))  # [B, 1]
        return F.reshape(r, (r.shape[0],))


def pairwise_reward_loss(r_chosen: Tensor, r_rejected: Tensor) -> Tensor:
    if r_chosen.shape != r_rejected.shape or r_chosen.ndim != 1:
        raise ValueError(f"rewards must be [B] and [B], got {r_chosen.shape}, {r_rejected.shape}")
    # -log sigmoid(r_c - r_r) = BCE with target 1 on the margin.
    return bce_with_logits(r_chosen - r_rejected, np.ones(r_chosen.shape, dtype=np.float32))


def _attn_value(name: str, m: Module) -> bool:
    return name.startswith("backbone.") and name.endswith((".q_proj", ".v_proj"))


def lora_classifier(
    clf: Module,
    r: int,
    alpha: float,
    target: Optional[Callable[[str, Module], bool]] = None,
    init: Literal["default", "pissa"] = "default",
    rng: Any = None,
) -> list[str]:
    names = inject_lora(clf, target or _attn_value, r, alpha, init=init, rng=rng)
    # The head is new: it trains in full (PEFT's modules_to_save).
    head = clf.classifier if hasattr(clf, "classifier") else clf.score
    for p in head.parameters():
        p.requires_grad = True
    return names


def train_classifier(
    clf: Module, ids: ArrayLike, labels: ArrayLike, steps: int, batch_size: int, lr: float, rng: Any,
    attn_mask: Optional[ArrayLike] = None, weight_decay: float = 0.0,
) -> list[float]:
    x, y = np.asarray(ids), np.asarray(labels)
    real = _real(x, attn_mask)
    if steps < 1 or batch_size < 1 or len(x) != len(y):
        raise ValueError(f"need steps, batch_size >= 1 and one label per row, got {steps}, {batch_size}")
    params = [p for p in clf.parameters() if p.requires_grad]
    opt = AdamW(params, lr=lr, weight_decay=weight_decay)
    clf.train()
    losses = []
    for _ in range(steps):
        idx = np.array([rng.below(len(x)) for _ in range(batch_size)])  # with replacement
        opt.zero_grad()
        _, loss = clf(x[idx], None, real[idx], y[idx])
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    clf.eval()
    return losses


def predict(
    clf: Module, ids: ArrayLike, attn_mask: Optional[ArrayLike] = None, batch_size: int = 64
) -> NDArray:
    x = np.asarray(ids)
    real = _real(x, attn_mask)
    clf.eval()
    out = []
    with no_grad():
        for i in range(0, len(x), batch_size):
            logits, _ = clf(x[i : i + batch_size], None, real[i : i + batch_size])
            out.append(np.argmax(logits.data, axis=-1))  # ties to the lowest class
    return np.concatenate(out).astype(np.int64)


def save_classifier(clf: SequenceClassifier, dir: str, arch: str, tokenizer: str = "file", labels: Sequence[str] = ()) -> None:
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    if any(isinstance(m, LoRALinear) for _, m in clf.named_modules()):
        raise ValueError("the classifier still holds LoRA adapters: merge_lora (L6.6) before saving")
    names = list(labels) or [str(i) for i in range(clf.n_classes)]
    if len(names) != clf.n_classes:
        raise ValueError(f"{len(names)} label names for {clf.n_classes} classes")
    out = Path(dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "tl_arch": arch,
        "tl_tokenizer": tokenizer,
        "tl_format": 1,
        "tl_backbone": asdict(clf.backbone.cfg),
        "tl_head": {"kind": "sequence", "n_classes": clf.n_classes, "pool": clf.pool, "d_model": clf.d_model, "labels": names},
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    save_safetensors(str(out / "model.safetensors"), clf.state_dict(), {"format": "tinyllm", "tl_arch": arch})


def load_classifier(dir: str) -> tuple[SequenceClassifier, dict[str, Any]]:
    d = Path(dir)
    cfg = json.loads((d / "config.json").read_text())
    arch, head = cfg.get("tl_arch"), cfg.get("tl_head")
    if arch not in ARCHS or not head or head.get("kind") != "sequence":
        raise ValueError(f"{d}/config.json is not a sequence classifier (tl_arch {arch!r}, tl_head {head!r})")
    b = dict(cfg["tl_backbone"])
    b["dropout"] = 0.0
    backbone = GPT(GPTConfig(**b)) if arch == "gpt" else BertEncoder(BertConfig(**b))
    clf = SequenceClassifier(backbone, head["d_model"], head["n_classes"], head["pool"])
    tensors, _ = load_safetensors(str(d / "model.safetensors"))
    clf.load_state_dict(tensors)
    clf.eval()
    return clf, cfg


# --- the linear policy head (D33) --------------------------------------------------------


def _unit_rows(E: ArrayLike) -> NDArray:
    X = np.asarray(E, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError(f"embeddings must be a finite 2-D array, got shape {X.shape}")
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X), where=n > 0)  # a zero vector stays zero


def fit_linear_head(
    embeddings: ArrayLike,
    labels: ArrayLike,
    classes: Sequence[str],
    l2: float,
    iters: int = 50,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if len(classes) != 2:
        raise ValueError(f"the policy head is binary (D33): need 2 classes, got {list(classes)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    X = _unit_rows(embeddings)  # fit on what the gateway will score: unit vectors
    y = np.asarray(labels)
    w = logistic_regression_fit(X, y, l2, iters)  # M07.7 IRLS; intercept last
    d = X.shape[1]
    # Two softmax rows: class 0 is the reference (all zeros), so
    # p(class 1) = softmax([0, w.e + c])[1] = sigmoid(w.e + c), exactly the fit.
    W = np.zeros((2, d))
    W[1] = w[:d]
    b = np.array([0.0, w[d]])
    return {"dim": d, "classes": list(classes), "W": W, "b": b, "threshold": float(threshold), "l2": float(l2)}


def head_probs(head: Mapping[str, Any], embeddings: ArrayLike) -> NDArray:
    X = _unit_rows(embeddings)
    W, b = np.asarray(head["W"], dtype=np.float64), np.asarray(head["b"], dtype=np.float64)
    if X.shape[1] != W.shape[1]:
        raise ValueError(f"embeddings have dim {X.shape[1]}, the head {W.shape[1]}")
    z = X @ W.T + b
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def head_metrics(head: Mapping[str, Any], embeddings: ArrayLike, labels: ArrayLike) -> dict[str, float]:
    y = np.asarray(labels).astype(np.int64)
    p1 = head_probs(head, embeddings)[:, -1]
    flag = p1 >= head["threshold"]
    tp, fp, fn = int(np.sum(flag & (y == 1))), int(np.sum(flag & (y == 0))), int(np.sum(~flag & (y == 1)))
    return {
        "accuracy": float(np.mean(flag == (y == 1))),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "auc": roc_auc(p1, y),  # M07.7
        "ece": ece(p1, y),  # M07.7
    }


def export_linear_head(
    head: Mapping[str, Any], embedding_model: str, path: str, metrics: Optional[Mapping[str, float]] = None
) -> None:
    W = np.asarray(head["W"], dtype=np.float64)
    b = np.asarray(head["b"], dtype=np.float64)
    classes = list(head["classes"])
    if not embedding_model:
        raise ValueError("embedding_model must name the model whose /v1/embeddings the head was fitted on")
    if W.shape != (len(classes), int(head["dim"])) or b.shape != (len(classes),) or len(classes) < 2:
        raise ValueError(f"W {W.shape} and b {b.shape} do not match {len(classes)} classes of dim {head['dim']}")
    if not (np.isfinite(W).all() and np.isfinite(b).all()):
        raise ValueError("the head has a non-finite weight")
    doc: dict[str, Any] = {
        "embedding_model": embedding_model,
        "dim": int(head["dim"]),
        "classes": classes,
        # float() of a float64 prints the shortest repr that round-trips: Go's
        # strconv.ParseFloat reads back the exact same double.
        "W": [[float(v) for v in row] for row in W],
        "b": [float(v) for v in b],
        "threshold": float(head["threshold"]),
    }
    if metrics:
        doc["metrics"] = {k: float(v) for k, v in metrics.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")


def load_linear_head(path: str) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    head = dict(doc)
    head["W"] = np.asarray(doc["W"], dtype=np.float64)
    head["b"] = np.asarray(doc["b"], dtype=np.float64)
    if head["W"].shape != (len(doc["classes"]), doc["dim"]):
        raise ValueError(f"{path}: W is {head['W'].shape}, expected {(len(doc['classes']), doc['dim'])}")
    return head
