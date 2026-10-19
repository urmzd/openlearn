"""The model zoo (L6.7): every family the course trains, loaded through
its checkpoint directory and scored on its own task, in one report.

One dispatch on config.json's tl_arch (the n-gram's file says it in its
safetensors metadata) finds the family's loader; one scorer per kind of
model turns it into a row: bits per byte for language models, exact match
with beam search for sequence-to-sequence models, accuracy for classifiers
and for the BERT and ELECTRA pretraining objectives, Spearman for word
vectors. Every row carries a 95% confidence interval. A row that cannot be
scored says why (status "error" or "skipped") instead of disappearing, so
one broken family never hides the others.

Contract: contracts/py/tinyllm/eval/zoo.pyi.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.mode import no_grad
from tinyllm.eval.lm import token_nlls
from tinyllm.eval.seqmetrics import exact_match
from tinyllm.infer.beam import beam_search
from tinyllm.io.safetensors import load_safetensors, save_safetensors
from tinyllm.io.tokens import open_tokens
from tinyllm.lm.bigram import BigramLM
from tinyllm.lm.ngram import NGramLM
from tinyllm.lm.nplm import load_nplm
from tinyllm.lm.word2vec import word_similarity
from tinyllm.obj.bert import load_bert, mlm_mask
from tinyllm.obj.electra import load_electra, rtd_accuracy
from tinyllm.obj.gpt import load_gpt
from tinyllm.obj.heads import load_classifier, predict
from tinyllm.prob.stats import mean_ci, normal_ppf, wilson_interval
from tinyllm.rnn.rnnlm import load_rnnlm
from tinyllm.seq2seq.model import load_seq2seq
from tinyllm.xfmr.transformer import load_transformer, translate

FORMAT = "tl.eval-results.v1"
# tl_arch -> (kind, metric, higher_is_better)
FAMILIES: dict[str, tuple[str, str, bool]] = {
    "bigram": ("lm", "bpb", False),
    "ngram": ("lm", "bpb", False),
    "nplm": ("lm", "bpb", False),
    "rnnlm": ("lm", "bpb", False),
    "gpt": ("lm", "bpb", False),
    "seq2seq": ("seq2seq", "em", True),
    "transformer": ("seq2seq", "em", True),
    "bert": ("mlm", "accuracy", True),
    "electra": ("rtd", "accuracy", True),
    "word2vec": ("embeddings", "spearman", True),
}
# tl_arch values of formats/eval-results.schema.json (the n-gram is not one yet).
SCHEMA_ARCHS = {"bigram", "nplm", "word2vec", "rnnlm", "seq2seq", "transformer", "gpt", "bert", "electra", "llama", "elmo", "t5"}
Z95 = 1.959963984540054


def read_config(dir: str) -> dict[str, Any]:
    d = Path(dir)
    if (d / "config.json").is_file():
        return json.loads((d / "config.json").read_text())
    # NGramLM.save writes one safetensors file whose metadata names the arch.
    for f in (d, d / "model.safetensors"):
        if f.is_file() and f.suffix == ".safetensors":
            _, meta = load_safetensors(str(f))
            if meta.get("tl_arch"):
                return dict(meta)
    raise ValueError(f"{d}: no config.json and no safetensors file naming its tl_arch")


def kind_of(cfg: dict[str, Any]) -> str:
    if cfg.get("tl_head"):
        return "classifier"  # a fine-tuned head over any backbone (L6.5)
    arch = cfg.get("tl_arch")
    if arch not in FAMILIES:
        raise KeyError(f"no zoo scorer for tl_arch {arch!r}")
    return FAMILIES[arch][0]


def save_word2vec(emb: ArrayLike, vocab: Sequence[str], dir: str) -> None:
    E = np.asarray(emb, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] != len(vocab):
        raise ValueError(f"embeddings {E.shape} for {len(vocab)} words")
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"tl_arch": "word2vec", "tl_tokenizer": "file", "tl_format": 1, "vocab_size": len(vocab), "hidden_size": int(E.shape[1])}) + "\n"
    )
    (d / "vocab.json").write_text(json.dumps(list(vocab)) + "\n")
    save_safetensors(str(d / "model.safetensors"), {"embeddings": E}, {"format": "tinyllm", "tl_arch": "word2vec"})


def load_model(dir: str) -> Any:
    d = Path(dir)
    cfg = read_config(dir)
    kind, arch = kind_of(cfg), cfg.get("tl_arch")
    if kind == "classifier":
        return load_classifier(dir)[0]  # L6.5
    if arch == "bigram":
        tensors, _ = load_safetensors(str(d / "model.safetensors"))
        return BigramLM(weight=tensors["bigram.weight"])
    if arch == "ngram":
        return NGramLM.load(str(d if d.is_file() else d / "model.safetensors"))
    loaders: dict[str, Callable[[str], Any]] = {
        "nplm": load_nplm,
        "rnnlm": load_rnnlm,
        "gpt": load_gpt,
        "seq2seq": load_seq2seq,
        "transformer": load_transformer,
        "bert": load_bert,
        "electra": lambda p: load_electra(p)[0],
    }
    if arch == "word2vec":
        tensors, _ = load_safetensors(str(d / "model.safetensors"))
        return tensors["embeddings"], json.loads((d / "vocab.json").read_text())
    return loaders[arch](dir)


def _params(model: Any) -> Optional[int]:
    if hasattr(model, "named_parameters"):
        return int(sum(p.data.size for _, p in model.named_parameters()))
    return None


def lm_token_nlls(model: Any, arch: str, cfg: dict[str, Any], ids: ArrayLike, entry: dict[str, Any]) -> NDArray:
    x = np.asarray(ids, dtype=np.int64)
    if arch == "bigram":
        W = np.asarray(model.weight, dtype=np.float64)
        z = W - W.max(axis=1, keepdims=True)
        lsm = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -lsm[x[:-1], x[1:]]
    if arch == "ngram":
        return np.asarray(model.nll(x.tolist()), dtype=np.float64)
    if arch in ("nplm", "rnnlm"):
        return np.asarray(model.nll(x), dtype=np.float64)
    if arch == "gpt":
        ctx = int(entry.get("ctx_len", model.cfg.n_ctx))
        return token_nlls(model, x, ctx, int(entry.get("stride", max(1, ctx // 2))))
    raise KeyError(f"no language-model scorer for tl_arch {arch!r}")


def _text_rows(entry: dict[str, Any], base: Path) -> tuple[list[str], NDArray]:
    """Sentences and labels of a split of a `split<TAB>sentence<TAB>label` file."""
    rows = [line.split("\t") for line in (base / entry["data"]).read_text().splitlines()[1:] if line.strip()]
    split = entry.get("split", "val")
    keep = [r for r in rows if r[0] == split]
    if not keep:
        raise ValueError(f"{entry['data']}: no rows in split {split!r}")
    return [r[1] for r in keep], np.array([int(r[2]) for r in keep], dtype=np.int64)


def encode_batch(texts: Sequence[str], max_len: int, cls_id: int = 1, sep_id: int = 2, pad_id: int = 0) -> tuple[NDArray, NDArray]:
    if max_len < 3:
        raise ValueError(f"max_len must be >= 3, got {max_len}")
    rows = [[cls_id] + list(t.encode("utf-8"))[: max_len - 2] + [sep_id] for t in texts]
    T = max(len(r) for r in rows)
    ids = np.full((len(rows), T), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), T), dtype=bool)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
        mask[i, : len(r)] = True
    return ids, mask


def _wilson_row(k: int, n: int) -> tuple[float, list[float]]:
    lo, hi = wilson_interval(k, n)  # M07.4
    return k / n, [float(lo), float(hi)]


def score_entry(entry: dict[str, Any], base: str, rng: Any) -> dict[str, Any]:
    with no_grad():  # scoring builds no graph
        return _score(entry, base, rng)


def _score(entry: dict[str, Any], base: str, rng: Any) -> dict[str, Any]:
    b = Path(base)
    path = str(b / entry["dir"])
    cfg = read_config(path)
    arch, kind = cfg.get("tl_arch"), kind_of(cfg)
    model = load_model(path)
    row: dict[str, Any] = {"model": entry["id"], "checkpoint": entry["dir"], "task": entry["task"]}
    if arch in SCHEMA_ARCHS:
        row["tl_arch"] = arch
    if kind == "lm":
        ids = np.asarray(open_tokens(str(b / entry["data"])), dtype=np.int64)
        nll = lm_token_nlls(model, arch, cfg, ids, entry)
        # A bare n-gram file has no tl_tokenizer: 256 ids means byte ids.
        default = "bytes" if str(cfg.get("vocab_size")) == "256" else "file"
        if entry.get("tokenizer", cfg.get("tl_tokenizer", default)) == "bytes":
            n_bytes = nll.size  # one scored token is one byte
        elif "n_bytes" in entry:
            n_bytes = int(entry["n_bytes"])
        else:
            raise ValueError("bpb needs the byte count: tl_tokenizer is not bytes and the entry gives no n_bytes")
        _, lo, hi = mean_ci(nll)  # M07.4: Student t over per-token NLLs
        scale = nll.size / (n_bytes * math.log(2.0))
        row.update(metric="bpb", value=float(nll.sum() / (n_bytes * math.log(2.0))), ci95=[lo * scale, hi * scale], n=int(nll.size))
    elif kind == "seq2seq":
        items = [json.loads(x) for x in (b / entry["data"]).read_text().splitlines() if x.strip()]
        bos, eos, beam = int(entry["bos"]), int(entry["eos"]), int(entry.get("beam", 4))
        max_len = int(entry.get("max_len", max(len(it["tgt"]) for it in items) + 2))
        hyps = []
        for it in items:
            src = np.asarray([it["src"]], dtype=np.int64)
            if arch == "transformer":
                out = translate(model, src, bos, eos, int(entry.get("pad_id", 0)), max_len, beam_size=beam)[0]
            else:
                enc = model.encode(src, np.array([src.shape[1]]))

                def step(state, y, m=model):
                    logits, new, _ = m.decode_step(y, state)
                    return logits.data, new

                best = beam_search(step, model.init_state(enc), bos, eos, beam, max_len)[0]  # L4.4
                out = best.tokens[:-1] if best.finished else best.tokens
            hyps.append(" ".join(map(str, out)))
        refs = [" ".join(map(str, it["tgt"])) for it in items]
        per = np.array([exact_match([h], [r]) for h, r in zip(hyps, refs)])  # L4.5
        k = int(round(per.sum()))
        v, ci = _wilson_row(k, len(items))
        row.update(metric="em", value=v, ci95=ci, n=len(items), decode="beam" if beam > 1 else "greedy", beam=beam)
    elif kind in ("classifier", "mlm", "rtd"):
        texts, labels = _text_rows(entry, b)
        cls_id, sep_id, pad_id = int(entry.get("cls_id", 1)), int(entry.get("sep_id", 2)), int(entry.get("pad_id", 0))
        if kind == "classifier":
            bb = model.backbone.cfg
            ids, real = encode_batch(texts, getattr(bb, "max_len", getattr(bb, "n_ctx", 512)), cls_id, sep_id, pad_id)
            right = predict(model, ids, real) == labels  # L6.5
        elif kind == "mlm":
            ids, real = encode_batch(texts, model.bert.cfg.max_len, cls_id, sep_id, pad_id)
            special = ~real | (ids == cls_id) | (ids == sep_id)
            inputs, lab = mlm_mask(ids, special, int(entry.get("mask_id", 3)), model.bert.cfg.vocab, float(entry.get("p", 0.15)), rng)
            logits, _ = model(inputs, None, real)
            picked = lab != -100
            right = np.argmax(logits.data, axis=-1)[picked] == lab[picked]
        else:
            ids, real = encode_batch(texts, model.discriminator.cfg.max_len, cls_id, sep_id, pad_id)
            special = np.isin(ids, cfg.get("tl_special_ids", [])) | (ids == cls_id) | (ids == sep_id)
            right = rtd_accuracy(model, ids, rng, mask_id=int(cfg["tl_mask_id"]), special_mask=special, attn_mask=real, p=float(entry.get("p", 0.15))) > 0.5  # L6.3
        v, ci = _wilson_row(int(np.sum(right)), int(right.size))
        row.update(metric="accuracy", value=v, ci95=ci, n=int(right.size))
    elif kind == "embeddings":
        emb, vocab = model
        pairs = []
        for line in (b / entry["data"]).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                a, c, s = line.split("\t")
                pairs.append((a, c, float(s)))
        rho = word_similarity(emb, vocab, pairs)  # L2.3
        index = set(vocab)
        n = sum(1 for a, c, _ in pairs if a in index and c in index)
        # Fisher z: atanh(rho) is about normal with sd 1 / sqrt(n - 3).
        z, se = math.atanh(max(min(rho, 1 - 1e-12), -1 + 1e-12)), 1.0 / math.sqrt(max(n - 3, 1))
        q = normal_ppf(0.975)
        row.update(metric="spearman", value=float(rho), ci95=[math.tanh(z - q * se), math.tanh(z + q * se)], n=n)
    row["higher_is_better"] = row["metric"] != "bpb"
    p = _params(model)
    if p is not None:
        row["params"] = p
    row["status"] = "ok"
    return row


def run_zoo(manifest: str, rng: Any, seed: int = 0) -> dict[str, Any]:
    m = json.loads(Path(manifest).read_text())
    base = str(Path(manifest).resolve().parent)
    rows = []
    for entry in m["models"]:
        try:
            rows.append(score_entry(entry, base, rng))
        except KeyError as e:
            if "no zoo scorer" in str(e):
                rows.append({"model": entry["id"], "task": entry.get("task", "?"), "metric": entry.get("metric", "score"),
                             "status": "skipped", "value": None, "reason": str(e).strip("'\"")})
                continue
            rows.append(_error_row(entry, e))
        except Exception as e:  # noqa: BLE001 - one broken family must not hide the rest
            rows.append(_error_row(entry, e))
    return {"format": FORMAT, "suite": m.get("suite", "zoo"), "seed": int(seed), "manifest": manifest, "rows": rows}


def _error_row(entry: dict[str, Any], e: BaseException) -> dict[str, Any]:
    return {"model": entry.get("id", "?"), "task": entry.get("task", "?"), "metric": entry.get("metric", "score"),
            "status": "error", "value": None, "reason": f"{type(e).__name__}: {e}"}


def write_report(report: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=1) + "\n")
