"""Pass 6 inference forms of the `tinyllm` role (course/milestones/MS-L8.toml).

Entry-point territory (D16): this file is yours. It is glue over L8.2
(generate, KVCache), L8.3 (PagedKVCache), L8.5 (quantize_model), L8.6
(speculative_generate and the drafts), L8.7 (constrained decoding), and
L7.9's Llama model (loaded by cli_modern). It claims only the forms MS-L8
adds; every other command line falls through to the next glue module.

    generate --model <llama dir> (--prompt T | --prompt-file F) [--max-tokens N]
             [--greedy | --temperature T] [--seed S]
             [--cache none|contiguous|paged] [--kv-dtype f32|f16]
             [--spec ngram|prompt-lookup --k K]
             [--json-schema FILE]
    eval ppl --model <llama dir> --data <text file> [--quant int8|q4_g32|fp8_e4m3|mxfp4] [--tokens N]
    bench decode --model <llama dir> [--tokens N] [--prompt T]

Final stdout line: one JSON object (spec/cli-roles.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from tinyllm.cli_modern import UsageError, is_llama_dir, load

CLAIMS = ("--cache", "--kv-dtype", "--spec", "--json-schema", "--quant")
KV_DTYPES = {"f32": np.float32, "f16": np.float16}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


class PagedHook:
    """L8.3's PagedKVCache as the cache L8.2's generate and L7.9's model
    take: one sequence (id 0), update appends and returns the gathered
    layer, seq_len and positions follow the committed length."""

    def __init__(self, paged, n_layers: int) -> None:
        self.p, self.n_layers = paged, n_layers
        self.p.add_seq(0)

    def update(self, layer: int, k_new, v_new):
        self.p.append(0, layer, np.asarray(k_new)[0], np.asarray(v_new)[0])
        k, v = self.p.gather(0, layer)
        return k[None].astype(np.float32), v[None].astype(np.float32)

    def seq_len(self, layer=None) -> int:
        if layer is not None:
            return self.p.seq_len(0, layer)
        return min(self.p.seq_len(0, i) for i in range(self.n_layers))

    def positions(self, t_new: int):
        s = self.seq_len()
        return np.arange(s, s + t_new, dtype=np.int64)

    def mask(self, t_new: int):
        from tinyllm.xfmr.masks import causal_mask

        s = self.seq_len()
        return causal_mask(t_new, s + t_new, q_offset=s)


def _prompts(a) -> list[str]:
    if bool(a.prompt) == bool(a.prompt_file):
        raise UsageError("generate needs exactly one of --prompt and --prompt-file")
    if a.prompt:
        return [a.prompt]
    return [line for line in Path(a.prompt_file).read_text().splitlines() if line]


def _params(a):
    from tinyllm.infer.sample import SamplingParams

    return SamplingParams(
        temperature=0.0 if a.greedy else a.temperature,
        seed=a.seed,
        max_tokens=a.max_tokens,
    )


def _paged_cache(model, prompt_len: int, max_tokens: int):
    from tinyllm.infer.generate import cache_dims
    from tinyllm.infer.paged import PagedKVCache

    n_layers, n_kv, d_head, _ = cache_dims(model)
    block = 16
    blocks = (prompt_len + max_tokens) // block + 2
    return PagedHook(
        PagedKVCache(blocks, block, n_layers, n_kv, d_head), n_layers
    )


def _constrained(m, text, prompt: str, p, schema: dict) -> tuple[list[int], bool]:
    """Decode under the schema's DFA until no token is allowed (the subset
    used here has a finite language, so every path ends) or max_tokens."""
    from tinyllm.infer.constrain import (
        Constraint,
        TokenIndex,
        constrained_sample,
        json_schema_to_regex,
        regex_to_dfa,
    )
    from tinyllm.infer.generate import cache_dims
    from tinyllm.infer.kvcache import KVCache
    from tinyllm.infer.sample import request_rng

    idx = TokenIndex(
        regex_to_dfa(json_schema_to_regex(schema)), [bytes([i]) for i in range(256)]
    )
    c = Constraint(idx)
    ids = text.encode(prompt)
    n_layers, n_kv, d_head, max_len = cache_dims(m)
    kv = KVCache(n_layers, n_kv, d_head, min(max_len, len(ids) + p.max_tokens))
    rng = request_rng(p.seed if p.seed is not None else 0)
    row = np.asarray(
        m.forward(np.asarray([ids]), positions=kv.positions(len(ids)), cache=kv).data[
            0, -1
        ],
        dtype=np.float64,
    )
    out: list[int] = []
    while len(out) < p.max_tokens and c.mask().any():
        tok, _ = constrained_sample(row, c, p, out, rng, prompt=ids)
        out.append(tok)
        if len(ids) + len(out) >= kv.max_len or not c.mask().any():
            break
        row = np.asarray(
            m.forward(np.asarray([[tok]]), positions=kv.positions(1), cache=kv).data[
                0, -1
            ],
            dtype=np.float64,
        )
    doc = bytes(out)
    try:
        ok = c.is_complete() and _valid(json.loads(doc), schema)
    except ValueError:
        ok = False
    return out, ok


def _valid(v, s) -> bool:
    if "enum" in s:
        return any(v == e and type(v) is type(e) for e in s["enum"])
    t = s.get("type")
    if t == "object":
        props = s.get("properties", {})
        return (
            isinstance(v, dict)
            and set(s.get("required", [])) <= set(v) <= set(props)
            and all(_valid(v[k], props[k]) for k in v)
        )
    if t == "array":
        return (
            isinstance(v, list)
            and s.get("minItems", 0) <= len(v) <= s.get("maxItems", 1 << 30)
            and all(_valid(x, s["items"]) for x in v)
        )
    return {"string": str, "boolean": bool, "null": type(None)}.get(t, object) is type(
        v
    ) or (
        t in ("integer", "number")
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    )


def cmd_generate(a) -> dict:
    from tinyllm.infer.generate import generate

    prompts = _prompts(a)
    m, text = load(a.model)
    p = _params(a)
    per, extra = [], {}
    if a.json_schema:
        schema = json.loads(Path(a.json_schema).read_text())
        valid = 0
        for pr in prompts:
            ids, ok = _constrained(m, text, pr, p, schema)
            per.append(ids)
            valid += int(ok)
        extra = {
            "documents": len(prompts),
            "valid": valid,
            "valid_fraction": valid / len(prompts),
        }
    elif a.spec:
        from tinyllm.infer.spec import (
            NGramDraft,
            PromptLookupDraft,
            speculative_generate,
        )
        from tinyllm.lm.ngram import NGramLM

        drafted = accepted = calls = 0
        for pr in prompts:
            if a.spec == "ngram":
                lm = NGramLM(3, vocab_size=m.config.vocab_size)
                lm.fit([text.encode(pr)])
                draft = NGramDraft(lm)
            else:
                draft = PromptLookupDraft()
            g = speculative_generate(
                m, draft, text, pr, p, k=a.k, kv_dtype=KV_DTYPES[a.kv_dtype]
            )
            per.append(g.ids)
            drafted += g.stats["drafted"]
            accepted += g.stats["accepted"]
            calls += g.stats["target_calls"]
        extra = {
            "spec": a.spec,
            "k": a.k,
            "drafted": drafted,
            "accepted": accepted,
            "target_calls": calls,
            "acceptance_rate": accepted / drafted if drafted else 0.0,
        }
    else:
        for pr in prompts:
            cache = a.cache
            if cache == "paged":
                cache = _paged_cache(m, len(text.encode(pr)), p.max_tokens)
            g = generate(m, text, pr, p, cache=cache, kv_dtype=KV_DTYPES[a.kv_dtype])
            per.append(g.ids)
        extra = {"cache": a.cache, "kv_dtype": a.kv_dtype}
    ids = [t for row in per for t in row]
    out = {"ids": ids, "text": "".join(text.decode(r) for r in per), **extra}
    if a.prompt_file:
        out["per_prompt"] = per
    return out


def _nll(m, ids: list[int], window: int) -> tuple[float, int, list[int]]:
    """Summed next-token NLL (nats) over windows of the ids, the number of
    predictions, and the argmax prediction at each position."""
    total, n, top = 0.0, 0, []
    for s in range(0, len(ids) - 1, window):
        chunk = ids[s : s + window + 1]
        if len(chunk) < 2:
            break
        z = np.asarray(m(np.array([chunk[:-1]])).data[0], dtype=np.float64)
        mx = z.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(z - mx).sum(axis=1))
        tgt = np.asarray(chunk[1:])
        total += float((lse - z[np.arange(len(tgt)), tgt]).sum())
        n += len(tgt)
        top += [int(i) for i in z.argmax(axis=1)]
    return total, n, top


def cmd_eval_ppl(a) -> dict:
    m, text = load(a.model)
    ids = text.encode(Path(a.data).read_text())[: a.tokens]
    window = min(128, m.config.max_position_embeddings - 1)
    base, n, top = _nll(m, ids, window)
    out = {"tokens": n, "ppl_fp32": math.exp(base / n)}
    if a.quant:
        from tinyllm.infer.quant import quantize_model

        q, _ = load(a.model)
        quantize_model(q, a.quant)
        qn, _, qtop = _nll(q, ids, window)
        out.update({"quant": a.quant, "ppl": math.exp(qn / n)})
        out["increase_pct"] = 100.0 * (out["ppl"] / out["ppl_fp32"] - 1.0)
        # On a random tiny model the perplexity itself means little; how
        # often the quantized model keeps the fp32 model's top-1 prediction,
        # and how far its mean NLL moves, are the checks that do.
        out["top1_agree"] = sum(int(x == y) for x, y in zip(top, qtop)) / n
        out["nll_delta"] = abs(qn - base) / n
    else:
        out["ppl"] = out["ppl_fp32"]
    return out


def cmd_bench_decode(a) -> dict:
    from tinyllm.infer.generate import generate
    from tinyllm.infer.sample import SamplingParams

    m, text = load(a.model)
    p = SamplingParams(temperature=0.0, max_tokens=a.tokens)
    rates = {}
    for mode in ("none", "contiguous"):
        best = 0.0
        for _ in range(2):
            t0 = time.perf_counter()
            g = generate(m, text, a.prompt, p, cache=mode)
            best = max(best, len(g.ids) / (time.perf_counter() - t0))
        rates[mode] = best
    return {
        "tokens": a.tokens,
        "tok_s_none": rates["none"],
        "tok_s_contiguous": rates["contiguous"],
        "cache_speedup": rates["contiguous"] / rates["none"],
    }


def parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="tinyllm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")
    g.add_argument("--max-tokens", type=int, default=32)
    g.add_argument("--greedy", action="store_true")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument(
        "--cache", choices=["none", "contiguous", "paged"], default="contiguous"
    )
    g.add_argument("--kv-dtype", choices=sorted(KV_DTYPES), default="f32")
    g.add_argument("--spec", choices=["ngram", "prompt-lookup"])
    g.add_argument("--k", type=int, default=4)
    g.add_argument("--json-schema")
    g.set_defaults(fn=cmd_generate)
    e = sub.add_parser("eval")
    e.add_argument("what", choices=["ppl"])
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--quant", choices=["int8", "q4_g32", "fp8_e4m3", "mxfp4"])
    e.add_argument("--tokens", type=int, default=2048)
    e.set_defaults(fn=cmd_eval_ppl)
    b = sub.add_parser("bench")
    b.add_argument("what", choices=["decode"])
    b.add_argument("--model", required=True)
    b.add_argument("--tokens", type=int, default=256)
    b.add_argument("--prompt", default="Once upon a time")
    b.set_defaults(fn=cmd_bench_decode)
    return ap


def _model_arg(argv: list[str]) -> str | None:
    for i, x in enumerate(argv):
        if x == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith("--model="):
            return x.split("=", 1)[1]
    return None


def intercept(argv: list[str]) -> int | None:
    """Claim MS-L8's forms on a Llama directory; return None for anything else."""
    if not argv or not is_llama_dir(_model_arg(argv)):
        return None
    claims = argv[0] in ("generate", "eval") and any(
        x.split("=")[0] in CLAIMS for x in argv
    )
    claims = claims or argv[:2] == ["bench", "decode"]
    if not claims:
        return None
    try:
        a = parser().parse_args(argv)
        result = a.fn(a)
    except Exception as e:  # noqa: BLE001 - every failure is reported, not traced
        code = getattr(e, "exit_code", 1)
        why = str(e) if hasattr(e, "exit_code") else f"{type(e).__name__}: {e}"
        print(f"tinyllm{'' if code == 2 else ' ' + argv[0]}: {why}", file=sys.stderr)
        return code
    print(json.dumps(result), flush=True)
    return 0
