#!/usr/bin/env python3
"""Open-loop Poisson load generator for OpenAI-compatible streaming endpoints (stdlib only).

Measures what a customer SLO is written in: TTFT, ITL, TPOT, E2E latency,
output tok/s, req/s, and goodput (requests meeting every SLO), at p50/p90/p99.

Open loop: arrivals follow a Poisson process at --rate req/s regardless of how
fast the server answers, so queueing delay shows up in the numbers (a closed
loop with N users hides it: a slow server simply receives fewer requests).

Usage:
  # offline self-test: starts a fake SSE server in-process
  python loadgen.py --mock --rate 8 --num-requests 200

  # sweep several rates and print the latency-throughput curve
  python loadgen.py --mock --sweep 2,4,8,16,32 --num-requests 150

  # against a real engine (vLLM / SGLang / TRT-LLM / Dynamo frontend)
  python loadgen.py --base-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
      --endpoint chat --rate 4 --num-requests 500 --input-len 1024 --output-len 256 \
      --slo-ttft-ms 500 --slo-tpot-ms 50

Token counts: output tokens are counted from streamed content chunks (one chunk
is one token on vLLM and SGLang) unless the server returns a usage block.
Input length is approximate: prompts are built from random words (~1 token each).
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import random
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu"
).split()


# ----------------------------------------------------------------------------- results


@dataclass
class Result:
    ok: bool = False
    error: str = ""
    sent_at: float = 0.0  # perf_counter seconds
    ttft: float = math.nan
    e2e: float = math.nan
    itls: list[float] = field(default_factory=list)
    output_tokens: int = 0
    input_tokens: int = 0

    @property
    def tpot(self) -> float:
        # Time per output token after the first, the decode-phase speed one user sees.
        if self.output_tokens < 2:
            return math.nan
        return (self.e2e - self.ttft) / (self.output_tokens - 1)


def pct(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ----------------------------------------------------------------------------- workload


def sample_len(rng: random.Random, mean: int, dist: str) -> int:
    if dist == "fixed":
        return mean
    if dist == "uniform":  # +-50% around the mean, like --random-range-ratio 0.5
        return max(1, int(rng.uniform(0.5 * mean, 1.5 * mean)))
    # lognormal with sigma=0.8: heavy right tail, the shape of real chat traffic
    sigma = 0.8
    mu = math.log(mean) - sigma**2 / 2  # so that E[len] = mean
    return max(1, int(rng.lognormvariate(mu, sigma)))


def make_prompt(
    rng: random.Random, n_words: int, shared_prefix: str, req_id: int
) -> str:
    # A unique nonce at the very start defeats prefix caching for the random part, so a
    # benchmark that reuses prompts does not quietly measure cache hits. --shared-prefix-len
    # puts a deliberate common prefix *before* the nonce to measure caching on purpose.
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return f"{shared_prefix}[req {req_id} {rng.getrandbits(64):x}] {body}"


# ----------------------------------------------------------------------------- client


def one_request(
    args: argparse.Namespace, prompt: str, in_len: int, out_len: int, res: Result
) -> None:
    u = urllib.parse.urlparse(args.base_url)
    conn_cls = (
        http.client.HTTPSConnection
        if u.scheme == "https"
        else http.client.HTTPConnection
    )
    if args.endpoint == "chat":
        path = "/v1/chat/completions"
        body: dict = {"messages": [{"role": "user", "content": prompt}]}
    else:
        path = "/v1/completions"
        body = {"prompt": prompt}
    body.update(
        model=args.model,
        max_tokens=out_len,
        stream=True,
        temperature=0.0,
        stream_options={"include_usage": True},
    )
    if args.ignore_eos:
        body["ignore_eos"] = True  # vLLM/SGLang extension: force exactly max_tokens
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    res.input_tokens = in_len
    res.sent_at = time.perf_counter()
    last = None
    try:
        conn = conn_cls(u.hostname, u.port, timeout=args.timeout)
        conn.request(
            "POST", (u.path.rstrip("/") + path), body=json.dumps(body), headers=headers
        )
        resp = conn.getresponse()
        if resp.status != 200:
            res.error = f"HTTP {resp.status}: {resp.read(200)!r}"
            return
        usage_out = None
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage_out = chunk["usage"].get("completion_tokens")
                res.input_tokens = chunk["usage"].get("prompt_tokens") or in_len
            choices = chunk.get("choices") or []
            if not choices:
                continue
            c = choices[0]
            text = (
                c.get("text") if "text" in c else (c.get("delta") or {}).get("content")
            )
            if not text:
                continue
            now = time.perf_counter()
            if last is None:
                res.ttft = now - res.sent_at
            else:
                res.itls.append(now - last)
            last = now
            res.output_tokens += 1
        conn.close()
        res.e2e = time.perf_counter() - res.sent_at
        if usage_out:
            res.output_tokens = usage_out
        res.ok = last is not None
        if not res.ok:
            res.error = "no tokens streamed"
    except Exception as e:  # network errors are results, not crashes
        res.error = f"{type(e).__name__}: {e}"


def run_load(args: argparse.Namespace, rate: float) -> dict:
    rng = random.Random(args.seed)
    shared = " ".join(rng.choice(WORDS) for _ in range(args.shared_prefix_len))
    shared = shared + " " if shared else ""

    # Warmup: sequential, excluded. Loads CUDA graphs / JIT, fills allocator pools.
    for i in range(args.warmup):
        one_request(args, make_prompt(rng, 32, shared, -1 - i), 32, 16, Result())

    results: list[Result] = []
    threads: list[threading.Thread] = []
    t0 = time.perf_counter()
    next_t = t0
    for i in range(args.num_requests):
        in_len = sample_len(rng, args.input_len, args.len_dist)
        out_len = sample_len(rng, args.output_len, args.len_dist)
        prompt = make_prompt(rng, in_len, shared, i)
        # Poisson process: exponential inter-arrival gaps with mean 1/rate.
        # burstiness < 1 makes it burstier (gamma shape), as in vllm bench serve --burstiness.
        if math.isfinite(rate):
            theta = 1.0 / (rate * args.burstiness)
            next_t += rng.gammavariate(args.burstiness, theta)
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        r = Result()
        results.append(r)
        th = threading.Thread(
            target=one_request, args=(args, prompt, in_len, out_len, r), daemon=True
        )
        th.start()  # one thread per request: arrivals never wait on completions (open loop)
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    return summarize(args, rate, results, wall)


def summarize(
    args: argparse.Namespace, rate: float, results: list[Result], wall: float
) -> dict:
    ok = [r for r in results if r.ok]
    ms = 1e3
    ttft = [r.ttft * ms for r in ok]
    tpot = [r.tpot * ms for r in ok]
    itl = [x * ms for r in ok for x in r.itls]
    e2e = [r.e2e * ms for r in ok]
    out_tok = sum(r.output_tokens for r in ok)

    def meets(r: Result) -> bool:
        good = True
        if args.slo_ttft_ms:
            good &= r.ttft * ms <= args.slo_ttft_ms
        if args.slo_tpot_ms and not math.isnan(r.tpot):
            good &= r.tpot * ms <= args.slo_tpot_ms
        return good

    good = [r for r in ok if meets(r)]
    s = {
        "rate": rate,
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "wall_s": wall,
        "req_per_s": len(ok) / wall,
        "output_tok_per_s": out_tok / wall,
        "goodput_req_per_s": len(good) / wall,
        "slo_attainment": len(good) / max(1, len(results)),
        # Little's law: mean in-flight requests = arrival rate x mean time in system.
        "mean_in_flight": (len(ok) / wall) * (sum(e2e) / max(1, len(e2e)) / ms),
    }
    for name, xs in (
        ("ttft_ms", ttft),
        ("tpot_ms", tpot),
        ("itl_ms", itl),
        ("e2e_ms", e2e),
    ):
        for q in (50, 90, 99):
            s[f"{name}_p{q}"] = pct(xs, q)
    first_err = next((r.error for r in results if not r.ok), "")
    if first_err:
        s["first_error"] = first_err
    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(
                json.dumps({"summary": s, "requests": [asdict(r) for r in results]})
                + "\n"
            )
    return s


def print_summary(s: dict) -> None:
    print(
        f"rate {s['rate']:g} req/s | {s['ok']}/{s['requests']} ok in {s['wall_s']:.1f}s | "
        f"{s['req_per_s']:.2f} req/s | {s['output_tok_per_s']:,.0f} out tok/s | "
        f"goodput {s['goodput_req_per_s']:.2f} req/s ({s['slo_attainment']:.0%} in SLO) | "
        f"L={s['mean_in_flight']:.1f} in flight"
    )
    print(f"{'metric':10s} {'p50':>9s} {'p90':>9s} {'p99':>9s}")
    for m in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
        print(f"{m:10s} {s[m + '_p50']:9.1f} {s[m + '_p90']:9.1f} {s[m + '_p99']:9.1f}")
    if "first_error" in s:
        print(f"first error: {s['first_error']}")


# ----------------------------------------------------------------------------- mock server


class MockEngine:
    """Toy model of a continuous-batching engine, so the curve has the right shape.

    TTFT = queue wait for a prefill slot + prefill time (prompt_len x per-token cost).
    ITL  = base_itl x (1 + running / knee): decode slows as the batch grows, because each
           step reads more KV cache. Only max_running requests decode at once; the rest wait.
    """

    def __init__(
        self, base_itl_ms: float, prefill_us_per_tok: float, knee: int, max_running: int
    ):
        self.base_itl = base_itl_ms / 1e3
        self.prefill_per_tok = prefill_us_per_tok / 1e6
        self.knee = knee
        self.slots = threading.Semaphore(max_running)
        self.prefill_lock = (
            threading.Lock()
        )  # one prefill at a time, like a non-chunked scheduler
        self.running = 0
        self.mu = threading.Lock()

    def itl(self) -> float:
        with self.mu:
            n = self.running
        return self.base_itl * (1 + n / self.knee)


def make_handler(engine: MockEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence per-request logging
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            chat = self.path.endswith("/chat/completions")
            prompt = req["messages"][-1]["content"] if chat else req.get("prompt", "")
            prompt_tokens = len(prompt.split())
            max_tokens = int(req.get("max_tokens", 16))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            engine.slots.acquire()
            with engine.mu:
                engine.running += 1
            try:
                with engine.prefill_lock:
                    time.sleep(prompt_tokens * engine.prefill_per_tok)
                for i in range(max_tokens):
                    if i:
                        time.sleep(engine.itl())
                    tok = random.choice(WORDS) + " "
                    choice = (
                        {"index": 0, "delta": {"content": tok}}
                        if chat
                        else {"index": 0, "text": tok}
                    )
                    self._sse(
                        {
                            "object": "chat.completion.chunk"
                            if chat
                            else "text_completion",
                            "choices": [choice],
                        }
                    )
                self._sse(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": max_tokens,
                        },
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with engine.mu:
                    engine.running -= 1
                engine.slots.release()

        def _sse(self, obj: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


def start_mock(args: argparse.Namespace) -> str:
    engine = MockEngine(
        args.mock_itl_ms, args.mock_prefill_us, args.mock_knee, args.mock_max_running
    )
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="mock")
    p.add_argument("--endpoint", choices=["completions", "chat"], default="completions")
    p.add_argument("--api-key")
    p.add_argument(
        "--rate",
        type=float,
        default=4.0,
        help="mean arrivals per second (inf = all at once)",
    )
    p.add_argument(
        "--sweep", help="comma-separated rates; prints the latency-throughput curve"
    )
    p.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="gamma shape; 1.0 = Poisson, <1 burstier",
    )
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument(
        "--len-dist", choices=["fixed", "uniform", "lognormal"], default="lognormal"
    )
    p.add_argument(
        "--shared-prefix-len",
        type=int,
        default=0,
        help="words of common prefix (tests prefix cache)",
    )
    p.add_argument(
        "--ignore-eos", action="store_true", help="send ignore_eos=true (vLLM/SGLang)"
    )
    p.add_argument("--slo-ttft-ms", type=float, default=0)
    p.add_argument("--slo-tpot-ms", type=float, default=0)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--json-out", help="append per-run JSON lines (summary + per-request) here"
    )
    m = p.add_argument_group("mock server")
    m.add_argument(
        "--mock",
        action="store_true",
        help="start an in-process fake SSE server and target it",
    )
    m.add_argument(
        "--mock-itl-ms", type=float, default=10.0, help="decode step time at batch 1"
    )
    m.add_argument(
        "--mock-prefill-us",
        type=float,
        default=50.0,
        help="prefill cost per prompt token",
    )
    m.add_argument(
        "--mock-knee", type=int, default=32, help="batch size at which ITL doubles"
    )
    m.add_argument(
        "--mock-max-running", type=int, default=64, help="like --max-num-seqs"
    )
    args = p.parse_args(argv)

    if args.mock:
        args.base_url = start_mock(args)
        print(f"mock engine at {args.base_url}")

    rates = [float(x) for x in args.sweep.split(",")] if args.sweep else [args.rate]
    summaries = []
    for r in rates:
        s = run_load(args, r)
        summaries.append(s)
        print_summary(s)
        print()
    if len(summaries) > 1:
        print("latency-throughput curve (one row per offered load)")
        print(
            f"{'rate':>6s} {'out tok/s':>10s} {'goodput':>8s} {'ttft p50':>9s} {'ttft p99':>9s} "
            f"{'tpot p50':>9s} {'tpot p99':>9s}"
        )
        for s in summaries:
            print(
                f"{s['rate']:6g} {s['output_tok_per_s']:10,.0f} {s['goodput_req_per_s']:8.2f} "
                f"{s['ttft_ms_p50']:9.1f} {s['ttft_ms_p99']:9.1f} {s['tpot_ms_p50']:9.1f} {s['tpot_ms_p99']:9.1f}"
            )
    return 0 if all(s["ok"] for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
