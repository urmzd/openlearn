"""Runs compared with each other: the `disagg` suite and `greedy-equal`.

The other matchers compare one output with a fixture. These two compare a
learner's engines with each other (DESIGN 4.3 MS-L10 steps 5 and 6):

`disagg` (match = "disagg") drives one prefill engine (P) and two decode
engines (D) the way the gateway does (gw.05, DESIGN 2.7 "Disaggregated
flow"), without the gateway: tokenize the prompt on P (`/v1/tokenize` with
`add_special`), call `tl.engine.v1.EngineControl/Prefill` on P with
`decode_target` = a D's `{kv_port}`, then POST the same request to that D
with `X-TL-KV-Handle: <handle_id>`. Its cases:

| case                     | checks                                                         |
|--------------------------|----------------------------------------------------------------|
| info                     | Info: P lists role prefill and each D role decode, one model, and every D reads P's kv_format |
| greedy.equals-unified    | per prompt (alternating D, plain and streamed), the greedy text equals the unified engine's |
| seeded.equals-unified    | temperature 0.9 with a seed: equal to the unified engine (the KvHandle RNG hand-off) |
| handle.unknown           | a resume with a handle nobody pushed is 404 `kv_handle_not_found` |
| handle.release           | Release on D frees the handle: the resume is 404 and D's used KV blocks are back to baseline |
| reset.prefill-fails      | P pushes through a proxy that resets the link mid-transfer: Prefill fails and, within 5 s, P's and D's used KV blocks are back to baseline |
| reset.recovers           | the next disaggregated request on that D equals the unified engine |

Used blocks are `tl_engine_kv_blocks{state="used"}` on the health port's
`/metrics` (otel/metrics.yaml). The reset is a TCP proxy in the harness
(the behaviour of testkit chaosproxy's ResetAfterBytes, counted in the
direction P sends): every connection through it is reset once more than
`reset_after_bytes` have gone upstream, so HasBlocks passes and PushKv is
cut inside the first block's envelope.

`greedy-equal` (match = "greedy-equal") sends each prompt greedily to every
engine named in `engines` and wants one text and one completion_tokens
count: the plain and the speculative engine (L10.8).

gRPC goes through grpcwire (stdlib HTTP/2); `Rpc` is the seam the tests
replace with a fake.
"""

from __future__ import annotations

import secrets
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

from . import conform, grpcwire, web

COMPLETIONS = "/v1/completions"
USED = 'tl_engine_kv_blocks{state="used"}'

# Fixture prompts: short (one partial block), long (several full blocks plus a
# tail at block size 16), and a repeat of the first on the other decode
# engine, whose second transfer can be deduplicated.
PROMPTS = [
    "Once upon a time",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over "
    "the lazy dog again, and then it rests.",
    "A long time ago in a galaxy far, far away, a small robot",
    "Once upon a time",
]
SEEDED = {"temperature": 0.9, "top_p": 1.0, "seed": 1234}
RESET_PROMPT = (
    "Every block of this prompt crosses the link before the decode engine "
    "can resume it, so a reset here lands mid-transfer."
)


@dataclass
class Engine:
    """One started engine, as the suite reaches it."""

    name: str
    base: str  # http://127.0.0.1:<port>
    health: str  # http://127.0.0.1:<health_port>
    grpc: str = ""  # 127.0.0.1:<grpc_port>
    kv: str = ""  # 127.0.0.1:<kv_port>


@dataclass
class Result:
    case: str
    status: str  # pass fail
    detail: str = ""


# ---------------------------------------------------------------------------
# the gRPC calls (tl.engine.v1, tl.kv.v1)


def encode_sampling(s: dict) -> bytes:
    g = grpcwire
    return b"".join(
        [
            g.field_float(1, s.get("temperature", 1.0)),
            g.field_float(2, s.get("top_p", 1.0)),
            g.field_varint(3, s.get("top_k", 0)),
            g.field_float(4, s.get("min_p", 0.0)),
            g.field_float(5, s.get("repetition_penalty", 1.0)),
            g.field_float(6, s.get("presence_penalty", 0.0)),
            g.field_float(7, s.get("frequency_penalty", 0.0)),
            g.field_varint(8, s.get("seed", 0)),
            g.field_varint(9, s.get("max_new_tokens", 0)),
            *(g.field_bytes(10, x) for x in s.get("stop", [])),
            g.field_varint(11, s.get("logprobs", 0)),
        ]
    )


def encode_prefill(req: dict) -> bytes:
    g = grpcwire
    return b"".join(
        [
            g.field_bytes(1, req.get("request_id", "")),
            g.field_bytes(2, req.get("model", "")),
            g.field_packed(3, req.get("prompt_ids", [])),
            g.field_message(4, encode_sampling(req.get("sampling", {}))),
            g.field_bytes(5, req.get("decode_target", "")),
            g.field_varint(6, req.get("deadline_unix_ms", 0)),
            g.field_varint(7, req.get("priority", 0)),
        ]
    )


def decode_handle(buf: bytes) -> dict:
    g = grpcwire
    f = g.decode(buf)
    return {
        "handle_id": g.one(f, 1, b"").decode(),
        "decode_target": g.one(f, 2, b"").decode(),
        "n_tokens": g.one(f, 3),
        "block_hashes": g.packed_varints(f, 4),
        "kv_format": g.one(f, 5),
        "first_token": g.one(f, 6),
        "prompt_tokens": g.one(f, 7),
        "rng_draws_consumed": g.one(f, 8),
    }


def decode_prefill(buf: bytes) -> dict:
    g = grpcwire
    f = g.decode(buf)
    return {
        "handle": decode_handle(g.one(f, 1, b"")),
        "first_token": g.one(f, 2),
        "first_logprob": g.as_float(g.one(f, 3)),
        "prompt_tokens": g.as_int32(g.one(f, 4)),
        "cached_prompt_tokens": g.as_int32(g.one(f, 5)),
        "ttft_ms": g.as_double(g.one(f, 6)),
    }


def decode_info(buf: bytes) -> dict:
    g = grpcwire
    f = g.decode(buf)
    return {
        "model": g.one(f, 1, b"").decode(),
        "roles": [r.decode() for r in f.get(2, [])],
        "kv_format": g.one(f, 3),
        "max_context": g.as_int32(g.one(f, 4)),
        "block_size": g.as_int32(g.one(f, 5)),
        "abi": g.one(f, 6),
        "kv_formats_read": g.packed_varints(f, 7),
    }


class Rpc:
    """EngineControl.Prefill and Info, KvTransferService.Release."""

    timeout = 30.0

    def prefill(self, addr: str, req: dict) -> dict:
        raw = grpcwire.call(
            addr,
            "/tl.engine.v1.EngineControl/Prefill",
            encode_prefill(req),
            self.timeout,
        )
        return decode_prefill(raw)

    def info(self, addr: str) -> dict:
        raw = grpcwire.call(addr, "/tl.engine.v1.EngineControl/Info", b"", self.timeout)
        return decode_info(raw)

    def release(self, addr: str, handle_id: str) -> None:
        grpcwire.call(
            addr,
            "/tl.kv.v1.KvTransferService/Release",
            grpcwire.field_bytes(1, handle_id),
            self.timeout,
        )


# ---------------------------------------------------------------------------
# the transfer reset


class ResetProxy:
    """A TCP proxy to `upstream` that resets both sides of a connection (RST,
    SO_LINGER 0) once more than `after` bytes have gone upstream on it."""

    def __init__(self, upstream: str, after: int):
        host, _, port = upstream.rpartition(":")
        self.upstream = (host or "127.0.0.1", int(port))
        self.after = after
        self.resets = 0
        self.connections = 0
        self._lock = threading.Lock()
        self._socks: list[socket.socket] = []
        self.ln = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ln.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ln.bind(("127.0.0.1", 0))
        self.ln.listen(16)
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def addr(self) -> str:
        return f"127.0.0.1:{self.ln.getsockname()[1]}"

    def _accept(self) -> None:
        while not self._closed:
            try:
                c, _ = self.ln.accept()
            except OSError:
                return
            try:
                u = socket.create_connection(self.upstream, timeout=10)
            except OSError:
                self._rst(c)
                continue
            with self._lock:
                self.connections += 1
                self._socks += [c, u]
            threading.Thread(target=self._up, args=(c, u), daemon=True).start()
            threading.Thread(target=self._down, args=(u, c), daemon=True).start()

    @staticmethod
    def _rst(s: socket.socket) -> None:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        except OSError:
            pass
        try:
            s.close()
        except OSError:
            pass

    def _up(self, c: socket.socket, u: socket.socket) -> None:
        sent = 0
        try:
            while True:
                chunk = c.recv(65536)
                if not chunk:
                    u.shutdown(socket.SHUT_WR)
                    return
                if sent + len(chunk) > self.after:
                    u.sendall(chunk[: self.after - sent])
                    with self._lock:
                        self.resets += 1
                    self._rst(c)
                    self._rst(u)
                    return
                u.sendall(chunk)
                sent += len(chunk)
        except OSError:
            self._rst(c)
            self._rst(u)

    def _down(self, u: socket.socket, c: socket.socket) -> None:
        try:
            while True:
                chunk = u.recv(65536)
                if not chunk:
                    c.shutdown(socket.SHUT_WR)
                    return
                c.sendall(chunk)
        except OSError:
            pass

    def close(self) -> None:
        self._closed = True
        try:
            self.ln.close()
        except OSError:
            pass
        with self._lock:
            for s in self._socks:
                try:
                    s.close()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# HTTP helpers


def used_blocks(e: Engine) -> float | None:
    """tl_engine_kv_blocks{state="used"} on the engine's /metrics."""
    r = web.get(e.health.rstrip("/") + "/metrics", timeout=5)
    if r.status != 200:
        return None
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        head, _, val = line.rpartition(" ")
        name, _, labels = head.partition("{")
        if name != "tl_engine_kv_blocks":
            continue
        pairs = {
            k.strip(): v.strip().strip('"')
            for k, _, v in (p.partition("=") for p in labels.rstrip("}").split(","))
        }
        if pairs.get("state") == "used":
            try:
                return float(val)
            except ValueError:
                return None
    return None


def _text(r: web.Response, stream: bool) -> tuple[str | None, list[str]]:
    """The completion text and usage tokens, or None and why."""
    if r.status != 200:
        return None, [f"HTTP {r.status}: {r.error or r.text[:300]}"]
    if stream:
        text, _, errs = conform._stream_text(r)
        return text, errs
    try:
        return r.json()["choices"][0]["text"], []
    except (ValueError, KeyError, IndexError, TypeError) as e:
        return None, [f"unreadable completion ({type(e).__name__}): {r.text[:300]}"]


@dataclass
class Suite:
    unified: Engine
    prefill: Engine
    decode: list[Engine]
    model: str
    max_tokens: int = 16
    prompts: list[str] = field(default_factory=lambda: list(PROMPTS))
    reset_after_bytes: int = 1024
    rpc: Rpc = field(default_factory=Rpc)
    timeout: float = 60.0
    settle_s: float = 5.0

    def body(self, prompt: str, stream: bool = False, **kw) -> dict:
        b = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": stream,
        }
        b.update(kw)
        return b

    def unified_text(self, body: dict) -> tuple[str | None, list[str]]:
        r = web.request(
            "POST",
            self.unified.base + COMPLETIONS,
            json_body={**body, "stream": False},
            timeout=self.timeout,
        )
        t, errs = _text(r, False)
        return t, [f"unified engine {self.unified.name}: {e}" for e in errs]

    def tokenize(self, prompt: str) -> list[int]:
        r = web.request(
            "POST",
            self.prefill.base + "/v1/tokenize",
            json_body={"model": self.model, "text": prompt, "add_special": True},
            timeout=self.timeout,
        )
        if r.status != 200:
            raise _Fail(
                f"POST {self.prefill.name} /v1/tokenize: HTTP {r.status} {r.error or r.text[:200]}"
            )
        ids = r.json().get("ids")
        if not isinstance(ids, list) or not ids:
            raise _Fail(
                f"/v1/tokenize on {self.prefill.name} returned no ids: {r.text[:200]}"
            )
        return [int(x) for x in ids]

    def prefill_req(self, body: dict, ids: list[int], target: str, rid: str) -> dict:
        return {
            "request_id": rid,
            "model": self.model,
            "prompt_ids": ids,
            "sampling": {
                "temperature": body.get("temperature", 1.0),
                "top_p": body.get("top_p", 1.0),
                "seed": body.get("seed", 0),
                "max_new_tokens": body.get("max_tokens", self.max_tokens),
                "repetition_penalty": 1.0,
            },
            "decode_target": target,
            "deadline_unix_ms": int(time.time() * 1000) + 30_000,
        }

    def run_prefill(self, body: dict, d: Engine, target: str | None = None) -> dict:
        ids = self.tokenize(body["prompt"])
        rid = f"disagg-{secrets.token_hex(6)}"
        try:
            out = self.rpc.prefill(
                self.prefill.grpc, self.prefill_req(body, ids, target or d.kv, rid)
            )
        except grpcwire.RpcError as e:
            raise _Fail(f"Prefill on {self.prefill.name} -> {d.name}: {e}") from None
        h = out.get("handle") or {}
        if not h.get("handle_id"):
            raise _Fail(f"Prefill on {self.prefill.name} returned no handle_id: {out}")
        if out.get("prompt_tokens") != len(ids):
            raise _Fail(
                f"Prefill reports prompt_tokens {out.get('prompt_tokens')}, "
                f"/v1/tokenize gave {len(ids)} ids"
            )
        return out

    def resume(self, d: Engine, body: dict, handle: str) -> web.Response:
        return web.request(
            "POST",
            d.base + COMPLETIONS,
            json_body=body,
            headers={"X-TL-KV-Handle": handle},
            timeout=self.timeout,
        )

    def disagg_text(self, body: dict, d: Engine) -> tuple[str | None, list[str]]:
        out = self.run_prefill(body, d)
        r = self.resume(d, body, out["handle"]["handle_id"])
        t, errs = _text(r, bool(body.get("stream")))
        return t, [f"resume on {d.name}: {e}" for e in errs]

    def compare(self, body: dict, d: Engine) -> list[str]:
        want, errs = self.unified_text(body)
        if errs:
            return errs
        got, errs = self.disagg_text(body, d)
        if errs:
            return errs
        if got != want:
            mode = "streamed" if body.get("stream") else "plain"
            return [
                f"{body['prompt'][:40]!r} via {self.prefill.name} -> {d.name} ({mode}): "
                f"{got!r} != unified {want!r}"
            ]
        return []

    # -- cases -------------------------------------------------------------------

    def case_info(self) -> list[str]:
        errs: list[str] = []
        try:
            p = self.rpc.info(self.prefill.grpc)
        except grpcwire.RpcError as e:
            return [f"Info on {self.prefill.name} ({self.prefill.grpc}): {e}"]
        if "prefill" not in p.get("roles", []):
            errs.append(f"{self.prefill.name} roles {p.get('roles')}, want prefill")
        for d in self.decode:
            try:
                i = self.rpc.info(d.grpc)
            except grpcwire.RpcError as e:
                errs.append(f"Info on {d.name} ({d.grpc}): {e}")
                continue
            if "decode" not in i.get("roles", []):
                errs.append(f"{d.name} roles {i.get('roles')}, want decode")
            if i.get("model") != p.get("model"):
                errs.append(
                    f"{d.name} serves {i.get('model')!r}, {self.prefill.name} {p.get('model')!r}"
                )
            reads = i.get("kv_formats_read") or [i.get("kv_format")]
            if p.get("kv_format") not in reads:
                errs.append(
                    f"{d.name} reads KV formats {reads}, {self.prefill.name} writes {p.get('kv_format')}"
                )
        return errs

    def case_greedy(self) -> list[str]:
        errs: list[str] = []
        for i, prompt in enumerate(self.prompts):
            d = self.decode[i % len(self.decode)]
            errs += self.compare(self.body(prompt, stream=i % 2 == 1), d)
        return errs

    def case_seeded(self) -> list[str]:
        return self.compare(self.body(self.prompts[0], **SEEDED), self.decode[0])

    def case_unknown(self) -> list[str]:
        d = self.decode[0]
        r = self.resume(
            d, self.body(self.prompts[0]), f"disagg-missing-{secrets.token_hex(6)}"
        )
        return _not_found(r, d, "an unknown handle")

    def _settled(self, e: Engine, base: float) -> tuple[bool, float | None]:
        deadline = time.monotonic() + self.settle_s
        while True:
            now = used_blocks(e)
            if now is not None and now <= base:
                return True, now
            if time.monotonic() >= deadline:
                return False, now
            time.sleep(0.1)

    def _baseline(self, e: Engine) -> float:
        v = used_blocks(e)
        if v is None:
            raise _Fail(f"{e.name}: no {USED} on {e.health}/metrics")
        return v

    def case_release(self) -> list[str]:
        d = self.decode[-1]
        base = self._baseline(d)
        body = self.body(self.prompts[1])
        h = self.run_prefill(body, d)["handle"]["handle_id"]
        try:
            self.rpc.release(d.kv, h)
            self.rpc.release(d.kv, h)  # idempotent
        except grpcwire.RpcError as e:
            return [f"Release on {d.name} ({d.kv}): {e}"]
        errs = _not_found(self.resume(d, body, h), d, "a released handle")
        ok, now = self._settled(d, base)
        if not ok:
            errs.append(f"{d.name}: {USED} {now} after Release, was {base}")
        return errs

    def case_reset(self) -> list[str]:
        d = self.decode[-1]
        base_p, base_d = self._baseline(self.prefill), self._baseline(d)
        proxy = ResetProxy(d.kv, self.reset_after_bytes)
        try:
            out = None
            try:
                out = self.run_prefill(self.body(RESET_PROMPT), d, target=proxy.addr)
            except _Fail as e:
                failed = str(e)
            if out is not None:
                return [
                    f"Prefill succeeded although the link to {d.name} was reset after "
                    f"{self.reset_after_bytes} bytes ({proxy.resets} resets): want an error"
                ]
            if proxy.connections == 0:
                return [
                    f"Prefill failed without connecting to its decode_target ({failed}); "
                    "P must push the KV to the decode_target it is given"
                ]
            if proxy.resets == 0:
                return [
                    f"the push never passed {self.reset_after_bytes} bytes, so nothing was reset "
                    f"({failed}); the prompt's KV should be larger"
                ]
        finally:
            proxy.close()
        errs = []
        for e, base in ((self.prefill, base_p), (d, base_d)):
            ok, now = self._settled(e, base)
            if not ok:
                errs.append(
                    f"{e.name}: {USED} is {now} {self.settle_s:g}s after the reset, was {base}: blocks leaked"
                )
        return errs

    def case_recovers(self) -> list[str]:
        return self.compare(self.body(self.prompts[1]), self.decode[-1])

    CASES = (
        ("info", "case_info"),
        ("greedy.equals-unified", "case_greedy"),
        ("seeded.equals-unified", "case_seeded"),
        ("handle.unknown", "case_unknown"),
        ("handle.release", "case_release"),
        ("reset.prefill-fails", "case_reset"),
        ("reset.recovers", "case_recovers"),
    )

    def run(self) -> list[Result]:
        out: list[Result] = []
        for name, meth in self.CASES:
            try:
                errs = getattr(self, meth)()
            except _Fail as e:
                errs = [str(e)]
            except Exception as e:  # a malformed answer fails the case, not the run
                errs = [f"{type(e).__name__}: {e}"]
            out.append(Result(name, "fail" if errs else "pass", "\n".join(errs)))
        return out


class _Fail(Exception):
    pass


def _not_found(r: web.Response, d: Engine, what: str) -> list[str]:
    if r.status != 404:
        return [
            f"{d.name}: a resume with {what} answered HTTP {r.status}, want 404: {r.text[:200]}"
        ]
    try:
        code = r.json()["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return [f"{d.name}: 404 without an error object: {r.text[:200]}"]
    if code != "kv_handle_not_found":
        return [f"{d.name}: 404 with code {code!r}, want kv_handle_not_found"]
    return []


# ---------------------------------------------------------------------------
# greedy-equal


def greedy_equal(
    engines: list[Engine],
    model: str,
    prompts: list[str],
    max_tokens: int,
    timeout: float = 60.0,
) -> list[str]:
    """Each prompt, greedy, on every engine: one text, one token count."""
    errs: list[str] = []
    for prompt in prompts:
        seen: list[tuple[str, str, int]] = []
        for e in engines:
            body = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": 0,
                "stream": False,
            }
            r = web.request(
                "POST", e.base + COMPLETIONS, json_body=body, timeout=timeout
            )
            if r.status != 200:
                errs.append(f"{e.name}: HTTP {r.status} {r.error or r.text[:200]}")
                continue
            try:
                doc = r.json()
                seen.append(
                    (
                        e.name,
                        doc["choices"][0]["text"],
                        int(doc["usage"]["completion_tokens"]),
                    )
                )
            except (ValueError, KeyError, IndexError, TypeError) as x:
                errs.append(
                    f"{e.name}: unreadable completion ({type(x).__name__}): {r.text[:200]}"
                )
        for name, text, n in seen[1:]:
            ref, rtext, rn = seen[0]
            if (text, n) != (rtext, rn):
                errs.append(
                    f"{prompt[:40]!r}: {name} {text!r} ({n} tokens) != {ref} {rtext!r} ({rn} tokens)"
                )
    return errs


def summary(results: list[Result]) -> tuple[bool, str]:
    bad = [r for r in results if r.status == "fail"]
    if bad:
        return False, "\n".join(f"{r.case}: {r.detail}" for r in bad)
    return True, f"{len(results)} cases pass ({', '.join(r.case for r in results)})"
