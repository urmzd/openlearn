"""OpenAPI conformance (DESIGN 5.8): the contract, not an implementation.

A suite is named `openapi[:<version>][:<tier>][:smoke]`, for example
`openapi:v0`, `openapi:v1:gateway:smoke`. The cases live in the course tree at
`course/conformance/openapi/cases/*.toml`:

    id       = "v0.stream.framing"
    title    = "SSE framing is byte exact"
    versions = ["v0"]                 # API versions whose suite includes it
    tiers    = ["engine", "gateway"]
    smoke    = true                   # part of the `:smoke` subset
    requires = []                     # module ids; until they pass, the case is `pending`
    check    = "stream_framing"       # a named check in CHECKS below
    [params]
    prompt = "Once upon a time"

Every response body is validated against the schema the contract
(`contracts/openapi/openai-subset.<version>.yaml`) declares for that path,
method, status, and content type. SSE chunks validate against the schema in
the 200 `text/event-stream` content's `x-tl-chunk` extension.

From v2 on, a client chooses its version per request: the v2 suite sends
`X-TL-API-Version: 2` on every request, the way a migrated client does, and
the server must answer as v2 whatever its default version is.
"""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, schema, web

VERSIONS = ("v0", "v1", "v2")
TIERS = ("engine", "gateway")


@dataclass
class Suite:
    version: str = "v0"
    tier: str | None = None
    smoke: bool = False

    @property
    def id(self) -> str:
        return "openapi:" + ":".join(
            [self.version, self.tier or "engine"] + (["smoke"] if self.smoke else [])
        )


def parse_suite(name: str, target: str | None = None) -> Suite:
    parts = name.split(":")
    if parts[0] != "openapi":
        raise HarnessError(
            f"unknown suite {name!r}: `ss conform` knows openapi[:v0|v1|v2][:engine|gateway][:smoke]"
        )
    s = Suite()
    for p in parts[1:]:
        if p in VERSIONS:
            s.version = p
        elif p in TIERS:
            s.tier = p
        elif p == "smoke":
            s.smoke = True
        else:
            raise HarnessError(f"suite {name!r}: unknown part {p!r}")
    if target:
        if target not in TIERS:
            raise HarnessError(f"--target must be engine or gateway, not {target!r}")
        if s.tier and s.tier != target:
            raise HarnessError(
                f"suite {name!r} names tier {s.tier} but --target is {target}"
            )
        s.tier = target
    s.tier = s.tier or "engine"
    return s


@dataclass
class Case:
    id: str
    title: str
    versions: list[str]
    tiers: list[str]
    check: str
    smoke: bool = False
    requires: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    upstream: str = (
        ""  # "fake": the gateway must run against the harness's recording upstream
    )


def load_cases(course: Path) -> list[Case]:
    d = course / "conformance" / "openapi" / "cases"
    out = []
    for p in sorted(d.glob("*.toml")) if d.is_dir() else []:
        try:
            raw = tomllib.loads(p.read_text())
        except tomllib.TOMLDecodeError as e:
            raise HarnessError(f"{p}: {e}") from None
        c = Case(
            id=raw.get("id", p.stem),
            title=raw.get("title", ""),
            versions=list(raw.get("versions", [])),
            tiers=list(raw.get("tiers", ["engine"])),
            check=raw.get("check", ""),
            smoke=bool(raw.get("smoke", False)),
            requires=list(raw.get("requires", [])),
            params=dict(raw.get("params", {})),
            upstream=str(raw.get("upstream", "")),
        )
        if c.upstream not in ("", "fake"):
            raise HarnessError(f"{p}: upstream {c.upstream!r} is '' or 'fake'")
        if c.check not in CHECKS:
            raise HarnessError(
                f"{p}: unknown check {c.check!r} (known: {', '.join(sorted(CHECKS))})"
            )
        out.append(c)
    return out


def load_spec(path: Path) -> dict:
    if not path.is_file():
        raise HarnessError(
            f"the contract {path} is missing; this suite needs it vendored (`ss contracts sync`)"
        )
    import yaml

    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise HarnessError(f"{path}: {e}") from None
    if not isinstance(doc, dict) or "paths" not in doc:
        raise HarnessError(f"{path}: not an OpenAPI document (no `paths`)")
    return doc


# ---------------------------------------------------------------------------
# a client bound to one base URL and the contract


@dataclass
class Client:
    base: str
    tier: str
    spec: dict
    model: str = "tracer"
    api_key: str | None = None
    health_base: str | None = None
    timeout: float = 30.0
    fake_upstream: object | None = None  # FakeUpstream when the gateway runs against it
    count_tokens: object | None = (
        None  # messages -> the learner tokenizer's prompt count
    )
    # The gateway cases share one key: a 429 left by the ratelimit case is
    # waited out (Retry-After, a few times) instead of failing an unrelated
    # case. The ratelimit case turns this off to see its own 429s.
    retry_429: bool = True
    # The X-TL-API-Version a migrated client sends (openai-subset.v2.yaml);
    # None sends no header and gets the server's default version.
    api_version: str | None = None

    def headers(self, auth: bool = True, key: str | None = None) -> dict:
        k = key if key is not None else self.api_key
        h = {"X-TL-API-Version": self.api_version} if self.api_version else {}
        if auth and self.tier == "gateway" and k:
            h["Authorization"] = f"Bearer {k}"
        return h

    def call(
        self,
        method: str,
        path: str,
        body=None,
        auth: bool = True,
        base: str | None = None,
        headers: dict | None = None,
        key: str | None = None,
    ) -> web.Response:
        for _ in range(5):
            r = web.request(
                method,
                (base or self.base).rstrip("/") + path,
                json_body=body,
                headers={**self.headers(auth, key), **(headers or {})},
                timeout=self.timeout,
            )
            if r.status != 429 or not self.retry_429:
                return r
            try:
                wait = float(r.headers.get("retry-after", "1"))
            except ValueError:
                wait = 1.0
            time.sleep(min(max(wait, 0.1), 5.0))
        return r

    def chat(self, **kw) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": kw.pop("prompt", "Once upon a time")}
            ],
            "max_tokens": 8,
            "temperature": 0,
            "seed": 0,
            "stream": False,
        }
        body.update(kw)
        return body

    def completion(self, **kw) -> dict:
        body = {
            "model": self.model,
            "prompt": "Once upon a time",
            "max_tokens": 8,
            "temperature": 0,
            "seed": 0,
            "stream": False,
        }
        body.update(kw)
        return body

    # -- schema --------------------------------------------------------------

    def _content(
        self, path: str, method: str, status: int
    ) -> tuple[dict | None, str | None]:
        op = (self.spec.get("paths", {}).get(path) or {}).get(method.lower())
        if op is None:
            return None, f"{method} {path} is not in the contract"
        resps = op.get("responses", {})
        resp = (
            resps.get(str(status))
            or resps.get(status)
            or resps.get(f"{str(status)[0]}XX")
            or resps.get("default")
        )
        if resp is None:
            return None, f"{method} {path}: the contract declares no {status} response"
        if "$ref" in resp:
            resp = schema.resolve_ref(resp["$ref"], self.spec)
        return resp.get("content") or {}, None

    def validate(self, path: str, method: str, r: web.Response) -> list[str]:
        if r.status == 0:
            return [r.error]
        content, err = self._content(path, method, r.status)
        if err:
            return [err]
        if not content:
            return []
        ctype = r.content_type
        if ctype not in content:
            return [
                f"{method} {path} {r.status}: content type {ctype or '(none)'} not in the contract ({', '.join(content)})"
            ]
        media = content[ctype] or {}
        if ctype == "text/event-stream":
            payloads, errs = web.sse_events(r.body)
            chunk = media.get("x-tl-chunk")
            for i, p in enumerate(payloads):
                if p == "[DONE]":
                    continue
                try:
                    obj = json.loads(p)
                except json.JSONDecodeError:
                    errs.append(f"chunk {i}: not JSON: {p[:80]!r}")
                    continue
                if chunk:
                    errs += [
                        f"chunk {i}: {e}"
                        for e in schema.validate(obj, chunk, self.spec)
                    ]
            return errs
        if "schema" not in media:
            return []
        try:
            body = r.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return [f"{method} {path} {r.status}: body is not JSON: {r.text[:120]!r}"]
        return [
            f"{method} {path} {r.status} {e}"
            for e in schema.validate(body, media["schema"], self.spec)
        ]


def _want(r: web.Response, status: int, what: str) -> list[str]:
    if r.status == 0:
        return [r.error]
    if r.status != status:
        return [f"{what}: HTTP {r.status}, want {status}: {r.text[:200]!r}"]
    return []


def _text(r: web.Response) -> str | None:
    try:
        return r.json()["choices"][0]["text"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _stream_text(r: web.Response) -> tuple[str, list[dict], list[str]]:
    payloads, errs = web.sse_events(r.body)
    chunks, out = [], []
    for p in payloads:
        if p == "[DONE]":
            continue
        try:
            obj = json.loads(p)
        except json.JSONDecodeError:
            errs.append(f"chunk not JSON: {p[:80]!r}")
            continue
        chunks.append(obj)
        try:
            out.append(obj["choices"][0].get("text") or "")
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
    return "".join(out), chunks, errs


# ---------------------------------------------------------------------------
# named checks: (client, params) -> errors


def check_healthz(c: Client, p: dict) -> list[str]:
    r = c.call("GET", p.get("path", "/healthz"), auth=False, base=c.health_base)
    return _want(r, 200, "GET /healthz")


def check_completion_schema(c: Client, p: dict) -> list[str]:
    r = c.call(
        "POST",
        "/v1/completions",
        c.completion(prompt=p.get("prompt", "Once upon a time")),
    )
    errs = _want(r, 200, "POST /v1/completions") or c.validate(
        "/v1/completions", "post", r
    )
    if not errs and not isinstance(_text(r), str):
        errs.append("choices[0].text is missing or not a string")
    return errs


def check_greedy_stable(c: Client, p: dict) -> list[str]:
    body = c.completion(
        prompt=p.get("prompt", "The cat"),
        max_tokens=p.get("max_tokens", 16),
        temperature=0,
    )
    a, b = (
        c.call("POST", "/v1/completions", body),
        c.call("POST", "/v1/completions", body),
    )
    errs = _want(a, 200, "first call") + _want(b, 200, "second call")
    if not errs and _text(a) != _text(b):
        errs.append(f"temperature 0 is not stable: {_text(a)!r} then {_text(b)!r}")
    return errs


def check_length(c: Client, p: dict) -> list[str]:
    n = int(p.get("max_tokens", 4))
    r = c.call(
        "POST",
        "/v1/completions",
        c.completion(max_tokens=n, prompt=p.get("prompt", "Hello")),
    )
    errs = _want(r, 200, "POST /v1/completions")
    if errs:
        return errs
    body = r.json()
    ch = (body.get("choices") or [{}])[0]
    if ch.get("finish_reason") != "length":
        errs.append(
            f"max_tokens={n}: finish_reason {ch.get('finish_reason')!r}, want 'length'"
        )
    usage = body.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") != n:
        errs.append(
            f"max_tokens={n}: usage.completion_tokens {usage.get('completion_tokens')!r}, want {n}"
        )
    return errs


def check_stream_framing(c: Client, p: dict) -> list[str]:
    r = c.call(
        "POST",
        "/v1/completions",
        c.completion(stream=True, max_tokens=p.get("max_tokens", 8)),
    )
    errs = _want(r, 200, "POST /v1/completions stream")
    if errs:
        return errs
    if r.content_type != "text/event-stream":
        errs.append(f"content-type {r.content_type!r}, want text/event-stream")
    errs += c.validate("/v1/completions", "post", r)
    _, chunks, _ = _stream_text(r)
    if not chunks:
        errs.append("no data chunks before [DONE]")
    elif (chunks[-1].get("choices") or [{}])[0].get("finish_reason") is None:
        errs.append("the last chunk carries no finish_reason")
    return errs


def check_stream_equals_nonstream(c: Client, p: dict) -> list[str]:
    kw = dict(
        prompt=p.get("prompt", "Once upon a time"),
        max_tokens=p.get("max_tokens", 12),
        temperature=0,
    )
    a = c.call("POST", "/v1/completions", c.completion(**kw))
    b = c.call("POST", "/v1/completions", c.completion(stream=True, **kw))
    errs = _want(a, 200, "non-stream") + _want(b, 200, "stream")
    if errs:
        return errs
    streamed, _, ferrs = _stream_text(b)
    if ferrs:
        return ferrs
    if streamed != _text(a):
        errs.append(f"stream {streamed!r} != non-stream {_text(a)!r} at temperature 0")
    return errs


def check_seed_stable(c: Client, p: dict) -> list[str]:
    body = c.completion(
        prompt=p.get("prompt", "Once"),
        max_tokens=p.get("max_tokens", 12),
        temperature=p.get("temperature", 1.0),
        seed=p.get("seed", 1234),
    )
    a, b = (
        c.call("POST", "/v1/completions", body),
        c.call("POST", "/v1/completions", body),
    )
    errs = _want(a, 200, "first call") + _want(b, 200, "second call")
    if not errs and _text(a) != _text(b):
        errs.append(f"same seed, different output: {_text(a)!r} then {_text(b)!r}")
    return errs


def _error_shape(
    r: web.Response, status: int, typ: str | None, code: str | None
) -> list[str]:
    errs = _want(r, status, f"want {status}")
    if errs:
        return errs
    try:
        e = r.json()["error"]
    except (ValueError, KeyError, TypeError):
        return [f'{status} body is not {{"error": {{...}}}}: {r.text[:200]!r}']
    for k in ("message", "type", "param", "code"):
        if not isinstance(e, dict) or k not in e:
            errs.append(f"{status} error object lacks {k!r}")
    if not errs and typ and e.get("type") != typ:
        errs.append(f"{status} error type {e.get('type')!r}, want {typ!r}")
    if not errs and code and e.get("code") != code:
        errs.append(f"{status} error code {e.get('code')!r}, want {code!r}")
    return errs


def check_error_400(c: Client, p: dict) -> list[str]:
    body = c.completion(**p.get("override", {"temperature": -1}))
    r = c.call("POST", "/v1/completions", body)
    return _error_shape(r, 400, "invalid_request_error", None) or c.validate(
        "/v1/completions", "post", r
    )


def check_auth_401(c: Client, p: dict) -> list[str]:
    r = c.call("POST", "/v1/completions", c.completion(), auth=False)
    return _error_shape(r, 401, "invalid_request_error", "invalid_api_key")


# ---------------------------------------------------------------------------
# API v1 (chat completions) and v2: engine and gateway tiers (DESIGN 5.8)

CHAT = "/v1/chat/completions"


class Pending(Exception):
    """A case that cannot run here (a missing key, a missing client): pending, not failed."""


def _chat_text(r: web.Response) -> str | None:
    try:
        return r.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _chat_stream(r: web.Response) -> tuple[str, list[dict], list[str]]:
    payloads, errs = web.sse_events(r.body)
    chunks, out = [], []
    for p in payloads:
        if p == "[DONE]":
            continue
        try:
            obj = json.loads(p)
        except json.JSONDecodeError:
            errs.append(f"chunk not JSON: {p[:80]!r}")
            continue
        chunks.append(obj)
        try:
            out.append((obj["choices"][0].get("delta") or {}).get("content") or "")
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
    return "".join(out), chunks, errs


def check_models_list(c: Client, p: dict) -> list[str]:
    r = c.call("GET", "/v1/models")
    errs = _want(r, 200, "GET /v1/models") or c.validate("/v1/models", "get", r)
    if errs:
        return errs
    ids = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict)]
    return (
        []
        if c.model in ids
        else [f"/v1/models lists {ids}, not the served model {c.model!r}"]
    )


def check_chat_schema(c: Client, p: dict) -> list[str]:
    r = c.call("POST", CHAT, c.chat(prompt=p.get("prompt", "Once upon a time")))
    errs = _want(r, 200, f"POST {CHAT}") or c.validate(CHAT, "post", r)
    if not errs and not isinstance(_chat_text(r), str):
        errs.append("choices[0].message.content is missing or not a string")
    return errs


def check_chat_greedy_stable(c: Client, p: dict) -> list[str]:
    body = c.chat(prompt=p.get("prompt", "The cat"), max_tokens=p.get("max_tokens", 16))
    a, b = c.call("POST", CHAT, body), c.call("POST", CHAT, body)
    errs = _want(a, 200, "first call") + _want(b, 200, "second call")
    if not errs and _chat_text(a) != _chat_text(b):
        errs.append(
            f"temperature 0 is not stable: {_chat_text(a)!r} then {_chat_text(b)!r}"
        )
    return errs


def check_chat_stream_framing(c: Client, p: dict) -> list[str]:
    r = c.call("POST", CHAT, c.chat(stream=True, max_tokens=p.get("max_tokens", 8)))
    errs = _want(r, 200, f"POST {CHAT} stream")
    if errs:
        return errs
    if r.content_type != "text/event-stream":
        errs.append(f"content-type {r.content_type!r}, want text/event-stream")
    errs += c.validate(CHAT, "post", r)
    _, chunks, ferrs = _chat_stream(r)
    errs += ferrs
    if not chunks:
        return errs + ["no data chunks before [DONE]"]
    first = (chunks[0].get("choices") or [{}])[0].get("delta") or {}
    if first.get("role") != "assistant":
        errs.append(f"the first delta has role {first.get('role')!r}, want 'assistant'")
    if (chunks[-1].get("choices") or [{}])[0].get("finish_reason") is None:
        errs.append("the last chunk carries no finish_reason")
    return errs


def check_chat_stream_equals_nonstream(c: Client, p: dict) -> list[str]:
    kw = dict(
        prompt=p.get("prompt", "Once upon a time"), max_tokens=p.get("max_tokens", 12)
    )
    a = c.call("POST", CHAT, c.chat(**kw))
    b = c.call("POST", CHAT, c.chat(stream=True, **kw))
    errs = _want(a, 200, "non-stream") + _want(b, 200, "stream")
    if errs:
        return errs
    streamed, _, ferrs = _chat_stream(b)
    if ferrs:
        return ferrs
    if streamed != _chat_text(a):
        errs.append(
            f"stream {streamed!r} != non-stream {_chat_text(a)!r} at temperature 0"
        )
    return errs


def check_chat_length(c: Client, p: dict) -> list[str]:
    n = int(p.get("max_tokens", 4))
    r = c.call("POST", CHAT, c.chat(max_tokens=n, prompt=p.get("prompt", "Hello")))
    errs = _want(r, 200, f"POST {CHAT}")
    if errs:
        return errs
    body = r.json()
    ch = (body.get("choices") or [{}])[0]
    if ch.get("finish_reason") != "length":
        errs.append(
            f"max_tokens={n}: finish_reason {ch.get('finish_reason')!r}, want 'length'"
        )
    usage = body.get("usage") or {}
    if usage.get("completion_tokens") != n:
        errs.append(
            f"max_tokens={n}: usage.completion_tokens {usage.get('completion_tokens')!r}, want {n}"
        )
    return errs


def check_chat_stop(c: Client, p: dict) -> list[str]:
    """A stop sequence two tokens long (it spans a token boundary) taken from
    the greedy output: the reply ends just before it, finish_reason stop."""
    prompt = p.get("prompt", "Once upon a time")
    full = c.call(
        "POST", CHAT, c.chat(prompt=prompt, max_tokens=p.get("max_tokens", 24))
    )
    errs = _want(full, 200, "greedy reference")
    if errs:
        return errs
    text = _chat_text(full) or ""
    k = next(
        (i for i in range(1, max(len(text) - 1, 1)) if text.find(text[i : i + 2]) == i),
        None,
    )
    if k is None or len(text) < 3:
        return [f"the greedy reply {text!r} is too short to pick a stop sequence from"]
    stop = text[k : k + 2]
    for stream in (False, True):
        r = c.call(
            "POST",
            CHAT,
            c.chat(
                prompt=prompt,
                max_tokens=p.get("max_tokens", 24),
                stop=[stop],
                stream=stream,
            ),
        )
        errs += _want(r, 200, f"stop={stop!r} stream={stream}")
        if errs:
            return errs
        if stream:
            got, chunks, ferrs = _chat_stream(r)
            errs += ferrs
            fin = (
                (chunks[-1].get("choices") or [{}])[0].get("finish_reason")
                if chunks
                else None
            )
        else:
            got = _chat_text(r)
            fin = (r.json().get("choices") or [{}])[0].get("finish_reason")
        if got != text[:k]:
            errs.append(
                f"stream={stream}: stop {stop!r}: got {got!r}, want {text[:k]!r} (the stop text excluded)"
            )
        if fin != "stop":
            errs.append(f"stream={stream}: finish_reason {fin!r}, want 'stop'")
    return errs


def check_chat_usage(c: Client, p: dict) -> list[str]:
    msgs = [{"role": "user", "content": p.get("prompt", "Count my tokens, please.")}]
    r = c.call("POST", CHAT, c.chat(messages=msgs, max_tokens=p.get("max_tokens", 5)))
    errs = _want(r, 200, f"POST {CHAT}")
    if errs:
        return errs
    u = r.json().get("usage") or {}
    pt, ct, tt = (
        u.get("prompt_tokens"),
        u.get("completion_tokens"),
        u.get("total_tokens"),
    )
    if not all(
        isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (pt, ct, tt)
    ):
        return [
            f"usage {u!r}: prompt, completion, and total tokens must be non-negative integers"
        ]
    if tt != pt + ct:
        errs.append(f"usage.total_tokens {tt} != prompt {pt} + completion {ct}")
    if pt == 0:
        errs.append("usage.prompt_tokens is 0 for a non-empty prompt")
    if c.count_tokens is not None:
        want = c.count_tokens(msgs)
        if want is not None and want != pt:
            errs.append(
                f"usage.prompt_tokens {pt}, but your tokenizer counts {want} for the templated prompt"
            )
    return errs


def check_chat_seed(c: Client, p: dict) -> list[str]:
    body = c.chat(
        prompt=p.get("prompt", "Once"),
        max_tokens=p.get("max_tokens", 12),
        temperature=p.get("temperature", 1.0),
        seed=p.get("seed", 1234),
    )
    a, b = c.call("POST", CHAT, body), c.call("POST", CHAT, body)
    errs = _want(a, 200, "first call") + _want(b, 200, "second call")
    if not errs and _chat_text(a) != _chat_text(b):
        errs.append(
            f"same seed, different output: {_chat_text(a)!r} then {_chat_text(b)!r}"
        )
    return errs


def check_chat_error(c: Client, p: dict) -> list[str]:
    status = int(p.get("status", 400))
    r = c.call("POST", CHAT, c.chat(**p.get("override", {"temperature": -1})))
    return _error_shape(
        r, status, p.get("type", "invalid_request_error"), p.get("code")
    ) or c.validate(CHAT, "post", r)


def _metric(c: Client, name: str) -> float | None:
    if not c.health_base:
        return None
    r = web.get(c.health_base.rstrip("/") + "/metrics", timeout=5)
    if r.status != 200:
        return None
    total, seen = 0.0, False
    for line in r.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        head, _, val = line.rpartition(" ")
        if head.split("{", 1)[0] == name:
            try:
                total += float(val)
                seen = True
            except ValueError:
                pass
    return total if seen else None


def check_cancel_disconnect(c: Client, p: dict) -> list[str]:
    """Close the client socket mid-stream; within 2 s the engine's
    tl.engine.active_sequences (Prometheus name below) is back to baseline."""
    import http.client
    import socket
    import urllib.parse

    name = p.get("metric", "tl_engine_active_sequences")
    if not c.health_base:
        return ["no health port to scrape /metrics from (pass --health-base)"]
    base = _metric(c, name)
    if base is None:
        return [f"{c.health_base}/metrics has no {name}"]
    u = urllib.parse.urlsplit(c.base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    body = json.dumps(
        c.chat(
            stream=True,
            max_tokens=int(p.get("max_tokens", 2000)),
            prompt=p.get("prompt", "Once upon a time"),
        )
    )
    conn.request(
        "POST",
        CHAT,
        body=body,
        headers={"Content-Type": "application/json", **c.headers()},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        return [f"stream request: HTTP {resp.status}"]
    # resp.readline() decodes chunked transfer encoding; the raw socket file
    # would hand back the chunk-size line instead of the first event.
    first = resp.readline()
    while first in (b"\r\n", b"\n"):
        first = resp.readline()
    peak = _metric(c, name) or 0.0
    sock = conn.sock
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)  # the client hangs up mid-stream
        except OSError:
            pass
    conn.close()
    if not first.startswith(b"data:"):
        return [f"the stream's first line is {first[:80]!r}"]
    if peak <= base:
        return [
            f"{name} did not rise during the stream ({base:g} before, {peak:g} during)"
        ]
    deadline = time.monotonic() + float(p.get("within_s", 2.0))
    while time.monotonic() < deadline:
        v = _metric(c, name)
        if v is not None and v <= base:
            return []
        time.sleep(0.05)
    return [
        f"{name} stayed at {_metric(c, name)} for {p.get('within_s', 2.0)}s after the client left (baseline {base:g})"
    ]


def check_concurrency(c: Client, p: dict) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor

    n = int(p.get("n", 16))
    prompts = [f"{p.get('prompt', 'Once upon a time')} {i}" for i in range(n)]
    serial = []
    for pr in prompts:
        r = c.call("POST", CHAT, c.chat(prompt=pr, max_tokens=p.get("max_tokens", 12)))
        if r.status != 200:
            return [f"serial run: HTTP {r.status}"]
        serial.append(_chat_text(r))

    def one(pr):
        return c.call(
            "POST",
            CHAT,
            c.chat(prompt=pr, max_tokens=p.get("max_tokens", 12), stream=True),
        )

    with ThreadPoolExecutor(max_workers=n) as ex:
        rs = list(ex.map(one, prompts))
    errs = []
    for i, (r, want) in enumerate(zip(rs, serial)):
        if r.status != 200:
            errs.append(f"stream {i}: HTTP {r.status} {r.error}")
            continue
        got, _, ferrs = _chat_stream(r)
        if ferrs:
            errs.append(f"stream {i}: {ferrs[0]}")
        elif got != want:
            errs.append(
                f"stream {i}: {got!r} != the serial run {want!r} (batch-invariant kernels, 2.4)"
            )
    return errs


# The city is an enum: a free string under a grammar only stays valid, and a
# random PR model can fill max_tokens without ever closing it; a closed set
# lets the constraint finish the call inside the budget (any model, any seed).
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "enum": ["Paris", "London", "Tokyo"]}
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def _tool_call_errs(calls, tools: list[dict]) -> list[str]:
    if not isinstance(calls, list) or not calls:
        return ["no tool_calls in the reply"]
    by = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    errs = []
    for i, tc in enumerate(calls):
        fn = (tc or {}).get("function") or {}
        if fn.get("name") not in by:
            errs.append(
                f"tool_calls[{i}] names {fn.get('name')!r}, not one of {sorted(by)}"
            )
            continue
        try:
            args = json.loads(fn.get("arguments") or "")
        except json.JSONDecodeError:
            errs.append(
                f"tool_calls[{i}].function.arguments is not JSON: {fn.get('arguments')!r}"
            )
            continue
        errs += [
            f"tool_calls[{i}] arguments {e}"
            for e in schema.validate(args, by[fn["name"]])
        ]
        if not (tc or {}).get("id"):
            errs.append(f"tool_calls[{i}] has no id")
    return errs


def check_tools_call(c: Client, p: dict) -> list[str]:
    r = c.call(
        "POST",
        CHAT,
        c.chat(
            prompt=p.get("prompt", "What is the weather in Paris?"),
            tools=[TOOL],
            tool_choice=p.get("tool_choice", "required"),
            max_tokens=p.get("max_tokens", 64),
        ),
    )
    errs = _want(r, 200, "tool request") or c.validate(CHAT, "post", r)
    if errs:
        return errs
    ch = r.json()["choices"][0]
    errs = _tool_call_errs((ch.get("message") or {}).get("tool_calls"), [TOOL])
    if ch.get("finish_reason") != "tool_calls":
        errs.append(f"finish_reason {ch.get('finish_reason')!r}, want 'tool_calls'")
    return errs


def check_tools_stream(c: Client, p: dict) -> list[str]:
    r = c.call(
        "POST",
        CHAT,
        c.chat(
            prompt=p.get("prompt", "What is the weather in Paris?"),
            tools=[TOOL],
            tool_choice=p.get("tool_choice", "required"),
            stream=True,
            max_tokens=p.get("max_tokens", 64),
        ),
    )
    errs = _want(r, 200, "tool stream")
    if errs:
        return errs
    _, chunks, ferrs = _chat_stream(r)
    if ferrs:
        return ferrs
    calls: dict[int, dict] = {}
    for ch in chunks:
        for d in ((ch.get("choices") or [{}])[0].get("delta") or {}).get(
            "tool_calls"
        ) or []:
            slot = calls.setdefault(
                int(d.get("index", 0)),
                {"id": None, "function": {"name": "", "arguments": ""}},
            )
            slot["id"] = slot["id"] or d.get("id")
            fn = d.get("function") or {}
            slot["function"]["name"] += fn.get("name") or ""
            slot["function"]["arguments"] += fn.get("arguments") or ""
    return _tool_call_errs([calls[i] for i in sorted(calls)], [TOOL])


def check_tools_choice(c: Client, p: dict) -> list[str]:
    forced = c.call(
        "POST",
        CHAT,
        c.chat(
            prompt=p.get("prompt", "Say hello."),
            tools=[TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_tokens=64,
        ),
    )
    errs = _want(forced, 200, "tool_choice forced")
    if not errs:
        calls = (forced.json()["choices"][0].get("message") or {}).get("tool_calls")
        errs += [f"forced: {e}" for e in _tool_call_errs(calls, [TOOL])]
    none = c.call(
        "POST",
        CHAT,
        c.chat(
            prompt=p.get("weather_prompt", "What is the weather in Paris?"),
            tools=[TOOL],
            tool_choice="none",
            max_tokens=16,
        ),
    )
    errs += _want(none, 200, "tool_choice none")
    if none.status == 200 and (none.json()["choices"][0].get("message") or {}).get(
        "tool_calls"
    ):
        errs.append("tool_choice none: the reply still calls a tool")
    return errs


# -- gateway tier --------------------------------------------------------------


def check_auth_401_chat(c: Client, p: dict) -> list[str]:
    errs = _error_shape(
        c.call("POST", CHAT, c.chat(), auth=False),
        401,
        "invalid_request_error",
        "invalid_api_key",
    )
    errs += [
        f"bad key: {e}"
        for e in _error_shape(
            c.call("POST", CHAT, c.chat(), key="tl_bogus_key"),
            401,
            "invalid_request_error",
            "invalid_api_key",
        )
    ]
    return errs


def check_auth_403(c: Client, p: dict) -> list[str]:
    import os

    envname = p.get("key_env", "TL_API_KEY_NOSCOPE")
    key = os.environ.get(envname)
    if not key:
        raise Pending(f"set ${envname} to a valid key without the chat scope")
    return _error_shape(
        c.call("POST", CHAT, c.chat(), key=key),
        403,
        p.get("type", "permission_error"),
        p.get("code"),
    )


def check_ratelimit_429(c: Client, p: dict) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor

    n = int(p.get("burst", 64))
    c.retry_429 = False
    try:
        with ThreadPoolExecutor(max_workers=min(n, 32)) as ex:
            rs = list(
                ex.map(lambda _: c.call("POST", CHAT, c.chat(max_tokens=1)), range(n))
            )
    finally:
        c.retry_429 = True
    limited = [r for r in rs if r.status == 429]
    if not limited:
        return [
            f"{n} requests at once never got a 429 ({sorted({r.status for r in rs})})"
        ]
    r = limited[0]
    errs = _error_shape(r, 429, "rate_limit_error", None)
    if "retry-after" not in r.headers:
        errs.append("429 without Retry-After")
    if not any(h.startswith("x-ratelimit-") for h in r.headers):
        errs.append("429 without x-ratelimit-* headers")
    try:  # leave the limiter as found: the cases after this one share the key
        time.sleep(min(float(r.headers.get("retry-after", "1")), 5.0))
    except ValueError:
        time.sleep(1.0)
    return errs


def check_route_model(c: Client, p: dict) -> list[str]:
    r = c.call("POST", CHAT, c.chat())
    errs = _want(r, 200, f"model {c.model!r}")
    if not errs and r.json().get("model") != c.model:
        errs.append(
            f"asked for {c.model!r}, the reply says model {r.json().get('model')!r}"
        )
    bad = c.call("POST", CHAT, c.chat(model=p.get("unknown", "no-such-model")))
    errs += [
        f"unknown model: {e}"
        for e in _error_shape(bad, 404, "invalid_request_error", "model_not_found")
    ]
    return errs


def check_cache_hit(c: Client, p: dict) -> list[str]:
    body = c.chat(prompt=p.get("prompt", "cache me"), max_tokens=p.get("max_tokens", 8))
    a = c.call("POST", CHAT, body)
    b = c.call("POST", CHAT, body)
    errs = _want(a, 200, "first call") + _want(b, 200, "second call")
    if errs:
        return errs
    if b.headers.get("x-tl-cache") != "hit":
        errs.append(
            f"the identical temperature-0 request got X-TL-Cache {b.headers.get('x-tl-cache')!r}, want 'hit'"
        )
    if _chat_text(a) != _chat_text(b):
        errs.append("a cache hit changed the reply")
    return errs


def check_priority_internal(c: Client, p: dict) -> list[str]:
    up = c.fake_upstream
    up.clear()
    r = c.call("POST", CHAT, c.chat(), headers={"X-TL-Priority": "forged-by-client"})
    errs = _want(r, 200, "request through the gateway")
    if errs:
        return errs
    seen = [
        h.get("x-tl-priority") for path, h, _ in up.requests if path.startswith("/v1/")
    ]
    if not seen:
        return ["the gateway never reached the upstream"]
    if seen[-1] == "forged-by-client":
        errs.append(
            "the client's X-TL-Priority reached the engine (it must be stripped)"
        )
    elif not seen[-1]:
        errs.append("the gateway did not set X-TL-Priority from the key's tier")
    return errs


def check_policy_451(c: Client, p: dict) -> list[str]:
    r = c.call(
        "POST",
        CHAT,
        c.chat(prompt=p.get("prompt", "Explain how to make a weapon at home.")),
    )
    # openai-subset PolicyBlocked: `policy_error` / `usage_policy`.
    return _error_shape(
        r, 451, p.get("type", "policy_error"), p.get("code", "usage_policy")
    )


SDK_SCRIPT = r"""
import json, sys
from openai import OpenAI
base, key, model = sys.argv[1], sys.argv[2], sys.argv[3]
c = OpenAI(base_url=base, api_key=key or "none", max_retries=0, timeout=30)
r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hello"}], max_tokens=8, temperature=0)
text = r.choices[0].message.content
parts = [ch.choices[0].delta.content or "" for ch in c.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Hello"}], max_tokens=8, temperature=0, stream=True) if ch.choices]
print(json.dumps({"text": text, "stream": "".join(parts)}))
"""


def check_openai_sdk(c: Client, p: dict) -> list[str]:
    import subprocess

    base = c.base.rstrip("/") + "/v1"
    for offline in (["--offline"], []):
        try:
            proc = subprocess.run(
                [
                    "uv",
                    "run",
                    *offline,
                    "--no-project",
                    "--quiet",
                    "--with",
                    "openai",
                    "python",
                    "-c",
                    SDK_SCRIPT,
                    base,
                    c.api_key or "",
                    c.model,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise Pending(f"cannot run the openai client: {e}") from None
        if proc.returncode == 0 or "openai" not in (proc.stderr or "") or not offline:
            break
    if proc.returncode != 0 and "No solution found" in (proc.stderr or ""):
        raise Pending(
            "the openai package is not available to uv (offline and no index)"
        )
    if proc.returncode != 0:
        return [f"the openai client failed:\n{(proc.stderr or proc.stdout)[-1500:]}"]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if not isinstance(out.get("text"), str):
        return ["the openai client got no message content"]
    if out["stream"] != out["text"]:
        return [
            f"the openai client's stream {out['stream']!r} != its non-stream {out['text']!r}"
        ]
    return []


def check_contract_paths(c: Client, p: dict) -> list[str]:
    """Every operation the contract marks with `x-tl-probe` (a request it
    must answer) answers with the probe's status, and its body validates.
    Run against the v2 contract, this is v1 and v2 side by side (ops.05)."""
    errs, n = [], 0
    for path, ops in (c.spec.get("paths") or {}).items():
        for method, op in (ops or {}).items():
            if not isinstance(op, dict) or "x-tl-probe" not in op:
                continue
            n += 1
            probe = op["x-tl-probe"] or {}
            body = probe.get("body")
            if isinstance(body, dict) and "model" in body:
                body = {**body, "model": c.model}
            r = c.call(method.upper(), path, body)
            want = int(probe.get("status", 200))
            e = _want(r, want, f"{method.upper()} {path}") or c.validate(
                path, method, r
            )
            errs += e
    return errs if n else ["the contract marks no operation with x-tl-probe"]


V1_CHECKS = {
    "models_list": check_models_list,
    "chat_schema": check_chat_schema,
    "chat_greedy_stable": check_chat_greedy_stable,
    "chat_stream_framing": check_chat_stream_framing,
    "chat_stream_equals_nonstream": check_chat_stream_equals_nonstream,
    "chat_length": check_chat_length,
    "chat_stop": check_chat_stop,
    "chat_usage": check_chat_usage,
    "chat_seed": check_chat_seed,
    "chat_error": check_chat_error,
    "cancel_disconnect": check_cancel_disconnect,
    "concurrency": check_concurrency,
    "tools_call": check_tools_call,
    "tools_stream": check_tools_stream,
    "tools_choice": check_tools_choice,
    "auth_401_chat": check_auth_401_chat,
    "auth_403": check_auth_403,
    "ratelimit_429": check_ratelimit_429,
    "route_model": check_route_model,
    "cache_hit": check_cache_hit,
    "priority_internal": check_priority_internal,
    "policy_451": check_policy_451,
    "openai_sdk": check_openai_sdk,
    "contract_paths": check_contract_paths,
}


# ---------------------------------------------------------------------------
# the recording upstream (priority.internal): a fake engine the gateway proxies to


class FakeUpstream:
    """A minimal OpenAI-subset engine on 127.0.0.1 that records every request
    (path, lower-cased headers, body) and answers deterministically."""

    def __init__(self, model: str = "tracer"):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.model = model
        self.requests: list[tuple[str, dict, bytes]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, status, obj, ctype="application/json"):
                data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append(
                    (self.path, {k.lower(): v for k, v in self.headers.items()}, b"")
                )
                if self.path in ("/healthz", "/readyz"):
                    return self._send(200, b"ok\n", "text/plain")
                if self.path == "/metrics":
                    return self._send(
                        200, b"tl_engine_active_sequences 0\n", "text/plain"
                    )
                if self.path == "/v1/models":
                    return self._send(
                        200,
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": outer.model,
                                    "object": "model",
                                    "created": 0,
                                    "owned_by": "ss",
                                }
                            ],
                        },
                    )
                return self._send(
                    404,
                    {
                        "error": {
                            "message": "not found",
                            "type": "invalid_request_error",
                            "param": None,
                            "code": "not_found",
                        }
                    },
                )

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                outer.requests.append(
                    (self.path, {k.lower(): v for k, v in self.headers.items()}, body)
                )
                try:
                    req = json.loads(body or b"{}")
                except json.JSONDecodeError:
                    req = {}
                model = req.get("model", outer.model)
                if req.get("stream"):
                    chunks = [
                        {
                            "id": "chatcmpl-up",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": "ok"},
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            "id": "chatcmpl-up",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                        },
                    ]
                    data = (
                        "".join(f"data: {json.dumps(x)}\n\n" for x in chunks)
                        + "data: [DONE]\n\n"
                    )
                    return self._send(200, data.encode(), "text/event-stream")
                if self.path == "/v1/completions":
                    return self._send(
                        200,
                        {
                            "id": "cmpl-up",
                            "object": "text_completion",
                            "created": 0,
                            "model": model,
                            "choices": [
                                {"index": 0, "text": "ok", "finish_reason": "stop"}
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )
                return self._send(
                    200,
                    {
                        "id": "chatcmpl-up",
                        "object": "chat.completion",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def clear(self) -> None:
        self.requests.clear()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


CHECKS = {
    "healthz": check_healthz,
    "completion_schema": check_completion_schema,
    "greedy_stable": check_greedy_stable,
    "length": check_length,
    "stream_framing": check_stream_framing,
    "stream_equals_nonstream": check_stream_equals_nonstream,
    "seed_stable": check_seed_stable,
    "error_400": check_error_400,
    "auth_401": check_auth_401,
    **V1_CHECKS,
}


@dataclass
class Result:
    case: str
    status: str  # pass fail pending
    detail: str = ""


def run(
    client: Client,
    cases: list[Case],
    suite: Suite,
    passed=lambda mid: True,
    overrides: dict | None = None,
) -> list[Result]:
    """overrides: {case id: params} merged over a case's [params] (a milestone
    step's `params`, e.g. the nightly instruct model choosing its own tool)."""
    out: list[Result] = []
    for c in cases:
        if (
            suite.version not in c.versions
            or client.tier not in c.tiers
            or (suite.smoke and not c.smoke)
        ):
            continue
        waiting = [m for m in c.requires if not passed(m)]
        if c.upstream == "fake" and client.fake_upstream is None:
            out.append(
                Result(
                    c.id,
                    "pending",
                    "needs the gateway running against the harness's recording upstream "
                    "(`ss conform` without --base starts it that way)",
                )
            )
            continue
        if c.upstream != "fake" and client.fake_upstream is not None:
            continue  # the fake-upstream pass runs only the cases that need it
        if c.check == "healthz" and not client.health_base:
            out.append(
                Result(
                    c.id,
                    "pending",
                    "no health endpoint to call: pass --health-base URL of the health "
                    "port (kind runs: set [deploy].gateway_health_url)",
                )
            )
            continue
        if waiting:
            out.append(Result(c.id, "pending", f"requires {', '.join(waiting)}"))
            continue
        try:
            params = {**c.params, **((overrides or {}).get(c.id) or {})}
            errs = CHECKS[c.check](client, params)
        except Pending as e:
            out.append(Result(c.id, "pending", str(e)))
            continue
        except (
            Exception
        ) as e:  # a malformed response must fail the case, not crash the suite
            errs = [f"{type(e).__name__}: {e}"]
        out.append(Result(c.id, "fail" if errs else "pass", "\n".join(errs)))
    return out
