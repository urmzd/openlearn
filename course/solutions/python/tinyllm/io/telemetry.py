"""OTLP/HTTP export from TRACEPARENT (dur.09, the D9 exception).

A Python subprocess activity has one link to the platform besides its files:
the OpenTelemetry collector. The worker passes the activity span's W3C
context as TRACEPARENT; every span made here continues that trace, so the
Python spans of a TrainRun hang under `activity train` in the same trace as
the CLI call that started it (otel/semconv.md, obs.05).

The exporter is hand-written OTLP JSON over urllib (the SDK is allowed too,
allowed-deps.toml): the format is small, and writing it once shows exactly
what a span is on the wire. Export failures are swallowed: telemetry must
never fail the training step it observes.

Contract: contracts/py/tinyllm/io/telemetry.pyi.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Iterator, Mapping, Optional, Sequence

_HEX = re.compile(r"^[0-9a-f]+$")


class SpanContext:
    def __init__(self, trace_id: str, span_id: str, sampled: bool) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.sampled = bool(sampled)

    def __repr__(self) -> str:
        return f"SpanContext({self.trace_id}, {self.span_id}, sampled={self.sampled})"


def _hex(s: str, n: int) -> bool:
    return len(s) == n and bool(_HEX.match(s)) and s != "0" * n


def parse_traceparent(value: Optional[str]) -> Optional[SpanContext]:
    if not value:
        return None
    parts = value.strip().split("-")
    if len(parts) < 4:
        return None
    version, trace_id, span_id, flags = parts[:4]
    if not (len(version) == 2 and _HEX.match(version)) or version == "ff":
        return None
    if version == "00" and len(parts) != 4:
        return None
    if not (_hex(trace_id, 32) and _hex(span_id, 16)):
        return None
    if not (len(flags) == 2 and _HEX.match(flags)):
        return None
    return SpanContext(trace_id, span_id, bool(int(flags, 16) & 1))


def format_traceparent(ctx: SpanContext) -> str:
    return f"00-{ctx.trace_id}-{ctx.span_id}-{'01' if ctx.sampled else '00'}"


class Span:
    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str,
        start_ns: int,
        attributes: Optional[Mapping[str, object]],
        sampled: bool,
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.start_ns = start_ns
        self.end_ns = 0
        self.attributes: dict[str, object] = dict(attributes or {})
        self.status_error: Optional[str] = None
        self._sampled = sampled

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_error(self, error_type: str) -> None:
        self.status_error = error_type
        self.attributes["error.type"] = error_type

    def context(self) -> SpanContext:
        return SpanContext(self.trace_id, self.span_id, self._sampled)


def _value(v: object) -> dict:
    # bool before int: bool is a subclass of int in Python.
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _attrs(d: Mapping[str, object]) -> list:
    return [{"key": k, "value": _value(v)} for k, v in d.items()]


def _resource(service_name: str, resource: Optional[Mapping[str, object]]) -> dict:
    return {"attributes": _attrs({"service.name": service_name, **dict(resource or {})})}


def otlp_traces_body(
    spans: Sequence[Span], service_name: str, resource: Optional[Mapping[str, object]] = None
) -> dict:
    out = []
    for s in spans:
        span = {
            "traceId": s.trace_id,
            "spanId": s.span_id,
            "name": s.name,
            "kind": 1,
            "startTimeUnixNano": str(s.start_ns),
            "endTimeUnixNano": str(s.end_ns),
            "attributes": _attrs(s.attributes),
            "status": {"code": 2} if s.status_error else {},
        }
        if s.parent_span_id:
            span["parentSpanId"] = s.parent_span_id
        out.append(span)
    return {
        "resourceSpans": [
            {
                "resource": _resource(service_name, resource),
                "scopeSpans": [{"scope": {"name": "tinyllm"}, "spans": out}],
            }
        ]
    }


def should_sample_step(step: int, every: int = 50) -> bool:
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    return step % every == 0


class Tracer:
    def __init__(
        self,
        endpoint: Optional[str],
        service_name: str,
        parent: Optional[SpanContext] = None,
        resource: Optional[Mapping[str, object]] = None,
        timeout_s: float = 2.0,
        ids: Callable[[int], bytes] = os.urandom,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.endpoint = (endpoint or "").rstrip("/")
        self.service_name = service_name
        self.parent = parent
        self.resource = dict(resource or {})
        self.timeout_s = timeout_s
        self._ids = ids
        self._clock_ns = clock_ns
        # A remote parent decided whether this trace is recorded; a new root
        # trace is.
        self.sampled = parent.sampled if parent is not None else True
        self._trace_id = parent.trace_id if parent is not None else self._new_id(16)
        self._stack: list[Span] = []
        self._queue: list[Span] = []
        self._gauges: dict[tuple, tuple] = {}
        self._closed = False

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Tracer":
        env = os.environ if env is None else env
        resource: dict[str, object] = {}
        for pair in (env.get("OTEL_RESOURCE_ATTRIBUTES") or "").split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                if k.strip():
                    resource[k.strip()] = v.strip()
        return cls(
            env.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None,
            env.get("OTEL_SERVICE_NAME") or "tinyllm-python",
            parse_traceparent(env.get("TRACEPARENT")),
            resource,
        )

    def _new_id(self, n: int) -> str:
        while True:
            b = self._ids(n)
            if any(b):
                return b.hex()

    def start_span(
        self,
        name: str,
        attributes: Optional[Mapping[str, object]] = None,
        parent: Optional[Span] = None,
    ) -> Span:
        if parent is None and self._stack:
            parent = self._stack[-1]
        if parent is not None:
            parent_id = parent.span_id
        elif self.parent is not None:
            parent_id = self.parent.span_id
        else:
            parent_id = ""
        return Span(name, self._trace_id, self._new_id(8), parent_id, self._clock_ns(), attributes, self.sampled)

    def end_span(self, span: Span) -> None:
        span.end_ns = max(self._clock_ns(), span.start_ns)
        if self.sampled and not self._closed:
            self._queue.append(span)

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Mapping[str, object]] = None,
        parent: Optional[Span] = None,
    ) -> Iterator[Span]:
        s = self.start_span(name, attributes, parent)
        self._stack.append(s)
        try:
            yield s
        except BaseException as e:
            s.set_error(type(e).__name__)
            raise
        finally:
            self._stack.remove(s)
            self.end_span(s)

    def gauge(
        self, name: str, value: float, attributes: Optional[Mapping[str, object]] = None
    ) -> None:
        attrs = dict(attributes or {})
        self._gauges[(name, tuple(sorted(attrs.items())))] = (float(value), attrs, self._clock_ns())

    def queued(self) -> Sequence[Span]:
        return list(self._queue)

    def _post(self, path: str, body: dict) -> bool:
        req = urllib.request.Request(
            self.endpoint + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _metrics_body(self) -> dict:
        metrics = []
        for (name, _), (value, attrs, ts) in sorted(self._gauges.items(), key=lambda kv: kv[0][0]):
            metrics.append(
                {
                    "name": name,
                    "gauge": {"dataPoints": [{"asDouble": value, "timeUnixNano": str(ts), "attributes": _attrs(attrs)}]},
                }
            )
        return {
            "resourceMetrics": [
                {
                    "resource": _resource(self.service_name, self.resource),
                    "scopeMetrics": [{"scope": {"name": "tinyllm"}, "metrics": metrics}],
                }
            ]
        }

    def flush(self) -> bool:
        spans, self._queue = self._queue, []
        gauges = bool(self._gauges) and self.sampled
        if not self.endpoint or (not spans and not gauges):
            return True
        ok = True
        if spans:
            ok &= self._post("/v1/traces", otlp_traces_body(spans, self.service_name, self.resource))
        if gauges:
            ok &= self._post("/v1/metrics", self._metrics_body())
            self._gauges.clear()
        return ok

    def shutdown(self) -> bool:
        ok = self.flush()
        self._closed = True
        return ok
