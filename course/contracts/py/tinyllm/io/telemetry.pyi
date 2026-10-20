# contracts/py/tinyllm/io/telemetry.pyi (dur.09): OTLP/HTTP export from TRACEPARENT
# chapter: content/courses/systems/07-durable-orchestration/09-subprocess-activities.md
#
# The D9 exception: a Python subprocess activity talks to nothing on the
# platform except the OpenTelemetry collector. The worker hands it the
# activity span's W3C context in TRACEPARENT (spec/subprocess-activity.md),
# and this module makes the Python spans (train.run, train.step every 50th
# step, train.checkpoint, corpus.stage <s>: otel/semconv.md) children of it,
# so one TrainRun is one trace from the CLI to the sampled steps (obs.05).
#
# Wire format: OTLP/HTTP with a JSON body, POST <endpoint>/v1/traces and
# <endpoint>/v1/metrics, Content-Type application/json, as the OTLP
# specification's JSON encoding defines it: trace and span ids as lowercase
# hex strings, times as decimal strings of Unix nanoseconds, attributes as
# [{"key": k, "value": {"stringValue" | "intValue" | "doubleValue" |
# "boolValue": v}}] (intValue is a decimal string), span kind 1 (INTERNAL).
#
# Telemetry must never fail the work it observes: no function here raises
# because the collector is down, slow, or answers an error.
from typing import Callable, ContextManager, Mapping, Optional, Sequence

class SpanContext:
    trace_id: str  # 32 lowercase hex digits, not all zero
    span_id: str  # 16 lowercase hex digits, not all zero
    sampled: bool  # trace-flags bit 0
    def __init__(self, trace_id: str, span_id: str, sampled: bool) -> None: ...

def parse_traceparent(value: Optional[str]) -> Optional[SpanContext]:
    """W3C trace-context `traceparent`: "<version>-<trace-id>-<parent-id>-<flags>"
    with version 00 (exactly four fields) or a later version (four or more;
    the first four are read), all lowercase hex. None when value is None,
    empty, malformed, version ff, or either id is all zeros: the caller then
    starts a new trace."""

def format_traceparent(ctx: SpanContext) -> str:
    """"00-<trace_id>-<span_id>-<01 when sampled else 00>"."""

class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str  # "" for a root span
    start_ns: int
    end_ns: int  # 0 while open
    attributes: dict[str, object]  # str, bool, int, or float values
    status_error: Optional[str]  # error.type when the span failed, else None
    def set_attribute(self, key: str, value: object) -> None: ...
    def set_error(self, error_type: str) -> None:
        """Status ERROR with attribute error.type = error_type."""
    def context(self) -> SpanContext: ...

class Tracer:
    service_name: str
    parent: Optional[SpanContext]  # the remote parent (TRACEPARENT), if any
    sampled: bool  # False: spans are created but never exported
    def __init__(
        self,
        endpoint: Optional[str],
        service_name: str,
        parent: Optional[SpanContext] = None,
        resource: Optional[Mapping[str, object]] = None,
        timeout_s: float = 2.0,
        ids: Callable[[int], bytes] = ...,
        clock_ns: Callable[[], int] = ...,
    ) -> None:
        """endpoint is the collector base URL ("http://host:4318"); None or ""
        exports nothing. A root trace (no parent) is sampled; otherwise the
        parent's sampled flag decides. ids(n) returns n random bytes (default
        os.urandom; an all-zero id is drawn again); clock_ns defaults to
        time.time_ns. resource adds attributes beside service.name."""
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Tracer":
        """From TRACEPARENT, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME
        (default "tinyllm-python"), and OTEL_RESOURCE_ATTRIBUTES
        ("k=v,k2=v2"), reading os.environ when env is None."""
    def start_span(
        self,
        name: str,
        attributes: Optional[Mapping[str, object]] = None,
        parent: Optional[Span] = None,
    ) -> Span:
        """A new span: child of parent when given, else of the innermost open
        span() of this tracer, else of the remote parent, else a new root."""
    def end_span(self, span: Span) -> None:
        """Set end_ns and queue the span for export (when sampled)."""
    def span(
        self,
        name: str,
        attributes: Optional[Mapping[str, object]] = None,
        parent: Optional[Span] = None,
    ) -> ContextManager[Span]:
        """start_span, yield, end_span. An exception inside marks the span
        with set_error(<exception class name>) and propagates."""
    def gauge(
        self, name: str, value: float, attributes: Optional[Mapping[str, object]] = None
    ) -> None:
        """Record the latest value of a gauge (otel/metrics.yaml names such as
        tl.train.loss); exported on flush."""
    def queued(self) -> Sequence[Span]:
        """Spans ended and not yet exported, in end order."""
    def flush(self) -> bool:
        """POST the queued spans to <endpoint>/v1/traces and the gauges to
        <endpoint>/v1/metrics, one request each, within timeout_s. True when
        there was nothing to send or every POST answered 2xx; False otherwise
        (the spans are dropped, never retried forever). Never raises."""
    def shutdown(self) -> bool:
        """flush(); later spans are not exported."""

def should_sample_step(step: int, every: int = 50) -> bool:
    """train.step spans are kept for every `every`-th step: step % every == 0.
    ValueError for every < 1."""

def otlp_traces_body(
    spans: Sequence[Span], service_name: str, resource: Optional[Mapping[str, object]] = None
) -> dict:
    """The OTLP/HTTP JSON request body for spans: one resourceSpans entry
    whose resource carries service.name and the resource attributes, one
    scopeSpans entry with scope name "tinyllm", the spans in order."""
