# Python helper and exporter

This part of [Subprocess activities: the Go runner and the Python helper](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Connect application progress/checkpoints to durable orchestration and export.

## Building the Python helper

The Python half is the other side of the same agreement: write events the runner can tail, publish outputs atomically, and turn exceptions into exit codes.

### One event, one line

`emit` validates before it writes, so a rejected event leaves the file untouched, and it writes one complete line and then flushes (Listing 12).

```python
    def emit(self, kind: str, **fields: object) -> None:
        if kind not in _KINDS:
            raise ValueError(f"progress kind {kind!r} is not one of {_KINDS}")
        for k, v in fields.items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"progress field {k}={v} is not finite (JSON has no NaN)")
        if self._progress is None:
            os.makedirs(self.work_dir, exist_ok=True)
            self._progress = open(self.progress_path, "a", encoding="utf-8")
        event = {"ts": self._clock(), "kind": kind, **fields}
        # One write of one complete line, then flush: the runner tails the
        # file and must never parse half an event.
        self._progress.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
        self._progress.flush()
```

`separators=(",", ":")` is what makes the worked example's lines byte for byte `{"ts":1760000000.0,"kind":"step",...}`.

### Checkpoints and outputs

A checkpoint path is made relative to `TL_ARTIFACTS`, because a path that is right on one pod must be right on the next, and it is announced only after `verify_step_dir` finds nothing wrong. Outputs are published through a temp file, fsync, rename, and a directory fsync (Listing 13).

```python
    def checkpoint(self, step: int, path: str) -> str:
        rel = self._artifact_path(path)  # relative, and never outside TL_ARTIFACTS
        problems = verify_step_dir(os.path.join(self.artifacts, rel))
        if problems:
            raise ValueError(f"checkpoint {rel} is not complete: " + "; ".join(problems))
        self.emit("ckpt", step=int(step), ckpt=rel)  # only now may it become a resume point
        return rel

    def publish(self, name: str, data: bytes) -> str:
        final = self.output(name)  # rejects absolute names and anything outside the work dir
        os.makedirs(os.path.dirname(final), exist_ok=True)
        tmp = final + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)              # atomic: old file or whole new file
        _fsync_dir(os.path.dirname(final))  # make the rename itself durable
        return final

    def done(self, outputs: Sequence[str]) -> bytes:
        rel = [self._artifact_path(p) for p in outputs]
        data = json.dumps({"outputs": rel}, separators=(",", ":")).encode()
        self.publish("DONE.json", data)  # DONE.json first, then the event
        self.emit("done", outputs=rel)
        return data
```

### Cancellation inside Python

The signal handler only sets a flag; the training loop calls `check_cancel()` at a safe point, after saving a checkpoint. The orphan check compares the parent pid with the one recorded at start (Listing 14).

```python
    @property
    def cancelled(self) -> bool:
        # A child whose worker was SIGKILLed is reparented (to init or a
        # subreaper): nobody will read its result, and the next attempt is
        # about to start in the same work directory. Stop as if cancelled.
        if not self._cancelled and os.getppid() != self._ppid:
            self._cancelled = True
        return self._cancelled

    def install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            self.request_cancel()  # no exit here: the loop checkpoints first
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
```

### `run`: exceptions to exit codes

In `run` the order of the steps is the whole protocol: lock, then the `DONE.json` check, then `main`, then `done` (Listing 15).

```python
def run(main, argv=None, env=None) -> int:
    try:
        act = Activity(sys.argv[1:] if argv is None else argv, env)
    except SpecError as e:          # missing --spec/--progress, bad TL_ATTEMPT
        print(f"activity: {e}", file=sys.stderr)
        return EXIT_DATAERR
    act.install_signal_handlers()
    try:
        if not act._lock():         # flock(LOCK_EX | LOCK_NB) on <dir>/.lock
            print(f"activity: another attempt holds {act.work_dir}/.lock", file=sys.stderr)
            return EXIT_TEMPFAIL    # try again once the orphan is gone
        if act.is_done():           # a duplicate delivery: one effect only
            with open(os.path.join(act.work_dir, "DONE.json"), "rb") as f:
                outputs = json.loads(f.read()).get("outputs", [])
            act.emit("done", outputs=outputs)
            return EXIT_OK
        outputs = main(act)
        act.done(list(outputs or []))
        return EXIT_OK
    except SpecError as e:
        # ...
        return EXIT_DATAERR
    except RetryableError as e:
        # ...
        return EXIT_TEMPFAIL
    except Cancelled as e:
        # ...
        return EXIT_CANCELED
    except Exception:  # a crash: retryable, with the traceback for Failure.message
        traceback.print_exc()
        return 1
    finally:
        act.close()  # closes progress, releases the lock
```

The full helper is [`activity.py`](../../../../../course/solutions/python/tinyllm/io/activity.py).

## Building the exporter

`parse_traceparent` is strict: version `ff` is invalid, version `00` must have exactly four parts, and all-zero ids are rejected. The sampled flag is bit 0 of the flags byte (Listing 16).

```python
def _hex(s: str, n: int) -> bool:
    return len(s) == n and bool(_HEX.match(s)) and s != "0" * n  # lowercase hex, not all zeros

def parse_traceparent(value: Optional[str]) -> Optional[SpanContext]:
    if not value:
        return None
    parts = value.strip().split("-")
    if len(parts) < 4:
        return None
    version, trace_id, span_id, flags = parts[:4]
    if not (len(version) == 2 and _HEX.match(version)) or version == "ff":
        return None
    if version == "00" and len(parts) != 4:  # future versions may append fields; 00 may not
        return None
    if not (_hex(trace_id, 32) and _hex(span_id, 16)):
        return None
    if not (len(flags) == 2 and _HEX.match(flags)):
        return None
    return SpanContext(trace_id, span_id, bool(int(flags, 16) & 1))
```

The tracer takes its sampling decision from the remote parent, encodes integers as strings, and never raises out of `flush` (Listing 17).

```python
        # A remote parent decided whether this trace is recorded; a new root trace is.
        self.sampled = parent.sampled if parent is not None else True
        self._trace_id = parent.trace_id if parent is not None else self._new_id(16)

def _value(v: object) -> dict:
    # bool before int: bool is a subclass of int in Python.
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}  # OTLP JSON: int64 as a string, exact past 2**53
    # ... doubleValue, stringValue

    def _post(self, path: str, body: dict) -> bool:
        # ...
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False  # telemetry never fails the work it observes

def should_sample_step(step: int, every: int = 50) -> bool:
    # ...
    return step % every == 0  # step 0, 50, 100, ...: step 50 yes, step 49 no
```

`end_span` queues a span only when `self.sampled and not self._closed`, which gives both "unsampled exports nothing" and "nothing queued after shutdown". `gauge` overwrites by `(name, attributes)`, so the latest loss wins. The full exporter is [`telemetry.py`](../../../../../course/solutions/python/tinyllm/io/telemetry.py).

## Review questions

1. What must a heartbeat avoid promising about a checkpoint?

   <details><summary>Answer</summary>

   Liveness does not establish that all checkpoint data is complete and durable.

   </details>

[Contents](README.md) · [Next: Failure tests and integration](04-failure-tests-and-integration.md)
