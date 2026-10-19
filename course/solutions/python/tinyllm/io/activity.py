"""The Python half of a subprocess activity (dur.09): spec/subprocess-activity.md.

The Go worker runs a Python entry as a child process: `--spec` and
`--progress` name files in the activity's work directory, the TL_* variables
carry the idempotency key and attempt, and the exit code is the verdict.
Everything an entry needs to keep that contract lives here, so `train`,
`eval`, `export`, and `corpus run --stage` behave the same under a kill:

- progress events are single JSON lines, flushed one at a time, so the
  runner's tail never reads half an event
- a `ckpt` event is emitted only for a complete checkpoint (L0.6), because
  the runner heartbeats it and the next attempt resumes from it
- outputs are published by rename, DONE.json last, so a rerun after success
  finds DONE.json and is a no-op
- SIGTERM only sets a flag; the loop checkpoints at its next safe point and
  exits 130
- one attempt at a time owns the work directory (an exclusive lock), so a
  child orphaned by a killed worker cannot write beside its successor

Contract: contracts/py/tinyllm/io/activity.pyi.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import sys
import time
import traceback
from typing import Callable, Mapping, Optional, Sequence

from tinyllm.io.checkpoint import verify_step_dir

EXIT_OK = 0
EXIT_DATAERR = 65
EXIT_TEMPFAIL = 75
EXIT_CANCELED = 130

_KINDS = ("step", "ckpt", "metric", "done")


class SpecError(Exception):
    """The spec (or another input) is wrong: run() exits 65."""


class RetryableError(Exception):
    """A failure the next attempt may not hit: run() exits 75."""


class Cancelled(Exception):
    """Raised by check_cancel() after SIGTERM: run() exits 130."""


def _flag(argv: Sequence[str], name: str) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Activity:
    def __init__(
        self,
        argv: Sequence[str],
        env: Optional[Mapping[str, str]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        env = os.environ if env is None else env
        spec, progress = _flag(argv, "--spec"), _flag(argv, "--progress")
        if not spec or not progress:
            raise SpecError("usage: <verb> --spec <dir>/spec.json --progress <dir>/progress.jsonl [--resume <ckpt>]")
        self.spec_path = os.path.abspath(spec)
        self.progress_path = os.path.abspath(progress)
        self.resume = _flag(argv, "--resume") or None
        self.key = env.get("TL_IDEMPOTENCY_KEY", "")
        try:
            self.attempt = int(env.get("TL_ATTEMPT", "1"))
        except ValueError:
            raise SpecError(f"TL_ATTEMPT={env.get('TL_ATTEMPT')!r} is not an integer") from None
        self.artifacts = os.path.abspath(env.get("TL_ARTIFACTS", "/artifacts"))
        self.work_dir = os.path.dirname(self.progress_path)
        self._clock = clock
        self._cancelled = False
        self._ppid = os.getppid()
        self._progress = None
        self._lock_fd: Optional[int] = None

    @property
    def cancelled(self) -> bool:
        # A child whose worker was SIGKILLed is reparented (to init or a
        # subreaper): nobody will read its result, and the next attempt is
        # about to start in the same work directory. Stop as if cancelled.
        if not self._cancelled and os.getppid() != self._ppid:
            self._cancelled = True
        return self._cancelled

    def request_cancel(self) -> None:
        self._cancelled = True

    def check_cancel(self) -> None:
        if self.cancelled:
            raise Cancelled("cancelled: SIGTERM received or the worker is gone")

    def install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            self.request_cancel()

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def load_spec(
        self, validate: Optional[Callable[[dict], Sequence[str]]] = None
    ) -> dict:
        try:
            with open(self.spec_path, "rb") as f:
                spec = json.loads(f.read().decode("utf-8"))
        except FileNotFoundError:
            raise SpecError(f"no spec at {self.spec_path}") from None
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SpecError(f"{self.spec_path} is not JSON: {e}") from None
        if not isinstance(spec, dict):
            raise SpecError(f"{self.spec_path}: the spec must be a JSON object")
        errs = list(validate(spec)) if validate else []
        if errs:
            raise SpecError(f"{self.spec_path} fails validation: " + "; ".join(errs))
        return spec

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

    def step(self, step: int, loss: float, lr: float, tokens: int) -> None:
        self.emit("step", step=int(step), loss=float(loss), lr=float(lr), tokens=int(tokens))

    def metric(self, name: str, value: float) -> None:
        self.emit("metric", name=str(name), value=float(value))

    def _artifact_path(self, path: str) -> str:
        full = os.path.normpath(path if os.path.isabs(path) else os.path.join(self.artifacts, path))
        rel = os.path.relpath(full, self.artifacts)
        if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
            raise ValueError(f"{path} is outside TL_ARTIFACTS={self.artifacts}")
        return rel

    def checkpoint(self, step: int, path: str) -> str:
        rel = self._artifact_path(path)
        problems = verify_step_dir(os.path.join(self.artifacts, rel))
        if problems:
            raise ValueError(f"checkpoint {rel} is not complete: " + "; ".join(problems))
        self.emit("ckpt", step=int(step), ckpt=rel)
        return rel

    def output(self, name: str) -> str:
        full = os.path.normpath(os.path.join(self.work_dir, name))
        if os.path.isabs(name) or not full.startswith(self.work_dir + os.sep):
            raise ValueError(f"{name!r} is not a file inside the work dir {self.work_dir}")
        return full

    def publish(self, name: str, data: bytes) -> str:
        final = self.output(name)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        tmp = final + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        _fsync_dir(os.path.dirname(final))
        return final

    def is_done(self) -> bool:
        return os.path.isfile(os.path.join(self.work_dir, "DONE.json"))

    def done(self, outputs: Sequence[str]) -> bytes:
        rel = [self._artifact_path(p) for p in outputs]
        data = json.dumps({"outputs": rel}, separators=(",", ":")).encode()
        self.publish("DONE.json", data)
        self.emit("done", outputs=rel)
        return data

    def _lock(self) -> bool:
        os.makedirs(self.work_dir, exist_ok=True)
        fd = os.open(os.path.join(self.work_dir, ".lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._lock_fd = fd
        return True

    def close(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None


def run(
    main: Callable[[Activity], Optional[Sequence[str]]],
    argv: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    try:
        act = Activity(sys.argv[1:] if argv is None else argv, env)
    except SpecError as e:
        print(f"activity: {e}", file=sys.stderr)
        return EXIT_DATAERR
    act.install_signal_handlers()
    try:
        if not act._lock():
            print(f"activity: another attempt holds {act.work_dir}/.lock", file=sys.stderr)
            return EXIT_TEMPFAIL
        if act.is_done():
            with open(os.path.join(act.work_dir, "DONE.json"), "rb") as f:
                outputs = json.loads(f.read()).get("outputs", [])
            act.emit("done", outputs=outputs)
            return EXIT_OK
        outputs = main(act)
        act.done(list(outputs or []))
        return EXIT_OK
    except SpecError as e:
        print(f"activity: {e}", file=sys.stderr)
        return EXIT_DATAERR
    except RetryableError as e:
        print(f"activity: retryable: {e}", file=sys.stderr)
        return EXIT_TEMPFAIL
    except Cancelled as e:
        print(f"activity: {e}", file=sys.stderr)
        return EXIT_CANCELED
    except Exception:  # noqa: BLE001 - a crash: retryable, with the traceback for Failure.message
        traceback.print_exc()
        return 1
    finally:
        act.close()
