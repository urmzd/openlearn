# contracts/py/tinyllm/io/activity.pyi (dur.09): the Python half of a subprocess activity
# chapter: content/courses/ai-platform-engineering/05-durable-orchestration-and-workers/09-subprocess-activities.md
#
# The whole interface between the Go worker and a Python entry is
# spec/subprocess-activity.md. This module is the helper every Python
# activity entry ({tinyllm} train, eval, export; {corpus} run --stage) runs
# under, so each of them gets the same progress file, signal handling,
# atomic outputs, and exit codes:
#
#     def train_main(act: Activity) -> list[str]:
#         spec = act.load_spec(validate)            # SpecError: exit 65
#         start = resume_from(act.resume)           # None on attempt 1
#         for step in range(start, spec["steps"]):
#             ...
#             act.step(step, loss, lr, tokens)
#             if step % 500 == 0 or act.cancelled:
#                 path = save_checkpoint(...)       # L0.6
#                 act.checkpoint(step, path)        # only after it is complete
#             act.check_cancel()                    # Cancelled: exit 130
#         return [final_dir]                        # outputs, relative to TL_ARTIFACTS
#
#     sys.exit(run(train_main, sys.argv[1:]))
#
# Words used below:
#   work dir   the directory holding --progress: the runner makes it
#              <TL_ARTIFACTS>/activities/<TL_IDEMPOTENCY_KEY>/, the same for
#              every attempt of one activity
#   artifact path
#              a path relative to TL_ARTIFACTS (absolute paths under it are
#              accepted and made relative); a path outside it is ValueError
#   publish    write <name>.tmp, flush and fsync it, rename it to <name>, and
#              fsync the directory: a reader sees the old file or the whole
#              new one, never a prefix
from typing import Callable, Mapping, Optional, Sequence

EXIT_OK: int  # 0
EXIT_DATAERR: int  # 65, EX_DATAERR: the input is wrong, never retried
EXIT_TEMPFAIL: int  # 75, EX_TEMPFAIL: retryable
EXIT_CANCELED: int  # 130: cancelled (SIGTERM honored)

class SpecError(Exception):
    """The spec (or another input) is wrong: run() exits 65."""

class RetryableError(Exception):
    """A failure the next attempt may not hit: run() exits 75."""

class Cancelled(Exception):
    """Raised by check_cancel() after SIGTERM: run() exits 130."""

class Activity:
    spec_path: str  # --spec
    progress_path: str  # --progress
    resume: Optional[str]  # --resume <artifact path of a checkpoint>, None without the flag
    key: str  # TL_IDEMPOTENCY_KEY ("" when unset)
    attempt: int  # TL_ATTEMPT (1 when unset)
    artifacts: str  # TL_ARTIFACTS (default "/artifacts"), absolute
    work_dir: str  # absolute; the directory of progress_path
    def __init__(
        self,
        argv: Sequence[str],
        env: Optional[Mapping[str, str]] = None,
        clock: Callable[[], float] = ...,
    ) -> None:
        """Parse --spec, --progress, and --resume out of argv (other arguments
        are left to the entry) and read the TL_* environment (os.environ when
        env is None). SpecError when --spec or --progress is missing. clock
        gives the "ts" of every progress event (default time.time)."""
    @property
    def cancelled(self) -> bool:
        """True once SIGTERM (or SIGINT) arrived, or the parent process died
        (the worker that started this child is gone)."""
    def request_cancel(self) -> None:
        """What the SIGTERM handler calls: only sets the flag. Checkpointing
        is the loop's job, at its next safe point."""
    def check_cancel(self) -> None:
        """Raise Cancelled if cancelled; otherwise nothing."""
    def install_signal_handlers(self) -> None:
        """Route SIGTERM and SIGINT to request_cancel (run() calls it)."""
    def load_spec(
        self, validate: Optional[Callable[[dict], Sequence[str]]] = None
    ) -> dict:
        """The parsed spec. SpecError when it is not a JSON object, or when
        validate returns any error strings (they are in the message)."""
    def emit(self, kind: str, **fields: object) -> None:
        """Append one progress event {"ts", "kind", **fields} as one JSON line
        and flush it. kind is one of step, ckpt, metric, done; ValueError
        otherwise, or for a non-finite float field."""
    def step(self, step: int, loss: float, lr: float, tokens: int) -> None:
        """emit("step", ...)."""
    def metric(self, name: str, value: float) -> None:
        """emit("metric", name=..., value=...)."""
    def checkpoint(self, step: int, path: str) -> str:
        """Emit a ckpt event for the checkpoint step directory at path (an
        artifact path or an absolute path under TL_ARTIFACTS) and return its
        artifact path. The event is the runner's heartbeat detail and the next
        attempt's --resume, so it is emitted only for a complete checkpoint:
        ValueError, and no event, when L0.6's verify_step_dir reports any
        problem."""
    def output(self, name: str) -> str:
        """Absolute path of <work dir>/<name>; ValueError for a name that
        leaves the work dir."""
    def publish(self, name: str, data: bytes) -> str:
        """Publish data as <work dir>/<name> (see "publish" above) and return
        the absolute path."""
    def is_done(self) -> bool:
        """<work dir>/DONE.json exists: an earlier attempt finished."""
    def done(self, outputs: Sequence[str]) -> bytes:
        """The last act of a successful run: publish DONE.json
        ({"outputs": [artifact paths]}), then emit the done event. Returns the
        DONE.json bytes (the activity's result)."""
    def close(self) -> None:
        """Close the progress file and release the work-dir lock."""

def run(
    main: Callable[[Activity], Optional[Sequence[str]]],
    argv: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Run one activity and return its exit code (the entry passes it to
    sys.exit):
      1. Activity(argv, env); install the signal handlers
      2. take an exclusive, non-blocking lock on <work dir>/.lock; when
         another live attempt holds it, return 75
      3. if is_done(): re-emit the done event from DONE.json and return 0
         without calling main (a rerun after success is a no-op)
      4. outputs = main(act); done(outputs or []); return 0
    Exceptions map to codes: SpecError 65, RetryableError 75, Cancelled 130,
    anything else (a crash) 1, after printing the traceback to stderr. The
    lock and the progress file are released on every path."""
