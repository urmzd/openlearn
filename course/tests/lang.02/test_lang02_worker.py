"""lang.02 course tests: your worker script (primers/lang.02/worker.sh).

Annotated exemplars (DESIGN 5.12). Each test starts the script in its own
process group with a private state directory, waits for its `ready` line,
sends a signal, and reads the exit status. A process that exits by itself
with status 143 shows up as returncode 143; one the signal killed shows up as
-15 (Python's way of saying "terminated by signal 15").
"""

import os
import select
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(os.environ["OL_PRIMER_DIR"]) / "worker.sh"


def _default_signals():
    # Signals ignored when a shell starts cannot be trapped by it, so reset
    # the two we send to their defaults before exec.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def start(state: Path, interval: str = "0.2") -> subprocess.Popen:
    env = {**os.environ, "WORKER_STATE_DIR": str(state), "WORKER_INTERVAL": interval}
    return subprocess.Popen(
        [str(SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=_default_signals,
    )


def read_line(p: subprocess.Popen, timeout: float = 5.0) -> str:
    fd = p.stdout.fileno()
    buf = b""
    end = time.monotonic() + timeout
    while not buf.endswith(b"\n"):
        left = end - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            raise AssertionError(
                f"no complete line on stdout within {timeout}s (got {buf!r})"
            )
        chunk = os.read(fd, 1)
        if not chunk:
            raise AssertionError(
                f"stdout closed before a full line (got {buf!r}); stderr: {p.stderr.read()!r}"
            )
        buf += chunk
    return buf.decode().rstrip("\n")


def ready(state: Path, interval: str = "0.2") -> subprocess.Popen:
    p = start(state, interval)
    try:
        assert read_line(p) == "ready", "the first line on stdout must be `ready`"
    except BaseException:
        stop(p)
        raise
    return p


def stop(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait(timeout=5)


def finish(p: subprocess.Popen, sig: int, timeout: float) -> int:
    os.kill(p.pid, sig)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop(p)
        raise AssertionError(f"still running {timeout}s after signal {sig}") from None


@pytest.fixture
def state(tmp_path):
    return tmp_path


def test_worker_is_an_executable_script():
    # WHY: `./worker.sh` runs only when the file has the execute bit (chmod +x)
    #      and starts with a #! line naming its interpreter; Kubernetes and CI
    #      exec your entry points the same way.
    # KIND: unit
    # CHAPTER: lang.02 section 4
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"
    assert os.access(SCRIPT, os.X_OK), (
        "worker.sh is not executable: chmod +x primers/lang.02/worker.sh"
    )
    assert SCRIPT.read_bytes().startswith(b"#!"), "worker.sh must start with a #! line"


def test_worker_writes_its_pid_to_the_lock(state):
    # WHY: section 3: once `ready` is printed, WORKER_STATE_DIR/worker.lock
    #      holds the worker's own PID ($$), which is what lets a second start
    #      tell a live worker from a dead one.
    # KIND: unit
    # CHAPTER: lang.02 section 3
    p = ready(state)
    try:
        assert (state / "worker.lock").read_text().strip() == str(p.pid)
    finally:
        stop(p)


def test_sigterm_exits_143_and_removes_the_lock(state):
    # WHY: SIGTERM is "please stop" (kill's default, Kubernetes' first signal).
    #      A trap gets to clean up, then exits 128 + 15 = 143 so the caller
    #      can still tell it was terminated. No trap: the shell dies with the
    #      signal (returncode -15) and the lock stays behind.
    # KIND: unit
    # CHAPTER: lang.02 section 5, pitfall 6
    p = ready(state)
    assert finish(p, signal.SIGTERM, timeout=5) == 143
    assert not (state / "worker.lock").exists(), "the TERM trap must remove the lock"


def test_sigint_exits_130_and_removes_the_lock(state):
    # WHY: SIGINT is Ctrl-C. Same cleanup, status 128 + 2 = 130.
    # KIND: unit
    p = ready(state)
    assert finish(p, signal.SIGINT, timeout=5) == 130
    assert not (state / "worker.lock").exists(), "the INT trap must remove the lock"


def test_trap_runs_promptly_during_a_long_sleep(state):
    # WHY: bash runs a trap only after the current foreground command ends.
    #      With WORKER_INTERVAL=30, a foreground `sleep 30` holds the trap for
    #      up to 30 s, past Kubernetes' grace period, and the pod gets SIGKILL.
    #      `sleep & wait $!` lets the trap run at once.
    # KIND: boundary
    # CHAPTER: lang.02 section 5, pitfall 7
    p = ready(state, interval="30")
    assert finish(p, signal.SIGTERM, timeout=3) == 143


def test_no_process_is_left_behind(state):
    # WHY: the background `sleep` is a separate process. If the trap does not
    #      kill it, it lives on as an orphan after the worker exits; a loop
    #      that restarts workers then leaks one process per restart.
    # KIND: boundary
    # CHAPTER: lang.02 section 5, pitfall 8
    p = ready(state, interval="30")
    assert finish(p, signal.SIGTERM, timeout=3) == 143
    end = time.monotonic() + 2
    while time.monotonic() < end:
        try:
            os.killpg(p.pid, 0)  # signal 0: does any process of the group exist?
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.killpg(p.pid, signal.SIGKILL)
    raise AssertionError(
        "a process of the worker's group outlived it (the background sleep?)"
    )


def test_second_start_refuses_while_the_first_runs(state):
    # WHY: two workers on one lock would both "own" it. The second start sees
    #      a lock whose PID is alive (kill -0 succeeds), prints why on stderr,
    #      and exits 1 without touching the lock.
    # KIND: unit
    p = ready(state)
    try:
        r = subprocess.run(
            [str(SCRIPT)],
            env={**os.environ, "WORKER_STATE_DIR": str(state)},
            capture_output=True,
            text=True,
            timeout=5,
            preexec_fn=_default_signals,
        )
        assert r.returncode == 1, (
            f"second start exited {r.returncode}; stdout {r.stdout!r}"
        )
        assert r.stderr.strip(), "say why on stderr"
        assert (state / "worker.lock").read_text().strip() == str(p.pid)
    finally:
        stop(p)


def test_stale_lock_after_sigkill_is_recovered(state):
    # WHY: SIGKILL cannot be trapped, so no cleanup runs and the lock stays.
    #      The next start must notice the PID is gone and take the lock over,
    #      or one crash blocks the worker forever.
    # KIND: fault
    # CHAPTER: lang.02 section 5, pitfall 9
    p = ready(state)
    os.killpg(p.pid, signal.SIGKILL)
    assert p.wait(timeout=5) == -signal.SIGKILL
    assert (state / "worker.lock").exists(), (
        "SIGKILL runs no trap: the lock must still be there"
    )
    q = ready(state)
    try:
        assert (state / "worker.lock").read_text().strip() == str(q.pid)
    finally:
        stop(q)
