"""Where things live. Every location can be overridden by an env var, which is
how the harness self-tests point `ol` at a throwaway course tree and learner.

  OL_ROOT          the openlearn checkout running the harness (set by ol)
  OL_COURSE_ROOT   the live course tree (default $OL_ROOT/course)
  OL_COURSE_HOME   the learner repo (default $OL_ROOT/.scratchpad/course)
  OL_CACHE         worktree cache (default $TINYLLM_CACHE or ~/.cache/openlearn)
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()


def root() -> Path:
    env = os.environ.get("OL_ROOT")
    return Path(env).resolve() if env else HERE.parents[4]


def live_course() -> Path:
    env = os.environ.get("OL_COURSE_ROOT")
    return Path(env).resolve() if env else root() / "course"


def site_root(course: Path | None = None) -> Path:
    """The tree chapters and paths are relative to: the course tree's parent."""
    return (course or live_course()).parent


def learner_home() -> Path:
    env = os.environ.get("OL_COURSE_HOME")
    return Path(env).resolve() if env else root() / ".scratchpad" / "course"


def cache_dir() -> Path:
    env = os.environ.get("OL_CACHE") or os.environ.get("TINYLLM_CACHE")
    return Path(env).resolve() if env else Path.home() / ".cache" / "openlearn"


def scratch() -> Path:
    env = os.environ.get("OL_SCRATCH")
    return Path(env).resolve() if env else root() / ".scratchpad"


# ---------------------------------------------------------------------------
# output

_TTY = sys.stdout.isatty()
RED = "\033[31m" if _TTY else ""
GRN = "\033[32m" if _TTY else ""
YEL = "\033[33m" if _TTY else ""
DIM = "\033[2m" if _TTY else ""
BLD = "\033[1m" if _TTY else ""
RST = "\033[0m" if _TTY else ""


def say(msg: str = "") -> None:
    print(msg, flush=True)


def err(msg: str) -> None:
    print(f"{RED}error:{RST} {msg}", file=sys.stderr, flush=True)


def indent(text: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.rstrip("\n").splitlines())


def tail(text: str, n: int = 40) -> str:
    lines = text.rstrip("\n").splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join([f"... ({len(lines) - n} lines above)"] + lines[-n:])


# ---------------------------------------------------------------------------
# subprocess


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: float | None = None,
) -> tuple[int, str]:
    """Run, merging stderr into stdout. A missing binary is rc 127, a timeout 124."""
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return p.returncode, p.stdout
    except FileNotFoundError as e:
        return 127, f"{cmd[0]}: not found ({e})"
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return 124, out + f"\n(timed out after {timeout}s)"


def git(args: list[str], cwd: Path) -> tuple[int, str]:
    return run(["git", *args], cwd=cwd)


def git_toplevel(p: Path) -> Path | None:
    rc, out = git(["rev-parse", "--show-toplevel"], p if p.is_dir() else p.parent)
    return Path(out.strip()).resolve() if rc == 0 else None


def git_head(repo: Path) -> str | None:
    rc, out = git(["rev-parse", "HEAD"], repo)
    return out.strip() if rc == 0 else None


def harness_sha() -> str:
    return (git_head(root()) or "unknown")[:12]


def base_env() -> dict:
    """The determinism environment of DESIGN 5.11, layered on the caller's."""
    env = dict(os.environ)
    for k in ("OL_ROOT",):
        env.pop(k, None)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "OL_SEED": os.environ.get("OL_SEED", "0"),
            "TZ": "UTC",
            "LC_ALL": "C.UTF-8",
            "GOFLAGS": "-count=1",
        }
    )
    return env


@contextlib.contextmanager
def lock(d: Path):
    """Serialize builds that share one directory (the Rust farm, the object
    cache): a second `ol check` or `ol verify` waits instead of racing."""
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "ol.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
