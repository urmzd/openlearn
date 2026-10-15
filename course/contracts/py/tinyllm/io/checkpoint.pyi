# contracts/py/tinyllm/io/checkpoint.pyi (L0.6): atomic checkpoints
# chapter: content/courses/build-an-llm/p00-foundations/06-safetensors-checkpoints-and-token-streams.md
#
# The on-disk layout is formats/checkpoint.md: <dir>/step-<nnnnnn>/ holding
# model.safetensors, optimizer.safetensors, trainer_state.json
# (trainer-state.schema.json), config.json when there is one, and
# MANIFEST.json (manifest.schema.json, kind "checkpoint") written last; and
# <dir>/LATEST, the name of the newest complete step directory and "\n".
#
# Words used below:
#   complete   MANIFEST.json exists, lists model.safetensors,
#              optimizer.safetensors, and trainer_state.json, lists every
#              other file in the directory, and every listed file has its
#              size and sha256
#   rng_state  the shuffle PCG32's (state, inc) (M06.3 PCG32.state()), stored
#              as 16 hex digits each so no JSON reader rounds them
#
# Optimizer state goes into optimizer.safetensors as F32 tensors named
# "<parameter name>.<state key>" (AdamW "fc.weight.exp_avg", SGD
# "fc.weight.momentum_buffer"), with the rest of opt.state_dict() (step
# count, hyperparameters) as JSON in the metadata key "state".
from typing import Any, Optional

from numpy.typing import NDArray

REQUIRED_FILES: tuple[str, ...]  # ("model.safetensors", "optimizer.safetensors", "trainer_state.json")

class Checkpoint:
    path: str  # the step directory that was loaded
    step: int
    model: dict[str, NDArray]  # for Module.load_state_dict (L0.4)
    opt: dict  # for the optimizer's load_state_dict (M10.2, M10.3)
    rng_state: tuple[int, int]  # (state, inc) for PCG32.set_state
    extra: dict  # tokens_seen, data_cursor, lr, config_sha256, git_sha [, loss_scale] [, config]

def step_name(step: int) -> str:
    """"step-%06d" (more digits past 999999). ValueError for step < 0."""

def save_checkpoint(
    dir: str,
    model: Any,
    opt: Any,
    step: int,
    rng_state: Any,
    extra: dict,
    keep: Optional[int] = None,
) -> str:
    """Write <dir>/step-<nnnnnn>/ atomically and point LATEST at it; returns
    its path. Steps, in order: every file into step-<nnnnnn>.tmp/, each
    fsynced; MANIFEST.json last, fsynced, then the directory; rename to
    step-<nnnnnn> (replacing an existing one) and fsync <dir>; write
    LATEST.tmp, fsync, rename to LATEST, fsync <dir>; then delete stale *.tmp
    directories and, with keep, every step directory but the keep newest.
    A crash at any instant leaves load_checkpoint returning the previous
    checkpoint or this one.
    model gives named_parameters() and state_dict() (F32), opt state_dict().
    rng_state is (state, inc) or a dict already holding pcg_state/pcg_inc
    hex. extra may hold only tokens_seen (default 0), data_cursor
    {"shard", "offset"} (default zeros), lr (default opt.lr), config_sha256
    (default the sha256 of config.json's bytes), git_sha (default
    "unknown"), loss_scale, and config (a dict written as config.json).
    ValueError for another key, a malformed value, an even inc, a non-F32
    array, or keep < 1."""

def verify_step_dir(path: str) -> list[str]:
    """What keeps the step directory from being complete; [] when it is."""

def load_checkpoint(dir: str, step: Optional[int] = None) -> Checkpoint:
    """With step: that directory, ValueError unless it is complete. Without:
    the directory LATEST names if it is complete, else the newest older
    complete one (directories newer than LATEST were never declared done);
    with no LATEST, the newest complete one. FileNotFoundError when none is
    complete. extra["config"] is the parsed config.json when present."""
