#!/usr/bin/env bash
# lang.02 reference worker: holds a lock file, ticks every $WORKER_INTERVAL
# seconds, and on SIGTERM or SIGINT removes its lock and exits 128 + signal.
set -euo pipefail

state="${WORKER_STATE_DIR:-${TMPDIR:-/tmp}}"
interval="${WORKER_INTERVAL:-30}"
lock="$state/worker.lock"

# A lock whose PID is gone was left by a process that could not clean up
# (SIGKILL, power loss). `kill -0` sends no signal; it only asks "does it exist?".
if [ -e "$lock" ]; then
  old="$(cat "$lock")"
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "worker: already running (pid $old)" >&2
    exit 1
  fi
  echo "worker: removing stale lock (pid ${old:-?})" >&2
  rm -f "$lock"
fi
echo "$$" > "$lock"

stop() {
  # Kill the background sleep too, or it outlives us as an orphan. `jobs -p`
  # lists every background child, including one started an instant before
  # the signal arrived.
  local kids
  kids="$(jobs -p)"
  # shellcheck disable=SC2086  # one word per PID: the splitting is the point
  if [ -n "$kids" ]; then kill $kids 2>/dev/null || true; fi
  rm -f "$lock"
  exit "$1"
}
trap 'stop 143' TERM   # 128 + 15
trap 'stop 130' INT    # 128 + 2

echo ready
while true; do
  echo tick
  # Sleep in the background and `wait` for it: bash runs a trap only between
  # commands, and `wait` returns as soon as a trapped signal arrives, while a
  # foreground `sleep 30` would delay the trap for up to 30 seconds.
  sleep "$interval" &
  wait $! || true
done
