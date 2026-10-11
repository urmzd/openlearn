#!/usr/bin/env bash
# Scripted responder for drill ops.00 engine-crashloop (DESIGN 5.10, "Drills in
# CI"): the reference fix, run by the nightly kind job against the reference
# system after `ol drill start engine-crashloop`, before `ol drill end`.
#
#   respond.sh [LEARNER_DIR]     (default: $OL_COURSE_HOME, else the current dir)
#
# It reads [system].name and [deploy] from LEARNER_DIR/system.toml, refuses
# any context that is not kind- or k3d- (the drill safety gate's rule), rolls
# the engine Deployment back, waits for the rollout, and writes the runbook
# and today's postmortem from course/ref/docs with <system> filled in.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
course="$(cd "$here/../.." && pwd)"
learner="$(cd "${1:-${OL_COURSE_HOME:-$PWD}}" && pwd)"

read -r name ctx ns target < <(python3 - "$learner/system.toml" <<'PY'
import sys, tomllib
s = tomllib.load(open(sys.argv[1], "rb"))
d = s.get("deploy", {})
print(s.get("system", {}).get("name", ""), d.get("kube_context", ""), d.get("namespace", ""),
      d.get("services", {}).get("engine", ""))
PY
)
for v in name ctx ns target; do
  [[ -n "${!v}" ]] || { echo "respond.sh: system.toml lacks the value for $v ([system].name, [deploy].kube_context, namespace, services.engine)" >&2; exit 2; }
done
case "$ctx" in kind-*|k3d-*) ;; *) echo "respond.sh: refusing context $ctx (not kind- or k3d-)" >&2; exit 2 ;; esac

k() { kubectl --context "$ctx" -n "$ns" "$@"; }

# Diagnose as the runbook says, so the CI log shows the same evidence a person sees.
k get pods -o wide || true
k rollout history "$target" || true
k logs "$target" --previous --tail=20 || true

# Mitigate: back to the last revision that worked, then wait for it to be Ready.
k rollout undo "$target"
k rollout status "$target" --timeout=180s

# Document: the runbook and today's postmortem.
fill() { sed "s/<system>/$name/g" "$1" > "$2"; }
mkdir -p "$learner/docs/runbooks" "$learner/docs/postmortems"
[[ -f "$learner/docs/runbooks/engine-crashloop.md" ]] || \
  fill "$course/ref/docs/runbooks/engine-crashloop.md" "$learner/docs/runbooks/engine-crashloop.md"
pm="$(ls "$course"/ref/docs/postmortems/*-engine-crashloop.md | head -1)"
fill "$pm" "$learner/docs/postmortems/$(date -u +%Y-%m-%d)-engine-crashloop.md"
echo "respond.sh: rolled back $target in $ctx/$ns; runbook and postmortem written"
