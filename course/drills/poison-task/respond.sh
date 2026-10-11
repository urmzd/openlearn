#!/usr/bin/env bash
# Scripted responder for drill ops.03 poison-task (DESIGN 5.10, "Drills in
# CI"): the reference response, run by the nightly kind job against the
# reference system after `ol drill run poison-task --respond` saw
# DurableDeadLetters fire, before `ol drill end`.
#
#   respond.sh [LEARNER_DIR]     (default: $OL_COURSE_HOME, else the current dir)
#
# The runbook's response: read the dead letter's failure, find what changed
# (a failpoint in the worker Deployment's environment), roll the workers back
# to a clean environment, redrive the dead letter, and wait for the build to
# complete. It refuses any context that is not kind- or k3d- (the drill safety
# gate's rule) and writes the runbook and today's postmortem from
# course/ref/docs.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
course="$(cd "$here/../.." && pwd)"
learner="$(cd "${1:-${OL_COURSE_HOME:-$PWD}}" && pwd)"

read -r name ctx ns worker < <(python3 - "$learner/system.toml" <<'PY'
import sys, tomllib
s = tomllib.load(open(sys.argv[1], "rb"))
d = s.get("deploy", {})
print(s.get("system", {}).get("name", ""), d.get("kube_context", ""), d.get("namespace", ""),
      d.get("services", {}).get("worker", ""))
PY
)
for v in name ctx ns worker; do
  [[ -n "${!v}" ]] || { echo "respond.sh: system.toml lacks the value for $v ([system].name, [deploy].kube_context, namespace, services.worker)" >&2; exit 2; }
done
case "$ctx" in kind-*|k3d-*) ;; *) echo "respond.sh: refusing context $ctx (not kind- or k3d-)" >&2; exit 2 ;; esac

k() { kubectl --context "$ctx" -n "$ns" "$@"; }
role() {
  local r="$1"; shift
  local argv
  argv="$(python3 - "$learner/system.toml" "$r" <<'PY'
import shlex, sys, tomllib
e = tomllib.load(open(sys.argv[1], "rb")).get("entry", {}).get(sys.argv[2])
if not e:
    sys.exit(f"respond.sh: system.toml has no [entry].{sys.argv[2]}")
print(shlex.join(e))
PY
)"
  (cd "$learner" && eval "$argv" '"$@"')
}
dur=(--durable 127.0.0.1:30733)

# Diagnose: what is dead-lettered, and why; then what changed.
role ctl wf dlq list data "${dur[@]}" --json
k rollout history "$worker" | tail -5 || true
fp="$(k get "$worker" -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="TL_FAILPOINTS")].value}')"
echo "respond.sh: worker TL_FAILPOINTS=${fp:-<unset>}"

# Mitigate: a clean worker environment, rolled out, then the redrive.
if [[ -n "$fp" ]]; then
  k set env "$worker" TL_FAILPOINTS-
fi
k rollout status "$worker" --timeout=300s
role ctl wf dlq redrive data --all "${dur[@]}" --json

# Verify: the build completes with nothing dead-lettered.
for _ in $(seq 1 60); do
  last="$(role ctl wf describe drill-poison "${dur[@]}" --json | tail -1)"
  status="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("status",""))' "$last")"
  [[ "$status" == RUNNING ]] || break
  sleep 10
done
python3 - "$last" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
if r.get("status") != "COMPLETED" or r.get("dead_lettered") != 0:
    sys.exit(f"respond.sh: drill-poison is {r.get('status')} with {r.get('dead_lettered')} dead letter(s)")
print("respond.sh: drill-poison COMPLETED, nothing dead-lettered")
PY

# Document: the runbook the page links to and today's postmortem.
fill() { sed "s/<system>/$name/g" "$1" > "$2"; }
mkdir -p "$learner/docs/runbooks" "$learner/docs/postmortems"
[[ -f "$learner/docs/runbooks/DurableDeadLetters.md" ]] || \
  fill "$course/ref/docs/runbooks/DurableDeadLetters.md" "$learner/docs/runbooks/DurableDeadLetters.md"
pm="$(ls "$course"/ref/docs/postmortems/*-poison-task.md | head -1)"
fill "$pm" "$learner/docs/postmortems/$(date -u +%Y-%m-%d)-poison-task.md"
echo "respond.sh: $worker clean in $ctx/$ns; dead letter redriven; runbook and postmortem written"
