#!/usr/bin/env bash
# Scripted responder for drill ops.02 durable-kill9 (DESIGN 5.10, "Drills in
# CI"): the reference response, run by the nightly kind job against the
# reference system after `ss drill run durable-kill9 --respond` injected the
# kills, before `ss drill end`.
#
#   respond.sh [LEARNER_DIR]     (default: $SS_COURSE_HOME, else the current dir)
#
# The fault heals itself (the StatefulSet recreates the pod on the same WAL
# claim, the server recovers its log, expired leases are redelivered), so the
# response is the runbook's: confirm the server and the workers are whole,
# wait for the build to finish, then prove the pass condition (the same
# checks `ss drill end` runs as [[resolve.run]]): the same start returns the
# finished run with the clean-run numbers, and the ledger has every row once. It
# refuses any context that is not kind- or k3d- (the drill safety gate's rule)
# and writes the runbook and today's postmortem from course/ref/docs.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
course="$(cd "$here/../.." && pwd)"
learner="$(cd "${1:-${SS_COURSE_HOME:-$PWD}}" && pwd)"

read -r name ctx ns durable worker < <(python3 - "$learner/system.toml" <<'PY'
import sys, tomllib
s = tomllib.load(open(sys.argv[1], "rb"))
d = s.get("deploy", {})
sv = d.get("services", {})
print(s.get("system", {}).get("name", ""), d.get("kube_context", ""), d.get("namespace", ""),
      sv.get("durable", ""), sv.get("worker", ""))
PY
)
for v in name ctx ns durable worker; do
  [[ -n "${!v}" ]] || { echo "respond.sh: system.toml lacks the value for $v ([system].name, [deploy].kube_context, namespace, services.durable, services.worker)" >&2; exit 2; }
done
case "$ctx" in kind-*|k3d-*) ;; *) echo "respond.sh: refusing context $ctx (not kind- or k3d-)" >&2; exit 2 ;; esac

k() { kubectl --context "$ctx" -n "$ns" "$@"; }
# role ROLE ARGS...: run an [entry] role of system.toml from the learner repo.
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
corpus_cfg="$course/fixtures/MS-corpus/corpus.toml"

# Diagnose as the runbook says, so the CI log shows what a person sees.
k get pods -o wide || true
k get events --sort-by=.lastTimestamp | tail -20 || true
k logs "$durable" --previous --tail=20 || true

# Mitigate: the server and the workers must be whole again.
k rollout status "$durable" --timeout=300s
k rollout status "$worker" --timeout=300s

# Verify: the same start re-attaches to the build (started = false) and waits
# for it; its numbers are a clean run's. Then the ledger has every row once.
out="$(role ctl data build --config "$corpus_cfg" --id drill-kill9 --durable 127.0.0.1:30733 --wait --json)"
last="$(printf '%s\n' "$out" | tail -1)"
python3 - "$last" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
want = {"n_docs": 169, "n_shards": 3, "train_tokens": 75164, "val_tokens": 6868}
bad = {k: r.get(k) for k, v in want.items() if r.get(k) != v}
if r.get("status") != "COMPLETED" or bad:
    sys.exit(f"respond.sh: drill-kill9 is {r.get('status')} with {bad or 'the clean numbers'}; want COMPLETED and {want}")
print("respond.sh: drill-kill9 COMPLETED with the clean-run numbers")
PY
TL_ARTIFACTS=artifacts role corpus ledger verify --config "$corpus_cfg"

# Document: the runbook the page links to and today's postmortem.
fill() { sed "s/<system>/$name/g" "$1" > "$2"; }
mkdir -p "$learner/docs/runbooks" "$learner/docs/postmortems"
[[ -f "$learner/docs/runbooks/DurableRedeliveriesSpike.md" ]] || \
  fill "$course/ref/docs/runbooks/DurableRedeliveriesSpike.md" "$learner/docs/runbooks/DurableRedeliveriesSpike.md"
pm="$(ls "$course"/ref/docs/postmortems/*-durable-kill9.md | head -1)"
fill "$pm" "$learner/docs/postmortems/$(date -u +%Y-%m-%d)-durable-kill9.md"
echo "respond.sh: $durable and $worker are whole in $ctx/$ns; build and ledger verified; runbook and postmortem written"
