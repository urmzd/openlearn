#!/usr/bin/env bash
# Scripted responder for drill ops.01 kill-decode (DESIGN 5.10, "Drills in
# CI"): the reference response, run by the nightly kind job against the
# reference system after `ol drill start kill-decode`, before `ol drill end`.
#
#   respond.sh [LEARNER_DIR]     (default: $OL_COURSE_HOME, else the current dir)
#
# The fault heals itself (the Deployment replaces the pod), so the response
# is the runbook's: diagnose, confirm the replacement becomes Ready, and wait
# for the alert's short window to clear. It refuses any context that is not
# kind- or k3d- (the drill safety gate's rule) and writes the runbook and
# today's postmortem from course/ref/docs with <system> filled in.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
course="$(cd "$here/../.." && pwd)"
learner="$(cd "${1:-${OL_COURSE_HOME:-$PWD}}" && pwd)"

read -r name ctx ns target < <(python3 - "$learner/system.toml" <<'PY'
import sys, tomllib
s = tomllib.load(open(sys.argv[1], "rb"))
d = s.get("deploy", {})
print(s.get("system", {}).get("name", ""), d.get("kube_context", ""), d.get("namespace", ""),
      d.get("services", {}).get("decode", ""))
PY
)
for v in name ctx ns target; do
  [[ -n "${!v}" ]] || { echo "respond.sh: system.toml lacks the value for $v ([system].name, [deploy].kube_context, namespace, services.decode)" >&2; exit 2; }
done
case "$ctx" in kind-*|k3d-*) ;; *) echo "respond.sh: refusing context $ctx (not kind- or k3d-)" >&2; exit 2 ;; esac

k() { kubectl --context "$ctx" -n "$ns" "$@"; }

# Diagnose as the runbook says, so the CI log shows the same evidence a person sees.
k get pods -o wide || true
k get events --sort-by=.lastTimestamp | tail -20 || true

# Mitigate: the replacement must become Ready; then the short window clears.
k rollout status "$target" --timeout=300s
sleep 60

# Document: the runbook the page links to and today's postmortem.
fill() { sed "s/<system>/$name/g" "$1" > "$2"; }
mkdir -p "$learner/docs/runbooks" "$learner/docs/postmortems"
[[ -f "$learner/docs/runbooks/TTFTBudgetBurnFast.md" ]] || \
  fill "$course/ref/docs/runbooks/TTFTBudgetBurnFast.md" "$learner/docs/runbooks/TTFTBudgetBurnFast.md"
pm="$(ls "$course"/ref/docs/postmortems/*-kill-decode.md | head -1)"
fill "$pm" "$learner/docs/postmortems/$(date -u +%Y-%m-%d)-kill-decode.md"
echo "respond.sh: $target is whole again in $ctx/$ns; runbook and postmortem written"
