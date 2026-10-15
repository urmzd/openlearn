#!/usr/bin/env bash
# check_diagrams.sh: fail when a D2 diagram's rendered SVG is missing or stale.
#
# Diagrams are committed twice: the .d2 source and the .svg it renders to,
# side by side (GitHub and the PDF book show the SVG, people edit the D2).
# Nothing re-renders in CI, so the only thing that can go wrong is someone
# editing the source and forgetting to re-render. Catch that two ways:
#
#   committed:   the SVG's last commit must be no older than the D2's
#   working tree: a D2 with uncommitted edits needs an edited SVG too
#
# The committed check needs history, so CI checks out with fetch-depth: 0.
# Re-render with: d2 --theme=<n> path/x.d2 path/x.svg (see each topic's README).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

rc=0
fail() { echo "FAIL $*"; rc=1; }

while IFS= read -r d2; do
  [[ -f "$d2" ]] || continue  # A tracked diagram may have been moved or deleted.
  svg="${d2%.d2}.svg"
  if [[ ! -f "$svg" ]]; then fail "$d2: no rendered $svg"; continue; fi
  if ! git ls-files --error-unmatch "$svg" >/dev/null 2>&1; then
    git ls-files --error-unmatch "$d2" >/dev/null 2>&1 && fail "$d2: $svg is not committed"
    continue
  fi
  d2_time="$(git log -1 --format=%ct -- "$d2")"
  svg_time="$(git log -1 --format=%ct -- "$svg")"
  if [[ -n "$d2_time" && -n "$svg_time" && "$svg_time" -lt "$d2_time" ]]; then
    fail "$d2: changed after $svg was last rendered"
  fi
  if ! git diff --quiet HEAD -- "$d2" && git diff --quiet HEAD -- "$svg"; then
    fail "$d2: has uncommitted edits but $svg does not"
  fi
done < <(git ls-files --cached --others --exclude-standard '*.d2')

[[ $rc -eq 0 ]] && echo "ok   every D2 diagram has an up-to-date SVG"
exit $rc
