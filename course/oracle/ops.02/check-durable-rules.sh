#!/usr/bin/env bash
# Maintainer check for the durable alerts of drills ops.02 and ops.03
# (course/ref/entry/deploy/observability/rules/durable.yaml): extracts the
# rule groups from the PrometheusRule and runs `promtool check rules` and
# `promtool test rules` with durable-rules.test.yaml (synthetic series: a
# redelivery storm fires, normal churn does not; one dead letter fires after
# its `for`, none never does). Uses promtool from PATH, else the pinned
# prom/prometheus image of course/tests/obs.03/_promtool.py, with no network.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
course="$(cd "$here/../.." && pwd)"
work="${1:-$(mktemp -d "${OL_SCRATCH:-$course/../.scratchpad}/promtest.XXXXXX")}"
mkdir -p "$work"
uv run --project "$course/harness" --with pyyaml python - "$course/ref/entry/deploy/observability/rules/durable.yaml" "$work/rules.yaml" <<'PY'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1]))
yaml.safe_dump({"groups": doc["spec"]["groups"]}, open(sys.argv[2], "w"))
PY
cp "$here/durable-rules.test.yaml" "$work/test.yaml"
if command -v promtool >/dev/null; then
  (cd "$work" && promtool check rules rules.yaml && promtool test rules test.yaml)
else
  img="prom/prometheus:v3.5.0@sha256:63805ebb8d2b3920190daf1cb14a60871b16fd38bed42b857a3182bc621f4996"
  for args in "check rules rules.yaml" "test rules test.yaml"; do
    # shellcheck disable=SC2086
    docker run --rm --network none -v "$work:/w" -w /w --entrypoint promtool "$img" $args
  done
fi
