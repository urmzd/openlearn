#!/usr/bin/env bash
# craft.01 reference: the three CI gates run locally, for a repo with no
# GitHub remote. system.toml declares it as `[ci] local = [["bash", "scripts/ci-local.sh"]]`,
# which is what MS-P0's ci-status step runs in that case. Set OL_BIN to your
# openlearn clone's practice/bin/ol.
# SOLUTION-BEGIN craft.01
set -euo pipefail

echo "== commit-lint (HEAD)"
msg="$(mktemp)"
trap 'rm -f "$msg"' EXIT
git log -1 --format=%B HEAD > "$msg"
.githooks/commit-msg "$msg"

echo "== native-tests"
make -s -C primers/lang.02 && primers/lang.02/greet ci
make -s -C primers/lang.02 clean
(cd primers/lang.01 && uv run --quiet python -c "import bigram, worksheet")

echo "== course-check"
# The openlearn checkout whose ol grades you: $OL_BIN when set (your local
# clone's practice/bin/ol), else the one the CI recipe clones (ol course ci).
ss="${OL_BIN:-.ol/openlearn/practice/bin/ol}"
[ -x "$ol" ] || { echo "no ol at $ss: set OL_BIN, or clone openlearn as \`ol course ci\` shows" >&2; exit 5; }
OL_COURSE_HOME="$PWD" "$ol" check --all --ci
# SOLUTION-END
