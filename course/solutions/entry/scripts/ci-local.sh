#!/usr/bin/env bash
# craft.01 reference: the CI gates of .github/workflows/ci.yml run locally,
# for a repo with no GitHub remote: commit-lint on HEAD, then the native test
# suites. Add each language's test command to native-tests as it lands.
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

