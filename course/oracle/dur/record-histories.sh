#!/usr/bin/env bash
# Regenerates course/fixtures/dur/histories/*.json: the recorded runs of the
# course test workflows (course/tests/go/dur_06/workflows_test.go), produced
# by the reference durable server and SDK in record mode (DUR06_RECORD, see
# course/tests/go/dur_06/record_test.go). The fake clock and the one-task-at-
# a-time driver make every file a deterministic function of the reference.
#
# usage: course/oracle/dur/record-histories.sh   (then update MANIFEST.tsv rows)
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
C="$ROOT/course"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/go"
rsync -a --exclude go.mod --exclude go.sum "$C/ref/go/" "$T/go/"
# go.work provides the contracts module, so its replace and require go away.
grep -v 'openlearn.urmzd.com/tl/contracts' "$C/ref/go/go.mod" > "$T/go/go.mod"
cp "$C/ref/go/go.sum" "$T/go/go.sum"
cat > "$T/go.work" <<WORK
go 1.25.0

use (
	$T/go
	$C/contracts/go
	$C/tests/go
	$C/testkit/go
)
WORK
OUT="$C/fixtures/dur/histories"
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$C/tests/go"
GOWORK="$T/go.work" GOTOOLCHAIN=local GOFLAGS=-count=1 DUR06_RECORD="$OUT" go test ./dur_06/ -run '^$'
ls -l "$OUT"
