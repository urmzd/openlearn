#!/usr/bin/env bash
# Builds and runs the tests. Threads need -pthread, so this exercise ships its
# own compile line rather than relying on a generic one.
#
# Built into a temp dir so nothing lands next to the source.
set -euo pipefail
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
cc -std=c11 -Wall -Wextra -pthread -o "$out/pool" ./*.c
"$out/pool"
