#!/usr/bin/env bash
# Builds and runs the tests. Threads need -pthread on Linux, so this exercise
# ships its own compile line. Built into a temp dir so nothing lands next to the source.
set -euo pipefail
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
c++ -std=c++20 -Wall -Wextra -pthread -o "$out/queue" main.cpp
"$out/queue"
