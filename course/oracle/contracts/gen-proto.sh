#!/usr/bin/env bash
# Maintainer-only: regenerate the committed protobuf code of the contracts
# (DESIGN 2.7) from course/contracts/proto. Learners never run this; lang.10
# shows the same steps.
#
#   course/oracle/contracts/gen-proto.sh          # rewrite the generated files
#   course/oracle/contracts/gen-proto.sh --check  # fail if they are stale
#
# Needs: buf, go, cargo. The plugins are built at pinned versions into
# $OL_CACHE (default ~/.cache/openlearn)/proto-plugins; nothing is
# installed on PATH.
set -euo pipefail

PROTOC_GEN_GO=v1.36.11
PROTOC_GEN_GO_GRPC=v1.5.1

here="$(cd "$(dirname "$0")" && pwd)"
course="$(cd "$here/../.." && pwd)"
contracts="$course/contracts"
cache="${OL_CACHE:-$HOME/.cache/openlearn}/proto-plugins"
check=0
[ "${1:-}" = "--check" ] && check=1

buf="$(command -v buf || true)"
[ -n "$buf" ] || buf="$(go env GOPATH)/bin/buf"
[ -x "$buf" ] || { echo "gen-proto: buf is not installed" >&2; exit 1; }

mkdir -p "$cache/bin"
for p in "google.golang.org/protobuf/cmd/protoc-gen-go@$PROTOC_GEN_GO" \
         "google.golang.org/grpc/cmd/protoc-gen-go-grpc@$PROTOC_GEN_GO_GRPC"; do
  name="${p%@*}"; name="${name##*/}"; ver="${p#*@}"
  stamp="$cache/bin/$name.$ver"
  if [ ! -f "$stamp" ]; then
    GOBIN="$cache/bin" GOFLAGS= go install "$p"
    touch "$stamp"
  fi
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$buf" build "$contracts/proto" -o "$work/fds.binpb"

# Go
mkdir -p "$work/go"
sed "s#../../contracts/go/gen#$work/go#" "$here/buf.gen.yaml" > "$work/buf.gen.yaml"
(cd "$here" && PATH="$cache/bin:$PATH" "$buf" generate "$contracts/proto" --template "$work/buf.gen.yaml")

# Rust
CARGO_TARGET_DIR="$cache/target" cargo run -q --release \
  --manifest-path "$here/tl-proto-gen/Cargo.toml" -- "$work/fds.binpb" "$work/rust"

if [ "$check" = 1 ]; then
  diff -r "$work/go" "$contracts/go/gen" && diff -r "$work/rust" "$contracts/rust/tl-proto/src/gen" \
    || { echo "gen-proto: generated code is stale; run course/oracle/contracts/gen-proto.sh" >&2; exit 1; }
  echo "gen-proto: generated code is current"
else
  rm -rf "$contracts/go/gen" "$contracts/rust/tl-proto/src/gen"
  cp -R "$work/go" "$contracts/go/gen"
  cp -R "$work/rust" "$contracts/rust/tl-proto/src/gen"
  echo "gen-proto: wrote contracts/go/gen and contracts/rust/tl-proto/src/gen"
fi
