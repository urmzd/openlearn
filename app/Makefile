.PHONY: all init dev build mcp package content test live lint vet fmt typecheck bindings check clean

# The Go binary embeds frontend/dist, so nothing that compiles Go works until
# the frontend has been built at least once. Targets that touch the Go build
# depend on the dist marker below rather than assuming it is there.
DIST := frontend/dist/index.html

# `go install` puts wails in GOPATH/bin, which is on an interactive shell's PATH
# but not necessarily on make's. Resolve it explicitly so a fresh clone works
# without the caller having to fix their environment first.
WAILS_VERSION := v2.11.0

# Wails 2.11 loads the app's packages with a type checker that cannot read
# packages compiled by Go 1.27 ("package time without types was imported"), so
# `wails dev` dies before it starts on a newer toolchain. Pinning the toolchain
# go.mod declares keeps the wails targets working whatever `go` is installed.
export GOTOOLCHAIN ?= go$(shell sed -n 's/^go //p' go.mod)
WAILS := $(shell command -v wails 2>/dev/null || echo $(shell go env GOPATH)/bin/wails)

all: check

init:
	go mod download
	cd frontend && npm install
	go install github.com/wailsapp/wails/v2/cmd/wails@$(WAILS_VERSION)

dev:
	$(WAILS) dev

# vite clears frontend/dist on every build, which takes the tracked .gitkeep
# with it and quietly stages a deletion that breaks //go:embed on a fresh
# clone. Put it back every time rather than relying on anyone noticing.
$(DIST):
	cd frontend && npm install && npm run build
	touch frontend/dist/.gitkeep

# Copies the course packs that ship inside the binary into internal/builtin.
# A build without this step still works; it just has no built-in course and
# relies on a pack found on disk.
PACKS := .. ../packs/languide
content:
	go run ./cmd/bundle -out internal/builtin/content $(PACKS)

build: $(DIST) content
	$(WAILS) build

# The MCP server on its own, without the desktop app's webview dependencies.
# The desktop binary serves the same thing as `openlearn mcp`.
mcp: content
	go build -o build/bin/openlearn-mcp ./cmd/openlearn-mcp

# Signed/notarized macOS bundle and platform installers.
package: $(DIST) content
	$(WAILS) build -clean -platform darwin/universal

test: $(DIST)
	go test ./...

# Exercises problem generation, the coach, and the reviewer against a real
# model. Needs ollama running. Override with MODEL=...
MODEL ?= qwen3.5:9b
live: $(DIST)
	OPENLEARN_LIVE=1 OPENLEARN_MODEL=$(MODEL) go test ./internal/harness -run Live -v

lint: $(DIST)
	golangci-lint run

vet: $(DIST)
	go vet ./...

fmt:
	gofmt -w .

typecheck:
	cd frontend && npx tsc --noEmit

# Regenerates frontend/wailsjs from the Go binding surface in app.go. Run this
# after changing any exported method on App or any type it returns.
bindings:
	$(WAILS) generate module

check: fmt vet lint test typecheck

# Leaves frontend/dist itself in place: the directory is tracked so //go:embed
# resolves on a fresh clone.
clean:
	rm -rf build/bin
	find frontend/dist -mindepth 1 ! -name .gitkeep -delete
