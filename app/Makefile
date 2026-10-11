.PHONY: all init dev build package test live lint vet fmt typecheck bindings check record clean

# The Go binary embeds frontend/dist, so nothing that compiles Go works until
# the frontend has been built at least once. Targets that touch the Go build
# depend on the dist marker below rather than assuming it is there.
DIST := frontend/dist/index.html

# `go install` puts wails in GOPATH/bin, which is on an interactive shell's PATH
# but not necessarily on make's. Resolve it explicitly so a fresh clone works
# without the caller having to fix their environment first.
WAILS_VERSION := v2.11.0
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

build: $(DIST)
	$(WAILS) build

# Signed/notarized macOS bundle and platform installers.
package: $(DIST)
	$(WAILS) build -clean -platform darwin/universal

test: $(DIST)
	go test ./...

# Exercises problem generation, the coach, and the reviewer against a real
# model. Needs ollama running. Override with MODEL=...
MODEL ?= qwen3.5:9b
live: $(DIST)
	WHITEBOARDY_LIVE=1 WHITEBOARDY_MODEL=$(MODEL) go test ./internal/harness -run Live -v

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

# Recaptures the reproducible showcase screens. teasr starts `wails dev` itself,
# so nothing may be listening on :34115. See the comment in teasr.toml for which
# screens it covers and which need a driven session.
record: $(DIST)
	teasr run

# Leaves frontend/dist itself in place: the directory is tracked so //go:embed
# resolves on a fresh clone.
clean:
	rm -rf build/bin
	find frontend/dist -mindepth 1 ! -name .gitkeep -delete
