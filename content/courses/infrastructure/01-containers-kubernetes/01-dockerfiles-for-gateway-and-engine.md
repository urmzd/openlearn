# Dockerfiles for the gateway and the engine

## Overview

| | |
|---|---|
| **Module** | `dep.01` · practice · ops · Pass 7 · 3 to 5 h |
| **Solution** | [`engine.Dockerfile`](../../../../course/solutions/entry/deploy/docker/engine.Dockerfile) and [`gateway.Dockerfile`](../../../../course/solutions/entry/deploy/docker/gateway.Dockerfile), hardened: bases pinned by digest, a `HEALTHCHECK`, OCI labels from build args, a static gateway binary; [`.dockerignore`](../../../../course/solutions/entry/.dockerignore) that keeps secrets out of the context |
| **Contract** | none of its own: the images run the roles of [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) on the fixed ports 8000, 8080, and 9464; section 4 is the required layout |
| **Needs** | `dep.00` the tracer images, `gw.07` the gateway with its pure-Go SQLite ledger; reading: [Containers, Kubernetes & Workloads](README.md), `L10.5` the engine's HTTP server |
| **Used by** | `dep.02` loads these images into the kind registry · `dep.03` runs them from the Helm charts. Part of the Pass 7 production platform |
| **Optional depth** | [Dockerfile reference: HEALTHCHECK](https://docs.docker.com/reference/dockerfile/#healthcheck) (free); [OCI image annotations](https://github.com/opencontainers/image-spec/blob/main/annotations.md) (free); [Docker build best practices](https://docs.docker.com/build/building/best-practices/) (free) |

## Key Takeaways

- A **tag moves, a digest does not**: `golang:1.25-alpine` is re-pushed with every patch release, `golang:1.25-alpine@sha256:...` names exactly one image. Pin every base by digest and keep the tag for readers.
- A **HEALTHCHECK** gives Docker its own view of whether the server works, not just whether the process runs; write it in exec form with a tool the base image actually has.
- **OCI labels** (`org.opencontainers.image.title`, `source`, `version`, `revision`) let anyone trace an image back to the commit that built it; `version` and `revision` come from build args so CI fills them.
- Every `ENV` and `ARG` value is **in the image history** for anyone who pulls it: secrets come from a Kubernetes Secret at run time, never from the build.
- The runtime stage ships **one binary on a small base**; a size budget of about twice the minimal image catches a compiler or a build cache shipped by mistake.

## How to read this chapter

Sections 1 to 3 derive what a production image needs beyond the tracer image and work the gateway's runtime stage by hand. Section 4 fixes the artifacts and what a correct pair of images must do. Section 5 walks through both solution Dockerfiles and explains each hardening line. Try to predict each line before reading it.

These commands, from the repo root, find a digest to pin, build both images with their build args, and show what anyone who pulls an image can read:

```bash
docker buildx imagetools inspect golang:1.25-alpine --format '{{json .Manifest.Digest}}'   # the digest to pin
docker build -f deploy/docker/gateway.Dockerfile -t forge-gateway:0.2.0 \
  --build-arg VERSION=0.2.0 --build-arg REVISION="$(git rev-parse HEAD)" .
docker build -f deploy/docker/engine.Dockerfile  -t forge-engine:0.2.0 \
  --build-arg VERSION=0.2.0 --build-arg REVISION="$(git rev-parse HEAD)" .
docker image inspect forge-gateway:0.2.0 --format '{{json .Config.Labels}} {{json .Config.Healthcheck}}'
docker history --no-trunc forge-gateway:0.2.0  # what anyone who pulls it can read
```

---

## 1. Why now

The images of `dep.00` were good enough for a tracer: they built, ran as non-root, and streamed. Pass 7 makes them the thing that runs the serving platform for the rest of the course, rebuilt by CI on every commit (`dep.05`), pulled by the cluster from a registry (`dep.02`), and rolled out by Helm (`dep.03`). Three things now break quietly. `FROM golang:1.24-alpine` builds a different image next month, so a build that passed CI is not the build you shipped. When a pod misbehaves, nothing in the image says which commit it came from. And the gateway now carries a SQLite ledger (`gw.07`): built with cgo, its binary would need a C library at run time that the small base does not have. This module hardens both Dockerfiles against all three and adds the health signal Docker itself can read.

## 2. Principles

`lang.07` and `dep.00` covered stages, layers, the build context, numeric users, and exec form. This section adds only what production images need on top.

### 2.1 Reproducible bases: tags and digests

An image reference `name:tag` is a **mutable pointer**: the registry maps the tag to a manifest, and the publisher moves it whenever they push (Docker's official images re-push `golang:1.25-alpine` for every Go patch release and every Alpine security fix). An image's **digest** is the SHA-256 of its manifest (for a multi-architecture image, of its index), so `name@sha256:<64 hex>` can only ever mean one image. Write both: `FROM golang:1.25-alpine@sha256:1ae0...b59 AS build`. Docker ignores the tag when a digest is present; the tag tells a reader which release the digest is. Moving to a new base is then a one-line change in a commit, reviewed and tested like any other dependency upgrade (`craft.15`).

### 2.2 One static binary

The gateway is Go. With `CGO_ENABLED=0` the linker produces a binary that needs no C library, so it runs on `alpine`, `distroless/static`, or even `scratch`. That is why `gw.07` uses `modernc.org/sqlite` (SQLite translated to Go) instead of `mattn/go-sqlite3` (a cgo binding): with cgo the binary would link the build stage's libc and fail at start-up on a different one. The Rust Candle engine is built without C bindings and uses `debian:bookworm-slim` for its runtime.

### 2.3 HEALTHCHECK

`HEALTHCHECK [--interval=D] [--timeout=D] [--start-period=D] [--retries=N] CMD ["exe", "arg", ...]` makes Docker run the command inside the container every interval. Exit 0 is healthy, anything else is a failure, and `retries` consecutive failures mark the container `unhealthy` (`docker ps`, `docker inspect .State.Health`, compose's `depends_on: condition: service_healthy`). Kubernetes ignores it and uses the chart's probes (`dep.03`), so the image and the chart must agree on the endpoint: `GET /healthz` on the health port 9464.

| Symbol | Meaning | Gateway | Engine |
|---|---|---|---|
| $i$ | interval between checks | 10 s | 10 s |
| $s$ | start period: failures do not count | 5 s | 20 s (model load) |
| $r$ | retries before `unhealthy` | 3 | 3 |
| $T_{\text{detect}}$ | worst-case time to `unhealthy` after a hang, about $s + r \cdot i$ at start, $r \cdot i$ later | 30 s | 30 s |

The command runs in the container, so it can only use what the image has. `alpine` has busybox `wget`; `debian:bookworm-slim` has neither `curl` nor `wget`, but it has `bash`, whose `/dev/tcp/<host>/<port>` redirection opens a TCP connection: write a request, read the status line, grep for 200. Exec form (`CMD [...]`) avoids `/bin/sh`, which a minimal base may not have.

### 2.4 OCI labels and build args

`LABEL key="value"` writes into the image config. The OCI image spec reserves `org.opencontainers.image.*` keys; four make an image traceable:

| Label | Value |
|---|---|
| `org.opencontainers.image.title` | `forge-gateway` |
| `org.opencontainers.image.source` | the repository URL |
| `org.opencontainers.image.version` | the release, from `--build-arg VERSION=0.2.0` |
| `org.opencontainers.image.revision` | the commit, from `--build-arg REVISION=$(git rev-parse HEAD)` |

An `ARG` is scoped to the stage it is declared in: an `ARG` before the first `FROM` is visible only to `FROM` lines, so declare `ARG VERSION` and `ARG REVISION` again inside the final stage before the `LABEL` that uses them.

### 2.5 What an image leaks

Everything in an image is readable by everyone who can pull it: every file in every layer (a file deleted in a later layer is still in the earlier one), every `ENV`, and every build instruction with its arguments (`docker history --no-trunc`). So a secret passed as `--build-arg API_KEY=...`, set with `ENV`, or copied in and deleted later, has shipped. Secrets reach the container at run time from a Kubernetes `Secret` (`dep.00`, `dep.03`); the build never sees them. The context filter matters too: `.dockerignore` keeps `.env`, `*.pem`, and `.git` out, so no `COPY` can sweep them into a layer.

## 3. Worked example by hand

The gateway's runtime stage, line by line:

```dockerfile
FROM alpine:3.19@sha256:6baf43584bcb78f2e5847d1de515f23499913ac9f12bdf834811a3145eb11ca1
ARG VERSION=0.0.0-dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="forge-gateway" \
      org.opencontainers.image.source="https://example.com/forge.git" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
COPY --from=build /out/gateway /usr/local/bin/gateway
USER 10001:10001
EXPOSE 8080 9464
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:9464/healthz"]
ENTRYPOINT ["/usr/local/bin/gateway"]
```

Built with `--build-arg VERSION=0.2.0 --build-arg REVISION=4f1c...`:

1. `FROM` resolves the digest, not the tag: the same bytes as every earlier build.
2. The labels expand to `version=0.2.0` and `revision=4f1c...`; without the `ARG` lines in this stage they would be empty.
3. The gateway starts at $t = 0$. Checks run at 10 s, 20 s, 30 s; until $s = 5$ s a failure would not count. `wget` exits 0 as soon as `/healthz` answers 200, so `docker inspect -f '{{.State.Health.Status}}'` turns from `starting` to `healthy` at the first check, at 10 s.
4. If the gateway later deadlocks with the process alive, three checks fail, each after its 3 s timeout, and the state is `unhealthy` about $3 \times 10 = 30$ s after the hang.
5. Size: alpine's filesystem is about 7.8 MiB and the static gateway binary about 14 MiB, so `docker export` of a container is about 22 MiB. A budget of twice that, 43.8 MiB, leaves room for growth. A single-stage build on `golang:1.25-alpine` would export more than 300 MiB and fail.

A correct image must satisfy exactly these lines and numbers (section 4).

## 4. The artifact and what it must do

| Path | Holds |
|---|---|
| `deploy/docker/engine.Dockerfile` | build stage pinned by digest (Rust, make), runtime stage `debian:bookworm-slim@sha256:...`; `ARG VERSION`, `ARG REVISION`, the four labels, numeric `USER`, `EXPOSE 8000 9464`, a `HEALTHCHECK` on `/healthz` port 9464 (bash `/dev/tcp`), exec-form `ENTRYPOINT` |
| `deploy/docker/gateway.Dockerfile` | build stage pinned by digest with `CGO_ENABLED=0 go build`, runtime stage `alpine@sha256:...` with the same labels, `USER`, `EXPOSE 8080 9464`, a `HEALTHCHECK` with `wget` |
| `.dockerignore` | build outputs and local state (dep.00) plus `.env`, `.git`, and `*.pem` |

A correct pair of images is proven in two steps: the static properties of the Dockerfiles, then a build with `--build-arg VERSION=... --build-arg REVISION=<HEAD>` and a run the way the charts run them: the engine on a model trained by `{tinyllm} train bigram` (as in `dep.00`), the gateway with a probe key, both with `--health-interval 1s`.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| layout present | static | the two Dockerfiles and `.dockerignore` | `dep.03` and CI build these files |
| dep.00's image rules still hold | static | stages, numeric user, exec form, ports | the charts' probes and `runAsNonRoot` |
| bases pinned by digest | static | every external `FROM` has `@sha256:<64 hex>` | reproducible builds; `craft.18` SBOMs |
| health check in exec form | static | one `HEALTHCHECK`, exec form, `--interval`, `--timeout`, `/healthz` on 9464 | Docker sees a hung server |
| OCI labels declared | static | title, source, version, revision; `ARG VERSION` and `ARG REVISION` in the final stage | traceable images |
| gateway builds a static binary | static | `CGO_ENABLED=0 go build` | the SQLite ledger runs on alpine |
| no secrets in Dockerfiles or context | static | no secret-looking `ENV`/`ARG` values or key material; `.env`, `.git`, `*.pem` ignored | nothing secret in a layer |
| images build | docker | both images build with the build args | |
| images carry the labels | docker | labels present; version and revision equal the build args | |
| images run as numeric non-root | boundary | the image config's user is a numeric uid other than 0 | `runAsNonRoot` |
| no secrets in history or env | docker | `docker history --no-trunc` and the config's `Env` hold no secret | a pulled image leaks nothing |
| images fit the size budget | boundary | `docker export` bytes at most about twice the minimal image | no toolchain in the runtime stage |
| containers report healthy and stream | conformance | both containers turn `healthy`, then a completion streams engine to gateway | the health command works in its base; packaging broke nothing |

## 5. The solution, step by step

### 5.1 From first principles

1. **A build must be repeatable from its commit.** The only input a Dockerfile does not pin by itself is the base image, and a tag is a mutable pointer (2.1). So every `FROM` carries a digest; the tag stays as documentation.
2. **The runtime stage must run with what it has.** The gateway's base has no libc that matches a cgo build, so the gateway binary must be static: `CGO_ENABLED=0`, which in turn requires the pure-Go SQLite driver of `gw.07`. The engine links glibc, so its runtime base is the same Debian release as its builder.
3. **Docker needs its own health signal.** A process that is alive but hung looks fine to `docker ps`. A `HEALTHCHECK` asking the same `/healthz` the chart probes gives Docker and Kubernetes one definition of healthy. It must use a tool the base has: busybox `wget` on alpine, bash's `/dev/tcp` on Debian slim.
4. **An image must name its source.** Labels are the only metadata that travels with the image to every registry and node; `version` and `revision` vary per build, so they come from `ARG`, re-declared in the final stage because `ARG` is stage-scoped.
5. **Nothing secret may enter the build.** History, `ENV`, and every layer are public to whoever pulls (2.5). So the Dockerfiles have no secret `ARG`/`ENV`, and `.dockerignore` keeps `.env`, `*.pem`, and `.git` out of the context.

### 5.2 The gateway image

```dockerfile
# Base images pinned by digest: a tag can be re-pushed, a digest cannot.
# `docker buildx imagetools inspect <tag>` prints it. The tag stays for humans.
FROM golang:1.25-alpine@sha256:1ae0735f00daffa3aaf1363a5184c0d2dc55c78e3db4ec70241cdac97bf84b59 AS build
WORKDIR /src
# The module files first: `go mod download` stays cached until they change.
# The contracts module (generated gRPC code) builds in workspace mode,
# so the small contracts/ tree comes along and a go.work names both modules.
COPY contracts/ contracts/
COPY go/go.mod go/go.sum* go/
RUN go work init ./go ./contracts/go && cd go && go mod download
COPY go/ go/
# CGO_ENABLED=0: a static binary. The usage ledger (gw.07) uses the pure-Go
# SQLite driver, so nothing needs a C toolchain or libc at run time.
RUN cd go && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM alpine:3.19@sha256:6baf43584bcb78f2e5847d1de515f23499913ac9f12bdf834811a3145eb11ca1
# ARG is stage-scoped: declared here, after this FROM, so LABEL can see it.
# The defaults mark a build that did not pass them (a laptop build).
ARG VERSION=0.0.0-dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="forge-gateway" \
      org.opencontainers.image.description="The forge gateway: keys, limits, cache, routing, SSE proxy, usage ledger" \
      org.opencontainers.image.source="https://example.com/forge.git" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
# Only the binary crosses from the build stage: no compiler, no module cache.
COPY --from=build /out/gateway /usr/local/bin/gateway
USER 10001:10001
# 8080: the API; 9464: /healthz and /readyz.
EXPOSE 8080 9464
# busybox wget is part of alpine; exec form needs no shell.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:9464/healthz"]
ENTRYPOINT ["/usr/local/bin/gateway"]
```

`-trimpath` removes local file system paths from the binary (they would otherwise leak your build directory and make two builds differ), and `-s -w` drops the symbol table and DWARF debug data, which the server never needs at run time and which make up a large share of an unstripped Go binary. The ARG defaults are deliberate: a label of `unknown` is honest, an empty one looks like a bug.

Full solution: [`gateway.Dockerfile`](../../../../course/solutions/entry/deploy/docker/gateway.Dockerfile).

### 5.3 The engine image

```dockerfile
FROM rust:1.88-slim-bookworm@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89 AS build
WORKDIR /src
# Least to most often changed: contracts, then Rust.
COPY contracts/ contracts/
COPY rust/ rust/
RUN cargo build --release --manifest-path rust/Cargo.toml -p tl-serve \
 && cp rust/target/release/tl-serve /tl-serve

# Same Debian release as the builder: tl-serve links this glibc.
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587
ARG VERSION=0.0.0-dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="forge-engine" \
      # ... description, source, licenses as in the gateway
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
COPY --from=build /tl-serve /usr/local/bin/tl-serve
# Numeric, so Kubernetes runAsNonRoot can verify it.
USER 10001:10001
# 8000: POST /v1/completions; 9464: /healthz.
EXPOSE 8000 9464
# Docker's own health signal; Kubernetes uses the chart's probes instead.
# The slim base has no curl, so bash asks /healthz itself:
#   open fd 3 as a TCP socket, send a minimal HTTP/1.0 request,
#   read the status line, succeed only on " 200 ".
# start-period 20s: the model load must not count as failures.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/9464 && printf 'GET /healthz HTTP/1.0\\r\\n\\r\\n' >&3 && head -n 1 <&3 | grep -q ' 200 '"]
# Exec form: tl-serve is PID 1, gets SIGTERM, and receives the chart's args.
ENTRYPOINT ["/usr/local/bin/tl-serve"]
```

The health command is exec form (`["bash", "-c", ...]`), so Docker runs bash directly rather than through `/bin/sh -c`; bash is needed anyway for `/dev/tcp`, which is a bash feature, not a device. HTTP/1.0 makes the server close the connection after one response, so `head -n 1` never waits on a kept-alive socket.

Full solution: [`engine.Dockerfile`](../../../../course/solutions/entry/deploy/docker/engine.Dockerfile).

### 5.4 The context filter

The `.dockerignore` of `dep.00` already lists `.git`, `.env`, and `*.pem` next to the build outputs, so no `COPY` can sweep a key, a credentials file, or the whole git history (which may contain an old secret) into a layer. Full solution: [`.dockerignore`](../../../../course/solutions/entry/.dockerignore).

### 5.5 Why this solution

- **Digest pins over tag pins or `latest`.** Only a digest makes "the image CI tested" and "the image we shipped" the same bytes. The cost is manual bumps, which Renovate or Dependabot automate (Going further).
- **alpine over `scratch` or distroless for the gateway.** `scratch` would work for a static binary but leaves nothing to run a `HEALTHCHECK` with and no CA certificates; distroless has no `wget`. alpine costs about 8 MiB and gives busybox. The tradeoff is a slightly larger attack surface.
- **Debian slim over alpine for the engine.** Rust binaries link glibc by default; running on musl would mean a different target and different performance characteristics for the numeric code.
- **bash `/dev/tcp` over installing curl.** Installing curl adds a package layer and its CVEs to every engine image for the sake of one health check.
- **Labels from build args over hard-coded values.** The Dockerfile stays the same for every release; CI supplies the variable part (`dep.05`).

### 5.6 Explain it back

1. Docker ignores the tag when a digest is present. Why keep the tag at all?
   <details><summary>Answer</summary>For readers and tools: a reviewer can see that the bump goes from Go 1.24 to 1.25, and Renovate uses the tag to find the newer digest. The digest is what makes the build reproducible; the tag is what makes it reviewable.</details>
2. Why does the gateway Dockerfile declare `ARG VERSION` after the second `FROM` and not at the top?
   <details><summary>Answer</summary>An <code>ARG</code> is scoped to the stage it is declared in, and one before the first <code>FROM</code> is visible only to <code>FROM</code> lines. Declared at the top, <code>${VERSION}</code> in the final <code>LABEL</code> would expand to an empty string.</details>
3. What would happen at start-up if the gateway were built with a cgo SQLite driver and no `CGO_ENABLED=0`?
   <details><summary>Answer</summary>The binary would be dynamically linked against the build stage's C library. On the runtime base it would fail to start (a missing or mismatched loader or <code>libc</code>), even though the build succeeded.</details>
4. The engine's `HEALTHCHECK` has a 20 s start period and the gateway's 5 s. Why the difference?
   <details><summary>Answer</summary>The engine loads a model before <code>/healthz</code> answers, and failures during the start period do not count towards <code>unhealthy</code>. The gateway has nothing to load, so a short start period detects a broken gateway sooner.</details>
5. A teammate deletes a `.env` file in a later `RUN rm .env` after `COPY . .`. Is the image clean?
   <details><summary>Answer</summary>No. The file is still in the layer created by <code>COPY</code>; the later layer only hides it. The fix is to keep it out of the context (<code>.dockerignore</code>), so it never enters any layer.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. `FROM golang:1.25-alpine` without a digest | a rebuild of an old commit produces a different image | every `FROM` carries `@sha256:...` (5.2, 5.3) |
| 2. `HEALTHCHECK CMD curl ...` on a base without curl | the container is `unhealthy` forever while serving fine | busybox `wget` on alpine, bash `/dev/tcp` on Debian slim (5.2, 5.3) |
| 3. one stage, or `COPY --from=build /src` instead of the binary | a 300 MiB to 2 GiB image with the compiler and build cache | `COPY --from=build` copies only the binary (5.2) |
| 4. `ARG API_KEY` or `ENV TL_GATEWAY_PEPPER=...` | the secret is in `docker history` for everyone who pulls | no secret `ARG` or `ENV`; secrets arrive at run time from a Kubernetes Secret (5.1, step 5) |
| 5. `ARG VERSION` only before the first `FROM` | the version label is empty | `ARG VERSION` and `ARG REVISION` re-declared in the final stage (5.2) |
| 6. a cgo SQLite driver, or `go build` without `CGO_ENABLED=0` | the gateway exits at start on alpine: the binary wants the build stage's libc | `CGO_ENABLED=0` with the pure-Go SQLite driver (5.2) |
| 7. `HEALTHCHECK CMD wget ...` in shell form | needs `/bin/sh`; on a distroless base the check cannot even start | both checks are exec form `CMD [...]` (5.2, 5.3) |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.00` | the tracer images these Dockerfiles harden |
| Back | `gw.07` | the pure-Go SQLite ledger that makes a static gateway possible |
| Forward | `dep.02` | tags these images into the local registry `localhost:5001` that kind pulls from |
| Forward | `dep.03` | the charts run them; their probes ask the same `/healthz` |
| Forward | `dep.05`, `craft.18` | CI builds them with `VERSION` and `REVISION`; SBOMs list the pinned bases |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| pinned digests by hand | Renovate, Dependabot | automated pull requests that bump digests and tags | [Renovate Docker support](https://docs.renovatebot.com/docker/) (free) |
| alpine plus one binary | distroless, Chainguard images | no shell or package manager at all; fewer CVEs | [distroless](https://github.com/GoogleContainerTools/distroless) (free) |
| OCI labels | SLSA provenance, cosign signatures | signed, verifiable statements of how and from what an image was built | [SLSA](https://slsa.dev/) (free), [Sigstore cosign](https://docs.sigstore.dev/cosign/signing/overview/) (free) |
