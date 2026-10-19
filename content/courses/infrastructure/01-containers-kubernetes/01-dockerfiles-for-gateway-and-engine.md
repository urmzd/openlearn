# Dockerfiles for the gateway and the engine

The images from [the tracer deploy](00-tracer-deploy.md) were good enough for a tracer: they built, ran as non-root, and streamed. From here on they become the thing that runs the serving platform, rebuilt by CI on every commit ([CI for the platform repo](05-ci-for-your-repo.md)), pulled by the cluster from a registry ([kind cluster, local registry, and port mappings](02-kind-cluster-and-local-registry.md)), and rolled out by Helm ([Helm charts and the observability stack](03-helm-charts-and-observability-stack.md)). Under that load, three things break quietly.

First, `FROM golang:1.24-alpine` builds a different image next month, so a build that passed CI is not the build you shipped. Second, when a Pod misbehaves, nothing in the image says which commit it came from. Third, the gateway now carries a SQLite usage ledger, built in [Usage ledger, metering, and admin API](../../ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md) in *AI Platform Engineering*. Built with cgo, its binary would need a C library at run time that the small base does not have. This section hardens both Dockerfiles against all three problems and adds a health signal that Docker itself can read.

It assumes the tracer images and the stage, layer, context, numeric-user, and exec-form rules from [the tracer deploy](00-tracer-deploy.md) and from [Containers and Kubernetes](../../programming-languages/02-language-and-tool-primers/07-containers-and-kubernetes.md) in *Programming Languages*. Background on the engine's HTTP server is in [tl-serve: an OpenAI-compatible server on tokio](../../build-an-llm/p10-serving/05-openai-server-on-tokio.md) in *Build Your Own LLM System*. There is no new contract: the images run the roles of [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) on the fixed ports 8000, 8080, and 9464.

> **Learning objectives**
> - Pin every base image by digest while keeping the tag readable, and explain why a tag alone is not reproducible.
> - Write an exec-form `HEALTHCHECK` that uses only tools the base image ships, and compute its worst-case detection time.
> - Stamp images with OCI labels from build args, declared in the right stage.
> - Explain everything a pulled image leaks, and keep secrets out of layers, `ENV`, and history.
> - Build a static Go binary and size-check a runtime image against a budget.

## The commands you will use

These commands, run from the repo root, find a digest to pin, build both images with their build args, and show what anyone who pulls an image can read:

```bash
docker buildx imagetools inspect golang:1.25-alpine --format '{{json .Manifest.Digest}}'   # the digest to pin
docker build -f deploy/docker/gateway.Dockerfile -t forge-gateway:0.2.0 \
  --build-arg VERSION=0.2.0 --build-arg REVISION="$(git rev-parse HEAD)" .
docker build -f deploy/docker/engine.Dockerfile  -t forge-engine:0.2.0 \
  --build-arg VERSION=0.2.0 --build-arg REVISION="$(git rev-parse HEAD)" .
docker image inspect forge-gateway:0.2.0 --format '{{json .Config.Labels}} {{json .Config.Healthcheck}}'
docker history --no-trunc forge-gateway:0.2.0  # what anyone who pulls it can read
```

The rest of the section explains what each of those outputs should show and why.

## Reproducible bases: tags and digests

An image reference `name:tag` is a **mutable pointer**. The registry maps the tag to a manifest, and the publisher moves it whenever they push; Docker's official images re-push `golang:1.25-alpine` for every Go patch release and every Alpine security fix. An image's **digest** is the SHA-256 of its manifest (for a multi-architecture image, of its index), so `name@sha256:<64 hex>` can only ever mean one image. A tag moves; a digest does not.

Write both: `FROM golang:1.25-alpine@sha256:1ae0...b59 AS build`. Docker ignores the tag when a digest is present, but the tag tells a reader which release the digest is. Moving to a new base is then a one-line change in a commit, reviewed and tested like any other dependency upgrade, as [Dependency upgrades](../../software-craftsmanship/07-maintenance/03-dependency-upgrades.md) in *Software Craftsmanship* describes.

## One static binary

Pinning fixes the inputs. The next question is whether the output can run on the small base you chose. The gateway is Go. With `CGO_ENABLED=0` the linker produces a binary that needs no C library, so it runs on `alpine`, `distroless/static`, or even `scratch`. That is why the usage ledger uses `modernc.org/sqlite` (SQLite translated to Go) instead of `mattn/go-sqlite3` (a cgo binding): with cgo the binary would link the build stage's libc and fail at start-up on a different one. The Rust Candle engine is built without C bindings and uses `debian:bookworm-slim` for its runtime, because Rust links glibc dynamically.

## HEALTHCHECK

A running process is not the same as a working server. `HEALTHCHECK [--interval=D] [--timeout=D] [--start-period=D] [--retries=N] CMD ["exe", "arg", ...]` makes Docker run the command inside the container every interval. Exit 0 is healthy, anything else is a failure, and `retries` consecutive failures mark the container `unhealthy`, which shows up in `docker ps`, `docker inspect .State.Health`, and compose's `depends_on: condition: service_healthy`. Kubernetes ignores it and uses the chart's probes instead, so the image and the chart must agree on the endpoint: `GET /healthz` on the health port 9464.

The timing parameters for the two images are:

| Symbol | Meaning | Gateway | Engine |
|---|---|---|---|
| $i$ | interval between checks | 10 s | 10 s |
| $s$ | start period: failures do not count | 5 s | 20 s (model load) |
| $r$ | retries before `unhealthy` | 3 | 3 |
| $T_{\text{detect}}$ | worst-case time to `unhealthy` after a hang, about $s + r \cdot i$ at start, $r \cdot i$ later | 30 s | 30 s |

The command runs in the container, so it can only use what the image has. `alpine` has busybox `wget`. `debian:bookworm-slim` has neither `curl` nor `wget`, but it has `bash`, whose `/dev/tcp/<host>/<port>` redirection opens a TCP connection: write a request, read the status line, grep for 200. Exec form (`CMD [...]`) avoids `/bin/sh`, which a minimal base may not have.

## OCI labels and build args

A health signal tells you whether an image works; labels tell you where it came from. `LABEL key="value"` writes into the image config. The OCI image spec reserves `org.opencontainers.image.*` keys, and four of them make an image traceable back to the commit that built it:

| Label | Value |
|---|---|
| `org.opencontainers.image.title` | `forge-gateway` |
| `org.opencontainers.image.source` | the repository URL |
| `org.opencontainers.image.version` | the release, from `--build-arg VERSION=0.2.0` |
| `org.opencontainers.image.revision` | the commit, from `--build-arg REVISION=$(git rev-parse HEAD)` |

`version` and `revision` come from build args so that CI can fill them. There is one trap: an `ARG` is scoped to the stage it is declared in, and an `ARG` before the first `FROM` is visible only to `FROM` lines. So you declare `ARG VERSION` and `ARG REVISION` again inside the final stage, before the `LABEL` that uses them.

## What an image leaks

Labels are metadata you want everyone to read. The flip side is that everything else in an image is readable too, by everyone who can pull it: every file in every layer (a file deleted in a later layer is still in the earlier one), every `ENV`, and every build instruction with its arguments (`docker history --no-trunc`). A secret passed as `--build-arg API_KEY=...`, set with `ENV`, or copied in and deleted later, has shipped.

So secrets reach the container at run time from a Kubernetes `Secret`, as in [the tracer deploy](00-tracer-deploy.md) and [Helm charts and the observability stack](03-helm-charts-and-observability-stack.md); the build never sees them. The context filter matters too: `.dockerignore` keeps `.env`, `*.pem`, and `.git` out, so no `COPY` can sweep them into a layer.

## Working the gateway's runtime stage by hand

With the principles in place, read the gateway's runtime stage line by line:

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

Suppose you build it with `--build-arg VERSION=0.2.0 --build-arg REVISION=4f1c...`.

1. `FROM` resolves the digest, not the tag: the same bytes as every earlier build.
2. The labels expand to `version=0.2.0` and `revision=4f1c...`. Without the `ARG` lines in this stage they would be empty.
3. The gateway starts at $t = 0$. Checks run at 10 s, 20 s, 30 s; until $s = 5$ s a failure would not count. `wget` exits 0 as soon as `/healthz` answers 200, so `docker inspect -f '{{.State.Health.Status}}'` turns from `starting` to `healthy` at the first check, at 10 s.
4. If the gateway later deadlocks with the process alive, three checks fail, each after its 3 s timeout, and the state is `unhealthy` about $3 \times 10 = 30$ s after the hang.
5. Size: alpine's filesystem is about 7.8 MiB and the static gateway binary about 14 MiB, so `docker export` of a container is about 22 MiB. A budget of twice that, 43.8 MiB, leaves room for growth. A single-stage build on `golang:1.25-alpine` would export more than 300 MiB and fail the budget.

The runtime stage ships one binary on a small base, and a size budget of about twice the minimal image is what catches a compiler or a build cache shipped by mistake. A correct image must satisfy exactly these lines and numbers.

## The files and what they must do

Three files carry the work:

| Path | Holds |
|---|---|
| `deploy/docker/engine.Dockerfile` | build stage pinned by digest (Rust, make), runtime stage `debian:bookworm-slim@sha256:...`; `ARG VERSION`, `ARG REVISION`, the four labels, numeric `USER`, `EXPOSE 8000 9464`, a `HEALTHCHECK` on `/healthz` port 9464 (bash `/dev/tcp`), exec-form `ENTRYPOINT` |
| `deploy/docker/gateway.Dockerfile` | build stage pinned by digest with `CGO_ENABLED=0 go build`, runtime stage `alpine@sha256:...` with the same labels, `USER`, `EXPOSE 8080 9464`, a `HEALTHCHECK` with `wget` |
| `.dockerignore` | build outputs and local state (from the tracer deploy) plus `.env`, `.git`, and `*.pem` |

You prove a correct pair of images in two steps. First check the static properties of the Dockerfiles; then build with `--build-arg VERSION=... --build-arg REVISION=<HEAD>` and run the images the way the charts run them: the engine on a model trained by `{tinyllm} train bigram` (as in the tracer deploy), the gateway with a probe key, both with `--health-interval 1s`.

### What a correct solution must handle

The static checks come first. The two Dockerfiles and `.dockerignore` must be present, because the Helm charts and CI build exactly these files. The tracer's image rules must still hold (stages, numeric user, exec form, ports), because the charts' probes and `runAsNonRoot` depend on them. Every external `FROM` must carry `@sha256:<64 hex>`, which makes builds reproducible and lets the SBOMs of [SBOM and supply chain](../../software-craftsmanship/08-security/02-sbom-and-supply-chain.md) name the bases. There must be one `HEALTHCHECK`, in exec form, with `--interval`, `--timeout`, and `/healthz` on 9464, so Docker sees a hung server. The labels (title, source, version, revision) must be declared, with `ARG VERSION` and `ARG REVISION` in the final stage. The gateway must build with `CGO_ENABLED=0 go build`, so the SQLite ledger runs on alpine. And there must be no secret-looking `ENV` or `ARG` values or key material, with `.env`, `.git`, and `*.pem` ignored, so nothing secret ends up in a layer.

Then the Docker checks. Both images must build with the build args and carry the labels, with version and revision equal to the build args. The image config's user must be a numeric uid other than 0, for `runAsNonRoot`. `docker history --no-trunc` and the config's `Env` must hold no secret, so a pulled image leaks nothing. The `docker export` size must be at most about twice the minimal image, proving no toolchain sits in the runtime stage. Finally, both containers must turn `healthy` and then stream a completion from engine to gateway, which proves the health command works in its base and that packaging broke nothing.

## Building it step by step

### Reasoning from first principles

Five requirements drive the hardening.

1. **A build must be repeatable from its commit.** The only input a Dockerfile does not pin by itself is the base image, and a tag is a mutable pointer. So every `FROM` carries a digest; the tag stays as documentation.
2. **The runtime stage must run with what it has.** The gateway's base has no libc that matches a cgo build, so the gateway binary must be static: `CGO_ENABLED=0`, which in turn requires the pure-Go SQLite driver of the usage ledger. The engine links glibc, so its runtime base is the same Debian release as its builder.
3. **Docker needs its own health signal.** A process that is alive but hung looks fine to `docker ps`. A `HEALTHCHECK` asking the same `/healthz` the chart probes gives Docker and Kubernetes one definition of healthy. It must use a tool the base has: busybox `wget` on alpine, bash's `/dev/tcp` on Debian slim.
4. **An image must name its source.** Labels are the only metadata that travels with the image to every registry and node. `version` and `revision` vary per build, so they come from `ARG`, re-declared in the final stage because `ARG` is stage-scoped.
5. **Nothing secret may enter the build.** History, `ENV`, and every layer are public to whoever pulls. So the Dockerfiles have no secret `ARG` or `ENV`, and `.dockerignore` keeps `.env`, `*.pem`, and `.git` out of the context.

### The gateway image

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

`-trimpath` removes local file system paths from the binary; they would otherwise leak your build directory and make two builds differ. `-s -w` drops the symbol table and DWARF debug data, which the server never needs at run time and which make up a large share of an unstripped Go binary. The `ARG` defaults are deliberate: a label of `unknown` is honest, while an empty one looks like a bug. The full solution is [`gateway.Dockerfile`](../../../../course/solutions/entry/deploy/docker/gateway.Dockerfile).

### The engine image

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

The health command is exec form (`["bash", "-c", ...]`), so Docker runs bash directly rather than through `/bin/sh -c`. Bash is needed anyway for `/dev/tcp`, which is a bash feature, not a device. HTTP/1.0 makes the server close the connection after one response, so `head -n 1` never waits on a kept-alive socket. The full solution is [`engine.Dockerfile`](../../../../course/solutions/entry/deploy/docker/engine.Dockerfile).

### The context filter

The `.dockerignore` from the tracer deploy already lists `.git`, `.env`, and `*.pem` next to the build outputs, so no `COPY` can sweep a key, a credentials file, or the whole git history (which may contain an old secret) into a layer. The full solution is [`.dockerignore`](../../../../course/solutions/entry/.dockerignore).

### Why this design

- **Digest pins over tag pins or `latest`.** Only a digest makes "the image CI tested" and "the image we shipped" the same bytes. The cost is manual bumps, which Renovate or Dependabot automate.
- **alpine over `scratch` or distroless for the gateway.** `scratch` would work for a static binary but leaves nothing to run a `HEALTHCHECK` with and no CA certificates; distroless has no `wget`. alpine costs about 8 MiB and gives busybox. The tradeoff is a slightly larger attack surface.
- **Debian slim over alpine for the engine.** Rust binaries link glibc by default; running on musl would mean a different target and different performance characteristics for the numeric code.
- **bash `/dev/tcp` over installing curl.** Installing curl adds a package layer and its CVEs to every engine image for the sake of one health check.
- **Labels from build args over hard-coded values.** The Dockerfile stays the same for every release; CI supplies the variable part.

## What can go wrong

Each of these mistakes produces an image that builds and often even runs, which is why they survive review:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `FROM golang:1.25-alpine` without a digest | a rebuild of an old commit produces a different image | every `FROM` carries `@sha256:...` |
| `HEALTHCHECK CMD curl ...` on a base without curl | the container is `unhealthy` forever while serving fine | busybox `wget` on alpine, bash `/dev/tcp` on Debian slim |
| one stage, or `COPY --from=build /src` instead of the binary | a 300 MiB to 2 GiB image with the compiler and build cache | `COPY --from=build` copies only the binary |
| `ARG API_KEY` or `ENV TL_GATEWAY_PEPPER=...` | the secret is in `docker history` for everyone who pulls | no secret `ARG` or `ENV`; secrets arrive at run time from a Kubernetes Secret |
| `ARG VERSION` only before the first `FROM` | the version label is empty | `ARG VERSION` and `ARG REVISION` re-declared in the final stage |
| a cgo SQLite driver, or `go build` without `CGO_ENABLED=0` | the gateway exits at start on alpine: the binary wants the build stage's libc | `CGO_ENABLED=0` with the pure-Go SQLite driver |
| `HEALTHCHECK CMD wget ...` in shell form | needs `/bin/sh`; on a distroless base the check cannot even start | both checks are exec form `CMD [...]` |

> **Warning:** Deleting a secret in a later layer does not remove it. The bytes remain in the layer that added them, and anyone who pulls the image can extract that layer.

## Where this leads

These Dockerfiles harden the tracer images from [the tracer deploy](00-tracer-deploy.md), and the static gateway is possible only because of the pure-Go SQLite ledger from [Usage ledger, metering, and admin API](../../ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md). Next, [kind cluster, local registry, and port mappings](02-kind-cluster-and-local-registry.md) tags these images into the local registry `localhost:5001` that kind pulls from. [Helm charts and the observability stack](03-helm-charts-and-observability-stack.md) runs them, with probes that ask the same `/healthz`. [CI for the platform repo](05-ci-for-your-repo.md) builds them with `VERSION` and `REVISION`, and [SBOM and supply chain](../../software-craftsmanship/08-security/02-sbom-and-supply-chain.md) in *Software Craftsmanship* lists the pinned bases in SBOMs.

Each piece here has a production equivalent worth knowing:

| Your piece | Production equivalent | What it adds |
|---|---|---|
| pinned digests by hand | Renovate, Dependabot | automated pull requests that bump digests and tags |
| alpine plus one binary | distroless, Chainguard images | no shell or package manager at all; fewer CVEs |
| OCI labels | SLSA provenance, cosign signatures | signed, verifiable statements of how and from what an image was built |

### Further reading

- [Dockerfile reference: HEALTHCHECK](https://docs.docker.com/reference/dockerfile/#healthcheck)
- [OCI image annotations](https://github.com/opencontainers/image-spec/blob/main/annotations.md)
- [Docker build best practices](https://docs.docker.com/build/building/best-practices/)
- [Renovate Docker support](https://docs.renovatebot.com/docker/)
- [distroless](https://github.com/GoogleContainerTools/distroless)
- [SLSA](https://slsa.dev/)
- [Sigstore cosign](https://docs.sigstore.dev/cosign/signing/overview/)

## Key terms

- **tag**: a mutable, human-readable pointer from a name to an image manifest, moved by the publisher.
- **digest**: the SHA-256 of an image's manifest (or index), naming exactly one image.
- **HEALTHCHECK**: a Dockerfile instruction that makes Docker run a command in the container periodically and mark it `healthy` or `unhealthy`.
- **start period**: the initial window during which health-check failures do not count.
- **OCI labels**: `org.opencontainers.image.*` keys in the image config that record title, source, version, and revision.
- **stage-scoped `ARG`**: a build argument visible only in the stage where it is declared.
- **static binary**: an executable that needs no C library at run time, produced by Go with `CGO_ENABLED=0`.

## Summary

- A tag moves, a digest does not: pin every base by digest and keep the tag for readers.
- A `HEALTHCHECK` gives Docker its own view of whether the server works, not just whether the process runs; write it in exec form with a tool the base image actually has.
- OCI labels (`title`, `source`, `version`, `revision`) let anyone trace an image to the commit that built it; `version` and `revision` come from build args, re-declared in the final stage, so CI fills them.
- Every `ENV`, `ARG`, and layer is in the image for anyone who pulls it: secrets come from a Kubernetes Secret at run time, never from the build.
- The runtime stage ships one binary on a small base; a size budget of about twice the minimal image catches a compiler or build cache shipped by mistake.

## Review questions

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
