<!-- ol:module dep.01 -->
# Dockerfiles for the gateway and the engine

## Overview

| | |
|---|---|
| **Module** | `dep.01` · practice · ops · Pass 7 · 3 to 5 h |
| **You build** | `deploy/docker/engine.Dockerfile` and `deploy/docker/gateway.Dockerfile`, hardened: bases pinned by digest, a `HEALTHCHECK`, OCI labels from build args, a static gateway binary; `.dockerignore` that keeps secrets out of the context |
| **Contract** | none of its own: the images run the roles of [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) on the ports of DESIGN 2.13; section 4 is the checked layout |
| **Tests** | `course/tests/dep.01/` (`check` runs `artifacts.py`; what each test checks: section 4), with the size budget in `course/fixtures/dep.01/image-budget.json` |
| **Needs** | `dep.00` the tracer images, `gw.07` the gateway with its pure-Go SQLite ledger; reading: [Containers, Kubernetes & Workloads](README.md), `L10.5` the engine's HTTP server |
| **Used by** | `dep.02` loads these images into the kind registry · `dep.03` runs them from the Helm charts |
| **Milestone** | MS-prod |
| **Optional depth** | [Dockerfile reference: HEALTHCHECK](https://docs.docker.com/reference/dockerfile/#healthcheck) (free); [OCI image annotations](https://github.com/opencontainers/image-spec/blob/main/annotations.md) (free); [Docker build best practices](https://docs.docker.com/build/building/best-practices/) (free) |

## Key Takeaways

- A **tag moves, a digest does not**: `golang:1.25-alpine` is re-pushed with every patch release, `golang:1.25-alpine@sha256:...` names exactly one image. Pin every base by digest and keep the tag for readers.
- A **HEALTHCHECK** gives Docker its own view of whether the server works, not just whether the process runs; write it in exec form with a tool the base image actually has.
- **OCI labels** (`org.opencontainers.image.title`, `source`, `version`, `revision`) let anyone trace an image back to the commit that built it; `version` and `revision` come from build args so CI fills them.
- Every `ENV` and `ARG` value is **in the image history** for anyone who pulls it: secrets come from a Kubernetes Secret at run time, never from the build.
- The runtime stage ships **one binary on a small base**; a size budget of twice the reference image catches a compiler or a build cache shipped by mistake.

## How to work this chapter

Every command runs from your repo root. `forge` stands for your `<system>` name.

```bash
ol start dep.01                               # records the start; there are no stubs
ol tests dep.01                               # read the test catalog first
docker buildx imagetools inspect golang:1.25-alpine --format '{{json .Manifest.Digest}}'   # the digest to pin
docker build -f deploy/docker/gateway.Dockerfile -t forge-gateway:0.2.0 \
  --build-arg VERSION=0.2.0 --build-arg REVISION="$(git rev-parse HEAD)" .
docker build -f deploy/docker/engine.Dockerfile  -t forge-engine:0.2.0 \
  --build-arg VERSION=0.2.0 --build-arg REVISION="$(git rev-parse HEAD)" .
docker image inspect forge-gateway:0.2.0 --format '{{json .Config.Labels}} {{json .Config.Healthcheck}}'
docker history --no-trunc forge-gateway:0.2.0  # what anyone who pulls it can read
ol check dep.01                               # exit code is the verdict
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
| `org.opencontainers.image.source` | your repository URL |
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
5. Size: alpine's filesystem is about 7.8 MiB and the static gateway binary about 14 MiB, so `docker export` of a container is about 22 MiB. The reference budget is twice that, 43.8 MiB. A single-stage build on `golang:1.25-alpine` would export more than 300 MiB and fail.

`test_healthcheck_in_exec_form`, `test_oci_labels_declared`, `test_images_carry_the_labels`, and `test_images_fit_the_size_budget` check exactly these lines and numbers.

## 4. The artifact and its check

| Path | Holds |
|---|---|
| `deploy/docker/engine.Dockerfile` | build stage pinned by digest (Rust, make), runtime stage `debian:bookworm-slim@sha256:...`; `ARG VERSION`, `ARG REVISION`, the four labels, numeric `USER`, `EXPOSE 8000 9464`, a `HEALTHCHECK` on `/healthz` port 9464 (bash `/dev/tcp`), exec-form `ENTRYPOINT` |
| `deploy/docker/gateway.Dockerfile` | build stage pinned by digest with `CGO_ENABLED=0 go build`, runtime stage `alpine@sha256:...` with the same labels, `USER`, `EXPOSE 8080 9464`, a `HEALTHCHECK` with `wget` |
| `.dockerignore` | build outputs and local state (dep.00) plus `.env`, `.git`, and `*.pem` |

The check runs the static tier first, then builds both images with `--build-arg VERSION=0.0.0-check --build-arg REVISION=<HEAD>` and runs them the way the charts do: the engine on a model trained by your `{tinyllm} train bigram` (as in `dep.00`), the gateway with a probe key, both with `--health-interval 1s`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_layout_present` | unit | the two Dockerfiles and `.dockerignore` | `dep.03` and CI build these files |
| `test_image_rules_still_hold` | unit | dep.00's rules: stages, numeric user, exec form, ports | the charts' probes and `runAsNonRoot` |
| `test_base_images_pinned_by_digest` | unit | every external `FROM` has `@sha256:<64 hex>` | reproducible builds; `craft.18` SBOMs |
| `test_healthcheck_in_exec_form` | unit | one `HEALTHCHECK`, exec form, `--interval`, `--timeout`, `/healthz` on 9464 | Docker sees a hung server |
| `test_oci_labels_declared` | unit | title, source, version, revision; `ARG VERSION` and `ARG REVISION` in the final stage | traceable images |
| `test_gateway_builds_a_static_binary` | unit | `CGO_ENABLED=0 go build` | the SQLite ledger runs on alpine |
| `test_no_secrets_in_dockerfiles_or_context` | unit | no secret-looking `ENV`/`ARG` values or key material; `.env`, `.git`, `*.pem` ignored | nothing secret in a layer |
| `test_images_build` | unit | both images build with the build args | |
| `test_images_carry_the_labels` | unit | labels present; version and revision equal the build args | |
| `test_images_run_as_numeric_non_root` | boundary | the image config's user is a numeric uid other than 0 | `runAsNonRoot` |
| `test_no_secrets_in_image_history_or_env` | unit | `docker history --no-trunc` and the config's `Env` hold no secret | a pulled image leaks nothing |
| `test_images_fit_the_size_budget` | boundary | `docker export` bytes at most twice the reference | no toolchain in the runtime stage |
| `test_containers_report_healthy_and_stream` | conformance | both containers turn `healthy`, then a completion streams engine to gateway | the health command works in its base; packaging broke nothing |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `FROM golang:1.25-alpine` without a digest | a rebuild of an old commit produces a different image | `test_base_images_pinned_by_digest` |
| 2. `HEALTHCHECK CMD curl ...` on a base without curl | the container is `unhealthy` forever while serving fine | `test_containers_report_healthy_and_stream` |
| 3. one stage, or `COPY --from=build /src` instead of the binary | a 300 MiB to 2 GiB image with the compiler and build cache | `test_images_fit_the_size_budget` |
| 4. `ARG API_KEY` or `ENV TL_GATEWAY_PEPPER=...` | the secret is in `docker history` for everyone who pulls | `test_no_secrets_in_dockerfiles_or_context`, `test_no_secrets_in_image_history_or_env` |
| 5. `ARG VERSION` only before the first `FROM` | the version label is empty | `test_oci_labels_declared`, `test_images_carry_the_labels` |
| 6. a cgo SQLite driver, or `go build` without `CGO_ENABLED=0` | the gateway exits at start on alpine: the binary wants the build stage's libc | `test_gateway_builds_a_static_binary`, `test_containers_report_healthy_and_stream` |
| 7. `HEALTHCHECK CMD wget ...` in shell form | needs `/bin/sh`; on a distroless base the check cannot even start | `test_healthcheck_in_exec_form` |

## 6. Where it's used next

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
