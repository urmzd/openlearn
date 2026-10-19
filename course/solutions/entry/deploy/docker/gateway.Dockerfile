# The gateway image (dep.00, hardened by dep.01). Build from the repo root:
#   docker build -f deploy/docker/gateway.Dockerfile -t forge-gateway:0.1.0 \
#     --build-arg VERSION=0.1.0 --build-arg REVISION="$(git rev-parse HEAD)" .
# Base images pinned by digest (see engine.Dockerfile).
FROM golang:1.25-alpine@sha256:1ae0735f00daffa3aaf1363a5184c0d2dc55c78e3db4ec70241cdac97bf84b59 AS build
WORKDIR /src
# The module files first: `go mod download` stays cached until they change.
# The --config form imports the vendored contracts module (generated gRPC
# code), which builds in workspace mode, so the whole small contracts/ tree
# comes along and a go.work names both modules (as durable.Dockerfile does).
COPY contracts/ contracts/
COPY go/go.mod go/go.sum* go/
RUN go work init ./go ./contracts/go && cd go && go mod download
COPY go/ go/
# CGO_ENABLED=0: a static binary. The usage ledger (gw.07) uses the pure-Go
# SQLite driver, so nothing needs a C toolchain or libc at run time.
RUN cd go && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM alpine:3.19@sha256:6baf43584bcb78f2e5847d1de515f23499913ac9f12bdf834811a3145eb11ca1
ARG VERSION=0.0.0-dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="forge-gateway" \
      org.opencontainers.image.description="The forge gateway: keys, limits, cache, routing, SSE proxy, usage ledger" \
      org.opencontainers.image.source="https://example.com/forge.git" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
COPY --from=build /out/gateway /usr/local/bin/gateway
USER 10001:10001
# 8080: the API; 9464: /healthz and /readyz.
EXPOSE 8080 9464
# busybox wget is part of alpine; the health port answers /healthz.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:9464/healthz"]
ENTRYPOINT ["/usr/local/bin/gateway"]
