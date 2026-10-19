# The durable server image (dep.06). Build from the repo root:
#   docker build -f deploy/docker/durable.Dockerfile -t forge-durable:0.1.0 \
#     --build-arg VERSION=0.1.0 --build-arg REVISION="$(git rev-parse HEAD)" .
# The dep.01 rules: multi-stage, bases pinned by digest, numeric non-root
# user, HEALTHCHECK in exec form, OCI labels.
FROM golang:1.25-alpine@sha256:1ae0735f00daffa3aaf1363a5184c0d2dc55c78e3db4ec70241cdac97bf84b59 AS build
WORKDIR /src
COPY contracts/ contracts/
COPY go/go.mod go/go.sum* go/
# go/go.mod replaces the contracts module but does not require it (the
# overlay provides it): a workspace over both is how it builds.
RUN go work init ./go ./contracts/go && cd go && go mod download
COPY go/ go/
RUN cd go && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/durable ./cmd/durable

FROM alpine:3.19@sha256:6baf43584bcb78f2e5847d1de515f23499913ac9f12bdf834811a3145eb11ca1
ARG VERSION=0.0.0-dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="forge-durable" \
      org.opencontainers.image.description="The forge durable execution server: event log, task queues, workflow history" \
      org.opencontainers.image.source="https://example.com/forge.git" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
COPY --from=build /out/durable /usr/local/bin/durable
# The WAL lives on the StatefulSet's volume at /var/lib/durable; the
# directory exists and belongs to the runtime user even without a volume.
RUN mkdir -p /var/lib/durable && chown 10001:10001 /var/lib/durable
USER 10001:10001
# 7233: tl.durable.v1 gRPC; 7234: Raft (dur.10); 9464: /healthz, /readyz, /metrics.
EXPOSE 7233 7234 9464
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:9464/healthz"]
ENTRYPOINT ["/usr/local/bin/durable"]
