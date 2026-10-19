# Build from the repository root. The agent worker (go/cmd/agent) runs the
# AgentRun workflow on the durable worker runtime and consumes only the queue
# configured by its Helm release.
FROM golang:1.25-alpine@sha256:1ae0735f00daffa3aaf1363a5184c0d2dc55c78e3db4ec70241cdac97bf84b59 AS build
WORKDIR /src
COPY go/go.mod go/go.sum* go/
RUN cd go && go mod download
COPY go/ go/
RUN cd go && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/agent ./cmd/agent

FROM alpine:3.22@sha256:4bcff63911fcb4448bd4fdacec207030997caf25e9bea4045fa6c8c44de311d1
RUN addgroup -S -g 10001 agent && adduser -S -D -H -u 10001 -G agent agent
WORKDIR /app
COPY --from=build --chown=10001:10001 /out/agent /usr/local/bin/agent
USER 10001:10001
EXPOSE 9464
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["/bin/sh", "-c", "wget -q -O /dev/null http://127.0.0.1:9464/healthz"]
ENTRYPOINT ["/usr/local/bin/agent"]
