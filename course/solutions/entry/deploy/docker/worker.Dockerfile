# The worker image (dep.06): the Go worker plus the Python environment its
# subprocess activities exec (dur.09): {tinyllm} train/eval/export and
# {corpus} run --stage. Build from the repo root:
#   docker build -f deploy/docker/worker.Dockerfile -t forge-worker:0.1.0 \
#     --build-arg VERSION=0.1.0 --build-arg REVISION="$(git rev-parse HEAD)" .
FROM golang:1.25-alpine@sha256:1ae0735f00daffa3aaf1363a5184c0d2dc55c78e3db4ec70241cdac97bf84b59 AS build
WORKDIR /src
COPY contracts/ contracts/
COPY go/go.mod go/go.sum* go/
# go/go.mod replaces the contracts module but does not require it (the
# overlay provides it): a workspace over both is how it builds.
RUN go work init ./go ./contracts/go && cd go && go mod download
COPY go/ go/
RUN cd go && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/worker ./cmd/worker

# The runtime is Python (glibc, for numpy and pyarrow wheels); the Go worker
# is a static binary copied in.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ARG VERSION=0.0.0-dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="forge-worker" \
      org.opencontainers.image.description="The forge worker: durable activities and workflows, Python training and corpus entries as subprocesses" \
      org.opencontainers.image.source="https://example.com/forge.git" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
# Python dependencies first (they change least), pinned to the versions
# allowed-deps.toml permits; no pip cache in the layer.
RUN pip install --no-cache-dir "numpy==2.3.3" "pyarrow==21.0.0" "zstandard==0.25.0"
WORKDIR /app
COPY python/ /app/python/
COPY --from=build /out/worker /usr/local/bin/worker
USER 10001:10001
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# 9464: /healthz, /readyz, /metrics.
EXPOSE 9464
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9464/healthz', timeout=2).status == 200 else 1)"]
ENTRYPOINT ["/usr/local/bin/worker"]
