# The engine image (dep.00, hardened by dep.01). Build from the repo root:
#   docker build -f deploy/docker/engine.Dockerfile -t forge-engine:0.1.0 \
#     --build-arg VERSION=0.1.0 --build-arg REVISION="$(git rev-parse HEAD)" .
#
# Stage 1 builds tl-serve and its Candle-backed Rust engine. Stage 2 ships
# the one binary on a small glibc base.
# Base images are pinned by digest (the tag is kept for humans): a tag can be
# re-pushed, a digest cannot. `docker buildx imagetools inspect <tag>` prints it.
FROM rust:1.88-slim-bookworm@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89 AS build
WORKDIR /src
# Least to most often changed: contracts, then Rust.
COPY contracts/ contracts/
COPY rust/ rust/
RUN cargo build --release --manifest-path rust/Cargo.toml -p tl-serve \
 && cp rust/target/release/tl-serve /tl-serve

FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587
ARG VERSION=0.0.0-dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="forge-engine" \
      org.opencontainers.image.description="tl-serve: the tinyllm inference engine (OpenAI-compatible HTTP and SSE)" \
      org.opencontainers.image.source="https://example.com/forge.git" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"
COPY --from=build /tl-serve /usr/local/bin/tl-serve
# Numeric, so Kubernetes runAsNonRoot can verify it.
USER 10001:10001
# 8000: POST /v1/completions; 9464: /healthz (DESIGN 2.13).
EXPOSE 8000 9464
# Docker's own health signal (docker ps, compose); Kubernetes uses the chart's
# probes instead. The slim base has no curl, so bash asks /healthz itself.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/9464 && printf 'GET /healthz HTTP/1.0\\r\\n\\r\\n' >&3 && head -n 1 <&3 | grep -q ' 200 '"]
# Exec form: tl-serve is PID 1, gets SIGTERM, and receives the chart's args.
ENTRYPOINT ["/usr/local/bin/tl-serve"]
