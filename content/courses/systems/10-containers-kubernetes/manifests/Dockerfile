# Multi-stage build for the Streamflow order-worker (Rust).
# Goal: a tiny, non-root, distroless runtime image with good layer caching.
#
# Layer-caching rule: copy dependency manifests and build deps BEFORE copying
# source, so a code edit doesn't invalidate the (slow) dependency layer.
#
# Build:  docker build -t streamflow/order-worker:$(git rev-parse --short HEAD) .
# Inspect layers: docker history streamflow/order-worker:<tag>

# ---- Stage 1: builder (fat; has the toolchain) ----
FROM rust:1.83-slim AS builder
WORKDIR /app

# 1) Cache dependencies: copy only manifests, build a stub, fetch+compile deps.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

# 2) Now copy real source. This layer is the only one a code change busts.
COPY src ./src
RUN cargo build --release --locked

# ---- Stage 2: runtime (tiny; no shell, no package manager) ----
# distroless = glibc + CA certs + your binary. Nothing to exploit, fast to pull.
FROM gcr.io/distroless/cc-debian12:nonroot
WORKDIR /app

# Copy ONLY the compiled binary out of the builder. The toolchain stays behind.
COPY --from=builder /app/target/release/order-worker /app/order-worker

# 12-factor: config from the environment, not baked into the image.
ENV RUST_LOG=info
# Run as a non-root user (distroless :nonroot is uid 65532).
USER nonroot

# No ENTRYPOINT shell wrapper: exec form so the process gets PID 1 and
# receives SIGTERM directly for graceful shutdown (see consumer code).
ENTRYPOINT ["/app/order-worker"]
