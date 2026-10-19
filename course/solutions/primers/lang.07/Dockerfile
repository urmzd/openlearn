# Stage 1: compile with the full Go toolchain (about 250 MB, never shipped).
FROM golang:1.24-alpine AS build
WORKDIR /src
# go.mod first: this layer stays cached until the module file changes.
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/hello .

# Stage 2: the runtime is a small base plus one static binary.
FROM alpine:3.19
COPY --from=build /out/hello /usr/local/bin/hello
# A numeric uid, so Kubernetes `runAsNonRoot` can prove it is not root.
USER 10001:10001
EXPOSE 8080
# Exec form: hello is PID 1 and receives SIGTERM directly.
ENTRYPOINT ["/usr/local/bin/hello"]
