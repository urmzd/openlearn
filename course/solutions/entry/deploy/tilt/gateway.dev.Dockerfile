# The gateway's dev image for Tilt (dep.04): only the binary that
# gateway-compile cross-built, on the same base as deploy/docker/gateway.Dockerfile.
# Tilt's restart_process live update needs a shell in the image, so the base
# is alpine, not distroless. Never pushed or released.
FROM alpine:3.19
COPY build/gateway /usr/local/bin/gateway
USER 10001:10001
EXPOSE 8080 9464
ENTRYPOINT ["/usr/local/bin/gateway"]
