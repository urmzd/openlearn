#!/usr/bin/env bash
# Create the kind cluster with a local registry (dep.02). From the repo root:
#   deploy/kind/up.sh            # cluster forge, registry localhost:5001
#
# Port mappings are fixed at creation, so an existing cluster is never
# patched: delete it first (`kind delete cluster --name forge`), then rerun.
# Images then flow: docker build -> docker push localhost:5001/<image>:<tag>
# -> the chart's image.repository localhost:5001/<image>.
set -euo pipefail
cd "$(dirname "$0")/../.."
name="$(awk '/^name:/ {print $2; exit}' deploy/kind/cluster.yaml)"
reg_name=kind-registry
reg_port=5001
reg_image="registry:2.8.3@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373"

if kind get clusters 2>/dev/null | grep -qx "$name"; then
  echo "cluster $name exists; its port mappings cannot change. Delete it first: kind delete cluster --name $name" >&2
  exit 1
fi

# 1. The registry container, reachable from the host at 127.0.0.1:5001.
if [ "$(docker inspect -f '{{.State.Running}}' "$reg_name" 2>/dev/null || true)" != "true" ]; then
  docker run -d --restart=always -p "127.0.0.1:${reg_port}:5000" --network bridge --name "$reg_name" "$reg_image"
fi

# 2. The cluster.
mkdir -p artifacts
kind create cluster --config deploy/kind/cluster.yaml

# 3. Every node resolves localhost:5001 to the registry container.
for node in $(kind get nodes --name "$name"); do
  docker exec "$node" mkdir -p "/etc/containerd/certs.d/localhost:${reg_port}"
  printf '[host."http://%s:5000"]\n' "$reg_name" |
    docker exec -i "$node" cp /dev/stdin "/etc/containerd/certs.d/localhost:${reg_port}/hosts.toml"
done

# 4. The registry joins the cluster's network, so nodes reach it by name.
if [ "$(docker inspect -f '{{json .NetworkSettings.Networks.kind}}' "$reg_name")" = "null" ]; then
  docker network connect kind "$reg_name"
fi

# 5. Advertise it (KEP-1755).
kubectl --context "kind-$name" apply -f deploy/kind/local-registry-hosting.yaml
kubectl --context "kind-$name" wait --for=condition=Ready nodes --all --timeout=120s
echo "kind-$name is up; push images to localhost:${reg_port}/"
