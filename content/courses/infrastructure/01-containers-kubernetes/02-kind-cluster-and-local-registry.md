# kind cluster + local registry + port mappings

## Overview

| | |
|---|---|
| **Module** | `dep.02` · practice · ops · Pass 7 · 2 to 3 h |
| **Solution** | [`deploy/kind/cluster.yaml`](../../../../course/solutions/entry/deploy/kind/cluster.yaml) recreated for Pass 7 (six NodePort mappings, the registry hosts directory), [`deploy/kind/up.sh`](../../../../course/solutions/entry/deploy/kind/up.sh) (registry container, cluster, node wiring), [`deploy/kind/local-registry-hosting.yaml`](../../../../course/solutions/entry/deploy/kind/local-registry-hosting.yaml) |
| **Contract** | none of its own: the fixed NodePorts of section 2.1 and [`helm/observability.md`](../../../../course/contracts/helm/observability.md); section 4 is the required layout |
| **Needs** | `dep.01` the images to push; reading: `dep.00` the tracer cluster, [Containers, Kubernetes & Workloads](README.md) |
| **Used by** | `dep.03` installs the charts on this cluster from this registry. Part of the Pass 7 production platform |
| **Optional depth** | [kind: local registry](https://kind.sigs.k8s.io/docs/user/local-registry/) (free); [kind: configuration, extra port mappings](https://kind.sigs.k8s.io/docs/user/configuration/#extra-port-mappings) (free); [containerd registry host configuration](https://github.com/containerd/containerd/blob/main/docs/hosts.md) (free); [KEP-1755, communicating a local registry](https://github.com/kubernetes/enhancements/tree/master/keps/sig-cluster-lifecycle/generic/1755-communicating-a-local-registry) (free) |

## Key Takeaways

- A kind "node" is a Docker container; a NodePort reaches your machine only through an **extraPortMapping**, and mappings are **fixed when the node is created**: adding Prometheus, Grafana, Tempo, and the durable server means recreating the cluster once.
- Map every port on **127.0.0.1**: the default listens on all interfaces and exposes your gateway to the network you are on.
- A **local registry** (`registry:2` at `localhost:5001`) replaces `kind load`: you `docker push` once and every node pulls, the way a real cluster works.
- Inside a node, `localhost` is the node: containerd finds the registry through **`/etc/containerd/certs.d/localhost:5001/hosts.toml`**, which points at the `kind-registry` container on the `kind` network.
- One script, `up.sh`, makes the cluster identical every time it is recreated.

## How to read this chapter

Sections 1 to 3 explain why the cluster is recreated, how a kind node reaches a registry on your machine, and follow one push and one request by hand. Section 4 fixes the three files. Section 5 walks through the solution's cluster file and `up.sh` and explains why each step is there. Try to predict each step of the script before reading it.

The solution's sequence, from the repo root:

```bash
kind delete cluster --name forge       # the Pass 1 cluster's port mappings cannot change
deploy/kind/up.sh                      # registry, cluster, node wiring, ConfigMap
docker tag forge-gateway:0.2.0 localhost:5001/forge-gateway:0.2.0
docker push localhost:5001/forge-gateway:0.2.0
```

---

## 1. Why now

The Pass 1 cluster maps two ports: the gateway (30080) and Jaeger (30686). Pass 7 adds Prometheus (30090), Grafana (30300), and Tempo's query API (30320), and Pass 8 adds the durable server (30733), so a worker on your machine can reach it. kind cannot add a mapping to a running node, so the cluster has to be recreated, and since it will be recreated again (a broken node, a new Kubernetes version, a teammate's machine), the recipe belongs in a script. Pass 1 also moved images with `kind load docker-image`, which copies a whole image into every node on every change and does not exist on a real cluster. From now on images go through a registry, as they will in CI (`dep.05`) and on every production cluster.

## 2. Principles

### 2.1 Port mappings

A NodePort Service listens on a port of every node. In kind the node is a container, so the port is open inside the container's network namespace, not on your machine. `extraPortMappings` asks Docker to publish `containerPort` of the node container on `hostPort` of your machine, exactly like `docker run -p 127.0.0.1:30080:30080`. Docker sets publishing when it creates a container and never changes it afterwards, so the mappings are fixed at `kind create cluster`. `listenAddress` is the host interface: the default `0.0.0.0` accepts connections from any machine that can reach yours, `127.0.0.1` only from your own.

| Port | Service | From |
|---|---|---|
| 30080 | gateway | Pass 1 |
| 30686 | Jaeger query | Pass 1 |
| 30090 | Prometheus | Pass 7 |
| 30300 | Grafana | Pass 7 |
| 30320 | Tempo query API | Pass 7 |
| 30733 | durable gRPC (a worker on the host) | Pass 8 |

### 2.2 A registry the nodes can reach

A **registry** stores images by name and digest and serves them over HTTP (the OCI distribution API); `registry:2` is the reference implementation in one container. The trick is names. You push to `localhost:5001/forge-gateway:0.2.0`, so that is the name in your chart. But when the kubelet on the node asks containerd to pull `localhost:5001/...`, localhost is the node container, where nothing listens on 5001. Three pieces fix that:

1. The registry container joins the Docker network `kind` that the nodes are on, so it is reachable from them by its name `kind-registry`, on its container port 5000.
2. `containerdConfigPatches` sets `config_path = "/etc/containerd/certs.d"`: containerd then reads one directory per registry host.
3. `up.sh` writes `/etc/containerd/certs.d/localhost:5001/hosts.toml` on every node with `[host."http://kind-registry:5000"]`: "for `localhost:5001`, talk to this host instead".

The ConfigMap `kube-public/local-registry-hosting` (KEP-1755) advertises the registry to tools: Tilt (`dep.04`) reads it to push images to the right place.

### 2.3 Recreate, never patch

The cluster holds nothing you cannot recreate: the model files live in `./artifacts` on your machine (mounted at `/artifacts`), the images in the registry (which `up.sh` keeps running across clusters), and everything else is a Helm release you reinstall (`dep.03`). That is what makes `kind delete cluster` safe, and why the script refuses to touch an existing cluster rather than pretend it could change its ports.

## 3. Worked example by hand

Push the gateway image and follow the bytes:

1. `docker push localhost:5001/forge-gateway:0.2.0` on your machine: Docker connects to 127.0.0.1:5001, published by the registry container (`-p 127.0.0.1:5001:5000`), and uploads the layers. Port 5001, not 5000, because macOS uses 5000 for AirPlay.
2. The chart's pod spec says `image: localhost:5001/forge-gateway:0.2.0`. The kubelet on node `forge-control-plane` asks containerd to pull it.
3. containerd reads `/etc/containerd/certs.d/localhost:5001/hosts.toml`, finds `http://kind-registry:5000`, and resolves `kind-registry` through Docker's DNS on the `kind` network.
4. The layers come from the registry container's port 5000, plain HTTP: allowed because `hosts.toml` names an `http://` host.
5. The pod starts. Then `curl http://127.0.0.1:30080/healthz` from your machine reaches the node container's port 30080 through the mapping `127.0.0.1:30080 -> 30080`, then the NodePort Service, then the pod.

Without step 3's file the pull fails with `ErrImagePull: dial tcp 127.0.0.1:5001: connect: connection refused` (the node's own localhost). Without the mapping of step 5, `curl` gets `connection refused` even though `kubectl get svc` shows the NodePort. These are the two links a correct setup must get right.

## 4. The artifact and what it must do

| Path | Holds |
|---|---|
| `deploy/kind/cluster.yaml` | `name: <system>`; `containerdConfigPatches` with `config_path = "/etc/containerd/certs.d"`; on the control-plane node, the six `extraPortMappings` of 2.1 with `listenAddress: "127.0.0.1"`; `extraMounts` of `./artifacts` at `/artifacts` |
| `deploy/kind/up.sh` | executable, from the repo root: start `kind-registry` (a pinned `registry` release, `127.0.0.1:5001:5000`) if it is not running; refuse if the cluster exists; `kind create cluster --config deploy/kind/cluster.yaml`; write `hosts.toml` on every node; `docker network connect kind kind-registry`; apply the ConfigMap |
| `deploy/kind/local-registry-hosting.yaml` | ConfigMap `kube-public/local-registry-hosting`, `data.localRegistryHosting.v1` with `host: "localhost:5001"` |

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| layout present | static | the three files | |
| every platform port mapped | static | name, six mappings on 127.0.0.1, `/artifacts` mount | the observability stack, drills, and a host worker reach the cluster on these ports |
| containerd reads registry hosts | static | the `config_path` patch | pulls from `localhost:5001` |
| registry hosting ConfigMap | static | name, namespace, `host: "localhost:5001"` | `dep.04` Tilt finds the registry |
| `up.sh` | static | executable, `bash -n` clean, each step of the table above, a pinned registry image | the same cluster every time |
| nodes ready | cluster | every node Ready | |
| node publishes the ports | cluster | the node container publishes all six ports | the running cluster is the new one |
| registry on the kind network | cluster | `kind-registry` running, 5001 on 127.0.0.1, on network `kind` | |
| ConfigMap applied | cluster | the ConfigMap is in the cluster | |
| a pushed image runs | cluster | push an image to `localhost:5001`, a pod with it runs | images are pullable without `kind load` |

## 5. The solution, step by step

### 5.1 From first principles

1. **Docker publishes a container's ports only at creation.** A kind node is a container, so every NodePort the rest of the course needs must be listed before `kind create cluster`. Pass 7 and 8 need six; so the cluster is recreated once with all six.
2. **Exposure is a choice of interface.** Publishing on `0.0.0.0` exposes the gateway and Grafana to the local network; `listenAddress: "127.0.0.1"` on every mapping keeps them on your machine.
3. **A pull happens inside the node, where `localhost` is the node.** So the name in the chart (`localhost:5001/...`, which works for `docker push` on your machine) must be translated on the node. containerd's per-host directory does that, which requires (a) containerd to read the directory (`config_path`), (b) a `hosts.toml` per node, and (c) the registry to be reachable from the node by name (the `kind` network).
4. **Recreation must give the same cluster.** Any order-dependent manual step will be forgotten on the next recreate, so the steps live in one script that refuses to run against an existing cluster (it could not change the ports anyway) and pins the registry image.
5. **Tools should not guess the registry.** KEP-1755's ConfigMap is the standard place to advertise it; Tilt reads it.

### 5.2 The cluster file

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: forge
# containerd looks for per-registry settings in /etc/containerd/certs.d;
# up.sh writes certs.d/localhost:5001/hosts.toml so that pulls of
# localhost:5001/<image> reach the kind-registry container.
containerdConfigPatches:
  - |-
    [plugins."io.containerd.grpc.v1.cri".registry]
      config_path = "/etc/containerd/certs.d"
nodes:
  - role: control-plane
    # Port mappings are fixed when the node container is created: adding one
    # means deleting and recreating the cluster. 127.0.0.1 only: nothing is
    # exposed to your LAN.
    extraPortMappings:
      - containerPort: 30080   # gateway NodePort
        hostPort: 30080
        listenAddress: "127.0.0.1"
      - containerPort: 30686   # Jaeger query NodePort (Pass 1)
        hostPort: 30686
        listenAddress: "127.0.0.1"
      - containerPort: 30090   # Prometheus
        hostPort: 30090
        listenAddress: "127.0.0.1"
      - containerPort: 30300   # Grafana
        hostPort: 30300
        listenAddress: "127.0.0.1"
      - containerPort: 30320   # Tempo query API
        hostPort: 30320
        listenAddress: "127.0.0.1"
      - containerPort: 30733   # durable gRPC, for a worker on the host (Pass 8)
        hostPort: 30733
        listenAddress: "127.0.0.1"
    # ./artifacts on your machine is /artifacts on the node.
    extraMounts:
      - hostPath: ./artifacts
        containerPath: /artifacts
        readOnly: true
```

`containerPort` equals `hostPort` everywhere so a port number means the same thing on both sides; a mismatch is a classic source of "it works in the cluster but not from my machine". `config_path` replaces the older inline `registry.mirrors` table, which current containerd ignores (pitfall 5).

Full solution: [`cluster.yaml`](../../../../course/solutions/entry/deploy/kind/cluster.yaml).

### 5.3 `up.sh`

```bash
set -euo pipefail                     # any failed step stops the script
cd "$(dirname "$0")/../.."            # the repo root, wherever it is run from:
                                      # extraMounts' ./artifacts is relative to it
name="$(awk '/^name:/ {print $2; exit}' deploy/kind/cluster.yaml)"   # one source of truth
reg_name=kind-registry
reg_port=5001                         # not 5000: macOS AirPlay listens there
reg_image="registry:2.8.3@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373"

# Refuse, do not patch: an existing node's port mappings cannot change.
if kind get clusters 2>/dev/null | grep -qx "$name"; then
  echo "cluster $name exists; its port mappings cannot change. Delete it first: kind delete cluster --name $name" >&2
  exit 1
fi

# 1. The registry container, reachable from the host at 127.0.0.1:5001.
#    Kept across clusters (--restart=always), so pushed images survive a recreate.
if [ "$(docker inspect -f '{{.State.Running}}' "$reg_name" 2>/dev/null || true)" != "true" ]; then
  docker run -d --restart=always -p "127.0.0.1:${reg_port}:5000" --network bridge --name "$reg_name" "$reg_image"
fi

# 2. The cluster. artifacts/ must exist, or the hostPath mount has nothing to mount.
mkdir -p artifacts
kind create cluster --config deploy/kind/cluster.yaml

# 3. Every node resolves localhost:5001 to the registry container.
for node in $(kind get nodes --name "$name"); do
  docker exec "$node" mkdir -p "/etc/containerd/certs.d/localhost:${reg_port}"
  printf '[host."http://%s:5000"]\n' "$reg_name" |
    docker exec -i "$node" cp /dev/stdin "/etc/containerd/certs.d/localhost:${reg_port}/hosts.toml"
done

# 4. The registry joins the cluster's network, so nodes reach it by name.
#    Idempotent: connect only if it is not already on "kind".
if [ "$(docker inspect -f '{{json .NetworkSettings.Networks.kind}}' "$reg_name")" = "null" ]; then
  docker network connect kind "$reg_name"
fi

# 5. Advertise it (KEP-1755), then wait until the cluster can schedule.
kubectl --context "kind-$name" apply -f deploy/kind/local-registry-hosting.yaml
kubectl --context "kind-$name" wait --for=condition=Ready nodes --all --timeout=120s
echo "kind-$name is up; push images to localhost:${reg_port}/"
```

Every `kubectl` names `--context "kind-$name"` explicitly, so the script cannot apply the ConfigMap to whatever cluster your shell happens to point at. The `hosts.toml` is written per node after creation because the node filesystem does not exist before; containerd reads `config_path` directories at pull time, so no restart is needed. Steps 1 and 4 check before acting, so a rerun after a failed creation is safe.

Full solution: [`up.sh`](../../../../course/solutions/entry/deploy/kind/up.sh), [`local-registry-hosting.yaml`](../../../../course/solutions/entry/deploy/kind/local-registry-hosting.yaml).

### 5.4 Why this solution

- **A registry over `kind load`.** `kind load` copies whole images into every node on every change and does not exist outside kind; a registry transfers only new layers and is how every real cluster works.
- **`hosts.toml` over a registry name like `kind-registry:5000` in the charts.** Charts would then hold a name that works only inside the cluster, and `docker push` from your machine would need a different name. With the translation, one name works on both sides.
- **A script over a README of commands.** The order matters (the registry before the network connect, the cluster before `hosts.toml`), and a script also makes the refusal and the pins enforceable.
- **Tradeoff accepted:** the registry is plain HTTP without authentication, acceptable only because it listens on 127.0.0.1 and the `kind` network.

### 5.5 Explain it back

1. Why can `up.sh` not simply add the Prometheus mapping to the existing Pass 1 cluster?
   <details><summary>Answer</summary>Docker fixes a container's published ports when it creates the container, and a kind node is a container. The only way to add a mapping is a new node, which means recreating the cluster; the script refuses rather than pretend otherwise.</details>
2. A pod shows `ErrImagePull ... dial tcp 127.0.0.1:5001: connect: connection refused`. What is missing, and why that address?
   <details><summary>Answer</summary>The node's <code>hosts.toml</code> (or the <code>config_path</code> patch). Without it containerd takes <code>localhost:5001</code> literally, and localhost inside the node container is the node itself, where nothing listens on 5001.</details>
3. Why is the registry started with `--restart=always` and kept outside the cluster?
   <details><summary>Answer</summary>So that it survives <code>kind delete cluster</code> and a Docker restart: images pushed before a recreate are still there afterwards, which is part of what makes recreating the cluster cheap.</details>
4. Why does the script `cd` to the repo root before anything else?
   <details><summary>Answer</summary><code>extraMounts</code> uses the relative path <code>./artifacts</code>, which kind resolves against the current directory. Run from elsewhere, the node would mount an empty or wrong directory and the engine would find no model.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. editing `cluster.yaml` without recreating the cluster | `curl 127.0.0.1:30090` is refused although Prometheus runs | `up.sh` refuses an existing cluster; delete and rerun (5.3) |
| 2. no `listenAddress` | the gateway and Grafana answer to every machine on your Wi-Fi | `listenAddress: "127.0.0.1"` on every mapping (5.2) |
| 3. `image: localhost:5001/...` without `hosts.toml` on the nodes | `ErrImagePull ... 127.0.0.1:5001: connection refused` | step 3 of `up.sh` writes `hosts.toml` on every node (5.3) |
| 4. the registry not on the `kind` network | `ErrImagePull ... lookup kind-registry: no such host` | step 4 of `up.sh`, `docker network connect kind` (5.3) |
| 5. the old `registry.mirrors` table | ignored by containerd 2 on current kind node images; pulls fail | `config_path` and per-host directories (5.2) |
| 6. `registry:latest` | the registry changes under you on the next recreate | `registry:2.8.3@sha256:...` pinned in `up.sh` (5.3) |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.01` | the images pushed to `localhost:5001` |
| Back | `dep.00` | the Pass 1 cluster this one replaces |
| Forward | `dep.03` | charts with `image.repository: localhost:5001/<system>-gateway`, the observability stack on 30090, 30300, 30320 |
| Forward | `dep.04` | Tilt pushes rebuilt images to the registry it finds in the ConfigMap |
| Forward | `dep.06` | a worker on your machine reaches the durable server on 30733 |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `registry:2` on localhost | Harbor, a cloud registry (ECR, GAR, ACR) | authentication, vulnerability scanning, replication, retention | [Harbor](https://goharbor.io/docs/) (free) |
| `extraPortMappings` | a cloud load balancer, Gateway API | one stable entry point with TLS and routing rules | [Gateway API](https://gateway-api.sigs.k8s.io/) (free) |
| `up.sh` | Cluster API, Terraform, ctlptl | declarative clusters with lifecycle management | [ctlptl](https://github.com/tilt-dev/ctlptl) (free), [Cluster API](https://cluster-api.sigs.k8s.io/) (free) |
