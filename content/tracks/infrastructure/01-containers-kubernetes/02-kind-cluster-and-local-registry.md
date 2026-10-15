<!-- ol:module dep.02 -->
# kind cluster + local registry + port mappings

## Overview

| | |
|---|---|
| **Module** | `dep.02` · practice · ops · Pass 7 · 2 to 3 h |
| **You build** | `deploy/kind/cluster.yaml` recreated for Pass 7 (six NodePort mappings, the registry hosts directory), `deploy/kind/up.sh` (registry container, cluster, node wiring), `deploy/kind/local-registry-hosting.yaml` |
| **Contract** | none of its own: the ports of DESIGN 2.13 and [`helm/observability.md`](../../../../course/contracts/helm/observability.md); section 4 is the checked layout |
| **Tests** | `course/tests/dep.02/` (`check` runs `artifacts.py`; what each test checks: section 4) |
| **Needs** | `dep.01` the images you push; reading: `dep.00` the tracer cluster, [Containers, Kubernetes & Workloads](README.md) |
| **Used by** | `dep.03` installs the charts on this cluster from this registry |
| **Milestone** | MS-prod |
| **Optional depth** | [kind: local registry](https://kind.sigs.k8s.io/docs/user/local-registry/) (free); [kind: configuration, extra port mappings](https://kind.sigs.k8s.io/docs/user/configuration/#extra-port-mappings) (free); [containerd registry host configuration](https://github.com/containerd/containerd/blob/main/docs/hosts.md) (free); [KEP-1755, communicating a local registry](https://github.com/kubernetes/enhancements/tree/master/keps/sig-cluster-lifecycle/generic/1755-communicating-a-local-registry) (free) |

## Key Takeaways

- A kind "node" is a Docker container; a NodePort reaches your machine only through an **extraPortMapping**, and mappings are **fixed when the node is created**: adding Prometheus, Grafana, Tempo, and the durable server means recreating the cluster once.
- Map every port on **127.0.0.1**: the default listens on all interfaces and exposes your gateway to the network you are on.
- A **local registry** (`registry:2` at `localhost:5001`) replaces `kind load`: you `docker push` once and every node pulls, the way a real cluster works.
- Inside a node, `localhost` is the node: containerd finds the registry through **`/etc/containerd/certs.d/localhost:5001/hosts.toml`**, which points at the `kind-registry` container on the `kind` network.
- One script, `up.sh`, makes the cluster identical every time it is recreated.

## How to work this chapter

```bash
ol start dep.02                        # records the start; there are no stubs
ol tests dep.02                        # read the test catalog first
kind delete cluster --name forge       # the Pass 1 cluster's port mappings cannot change
deploy/kind/up.sh                      # registry, cluster, node wiring, ConfigMap
docker tag forge-gateway:0.2.0 localhost:5001/forge-gateway:0.2.0
docker push localhost:5001/forge-gateway:0.2.0
ol check dep.02                        # exit code is the verdict
OL_SMOKE=1 ol check dep.02             # without a cluster: the static tier only
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

Without step 3's file the pull fails with `ErrImagePull: dial tcp 127.0.0.1:5001: connect: connection refused` (the node's own localhost). Without the mapping of step 5, `curl` gets `connection refused` even though `kubectl get svc` shows the NodePort. `test_image_pushed_to_the_registry_runs` and `test_cluster_maps_every_platform_port` check exactly these two links.

## 4. The artifact and its check

| Path | Holds |
|---|---|
| `deploy/kind/cluster.yaml` | `name: <system>`; `containerdConfigPatches` with `config_path = "/etc/containerd/certs.d"`; on the control-plane node, the six `extraPortMappings` of 2.1 with `listenAddress: "127.0.0.1"`; `extraMounts` of `./artifacts` at `/artifacts` |
| `deploy/kind/up.sh` | executable, from the repo root: start `kind-registry` (a pinned `registry` release, `127.0.0.1:5001:5000`) if it is not running; refuse if the cluster exists; `kind create cluster --config deploy/kind/cluster.yaml`; write `hosts.toml` on every node; `docker network connect kind kind-registry`; apply the ConfigMap |
| `deploy/kind/local-registry-hosting.yaml` | ConfigMap `kube-public/local-registry-hosting`, `data.localRegistryHosting.v1` with `host: "localhost:5001"` |
| `system.toml` `[deploy]` | unchanged from `dep.00`: `kube_context = "kind-<system>"`, `namespace = "<system>"` |

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_layout_present` | unit | the three files | |
| `test_cluster_maps_every_platform_port` | unit | name, six mappings on 127.0.0.1, `/artifacts` mount | MS-prod, obs, and drills reach the cluster on these ports |
| `test_containerd_reads_registry_hosts` | unit | the `config_path` patch | pulls from `localhost:5001` |
| `test_registry_hosting_configmap` | unit | name, namespace, `host: "localhost:5001"` | `dep.04` Tilt finds the registry |
| `test_up_script` | unit | executable, `bash -n`, each step of the table above, a pinned registry image | the same cluster every time |
| `test_system_toml_deploy_section` | unit | `kube_context`, `namespace` | the drill safety gate |
| `test_nodes_ready` | conformance | every node Ready | |
| `test_node_publishes_the_ports` | conformance | the node container publishes all six ports | the running cluster is the new one |
| `test_registry_runs_on_the_kind_network` | conformance | `kind-registry` running, 5001 on 127.0.0.1, on network `kind` | |
| `test_registry_hosting_configmap_applied` | conformance | the ConfigMap is in the cluster | |
| `test_image_pushed_to_the_registry_runs` | conformance | push an image to `localhost:5001`, a pod with it runs | images are pullable without `kind load` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. editing `cluster.yaml` without recreating the cluster | `curl 127.0.0.1:30090` is refused although Prometheus runs | `test_node_publishes_the_ports` |
| 2. no `listenAddress` | the gateway and Grafana answer to every machine on your Wi-Fi | `test_cluster_maps_every_platform_port` |
| 3. `image: localhost:5001/...` without `hosts.toml` on the nodes | `ErrImagePull ... 127.0.0.1:5001: connection refused` | `test_image_pushed_to_the_registry_runs` |
| 4. the registry not on the `kind` network | `ErrImagePull ... lookup kind-registry: no such host` | `test_registry_runs_on_the_kind_network` |
| 5. the old `registry.mirrors` table | ignored by containerd 2 on current kind node images; pulls fail | `test_containerd_reads_registry_hosts` |
| 6. `registry:latest` | the registry changes under you on the next recreate | `test_up_script` |

## 6. Where it's used next

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
