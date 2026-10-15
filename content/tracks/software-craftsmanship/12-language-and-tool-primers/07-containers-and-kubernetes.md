<!-- ol:module lang.07 -->
# Containers and Kubernetes: images, layers, Pods, Deployments, Services, Helm, kind

## Overview

| | |
|---|---|
| **Module** | `lang.07` · practice · ops · Pass 1 · 6 to 8 h |
| **You build** | `primers/lang.07/`: a tiny Go server, its `Dockerfile`, a Helm chart, and a kind cluster config, deployed and reached through a NodePort |
| **Contract** | none: the layout and rules in section 4 are the contract |
| **Tests** | `course/tests/lang.07/` (`check` runs `artifacts.py`; what each test checks: section 4) |
| **Needs** | `lang.06` Go (reading), [`infrastructure/01-containers-kubernetes`](../../infrastructure/01-containers-kubernetes/README.md) (reading) |
| **Used by** | `dep.00` tracer deploy (the same skills, applied to your engine and gateway) |
| **Milestone** | MS-P1 |
| **Optional depth** | [Kubernetes documentation](https://kubernetes.io/docs/concepts/) (free), [Docker build docs](https://docs.docker.com/build/) (free), [OCI image spec](https://github.com/opencontainers/image-spec) (free), [Helm docs](https://helm.sh/docs/) (free), [kind docs](https://kind.sigs.k8s.io/) (free) |

## Key Takeaways

- An **image** is a stack of content-addressed filesystem layers plus a config (entrypoint, user, ports, env). A **container** is an ordinary Linux process started from that config, with its own view of the filesystem, processes, and network.
- A **multi-stage build** compiles in a fat stage and copies one binary into a small runtime stage that runs as a **numeric non-root user**. The tests prove the user and the exit code.
- Your server is **PID 1** in its container. It must handle `SIGTERM` itself and exit `0`, or every `docker stop` and every rollout kills it after the grace period.
- Kubernetes is a set of **controllers** that drive actual state toward the objects you declare. A **Deployment** keeps N **Pods** running; a **Service** finds them by label and gives them one stable address; a **NodePort** opens that address on every node.
- **Helm** renders templates plus values into those objects. **kind** runs a whole cluster as Docker containers, which is why a NodePort needs a port mapping fixed when the cluster is created.

## How to work this chapter

```bash
ol start lang.07                 # records the start; there are no stubs: you write every file
ol tests lang.07                 # read the test catalog first
# write primers/lang.07/{main.go,go.mod,Dockerfile,kind.yaml,chart/}  (section 4)
docker build -t hello:0.1.0 primers/lang.07
kind create cluster --config primers/lang.07/kind.yaml
kind load docker-image hello:0.1.0 --name lang07
helm install hello primers/lang.07/chart --kube-context kind-lang07
curl -s http://127.0.0.1:31007/
ol check lang.07                 # exit code is the verdict
OL_SMOKE=1 ol check lang.07      # without a cluster: everything except the cluster tier
kind delete cluster --name lang07   # when you are done; dep.00 makes its own cluster
```

---

## 1. Why now

At this point in Pass 1 your system runs as processes on your laptop: a Rust engine you start with `cargo run`, a Go gateway you start with `go run`, both reading files from paths that exist only on your machine. MS-P1 asks for more: the same two programs must run on a Kubernetes cluster, reachable from outside it, restartable by a controller, and observable. None of that is possible until each program is an image with a known entrypoint, user, and port, and until you can describe "run one of these, keep it healthy, and give it an address" as data the cluster understands. This primer teaches exactly those pieces on a server small enough that nothing else can go wrong. In `dep.00` you apply them, unchanged, to your engine and gateway.

## 2. Principles

Every term is defined before it is used. Read this section once straight through; section 4 tells you what to build.

### 2.1 A process, isolated

A **process** is a running program: memory, open files, a current directory, environment variables, a user id (uid), and a process id (pid). On Linux, the kernel can give a process private **namespaces**, each a separate view of one kind of resource:

| Namespace | What the process sees |
|---|---|
| mount | its own root filesystem (`/`), so `/usr/bin` is whatever the image holds |
| pid | its own process table; the first process in it is **pid 1** |
| network | its own interfaces, IP address, and ports |
| uts | its own hostname |
| user | its own mapping of user ids |

**cgroups** (control groups) put limits on what the process may use: CPU time and memory. A **container** is a process started with its own namespaces and a cgroup. It shares the host's kernel. That is why a container starts in milliseconds and why an image built for Linux on `arm64` does not run on an `amd64` kernel without emulation. On macOS, Docker Desktop (or colima) runs a small Linux virtual machine, and your containers are processes in it.

### 2.2 Images and layers

An **image** is what a container starts from. It has two parts:

1. **Layers.** Each layer is a tar archive of file changes (added, changed, or deleted files). Stacked in order, the layers form the container's root filesystem. A layer is named by the SHA-256 hash of its bytes, its **digest**, so the same bytes always have the same name and are stored and downloaded once.
2. **Config.** JSON that says how to start the process: `Entrypoint` and `Cmd` (the argv), `Env`, `User`, `WorkingDir`, `ExposedPorts`, labels.

An image is named by `repository:tag`, for example `golang:1.24-alpine`. A **tag** is a movable pointer: the registry can make `golang:1.24-alpine` point to new bytes tomorrow. A digest (`golang@sha256:...`) is immutable. The tag `latest` is only a convention for "whatever was pushed last" and is the default when you write no tag at all.

### 2.3 The Dockerfile and the build cache

A **Dockerfile** is a program that builds an image, one instruction per line:

| Instruction | Effect |
|---|---|
| `FROM image AS name` | start a **stage** from a base image |
| `WORKDIR /src` | set the directory for later instructions |
| `COPY src dst` | add files from the **build context** (the directory you pass to `docker build`); makes a layer |
| `COPY --from=name src dst` | add files from an earlier stage |
| `RUN cmd` | run a command in the image being built; its file changes become a layer |
| `ENV K=V`, `USER uid:gid`, `EXPOSE 8080`, `ENTRYPOINT [...]` | set config, no layer |

The builder caches each layer by the instruction and its inputs. If the instruction and everything it depends on are unchanged, it reuses the cached layer and every later one. The first instruction whose input changed, and everything after it, rebuild. So instructions are ordered from least to most often changed: dependency manifests (`go.mod`, `Cargo.toml`) before source code.

A **multi-stage build** has two or more `FROM` lines. The early stage holds the compiler (the `golang` image is about 250 MB). The last stage is what ships, and it receives only the compiled binary through `COPY --from`. Nothing else from the build stage is in the final image.

### 2.4 Running a container

`docker run` creates a container from an image and starts its entrypoint:

```bash
docker run -d --name h -e GREETING=hi -p 127.0.0.1:8081:8080 hello:0.1.0
```

- `-e K=V` sets an environment variable. Programs that read their configuration from the environment can be configured without rebuilding the image. That is a rule of the **twelve-factor** app.
- `-p 127.0.0.1:8081:8080` **publishes** a port: connections to port 8081 on the host's loopback address are forwarded to port 8080 inside the container's network namespace. `EXPOSE` in the Dockerfile only documents the port; `-p` is what opens it.
- The container's hostname (uts namespace) defaults to the first 12 hex digits of its id.

**Signals and pid 1.** `docker stop` sends `SIGTERM` to the container's pid 1, waits a **grace period** (10 s by default, `-t` to change), then sends `SIGKILL`, which cannot be caught. Three details decide what happens:

1. The kernel does not apply the default action of `SIGTERM` (terminate) to pid 1. A pid 1 that installs no handler ignores `SIGTERM` and is killed at the end of the grace period, with exit code `137` (128 + 9, killed by signal 9). Section 3 shows it happening.
2. Language runtimes may install a handler for you, with their own exit code. A Go program that never calls `signal.Notify` still dies on `SIGTERM`, but with exit code `2`, and without finishing the requests in flight.
3. In **shell form**, `ENTRYPOINT hello` runs `/bin/sh -c hello`. Whether `hello` then becomes pid 1 depends on the shell: BusyBox `sh` replaces itself with a single simple command, but a compound command (`cd /app && ./hello`) leaves the shell as pid 1, and a shell does not forward `SIGTERM` to its child. Shell form also ignores any arguments passed at run time (`docker run IMAGE --flag`, or `args:` in a Kubernetes Pod), which `dep.00` relies on. In **exec form**, `ENTRYPOINT ["/usr/local/bin/hello"]`, your server is pid 1 and receives its arguments, with no shell to reason about.

A server that handles `SIGTERM` stops accepting connections, finishes the requests in flight, and exits `0`. In Go that is `signal.NotifyContext` plus `http.Server.Shutdown`.

**Users.** A process runs as uid 0 (root) unless the image's `USER` says otherwise. Root inside a container is the host's root if the container escapes its isolation, so production images run as an unprivileged uid such as `10001`. Kubernetes can refuse root images (`runAsNonRoot: true`), but it can verify only a **numeric** uid: `USER app` names a user in `/etc/passwd` that the kubelet cannot read, so the pod fails to start with `CreateContainerConfigError`.

### 2.5 Kubernetes: declared state and controllers

A **cluster** is a set of machines, the **nodes**, plus a **control plane**: an API server that stores objects (in etcd) and **controllers**, loops that each watch one kind of object and act to make the world match it. You never start a container on Kubernetes directly. You write an **object** (YAML with `apiVersion`, `kind`, `metadata.name`, `spec`), submit it to the API server, and a controller does the work. On each node, the **kubelet** starts the containers the control plane assigned to that node.

| Object | What it declares | Who acts on it |
|---|---|---|
| **Pod** | one or more containers that share a network namespace (one IP) and volumes; the smallest unit scheduled | the scheduler picks a node; that node's kubelet runs it |
| **ReplicaSet** | "keep N Pods matching this label selector" | the ReplicaSet controller creates or deletes Pods |
| **Deployment** | a Pod template plus replicas and an update strategy; owns one ReplicaSet per template version | the Deployment controller rolls from the old ReplicaSet to the new one |
| **Service** | a stable name and virtual IP for the Pods whose labels match its `selector` | kube-proxy programs each node to forward the IP to a ready Pod |

**Labels and selectors.** A label is a key-value pair on an object (`app.kubernetes.io/name: hello`). A selector is a set of required labels. A Service sends traffic to exactly the Pods whose labels contain all of its selector's pairs, and only to those that are **ready**. The list of ready Pod addresses behind a Service is its **endpoints** (stored as EndpointSlice objects).

**Probes.** The kubelet asks each container two questions on a timer:

- **readiness** ("can you take traffic now?"): while it fails, the Pod is removed from every Service's endpoints. It is not restarted.
- **liveness** ("are you stuck?"): after enough failures, the container is killed and restarted.

A liveness probe must check only the process itself. If it checks a dependency, one dependency blip restarts every replica at once.

**Resources.** `requests` is what the scheduler reserves on a node for the container; `limits` is a hard cap. A container that exceeds its memory limit is killed (`OOMKilled`); one that exceeds its CPU limit is slowed down.

**Ports.** A Service has a `port` (its own), a `targetPort` (the container's port, by number or by the name given in the container's `ports` list), and for `type: NodePort` a `nodePort` in 30000 to 32767 that every node opens. A request to `<any node IP>:<nodePort>` reaches the Service, which forwards it to one ready Pod's `targetPort`.

**Rollouts.** When the Pod template changes (a new image tag, a new env value), the Deployment creates a new ReplicaSet and shifts Pods to it. With `replicas` $R$, `maxSurge` $s$, and `maxUnavailable` $u$ (both default to 25% of $R$), it keeps the count of Pods between a floor and a ceiling. Each terminated old Pod gets `SIGTERM`, then `SIGKILL` after `terminationGracePeriodSeconds` (default 30 s).

| Symbol | Meaning | Type |
|---|---|---|
| $R$ | desired replicas (`spec.replicas`) | integer |
| $s$ | `maxSurge`: extra Pods allowed above $R$; a percentage rounds **up** | integer |
| $u$ | `maxUnavailable`: Pods allowed below $R$ ready; a percentage rounds **down** | integer |

During a rollout, at most $R + s$ Pods exist and at least $R - u$ are ready.

### 2.6 Helm: charts, values, releases

Writing the same Deployment by hand for every environment does not scale. **Helm** packages Kubernetes YAML as a **chart**: a directory with `Chart.yaml` (name, version), `values.yaml` (defaults), and `templates/*.yaml`, which are Go templates. `{{ .Values.greeting }}` reads a value; `{{ .Release.Name }}` is the name you install under; `{{ include "x" . }}` reuses a named template (by convention in `_helpers.tpl`); `| quote`, `| nindent 4`, and `toYaml` are functions that format output.

| Command | Does |
|---|---|
| `helm template NAME CHART [--set k=v]` | render the YAML locally; no cluster needed |
| `helm lint CHART` | check the chart's structure and templates |
| `helm install NAME CHART` | render and create the objects; the set is a **release** named `NAME` |
| `helm upgrade --install NAME CHART` | the same, or update an existing release (a new **revision**) |
| `helm uninstall NAME` | delete the release's objects |

A chart is a function from values to manifests. `helm template` is how you test that function without a cluster, and it is what the course tests do.

### 2.7 kind: a cluster made of containers

**kind** (Kubernetes IN Docker) runs each node as a Docker container that itself runs a kubelet and a container runtime. Three consequences:

1. **Images.** The node has its own image store. An image you built with `docker build` is not in it until you run `kind load docker-image IMAGE --name CLUSTER`. If the Pod's `imagePullPolicy` is `Always`, the kubelet tries a registry anyway and the Pod sits in `ErrImagePull`. The default policy is `Always` when the tag is `latest` (or missing) and `IfNotPresent` otherwise.
2. **Ports.** A NodePort opens on the node, which is a container. Your laptop reaches it only through a Docker port mapping on that node container, declared as `extraPortMappings` in the cluster config, and Docker fixes port mappings when a container is created. A mapping added later needs `kind delete cluster` and `kind create cluster`.
3. **Contexts.** `kind create cluster --name lang07` adds a **kube context** named `kind-lang07` to `~/.kube/config`. `kubectl --context kind-lang07` and `helm --kube-context kind-lang07` talk to it, whatever your current context is.

## 3. Worked example by hand

A different program, so the exercise stays yours: a static page served by BusyBox `httpd`, which is in the official `busybox` image. Everything below can be checked with a pencil before you run it, and every number was then checked by running it.

**The Dockerfile**, next to a 114-byte `index.html`:

```dockerfile
FROM busybox:1.36.1                    # layer 1: the base, 4.24 MB
COPY index.html /www/index.html        # layer 2: one file
USER 10001:10001                       # config only
EXPOSE 8080                            # config only
ENTRYPOINT ["busybox", "httpd", "-f", "-p", "8080", "-h", "/www"]
```

- Layers: 2, the base and one `COPY`. `USER`, `EXPOSE`, and `ENTRYPOINT` change only the config, so `docker history page:1.0.0` lists them with size `0B`. The `COPY` layer shows `12.3kB` for a 114-byte file: a layer is a tar archive, and tar pads every archive to 10 KiB records and adds a 512-byte header per entry.
- Rebuild after editing `index.html`: the `FROM` layer comes from the cache; the `COPY` layer and everything after it rebuild. Rebuild after changing only `EXPOSE`: the `COPY` layer is still cached.
- The user is `10001`, numeric, so `runAsNonRoot` can verify it. Port 8080 is above 1023, so an unprivileged user may bind it.

**Run it, then stop it.**

```bash
docker build -t page:1.0.0 .
docker run -d --name p -p 127.0.0.1:8081:8080 page:1.0.0
curl -s http://127.0.0.1:8081/          # the 114 bytes of index.html
docker stop -t 5 p                      # takes 5 s
docker inspect -f '{{.State.ExitCode}}' p   # 137
```

The request path: `127.0.0.1:8081` on the host, Docker's forward into the container's network namespace, port 8080, `httpd`. The stop is the lesson. The exec-form `ENTRYPOINT` makes `httpd` pid 1, but BusyBox `httpd` installs no `SIGTERM` handler, and the kernel ignores an unhandled `SIGTERM` sent to pid 1 (section 2.4). So `docker stop` waits the whole 5 s grace period, sends `SIGKILL`, and the exit code is 128 + 9 = 137. Your Go server in section 4 must stop in well under the grace period and exit `0`, and `test_container_exits_zero_on_sigterm` measures exactly that.

**The Kubernetes objects**, written as a chart with `values.yaml` holding `replicas: 4` and `nodePort: 31008`. Render `templates/deployment.yaml` by hand with `helm install page ./chart`:

```yaml
# template                                         # rendered, Release.Name = page
metadata:                                          metadata:
  name: {{ .Release.Name }}-page                     name: page-page
spec:                                              spec:
  replicas: {{ .Values.replicas }}                   replicas: 4
```

The Service selects `app.kubernetes.io/name: page`, `targetPort: http`, and the container lists `ports: [{name: http, containerPort: 8080}]`, so `targetPort: http` resolves to 8080.

**The rollout arithmetic.** Change the image tag to `page:1.0.1` with $R = 4$ and the default 25%:

- $s = \lceil 0.25 \times 4 \rceil = 1$, so at most $4 + 1 = 5$ Pods exist.
- $u = \lfloor 0.25 \times 4 \rfloor = 1$, so at least $4 - 1 = 3$ are ready.

With $R = 2$ instead: $s = \lceil 0.5 \rceil = 1$ and $u = \lfloor 0.5 \rfloor = 0$, so the controller starts one new Pod, waits for it to be ready, and only then stops an old one. Two replicas never drop below two ready Pods during a rollout. That is why the exercise chart defaults to `replicas: 2`.

**One request through kind**, `curl http://127.0.0.1:31008/` with the cluster config mapping 31008:

1. `127.0.0.1:31008` on your laptop: Docker's port mapping on the kind node container (`extraPortMappings`).
2. Port 31008 on the node: kube-proxy's NodePort rule for Service `page-page`.
3. The Service picks one ready endpoint, for example `10.244.0.7:8080`, a Pod's IP and `targetPort`.
4. `httpd` in that Pod answers.

If step 3 has no ready endpoint (the image was never loaded, the selector is wrong, the readiness probe fails), the connection is accepted at step 2 and then reset. `kubectl get endpointslices -l kubernetes.io/service-name=page-page` shows which.

## 4. The interface

The exercise lives in your repo at `primers/lang.07/` and is not part of your system. The check runs from your repo root.

| Path | Requirement |
|---|---|
| `primers/lang.07/go.mod`, `main.go` | Go module `primer`. Listens on `:8080` (`PORT` overrides). `GET /healthz` answers `200`. `GET /` answers `200` with JSON `{"greeting": <$GREETING>, "pod": <hostname>}`. On `SIGTERM` it stops accepting connections, drains, and exits `0` |
| `primers/lang.07/Dockerfile` | build context `primers/lang.07`; at least two stages; every `FROM` pinned to a tag other than `latest` (or a digest); final stage `USER` a numeric uid other than 0; `EXPOSE 8080`; exec-form `ENTRYPOINT` or `CMD` |
| `primers/lang.07/kind.yaml` | `kind: Cluster`, `apiVersion: kind.x-k8s.io/v1alpha4`, `name: lang07`; a node maps `containerPort: 31007` to `hostPort: 31007` |
| `primers/lang.07/chart/` | `helm lint` clean; renders exactly one Deployment and one Service. Service: `type: NodePort`, `nodePort: 31007`, selector matching the Pod labels, `targetPort` resolving to container port 8080. Container: readiness `httpGet` on `/healthz`, `resources.limits`, `runAsNonRoot: true`, a pinned image tag, `imagePullPolicy` not `Always`. The value `greeting` sets the env var `GREETING` |
| on kind | release installed on context `kind-lang07`; the Service has ready endpoints; `http://127.0.0.1:31007/` answers from one of its Pods |

### What the tests check

`ol check lang.07` runs `course/tests/lang.07/check` in your repo. Tests run in order, in three tiers.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_files_present` | unit | the files of the table above exist | every later test names its file |
| `test_dockerfile_multi_stage_pinned_non_root` | unit | the Dockerfile rules, statically | `dep.00` applies them to the engine and gateway images |
| `test_kind_config_maps_the_nodeport` | unit | `name: lang07` and the 31007 mapping | `dep.00` maps 30080 and 30686 the same way |
| `test_chart_lints` | conformance | `helm lint` exits 0 | template errors surface before a cluster sees them |
| `test_chart_renders_deployment_and_nodeport_service` | unit | one Deployment, one NodePort Service, selector and `targetPort` consistent | a mismatch serves nothing |
| `test_pod_spec_is_production_shaped` | unit | probe, limits, `runAsNonRoot`, pinned tag, pull policy | the chart policy `dep.03` enforces |
| `test_values_reach_the_pod` | unit | `--set greeting=ol-probe-lang07` reaches `GREETING` | charts are functions of values |
| `test_image_builds` | unit | `docker build` of `primers/lang.07` | a clean-context build is what CI does |
| `test_image_user_is_numeric_non_root` | boundary | the image config's `User` | what the kubelet verifies |
| `test_container_serves_health_and_greeting` | unit | `/healthz`, then `/` returns the env greeting and the container hostname | configuration through the environment |
| `test_container_exits_zero_on_sigterm` | fault | `docker stop -t 5` ends with exit 0 well inside 5 s | rollouts and drills in `ops.00` |
| `test_service_has_ready_endpoints_on_kind` | conformance | ready endpoints behind the 31007 Service on `kind-lang07` | the image was loaded and the probe passes |
| `test_nodeport_answers_from_a_pod` | conformance | `GET http://127.0.0.1:31007/` names a running Pod | the full path of section 3 |

The last two form the **cluster tier**. Without the context `kind-lang07` they fail and say so; `OL_SMOKE=1 ol check lang.07` skips them with the reason and checks everything else (the summary line ends `(smoke: cluster tier skipped)`).

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. One stage, or `FROM golang` without a tag | a 300 MB image with a compiler in it; builds that change under you when `latest` moves | `test_dockerfile_multi_stage_pinned_non_root` |
| 2. No `SIGTERM` handler, or a shell-form `ENTRYPOINT` | without a handler Go exits `2` mid-request; a shell left as pid 1 makes `docker stop` take the full grace period and exit `137`; shell form also drops the `args:` a chart passes | `test_container_exits_zero_on_sigterm`, `test_dockerfile_multi_stage_pinned_non_root` |
| 3. `USER app` (a name) or no `USER` | the Pod never starts: `CreateContainerConfigError: image has non-numeric user` (or it runs as root) | `test_image_user_is_numeric_non_root`, `test_dockerfile_multi_stage_pinned_non_root` |
| 4. `image: hello:latest` or `imagePullPolicy: Always` on kind | `ErrImagePull`, `ImagePullBackOff`: the node tries a registry that has no such image | `test_pod_spec_is_production_shaped` |
| 5. NodePort mapping missing from `kind.yaml`, or added after `kind create cluster` | `curl: (7) Failed to connect to 127.0.0.1 port 31007` while `kubectl get pods` looks healthy | `test_kind_config_maps_the_nodeport`, `test_nodeport_answers_from_a_pod` |
| 6. Service selector or `targetPort` that does not match the Pod | `curl: (52) Empty reply` or a reset; the Service has no endpoints | `test_chart_renders_deployment_and_nodeport_service` |
| 7. Forgot `kind load docker-image` after a rebuild | old code keeps running (same tag), or `ErrImagePull` on a new tag | `test_service_has_ready_endpoints_on_kind` |
| 8. A liveness probe that calls a dependency | every replica restarts when the dependency blips | review only: the exercise has no dependency; in dep.00 the gateway's readiness, never its liveness, depends on the engine |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | the Go server you containerize here |
| Back | `lang.02` | processes, exit codes, and signals (`SIGTERM`, `SIGKILL`) |
| Forward | `dep.00` | engine and gateway images, two Helm charts, a kind cluster with NodePorts 30080 and 30686 |
| Forward | `obs.00` | Jaeger runs as one more Deployment and Service on the same cluster |
| Forward | `ops.00` | `kubectl describe`, `logs --previous`, and `rollout undo` on a crashlooping engine |
| Forward | `dep.01` to `dep.03` | pinned digests, `HEALTHCHECK`, registries, and the chart policy, enforced |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `docker build` | BuildKit with cache mounts and `docker buildx` | parallel stages, `--mount=type=cache` for module caches, multi-arch images | [BuildKit docs](https://docs.docker.com/build/buildkit/) |
| `alpine` runtime stage | distroless or `scratch` | no shell or package manager at all; smaller attack surface | [distroless](https://github.com/GoogleContainerTools/distroless) |
| NodePort | Ingress or the Gateway API with a load balancer | one entry point with hostnames, TLS, and routing rules | [Gateway API](https://gateway-api.sigs.k8s.io/) |
| a Helm chart | Kustomize overlays, or Helm with a GitOps controller (Argo CD, Flux) | the cluster pulls declared state from git | [Argo CD](https://argo-cd.readthedocs.io/) |
| kind | k3d, minikube, or a managed cluster (GKE, EKS) | real nodes, cloud load balancers, autoscaling | [k3d](https://k3d.io/) |
