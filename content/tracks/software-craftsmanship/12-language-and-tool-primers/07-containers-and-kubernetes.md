# Containers and Kubernetes: images, layers, Pods, Deployments, Services, Helm, kind

## Overview

| | |
|---|---|
| **Module** | `lang.07` · practice · ops · Pass 1 · 6 to 8 h |
| **Solution** | [`primers/lang.07/`](../../../../course/solutions/primers/lang.07/): a tiny Go server [`main.go`](../../../../course/solutions/primers/lang.07/main.go) (`main`: `/healthz`, `/`, `signal.NotifyContext` plus `Shutdown`), its [`Dockerfile`](../../../../course/solutions/primers/lang.07/Dockerfile), a Helm chart [`chart/`](../../../../course/solutions/primers/lang.07/chart/) (one Deployment, one NodePort Service), and a kind cluster config [`kind.yaml`](../../../../course/solutions/primers/lang.07/kind.yaml), deployed and reached through a NodePort |
| **Contract** | none: the layout and rules in section 4 are the contract |
| **Needs** | `lang.06` Go (reading), [`infrastructure/01-containers-kubernetes`](../../infrastructure/01-containers-kubernetes/README.md) (reading) |
| **Used by** | `dep.00` tracer deploy (the same skills, applied to the engine and gateway, so that by the end of Pass 1 the whole tracer runs on a cluster) |
| **Optional depth** | [Kubernetes documentation](https://kubernetes.io/docs/concepts/) (free), [Docker build docs](https://docs.docker.com/build/) (free), [OCI image spec](https://github.com/opencontainers/image-spec) (free), [Helm docs](https://helm.sh/docs/) (free), [kind docs](https://kind.sigs.k8s.io/) (free) |

## Key Takeaways

- An **image** is a stack of content-addressed filesystem layers plus a config (entrypoint, user, ports, env). A **container** is an ordinary Linux process started from that config, with its own view of the filesystem, processes, and network.
- A **multi-stage build** compiles in a fat stage and copies one binary into a small runtime stage that runs as a **numeric non-root user**. The image config records the user; `docker stop` shows the exit code.
- Your server is **PID 1** in its container. It must handle `SIGTERM` itself and exit `0`, or every `docker stop` and every rollout kills it after the grace period.
- Kubernetes is a set of **controllers** that drive actual state toward the objects you declare. A **Deployment** keeps N **Pods** running; a **Service** finds them by label and gives them one stable address; a **NodePort** opens that address on every node.
- **Helm** renders templates plus values into those objects. **kind** runs a whole cluster as Docker containers, which is why a NodePort needs a port mapping fixed when the cluster is created.

## How to read this chapter

Sections 1 and 2 build containers and Kubernetes up from processes and namespaces. Section 3 works a different image and chart by hand, including the rollout arithmetic. Section 4 fixes the layout and rules, and section 5 walks through the solution file by file and explains why each line is there. Try to predict each file before reading it. The solution is used like this, from the repo root:

```bash
docker build -t hello:0.1.0 primers/lang.07
kind create cluster --config primers/lang.07/kind.yaml
kind load docker-image hello:0.1.0 --name lang07
helm install hello primers/lang.07/chart --kube-context kind-lang07
curl -s http://127.0.0.1:31007/
kind delete cluster --name lang07   # when you are done; dep.00 makes its own cluster
```

---

## 1. Why now

At this point in Pass 1 the system runs as processes on your laptop: a Rust engine you start with `cargo run`, a Go gateway you start with `go run`, both reading files from paths that exist only on your machine. The end of Pass 1 asks for more: the same two programs must run on a Kubernetes cluster, reachable from outside it, restartable by a controller, and observable. None of that is possible until each program is an image with a known entrypoint, user, and port, and until you can describe "run one of these, keep it healthy, and give it an address" as data the cluster understands. This primer teaches exactly those pieces on a server small enough that nothing else can go wrong. In `dep.00` they are applied, unchanged, to the engine and gateway.

## 2. Principles

Every term is defined before it is used. Read this section once straight through; section 4 fixes what the solution must contain.

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

A chart is a function from values to manifests. `helm template` is how you test that function without a cluster.

### 2.7 kind: a cluster made of containers

**kind** (Kubernetes IN Docker) runs each node as a Docker container that itself runs a kubelet and a container runtime. Three consequences:

1. **Images.** The node has its own image store. An image you built with `docker build` is not in it until you run `kind load docker-image IMAGE --name CLUSTER`. If the Pod's `imagePullPolicy` is `Always`, the kubelet tries a registry anyway and the Pod sits in `ErrImagePull`. The default policy is `Always` when the tag is `latest` (or missing) and `IfNotPresent` otherwise.
2. **Ports.** A NodePort opens on the node, which is a container. Your laptop reaches it only through a Docker port mapping on that node container, declared as `extraPortMappings` in the cluster config, and Docker fixes port mappings when a container is created. A mapping added later needs `kind delete cluster` and `kind create cluster`.
3. **Contexts.** `kind create cluster --name lang07` adds a **kube context** named `kind-lang07` to `~/.kube/config`. `kubectl --context kind-lang07` and `helm --kube-context kind-lang07` talk to it, whatever your current context is.

## 3. Worked example by hand

A different program, so the solution in section 5 is not given away: a static page served by BusyBox `httpd`, which is in the official `busybox` image. Everything below can be checked with a pencil before you run it, and every number was then checked by running it.

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

The request path: `127.0.0.1:8081` on the host, Docker's forward into the container's network namespace, port 8080, `httpd`. The stop is the lesson. The exec-form `ENTRYPOINT` makes `httpd` pid 1, but BusyBox `httpd` installs no `SIGTERM` handler, and the kernel ignores an unhandled `SIGTERM` sent to pid 1 (section 2.4). So `docker stop` waits the whole 5 s grace period, sends `SIGKILL`, and the exit code is 128 + 9 = 137. The Go server of section 5 must stop in well under the grace period and exit `0`; `docker stop -t 5` followed by `docker inspect` measures exactly that.

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

The exercise lives at `primers/lang.07/` and is not part of the system. Paths are relative to the repo root.

| Path | Requirement |
|---|---|
| `primers/lang.07/go.mod`, `main.go` | Go module `primer`. Listens on `:8080` (`PORT` overrides). `GET /healthz` answers `200`. `GET /` answers `200` with JSON `{"greeting": <$GREETING>, "pod": <hostname>}`. On `SIGTERM` it stops accepting connections, drains, and exits `0` |
| `primers/lang.07/Dockerfile` | build context `primers/lang.07`; at least two stages; every `FROM` pinned to a tag other than `latest` (or a digest); final stage `USER` a numeric uid other than 0; `EXPOSE 8080`; exec-form `ENTRYPOINT` or `CMD` |
| `primers/lang.07/kind.yaml` | `kind: Cluster`, `apiVersion: kind.x-k8s.io/v1alpha4`, `name: lang07`; a node maps `containerPort: 31007` to `hostPort: 31007` |
| `primers/lang.07/chart/` | `helm lint` clean; renders exactly one Deployment and one Service. Service: `type: NodePort`, `nodePort: 31007`, selector matching the Pod labels, `targetPort` resolving to container port 8080. Container: readiness `httpGet` on `/healthz`, `resources.limits`, `runAsNonRoot: true`, a pinned image tag, `imagePullPolicy` not `Always`. The value `greeting` sets the env var `GREETING` |
| on kind | release installed on context `kind-lang07`; the Service has ready endpoints; `http://127.0.0.1:31007/` answers from one of its Pods |

### What a correct solution must handle

The cases come in three tiers: static checks of the files, checks of the built image and a running container, and a cluster tier on `kind-lang07`.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Every file present | unit | the files of the table above exist | every later case names its file |
| Dockerfile multi-stage, pinned, non-root | unit | the Dockerfile rules, statically | `dep.00` applies them to the engine and gateway images |
| kind config maps the NodePort | unit | `name: lang07` and the 31007 mapping | `dep.00` maps 30080 and 30686 the same way |
| Chart lints | conformance | `helm lint` exits 0 | template errors surface before a cluster sees them |
| Chart renders a Deployment and a NodePort Service | unit | one Deployment, one NodePort Service, selector and `targetPort` consistent | a mismatch serves nothing |
| Pod spec is production-shaped | unit | probe, limits, `runAsNonRoot`, pinned tag, pull policy | the chart policy `dep.03` enforces |
| Values reach the Pod | unit | `--set greeting=...` reaches `GREETING` | charts are functions of values |
| Image builds | unit | `docker build` of `primers/lang.07` | a clean-context build is what CI does |
| Image user is numeric, non-root | boundary | the image config's `User` | what the kubelet verifies |
| Container serves health and greeting | unit | `/healthz`, then `/` returns the env greeting and the container hostname | configuration through the environment |
| Container exits 0 on `SIGTERM` | fault | `docker stop -t 5` ends with exit 0 well inside 5 s | rollouts and drills in `ops.00` |
| Service has ready endpoints on kind | conformance | ready endpoints behind the 31007 Service on `kind-lang07` | the image was loaded and the probe passes |
| NodePort answers from a Pod | conformance | `GET http://127.0.0.1:31007/` names a running Pod | the full path of section 3 |

The last two need a running cluster with the context `kind-lang07`; everything else runs with Docker and Helm alone.

## 5. The solution, step by step

### 5.1 From first principles

1. **The program must be configurable without a rebuild.** The image is built once and run in many places, so everything that differs per place (greeting, port) comes from the environment, with defaults. The pod name is not configuration at all: it is the hostname the uts namespace gives the container, which is also the Pod name on Kubernetes.
2. **The program is pid 1, so it owns `SIGTERM`.** Section 2.4: an unhandled `SIGTERM` to pid 1 is ignored, and Go's default handler exits `2` without draining. So the server must catch `SIGTERM`, stop accepting, finish what is in flight, and return from `main` normally (exit `0`), all well inside the grace period.
3. **What ships is only the binary.** The compiler is needed to build, not to run, so two stages: compile with `golang`, copy one static binary (`CGO_ENABLED=0`, so it needs no libc from the build stage) into a small runtime stage. Every `FROM` is pinned to a tag so the build does not change under you.
4. **The user must be provably non-root.** The kubelet can only compare a number with 0, so `USER 10001:10001`, and port 8080 (above 1023) so that user may bind it.
5. **The entrypoint must not put a shell between the kernel and the server.** Exec form makes the binary pid 1 and lets it receive `args:`.
6. **The cluster objects must agree with each other.** One set of labels, defined once, is used for the Deployment's selector, the Pod template, and the Service's selector; the container names its port `http`, and both the probe and the Service's `targetPort` refer to that name. Nothing can drift because nothing is written twice.
7. **kind needs two things decided up front.** The image has to be in the node's store and never pulled (`IfNotPresent` with a pinned tag), and the NodePort has to be mapped to the host when the cluster is created.

### 5.2 The code, annotated

**The server.** Configuration from the environment with defaults, then the two routes.

```go
func main() {
	greeting := os.Getenv("GREETING") // twelve-factor: set per environment, never baked in
	if greeting == "" {
		greeting = "hello"
	}
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	pod, _ := os.Hostname() // the uts namespace: the container id prefix, or the Pod name

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n")) // checks the process only: safe for liveness too
	})
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"greeting": greeting, "pod": pod})
	})
	// ...
```

`GET /{$}` (Go 1.22 patterns) matches exactly `/`, not every path, so a typo in a URL is a 404 instead of a greeting. The hostname is read once at start: it cannot change for the life of the container.

**Shutdown.** This is the part that decides the exit code.

```go
	srv := &http.Server{Addr: ":" + port, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	// ctx is cancelled on SIGTERM (docker stop, a rollout) or Ctrl-C.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()

	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	log.Printf("listening on :%s", port)

	select {
	case err := <-errc:
		log.Fatal(err) // could not bind: exit non-zero, the controller will restart us
	case <-ctx.Done(): // SIGTERM: fall through to the drain
	}
	// Stop accepting, wait for in-flight requests, but never longer than 5 s:
	// well inside docker's 10 s and Kubernetes' 30 s grace periods.
	shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdown); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	// main returns: exit code 0
}
```

`signal.NotifyContext` replaces Go's default `SIGTERM` behaviour (exit `2`) with a cancelled context. `ListenAndServe` runs in a goroutine so `main` can wait on either a bind error or the signal. `Shutdown` closes the listeners first, so no new connection is accepted, then waits for active ones. `ReadHeaderTimeout` stops a slow client from holding a connection open forever, which would also hold up the drain.

**The Dockerfile.**

```dockerfile
# Stage 1: compile with the full Go toolchain (about 250 MB, never shipped).
FROM golang:1.24-alpine AS build
WORKDIR /src
# go.mod first: this layer stays cached until the module file changes.
COPY go.mod ./
COPY *.go ./
# Static binary (no cgo, so no libc dependency on the build stage); -trimpath and
# -s -w strip local paths and symbols, so the binary is smaller and reproducible.
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/hello .

# Stage 2: the runtime is a small base plus one static binary.
FROM alpine:3.19
COPY --from=build /out/hello /usr/local/bin/hello
# A numeric uid, so Kubernetes `runAsNonRoot` can prove it is not root.
USER 10001:10001
EXPOSE 8080
# Exec form: hello is PID 1 and receives SIGTERM directly.
ENTRYPOINT ["/usr/local/bin/hello"]
```

The build context is `primers/lang.07`, and [`.dockerignore`](../../../../course/solutions/primers/lang.07/.dockerignore) leaves out `chart/`, `kind.yaml`, and Markdown, so editing the chart does not invalidate the `COPY` cache or ship into the build context.

**The kind cluster.** The mapping must exist at creation (section 2.7), and binds only to loopback.

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: lang07                   # context kind-lang07
nodes:
  - role: control-plane
    extraPortMappings:         # host 127.0.0.1:31007 -> node container :31007
      - containerPort: 31007
        hostPort: 31007
        listenAddress: "127.0.0.1"
        protocol: TCP
```

**The chart.** The labels are defined once in `_helpers.tpl` and included everywhere a selector or label appears:

```yaml
{{- define "hello.labels" -}}
app.kubernetes.io/name: hello
app.kubernetes.io/instance: {{ .Release.Name }}  # two releases never select each other's Pods
{{- end -}}
```

The Deployment's container (the part that carries the policy):

```yaml
    spec:
      securityContext:
        runAsNonRoot: true                       # kubelet refuses uid 0; works because USER is numeric
      containers:
        - name: hello
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"   # hello:0.1.0, pinned
          imagePullPolicy: {{ .Values.image.pullPolicy }}                   # IfNotPresent: use kind's copy
          env:
            - name: GREETING
              value: {{ .Values.greeting | quote }}   # quote: a value like "yes" stays a string
          ports:
            - name: http                         # the name the probe and the Service use
              containerPort: 8080
          readinessProbe:                        # out of the endpoints while it fails
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 2
          livenessProbe:                         # restarted only if the process itself is stuck
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 10
          resources:
            {{- toYaml .Values.resources | nindent 12 }}   # requests and limits from values
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true         # the server writes nothing to disk
```

And the Service, which selects the same labels and targets the port by name:

```yaml
spec:
  type: NodePort
  selector:
    {{- include "hello.labels" . | nindent 4 }}
  ports:
    - name: http
      port: {{ .Values.service.port }}           # 80, the in-cluster port
      targetPort: http                           # resolves to containerPort 8080
      nodePort: {{ .Values.service.nodePort }}   # 31007, the one kind.yaml maps
```

[`values.yaml`](../../../../course/solutions/primers/lang.07/chart/values.yaml) sets `replicas: 2` (section 3: with $R = 2$ a rollout never drops below two ready Pods), the pinned tag `"0.1.0"` with `pullPolicy: IfNotPresent`, and small requests and limits (`10m`/`16Mi`, capped at `100m`/`32Mi`).

Full solution: [`main.go`](../../../../course/solutions/primers/lang.07/main.go), [`Dockerfile`](../../../../course/solutions/primers/lang.07/Dockerfile), [`kind.yaml`](../../../../course/solutions/primers/lang.07/kind.yaml), [`chart/`](../../../../course/solutions/primers/lang.07/chart/).

### 5.3 Why this solution

- **Image size and attack surface.** The runtime image is Alpine (a few MB) plus a stripped static binary, instead of a 300 MB toolchain image. `scratch` or distroless would be smaller still (see Going further); Alpine keeps a shell for `kubectl exec` while learning.
- **Graceful shutdown costs a dozen lines.** The alternative, relying on the runtime's default, exits `2` mid-request on every rollout. A `tini`-style init process would forward signals but not drain HTTP requests; only the server knows what "in flight" means.
- **Port by name, labels by include.** Writing `targetPort: 8080` and copying the labels three times works until one copy changes. Using one source for each makes the Service-to-Pod link impossible to break by editing one file.
- **NodePort on kind** is the smallest way to reach a Service from the laptop. It costs a fixed port mapping at cluster creation; an Ingress would need a controller installed first.
- **Accepted tradeoff:** the same `/healthz` serves readiness and liveness. That is right here because the server has no dependency; once it has one (the gateway and the engine in `dep.00`), readiness checks the dependency and liveness must not.

### 5.4 Explain it back

1. Why does the server exit `0` on `SIGTERM`, and what would `docker stop` report without `signal.NotifyContext`?
   <details><summary>Answer</summary>
   `NotifyContext` turns `SIGTERM` into a cancelled context; `main` then calls `Shutdown` and returns normally, which is exit `0`. Without it, Go's runtime handles `SIGTERM` itself and exits with code `2`, without finishing the requests in flight. (A pid 1 with no handler at all, like BusyBox `httpd` in section 3, is not terminated by `SIGTERM` and is killed at the end of the grace period with `137`.)
   </details>
2. Why `USER 10001:10001` and not `USER app` with an `adduser` line?
   <details><summary>Answer</summary>
   `runAsNonRoot: true` makes the kubelet verify the uid before starting the container, and it can only compare a number with 0. A user name would need `/etc/passwd` from inside the image, which the kubelet does not read, so the Pod fails with `CreateContainerConfigError`.
   </details>
3. What happens if `values.yaml` sets the tag to `latest` and you remove `pullPolicy`?
   <details><summary>Answer</summary>
   With tag `latest` the default policy is `Always`, so the kubelet ignores the image `kind load` put in the node's store and tries a registry that has no `hello` image: `ErrImagePull`, then `ImagePullBackOff`. A pinned tag with `IfNotPresent` uses the loaded copy.
   </details>
4. Why does the Service say `targetPort: http` rather than `8080`, and why do all selectors come from `hello.labels`?
   <details><summary>Answer</summary>
   Both remove a second copy of a fact. If the container port changes, the named port follows it; if a label changes, the Deployment selector, the Pod labels, and the Service selector change together. A mismatch in either would leave the Service with no endpoints: connections accepted and then reset.
   </details>
5. Why must `extraPortMappings` be in `kind.yaml` before `kind create cluster`, when the NodePort itself is created later by Helm?
   <details><summary>Answer</summary>
   The NodePort opens port 31007 on the node, and the node is a Docker container. The laptop reaches it only through a Docker port mapping on that container, and Docker fixes port mappings when a container is created. The Service can come later; the mapping cannot.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. One stage, or `FROM golang` without a tag | a 300 MB image with a compiler in it; builds that change under you when `latest` moves | two stages, `golang:1.24-alpine` and `alpine:3.19`, both pinned |
| 2. No `SIGTERM` handler, or a shell-form `ENTRYPOINT` | without a handler Go exits `2` mid-request; a shell left as pid 1 makes `docker stop` take the full grace period and exit `137`; shell form also drops the `args:` a chart passes | `signal.NotifyContext` plus `Shutdown` with a 5 s cap; exec-form `ENTRYPOINT` (step 2 and 5) |
| 3. `USER app` (a name) or no `USER` | the Pod never starts: `CreateContainerConfigError: image has non-numeric user` (or it runs as root) | `USER 10001:10001` in the final stage |
| 4. `image: hello:latest` or `imagePullPolicy: Always` on kind | `ErrImagePull`, `ImagePullBackOff`: the node tries a registry that has no such image | tag `"0.1.0"` and `pullPolicy: IfNotPresent` in `values.yaml` |
| 5. NodePort mapping missing from `kind.yaml`, or added after `kind create cluster` | `curl: (7) Failed to connect to 127.0.0.1 port 31007` while `kubectl get pods` looks healthy | `extraPortMappings` for 31007 in `kind.yaml`, used at creation |
| 6. Service selector or `targetPort` that does not match the Pod | `curl: (52) Empty reply` or a reset; the Service has no endpoints | one `hello.labels` helper for every selector; `targetPort: http` by name |
| 7. Forgot `kind load docker-image` after a rebuild | old code keeps running (same tag), or `ErrImagePull` on a new tag | the usage sequence loads after every build; bump the tag when the code changes |
| 8. A liveness probe that calls a dependency | every replica restarts when the dependency blips | `/healthz` checks only the process; in dep.00 the gateway's readiness, never its liveness, depends on the engine |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | the Go server containerized here |
| Back | `lang.02` | processes, exit codes, and signals (`SIGTERM`, `SIGKILL`) |
| Forward | `dep.00` | engine and gateway images, two Helm charts, a kind cluster with NodePorts 30080 and 30686 |
| Forward | `obs.00` | Jaeger runs as one more Deployment and Service on the same cluster |
| Forward | `ops.00` | `kubectl describe`, `logs --previous`, and `rollout undo` on a crashlooping engine |
| Forward | `dep.01` to `dep.03` | pinned digests, `HEALTHCHECK`, registries, and the chart policy, enforced |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `docker build` | BuildKit with cache mounts and `docker buildx` | parallel stages, `--mount=type=cache` for module caches, multi-arch images | [BuildKit docs](https://docs.docker.com/build/buildkit/) |
| `alpine` runtime stage | distroless or `scratch` | no shell or package manager at all; smaller attack surface | [distroless](https://github.com/GoogleContainerTools/distroless) |
| NodePort | Ingress or the Gateway API with a load balancer | one entry point with hostnames, TLS, and routing rules | [Gateway API](https://gateway-api.sigs.k8s.io/) |
| a Helm chart | Kustomize overlays, or Helm with a GitOps controller (Argo CD, Flux) | the cluster pulls declared state from git | [Argo CD](https://argo-cd.readthedocs.io/) |
| kind | k3d, minikube, or a managed cluster (GKE, EKS) | real nodes, cloud load balancers, autoscaling | [k3d](https://k3d.io/) |
