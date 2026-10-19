# Agent image and chart

## Overview

| | |
|---|---|
| **Module** | `dep.07` · practice · Docker and Helm · Pass 10 · 1 to 2 h |
| **Solution** | [`deploy/docker/agent.Dockerfile`](../../../../course/solutions/entry/deploy/docker/agent.Dockerfile) and [`deploy/helm/forge-agent/`](../../../../course/solutions/entry/deploy/helm/forge-agent/values.yaml) (`templates/deployment.yaml`, `templates/pdb.yaml`, `values.schema.json`), running the agent worker [`go/cmd/agent`](../../../../course/solutions/entry/go/cmd/agent/main.go) |
| **Contract** | [`agent.values.schema.json`](../../../../course/contracts/helm/agent.values.schema.json) |
| **Needs** | `dep.06` worker chart, `ag.05` durable agent runs |
| **Used by** | the Pass 10 agent platform runs the worker on the `agent` queue next to the durable service and the gateway · `ops.10` the runaway-agent drill |
| **Optional depth** | Separate agent and tool sandbox workloads |

## Key takeaways

- The image runs the agent worker as a numeric non-root user, built static from Go onto a digest-pinned alpine.
- The chart consumes the durable address and the `agent` queue.
- Provider keys are injected only through Kubernetes Secret references.
- Resource limits and probes make failure visible to the scheduler.

## How to read this chapter

Sections 1 to 3 explain why the agent worker gets its own workload and trace its configuration by hand. Section 4 fixes what a correct image and chart must do. Section 5 walks through the solution Dockerfile and chart, explains each choice, and points out two values worth changing. Try to predict each part before reading it.

## 1. Why now

`ag.05` can resume durable agent runs, but a worker process must be packaged and scheduled for the agent queue. A dedicated chart lets it scale and restart independently from gateway and engine services.

## 2. Principles

Use a pinned multi-stage base image, copy only the built worker and runtime files, set a non-root numeric uid, and provide CPU and memory requests and limits. Readiness and liveness probes use the worker health endpoint. Configuration identifies the durable service, queue, provider URL, model, artifacts mount, and optional OTel endpoint. Any API key comes from a Secret key reference.

| Setting | Purpose |
|---|---|
| `durableAddress` | gRPC address for durable workflow service |
| `provider.baseUrl` | the system's gateway or an explicitly configured frontier endpoint |
| `provider.apiKey` | Secret reference, never a literal |
| `replicaCount` | bounded worker count for this local chart (the schema allows exactly 1) |

## 3. Worked example

The chart points `durableAddress` at `forge-durable:7233` (the Service of `dep.06`), points the provider at `http://forge-gateway:8080` with model `smollm2-135m-instruct`, mounts `/artifacts` read-write for run state, and reads `PROVIDER_API_KEY` from Secret `agent-provider`, key `api-key`. The worker resolves its settings in a fixed order: defaults (queue `agent`, health on `:9464`, artifacts at `/artifacts`), then a `--config` file if given, then flags (`--queue agent --durable-address forge-durable:7233`), then the `PROVIDER_*` environment. Its run journal lands in `/artifacts/agent-runs`, so a retried activity replays the completed model and tool steps instead of repeating them. A readiness failure removes the pod from service while its durable workflow lease can expire and be reclaimed.

## 4. The artifact and what it must do

The image must have a pinned base, a numeric non-root runtime user, and only the built binary in its final stage. The chart must render under the contract schema, select the `agent` queue, read `PROVIDER_API_KEY` with `secretKeyRef` (the values file holds only the Secret name and key), and carry probes, resource bounds, and a PodDisruptionBudget. A chart must not carry provider credentials in values files or image layers.

### What a correct solution must handle

| Case | What must hold | Why it matters downstream |
|---|---|---|
| image policy | no root execution, no mutable base tags, no broad source copies (`.env` stays out) | the dep.01 rules for every image |
| chart schema | the chart's `values.schema.json` equals the published contract | the same shape as every other chart |
| queue and secret | the `agent` queue, the Secret reference, probes, resource bounds, a PodDisruptionBudget | the worker takes agent work only, and no key leaks |

## 5. The solution, step by step

### 5.1 From first principles

1. **The agent worker is a separate failure domain.** Agent runs are long and call out to a model and tools; sharing a pod with other workers would let a runaway run starve them. So it gets its own image and Deployment, consuming one queue.
2. **The queue is the routing.** The durable server hands `AgentRun` tasks only to pollers of queue `agent`, so the chart must pass `--queue agent` explicitly rather than rely on a default that could change.
3. **The key never touches Helm.** The provider key enters as `PROVIDER_API_KEY` through `secretKeyRef`; values hold only `{name, key}`, which the contract schema enforces.
4. **Run state must survive the pod.** The durable agent journal under `/artifacts/agent-runs` is what makes a retried run resume, so `/artifacts` is mounted read-write, and the root filesystem stays read-only with an `emptyDir` at `/tmp`.
5. **The rest is the dep.01 and dep.03 rules applied once more:** a static binary on a pinned base, a numeric user, probes on the health port, requests and limits.

### 5.2 The image

```dockerfile
# Build from the repository root. The agent worker (go/cmd/agent) runs the
# AgentRun workflow on the durable worker runtime and consumes only the queue
# configured by its Helm release.
FROM golang:1.25-alpine@sha256:1ae0735f00daffa3aaf1363a5184c0d2dc55c78e3db4ec70241cdac97bf84b59 AS build
WORKDIR /src
# Module files first, so the download layer is cached across source edits.
COPY go/go.mod go/go.sum* go/
RUN cd go && go mod download
# Only go/: no .env, no artifacts, no other languages.
COPY go/ go/
RUN cd go && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/agent ./cmd/agent

FROM alpine:3.22@sha256:4bcff63911fcb4448bd4fdacec207030997caf25e9bea4045fa6c8c44de311d1
# A named user for tools that read /etc/passwd; the numeric USER is what
# runAsNonRoot checks.
RUN addgroup -S -g 10001 agent && adduser -S -D -H -u 10001 -G agent agent
WORKDIR /app
COPY --from=build --chown=10001:10001 /out/agent /usr/local/bin/agent
USER 10001:10001
EXPOSE 9464
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["/bin/sh", "-c", "wget -q -O /dev/null http://127.0.0.1:9464/healthz"]
ENTRYPOINT ["/usr/local/bin/agent"]
```

The health check goes through `/bin/sh -c`, which alpine has; the exec-form `["wget", ...]` of `dep.01` would avoid the shell entirely. Unlike the gateway image, this one carries no OCI labels; adding the `ARG VERSION`, `ARG REVISION`, and `LABEL` block of `dep.01` is a straightforward improvement.

Full solution: [`agent.Dockerfile`](../../../../course/solutions/entry/deploy/docker/agent.Dockerfile).

### 5.3 The Deployment

```yaml
spec:
  replicas: {{ .Values.replicaCount }}          # the schema allows exactly 1
  selector:
    matchLabels:
      app.kubernetes.io/name: forge-agent
  template:
    spec:
      securityContext:
        runAsNonRoot: {{ .Values.securityContext.runAsNonRoot }}
        runAsUser: {{ .Values.securityContext.runAsUser }}
        runAsGroup: {{ .Values.securityContext.runAsGroup }}
      containers:
        - name: agent
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          # The queue is fixed in the template: this release takes agent work only.
          args: ["--queue", "agent", "--durable-address", "{{ .Values.durableAddress }}"]
          env:
            - name: PROVIDER_BASE_URL
              value: {{ .Values.provider.baseUrl | quote }}
            - name: PROVIDER_MODEL
              value: {{ .Values.provider.model | quote }}
            - name: PROVIDER_API_KEY            # wins over [agent].api_key_env
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.provider.apiKey.name }}   # agent-provider
                  key: {{ .Values.provider.apiKey.key }}     # api-key
          ports:
            - name: health
              containerPort: 9464
          livenessProbe:
            httpGet: { path: {{ .Values.probes.liveness.path }}, port: health }    # /healthz
          readinessProbe:
            httpGet: { path: {{ .Values.probes.readiness.path }}, port: health }   # /readyz: 503 while draining
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          securityContext:
            allowPrivilegeEscalation: {{ .Values.securityContext.allowPrivilegeEscalation }}
            readOnlyRootFilesystem: {{ .Values.securityContext.readOnlyRootFilesystem }}
          volumeMounts:
            - name: artifacts
              mountPath: {{ .Values.artifacts.mountPath }}   # read-write: agent-runs journal
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: artifacts
          hostPath: { path: {{ .Values.artifacts.hostPath }}, type: Directory }
        - name: tmp
          emptyDir: {}
```

The worker needs no `TL_ARTIFACTS` variable: its configuration default for `[paths].artifacts` is already `/artifacts`, and the default health port is 9464, so the template sets only what differs per deployment. The image line has no digest suffix; the gateway and engine charts' `{{ with .Values.image.digest }}@{{ . }}{{ end }}` would let the agent be pinned by digest too.

Full solution: [`deployment.yaml`](../../../../course/solutions/entry/deploy/helm/forge-agent/templates/deployment.yaml), [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-agent/values.yaml).

### 5.4 Two values worth a second look

The solution's chart has two settings that the principles of earlier chapters argue against; reading them critically is part of understanding the design.

- **`pdb.maxUnavailable: 0` with one replica.** By the arithmetic of `dep.03` (section 2.2), $a = \min(0, 1) = 0$: no voluntary eviction is ever allowed, so `kubectl drain` of the node hangs. The intent (never interrupt a run) is better served by the durable journal, which already makes an interrupted run resume. `maxUnavailable: 1`, as in the worker chart, keeps drains finishing.
- **No `terminationGracePeriodSeconds`.** The agent drains for up to 60 s on SIGTERM (`go/cmd/agent`), but the pod default is 30 s, so a long run can be killed mid-step during a rollout. As with the worker chart of `dep.06`, the grace period should exceed the drain budget (for example 75 s). The journal limits the damage to repeating the step that was in flight.

The selector also hard-codes `app.kubernetes.io/name: forge-agent` instead of `{{ .Release.Name }}`, which is fine for one release but would collide if the chart were installed twice.

### 5.5 Why this solution

- **A dedicated release over adding `agent` to the worker's queues.** Separate resource limits, separate scaling, and a runaway run (`ops.10`) that cannot exhaust the training worker's memory.
- **Provider settings in the environment over a ConfigMap.** Three values do not need a file; the environment also lets `PROVIDER_API_KEY` come straight from a Secret.
- **alpine over the Python base of the worker.** The agent runs no Python activities, so it does not need the worker's heavy runtime.
- **Tradeoff accepted:** `hostPath` run state ties the agent to one node, as everywhere else on kind.

### 5.6 Explain it back

1. Why does the chart pass `--queue agent` even though `agent` is already the worker's default queue?
   <details><summary>Answer</summary>The queue decides which work the pod receives. Making it explicit in the template means a change to the program's default, or a <code>--config</code> file with another <code>task_queue</code>, cannot silently move the agent worker onto another queue (flags win over the file).</details>
2. A provider key appears in `helm get values`. What went wrong, and what prevents it?
   <details><summary>Answer</summary>Someone put the key into values (for example as an env entry with a literal <code>value</code>). The contract schema rejects a secret-looking name without <code>valueFrom</code> and accepts <code>provider.apiKey</code> only as <code>{name, key}</code>, so the key can only arrive through <code>secretKeyRef</code>.</details>
3. Why must `/artifacts` be writable for the agent but not for the engine?
   <details><summary>Answer</summary>The agent writes its durable run journal under <code>/artifacts/agent-runs</code>; that journal is what lets a retried activity replay completed steps. The engine only reads model files.</details>
4. With the solution's PDB, what happens when an operator drains the agent's node?
   <details><summary>Answer</summary><code>maxUnavailable: 0</code> on one replica allows zero evictions, so the drain waits forever. Changing it to 1 lets the pod be evicted; the run resumes from its journal on the replacement pod.</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Copying a developer `.env` into the image | the build stage copies only `go/`; the final stage only the binary (5.2) |
| Starting the default queue instead of `agent` | `--queue agent` fixed in the template's `args` (5.3) |
| Omitting resource limits | `resources.requests` and `limits` in values, required by the contract schema (5.3) |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.06` | Reuses the worker chart conventions for probes, resources, and secrets. |
| Back | `ag.05` | Packages the durable agent worker and its queue configuration. |
| Forward | Pass 10 agent platform | Runs the agent worker with the durable service and the system's gateway. |
| Forward | `ops.10` | Injects a runaway agent and checks its budget and resume behavior. |

## Going further

Production systems isolate untrusted tool execution, use workload identity for secrets, and persist artifacts in a multi-node store. Those changes need explicit threat models and storage recovery tests.
