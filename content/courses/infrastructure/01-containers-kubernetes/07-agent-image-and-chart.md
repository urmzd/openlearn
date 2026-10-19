# The agent image and chart

[Durable agent runs](../../ai-platform-engineering/13-agent-sdk/05-durable-agent-runs.md) in *AI Platform Engineering* can resume an interrupted agent run, but only if some worker process is packaged and scheduled to poll the `agent` queue. This section provides that process: an image for the agent worker [`go/cmd/agent`](../../../../course/solutions/entry/go/cmd/agent/main.go) and a Helm chart, `deploy/helm/forge-agent/`, that runs it next to the durable service and the gateway. A dedicated chart lets the agent worker scale and restart independently of the gateway and engine services.

The work is mostly the rules you already know, applied once more. The chart follows the conventions of the worker chart from [Durable and worker images and charts](06-durable-and-worker-charts.md) for probes, resources, and secrets, and the image follows [Dockerfiles for the gateway and the engine](01-dockerfiles-for-gateway-and-engine.md). The contract is [`agent.values.schema.json`](../../../../course/contracts/helm/agent.values.schema.json). The section is also the first place where the solution is worth reading critically: two of its values contradict principles from earlier sections, and you will see why.

> **Learning objectives**
> - Explain why the agent worker is its own failure domain with its own image and Deployment.
> - Trace how the worker resolves its settings and where its durable run journal lives.
> - Inject provider credentials only through Secret references enforced by a values schema.
> - Critique a chart's disruption budget and grace period against the arithmetic of earlier sections.

## What the chart must configure

The image is a pinned multi-stage build that copies only the built worker and its runtime files, sets a numeric non-root uid, and runs a static Go binary on a digest-pinned alpine. The chart gives the container CPU and memory requests and limits, and readiness and liveness probes on the worker's health endpoint, so failure is visible to the scheduler. Its configuration identifies the durable service, the queue, the provider URL, the model, the artifacts mount, and an optional OTel endpoint. Any API key comes from a Secret key reference.

Four settings carry most of the meaning:

| Setting | Purpose |
|---|---|
| `durableAddress` | gRPC address of the durable workflow service |
| `provider.baseUrl` | the system's gateway or an explicitly configured frontier endpoint |
| `provider.apiKey` | a Secret reference, never a literal |
| `replicaCount` | a bounded worker count for this local chart (the schema allows exactly 1) |

## Tracing the configuration by hand

The chart points `durableAddress` at `forge-durable:7233`, the Service from the durable chart. It points the provider at `http://forge-gateway:8080` with model `smollm2-135m-instruct`, mounts `/artifacts` read-write for run state, and reads `PROVIDER_API_KEY` from Secret `agent-provider`, key `api-key`.

The worker resolves its settings in a fixed order: defaults first (queue `agent`, health on `:9464`, artifacts at `/artifacts`), then a `--config` file if given, then flags (`--queue agent --durable-address forge-durable:7233`), then the `PROVIDER_*` environment. Its run journal lands in `/artifacts/agent-runs`, so a retried activity replays the completed model and tool steps instead of repeating them. A readiness failure removes the Pod from service, while its durable workflow lease can expire and be reclaimed by another worker.

## What a correct image and chart must do

The image must have a pinned base, a numeric non-root runtime user, and only the built binary in its final stage: no root execution, no mutable base tags, and no broad source copies, so a developer's `.env` stays out. These are the image rules every image in this chapter follows.

The chart must render under the contract schema, and its `values.schema.json` must equal the published contract, giving it the same shape as every other chart. It must select the `agent` queue, read `PROVIDER_API_KEY` with `secretKeyRef` (the values file holds only the Secret name and key), and carry probes, resource bounds, and a PodDisruptionBudget. Together these make the worker take agent work only, and ensure no key leaks. A chart must never carry provider credentials in values files or image layers.

## Building it step by step

### Reasoning from first principles

1. **The agent worker is a separate failure domain.** Agent runs are long and call out to a model and tools; sharing a Pod with other workers would let a runaway run starve them. So it gets its own image and Deployment, consuming one queue.
2. **The queue is the routing.** The durable server hands `AgentRun` tasks only to pollers of queue `agent`, so the chart must pass `--queue agent` explicitly rather than rely on a default that could change.
3. **The key never touches Helm.** The provider key enters as `PROVIDER_API_KEY` through `secretKeyRef`; values hold only `{name, key}`, which the contract schema enforces.
4. **Run state must survive the Pod.** The durable agent journal under `/artifacts/agent-runs` is what makes a retried run resume, so `/artifacts` is mounted read-write, and the root filesystem stays read-only with an `emptyDir` at `/tmp`.
5. **The rest is the image and chart rules applied once more:** a static binary on a pinned base, a numeric user, probes on the health port, requests and limits.

### The image

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

The health check goes through `/bin/sh -c`, which alpine has; the exec-form `["wget", ...]` used by the gateway image would avoid the shell entirely. Unlike the gateway image, this one carries no OCI labels, so adding the `ARG VERSION`, `ARG REVISION`, and `LABEL` block from [Dockerfiles for the gateway and the engine](01-dockerfiles-for-gateway-and-engine.md) is a straightforward improvement. The full solution is [`agent.Dockerfile`](../../../../course/solutions/entry/deploy/docker/agent.Dockerfile).

### The Deployment

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

The worker needs no `TL_ARTIFACTS` variable. Its configuration default for `[paths].artifacts` is already `/artifacts`, and the default health port is 9464, so the template sets only what differs per deployment. The image line has no digest suffix; the gateway and engine charts' `{{ with .Values.image.digest }}@{{ . }}{{ end }}` would let the agent be pinned by digest too.

The full solution is [`deployment.yaml`](../../../../course/solutions/entry/deploy/helm/forge-agent/templates/deployment.yaml) and [`values.yaml`](../../../../course/solutions/entry/deploy/helm/forge-agent/values.yaml).

### Two values worth a second look

The solution's chart has two settings that the principles of earlier sections argue against. Reading them critically is part of understanding the design.

> **Warning:** `pdb.maxUnavailable: 0` with one replica. By the disruption-budget arithmetic of [Helm charts and the observability stack](03-helm-charts-and-observability-stack.md), $a = \min(0, 1) = 0$: no voluntary eviction is ever allowed, so `kubectl drain` of the node hangs. The intent (never interrupt a run) is better served by the durable journal, which already makes an interrupted run resume. `maxUnavailable: 1`, as in the worker chart, keeps drains finishing.

> **Warning:** No `terminationGracePeriodSeconds`. The agent drains for up to 60 s on SIGTERM (`go/cmd/agent`), but the Pod default is 30 s, so a long run can be killed mid-step during a rollout. As with the worker chart of [Durable and worker images and charts](06-durable-and-worker-charts.md), the grace period should exceed the drain budget, for example 75 s. The journal limits the damage to repeating the step that was in flight.

The selector also hard-codes `app.kubernetes.io/name: forge-agent` instead of `{{ .Release.Name }}`. That is fine for one release but would collide if the chart were installed twice.

### Why this design

- **A dedicated release over adding `agent` to the worker's queues.** Separate resource limits, separate scaling, and a runaway run (the subject of the [runaway agent drill](../../systems/05-incident-response-and-chaos/10-runaway-agent.md)) that cannot exhaust the training worker's memory.
- **Provider settings in the environment over a ConfigMap.** Three values do not need a file, and the environment also lets `PROVIDER_API_KEY` come straight from a Secret.
- **alpine over the worker's Python base.** The agent runs no Python activities, so it does not need the worker's heavy runtime.
- **Tradeoff accepted:** `hostPath` run state ties the agent to one node, as everywhere else on kind.

## What can go wrong

Beyond the two values above, three mistakes are easy to make with a chart like this:

| Pitfall | How the solution avoids it |
|---|---|
| Copying a developer `.env` into the image | the build stage copies only `go/`; the final stage only the binary |
| Starting on the default queue instead of `agent` | `--queue agent` fixed in the template's `args` |
| Omitting resource limits | `resources.requests` and `limits` in values, required by the contract schema |

## Where this leads

This chart reuses the worker chart's conventions for probes, resources, and secrets from [Durable and worker images and charts](06-durable-and-worker-charts.md), and packages the durable agent worker and its queue configuration from [Durable agent runs](../../ai-platform-engineering/13-agent-sdk/05-durable-agent-runs.md). The agent platform runs this worker alongside the durable service and the system's gateway, and the [runaway agent drill](../../systems/05-incident-response-and-chaos/10-runaway-agent.md) in *Systems and Architecture* injects a runaway agent and checks its budget and resume behavior.

Production systems go further: they isolate untrusted tool execution in separate sandbox workloads, use workload identity for secrets, and persist artifacts in a multi-node store. Each of those changes needs an explicit threat model and storage recovery tests.

## Key terms

- **failure domain**: a unit (here, a Deployment) whose failure or overload does not take others down with it.
- **task queue**: the durable server's routing key; only pollers of queue `agent` receive `AgentRun` tasks.
- **run journal**: the durable record under `/artifacts/agent-runs` that lets a retried activity replay completed steps.
- **settings resolution order**: defaults, then a `--config` file, then flags, then the `PROVIDER_*` environment.

## Summary

- The image runs the agent worker as a numeric non-root user, built static from Go onto a digest-pinned alpine.
- The chart consumes the durable address and fixes the `agent` queue in its args.
- Provider keys are injected only through Kubernetes Secret references, which the contract schema enforces.
- Resource limits and probes make failure visible to the scheduler, and `/artifacts` is writable so the run journal survives the Pod.
- The solution's `maxUnavailable: 0` and default 30 s grace period are worth changing to 1 and about 75 s.

## Review questions

1. Why does the chart pass `--queue agent` even though `agent` is already the worker's default queue?
   <details><summary>Answer</summary>The queue decides which work the Pod receives. Making it explicit in the template means a change to the program's default, or a <code>--config</code> file with another <code>task_queue</code>, cannot silently move the agent worker onto another queue (flags win over the file).</details>
2. A provider key appears in `helm get values`. What went wrong, and what prevents it?
   <details><summary>Answer</summary>Someone put the key into values (for example as an env entry with a literal <code>value</code>). The contract schema rejects a secret-looking name without <code>valueFrom</code> and accepts <code>provider.apiKey</code> only as <code>{name, key}</code>, so the key can only arrive through <code>secretKeyRef</code>.</details>
3. Why must `/artifacts` be writable for the agent but not for the engine?
   <details><summary>Answer</summary>The agent writes its durable run journal under <code>/artifacts/agent-runs</code>; that journal is what lets a retried activity replay completed steps. The engine only reads model files.</details>
4. With the solution's PDB, what happens when an operator drains the agent's node?
   <details><summary>Answer</summary><code>maxUnavailable: 0</code> on one replica allows zero evictions, so the drain waits forever. Changing it to 1 lets the Pod be evicted; the run resumes from its journal on the replacement Pod.</details>
