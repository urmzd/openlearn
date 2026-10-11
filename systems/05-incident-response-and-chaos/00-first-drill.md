<!-- ol:module ops.00 -->
# First drill: the engine crashloops

## Overview

| | |
|---|---|
| **Module** | `ops.00` · drill · ops, docs · Pass 1 · 1 to 2 h |
| **You build** | `docs/runbooks/engine-crashloop.md` (Symptoms, Diagnosis, Mitigation) and `docs/postmortems/<date>-engine-crashloop.md`, and you restore the service |
| **Contract** | the drill spec `course/drills/engine-crashloop/drill.toml` (what is injected and what `ol drill end` checks: section 4) |
| **Tests** | graded by `ol drill end`; no course tests to read (section 4 lists every check) |
| **Needs** | [`dep.00`](../../infrastructure/01-containers-kubernetes/00-tracer-deploy.md) (your engine and gateway on kind); you triage what [`L10.0`](../../ml/08-tinyllm/p10-serving/00-your-first-endpoint.md) and [`gw.00`](../../ai-platform-engineering/12-gateway/00-streaming-proxy.md) built, with the `kubectl` of [`lang.07`](../../software-craftsmanship/12-language-and-tool-primers/07-containers-and-kubernetes.md) |
| **Used by** | no call site: a drill. Its runbook is the first of the runbooks craft.10 requires per alert |
| **Milestone** | [MS-P1](../../paths/course-p01-tracer/milestone.md) |
| **Optional depth** | *Site Reliability Engineering*, ch. 12 Effective Troubleshooting, ch. 14 Managing Incidents, ch. 15 Postmortem Culture ([free](https://sre.google/sre-book/table-of-contents/)); Kubernetes docs, [Debug Running Pods](https://kubernetes.io/docs/tasks/debug/debug-application/debug-running-pod/) (free) |

## Key Takeaways

- A crashlooping pod tells you why it died in three places: `Last State` and the exit code in `kubectl describe`, and the log of the previous run in `kubectl logs --previous`.
- Mitigate first, then find the root cause: `kubectl rollout undo` restores service in seconds; the durable fix goes through the chart.
- Whether users see an outage depends on your rollout strategy and readiness probe, not only on the bug.
- A runbook is a how-to guide written for the next person at 3 a.m.: symptoms they will see, commands that narrow the cause, steps that restore service.
- A blameless postmortem has fixed sections, and `ol drill end` checks that yours does.

## How to work this chapter

```bash
ol drill list                      # engine-crashloop is ops.00
ol drill start engine-crashloop    # injects the fault, prints only the page
# ... triage with kubectl, restore service, write the runbook and the postmortem ...
ol drill end                       # grades: resolved, runbook, postmortem, time limit
ol drill reset                     # replays the undo journal; always run it after `end`
```

---

## 1. Why now

Your tracer runs on kind: one engine pod behind one gateway pod, deployed by your Helm charts (`dep.00`). There is no Prometheus and no alert yet (they arrive in Pass 7), and the engine has a single replica. So when a rollout goes wrong, nothing pages you except a user, and nothing tells you why except `kubectl`. This drill makes that happen on purpose: a change to the engine Deployment points `--model-dir` at a directory that does not exist. The engine exits on startup, Kubernetes restarts it again and again, and depending on how your chart rolls out, the gateway's `/readyz` turns 503 and every completion fails. You will find the cause by hand, restore the service, and write the runbook and the postmortem that make the second occurrence a five-minute fix.

## 2. Principles

**The objects involved.** A **Deployment** declares "run N copies of this pod template". It owns **ReplicaSets**, one per version of the template; each ReplicaSet creates **Pods**. Changing anything in the template (an image, an arg, an env var) creates a new ReplicaSet: that is a **rollout**, and `kubectl rollout history` lists each one as a revision. A **Service** sends traffic only to pods that are **Ready**.

**Restart policy and back-off.** A Deployment's pods have `restartPolicy: Always`. When the container exits, the kubelet restarts it in place, but waits longer each time so a broken container does not spin the node. The pod's status shows `CrashLoopBackOff` while it waits.

| Symbol | Meaning | Unit |
|---|---|---|
| $k$ | the restart number, $k = 1, 2, \dots$ | count |
| $d_k$ | the wait before restart $k$ | seconds |
| $r$ | how long the container runs before it dies | seconds |
| $s_k$ | when attempt $k$ starts; $s_0 = 0$ is the first start | seconds after the fault |
| $t_{inj}, t_{det}, t_{mit}$ | when the fault was injected, noticed, and mitigated | clock time |

With the default kubelet settings, the wait doubles from 10 s and is capped at 5 minutes, and it resets once a container has run for 10 minutes:

$$d_k = \min(10 \cdot 2^{k-1},\ 300), \qquad s_k = s_{k-1} + r + d_k .$$

So the `RESTARTS` column grows fast at first and then about once every 5 minutes. A restart count that is still climbing slowly does not mean the problem is going away.

**Exit codes say who stopped the process.** A code below 128 is the program's own choice: `1` usually means it detected a problem and quit, and its last log lines say which. A code of `128 + n` means it died from signal $n$: `137` is SIGKILL (9), which in a pod almost always means the kernel killed it for exceeding its memory limit (`Reason: OOMKilled`); `143` is SIGTERM (15), a normal shutdown.

**Logs of the dead.** `kubectl logs` shows the current attempt, which in a crashloop is often empty or not started. `kubectl logs --previous` shows the attempt that just died, which is where the reason is.

**Readiness decides whether users notice.** The default Deployment strategy is `RollingUpdate` with `maxSurge: 25%` and `maxUnavailable: 25%`, rounded up and down. With one replica that is one extra pod and zero unavailable: the new pod is created first and the old one is removed only when the new one is Ready. So:

| Your engine chart has | During a bad rollout | Users see |
|---|---|---|
| a readiness probe on `/healthz` (health port) | the new pod never becomes Ready; the old pod keeps serving; the rollout stalls | nothing yet, but the next restart of the old pod is an outage |
| no readiness probe | a container counts as Ready the moment it starts, so the old pod is removed during the new pod's first second alive | 503 `no_capacity` on every completion |
| `strategy: Recreate` | the old pod is removed first, always | 503 `no_capacity` on every completion |

Either way the Deployment is broken until you act. The drill grades the service you restore, not which row you were in.

**Mitigate, then fix, then learn.** During an incident the first goal is to stop user impact, even if you do not yet understand the cause. Rolling back the change that started it is the fastest safe action, because the previous version is known to work. The root cause and the durable fix come after, and the postmortem comes last.

**The cluster drifts from the chart.** Helm renders your chart into objects and applies them. A `kubectl patch` or `kubectl edit` changes the live object without changing the chart, so the cluster no longer matches what you would deploy. The durable fix is always in the chart, applied with `helm upgrade`.

**Runbooks and postmortems are different documents.** A **runbook** is a how-to guide for one symptom: what you see, how to narrow the cause, how to restore service. It is written before the next incident and kept current. A **postmortem** is the record of one incident: what happened, when, why, and what will change. It is **blameless**: it names systems and processes, never a person at fault, so people report what really happened. Its sections are fixed: Summary, Impact, Timeline, Root cause, Detection, Resolution, Action items.

**Time to detect, time to mitigate.** $\text{TTD} = t_{det} - t_{inj}$ and $\text{TTM} = t_{mit} - t_{inj}$. Pass 1 has no alerts, so TTD is however long it takes a person to notice. From Pass 7, burn-rate alerts (`obs.03`) cut it to minutes, and `ol drill end` measures both from Prometheus.

## 3. Worked example by hand

A different incident with the same shape, so you can check every number: someone lowers the engine's memory limit to 8 MiB. The engine is killed while loading its model, after running $r = 2$ s each time. Your chart has no readiness probe, so users see the outage at once. The fault lands at 10:00:00.

**The restart schedule.** Apply $d_k = \min(10 \cdot 2^{k-1}, 300)$ and $s_k = s_{k-1} + r + d_k$:

| $k$ | $d_k$ (s) | $s_k$ (s) | how $s_k$ was computed | clock time |
|---|---|---|---|---|
| 0 | | 0 | first start | 10:00:00 |
| 1 | 10 | 12 | $0 + 2 + 10$ | 10:00:12 |
| 2 | 20 | 34 | $12 + 2 + 20$ | 10:00:34 |
| 3 | 40 | 76 | $34 + 2 + 40$ | 10:01:16 |
| 4 | 80 | 158 | $76 + 2 + 80$ | 10:02:38 |
| 5 | 160 | 320 | $158 + 2 + 160$ | 10:05:20 |
| 6 | 300 | 622 | $320 + 2 + \min(320, 300)$ | 10:10:22 |

At 10:03:00 ($t = 180$ s) `kubectl get pods` shows `RESTARTS 4`: attempts 1 to 4 started before 180 s, attempt 5 starts at 320 s.

**The timeline you would write.**

| Time | Event | Evidence |
|---|---|---|
| 10:00:00 | memory limit lowered to 8 MiB by a manual `kubectl patch` | `kubectl rollout history`: new revision |
| 10:00:02 | first kill | `Last State: Terminated, Reason: OOMKilled, Exit Code: 137` |
| 10:01:10 | a user reports 503 `no_capacity` | gateway response |
| 10:03:00 | triage: engine pod `CrashLoopBackOff`, `RESTARTS 4`; exit 137 points at memory, not at the program | `kubectl get pods`, `kubectl describe pod` |
| 10:04:40 | `kubectl rollout undo deploy/<system>-engine` | |
| 10:05:00 | new pod Ready; completions stream again | `kubectl rollout status`, `curl` |

**The two numbers.** $t_{inj}$ = 10:00:00, $t_{det}$ = 10:01:10, $t_{mit}$ = 10:05:00, so $\text{TTD} = 70$ s and $\text{TTM} = 300$ s. Note what dominated TTM: not the fix (20 s) but the time between the report and reading `Exit Code: 137`. That is what a runbook removes.

**What differs in your drill.** Your fault is not memory: expect `Exit Code: 1` and a log line about the missing model directory. The method is the same.

## 4. Inject, detect, mitigate, verify

**Before you start.** `dep.00` passes, and your `system.toml` has a `[deploy]` table:

```toml
[deploy]
kube_context = "kind-<system>"           # ol refuses any context that is not kind- or k3d-
namespace    = "<system>"
gateway_url  = "http://127.0.0.1:30080"  # the gateway NodePort
traces       = "http://127.0.0.1:30686"  # Jaeger query
services     = { engine = "deploy/<system>-engine", gateway = "deploy/<system>-gateway" }
```

Export your gateway key in the variable `[endpoints].api_key_env` names (default `TL_API_KEY`); the second resolve check calls the gateway with it.

**Inject.** `ol drill start engine-crashloop` checks the safety gate (current context equals `kube_context`, which starts with `kind-` or `k3d-`; the namespace exists), then patches the first container of `services.engine`: its args become the tracer engine's flags (`spec/cli-roles.md`) with `--model-dir /missing`, at the Kubernetes ports `8000` and `9464`. The undo (your original args) goes to `.ol/drills/<run>/journal.jsonl`. You see only the page.

**Detect.** There is no alert in Pass 1. Work outside in: which workload is unhealthy, why its last container exited, what its previous log says, and what changed. These commands answer the four questions in that order; your runbook's Diagnosis section says what each output means, in your own words:

```bash
kubectl get deploy,pods -n <system> -o wide                       # which workload: the engine pod is not 1/1 Running
kubectl describe pod -n <system> <engine-pod>                     # why it exited: Last State, Exit Code, Restart Count, Events
kubectl logs -n <system> deploy/<system>-engine --previous        # what the crashed container said before it died
kubectl rollout history deploy/<system>-engine -n <system>        # what changed: a new revision
kubectl get deploy <system>-engine -n <system> -o jsonpath='{.spec.template.spec.containers[0].args}'
```

**Mitigate.** Restore the last revision that worked, wait for it, then make the chart the truth again:

```bash
kubectl rollout undo deploy/<system>-engine -n <system>
kubectl rollout status deploy/<system>-engine -n <system> --timeout=120s
helm upgrade --install <system>-engine deploy/helm/<system>-engine -n <system>
```

**Write the runbook.** `docs/runbooks/engine-crashloop.md`, with at least these headings (any level, any extra sections you like):

```markdown
# Runbook: engine crashloop

## Symptoms
What the person paged will see: the client error, the gateway's /readyz,
the pod status and restart count.

## Diagnosis
Numbered steps, each a command and what its output means, from "which
workload" to "what changed".

## Mitigation
Numbered steps that restore service, how to confirm it, and the durable fix.
```

**Write the postmortem.** `docs/postmortems/<YYYY-MM-DD>-engine-crashloop.md` (the date in UTC), with the headings Summary, Impact, Timeline, Root cause, Detection, Resolution, Action items. The timeline is a table like the one in section 3; the action items say how the next rollout could not cause this (a readiness probe, `maxUnavailable: 0`, changes only through `helm upgrade`).

**Verify.** `ol drill end` grades, then prints the seed and what was injected:

| Check | Passes when | Why it matters |
|---|---|---|
| detected | always, with "graded manually": this drill has no `[detect]` block | Pass 1 has no Prometheus; you are the detector |
| resolve 1 | the engine rollout (`services.engine`) has converged within 60 s: every replica updated and available | Kubernetes, not you, now keeps a Ready engine running |
| resolve 2 | `ol conform openapi:v0:gateway:smoke` against `[deploy].gateway_url` passes: the v0 smoke cases (completion schema, SSE framing, and 401 without a key; the health case is pending unless `[deploy].gateway_health_url` is set) through your gateway NodePort | completions flow again, so a Ready engine is behind the gateway |
| postmortem | the newest `docs/postmortems/*-engine-crashloop.md` has all seven headings | the incident is recorded in the fixed shape |
| runbook | `docs/runbooks/engine-crashloop.md` has Symptoms, Diagnosis, Mitigation headings (the drill's `[[doc]]` table) | the next person needs minutes, not your 45 |
| time limit | `end` within 45 minutes of `start` | keeps the drill honest |

`ol drill end` grades the runbook and the postmortem the same way: a missing file or heading fails the drill. It records the verdict under `ops.00`, which MS-P1 requires. Then run `ol drill reset`: it replays the journal in reverse (after your rollback the replayed undo changes nothing).

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | Reading `kubectl logs` without `--previous` | an empty log, or "container is waiting to start" | no check; the runbook's Diagnosis step 3 |
| 2 | Deleting the crashing pod to "restart it" | the ReplicaSet makes an identical pod that crashes the same way | resolve 1 and resolve 2 stay red |
| 3 | Scaling the engine to zero to stop the restarts | the restarts stop and so does the service | resolve 2 stays red |
| 4 | Fixing the args with `kubectl edit` and never touching the chart | it works until the next `helm upgrade` reapplies whatever the chart says | the postmortem's action items; review |
| 5 | Running `ol drill reset` before `ol drill end` | the run is marked reset and never graded | ol drill end refuses: no active drill |
| 6 | A postmortem that names who broke it, or skips a heading | the team stops reporting honestly; `end` fails on the missing heading | the postmortem check |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.00` | the Deployment, Service, and NodePort the drill injects into and checks through |
| Back | `L10.0` | the engine whose startup fails on a missing model directory |
| Back | `gw.00` | the gateway whose `/readyz` and 503 `no_capacity` are the symptom |
| Back | `lang.07` | `kubectl get`, `describe`, `logs`, and Helm |
| Forward | `obs.03` | burn-rate alerts that page you before a user does |
| Forward | `ops.01` | the first drill with a `[detect]` block, graded on TTD from Prometheus |
| Forward | `dep.03` | chart policy: readiness probes and disruption budgets, the action items here |
| Forward | `craft.10` | a runbook for every required alert, in the format you start here |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| manual `rollout undo` | progressive delivery (Argo Rollouts, Flagger) | automatic rollback when the new version's health or SLO checks fail | Argo Rollouts analysis templates |
| reading exit codes by hand | `startupProbe`, Kubernetes events, `kubectl debug` | separates slow starts from crashes; ephemeral debug containers in a broken pod | Kubernetes docs, Configure Liveness, Readiness and Startup Probes |
| the fixed 10 s to 5 min back-off | KEP-4603, tunable CrashLoopBackOff | faster restarts for quick-failing containers | `kubernetes/enhancements`, keps/sig-node/4603 |
| `docs/runbooks/*.md` | incident tooling (PagerDuty, incident.io, Rootly) | runbooks linked from the alert, incident roles, automated timelines | Google SRE book ch. 14 (free) |
| your postmortem | blameless postmortem programs | review meetings, tracked action items, shared learning | Google SRE book ch. 15 (free); Etsy, *Debriefing Facilitation Guide* (free) |

## Company Relevance

| Company | Practice | Why it matters |
|---|---|---|
| Google | SRE incident management and blameless postmortems | the origin of most of this chapter's vocabulary |
| Netflix | chaos engineering in production | drills are how a team learns its system before a real outage teaches it |
| Any on-call team | runbooks linked from every alert | the difference between a 5-minute and a 45-minute mitigation |
