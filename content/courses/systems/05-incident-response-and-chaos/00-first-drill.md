# First drill: the engine crashloops

## Overview

| | |
|---|---|
| **Module** | `ops.00` · drill · ops, docs · Pass 1 · 1 to 2 h |
| **Solution** | a worked incident (section 5): diagnosis and mitigation, the model runbook [`docs/runbooks/engine-crashloop.md`](../../../../course/solutions/docs/runbooks/engine-crashloop.md) (Symptoms, Diagnosis, Mitigation), and the model postmortem [`docs/postmortems/2026-10-08-engine-crashloop.md`](../../../../course/solutions/docs/postmortems/2026-10-08-engine-crashloop.md) |
| **Contract** | the fault and the resolve conditions in section 4 |
| **Needs** | [`dep.00`](../../infrastructure/01-containers-kubernetes/00-tracer-deploy.md) (the engine and gateway on kind); the incident is in what [`L10.0`](../../../courses/build-an-llm/p10-serving/00-your-first-endpoint.md) and [`gw.00`](../../ai-platform-engineering/12-gateway/00-streaming-proxy.md) built, with the `kubectl` of [`lang.07`](../../programming-languages/02-language-and-tool-primers/07-containers-and-kubernetes.md) |
| **Used by** | no call site: a drill. Its runbook is the first of the runbooks craft.10 requires per alert; part of [the Pass 1 tracer](../../../courses/build-an-llm/paths/course-p01-tracer/milestone.md) |
| **Optional depth** | *Site Reliability Engineering*, ch. 12 Effective Troubleshooting, ch. 14 Managing Incidents, ch. 15 Postmortem Culture ([free](https://sre.google/sre-book/table-of-contents/)); Kubernetes docs, [Debug Running Pods](https://kubernetes.io/docs/tasks/debug/debug-application/debug-running-pod/) (free) |

## Key Takeaways

- A crashlooping pod says why it died in three places: `Last State` and the exit code in `kubectl describe`, and the log of the previous run in `kubectl logs --previous`.
- Mitigate first, then find the root cause: `kubectl rollout undo` restores service in seconds; the durable fix goes through the chart.
- Whether users see an outage depends on the rollout strategy and readiness probe, not only on the bug.
- A runbook is a how-to guide written for the next person at 3 a.m.: symptoms they will see, commands that narrow the cause, steps that restore service.
- A blameless postmortem has fixed sections, and its root cause names system properties, not people.

## How to read this chapter

This chapter is a worked incident. Sections 1 to 3 build the Kubernetes and incident-response ideas from first principles and work a similar crashloop by hand. Section 4 states what is injected and what "resolved" means. Section 5 walks through the incident from page to postmortem, with the model runbook and postmortem, and explains why each step is the right one. Before reading each diagnosis step, predict what its output will show.

---

## 1. Why now

The tracer runs on kind: one engine pod behind one gateway pod, deployed by the Helm charts (`dep.00`). There is no Prometheus and no alert yet (they arrive in Pass 7), and the engine has a single replica. So when a rollout goes wrong, nothing pages except a user, and nothing says why except `kubectl`. This drill makes that happen on purpose: a change to the engine Deployment points `--model-dir` at a directory that does not exist. The engine exits on startup, Kubernetes restarts it again and again, and depending on how the chart rolls out, the gateway's `/readyz` turns 503 and every completion fails. The walkthrough finds the cause by hand, restores the service, and writes the runbook and the postmortem that make the second occurrence a five-minute fix.

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

| The engine chart has | During a bad rollout | Users see |
|---|---|---|
| a readiness probe on `/healthz` (health port) | the new pod never becomes Ready; the old pod keeps serving; the rollout stalls | nothing yet, but the next restart of the old pod is an outage |
| no readiness probe | a container counts as Ready the moment it starts, so the old pod is removed during the new pod's first second alive | 503 `no_capacity` on every completion |
| `strategy: Recreate` | the old pod is removed first, always | 503 `no_capacity` on every completion |

Either way the Deployment is broken until someone acts. What matters is the service restored, not which row the chart was in.

**Mitigate, then fix, then learn.** During an incident the first goal is to stop user impact, even if you do not yet understand the cause. Rolling back the change that started it is the fastest safe action, because the previous version is known to work. The root cause and the durable fix come after, and the postmortem comes last.

**The cluster drifts from the chart.** Helm renders the chart into objects and applies them. A `kubectl patch` or `kubectl edit` changes the live object without changing the chart, so the cluster no longer matches what the chart would deploy. The durable fix is always in the chart, applied with `helm upgrade`.

**Runbooks and postmortems are different documents.** A **runbook** is a how-to guide for one symptom: what you see, how to narrow the cause, how to restore service. It is written before the next incident and kept current. A **postmortem** is the record of one incident: what happened, when, why, and what will change. It is **blameless**: it names systems and processes, never a person at fault, so people report what really happened. Its sections are fixed: Summary, Impact, Timeline, Root cause, Detection, Resolution, Action items.

**Time to detect, time to mitigate.** $\text{TTD} = t_{det} - t_{inj}$ and $\text{TTM} = t_{mit} - t_{inj}$. Pass 1 has no alerts, so TTD is however long it takes a person to notice. From Pass 7, burn-rate alerts (`obs.03`) cut it to minutes, and both can be measured from Prometheus.

## 3. Worked example by hand

A different incident with the same shape, with every number checkable: someone lowers the engine's memory limit to 8 MiB. The engine is killed while loading its model, after running $r = 2$ s each time. The chart has no readiness probe, so users see the outage at once. The fault lands at 10:00:00.

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

**The timeline to write.**

| Time | Event | Evidence |
|---|---|---|
| 10:00:00 | memory limit lowered to 8 MiB by a manual `kubectl patch` | `kubectl rollout history`: new revision |
| 10:00:02 | first kill | `Last State: Terminated, Reason: OOMKilled, Exit Code: 137` |
| 10:01:10 | a user reports 503 `no_capacity` | gateway response |
| 10:03:00 | triage: engine pod `CrashLoopBackOff`, `RESTARTS 4`; exit 137 points at memory, not at the program | `kubectl get pods`, `kubectl describe pod` |
| 10:04:40 | `kubectl rollout undo deploy/<system>-engine` | |
| 10:05:00 | new pod Ready; completions stream again | `kubectl rollout status`, `curl` |

**The two numbers.** $t_{inj}$ = 10:00:00, $t_{det}$ = 10:01:10, $t_{mit}$ = 10:05:00, so $\text{TTD} = 70$ s and $\text{TTM} = 300$ s. Note what dominated TTM: not the fix (20 s) but the time between the report and reading `Exit Code: 137`. That is what a runbook removes.

**What differs in this drill.** The fault is not memory: expect `Exit Code: 1` and a log line about the missing model directory. The method is the same.

## 4. The incident: what is injected, what resolved means

**The setup.** The Pass 1 tracer on kind: namespace `<system>`, `deploy/<system>-engine` (one replica) behind `deploy/<system>-gateway`, the gateway on NodePort 30080, Jaeger on 30686.

**The fault.** A change to the first container of the engine Deployment replaces its args with the tracer engine's flags (`spec/cli-roles.md`) pointing at a directory that does not exist, at the Kubernetes ports:

```
--model-dir /missing --port 8000 --health-port 9464
```

Because the pod template changed, this is a rollout: a new ReplicaSet and a new revision in `kubectl rollout history`. The page says only: *the engine has been restarting in a loop since the last rollout; users may get 503 no_capacity*.

**Resolved means**, in this order of importance:

| Condition | How to confirm | Why it matters |
|---|---|---|
| the engine rollout has converged: every replica updated and available | `kubectl rollout status deploy/<system>-engine --timeout=60s` | Kubernetes, not a person, now keeps a Ready engine running |
| completions flow through the gateway NodePort: completion schema, SSE framing, 401 without a key | the `curl` of section 5.3, and one without the key | a Ready engine is behind the gateway, end to end |
| the runbook exists with Symptoms, Diagnosis, Mitigation | `docs/runbooks/engine-crashloop.md` | the next person needs minutes, not 45 |
| the postmortem exists with Summary, Impact, Timeline, Root cause, Detection, Resolution, Action items | `docs/postmortems/<YYYY-MM-DD>-engine-crashloop.md` (UTC date) | the incident is recorded in the fixed shape |
| all of it within 45 minutes | the timeline | a realistic bound for a Pass 1 outage with no alerting |

There is no automated detection: Pass 1 has no Prometheus, so the person on call is the detector.

## 5. The solution, step by step

### 5.1 From first principles

1. **The page names a symptom, not a cause.** "Restarting in a loop" means the container keeps exiting and the kubelet keeps restarting it (section 2, restart policy). The questions are: which workload, why did it exit, what did it say, what changed. Answer them in that order, outside in.
2. **Why it exited is recorded by the kubelet**, in `Last State` and `Exit Code`. The exit code splits the space in two (section 2): below 128 the program chose to stop, so its log names the reason; $128 + n$ means a signal killed it, usually the OOM killer (137).
3. **The reason is in the dead container's log.** The current attempt is waiting in back-off, so `kubectl logs --previous` is the only place the message exists.
4. **What changed is in the rollout history.** A crashloop that starts at a rollout is caused by the rollout until proven otherwise. Comparing the live args with what Helm renders shows whether the change came from the chart or from outside it.
5. **Mitigation must not depend on understanding.** The previous revision is known to work, so `kubectl rollout undo` restores service even before the root cause is certain. Deleting the pod or scaling to zero does not: the ReplicaSet still holds the broken template.
6. **The chart must become the truth again.** A rollback restores the live object, but the cluster now differs from the chart in history if not in content. Re-applying the chart with `helm upgrade` makes them agree.
7. **Learning is written down in two shapes.** A runbook for the symptom (reusable, kept current) and a blameless postmortem for this incident (a record with fixed sections and action items that prevent recurrence).

### 5.2 Diagnosis, annotated

```bash
# 1. Which workload? The gateway pod is 1/1 Running; the engine pod is not.
kubectl get deploy,pods -n <system> -o wide

# 2. Why did it exit? Last State: Terminated, Reason: Error, Exit Code: 1;
#    Events: Back-off restarting failed container. Exit 1 at startup: the
#    program chose to exit, so its own log says why (137 would be OOM, a
#    different runbook).
kubectl describe pod -n <system> <engine-pod>

# 3. What did it say? The crashed run's stderr, not the waiting one:
#    cannot read /missing/config.json: No such file or directory
kubectl logs -n <system> deploy/<system>-engine --previous

# 4. What changed? A new revision; live args that differ from the chart.
kubectl rollout history deploy/<system>-engine -n <system>
kubectl get deploy <system>-engine -n <system> -o jsonpath='{.spec.template.spec.containers[0].args}'
helm get manifest <system>-engine -n <system> | grep -A6 'args:'
```

Applying section 2's readiness table: whether users already see 503s depends on the chart. With a readiness probe and the default `RollingUpdate`, the old pod keeps serving and the rollout stalls; without one, or with `Recreate`, the old pod is gone and every completion fails with `503 no_capacity`. Either way the Deployment is broken until someone acts.

### 5.3 Mitigation, annotated

```bash
# Restore the last revision that worked; wait until Kubernetes says it is done.
kubectl rollout undo deploy/<system>-engine -n <system>
kubectl rollout status deploy/<system>-engine -n <system> --timeout=120s

# Confirm from the user's side: eight data: chunks, then data: [DONE].
curl -sS -N http://127.0.0.1:30080/v1/completions \
  -H "Authorization: Bearer $TL_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model": "tracer", "prompt": "Once", "max_tokens": 8, "stream": true}'

# Make the chart the source of truth again (never fix args with kubectl edit).
helm upgrade --install <system>-engine deploy/helm/<system>-engine -n <system>
```

`rollout status` is the check that the fix took: it returns only when every replica is updated and available. The `curl` is the check that users are served, which is what the incident was about.

### 5.4 The model runbook

The runbook is written for the next person, who has the page and nothing else. Its structure mirrors 5.1: what you will see, then commands that narrow the cause, then steps that restore service. Excerpt:

```markdown
## Symptoms

- Clients get `503` with `"code": "no_capacity"` from `POST /v1/completions`
  through the gateway, before any token.
- `GET /readyz` on the gateway health port answers `503`: the gateway cannot
  reach a healthy engine. `GET /healthz` on the gateway still answers `200`,
  so the gateway itself is fine.
- `kubectl get pods -n <system>` shows the engine pod with `STATUS`
  `CrashLoopBackOff` or `Error`, `READY 0/1`, and a `RESTARTS` count that
  keeps growing. The gap between restarts doubles each time (10 s, 20 s,
  40 s, up to 5 min).

## Diagnosis

Work from the outside in. Each step either names the cause or tells you
where to look next.
<!-- steps 1 to 4 of section 5.2, each with what its output means -->

## Mitigation

Restore service first, then fix the cause.
<!-- the three steps of section 5.3 -->
```

The `/healthz` versus `/readyz` distinction in the Symptoms is what saves the most time: it rules out the gateway in one request. The model also has Verification and Escalation sections (a malformed `config.json` or a bad safetensors header means the artifact itself is bad: roll back, rebuild the model directory with `{tinyllm} train bigram`, redeploy). Full runbook: [`engine-crashloop.md`](../../../../course/solutions/docs/runbooks/engine-crashloop.md).

### 5.5 The model postmortem

The postmortem records this incident, blamelessly. Its root cause is not "someone typed the wrong path" but the three system properties that let a typo become an outage:

```markdown
## Root cause

The engine resolves `--model-dir` once, at startup, and exits when
`config.json` is missing. The Deployment was changed outside Helm, so nothing
rendered or reviewed the new args before they reached the cluster. The
Deployment's rolling update removed the only healthy pod before its
replacement was ready, because the engine has one replica and no readiness
gate that would have held the old pod.
```

Each action item then removes one of those properties, and is typed by what it does (mitigate, detect, prevent):

| Action | Type | Owner | Status |
|---|---|---|---|
| Write `docs/runbooks/engine-crashloop.md` | mitigate | engine | done |
| Make the engine log the resolved model path and exit with a clear message | detect | engine | done |
| Set `maxUnavailable: 0` on the engine rollout so the old pod stays until the new one is Ready | prevent | deploy | open |
| Alert on gateway `/readyz` failing for 1 minute (arrives with Prometheus in Pass 7) | detect | observability | open |
| Change the cluster only through `helm upgrade`, never `kubectl edit` or `patch` | prevent | team | open |

The timeline is a table in UTC with the evidence for each line, like section 3's. Full postmortem: [`2026-10-08-engine-crashloop.md`](../../../../course/solutions/docs/postmortems/2026-10-08-engine-crashloop.md).

### 5.6 Why this solution

- **Rollback before root cause.** It is the only action whose outcome is known in advance. Debugging in place keeps users down for the whole investigation; in section 3, the fix took 20 s of a 300 s TTM.
- **Why not delete the pod?** The ReplicaSet recreates it from the same broken template. Why not `kubectl edit` the args back? It works, but leaves the cluster diverged from the chart, and the next `helm upgrade` decides what runs.
- **Why a runbook and a postmortem?** They answer different readers: the runbook serves the next incident, the postmortem serves the team deciding what to change. Merging them produces a document that is neither current nor a faithful record.

### 5.7 Explain it back

1. The engine pod shows `Exit Code: 137`. Which branch of the diagnosis do you take, and why?
   <details><summary>Answer</summary>137 is $128 + 9$: the process was killed by SIGKILL, in a pod almost always the OOM killer (`Reason: OOMKilled`). The program did not choose to exit, so its log may say nothing; look at the memory limit and the rollout that changed it, as in section 3.</details>
2. Why is `kubectl logs` without `--previous` often empty in a crashloop?
   <details><summary>Answer</summary>It shows the current container attempt, which is usually waiting in back-off and has not started or has only just started. The message is in the attempt that died.</details>
3. With one replica, a readiness probe, and the default rolling update, do users see an outage during this bad rollout?
   <details><summary>Answer</summary>Not yet. `maxSurge` rounds up to 1 and `maxUnavailable` down to 0, so the new pod is added first and the old one is removed only when the new one is Ready, which never happens. The old pod keeps serving, but the rollout is stuck, and any restart of the old pod becomes an outage.</details>
4. Why is "someone changed the args" not the root cause in the postmortem?
   <details><summary>Answer</summary>A blameless postmortem asks why the system let a change cause an outage: no review path outside Helm, an engine that only fails at startup, and a rollout that removed the only healthy pod. Fixing those prevents the next incident regardless of who makes the next mistake.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Reading `kubectl logs` without `--previous` | an empty log, or "container is waiting to start" | diagnosis step 3 reads the crashed run |
| 2 | Deleting the crashing pod to "restart it" | the ReplicaSet makes an identical pod that crashes the same way | mitigation rolls back the template instead |
| 3 | Scaling the engine to zero to stop the restarts | the restarts stop and so does the service | mitigation restores a working revision; success is measured by a completion through the gateway |
| 4 | Fixing the args with `kubectl edit` and never touching the chart | it works until the next `helm upgrade` reapplies whatever the chart says | mitigation step 3 re-applies the chart; a postmortem action item forbids out-of-band changes |
| 5 | A postmortem that names who broke it, or skips a heading | the team stops reporting honestly; the record is incomplete | the model postmortem names system properties and has all seven sections |

## 7. Where it's used next
## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.00` | the Deployment, Service, and NodePort the drill injects into and checks through |
| Back | `L10.0` | the engine whose startup fails on a missing model directory |
| Back | `gw.00` | the gateway whose `/readyz` and 503 `no_capacity` are the symptom |
| Back | `lang.07` | `kubectl get`, `describe`, `logs`, and Helm |
| Forward | `obs.03` | burn-rate alerts that page before a user does |
| Forward | `ops.01` | the first drill detected by an alert, with TTD measured from Prometheus |
| Forward | `dep.03` | chart policy: readiness probes and disruption budgets, the action items here |
| Forward | `craft.10` | a runbook for every required alert, in the format started here |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| manual `rollout undo` | progressive delivery (Argo Rollouts, Flagger) | automatic rollback when the new version's health or SLO checks fail | Argo Rollouts analysis templates |
| reading exit codes by hand | `startupProbe`, Kubernetes events, `kubectl debug` | separates slow starts from crashes; ephemeral debug containers in a broken pod | Kubernetes docs, Configure Liveness, Readiness and Startup Probes |
| the fixed 10 s to 5 min back-off | KEP-4603, tunable CrashLoopBackOff | faster restarts for quick-failing containers | `kubernetes/enhancements`, keps/sig-node/4603 |
| `docs/runbooks/*.md` | incident tooling (PagerDuty, incident.io, Rootly) | runbooks linked from the alert, incident roles, automated timelines | Google SRE book ch. 14 (free) |
| the postmortem | blameless postmortem programs | review meetings, tracked action items, shared learning | Google SRE book ch. 15 (free); Etsy, *Debriefing Facilitation Guide* (free) |

## Company Relevance

| Company | Practice | Why it matters |
|---|---|---|
| Google | SRE incident management and blameless postmortems | the origin of most of this chapter's vocabulary |
| Netflix | chaos engineering in production | drills are how a team learns its system before a real outage teaches it |
| Any on-call team | runbooks linked from every alert | the difference between a 5-minute and a 45-minute mitigation |
