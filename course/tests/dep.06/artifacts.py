"""dep.06 artifact check: durable and worker images and charts (WAL PVC, StatefulSet, KEDA).

Run by `ol check dep.06` in your repo. Three tiers, in order:

  static   deploy/docker/{durable,worker}.Dockerfile follow the dep.01 rules;
           deploy/helm/<system>-{durable,worker} carry
           contracts/helm/{durable,worker}.values.schema.json and Helm
           enforces it; the dep.03 policy over `helm template`; the durable
           server is a one-replica StatefulSet whose WAL is on a
           volumeClaimTemplate a non-root user can write, with
           [durable].wal_max_bytes rendered from walMaxBytes; the worker
           drains before Kubernetes kills it, writes /artifacts, and is
           scaled by a KEDA ScaledObject on tl_durable_task_queue_depth;
           deploy/keda pins KEDA as contracts/helm/observability.md says
  docker   both images build from your entry points (go/cmd/durable,
           go/cmd/worker), run as a numeric non-root user, and the worker
           image can import the Python units its activities exec
  cluster  on [deploy].kube_context: KEDA is installed, the rendered charts pass
           `kubectl apply --dry-run=server`, the durable StatefulSet is ready
           with a bound WAL claim, the worker is available, and KEDA made an
           HPA for it

`<system>` below is [system].name from system.toml. The cluster tier fails when
the context is missing or down; with OL_SMOKE=1 it is skipped instead. The
queue-burst scale-up and scale-down on kind is a step of MS-durable's kind run.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import (  # noqa: E402
    Ctx,
    Fail,
    check_dockerfile,
    containers,
    dockerfile_stages,
    image_tag_problem,
    objects,
    one,
    pod_spec,
    run,
)

PARTS = ("durable", "worker")
SECRETISH = re.compile(r"(KEY|SECRET|PEPPER|TOKEN|PASSWORD|CREDENTIAL)", re.I)
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
QUEUE_METRIC = "tl_durable_task_queue_depth"
VERSION = "0.0.0-check"
LABELS = [
    "org.opencontainers.image.title",
    "org.opencontainers.image.source",
    "org.opencontainers.image.version",
    "org.opencontainers.image.revision",
]


def course_tree() -> Path:
    return Path(os.environ.get("OL_COURSE_TREE") or Path(__file__).resolve().parents[2])


def chart(c: Ctx, part: str) -> str:
    return f"deploy/helm/{c.system_name()}-{part}"


def render(c: Ctx, part: str, sets: dict | None = None) -> list:
    n = c.system_name()
    return c.helm_template(chart(c, part), f"{n}-{part}", sets=sets, namespace=n)


def contract_path(c: Ctx, rel: str) -> Path:
    for p in (c.path(f"contracts/{rel}"), course_tree() / "contracts" / rel):
        if p.is_file():
            return p
    raise Fail(f"contracts/{rel} is missing: run `ol contracts sync`")


def workloads(docs: list) -> list[dict]:
    return objects(docs, "Deployment") + objects(docs, "StatefulSet")


def labels_match(selector: dict, labels: dict) -> bool:
    ml = (selector or {}).get("matchLabels") or {}
    return bool(ml) and all(labels.get(k) == v for k, v in ml.items())


def runtime_toml(c: Ctx, docs: list, where: str) -> dict:
    """The runtime.toml a workload mounts, parsed."""
    for cm in objects(docs, "ConfigMap"):
        text = (cm.get("data") or {}).get("runtime.toml")
        if text is not None:
            try:
                return tomllib.loads(text)
            except tomllib.TOMLDecodeError as e:
                raise Fail(f"{where}: runtime.toml does not parse: {e}") from None
    raise Fail(f"{where}: no ConfigMap carries runtime.toml")


def mount_of(ctr: dict, volume: str) -> dict | None:
    return next(
        (m for m in ctr.get("volumeMounts") or [] if m.get("name") == volume), None
    )


# -- static: images -----------------------------------------------------------


def test_layout_present(c: Ctx) -> None:
    # WHY: the images, charts, and the KEDA pin live where MS-durable, the
    #      drills (durable-kill9, poison-task, eventlog-disk-full), and your
    #      Tiltfile look for them.
    # KIND: unit
    # CHAPTER: dep.06 section 4, The artifact and its check
    for rel in (
        "deploy/docker/durable.Dockerfile",
        "deploy/docker/worker.Dockerfile",
        f"{chart(c, 'durable')}/Chart.yaml",
        f"{chart(c, 'durable')}/values.schema.json",
        f"{chart(c, 'worker')}/Chart.yaml",
        f"{chart(c, 'worker')}/values.schema.json",
        "deploy/keda/Chart.yaml",
        "deploy/keda/Chart.lock",
    ):
        c.require_file(rel)


def test_dockerfiles_follow_the_image_rules(c: Ctx) -> None:
    # WHY: the dep.01 rules hold for every image of the platform: multi-stage,
    #      every base pinned by digest, a numeric non-root USER, exec-form
    #      ENTRYPOINT so the process is PID 1 and gets SIGTERM (a worker that
    #      never sees SIGTERM cannot drain), a HEALTHCHECK in exec form on the
    #      health port, and the OCI labels.
    # KIND: unit
    # CHAPTER: dep.06 section 2.1
    errs = []
    for part in PARTS:
        rel = f"deploy/docker/{part}.Dockerfile"
        text = c.require_file(rel).read_text()
        errs += check_dockerfile(text, rel, "9464")
        stages = dockerfile_stages(text)
        names = {s["as"] for s in stages if s["as"]}
        for s in stages:
            if (
                s["from"] not in names
                and s["from"] != "scratch"
                and not DIGEST.search(s["from"])
            ):
                errs.append(
                    f"{rel}: FROM {s['from']} is not pinned by digest (@sha256:...)"
                )
        final = "\n".join(stages[-1]["lines"]) if stages else ""
        hc = re.search(r"^HEALTHCHECK\b.*?\bCMD\s+(\S)", final, re.M | re.S)
        if not hc or hc.group(1) != "[":
            errs.append(
                f'{rel}: the final stage needs HEALTHCHECK ... CMD ["..."] (exec form)'
            )
        for label in LABELS:
            if label not in final:
                errs.append(f"{rel}: the final stage has no LABEL {label}")
    worker = c.require_file("deploy/docker/worker.Dockerfile").read_text()
    if not re.search(r"^COPY\s+(--\S+\s+)*python/", worker, re.M):
        errs.append(
            "deploy/docker/worker.Dockerfile: the image must carry python/ (the subprocess activities exec it)"
        )
    if "./cmd/worker" not in worker:
        errs.append("deploy/docker/worker.Dockerfile: no build of ./cmd/worker")
    if (
        "./cmd/durable"
        not in c.require_file("deploy/docker/durable.Dockerfile").read_text()
    ):
        errs.append("deploy/docker/durable.Dockerfile: no build of ./cmd/durable")
    if errs:
        raise Fail("\n".join(errs))


# -- static: charts -------------------------------------------------------------


def test_values_schemas_are_the_contract(c: Ctx) -> None:
    # WHY: the course writes the values contract and every drill relies on
    #      its shape (walMaxBytes is how ops.11 lowers the quota); a chart
    #      whose schema drifted or was loosened protects nothing.
    # KIND: conformance
    # CHAPTER: dep.06 section 2.2
    errs = []
    for part in PARTS:
        want = json.loads(
            contract_path(c, f"helm/{part}.values.schema.json").read_text()
        )
        try:
            got = json.loads(
                c.require_file(f"{chart(c, part)}/values.schema.json").read_text()
            )
        except json.JSONDecodeError as e:
            raise Fail(
                f"{chart(c, part)}/values.schema.json is not JSON: {e}"
            ) from None
        if got != want:
            errs.append(
                f"{chart(c, part)}/values.schema.json differs from contracts/helm/{part}.values.schema.json: copy it unchanged"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_charts_lint(c: Ctx) -> None:
    # WHY: `helm lint` validates values.yaml against the schema and the
    #      templates against Kubernetes' shapes before anything reaches kind.
    # KIND: conformance
    # CHAPTER: dep.06 section 4, The artifact and its check
    c.sh(["helm", "lint", chart(c, "durable"), chart(c, "worker")], timeout=60)


def test_schema_rejects_unsafe_values(c: Ctx) -> None:
    # WHY: each of these must stop `helm template` before an install: tag
    #      latest; --test-clock in a cluster; no WAL volume; no WAL quota; two
    #      durable replicas without Raft; a worker with no queue or a durable
    #      address that is not host:port.
    # KIND: boundary
    # CHAPTER: dep.06 section 5, Pitfalls
    n = c.system_name()
    cases = [
        ("durable", ["--set", "image.tag=latest"], "image.tag=latest"),
        ("durable", ["--set", "testClock=true"], "testClock=true"),
        (
            "durable",
            ["--set", "persistence.enabled=false"],
            "persistence.enabled=false",
        ),
        ("durable", ["--set", "walMaxBytes=0"], "walMaxBytes=0"),
        ("durable", ["--set", "replicaCount=2"], "two replicas"),
        ("worker", ["--set", "image.tag=latest"], "image.tag=latest"),
        ("worker", ["--set", "taskQueues=null"], "no task queues"),
        (
            "worker",
            ["--set-string", "durableAddress=forge-durable"],
            "a durable address without a port",
        ),
    ]
    render(c, "durable")
    render(c, "worker")
    errs = []
    for part, args, what in cases:
        r = c.sh(
            ["helm", "template", f"{n}-{part}", chart(c, part), *args],
            timeout=60,
            check=False,
        )
        if r.returncode == 0:
            errs.append(
                f"{chart(c, part)} renders with {what}: values.schema.json is not enforced"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_policy_over_rendered_charts(c: Ctx) -> None:
    # WHY: the chart policy of DESIGN 2.13 on what Kubernetes will receive:
    #      probes, CPU and memory requests and limits, a numeric non-root user
    #      without privilege escalation, a pinned image, and a
    #      PodDisruptionBudget for every workload.
    # KIND: unit
    # CHAPTER: dep.06 section 2.2
    errs = []
    for part in PARTS:
        docs = render(c, part)
        pdbs = objects(docs, "PodDisruptionBudget")
        ws = workloads(docs)
        if not ws:
            errs.append(f"{chart(c, part)} renders no Deployment or StatefulSet")
        for w in ws:
            where = f"{chart(c, part)}: {w['kind']} {w['metadata'].get('name')}"
            ps = pod_spec(w)
            psc = ps.get("securityContext") or {}
            labels = ((w["spec"].get("template") or {}).get("metadata") or {}).get(
                "labels"
            ) or {}
            for ctr in containers(w):
                sc = ctr.get("securityContext") or {}
                for probe in ("livenessProbe", "readinessProbe"):
                    if not ctr.get(probe):
                        errs.append(f"{where}: container {ctr['name']} has no {probe}")
                res = ctr.get("resources") or {}
                for kind in ("requests", "limits"):
                    if not {"cpu", "memory"} <= set(res.get(kind) or {}):
                        errs.append(
                            f"{where}: container {ctr['name']} lacks resources.{kind}.cpu and .memory"
                        )
                if not sc.get("runAsNonRoot", psc.get("runAsNonRoot")):
                    errs.append(f"{where}: runAsNonRoot is not true")
                uid = sc.get("runAsUser", psc.get("runAsUser"))
                if not isinstance(uid, int) or uid <= 0:
                    errs.append(
                        f"{where}: runAsUser is {uid!r}; want a numeric uid above 0"
                    )
                if sc.get("allowPrivilegeEscalation") is not False:
                    errs.append(f"{where}: allowPrivilegeEscalation is not false")
                why = image_tag_problem(str(ctr.get("image", "")))
                if why:
                    errs.append(f"{where}: {why}")
            if not any(
                labels_match((p.get("spec") or {}).get("selector"), labels)
                for p in pdbs
            ):
                errs.append(f"{where}: no PodDisruptionBudget selects its pods")
            for e in [e for ctr in containers(w) for e in ctr.get("env") or []]:
                if SECRETISH.search(e.get("name", "")) and "value" in e:
                    errs.append(
                        f"{where}: env {e['name']} has a literal value; use valueFrom.secretKeyRef"
                    )
        for s in objects(docs, "Secret"):
            if s.get("data") or s.get("stringData"):
                errs.append(
                    f"{chart(c, part)}: renders Secret {s['metadata'].get('name')} with values"
                )
    if errs:
        raise Fail("\n".join(dict.fromkeys(errs)))


def test_durable_is_a_statefulset_with_a_wal_volume(c: Ctx) -> None:
    # WHY: the WAL is the durable server's only state. In a Deployment with
    #      an emptyDir it dies with the pod and every workflow is gone; a
    #      StatefulSet gives the pod a PersistentVolumeClaim of its own that
    #      survives restarts and rescheduling. One replica (one writer; three
    #      only with Raft), the claim mounted where [durable].wal_dir points,
    #      and an fsGroup so a non-root server can write it.
    # KIND: unit
    # CHAPTER: dep.06 section 2.3
    docs = render(c, "durable")
    if objects(docs, "Deployment"):
        raise Fail(
            f"{chart(c, 'durable')} renders a Deployment: the server must be a StatefulSet"
        )
    ss = one(docs, "StatefulSet", chart(c, "durable"))
    errs = []
    if ss["spec"].get("replicas") != 1:
        errs.append(
            f"the StatefulSet has {ss['spec'].get('replicas')} replicas; want 1 (3 only with Raft, dur.10)"
        )
    vcts = ss["spec"].get("volumeClaimTemplates") or []
    if not vcts:
        errs.append("no volumeClaimTemplates: the WAL has no PersistentVolumeClaim")
    ctr = containers(ss)[0]
    wal_dir = str(
        (runtime_toml(c, docs, chart(c, "durable")).get("durable") or {}).get(
            "wal_dir", ""
        )
    )
    covered = False
    for v in vcts:
        name = (v.get("metadata") or {}).get("name")
        size = (
            ((v.get("spec") or {}).get("resources") or {}).get("requests") or {}
        ).get("storage")
        if not size:
            errs.append(f"volumeClaimTemplate {name} requests no storage")
        m = mount_of(ctr, name)
        if m is None:
            errs.append(f"volumeClaimTemplate {name} is not mounted in the container")
        elif wal_dir and (
            wal_dir == m["mountPath"]
            or wal_dir.startswith(m["mountPath"].rstrip("/") + "/")
        ):
            covered = True
    if not wal_dir:
        errs.append(
            "runtime.toml has no [durable].wal_dir: the server would write its WAL somewhere unmounted"
        )
    elif not covered:
        errs.append(
            f"[durable].wal_dir {wal_dir} is not on the claim's mount: the WAL would land on the read-only root filesystem"
        )
    if not (pod_spec(ss).get("securityContext") or {}).get("fsGroup"):
        errs.append(
            "the pod sets no securityContext.fsGroup: a non-root server cannot write a freshly provisioned volume"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_wal_quota_is_rendered(c: Ctx) -> None:
    # WHY: kind's local-path volumes do not enforce their size, so
    #      walMaxBytes is the only disk quota (ops.11): it must reach
    #      [durable].wal_max_bytes as an integer, follow the value given at
    #      install time, and stay below the claim's size.
    # KIND: unit
    # CHAPTER: dep.06 section 3, Worked example by hand
    errs = []
    n = c.system_name()
    for value in (None, 123456789):
        if value is None:
            docs = render(c, "durable")
        else:
            out = c.sh(
                [
                    "helm",
                    "template",
                    f"{n}-durable",
                    chart(c, "durable"),
                    "--set",
                    f"walMaxBytes={value}",
                ],
                timeout=60,
            ).stdout
            docs = c.yaml_docs(out, "helm template --set walMaxBytes")
        got = (runtime_toml(c, docs, chart(c, "durable")).get("durable") or {}).get(
            "wal_max_bytes"
        )
        want = value
        if want is None:
            vals = c.yaml_file(f"{chart(c, 'durable')}/values.yaml")[0] or {}
            want = vals.get("walMaxBytes")
        if not isinstance(got, int) or isinstance(got, bool):
            errs.append(
                f"wal_max_bytes is {got!r}: runtime.schema.json wants an integer (Helm reads YAML numbers as floats; cast with int64)"
            )
        elif got != want:
            errs.append(
                f"with walMaxBytes={want} the runtime.toml says wal_max_bytes = {got}"
            )
    docs = render(c, "durable")
    ss = one(docs, "StatefulSet", chart(c, "durable"))
    size = None
    for v in ss["spec"].get("volumeClaimTemplates") or []:
        size = (
            ((v.get("spec") or {}).get("resources") or {}).get("requests") or {}
        ).get("storage")
    mult = {"Mi": 2**20, "Gi": 2**30, "Ti": 2**40}
    m = re.fullmatch(r"([0-9]+)(Mi|Gi|Ti)", str(size or ""))
    quota = (runtime_toml(c, docs, "").get("durable") or {}).get("wal_max_bytes")
    if m and isinstance(quota, int) and quota >= int(m.group(1)) * mult[m.group(2)]:
        errs.append(
            f"wal_max_bytes {quota} is not below the claim's {size}: the disk fills before the quota trips"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_durable_service_ports(c: Ctx) -> None:
    # WHY: workers in the cluster reach the server at <system>-durable:7233;
    #      a training worker on your Mac reaches it through NodePort 30733,
    #      which dep.02's cluster maps to 127.0.0.1 (DESIGN 2.13); Prometheus
    #      scrapes the health port.
    # KIND: unit
    # CHAPTER: dep.06 section 2.3
    n = c.system_name()
    docs = render(c, "durable")
    svc = next(
        (
            s
            for s in objects(docs, "Service")
            if s["metadata"].get("name") == f"{n}-durable"
        ),
        None,
    )
    if svc is None:
        raise Fail(f"no Service named {n}-durable")
    ports = {p.get("port"): p for p in (svc.get("spec") or {}).get("ports") or []}
    errs = []
    if 7233 not in ports:
        errs.append("Service port 7233 (gRPC) is missing")
    elif (svc["spec"].get("type") == "NodePort") and ports[7233].get(
        "nodePort"
    ) != 30733:
        errs.append(f"gRPC nodePort is {ports[7233].get('nodePort')}; want 30733")
    elif svc["spec"].get("type") != "NodePort":
        errs.append(
            "the Service is not a NodePort: a worker on the host cannot reach it"
        )
    if 9464 not in ports:
        errs.append("Service port 9464 (health and /metrics) is missing")
    if errs:
        raise Fail("\n".join(errs))


def test_worker_drains_before_it_is_killed(c: Ctx) -> None:
    # WHY: on SIGTERM a worker stops polling and lets in-flight activities
    #      finish (dur.04, up to 30 s), and a Python child gets 30 s between
    #      SIGTERM and SIGKILL to checkpoint (dur.09). Kubernetes SIGKILLs the
    #      pod after terminationGracePeriodSeconds (default 30): anything
    #      shorter than 60 kills training between checkpoints on every
    #      rollout and every KEDA scale-down.
    # KIND: boundary
    # CHAPTER: dep.06 section 2.4
    dep = one(render(c, "worker"), "Deployment", chart(c, "worker"))
    g = pod_spec(dep).get("terminationGracePeriodSeconds", 30)
    if not isinstance(g, int) or g < 60:
        raise Fail(
            f"terminationGracePeriodSeconds is {g}; want at least 60 (drain plus the child's SIGTERM grace)"
        )


def test_worker_writes_artifacts(c: Ctx) -> None:
    # WHY: subprocess activities write checkpoints, shards, and DONE.json
    #      under /artifacts (TL_ARTIFACTS). The engine mounts it read-only;
    #      the worker must mount it read-write, and the root filesystem stays
    #      read-only, so scratch space is an emptyDir.
    # KIND: unit
    # CHAPTER: dep.06 section 2.4
    dep = one(render(c, "worker"), "Deployment", chart(c, "worker"))
    ctr = containers(dep)[0]
    vols = {v["name"]: v for v in pod_spec(dep).get("volumes") or []}
    art = next(
        (
            m
            for m in ctr.get("volumeMounts") or []
            if m.get("mountPath") == "/artifacts"
        ),
        None,
    )
    errs = []
    if art is None:
        errs.append("the worker does not mount /artifacts")
    elif art.get("readOnly"):
        errs.append(
            "/artifacts is mounted read-only: no activity can write its outputs"
        )
    elif "emptyDir" in vols.get(art["name"], {}):
        errs.append(
            "/artifacts is an emptyDir: outputs die with the pod and the engine never sees them"
        )
    sc = ctr.get("securityContext") or {}
    if sc.get("readOnlyRootFilesystem") is not True:
        errs.append(
            "the worker's root filesystem is writable: set readOnlyRootFilesystem and give it an emptyDir for /tmp"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_keda_scales_on_queue_depth(c: Ctx) -> None:
    # WHY: queue depth, not CPU, says whether workers are missing: a burst of
    #      200 corpus shards is 200 pending tasks long before any CPU rises,
    #      and a worker waiting on a subprocess is idle by CPU. The
    #      ScaledObject targets the worker Deployment, reads
    #      tl_durable_task_queue_depth for the worker's queue from Prometheus,
    #      and owns the replica count: the Deployment sets none and no other
    #      HPA competes for it.
    # KIND: unit
    # CHAPTER: dep.06 section 2.5
    n = c.system_name()
    docs = render(c, "worker")
    vals = c.yaml_file(f"{chart(c, 'worker')}/values.yaml")[0] or {}
    queue = str(((vals.get("autoscaling") or {}).get("queue")) or "")
    so = one(docs, "ScaledObject", chart(c, "worker"))
    spec = so.get("spec") or {}
    errs = []
    if not str(so.get("apiVersion", "")).startswith("keda.sh/"):
        errs.append(
            f"ScaledObject apiVersion is {so.get('apiVersion')}; want keda.sh/v1alpha1"
        )
    tgt = spec.get("scaleTargetRef") or {}
    if (
        tgt.get("name") != f"{n}-worker"
        or tgt.get("kind", "Deployment") != "Deployment"
    ):
        errs.append(f"scaleTargetRef is {tgt}; want the Deployment {n}-worker")
    lo, hi = spec.get("minReplicaCount", 0), spec.get("maxReplicaCount", 100)
    if not (isinstance(lo, int) and isinstance(hi, int) and hi > lo >= 1):
        errs.append(
            f"minReplicaCount {lo}, maxReplicaCount {hi}: want 1 <= min < max (min 0 parks the queue until KEDA's activation)"
        )
    trig = [t for t in spec.get("triggers") or [] if t.get("type") == "prometheus"]
    if not trig:
        errs.append("no prometheus trigger")
    for t in trig:
        md = t.get("metadata") or {}
        q = str(md.get("query", ""))
        if QUEUE_METRIC not in q:
            errs.append(f"the trigger query {q!r} does not read {QUEUE_METRIC}")
        if queue and f'"{queue}"' not in q:
            errs.append(
                f'the trigger query {q!r} does not select queue "{queue}" (autoscaling.queue)'
            )
        try:
            if float(md.get("threshold", "0")) <= 0:
                raise ValueError
        except ValueError:
            errs.append(
                f"threshold {md.get('threshold')!r} must be a positive number (tasks per worker)"
            )
        if not str(md.get("serverAddress", "")).startswith("http"):
            errs.append("the trigger has no Prometheus serverAddress")
    dep = one(docs, "Deployment", chart(c, "worker"))
    if "replicas" in dep["spec"]:
        errs.append(
            "the Deployment sets spec.replicas while KEDA scales it: every helm upgrade resets the count"
        )
    if objects(docs, "HorizontalPodAutoscaler"):
        errs.append(
            "the chart also renders an HPA: two controllers would fight over the replica count"
        )
    if errs:
        raise Fail("\n".join(errs))


def pins(c: Ctx) -> dict[str, tuple[str, str]]:
    out = {}
    for line in contract_path(c, "helm/observability.md").read_text().splitlines():
        cells = re.findall(r"`([^`]+)`", line)
        if line.startswith("|") and len(cells) >= 3 and cells[1].startswith("https://"):
            out[cells[0]] = (cells[1], cells[2])
    return out


def test_keda_is_pinned(c: Ctx) -> None:
    # WHY: the course tests against one KEDA version; deploy/keda/Chart.yaml
    #      names it without a range and Chart.lock records what
    #      `helm dependency update` resolved, so a newer KEDA is an explicit
    #      upgrade (craft.15), never drift.
    # KIND: conformance
    # CHAPTER: dep.06 section 2.5
    repo, ver = pins(c)["keda"]
    meta = c.yaml_file("deploy/keda/Chart.yaml")[0] or {}
    lock = c.yaml_file("deploy/keda/Chart.lock")[0] or {}
    d = {x.get("name"): x for x in meta.get("dependencies") or []}.get("keda")
    lk = {x.get("name"): x for x in lock.get("dependencies") or []}.get("keda")
    errs = []
    if d is None:
        errs.append("deploy/keda/Chart.yaml has no dependency keda")
    elif str(d.get("version")) != ver or d.get("repository") != repo:
        errs.append(
            f"keda: Chart.yaml has {d.get('version')} from {d.get('repository')}; the pin is {ver} from {repo}"
        )
    if lk is None or str(lk.get("version")) != ver or lk.get("repository") != repo:
        errs.append(
            f"keda: Chart.lock does not record {ver} from {repo}: run helm dependency update deploy/keda"
        )
    if not str(lock.get("digest", "")).startswith("sha256:"):
        errs.append(
            "deploy/keda/Chart.lock has no digest: it was not written by helm dependency update"
        )
    if errs:
        raise Fail("\n".join(errs))


# -- docker ---------------------------------------------------------------------


def images(c: Ctx) -> dict[str, str]:
    if "images" not in c.cache:
        c.need_docker()
        c.cache["images"] = None
        for cmd in ("go/cmd/durable", "go/cmd/worker"):
            if not c.path(cmd).is_dir():
                raise Fail(
                    f"{cmd} is missing: the images build your entry points (dur.02 server, dur.04 worker)"
                )
        out = {}
        for part in PARTS:
            tag = f"ol-check/{c.system_name()}-{part}:dep06"
            c.sh(
                [
                    "docker",
                    "build",
                    "-q",
                    "-f",
                    f"deploy/docker/{part}.Dockerfile",
                    "--build-arg",
                    f"VERSION={VERSION}",
                    "--build-arg",
                    "REVISION=check",
                    "-t",
                    tag,
                    ".",
                ],
                timeout=1800,
            )
            out[part] = tag
        c.cache["images"] = out
    if c.cache["images"] is None:
        raise Fail("the images did not build (see test_images_build)")
    return c.cache["images"]


def test_images_build(c: Ctx) -> None:
    # WHY: both images build from the repo root with the pinned bases, the
    #      way CI and Tilt build them.
    # KIND: unit
    # CHAPTER: dep.06 section 4, The artifact and its check
    images(c)


def test_images_run_as_numeric_non_root(c: Ctx) -> None:
    # WHY: runAsNonRoot can only verify a numeric USER; a root image fails
    #      admission in the chart, or worse, writes /artifacts as root.
    # KIND: unit
    # CHAPTER: dep.06 section 2.1
    errs = []
    for part, tag in images(c).items():
        user = json.loads(c.sh(["docker", "image", "inspect", tag]).stdout)[0][
            "Config"
        ].get("User", "")
        uid = user.split(":")[0]
        if not uid.isdigit() or uid == "0":
            errs.append(f"{part}: image USER is {user!r}; want a numeric uid above 0")
    if errs:
        raise Fail("\n".join(errs))


def test_worker_image_has_the_python_units(c: Ctx) -> None:
    # WHY: the worker execs Python entries; an image that builds but cannot
    #      import numpy or tinyllm fails every subprocess activity with exit 1
    #      only once it runs on kind.
    # KIND: unit
    # CHAPTER: dep.06 section 2.1
    tag = images(c)["worker"]
    code = "import numpy, sys; sys.path.insert(0, '/app/python'); import tinyllm.io.activity, tinyllm.io.telemetry"
    r = c.sh(
        ["docker", "run", "--rm", "--entrypoint", "python", tag, "-c", code],
        timeout=120,
        check=False,
    )
    if r.returncode != 0:
        raise Fail(
            "the worker image cannot import its Python units:\n"
            + (r.stdout + r.stderr)[-1500:]
        )


# -- cluster ----------------------------------------------------------------------


def context(c: Ctx) -> tuple[str, str]:
    d = c.system().get("deploy") or {}
    ctx = d.get("kube_context") or f"kind-{c.system_name()}"
    c.need_cluster(ctx)
    return ctx, d.get("namespace") or c.system_name()


def test_keda_installed(c: Ctx) -> None:
    # WHY: a ScaledObject without KEDA's CRD is rejected by the API server,
    #      and without its operator nothing scales.
    # KIND: conformance
    # CHAPTER: dep.06 section 4, The artifact and its check
    ctx, _ = context(c)
    r = c.sh(
        ["kubectl", "--context", ctx, "get", "crd", "scaledobjects.keda.sh"],
        check=False,
    )
    if r.returncode != 0:
        raise Fail(
            "CRD scaledobjects.keda.sh is missing: helm upgrade --install keda deploy/keda -n keda --create-namespace"
        )
    r = c.sh(
        ["helm", "--kube-context", ctx, "-n", "keda", "status", "keda", "-o", "json"],
        check=False,
    )
    if (
        r.returncode != 0
        or (json.loads(r.stdout).get("info") or {}).get("status") != "deployed"
    ):
        raise Fail("helm release keda (namespace keda) is not deployed")


def test_server_side_dry_run(c: Ctx) -> None:
    # WHY: the API server validates what `helm template` cannot: the
    #      ScaledObject against KEDA's CRD, the StatefulSet's claim template,
    #      and the namespace.
    # KIND: conformance
    # CHAPTER: dep.06 section 4, The artifact and its check
    ctx, ns = context(c)
    n = c.system_name()
    for part in PARTS:
        out = c.sh(
            ["helm", "template", f"{n}-{part}", chart(c, part), "-n", ns], timeout=60
        ).stdout
        c.sh(
            [
                "kubectl",
                "--context",
                ctx,
                "-n",
                ns,
                "apply",
                "--dry-run=server",
                "-f",
                "-",
            ],
            timeout=60,
            input=out,
        )


def test_releases_ready(c: Ctx) -> None:
    # WHY: installed, not just renderable: the durable pod is ready with its
    #      WAL claim bound, the worker is available, and KEDA turned the
    #      ScaledObject into an HPA on the worker.
    # KIND: conformance
    # CHAPTER: dep.06 section 3, Worked example by hand
    ctx, ns = context(c)
    n = c.system_name()
    errs = []
    r = c.sh(
        [
            "kubectl",
            "--context",
            ctx,
            "-n",
            ns,
            "get",
            "statefulset",
            f"{n}-durable",
            "-o",
            "json",
        ],
        check=False,
    )
    if (
        r.returncode != 0
        or int((json.loads(r.stdout).get("status") or {}).get("readyReplicas") or 0) < 1
    ):
        errs.append(f"statefulset/{n}-durable has no ready replica")
    r = c.sh(
        ["kubectl", "--context", ctx, "-n", ns, "get", "pvc", "-o", "json"], check=False
    )
    claims = json.loads(r.stdout).get("items", []) if r.returncode == 0 else []
    if not any(
        p["metadata"]["name"].startswith(f"wal-{n}-durable-")
        and (p.get("status") or {}).get("phase") == "Bound"
        for p in claims
    ):
        errs.append(f"no bound claim wal-{n}-durable-0")
    r = c.sh(
        [
            "kubectl",
            "--context",
            ctx,
            "-n",
            ns,
            "get",
            "deploy",
            f"{n}-worker",
            "-o",
            "json",
        ],
        check=False,
    )
    if (
        r.returncode != 0
        or int((json.loads(r.stdout).get("status") or {}).get("availableReplicas") or 0)
        < 1
    ):
        errs.append(f"deploy/{n}-worker is not available")
    r = c.sh(
        [
            "kubectl",
            "--context",
            ctx,
            "-n",
            ns,
            "get",
            "hpa",
            f"keda-hpa-{n}-worker",
            "-o",
            "json",
        ],
        check=False,
    )
    if r.returncode != 0:
        errs.append(
            f"KEDA made no HPA keda-hpa-{n}-worker: is the ScaledObject Ready? (kubectl describe scaledobject {n}-worker)"
        )
    if errs:
        raise Fail("\n".join(errs))


if __name__ == "__main__":
    raise SystemExit(run(globals()))
