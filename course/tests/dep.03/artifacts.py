"""dep.03 artifact check: Helm charts for the gateway and the engine, plus the observability stack.

Run by `ol check dep.03` in your repo. Two tiers, in order:

  static   each chart carries contracts/helm/<part>.values.schema.json and Helm
           enforces it; the policy over `helm template` (probes, requests and
           limits, non-root, no :latest, a PodDisruptionBudget that lets a node
           drain, secrets only by reference); one gateway replica whose ledger
           sits on a PersistentVolumeClaim; one engine release per role; the
           observability umbrella chart pinned to contracts/helm/observability.md
           with the default PrometheusRule selector; [deploy].prometheus
  cluster  on [deploy].kube_context: every rendered chart passes
           `kubectl apply --dry-run=server`, the gateway and engine Deployments
           are available, and the observability release is deployed

`<system>` below is [system].name from system.toml. The cluster tier fails when
the context is missing or down; with OL_SMOKE=1 it is skipped instead.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import (
    Ctx,
    Fail,
    containers,
    image_tag_problem,
    objects,
    one,
    pod_spec,
    run,
)  # noqa: E402

ROLES = ("unified", "prefill", "decode")
SECRETISH = re.compile(r"(KEY|SECRET|PEPPER|TOKEN|PASSWORD|CREDENTIAL)", re.I)
STACK = ("opentelemetry-collector", "kube-prometheus-stack", "tempo")


def course_tree() -> Path:
    return Path(os.environ.get("OL_COURSE_TREE") or Path(__file__).resolve().parents[2])


def chart(c: Ctx, part: str) -> str:
    return f"deploy/helm/{c.system_name()}-{part}"


def render(
    c: Ctx, part: str, release: str | None = None, sets: dict | None = None
) -> list:
    n = c.system_name()
    return c.helm_template(
        chart(c, part), release or f"{n}-{part}", sets=sets, namespace=n
    )


def contract_path(c: Ctx, rel: str) -> Path:
    for p in (c.path(f"contracts/{rel}"), course_tree() / "contracts" / rel):
        if p.is_file():
            return p
    raise Fail(f"contracts/{rel} is missing: run `ol contracts sync`")


def renders(c: Ctx) -> list[tuple[str, list]]:
    """Every rendering the policy covers: the gateway, and the engine per role."""
    n = c.system_name()
    out = [(f"{n}-gateway", render(c, "gateway"))]
    for role in ROLES:
        rel = f"{n}-engine" if role == "unified" else f"{n}-engine-{role}"
        out.append((f"{rel} (role {role})", render(c, "engine", rel, {"role": role})))
    return out


def workloads(docs: list) -> list[dict]:
    return objects(docs, "Deployment") + objects(docs, "StatefulSet")


def labels_match(selector: dict, labels: dict) -> bool:
    ml = (selector or {}).get("matchLabels") or {}
    return bool(ml) and all(labels.get(k) == v for k, v in ml.items())


# -- static -------------------------------------------------------------------


def test_layout_present(c: Ctx) -> None:
    # WHY: the charts and the stack live where MS-prod, the drills, and dep.04's
    #      Tiltfile look for them.
    # KIND: unit
    # CHAPTER: dep.03 section 4, The artifact and its check
    for rel in (
        f"{chart(c, 'gateway')}/Chart.yaml",
        f"{chart(c, 'engine')}/Chart.yaml",
        f"{chart(c, 'gateway')}/values.schema.json",
        f"{chart(c, 'engine')}/values.schema.json",
        "deploy/observability/Chart.yaml",
        "deploy/observability/Chart.lock",
        "deploy/observability/values.yaml",
    ):
        c.require_file(rel)


def test_values_schemas_are_the_contract(c: Ctx) -> None:
    # WHY: the course writes the values contract (contracts/helm/*.values.schema.json)
    #      and every drill and milestone relies on its shape; a chart whose
    #      schema drifted, or was loosened, no longer protects anything.
    # KIND: conformance
    # CHAPTER: dep.03 section 2.1
    errs = []
    for part in ("gateway", "engine"):
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
    #      templates against Kubernetes' shapes before anything reaches a cluster.
    # KIND: conformance
    c.sh(["helm", "lint", chart(c, "gateway"), chart(c, "engine")], timeout=60)


def test_schema_rejects_unsafe_values(c: Ctx) -> None:
    # WHY: a schema only helps if Helm sees it: each of these values must make
    #      `helm template` fail before an install: tag latest, a pepper written
    #      as a literal string, a second gateway replica, an unknown engine role.
    # KIND: boundary
    # CHAPTER: dep.03 section 5, Pitfall 1
    n = c.system_name()
    cases = [
        ("gateway", ["--set", "image.tag=latest"], "image.tag=latest"),
        ("gateway", ["--set-string", "pepper=not-a-reference"], "a literal pepper"),
        ("gateway", ["--set", "replicaCount=2"], "two gateway replicas"),
        ("engine", ["--set", "image.tag=latest"], "image.tag=latest"),
        ("engine", ["--set", "role=leader"], "role=leader"),
    ]
    render(c, "gateway")  # the defaults must render, or a refusal below proves nothing
    render(c, "engine")
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
    # WHY: the chart policy of DESIGN 2.13, checked on what Kubernetes will
    #      actually receive: every container has liveness and readiness probes
    #      and CPU and memory requests and limits, runs as a numeric non-root
    #      user without privilege escalation, uses a pinned image that kind
    #      will not re-pull, and every workload has a PodDisruptionBudget.
    # KIND: unit
    # CHAPTER: dep.03 section 2.2
    errs = []
    for name, docs in renders(c):
        pdbs = objects(docs, "PodDisruptionBudget")
        for w in workloads(docs):
            where = f"{name}: {w['kind']} {w['metadata'].get('name')}"
            ps = pod_spec(w)
            labels = ((w["spec"].get("template") or {}).get("metadata") or {}).get(
                "labels"
            ) or {}
            psc = ps.get("securityContext") or {}
            for ctr in containers(w):
                sc = ctr.get("securityContext") or {}
                for probe in ("livenessProbe", "readinessProbe"):
                    if not ctr.get(probe):
                        errs.append(f"{where}: container {ctr['name']} has no {probe}")
                res = ctr.get("resources") or {}
                for kind in ("requests", "limits"):
                    if not {"cpu", "memory"} <= set((res.get(kind) or {})):
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
                if ctr.get("imagePullPolicy") == "Always":
                    errs.append(
                        f"{where}: imagePullPolicy Always re-pulls on every start"
                    )
            if not any(
                labels_match((p.get("spec") or {}).get("selector"), labels)
                for p in pdbs
            ):
                errs.append(f"{where}: no PodDisruptionBudget selects its pods")
    if errs:
        raise Fail("\n".join(dict.fromkeys(errs)))


def test_secrets_only_by_reference(c: Ctx) -> None:
    # WHY: the pepper and the API keys exist in exactly one place, Secrets you
    #      create by hand. A chart that renders a Secret with data, or an env
    #      var with a secret-looking name and a literal value, puts the secret
    #      in git and in `helm get values`.
    # KIND: unit
    # CHAPTER: dep.03 section 2.3
    errs = []
    for name, docs in renders(c):
        for s in objects(docs, "Secret"):
            if s.get("data") or s.get("stringData"):
                errs.append(
                    f"{name}: renders Secret {s['metadata'].get('name')} with values"
                )
        for w in workloads(docs):
            for ctr in containers(w):
                for e in ctr.get("env") or []:
                    if SECRETISH.search(e.get("name", "")) and "value" in e:
                        errs.append(
                            f"{name}: env {e['name']} has a literal value; use valueFrom.secretKeyRef"
                        )
    gw = one(render(c, "gateway"), "Deployment", chart(c, "gateway"))
    refs = {
        (e.get("valueFrom") or {}).get("secretKeyRef", {}).get("name")
        for ct in containers(gw)
        for e in ct.get("env") or []
    }
    if not refs - {None}:
        errs.append(
            f"{chart(c, 'gateway')}: no env var reads the pepper from a Secret (values: pepper.name, pepper.key)"
        )
    if errs:
        raise Fail("\n".join(dict.fromkeys(errs)))


def test_gateway_one_replica_that_can_drain(c: Ctx) -> None:
    # WHY: the gateway keeps limits, the cache, the ledger, and the registry in
    #      one process, so it runs one replica; its PodDisruptionBudget must
    #      still allow one pod down (maxUnavailable 1), or `kubectl drain`
    #      waits forever on a single-replica Deployment.
    # KIND: boundary
    # CHAPTER: dep.03 section 5, Pitfall 3
    docs = render(c, "gateway")
    dep = one(docs, "Deployment", chart(c, "gateway"))
    errs = []
    if dep["spec"].get("replicas") != 1:
        errs.append(
            f"the gateway Deployment has {dep['spec'].get('replicas')} replicas; want 1"
        )
    for p in objects(docs, "PodDisruptionBudget"):
        spec = p.get("spec") or {}
        if spec.get("minAvailable") in (1, "100%") or spec.get("maxUnavailable") in (
            0,
            "0%",
        ):
            errs.append(
                f"PodDisruptionBudget {p['metadata'].get('name')} allows no pod down: a node drain never finishes"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_gateway_ledger_on_a_volume_claim(c: Ctx) -> None:
    # WHY: the usage ledger (gw.07) is a SQLite file. The root filesystem is
    #      read-only, the node's /artifacts mount is read-only, and an emptyDir
    #      dies with the pod: the bill must live on a PersistentVolumeClaim.
    # KIND: unit
    # CHAPTER: dep.03 section 2.4
    docs = render(c, "gateway")
    dep = one(docs, "Deployment", chart(c, "gateway"))
    vols = {v["name"]: v for v in pod_spec(dep).get("volumes") or []}
    claims = {
        v["persistentVolumeClaim"]["claimName"]
        for v in vols.values()
        if v.get("persistentVolumeClaim")
    }
    if not claims:
        raise Fail("the gateway mounts no PersistentVolumeClaim for its usage ledger")
    rendered = {p["metadata"]["name"] for p in objects(docs, "PersistentVolumeClaim")}
    if not claims <= rendered:
        raise Fail(
            f"the gateway mounts claims {sorted(claims)} that the chart does not create ({sorted(rendered)})"
        )


def test_engine_one_release_per_role(c: Ctx) -> None:
    # WHY: disaggregated serving runs a prefill and a decode release of the
    #      same chart; [deploy].services finds them as <system>-engine-prefill
    #      and -decode, and each engine must learn its role (env
    #      TL_ENGINE__ROLE, a --role flag, or role in its runtime.toml).
    # KIND: unit
    # CHAPTER: dep.03 section 3, Worked example by hand
    n = c.system_name()
    errs = []
    for role in ROLES:
        rel = f"{n}-engine" if role == "unified" else f"{n}-engine-{role}"
        docs = render(c, "engine", rel, {"role": role})
        dep = one(docs, "Deployment", f"{chart(c, 'engine')} role {role}")
        if dep["metadata"].get("name") != rel:
            errs.append(
                f"release {rel}: the Deployment is named {dep['metadata'].get('name')!r}; want {rel!r}"
            )
        ctr = containers(dep)[0]
        args = [str(a) for a in (ctr.get("command") or []) + (ctr.get("args") or [])]
        env = {e.get("name"): e.get("value") for e in ctr.get("env") or []}
        cms = " ".join(
            json.dumps(cm.get("data") or {}) for cm in objects(docs, "ConfigMap")
        )
        told = (
            env.get("TL_ENGINE__ROLE") == role
            or ("--role" in args and args[args.index("--role") + 1 :][:1] == [role])
            or re.search(r'role\s*=\s*\\?"' + role, cms)
        )
        if not told:
            errs.append(f"release {rel}: nothing tells the engine its role {role}")
    if errs:
        raise Fail("\n".join(errs))


def pins(c: Ctx) -> dict[str, tuple[str, str]]:
    """name -> (repository, version) from contracts/helm/observability.md."""
    out = {}
    for line in contract_path(c, "helm/observability.md").read_text().splitlines():
        cells = re.findall(r"`([^`]+)`", line)
        if line.startswith("|") and len(cells) >= 3 and cells[1].startswith("https://"):
            out[cells[0]] = (cells[1], cells[2])
    return out


def test_observability_stack_is_pinned(c: Ctx) -> None:
    # WHY: the course tests against exact upstream chart versions; Chart.yaml
    #      names them without ranges, and Chart.lock records what
    #      `helm dependency update` resolved, so every install is the same and
    #      a newer chart is an explicit upgrade (craft.15, drill ops.06).
    # KIND: conformance
    # CHAPTER: dep.03 section 2.5
    want = pins(c)
    meta = c.yaml_file("deploy/observability/Chart.yaml")[0] or {}
    lock = c.yaml_file("deploy/observability/Chart.lock")[0] or {}
    deps = {d.get("name"): d for d in meta.get("dependencies") or []}
    locked = {d.get("name"): d for d in lock.get("dependencies") or []}
    errs = []
    for name in STACK:
        repo, ver = want[name]
        d, lk = deps.get(name), locked.get(name)
        if d is None:
            errs.append(f"deploy/observability/Chart.yaml has no dependency {name}")
            continue
        if str(d.get("version")) != ver or d.get("repository") != repo:
            errs.append(
                f"{name}: Chart.yaml has {d.get('version')} from {d.get('repository')}; the pin is {ver} from {repo}"
            )
        if lk is None or str(lk.get("version")) != ver or lk.get("repository") != repo:
            errs.append(
                f"{name}: Chart.lock does not record {ver} from {repo}: run helm dependency update"
            )
    if not str(lock.get("digest", "")).startswith("sha256:"):
        errs.append(
            "Chart.lock has no digest: it was not written by helm dependency update"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_prometheus_rule_selector_and_ports(c: Ctx) -> None:
    # WHY: Prometheus loads only the PrometheusRules its rule selector matches;
    #      the course keeps kube-prometheus-stack's default (rules labelled
    #      release: observability), and overriding it makes obs.03's alerts
    #      valid YAML that never fires. Prometheus and Grafana answer on the
    #      NodePorts kind maps (30090, 30300), and Grafana's admin password
    #      comes from a Secret.
    # KIND: unit
    # CHAPTER: dep.03 section 5, Pitfall 5
    vals = c.yaml_file("deploy/observability/values.yaml")[0] or {}
    kps = vals.get("kube-prometheus-stack") or {}
    spec = (kps.get("prometheus") or {}).get("prometheusSpec") or {}
    errs = []
    for k in (
        "ruleSelectorNilUsesHelmValues",
        "serviceMonitorSelectorNilUsesHelmValues",
    ):
        if spec.get(k) is False:
            errs.append(
                f"prometheusSpec.{k} is false: Prometheus would ignore the release: observability label"
            )
    for k in ("ruleSelector", "serviceMonitorSelector"):
        if spec.get(k):
            errs.append(
                f"prometheusSpec.{k} is overridden: keep the default selector (contracts/helm/observability.md)"
            )
    if (((kps.get("prometheus") or {}).get("service") or {}).get("nodePort")) != 30090:
        errs.append("kube-prometheus-stack.prometheus.service.nodePort is not 30090")
    graf = kps.get("grafana") or {}
    if ((graf.get("service") or {}).get("nodePort")) != 30300:
        errs.append("kube-prometheus-stack.grafana.service.nodePort is not 30300")
    if graf.get("adminPassword") or not (
        (graf.get("admin") or {}).get("existingSecret")
    ):
        errs.append(
            "Grafana's admin password must come from admin.existingSecret, never adminPassword"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_system_toml_prometheus(c: Ctx) -> None:
    # WHY: MS-prod's promql steps and every drill query Prometheus through
    #      [deploy].prometheus, the NodePort kind maps to your machine.
    # KIND: unit
    d = c.system().get("deploy") or {}
    if str(d.get("prometheus", "")).rstrip("/") != "http://127.0.0.1:30090":
        raise Fail(
            f"[deploy].prometheus is {d.get('prometheus')!r}; want 'http://127.0.0.1:30090'"
        )


# -- cluster ------------------------------------------------------------------


def context(c: Ctx) -> tuple[str, str]:
    d = c.system().get("deploy") or {}
    ctx = d.get("kube_context") or f"kind-{c.system_name()}"
    c.need_cluster(ctx)
    return ctx, d.get("namespace") or c.system_name()


def test_server_side_dry_run(c: Ctx) -> None:
    # WHY: the API server validates what `helm template` cannot: field names
    #      against the live schema, PodDisruptionBudget and PVC admission, and
    #      the namespace.
    # KIND: conformance
    # CHAPTER: dep.03 section 4, The artifact and its check
    ctx, ns = context(c)
    n = c.system_name()
    for part, rel, sets in (
        ("gateway", f"{n}-gateway", []),
        ("engine", f"{n}-engine", []),
        ("engine", f"{n}-engine-decode", ["--set", "role=decode"]),
    ):
        out = c.sh(
            ["helm", "template", rel, chart(c, part), "-n", ns, *sets], timeout=60
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


def test_releases_available(c: Ctx) -> None:
    # WHY: installed, not just renderable: the gateway and the unified engine
    #      are available, and the observability release is deployed.
    # KIND: conformance
    ctx, ns = context(c)
    n = c.system_name()
    errs = []
    for name in (f"{n}-gateway", f"{n}-engine"):
        r = c.sh(
            [
                "kubectl",
                "--context",
                ctx,
                "-n",
                ns,
                "get",
                "deploy",
                name,
                "-o",
                "json",
            ],
            check=False,
        )
        if (
            r.returncode != 0
            or int(
                (json.loads(r.stdout).get("status") or {}).get("availableReplicas") or 0
            )
            < 1
        ):
            errs.append(f"deploy/{name} is not available in {ns}")
    r = c.sh(
        [
            "helm",
            "--kube-context",
            ctx,
            "-n",
            "observability",
            "status",
            "observability",
            "-o",
            "json",
        ],
        check=False,
    )
    if (
        r.returncode != 0
        or (json.loads(r.stdout).get("info") or {}).get("status") != "deployed"
    ):
        errs.append(
            "helm release observability (namespace observability) is not deployed"
        )
    if errs:
        raise Fail("\n".join(errs))


if __name__ == "__main__":
    raise SystemExit(run(globals()))
