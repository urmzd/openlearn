"""dep.02 artifact check: the Pass 7 kind cluster with a local registry.

Run by `ol check dep.02` in your repo. Two tiers, in order:

  static   deploy/kind/cluster.yaml maps every platform NodePort on 127.0.0.1,
           mounts ./artifacts, and points containerd at /etc/containerd/certs.d;
           the local-registry-hosting ConfigMap; deploy/kind/up.sh; [deploy]
  cluster  on [deploy].kube_context: nodes Ready, the node container publishes
           the ports (the cluster was recreated), the kind-registry container
           runs on 127.0.0.1:5001 and joins the kind network, the ConfigMap is
           applied, and an image pushed to localhost:5001 runs in a pod

The cluster tier fails when the context is missing or down; with OL_SMOKE=1
it is skipped with the reason instead (kind is not installed on every machine
that runs the course's own verification).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import Ctx, Fail, run  # noqa: E402

PORTS = {
    30080: "gateway",
    30686: "Jaeger query",
    30090: "Prometheus",
    30300: "Grafana",
    30320: "Tempo query",
    30733: "durable gRPC",
}
REGISTRY = "localhost:5001"
REG_NAME = "kind-registry"


def cluster_cfg(c: Ctx) -> dict:
    docs = c.yaml_file("deploy/kind/cluster.yaml")
    cfg = docs[0] if docs else {}
    if cfg.get("kind") != "Cluster":
        raise Fail("deploy/kind/cluster.yaml is not a kind Cluster config")
    return cfg


# -- static -------------------------------------------------------------------


def test_layout_present(c: Ctx) -> None:
    # WHY: three files make the cluster reproducible: its config, the script
    #      that creates it with the registry, and the registry's ConfigMap.
    # KIND: unit
    # CHAPTER: dep.02 section 4, The artifact and its check
    for rel in (
        "deploy/kind/cluster.yaml",
        "deploy/kind/up.sh",
        "deploy/kind/local-registry-hosting.yaml",
    ):
        c.require_file(rel)


def test_cluster_maps_every_platform_port(c: Ctx) -> None:
    # WHY: kind reaches a NodePort from your machine only through an
    #      extraPortMapping, and mappings are fixed when the node is created:
    #      Pass 7 needs the gateway (30080), Jaeger (30686), Prometheus
    #      (30090), Grafana (30300), Tempo (30320), and Pass 8's durable server
    #      (30733) now, each on 127.0.0.1 so nothing is exposed to your network.
    # KIND: unit
    # CHAPTER: dep.02 section 3, Worked example by hand
    cfg = cluster_cfg(c)
    errs = []
    if cfg.get("name") != c.system_name():
        errs.append(
            f"the cluster name is {cfg.get('name')!r}; want {c.system_name()!r} (context kind-{c.system_name()})"
        )
    maps = {}
    for n in cfg.get("nodes") or []:
        for m in n.get("extraPortMappings") or []:
            maps[int(m.get("containerPort", 0))] = m
    for port, what in PORTS.items():
        m = maps.get(port)
        if m is None or int(m.get("hostPort", 0)) != port:
            errs.append(
                f"no mapping of containerPort {port} to hostPort {port} ({what})"
            )
        elif m.get("listenAddress") != "127.0.0.1":
            errs.append(
                f"port {port} listens on {m.get('listenAddress', '0.0.0.0 (the default)')}; want 127.0.0.1"
            )
    mounts = [m for n in cfg.get("nodes") or [] for m in n.get("extraMounts") or []]
    if not any(m.get("containerPath") == "/artifacts" for m in mounts):
        errs.append("no extraMounts entry has containerPath /artifacts")
    if errs:
        raise Fail("\n".join(errs))


def test_containerd_reads_registry_hosts(c: Ctx) -> None:
    # WHY: inside the node, "localhost:5001" is the node itself, not your
    #      machine. containerd must look up per-registry hosts in
    #      /etc/containerd/certs.d, where up.sh maps localhost:5001 to the
    #      kind-registry container (the old registry.mirrors table is gone in
    #      containerd 2).
    # KIND: unit
    # CHAPTER: dep.02 section 2.2
    patches = "\n".join(cluster_cfg(c).get("containerdConfigPatches") or [])
    if not re.search(r'config_path\s*=\s*"/etc/containerd/certs\.d"', patches):
        raise Fail(
            'containerdConfigPatches must set config_path = "/etc/containerd/certs.d" under '
            '[plugins."io.containerd.grpc.v1.cri".registry]'
        )


def test_registry_hosting_configmap(c: Ctx) -> None:
    # WHY: KEP-1755: tools (Tilt in dep.04, ko, skaffold) find the cluster's
    #      registry in ConfigMap kube-public/local-registry-hosting.
    # KIND: unit
    # CHAPTER: dep.02 section 2.2
    docs = c.yaml_file("deploy/kind/local-registry-hosting.yaml")
    cm = next((d for d in docs if d.get("kind") == "ConfigMap"), None)
    if cm is None:
        raise Fail("deploy/kind/local-registry-hosting.yaml holds no ConfigMap")
    meta = cm.get("metadata") or {}
    if (
        meta.get("name") != "local-registry-hosting"
        or meta.get("namespace") != "kube-public"
    ):
        raise Fail(
            f"the ConfigMap is {meta.get('namespace')}/{meta.get('name')}; want kube-public/local-registry-hosting"
        )
    body = (cm.get("data") or {}).get("localRegistryHosting.v1")
    hosting = c.yaml_docs(body or "", "localRegistryHosting.v1")
    if not hosting or (hosting[0] or {}).get("host") != REGISTRY:
        raise Fail(f'data.localRegistryHosting.v1 must say host: "{REGISTRY}"')


def test_up_script(c: Ctx) -> None:
    # WHY: the cluster is recreated more than once in the course; a script
    #      makes it the same every time: the registry container on
    #      127.0.0.1:5001 from a pinned image, the cluster from cluster.yaml,
    #      hosts.toml on every node, the registry on the kind network, and the
    #      ConfigMap.
    # KIND: unit
    # CHAPTER: dep.02 section 4, The artifact and its check
    p = c.require_file("deploy/kind/up.sh")
    if not os.access(p, os.X_OK):
        raise Fail("deploy/kind/up.sh is not executable (chmod +x)")
    c.sh(["bash", "-n", str(p)], timeout=10)
    text = p.read_text()
    errs = []
    for needle, why in (
        (
            "deploy/kind/cluster.yaml",
            "creates the cluster from deploy/kind/cluster.yaml",
        ),
        (
            "certs.d/localhost:",
            "writes /etc/containerd/certs.d/localhost:5001/hosts.toml on each node",
        ),
        ("hosts.toml", "writes hosts.toml"),
        (
            "docker network connect",
            "connects the registry container to the kind network",
        ),
        ("local-registry-hosting", "applies the local-registry-hosting ConfigMap"),
        ("127.0.0.1:", "binds the registry to 127.0.0.1"),
    ):
        if needle not in text:
            errs.append(f"up.sh never {why}")
    m = re.search(r"\bregistry:[^\s\"']*", text)
    if not m:
        errs.append("up.sh runs no registry image (registry:<version>@sha256:...)")
    elif "@sha256:" not in m.group(0) and not re.match(
        r"registry:\d+\.\d+\.\d+", m.group(0)
    ):
        errs.append(f"up.sh runs {m.group(0)}: pin a release (and its digest)")
    if errs:
        raise Fail("\n".join(errs))


def test_system_toml_deploy_section(c: Ctx) -> None:
    # WHY: every kind step, drill, and MS-prod reaches the cluster through
    #      [deploy]: the recreated cluster keeps the name, so the context and
    #      namespace stay what dep.00 declared.
    # KIND: unit
    n = c.system_name()
    d = c.system().get("deploy") or {}
    errs = [
        f"[deploy].{k} is {d.get(k)!r}; want {v!r}"
        for k, v in (("kube_context", f"kind-{n}"), ("namespace", n))
        if d.get(k) != v
    ]
    if errs:
        raise Fail("\n".join(errs))


# -- cluster ------------------------------------------------------------------


def context(c: Ctx) -> str:
    ctx = (c.system().get("deploy") or {}).get(
        "kube_context"
    ) or f"kind-{c.system_name()}"
    c.need_cluster(ctx)
    return ctx


def test_nodes_ready(c: Ctx) -> None:
    # WHY: a node that is not Ready schedules nothing; this is the first thing
    #      to look at when a fresh cluster misbehaves.
    # KIND: conformance
    ctx = context(c)
    nodes = c.kubectl_json(ctx, ["get", "nodes"]).get("items") or []
    bad = [
        n["metadata"]["name"]
        for n in nodes
        if not any(
            x.get("type") == "Ready" and x.get("status") == "True"
            for x in n["status"].get("conditions") or []
        )
    ]
    if not nodes or bad:
        raise Fail(f"nodes not Ready: {bad or 'no nodes'}")


def test_node_publishes_the_ports(c: Ctx) -> None:
    # WHY: editing cluster.yaml changes nothing until the cluster is
    #      recreated; the node container's published ports show which config
    #      the running cluster was created from.
    # KIND: conformance
    # CHAPTER: dep.02 section 5, Pitfall 1
    context(c)
    node = f"{c.system_name()}-control-plane"
    info = json.loads(c.sh(["docker", "inspect", node]).stdout)[0]
    bind = (info.get("HostConfig") or {}).get("PortBindings") or {}
    missing = [
        p
        for p in PORTS
        if not any(b.get("HostPort") == str(p) for b in bind.get(f"{p}/tcp") or [])
    ]
    if missing:
        raise Fail(
            f"{node} publishes no host port for {missing}: delete the cluster and rerun deploy/kind/up.sh"
        )


def test_registry_runs_on_the_kind_network(c: Ctx) -> None:
    # WHY: the registry container must answer on 127.0.0.1:5001 (where you
    #      push) and sit on the kind network (where nodes pull it by name).
    # KIND: conformance
    # CHAPTER: dep.02 section 2.2
    context(c)
    r = c.sh(["docker", "inspect", REG_NAME], check=False)
    if r.returncode != 0:
        raise Fail(f"no container {REG_NAME}: run deploy/kind/up.sh")
    info = json.loads(r.stdout)[0]
    errs = []
    if not (info.get("State") or {}).get("Running"):
        errs.append(f"{REG_NAME} is not running")
    binds = ((info.get("HostConfig") or {}).get("PortBindings") or {}).get(
        "5000/tcp"
    ) or []
    if not any(
        b.get("HostPort") == "5001" and b.get("HostIp") in ("127.0.0.1", "")
        for b in binds
    ):
        errs.append(f"{REG_NAME} does not publish 5000 on host port 5001")
    if "kind" not in ((info.get("NetworkSettings") or {}).get("Networks") or {}):
        errs.append(
            f"{REG_NAME} is not on the kind network: docker network connect kind {REG_NAME}"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_registry_hosting_configmap_applied(c: Ctx) -> None:
    # WHY: the ConfigMap file only helps once it is in the cluster.
    # KIND: conformance
    ctx = context(c)
    r = c.sh(
        [
            "kubectl",
            "--context",
            ctx,
            "-n",
            "kube-public",
            "get",
            "configmap",
            "local-registry-hosting",
        ],
        check=False,
    )
    if r.returncode != 0:
        raise Fail(
            "kube-public/local-registry-hosting is not applied: kubectl apply -f deploy/kind/local-registry-hosting.yaml"
        )


def test_image_pushed_to_the_registry_runs(c: Ctx) -> None:
    # WHY: the whole point: an image pushed to localhost:5001 from your
    #      machine is pulled by the node and runs in a pod, with no
    #      `kind load` in between.
    # KIND: conformance
    # CHAPTER: dep.02 section 3, Worked example by hand
    ctx = context(c)
    ns = (c.system().get("deploy") or {}).get("namespace") or c.system_name()
    src = "alpine:3.19@sha256:6baf43584bcb78f2e5847d1de515f23499913ac9f12bdf834811a3145eb11ca1"
    dst = f"{REGISTRY}/ol-check/alpine:dep02"
    c.sh(["docker", "pull", src], timeout=300)
    c.sh(["docker", "tag", src, dst])
    c.sh(["docker", "push", dst], timeout=300)
    pod = f"ol-dep02-pull-{os.getpid()}"
    c.sh(
        [
            "kubectl",
            "--context",
            ctx,
            "-n",
            ns,
            "run",
            pod,
            f"--image={dst}",
            "--restart=Never",
            "--image-pull-policy=Always",
            "--command",
            "--",
            "true",
        ],
        timeout=60,
    )
    c.cleanups.append(
        lambda: c.sh(
            [
                "kubectl",
                "--context",
                ctx,
                "-n",
                ns,
                "delete",
                "pod",
                pod,
                "--wait=false",
            ],
            check=False,
        )
    )
    deadline = time.monotonic() + 120
    phase, why = "", ""
    while time.monotonic() < deadline:
        st = c.kubectl_json(ctx, ["-n", ns, "get", "pod", pod]).get("status") or {}
        phase = st.get("phase", "")
        waiting = ((st.get("containerStatuses") or [{}])[0].get("state") or {}).get(
            "waiting"
        ) or {}
        why = waiting.get("reason", "")
        if phase in ("Succeeded", "Running") or why in (
            "ErrImagePull",
            "ImagePullBackOff",
            "InvalidImageName",
        ):
            break
        time.sleep(2)
    if phase not in ("Succeeded", "Running"):
        raise Fail(
            f"pod {pod} with {dst}: phase {phase or 'unknown'} {why}: the node cannot pull from {REGISTRY}"
        )


if __name__ == "__main__":
    raise SystemExit(run(globals()))
