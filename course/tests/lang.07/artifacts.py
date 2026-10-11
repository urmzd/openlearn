"""lang.07 artifact check: primers/lang.07 builds an image, renders a chart, and serves on kind.

Run by `ol check lang.07` in your repo. Three tiers, in order:

  static   files, Dockerfile rules, kind config, `helm lint`, the rendered chart
  docker   the image builds, runs as a numeric non-root user, serves, stops on SIGTERM
  cluster  on kube context kind-lang07: the NodePort Service has ready endpoints
           and http://127.0.0.1:31007/ answers from one of its pods

The cluster tier fails when kind-lang07 is missing or down. With OL_SMOKE=1 it
is skipped with the reason instead (everything else still runs).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import (
    Ctx,
    Fail,
    check_dockerfile,
    containers,
    image_tag_problem,
    one,
    pod_spec,
    run,
)  # noqa: E402

DIR = "primers/lang.07"
CHART = f"{DIR}/chart"
CONTEXT = "kind-lang07"
NODEPORT = 31007
IMAGE = "ol-check/lang07:local"
PROBE = "ol-probe-lang07"


def _rendered(c: Ctx, sets: dict | None = None) -> list:
    return c.helm_template(CHART, "hello", sets)


def _container_port(ctr: dict, ref) -> int | None:
    """The container port a Service targetPort (a number or a port name) resolves to."""
    for p in ctr.get("ports") or []:
        if ref == p.get("name") or str(ref) == str(p.get("containerPort")):
            return int(p["containerPort"])
    return int(ref) if str(ref).isdigit() else None


# -- static -------------------------------------------------------------------


def test_files_present(c: Ctx) -> None:
    # WHY: the check and the chapter agree on one layout; a missing file is reported
    #      by name instead of as a confusing failure three steps later.
    # KIND: unit
    # CHAPTER: lang.07 section 4, The interface
    for rel in (
        "main.go",
        "go.mod",
        "Dockerfile",
        "kind.yaml",
        "chart/Chart.yaml",
        "chart/values.yaml",
    ):
        c.require_file(f"{DIR}/{rel}")


def test_dockerfile_multi_stage_pinned_non_root(c: Ctx) -> None:
    # WHY: a build stage plus a small runtime stage keeps the toolchain out of the
    #      image; a pinned tag makes the build repeatable; a numeric non-root USER is
    #      what Kubernetes runAsNonRoot can verify; exec-form ENTRYPOINT makes your
    #      server PID 1 so it receives SIGTERM. dep.00 applies the same rules to the
    #      engine and gateway images.
    # KIND: unit
    # CHAPTER: lang.07 section 5, Pitfalls 1 to 3
    errs = check_dockerfile(
        c.require_file(f"{DIR}/Dockerfile").read_text(),
        f"{DIR}/Dockerfile",
        port="8080",
    )
    if errs:
        raise Fail("\n".join(errs))


def test_kind_config_maps_the_nodeport(c: Ctx) -> None:
    # WHY: kind's node is a container, so NodePort 31007 is reachable from your
    #      machine only through an extraPortMappings entry, and only if it was there
    #      when the cluster was created.
    # KIND: unit
    # CHAPTER: lang.07 section 5, Pitfall 5
    docs = c.yaml_file(f"{DIR}/kind.yaml")
    cfg = docs[0] if docs else {}
    if cfg.get("kind") != "Cluster" or not str(cfg.get("apiVersion", "")).startswith(
        "kind.x-k8s.io/"
    ):
        raise Fail(
            f"{DIR}/kind.yaml must be a kind Cluster config (kind: Cluster, apiVersion: kind.x-k8s.io/v1alpha4)"
        )
    if cfg.get("name") != "lang07":
        raise Fail(
            f"{DIR}/kind.yaml must name the cluster lang07 (the check looks for context {CONTEXT})"
        )
    maps = [m for n in cfg.get("nodes") or [] for m in n.get("extraPortMappings") or []]
    if not any(
        int(m.get("containerPort", 0)) == NODEPORT
        and int(m.get("hostPort", 0)) == NODEPORT
        for m in maps
    ):
        raise Fail(
            f"no node maps containerPort {NODEPORT} to hostPort {NODEPORT} (found {maps or 'none'})"
        )


def test_chart_lints(c: Ctx) -> None:
    # WHY: `helm lint` catches template and Chart.yaml errors before a cluster sees them.
    # KIND: conformance
    c.sh(["helm", "lint", CHART], timeout=60)


def test_chart_renders_deployment_and_nodeport_service(c: Ctx) -> None:
    # WHY: a Service finds pods only through its selector, and reaches them only
    #      through a targetPort the container really listens on; a mismatch renders
    #      fine and serves nothing (no endpoints).
    # KIND: unit
    # CHAPTER: lang.07 section 5, Pitfall 6
    docs = _rendered(c)
    dep = one(docs, "Deployment", CHART)
    svc = one(docs, "Service", CHART)
    spec = svc.get("spec") or {}
    if spec.get("type") != "NodePort":
        raise Fail(f"the Service is type {spec.get('type', 'ClusterIP')}, not NodePort")
    ports = spec.get("ports") or []
    hit = [p for p in ports if int(p.get("nodePort", 0)) == NODEPORT]
    if not hit:
        raise Fail(f"no Service port has nodePort {NODEPORT} (ports: {ports})")
    labels = ((dep.get("spec") or {}).get("template") or {}).get("metadata", {}).get(
        "labels"
    ) or {}
    sel = spec.get("selector") or {}
    if not sel or any(labels.get(k) != v for k, v in sel.items()):
        raise Fail(f"Service selector {sel} does not match the pod labels {labels}")
    ctrs = containers(dep)
    if not ctrs:
        raise Fail("the Deployment has no containers")
    target = hit[0].get("targetPort", hit[0].get("port"))
    if _container_port(ctrs[0], target) != 8080:
        raise Fail(f"targetPort {target!r} does not resolve to container port 8080")


def test_pod_spec_is_production_shaped(c: Ctx) -> None:
    # WHY: readiness gates traffic, limits bound a runaway pod, runAsNonRoot makes
    #      the kubelet refuse a root image, and a pinned tag with a non-Always pull
    #      policy is what lets kind run an image you loaded instead of pulling one.
    # KIND: unit
    # CHAPTER: lang.07 section 5, Pitfalls 3 and 4
    dep = one(_rendered(c), "Deployment", CHART)
    ctr = containers(dep)[0]
    errs = []
    probe = ctr.get("readinessProbe") or {}
    if (probe.get("httpGet") or {}).get("path") != "/healthz":
        errs.append("readinessProbe must be an httpGet on /healthz")
    if not ((ctr.get("resources") or {}).get("limits") or {}):
        errs.append("resources.limits is not set")
    pod_sc = pod_spec(dep).get("securityContext") or {}
    ctr_sc = ctr.get("securityContext") or {}
    if not (ctr_sc.get("runAsNonRoot", pod_sc.get("runAsNonRoot"))):
        errs.append("runAsNonRoot: true is not set (pod or container securityContext)")
    image = str(ctr.get("image", ""))
    why = image_tag_problem(image)
    if why:
        errs.append(why)
    policy = ctr.get("imagePullPolicy", "Always" if why else "IfNotPresent")
    if policy == "Always":
        errs.append(
            "imagePullPolicy is Always: kind would try to pull your local image from a registry"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_values_reach_the_pod(c: Ctx) -> None:
    # WHY: a chart is a function from values to manifests; `greeting` must flow
    #      from values.yaml (or --set) into the container's GREETING variable.
    # KIND: unit
    dep = one(_rendered(c, {"greeting": PROBE}), "Deployment", CHART)
    env = {e.get("name"): e.get("value") for e in containers(dep)[0].get("env") or []}
    if env.get("GREETING") != PROBE:
        raise Fail(f"--set greeting={PROBE} rendered GREETING={env.get('GREETING')!r}")


# -- docker -------------------------------------------------------------------


def _image(c: Ctx) -> str:
    if "image" not in c.cache:
        c.need_docker()
        c.cache["image"] = None
        c.sh(["docker", "build", "-q", "-t", IMAGE, DIR], timeout=600)
        c.cache["image"] = IMAGE
    if c.cache["image"] is None:
        raise Fail("the image did not build (see test_image_builds)")
    return c.cache["image"]


def test_image_builds(c: Ctx) -> None:
    # WHY: the Dockerfile must build from a clean context with only the files it
    #      copies; a build that works only from a dirty cache fails in CI.
    # KIND: unit
    _image(c)


def test_image_user_is_numeric_non_root(c: Ctx) -> None:
    # WHY: what the image metadata says is what the kubelet checks; a name such as
    #      `app` cannot be verified as non-root and the pod never starts.
    # KIND: boundary
    # CHAPTER: lang.07 section 5, Pitfall 3
    user = c.sh(
        ["docker", "image", "inspect", "-f", "{{.Config.User}}", _image(c)]
    ).stdout.strip()
    uid = user.split(":")[0]
    if not uid.isdigit() or uid == "0":
        raise Fail(
            f"image user is {user or '(empty, so root)'}; want a numeric uid other than 0"
        )


def _container(c: Ctx) -> tuple[str, int]:
    if "ctr" not in c.cache:
        img = _image(c)
        name = f"ol-check-lang07-{int(time.time() * 1000) % 10**9}"
        c.cache["ctr"] = None
        c.sh(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-e",
                f"GREETING={PROBE}",
                "-p",
                "127.0.0.1::8080",
                img,
            ]
        )
        c.cleanups.append(lambda: c.sh(["docker", "rm", "-f", name], check=False))
        out = c.sh(["docker", "port", name, "8080/tcp"]).stdout.strip().splitlines()
        port = int(out[0].rsplit(":", 1)[1])
        c.cache["ctr"] = (name, port)
    if c.cache["ctr"] is None:
        raise Fail(
            "the container did not start (see test_container_serves_health_and_greeting)"
        )
    return c.cache["ctr"]


def test_container_serves_health_and_greeting(c: Ctx) -> None:
    # WHY: the image's ENTRYPOINT must start the server on 8080 and read its config
    #      from the environment (twelve-factor), which is how the chart configures it.
    # KIND: unit
    name, port = _container(c)
    c.wait_http(f"http://127.0.0.1:{port}/healthz", timeout=15)
    status, _, body = c.http("GET", f"http://127.0.0.1:{port}/")
    if status != 200:
        raise Fail(f"GET / answered {status}")
    try:
        doc = json.loads(body)
    except json.JSONDecodeError:
        raise Fail(f"GET / is not JSON: {body[:200]!r}") from None
    if doc.get("greeting") != PROBE:
        raise Fail(
            f"greeting is {doc.get('greeting')!r}; the container ran with GREETING={PROBE}"
        )
    host = c.sh(
        ["docker", "inspect", "-f", "{{.Config.Hostname}}", name]
    ).stdout.strip()
    if doc.get("pod") != host:
        raise Fail(f"pod is {doc.get('pod')!r}; want the container hostname {host!r}")


def test_container_exits_zero_on_sigterm(c: Ctx) -> None:
    # WHY: `docker stop` and every Kubernetes rollout send SIGTERM, wait, then
    #      SIGKILL. Without its own handler a Go server dies mid-request with exit 2;
    #      one hidden behind a shell as PID 1 never sees SIGTERM and is killed after
    #      the grace period with exit 137. Either way requests are dropped.
    # KIND: fault
    # CHAPTER: lang.07 section 5, Pitfall 2
    name, _ = _container(c)
    t0 = time.monotonic()
    c.sh(["docker", "stop", "-t", "5", name], timeout=30)
    took = time.monotonic() - t0
    code = c.sh(["docker", "inspect", "-f", "{{.State.ExitCode}}", name]).stdout.strip()
    if code != "0" or took >= 4.5:
        raise Fail(
            f"after SIGTERM the container exited {code} in {took:.1f}s; want exit 0 well inside the 5 s grace period"
        )


# -- cluster --------------------------------------------------------------------


def _service(c: Ctx) -> dict:
    if "svc" not in c.cache:
        c.need_cluster(CONTEXT)
        svcs = c.kubectl_json(CONTEXT, ["get", "services", "-A"])["items"]
        hit = [
            s
            for s in svcs
            if any(
                int(p.get("nodePort", 0)) == NODEPORT
                for p in s["spec"].get("ports") or []
            )
        ]
        if not hit:
            raise Fail(
                f"no Service on {CONTEXT} has nodePort {NODEPORT}: helm install your chart (chapter section 4)"
            )
        c.cache["svc"] = hit[0]
    return c.cache["svc"]


def test_service_has_ready_endpoints_on_kind(c: Ctx) -> None:
    # WHY: a NodePort with zero ready endpoints accepts connections and resets
    #      them; ready endpoints mean the image was loaded, the pod passed its
    #      readiness probe, and the selector matched.
    # KIND: conformance
    svc = _service(c)
    ns, name = svc["metadata"]["namespace"], svc["metadata"]["name"]
    slices = c.kubectl_json(
        CONTEXT,
        ["get", "endpointslices", "-n", ns, "-l", f"kubernetes.io/service-name={name}"],
    )
    ready = [
        e
        for s in slices["items"]
        for e in s.get("endpoints") or []
        if (e.get("conditions") or {}).get("ready")
    ]
    if not ready:
        raise Fail(
            f"Service {ns}/{name} has no ready endpoints: "
            f"`kubectl --context {CONTEXT} -n {ns} get pods` and `describe pod` say why"
        )


def test_nodeport_answers_from_a_pod(c: Ctx) -> None:
    # WHY: the end-to-end path: your machine, the kind port mapping, the NodePort,
    #      kube-proxy, and one of the Deployment's pods, whose name comes back.
    # KIND: conformance
    svc = _service(c)
    ns = svc["metadata"]["namespace"]
    status, _, body = c.http("GET", f"http://127.0.0.1:{NODEPORT}/", timeout=10)
    if status != 200:
        raise Fail(f"GET http://127.0.0.1:{NODEPORT}/ answered {status}")
    doc = json.loads(body)
    pods = {
        p["metadata"]["name"]
        for p in c.kubectl_json(CONTEXT, ["get", "pods", "-n", ns])["items"]
        if p.get("status", {}).get("phase") == "Running"
    }
    if doc.get("pod") not in pods:
        raise Fail(
            f"answer came from {doc.get('pod')!r}, which is not a running pod in {ns} ({sorted(pods)})"
        )
    if not doc.get("greeting"):
        raise Fail("the greeting is empty")


if __name__ == "__main__":
    raise SystemExit(run(globals()))
