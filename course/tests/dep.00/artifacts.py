"""dep.00 artifact check: your engine and gateway as images, charts, and a release on kind.

Run by `ol check dep.00` in your repo. Three tiers, in order:

  static   files, the image rules, the kind config, `helm lint`, the rendered charts,
           the Jaeger manifest, and the [deploy] section of system.toml
  docker   both images build, run as numeric non-root users, stream a completion
           engine -> gateway on a Docker network, reject a missing key, and exit 0
           on SIGTERM
  cluster  on [deploy].kube_context: both Deployments available, a stream through
           [deploy].gateway_url, and the Jaeger query API at [deploy].traces

`<system>` below is [system].name from system.toml. The cluster tier fails when the
context is missing or down; with OL_SMOKE=1 it is skipped with the reason instead.
"""

from __future__ import annotations

import json
import os
import re
import shutil
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
    objects,
    one,  # noqa: E402
    pod_spec,
    run,
)

PROBE_KEY = "tl_check_dep00probe"
# A fixed training text for the docker tier: any bytes give a valid bigram.
CORPUS = (
    "Once upon a time there was a little engine. The engine read the bytes and "
    "the gateway checked the key. Once the key was good, the tokens streamed.\n"
) * 8


def _name(c: Ctx) -> str:
    return c.system_name()


def _chart(c: Ctx, part: str) -> str:
    return f"deploy/helm/{_name(c)}-{part}"


def _render(c: Ctx, part: str) -> list:
    n = _name(c)
    return c.helm_template(_chart(c, part), f"{n}-{part}", namespace=n)


def _deployment(c: Ctx, part: str) -> dict:
    n = _name(c)
    dep = one(_render(c, part), "Deployment", _chart(c, part))
    if dep["metadata"].get("name") != f"{n}-{part}":
        raise Fail(
            f"{_chart(c, part)} names its Deployment {dep['metadata'].get('name')!r}; "
            f"want {n}-{part} (ops.00 and [deploy].services find it by that name)"
        )
    return dep


def _args(ctr: dict) -> list[str]:
    return [str(a) for a in (ctr.get("command") or []) + (ctr.get("args") or [])]


def _flag(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _env(ctr: dict) -> dict:
    return {e.get("name"): e for e in ctr.get("env") or []}


def _probe_port(ctr: dict, probe: dict) -> int | None:
    port = (probe.get("httpGet") or {}).get("port")
    for p in ctr.get("ports") or []:
        if port == p.get("name"):
            return int(p["containerPort"])
    return int(port) if str(port).isdigit() else None


def _service_errors(
    c: Ctx, part: str, dep: dict, port: int, node_port: int | None
) -> list[str]:
    errs = []
    svcs = objects(_render(c, part), "Service")
    labels = (dep["spec"]["template"].get("metadata") or {}).get("labels") or {}
    want = f"{_name(c)}-{part}"
    svc = next((s for s in svcs if s["metadata"].get("name") == want), None)
    if svc is None:
        return [f"{_chart(c, part)} renders no Service named {want}"]
    sel = svc["spec"].get("selector") or {}
    if not sel or any(labels.get(k) != v for k, v in sel.items()):
        errs.append(
            f"Service {want} selector {sel} does not match the pod labels {labels}"
        )
    ports = svc["spec"].get("ports") or []
    ctr = containers(dep)[0]
    ok = False
    for p in ports:
        tgt = p.get("targetPort", p.get("port"))
        cport = next(
            (
                int(x["containerPort"])
                for x in ctr.get("ports") or []
                if tgt in (x.get("name"), x.get("containerPort"))
            ),
            None,
        )
        if cport is None and str(tgt).isdigit():
            cport = int(tgt)
        if cport == port and (
            node_port is None or int(p.get("nodePort", 0)) == node_port
        ):
            ok = True
    if not ok:
        want_np = f" with nodePort {node_port}" if node_port else ""
        errs.append(
            f"Service {want} has no port targeting container port {port}{want_np} (ports: {ports})"
        )
    if node_port and svc["spec"].get("type") != "NodePort":
        errs.append(
            f"Service {want} is type {svc['spec'].get('type', 'ClusterIP')}, not NodePort"
        )
    return errs


def _common_pod_errors(
    dep: dict, health_port: int, ready_path: str, live_path: str, service: str
) -> list[str]:
    errs = []
    ctr = containers(dep)[0]
    for kind, path in (("readinessProbe", ready_path), ("livenessProbe", live_path)):
        probe = ctr.get(kind) or {}
        if (probe.get("httpGet") or {}).get("path") != path or _probe_port(
            ctr, probe
        ) != health_port:
            errs.append(f"{kind} must be an httpGet on {path} at port {health_port}")
    if not ((ctr.get("resources") or {}).get("limits") or {}):
        errs.append("resources.limits is not set")
    sc = ctr.get("securityContext") or {}
    if not sc.get(
        "runAsNonRoot", (pod_spec(dep).get("securityContext") or {}).get("runAsNonRoot")
    ):
        errs.append("runAsNonRoot: true is not set")
    image = str(ctr.get("image", ""))
    why = image_tag_problem(image)
    if why:
        errs.append(why)
    if ctr.get("imagePullPolicy", "Always" if why else "IfNotPresent") == "Always":
        errs.append(
            "imagePullPolicy is Always: kind would pull instead of using the image you loaded"
        )
    env = _env(ctr)
    if (env.get("OTEL_SERVICE_NAME") or {}).get("value") != service:
        errs.append(
            f"env OTEL_SERVICE_NAME must be {service} (obs.00 and MS-P1 find the trace by it)"
        )
    if not (env.get("OTEL_EXPORTER_OTLP_ENDPOINT") or {}).get("value"):
        errs.append(
            "env OTEL_EXPORTER_OTLP_ENDPOINT is not set (the Jaeger Service, for example http://jaeger:4318)"
        )
    return errs


# -- static -------------------------------------------------------------------


def test_layout_present(c: Ctx) -> None:
    # WHY: the deploy layout is fixed (DESIGN 2.15) because ops.00, MS-P1, and later
    #      dep modules find your files and objects by these names.
    # KIND: unit
    # CHAPTER: dep.00 section 4, The interface
    n = _name(c)
    for rel in (
        "deploy/docker/engine.Dockerfile",
        "deploy/docker/gateway.Dockerfile",
        "deploy/kind/cluster.yaml",
        f"deploy/helm/{n}-engine/Chart.yaml",
        f"deploy/helm/{n}-gateway/Chart.yaml",
        "deploy/observability/jaeger.yaml",
    ):
        c.require_file(rel)


def test_dockerignore_keeps_build_outputs_out(c: Ctx) -> None:
    # WHY: the build context is the repo root, so without a .dockerignore every
    #      `docker build` uploads rust/target (gigabytes), .ol, and artifacts/, and
    #      a COPY of rust/ bakes your laptop's build cache into the image.
    # KIND: unit
    # CHAPTER: dep.00 section 5, Pitfall 1
    lines = {
        ln.strip().strip("/")
        for ln in c.require_file(".dockerignore").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    }
    missing = [
        p
        for p in ("target", ".ol", "artifacts")
        if not ({p, f"**/{p}", f"rust/{p}"} & lines)
    ]
    if missing:
        raise Fail(f".dockerignore does not exclude {', '.join(missing)}")


def test_dockerfiles_follow_the_image_rules(c: Ctx) -> None:
    # WHY: the lang.07 image rules, now on real programs: a build stage the image
    #      never ships, pinned bases, a numeric non-root user, exec form so the
    #      chart's args reach your binary, and the ports the charts target.
    # KIND: unit
    # CHAPTER: dep.00 section 5, Pitfalls 1 and 2
    errs = []
    for part, ports in (("engine", ("8000", "9464")), ("gateway", ("8080", "9464"))):
        rel = f"deploy/docker/{part}.Dockerfile"
        text = c.require_file(rel).read_text()
        for p in ports:
            errs += check_dockerfile(text, rel, port=p)
    if errs:
        raise Fail("\n".join(dict.fromkeys(errs)))


def test_kind_cluster_maps_gateway_jaeger_and_artifacts(c: Ctx) -> None:
    # WHY: the cluster is named after your system (context kind-<system>), the
    #      gateway (30080) and Jaeger query (30686) NodePorts reach your machine only
    #      through mappings fixed at creation, and the engine reads its model from
    #      /artifacts on the node.
    # KIND: unit
    # CHAPTER: dep.00 section 5, Pitfall 4
    docs = c.yaml_file("deploy/kind/cluster.yaml")
    cfg = docs[0] if docs else {}
    errs = []
    if cfg.get("kind") != "Cluster":
        errs.append("deploy/kind/cluster.yaml is not a kind Cluster config")
    if cfg.get("name") != _name(c):
        errs.append(
            f"the cluster name is {cfg.get('name')!r}; want {_name(c)!r} so the context is kind-{_name(c)}"
        )
    nodes = cfg.get("nodes") or []
    maps = {
        (int(m.get("containerPort", 0)), int(m.get("hostPort", 0)))
        for n in nodes
        for m in n.get("extraPortMappings") or []
    }
    for port in (30080, 30686):
        if (port, port) not in maps:
            errs.append(f"no node maps containerPort {port} to hostPort {port}")
    mounts = [m for n in nodes for m in n.get("extraMounts") or []]
    if not any(m.get("containerPath") == "/artifacts" for m in mounts):
        errs.append("no extraMounts entry has containerPath /artifacts")
    if errs:
        raise Fail("\n".join(errs))


def test_charts_lint(c: Ctx) -> None:
    # WHY: template and Chart.yaml errors surface here, not halfway through an install.
    # KIND: conformance
    c.sh(["helm", "lint", _chart(c, "engine"), _chart(c, "gateway")], timeout=60)


def test_engine_chart_renders(c: Ctx) -> None:
    # WHY: the engine Deployment runs the tracer form of spec/cli-roles.md with its
    #      model under /artifacts, listens on 8000 and 9464 (DESIGN 2.13), is ready
    #      only when /healthz answers, and exports spans to Jaeger.
    # KIND: unit
    # CHAPTER: dep.00 section 5, Pitfall 5
    n = _name(c)
    dep = _deployment(c, "engine")
    ctr = containers(dep)[0]
    args = _args(ctr)
    errs = []
    model = _flag(args, "--model-dir") or ""
    if not model.startswith("/artifacts/"):
        errs.append(
            f"--model-dir is {model or 'missing'}; want a directory under /artifacts/"
        )
    if _flag(args, "--port") != "8000":
        errs.append(f"--port is {_flag(args, '--port')}; want 8000")
    if _flag(args, "--health-port") != "9464":
        errs.append(f"--health-port is {_flag(args, '--health-port')}; want 9464")
    vols = {v["name"]: v for v in pod_spec(dep).get("volumes") or []}
    mounts = {m.get("mountPath"): m.get("name") for m in ctr.get("volumeMounts") or []}
    vol = vols.get(mounts.get("/artifacts", ""), {})
    if (vol.get("hostPath") or {}).get("path") != "/artifacts":
        errs.append(
            "/artifacts must be a hostPath volume of the node's /artifacts, mounted at /artifacts"
        )
    errs += _common_pod_errors(dep, 9464, "/healthz", "/healthz", f"{n}-engine")
    errs += _service_errors(c, "engine", dep, 8000, None)
    if errs:
        raise Fail("\n".join(errs))


def test_gateway_chart_renders(c: Ctx) -> None:
    # WHY: the gateway reaches the engine through its Service name, is ready only
    #      when the engine is (/readyz) but restarts only when it is itself stuck
    #      (/healthz), and is the one thing exposed: NodePort 30080.
    # KIND: unit
    # CHAPTER: dep.00 section 5, Pitfalls 3 and 6
    n = _name(c)
    dep = _deployment(c, "gateway")
    args = _args(containers(dep)[0])
    errs = []
    if _flag(args, "--port") != "8080":
        errs.append(f"--port is {_flag(args, '--port')}; want 8080")
    if _flag(args, "--health-port") != "9464":
        errs.append(f"--health-port is {_flag(args, '--health-port')}; want 9464")
    up = _flag(args, "--upstream") or ""
    if not up.startswith((f"http://{n}-engine:8000", f"http://{n}-engine.{n}")):
        errs.append(
            f"--upstream is {up or 'missing'}; want the engine Service, http://{n}-engine:8000"
        )
    errs += _common_pod_errors(dep, 9464, "/readyz", "/healthz", f"{n}-gateway")
    errs += _service_errors(c, "gateway", dep, 8080, 30080)
    if errs:
        raise Fail("\n".join(errs))


def test_api_key_only_from_a_secret(c: Ctx) -> None:
    # WHY: a key written into values.yaml or a manifest lands in git and in
    #      `helm get values`; a secretKeyRef keeps it in one Secret you create by hand.
    # KIND: unit
    # CHAPTER: dep.00 section 5, Pitfall 3
    ctr = containers(_deployment(c, "gateway"))[0]
    env = _env(ctr).get("TL_API_KEY")
    if env is None:
        raise Fail("the gateway container has no TL_API_KEY env entry")
    if "value" in env or not (
        (env.get("valueFrom") or {}).get("secretKeyRef") or {}
    ).get("name"):
        raise Fail(
            "TL_API_KEY must come from valueFrom.secretKeyRef, never a literal value"
        )
    pat = re.compile(r"\btl_[A-Za-z0-9]+_[A-Za-z0-9]{8,}")
    for part in ("engine", "gateway"):
        for p in sorted(c.path(_chart(c, part)).rglob("*")):
            if p.is_file():
                hit = pat.search(p.read_text(errors="replace"))
                if hit:
                    raise Fail(
                        f"{p.relative_to(c.root)} holds what looks like an API key ({hit.group(0)[:8]}...)"
                    )


def test_jaeger_manifest(c: Ctx) -> None:
    # WHY: Jaeger all-in-one receives OTLP/HTTP on 4318 inside the cluster and
    #      serves its query API on NodePort 30686, where obs.00 and the MS-P1
    #      trace step look for your trace.
    # KIND: unit
    docs = c.yaml_file("deploy/observability/jaeger.yaml")
    errs = []
    deps = objects(docs, "Deployment")
    if len(deps) != 1:
        raise Fail(
            f"deploy/observability/jaeger.yaml must hold one Deployment (found {len(deps)})"
        )
    image = str(containers(deps[0])[0].get("image", "")) if containers(deps[0]) else ""
    if "jaeger" not in image:
        errs.append(f"the Deployment runs {image or 'nothing'}, not a Jaeger image")
    why = image_tag_problem(image)
    if why:
        errs.append(why)
    svcs = objects(docs, "Service")
    sp = [(s, p) for s in svcs for p in s["spec"].get("ports") or []]
    if not any(int(p.get("port", 0)) == 4318 for _, p in sp):
        errs.append("no Service port 4318 (OTLP/HTTP)")
    if not any(
        s["spec"].get("type") == "NodePort" and int(p.get("nodePort", 0)) == 30686
        for s, p in sp
    ):
        errs.append("no NodePort Service on 30686 for the query API")
    if errs:
        raise Fail("\n".join(errs))


def test_system_toml_deploy_section(c: Ctx) -> None:
    # WHY: the harness reaches your cluster only through [deploy]: MS-P1's kind steps
    #      use gateway_url and traces, and ops.00 patches [deploy].services.engine
    #      after the drill safety gate checks kube_context and namespace.
    # KIND: unit
    n = _name(c)
    d = c.system().get("deploy") or {}
    want = {
        "kube_context": f"kind-{n}",
        "namespace": n,
        "gateway_url": "http://127.0.0.1:30080",
        "traces": "http://127.0.0.1:30686",
    }
    errs = [
        f"[deploy].{k} is {d.get(k)!r}; want {v!r}"
        for k, v in want.items()
        if str(d.get(k, "")).rstrip("/") != v
    ]
    svcs = d.get("services") or {}
    for part in ("engine", "gateway"):
        if svcs.get(part) != f"deploy/{n}-{part}":
            errs.append(
                f"[deploy].services.{part} is {svcs.get(part)!r}; want 'deploy/{n}-{part}'"
            )
    if errs:
        raise Fail("\n".join(errs))


# -- docker -------------------------------------------------------------------


def _images(c: Ctx) -> dict[str, str]:
    if "images" not in c.cache:
        c.need_docker()
        c.cache["images"] = None
        out = {}
        for part in ("engine", "gateway"):
            tag = f"ol-check/{_name(c)}-{part}:local"
            c.sh(
                [
                    "docker",
                    "build",
                    "-q",
                    "-f",
                    f"deploy/docker/{part}.Dockerfile",
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
    # WHY: each image builds from the repo root with only what its Dockerfile
    #      copies, the way CI and a teammate build it: no target/ or build/ from
    #      your machine leaks in (.dockerignore).
    # KIND: unit
    _images(c)


def test_images_run_as_numeric_non_root(c: Ctx) -> None:
    # WHY: the kubelet enforces runAsNonRoot from the image config's numeric user.
    # KIND: boundary
    for part, tag in _images(c).items():
        user = c.sh(
            ["docker", "image", "inspect", "-f", "{{.Config.User}}", tag]
        ).stdout.strip()
        uid = user.split(":")[0]
        if not uid.isdigit() or uid == "0":
            raise Fail(
                f"the {part} image runs as {user or 'root'}; want a numeric uid other than 0"
            )


def _model(c: Ctx) -> Path:
    """A model directory trained by your own `tinyllm` entry (cli-roles `train bigram`)."""
    if "model" not in c.cache:
        c.cache["model"] = None
        argv = (c.system().get("entry") or {}).get("tinyllm")
        if not argv:
            raise Fail(
                "system.toml has no [entry].tinyllm, which trains the model the engine image serves"
            )
        work = c.path(".ol/check/dep.00")
        shutil.rmtree(work, ignore_errors=True)
        (work / "model").parent.mkdir(parents=True, exist_ok=True)
        (work / "corpus.txt").write_text(CORPUS)
        c.sh(
            [
                *argv,
                "train",
                "bigram",
                "--data",
                str(work / "corpus.txt"),
                "--out",
                str(work / "model"),
            ],
            timeout=300,
        )
        stage = work / "stage" / "artifacts" / "models"
        stage.mkdir(parents=True)
        shutil.copytree(work / "model", stage / "bigram")
        for p in [work / "stage", *(work / "stage").rglob("*")]:
            os.chmod(
                p, 0o755 if p.is_dir() else 0o644
            )  # readable by the image's non-root user
        c.cache["model"] = work / "model"
    if c.cache["model"] is None:
        raise Fail(
            "training the model failed (see test_containers_stream_a_completion)"
        )
    return c.cache["model"]


def _stack(c: Ctx) -> dict:
    """engine and gateway containers on one Docker network, as the charts run them."""
    if "stack" not in c.cache:
        imgs = _images(c)
        model = _model(c)
        c.cache["stack"] = None
        tag = f"ol-dep00-{os.getpid()}"
        c.sh(["docker", "network", "create", tag])
        c.cleanups.append(lambda: c.sh(["docker", "network", "rm", tag], check=False))
        eng, gw = f"{tag}-engine", f"{tag}-gateway"
        c.cleanups.append(lambda: c.sh(["docker", "rm", "-f", eng, gw], check=False))
        # The model is copied in, not bind-mounted: Docker Desktop shares only some
        # host directories, and a missing share shows up as an empty mount.
        c.sh(
            [
                "docker",
                "create",
                "--name",
                eng,
                "--network",
                tag,
                "--network-alias",
                "engine",
                "-p",
                "127.0.0.1::9464",
                imgs["engine"],
                "--model-dir",
                "/artifacts/models/bigram",
                "--port",
                "8000",
                "--health-port",
                "9464",
            ]
        )
        c.sh(["docker", "cp", str(model.parent / "stage" / "artifacts"), f"{eng}:/"])
        c.sh(["docker", "start", eng])
        c.sh(
            [
                "docker",
                "run",
                "-d",
                "--name",
                gw,
                "--network",
                tag,
                "-e",
                f"TL_API_KEY={PROBE_KEY}",
                "-p",
                "127.0.0.1::8080",
                "-p",
                "127.0.0.1::9464",
                imgs["gateway"],
                "--port",
                "8080",
                "--health-port",
                "9464",
                "--upstream",
                "http://engine:8000",
            ]
        )

        def port(name: str, p: int) -> int:
            return int(
                c.sh(["docker", "port", name, f"{p}/tcp"])
                .stdout.splitlines()[0]
                .rsplit(":", 1)[1]
            )

        st = {
            "engine": eng,
            "gateway": gw,
            "api": port(gw, 8080),
            "gw_health": port(gw, 9464),
            "engine_health": port(eng, 9464),
        }
        try:
            c.wait_http(f"http://127.0.0.1:{st['engine_health']}/healthz", timeout=30)
            c.wait_http(f"http://127.0.0.1:{st['gw_health']}/readyz", timeout=30)
        except Fail as e:
            logs = "\n".join(
                f"--- {n} ---\n"
                + c.sh(["docker", "logs", "--tail", "15", n], check=False).stdout
                + c.sh(["docker", "logs", "--tail", "15", n], check=False).stderr
                for n in (eng, gw)
            )
            raise Fail(f"{e}\n{logs}") from None
        c.cache["stack"] = st
    if c.cache["stack"] is None:
        raise Fail(
            "the containers did not come up (see test_containers_stream_a_completion)"
        )
    return c.cache["stack"]


def _stream(
    c: Ctx, url: str, key: str, max_tokens: int = 24, timeout: float = 30
) -> tuple[int, list[str]]:
    """POST a streamed completion; the status and the `data:` payloads in order."""
    status, _, body = c.http(
        "POST",
        url,
        {
            "model": "tracer",
            "prompt": "Once",
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        {"Authorization": f"Bearer {key}"},
        timeout=timeout,
    )
    lines = body.decode("utf-8", "replace").splitlines()
    return status, [
        ln[len("data:") :].strip() for ln in lines if ln.startswith("data:")
    ]


def test_containers_stream_a_completion(c: Ctx) -> None:
    # WHY: the two images, wired the way the charts wire them (engine by its network
    #      name, model mounted under /artifacts, key from the environment), stream
    #      tokens end to end before any cluster is involved.
    # KIND: conformance
    st = _stack(c)
    code, data = _stream(c, f"http://127.0.0.1:{st['api']}/v1/completions", PROBE_KEY)
    if code != 200:
        raise Fail(
            f"POST /v1/completions through the gateway container answered {code}"
        )
    chunks = [d for d in data if d != "[DONE]"]
    if len(chunks) < 24 or not data or data[-1] != "[DONE]":
        raise Fail(
            f"want 24 chunks then [DONE]; got {len(chunks)} chunks, last {data[-1:] or 'nothing'}"
        )


def test_gateway_container_rejects_a_missing_key(c: Ctx) -> None:
    # WHY: the key check must survive packaging: an image that dropped TL_API_KEY
    #      handling, or read it from a file that is not in the image, fails open or closed here.
    # KIND: boundary
    st = _stack(c)
    status, _, _ = c.http(
        "POST",
        f"http://127.0.0.1:{st['api']}/v1/completions",
        {"model": "tracer", "prompt": "Once", "max_tokens": 1},
    )
    if status != 401:
        raise Fail(f"a request without a key answered {status}; want 401")


def test_containers_exit_zero_on_sigterm(c: Ctx) -> None:
    # WHY: spec/cli-roles.md: servers exit 0 on SIGTERM. Every rollout, scale-down,
    #      and the ops.00 drill stop your pods this way.
    # KIND: fault
    # CHAPTER: dep.00 section 5, Pitfall 2
    st = _stack(c)
    errs = []
    for part in ("gateway", "engine"):
        name = st[part]
        t0 = time.monotonic()
        c.sh(["docker", "stop", "-t", "8", name], timeout=30)
        took = time.monotonic() - t0
        code = c.sh(
            ["docker", "inspect", "-f", "{{.State.ExitCode}}", name]
        ).stdout.strip()
        if code != "0" or took >= 7.5:
            errs.append(
                f"{part}: exit {code} after {took:.1f}s; want exit 0 inside the 8 s grace period"
            )
    if errs:
        raise Fail("\n".join(errs))


# -- cluster ------------------------------------------------------------------


def _deploy(c: Ctx) -> dict:
    d = c.system().get("deploy") or {}
    ctx = d.get("kube_context")
    if not ctx:
        raise Fail("system.toml has no [deploy].kube_context")
    c.need_cluster(ctx)
    return d


def test_workloads_available_on_kind(c: Ctx) -> None:
    # WHY: Available means the image was loaded, the pod started as non-root, read
    #      its model from /artifacts, and passed readiness: the whole chart worked.
    # KIND: conformance
    d = _deploy(c)
    n, ns = _name(c), d.get("namespace", _name(c))
    errs = []
    for part in ("engine", "gateway"):
        r = c.sh(
            [
                "kubectl",
                "--context",
                d["kube_context"],
                "-n",
                ns,
                "get",
                "deploy",
                f"{n}-{part}",
                "-o",
                "json",
            ],
            timeout=30,
            check=False,
        )
        if r.returncode != 0:
            errs.append(
                f"deploy/{n}-{part} in namespace {ns}: {r.stderr.strip()[:200]}"
            )
            continue
        st = json.loads(r.stdout).get("status") or {}
        if int(st.get("availableReplicas") or 0) < 1:
            errs.append(
                f"deploy/{n}-{part} has no available replica: `kubectl -n {ns} describe pods` says why"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_stream_through_the_gateway_nodeport(c: Ctx) -> None:
    # WHY: the tracer's promise on kind: curl with your key at 127.0.0.1:30080
    #      streams tokens from your engine through your gateway (MS-P1).
    # KIND: conformance
    d = _deploy(c)
    env_name = (c.system().get("endpoints") or {}).get("api_key_env", "TL_API_KEY")
    key = os.environ.get(env_name)
    if not key:
        raise Fail(f"export {env_name} with the key stored in your API key Secret")
    code, data = _stream(
        c, d["gateway_url"].rstrip("/") + "/v1/completions", key, max_tokens=48
    )
    chunks = [x for x in data if x != "[DONE]"]
    if code != 200 or len(chunks) < 32 or data[-1:] != ["[DONE]"]:
        raise Fail(
            f"through {d['gateway_url']}: HTTP {code}, {len(chunks)} chunks; want 200, at least 32, then [DONE]"
        )


def test_jaeger_query_reachable(c: Ctx) -> None:
    # WHY: obs.00 and the MS-P1 trace step read your trace from the Jaeger query
    #      API at [deploy].traces; it must answer before there is a trace to find.
    # KIND: conformance
    d = _deploy(c)
    status, _, body = c.http(
        "GET", d["traces"].rstrip("/") + "/api/services", timeout=10
    )
    if status != 200:
        raise Fail(f"GET {d['traces']}/api/services answered {status}")
    try:
        json.loads(body)
    except json.JSONDecodeError:
        raise Fail("the Jaeger query API did not return JSON") from None


if __name__ == "__main__":
    raise SystemExit(run(globals()))
