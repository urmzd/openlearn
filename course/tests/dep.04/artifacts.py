"""dep.04 artifact check: the Tilt dev loop (deploy/Tiltfile).

Run by `ol check dep.04` in your repo. Two tiers:

  static   deploy/Tiltfile is evaluated by the course's recording evaluator
           (_tiltfile.py: every Tilt builtin is a fake that records its call;
           helm() renders your charts with the real `helm template`). The
           recorded graph must: refuse any context but kind-<system>; build the
           engine and gateway images under the names your charts deploy; build
           the engine (Rust) by a full image rebuild and live-update the gateway
           (Go) with a host-compiled linux binary and a restart; deploy both
           charts; start the gateway only after the engine; bound `tilt ci`
  cluster  `tilt ci -f deploy/Tiltfile` on [deploy].kube_context exits 0 within
           the budget (needs tilt on PATH)

The cluster tier fails when the context is missing or down; with OL_SMOKE=1 it
is skipped with the reason instead.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tiltfile as tf  # noqa: E402
from _lib.practice import Ctx, Fail, Skip, objects, run, smoke_mode  # noqa: E402

TILTFILE = "deploy/Tiltfile"
MAX_CI_TIMEOUT_S = 20 * 60  # tilt ci's own default is 30 min


def _eval(c: Ctx, context: str) -> tf.Result:
    def cli(argv: list[str]) -> str:
        return c.sh(argv, timeout=60).stdout

    return tf.evaluate(c.require_file(TILTFILE), context, tf.helm_template_with(cli))


def _graph(c: Ctx) -> tf.Result:
    if "graph" not in c.cache:
        c.cache["graph"] = None
        ctx = f"kind-{c.system_name()}"
        try:
            c.cache["graph"] = _eval(c, ctx)
        except tf.TiltfileFail as e:
            raise Fail(f"{TILTFILE} called fail() with context {ctx}: {e}") from None
        except tf.Unsupported as e:
            raise Fail(f"{TILTFILE}: {e}") from None
        except Exception as e:  # noqa: BLE001  the learner's file raised
            raise Fail(
                f"{TILTFILE} does not evaluate: {type(e).__name__}: {e}"
            ) from None
    if c.cache["graph"] is None:
        raise Fail(f"{TILTFILE} did not evaluate (see test_tiltfile_evaluates)")
    return c.cache["graph"]


def _objects(c: Ctx) -> list[dict]:
    if "objs" not in c.cache:
        docs = []
        for text in _graph(c).yaml:
            docs += c.yaml_docs(text, "the YAML your Tiltfile deploys")
        c.cache["objs"] = [d for d in docs if isinstance(d, dict)]
    return c.cache["objs"]


def _workload(c: Ctx, part: str) -> dict:
    name = f"{c.system_name()}-{part}"
    for kind in ("Deployment", "StatefulSet"):
        for o in objects(_objects(c), kind):
            if (o.get("metadata") or {}).get("name") == name:
                return o
    raise Fail(f"{TILTFILE} deploys no Deployment or StatefulSet named {name}")


def _image_for(c: Ctx, part: str) -> tf.Image:
    """The image target Tilt builds for the workload's first container."""
    w = _workload(c, part)
    ctrs = ((w.get("spec") or {}).get("template") or {}).get("spec", {}).get(
        "containers"
    ) or []
    want = str(ctrs[0].get("image", "")).split("@")[0]
    repo = want.rsplit(":", 1)[0] if ":" in want.rsplit("/", 1)[-1] else want
    for img in _graph(c).images:
        if img.ref.split(":")[0] == repo:
            return img
    refs = [i.ref for i in _graph(c).images]
    raise Fail(
        f"the {part} chart runs image {want!r}, but the Tiltfile builds {refs or 'no images'}: "
        f"Tilt replaces an image only when docker_build's ref matches the chart's repository ({repo})"
    )


def _resource_name(c: Ctx, part: str) -> str:
    """Tilt names a workload's resource after the workload unless k8s_resource renames it."""
    name = f"{c.system_name()}-{part}"
    r = _graph(c).resources.get(name) or {}
    return r.get("new_name") or name


# -- static -----------------------------------------------------------------------


def test_tiltfile_evaluates(c: Ctx) -> None:
    # WHY: Tilt reads deploy/Tiltfile on every `tilt up` and `tilt ci`; an error
    #      there means no dev loop and a red CI job (dep.05). The course runs it
    #      with recording fakes of the Tilt builtins, so this needs no cluster.
    # KIND: unit
    # CHAPTER: dep.04 section 4, The interface
    _graph(c)


def test_refuses_other_clusters(c: Ctx) -> None:
    # WHY: `tilt up` deploys to whatever kubectl's current context is. A
    #      Tiltfile that does not check it will one day push your dev images
    #      and charts into a cluster you did not mean (a work cluster, a shared
    #      staging). Evaluated with any context but kind-<system>, it must fail().
    # KIND: fault
    # CHAPTER: dep.04 section 5, Pitfall 1
    try:
        _eval(c, "prod-cluster")
    except tf.TiltfileFail:
        return
    except Exception:  # noqa: BLE001  any refusal counts, the message is the learner's
        return
    raise Fail(
        f"{TILTFILE} evaluates happily with k8s_context() == 'prod-cluster'; fail() unless it is kind-{c.system_name()}"
    )


def test_images_match_the_charts(c: Ctx) -> None:
    # WHY: Tilt injects an image it built only into containers whose image
    #      repository equals docker_build's ref. With a mismatch, Tilt builds
    #      your code and deploys the chart's old image: edits never show up and
    #      nothing says why.
    # KIND: conformance
    # CHAPTER: dep.04 section 5, Pitfall 2
    _image_for(c, "engine")
    _image_for(c, "gateway")


def test_engine_rebuilds_from_its_dockerfile(c: Ctx) -> None:
    # WHY: the engine is Rust linked against your C library: a change needs the
    #      toolchain and a relink, so the loop is a full rebuild of the dep.00
    #      image (cargo's cached layers keep it short), from the repo root.
    #      A live update that syncs sources into a runtime image has no
    #      compiler to use them.
    # KIND: unit
    # CHAPTER: dep.04 section 2, Principles
    img = _image_for(c, "engine")
    errs = []
    if img.live_update:
        errs.append(
            "the engine image has a live_update; Rust changes rebuild the image"
        )
    if img.kind != "custom_build":
        want = c.path("deploy/docker/engine.Dockerfile").resolve()
        if Path(img.dockerfile) != want:
            errs.append(
                f"the engine builds from {img.dockerfile}; want deploy/docker/engine.Dockerfile"
            )
        if Path(img.context) != c.root:
            errs.append(
                f"the engine's build context is {img.context}; the Dockerfile expects the repo root"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_gateway_live_updates_a_linux_binary(c: Ctx) -> None:
    # WHY: the Go gateway compiles in seconds on your machine, so the loop is:
    #      a local_resource cross-compiles it (GOOS=linux, CGO_ENABLED=0: the
    #      kind node is linux, the binary must be static), live_update syncs the
    #      one file into the running container, and the process restarts. A
    #      darwin binary in the container dies with "exec format error".
    # KIND: unit
    # CHAPTER: dep.04 section 5, Pitfall 3
    g = _graph(c)
    img = _image_for(c, "gateway")
    steps = img.live_update
    syncs = [s for s in steps if isinstance(s, tuple) and s[0] == "sync"]
    errs = []
    if not syncs:
        errs.append("the gateway image has no live_update sync step")
    restarts = img.restart or any(
        isinstance(s, tuple) and s[0] in ("run", "restart") for s in steps
    )
    if not restarts:
        errs.append(
            "nothing restarts the gateway after a sync (docker_build_with_restart from ext://restart_process, or a run step)"
        )
    go_dir = str(c.path("go"))
    source_sync = any(s[1].startswith(go_dir) for s in syncs)
    in_image = any(
        isinstance(s, tuple) and s[0] == "run" and re.search(r"\bgo\s+build\b", s[1])
        for s in steps
    )
    builders = [
        (n, r) for n, r in g.local.items() if re.search(r"\bgo\s+build\b", r["cmd"])
    ]
    if syncs and not builders and not (source_sync and in_image):
        errs.append(
            "nothing compiles the gateway: a local_resource running `go build` (or Go sources synced and a run('go build ...') step)"
        )
    for n, r in builders:
        if "GOOS=linux" not in r["cmd"] or "CGO_ENABLED=0" not in r["cmd"]:
            errs.append(
                f"local_resource {n!r} must build with CGO_ENABLED=0 GOOS=linux (the kind node runs linux): {r['cmd']!r}"
            )
        if not any(d == go_dir or d.startswith(go_dir + "/") for d in r["deps"]):
            errs.append(
                f"local_resource {n!r} must watch ../go (deps), or edits never trigger it"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_charts_deployed(c: Ctx) -> None:
    # WHY: the loop deploys the same charts as production (dep.03), into your
    #      system's namespace: no second set of dev manifests to drift.
    # KIND: unit
    n = c.system_name()
    g = _graph(c)
    errs = []
    for part in ("engine", "gateway"):
        chart = c.path(f"deploy/helm/{n}-{part}").resolve()
        calls = [h for h in g.helm if Path(h["chart"]) == chart]
        if not calls:
            errs.append(f"no helm() call renders deploy/helm/{n}-{part}")
        elif calls[0]["namespace"] not in ("", n):
            errs.append(
                f"deploy/helm/{n}-{part} is rendered into namespace {calls[0]['namespace']!r}; want {n!r}"
            )
    if errs:
        raise Fail("\n".join(errs))
    _workload(c, "engine")
    _workload(c, "gateway")


def test_gateway_starts_after_the_engine(c: Ctx) -> None:
    # WHY: the gateway is ready only when the engine is (/readyz); started
    #      first, it crash-loops and its logs are noise. resource_deps orders the
    #      graph; once dep.06 lands, durable precedes the workers the same way.
    # KIND: unit
    # CHAPTER: dep.04 section 5, Pitfall 4
    n = c.system_name()
    g = _graph(c)
    gw = g.resources.get(f"{n}-gateway")
    if not gw or _resource_name(c, "engine") not in gw["resource_deps"]:
        raise Fail(
            f"k8s_resource('{n}-gateway', resource_deps=['{_resource_name(c, 'engine')}', ...]) is missing"
        )
    if (
        c.path(f"deploy/helm/{n}-worker").is_dir()
        and c.path(f"deploy/helm/{n}-durable").is_dir()
    ):
        w = g.resources.get(f"{n}-worker") or {}
        if f"{n}-durable" not in w.get("resource_deps", []):
            raise Fail(
                f"k8s_resource('{n}-worker', resource_deps=['{n}-durable']) is missing"
            )


def test_ci_has_a_budget(c: Ctx) -> None:
    # WHY: `tilt ci` waits for every resource to be ready. Without a budget a
    #      pod stuck in ImagePullBackOff holds the CI job for tilt's default
    #      30 minutes; ci_settings(timeout=...) ends it sooner with the reason.
    # KIND: boundary
    t = str(_graph(c).ci_settings.get("timeout", ""))
    secs = sum(
        float(x) * {"s": 1, "m": 60, "h": 3600}[u]
        for x, u in re.findall(r"(\d+(?:\.\d+)?)([smh])", t)
    )
    if not t or secs > MAX_CI_TIMEOUT_S:
        raise Fail(
            f"ci_settings(timeout=...) is {t or 'unset'}; set {MAX_CI_TIMEOUT_S // 60}m or less"
        )


# -- cluster ----------------------------------------------------------------------


def test_tilt_ci_on_kind(c: Ctx) -> None:
    # WHY: the real loop, once: Tilt builds both images, deploys both charts in
    #      order, and exits 0 when every resource is ready (what dep.05's kind
    #      job runs).
    # KIND: conformance
    d = c.system().get("deploy") or {}
    ctx = str(d.get("kube_context") or "")
    if not ctx:
        if smoke_mode():
            c.skipped_cluster = True
            raise Skip(
                "cluster tier skipped under OL_SMOKE=1: system.toml has no [deploy].kube_context"
            )
        raise Fail("system.toml has no [deploy].kube_context")
    c.need_cluster(ctx)
    if shutil.which("tilt") is None:
        raise Fail(
            "tilt is not on PATH (`brew install tilt`; ol doctor lists it from Pass 7)"
        )
    budget = int(min(MAX_CI_TIMEOUT_S, 1800))
    c.sh(
        ["tilt", "ci", "-f", TILTFILE, "--context", ctx, "--timeout", f"{budget}s"],
        timeout=budget + 60,
    )


if __name__ == "__main__":
    raise SystemExit(run(globals()))
