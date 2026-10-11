"""Evaluate a Tiltfile without Tilt, recording what it asks for (dep.04).

A Tiltfile is Starlark, whose syntax is a subset of Python's, so the course
runs it with Python's evaluator and a fake of each Tilt builtin that records
its call instead of building or deploying. That makes the static tier of
`ol check dep.04` deterministic and fast, and needs no cluster. `tilt ci`
itself runs in the cluster tier.

Supported: docker_build, custom_build, docker_build_with_restart (from
load('ext://restart_process', ...)), k8s_yaml, helm (renders the chart with
the real `helm template`), k8s_resource, local_resource, allow_k8s_contexts,
k8s_context, k8s_namespace, ci_settings, update_settings, watch_settings,
default_registry, sync, run, fall_back_on, restart_container, read_file,
blob, local (returns '' and runs nothing), os.getenv, os.path.*, config.*,
listdir, read_yaml, decode_yaml, struct, fail, print. Anything else fails
with the name it did not know.
"""

from __future__ import annotations

import os
import subprocess
import types
from dataclasses import dataclass, field
from pathlib import Path


class TiltfileFail(Exception):
    """The Tiltfile called fail(msg)."""


class Unsupported(Exception):
    """The Tiltfile used something this evaluator does not model."""


class Blob(str):
    """Tilt's Blob: text that k8s_yaml accepts."""


@dataclass
class Image:
    ref: str
    context: str  # absolute
    dockerfile: str  # absolute
    live_update: list = field(default_factory=list)
    restart: bool = False  # docker_build_with_restart
    only: list = field(default_factory=list)
    kind: str = "docker_build"


@dataclass
class Result:
    images: list[Image] = field(default_factory=list)
    yaml: list[str] = field(
        default_factory=list
    )  # every document k8s_yaml received, as text
    helm: list[dict] = field(default_factory=list)  # {chart, name, namespace}
    resources: dict = field(default_factory=dict)  # k8s_resource workload -> kwargs
    local: dict = field(default_factory=dict)  # local_resource name -> kwargs
    allow_contexts: list = field(default_factory=list)
    ci_settings: dict = field(default_factory=dict)
    loads: list = field(default_factory=list)


ALLOWED_BUILTINS = {
    n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
    for n in (
        "len",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "tuple",
        "range",
        "enumerate",
        "sorted",
        "reversed",
        "zip",
        "min",
        "max",
        "any",
        "all",
        "repr",
        "hasattr",
        "getattr",
        "type",
        "isinstance",
        "abs",
        "set",
    )
}


def evaluate(tiltfile: Path, context: str, helm_template=None) -> Result:
    """Run `tiltfile` with k8s_context() == context; raise TiltfileFail,
    Unsupported, or the Python error the file raised."""
    here = tiltfile.parent.resolve()
    res = Result()

    def p(rel: str) -> str:
        return str((here / str(rel)).resolve())

    def _docker_build(
        ref,
        context,
        dockerfile="Dockerfile",
        live_update=None,
        only=None,
        kind="docker_build",
        restart=False,
        **_,
    ):
        ctx_dir = p(context)
        df = (
            str((Path(ctx_dir) / dockerfile).resolve())
            if dockerfile == "Dockerfile"
            else p(dockerfile)
        )
        res.images.append(
            Image(
                str(ref),
                ctx_dir,
                df,
                list(live_update or []),
                restart,
                list(only or []),
                kind,
            )
        )

    def docker_build(ref, context, dockerfile="Dockerfile", **kw):
        _docker_build(ref, context, dockerfile, **kw)

    def custom_build(ref, command, deps, live_update=None, **kw):
        res.images.append(
            Image(
                str(ref), p("."), "", list(live_update or []), False, [], "custom_build"
            )
        )

    def docker_build_with_restart(
        ref, context, entrypoint, dockerfile="Dockerfile", live_update=None, **kw
    ):
        _docker_build(
            ref,
            context,
            dockerfile,
            live_update=live_update,
            restart=True,
            kind="docker_build_with_restart",
            **kw,
        )

    def k8s_yaml(y, allow_duplicates=False):
        items = y if isinstance(y, (list, tuple)) else [y]
        for it in items:
            if isinstance(it, Blob):
                res.yaml.append(str(it))
            else:
                res.yaml.append(Path(p(it)).read_text())

    def helm(chart, name="", namespace="", values=None, set=None, **_):
        res.helm.append({"chart": p(chart), "name": name, "namespace": namespace})
        if helm_template is None:
            return Blob("")
        return Blob(
            helm_template(
                p(chart),
                name or Path(chart).name,
                namespace or "default",
                list(values or []),
                list(set or []),
            )
        )

    def k8s_resource(
        workload="",
        new_name="",
        resource_deps=None,
        port_forwards=None,
        labels=None,
        objects=None,
        **kw,
    ):
        res.resources[str(workload or new_name)] = {
            "new_name": new_name,
            "resource_deps": list(resource_deps or []),
            **kw,
        }

    def local_resource(name, cmd="", deps=None, resource_deps=None, serve_cmd="", **kw):
        res.local[str(name)] = {
            "cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
            "deps": [p(d) for d in (deps or [])],
            "resource_deps": list(resource_deps or []),
            "serve_cmd": serve_cmd,
            **kw,
        }

    def allow_k8s_contexts(c):
        res.allow_contexts += list(c) if isinstance(c, (list, tuple)) else [c]

    def ci_settings(**kw):
        res.ci_settings.update(kw)

    def fail(msg):
        raise TiltfileFail(str(msg))

    def load(module, *names, **aliases):
        res.loads.append(module)
        known = {
            "ext://restart_process": {
                "docker_build_with_restart": docker_build_with_restart
            }
        }
        if module not in known:
            raise Unsupported(
                f"load({module!r}): the course evaluator knows only {sorted(known)}"
            )
        for n in names:
            if n not in known[module]:
                raise Unsupported(f"load({module!r}, {n!r}): not modelled")
            ns[n] = known[module][n]
        for alias, n in aliases.items():
            ns[alias] = known[module][n]

    def noop(*a, **k):
        return None

    def read_file(path, default=None):
        f = Path(p(path))
        if not f.is_file():
            if default is not None:
                return Blob(default)
            raise TiltfileFail(f"read_file: {path} does not exist")
        return Blob(f.read_text())

    def read_yaml(path, default=None):
        import yaml

        f = Path(p(path))
        return yaml.safe_load(f.read_text()) if f.is_file() else default

    def decode_yaml(s):
        import yaml

        return yaml.safe_load(str(s))

    os_mod = types.SimpleNamespace(
        getenv=lambda k, d="": os.environ.get(k, d) if k.startswith("TILT_") else d,
        environ={},
        path=types.SimpleNamespace(
            join=os.path.join,
            exists=lambda x: Path(p(x)).exists(),
            basename=os.path.basename,
            dirname=os.path.dirname,
            abspath=lambda x: p(x),
        ),
        name="posix",
    )
    config_mod = types.SimpleNamespace(
        parse=lambda: {},
        define_string=noop,
        define_string_list=noop,
        define_bool=noop,
        tilt_subcommand="ci",
        main_path=str(tiltfile),
        main_dir=str(here),
    )

    def unknown(name):
        def f(*a, **k):
            raise Unsupported(
                f"{name}() is not modelled by the course's Tiltfile evaluator"
            )

        return f

    ns: dict = {
        "__builtins__": dict(ALLOWED_BUILTINS, print=noop),
        "docker_build": docker_build,
        "custom_build": custom_build,
        "k8s_yaml": k8s_yaml,
        "helm": helm,
        "k8s_resource": k8s_resource,
        "local_resource": local_resource,
        "allow_k8s_contexts": allow_k8s_contexts,
        "k8s_context": lambda: context,
        "k8s_namespace": lambda: "default",
        "ci_settings": ci_settings,
        "update_settings": noop,
        "watch_settings": noop,
        "default_registry": noop,
        "set_team": noop,
        "analytics_settings": noop,
        "secret_settings": noop,
        "version_settings": noop,
        "watch_file": noop,
        "sync": lambda src, dest: ("sync", p(src), str(dest)),
        "run": lambda cmd, trigger=None, echo_off=False: (
            "run",
            cmd if isinstance(cmd, str) else " ".join(cmd),
            list(trigger or []),
        ),
        "fall_back_on": lambda files: (
            "fall_back_on",
            [p(f) for f in (files if isinstance(files, (list, tuple)) else [files])],
        ),
        "restart_container": lambda: ("restart",),
        "read_file": read_file,
        "blob": Blob,
        "local": lambda *a, **k: Blob(""),
        "read_yaml": read_yaml,
        "decode_yaml": decode_yaml,
        "listdir": lambda d, recursive=False: sorted(
            str(x) for x in Path(p(d)).iterdir()
        ),
        "struct": lambda **kw: types.SimpleNamespace(**kw),
        "fail": fail,
        "load": load,
        "os": os_mod,
        "config": config_mod,
        "k8s_custom_deploy": unknown("k8s_custom_deploy"),
        "docker_compose": unknown("docker_compose"),
        "include": unknown("include"),
    }
    code = compile(tiltfile.read_text(), str(tiltfile), "exec")
    exec(code, ns)  # noqa: S102  the learner's own Tiltfile, with fake builtins only
    return res


def helm_template_with(cli_run):
    """A helm() renderer that shells out to `helm template` through cli_run(argv) -> stdout."""

    def render(chart: str, name: str, namespace: str, values: list, sets: list) -> str:
        argv = ["helm", "template", name, chart, "--namespace", namespace]
        for v in values:
            argv += ["--values", v]
        for s in sets:
            argv += ["--set", s]
        return cli_run(argv)

    return render


def run_cli(argv: list[str], cwd: Path) -> str:
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise TiltfileFail(f"`{' '.join(argv)}` failed: {r.stderr.strip()[-400:]}")
    return r.stdout
