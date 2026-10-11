"""The learner's harness manifest, system.toml (DESIGN 2.16).

[system]   name, version, course_version
[build]    steps = [[argv...], ...]          run in order before services
[entry]    <role> = [argv...]                spec/cli-roles.md roles
[services.<name>]                            local processes the runner starts
           entry  = "<role>"                 default: the service name
           section = "engine"                runtime.toml section; default: the name
           config = "deploy/runtime.dev.toml" template with {port} ... filled in
           health = "http://127.0.0.1:{health_port}/healthz"
           ready_timeout_s = 60
           after  = ["engine"]
[endpoints] api_base, api_key_env
[ci]       local = [[argv...], ...]          ci-status without a GitHub remote
[deploy]   kube_context, namespace, gateway_url, prometheus, traces, services
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError

PORT_NAMES = ("port", "health_port", "grpc_port", "kv_port", "registry_port")


@dataclass
class Service:
    name: str
    entry: str
    section: str
    config: str = ""
    health: str = "http://127.0.0.1:{health_port}/healthz"
    ready_timeout_s: float = 60.0
    after: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)


@dataclass
class System:
    path: Path
    raw: dict

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def name(self) -> str:
        return str(self.raw.get("system", {}).get("name", "system"))

    @property
    def build_steps(self) -> list[list[str]]:
        steps = self.raw.get("build", {}).get("steps", [])
        if not isinstance(steps, list) or not all(
            isinstance(s, list) and s and all(isinstance(x, str) for x in s)
            for s in steps
        ):
            raise HarnessError("system.toml [build].steps must be a list of argv lists")
        return steps

    @property
    def entries(self) -> dict[str, list[str]]:
        e = self.raw.get("entry", {})
        for k, v in e.items():
            if not (isinstance(v, list) and v and all(isinstance(x, str) for x in v)):
                raise HarnessError(
                    f"system.toml [entry].{k} must be a non-empty argv list"
                )
        return e

    def entry(self, role: str, who: str) -> list[str]:
        e = self.entries
        if role not in e:
            raise HarnessError(f"{who} needs [entry].{role} in system.toml")
        return list(e[role])

    @property
    def services(self) -> dict[str, Service]:
        out = {}
        for name, s in (self.raw.get("services") or {}).items():
            if not isinstance(s, dict):
                raise HarnessError(f"system.toml [services.{name}] must be a table")
            out[name] = Service(
                name=name,
                entry=s.get("entry", name),
                section=s.get("section", name),
                config=s.get("config", ""),
                health=s.get("health", Service.health),
                ready_timeout_s=float(s.get("ready_timeout_s", 60)),
                after=list(s.get("after", [])),
                args=list(s.get("args", [])),
            )
        for s in out.values():
            for a in s.after:
                if a not in out:
                    raise HarnessError(
                        f"system.toml [services.{s.name}].after names {a!r}, which is not a service"
                    )
        return out

    def service(self, name: str, who: str) -> Service:
        svcs = self.services
        if name not in svcs:
            raise HarnessError(f"{who} needs [services.{name}] in system.toml")
        return svcs[name]

    def start_order(self, names: list[str], who: str = "this run") -> list[str]:
        """names plus everything they come `after`, dependencies first."""
        svcs = self.services
        order: list[str] = []

        def visit(n: str, stack: tuple[str, ...]) -> None:
            if n in stack:
                raise HarnessError(
                    "service cycle in system.toml: " + " -> ".join(stack + (n,))
                )
            if n in order:
                return
            if n not in svcs:
                raise HarnessError(f"{who} needs [services.{n}] in system.toml")
            for a in svcs[n].after:
                visit(a, stack + (n,))
            order.append(n)

        for n in names:
            visit(n, ())
        return order

    @property
    def deploy(self) -> dict:
        return dict(self.raw.get("deploy") or {})

    @property
    def endpoints(self) -> dict:
        return dict(self.raw.get("endpoints") or {})

    @property
    def api_key_env(self) -> str:
        return str(self.endpoints.get("api_key_env", "TL_API_KEY"))

    @property
    def ci_local(self) -> list[list[str]]:
        steps = (self.raw.get("ci") or {}).get("local", [])
        if not isinstance(steps, list) or not all(
            isinstance(s, list) and s for s in steps
        ):
            raise HarnessError("system.toml [ci].local must be a list of argv lists")
        return steps


def schema_errors(learner: Path, raw: dict) -> list[str]:
    """system.toml against the vendored contracts/config/system.schema.json
    (DESIGN 2.16). No vendored schema (an older contracts/VERSION): no check."""
    import json

    from . import schema

    p = learner / "contracts" / "config" / "system.schema.json"
    if not p.is_file():
        return []
    try:
        sch = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"{p}: {e}"]
    return schema.validate(raw, sch, sch)


def load(learner: Path) -> System:
    p = learner / "system.toml"
    try:
        return System(p, tomllib.loads(p.read_text()))
    except OSError as e:
        raise HarnessError(f"system.toml: {e}") from None
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(f"system.toml: {e}") from None
