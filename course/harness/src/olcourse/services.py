"""Local service instances for milestones and `ol conform` (DESIGN 2.16, 5.7).

Ports are never fixed locally. Every instance gets `{port}`, `{health_port}`,
`{grpc_port}`, `{kv_port}`, and `{registry_port}` from the OS, a generated
runtime.toml (`{config}`) whose listen keys carry those ports, the same values
as `TL_<SECTION>__<KEY>` overrides (2.12), and a state directory (`{data}`).
Other entries see them as `{<service>.<port>}` and `{<service>.api_base}`.

The learner's entry argv is started in its own process group, so teardown
reaches every child (`go run` and `uv run` fork the real server).
"""

from __future__ import annotations

import os
import random
import shlex
import signal
import socket
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, ctx, placeholders, tomlw, web
from .system import PORT_NAMES, Service, System

# Which runtime.toml keys (2.12) carry which allocated port, per section.
# Every section also gets `health_listen`: health and metrics live on the
# health port locally, as :9464 does in k8s (2.6, 2.13).
SECTION_PORTS = {
    "gateway": {"listen": "port", "registry_listen": "registry_port"},
    "engine": {
        "http_listen": "port",
        "grpc_listen": "grpc_port",
        "kv_listen": "kv_port",
    },
    "durable": {"grpc_listen": "grpc_port"},
    "worker": {},
    "agent": {},
}


class PortPool:
    """Free TCP ports, never handing the same one out twice.

    Ports come from 20000 to 31999, below the ephemeral ranges of Linux
    (32768+) and macOS (49152+): a port the OS picks for `bind(0)` can be handed
    to some process's outgoing connection (an engine posting spans) before the
    service that was promised it binds, which fails with EADDRINUSE. Each
    candidate is test-bound on all interfaces, as services listen there."""

    LOW, HIGH = 20000, 32000

    def __init__(self) -> None:
        self.used: set[int] = set()
        self.rng = random.Random()

    @staticmethod
    def _free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                return False
        return True

    def take(self) -> int:
        for _ in range(256):
            p = self.rng.randrange(self.LOW, self.HIGH)
            if p not in self.used and self._free(p):
                self.used.add(p)
                return p
        for _ in range(64):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                p = s.getsockname()[1]
            if p not in self.used:
                self.used.add(p)
                return p
        raise HarnessError("could not allocate a free port")


@dataclass
class Instance:
    svc: Service
    ports: dict[str, int]
    data: Path
    config: Path
    log: Path
    env: dict = field(default_factory=dict)
    argv: list[str] = field(default_factory=list)
    proc: subprocess.Popen | None = None
    health_url: str = ""

    @property
    def name(self) -> str:
        return self.svc.name

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.ports['port']}"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def listen_keys(section: str) -> dict[str, str]:
    keys = dict(SECTION_PORTS.get(section, {}))
    keys["health_listen"] = "health_port"
    return keys


def runtime_toml(
    template: str, section: str, ports: dict[str, int], lookup, where: str
) -> tuple[str, dict]:
    """Fill a runtime.toml template, then force the section's listen keys to
    the allocated ports. Returns (text, TL_* overrides)."""
    text = placeholders.expand(template, lookup, where) if template else ""
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(
            f"{where}: not valid TOML after filling placeholders: {e}"
        ) from None
    sect = doc.setdefault(section, {})
    if not isinstance(sect, dict):
        raise HarnessError(f"{where}: [{section}] is not a table")
    env = {}
    for key, pname in listen_keys(section).items():
        val = f"127.0.0.1:{ports[pname]}"
        sect[key] = val
        env[f"TL_{section.upper()}__{key.upper()}"] = val
    return tomlw.dumps(doc), env


class Stack:
    """Starts and stops the learner's services for one run."""

    def __init__(self, system: System, logdir: Path, base_lookup, env: dict, who: str):
        self.system = system
        self.logdir = logdir
        self.base_lookup = base_lookup
        self.env = env
        self.who = who
        self.pool = PortPool()
        self.instances: dict[str, Instance] = {}

    # Lookups other entries use: {engine.port}, {engine.api_base}, {engine.config}.
    def lookup(self, name: str):
        svc, _, key = name.partition(".")
        inst = self.instances.get(svc)
        if inst is None or not key:
            return None
        if key in inst.ports:
            return str(inst.ports[key])
        if key == "api_base":
            return f"{inst.base}/v1"
        if key == "base":
            return inst.base
        if key in ("config", "data"):
            return str(getattr(inst, key))
        return None

    def _instance_lookup(self, inst: Instance):
        local = {k: str(v) for k, v in inst.ports.items()}
        local.update({"config": str(inst.config), "data": str(inst.data)})
        return placeholders.chain(local.get, self.lookup, self.base_lookup)

    def start(self, names: list[str]) -> None:
        for n in self.system.start_order(names, self.who):
            if n not in self.instances:
                self._start_one(self.system.service(n, self.who))

    def _start_one(self, svc: Service) -> None:
        d = self.logdir / "services"
        d.mkdir(parents=True, exist_ok=True)
        ports = {p: self.pool.take() for p in PORT_NAMES}
        data = self.logdir / "out" / svc.name / "data"
        data.mkdir(parents=True, exist_ok=True)
        inst = Instance(
            svc, ports, data, d / f"{svc.name}.runtime.toml", d / f"{svc.name}.log"
        )
        self.instances[svc.name] = inst
        lk = self._instance_lookup(inst)
        template = ""
        if svc.config:
            tp = self.system.root / svc.config
            if not tp.is_file():
                raise HarnessError(
                    f"[services.{svc.name}].config names {svc.config}, which does not exist"
                )
            template = tp.read_text()
        text, overrides = runtime_toml(
            template, svc.section, ports, lk, f"[services.{svc.name}].config"
        )
        inst.config.write_text(text)
        argv = placeholders.expand_argv(
            self.system.entry(svc.entry, self.who) + svc.args,
            lk,
            f"[entry].{svc.entry}",
        )
        inst.argv = argv
        inst.env = {**self.env, **overrides, "TL_CONFIG": str(inst.config)}
        inst.health_url = placeholders.expand(
            svc.health, lk, f"[services.{svc.name}].health"
        )
        self._spawn(inst, "w")
        ok, why = web.wait_ok(inst.health_url, svc.ready_timeout_s, inst.alive)
        if not ok:
            log = inst.log.read_text(errors="replace") if inst.log.is_file() else ""
            self.stop()
            raise ServiceError(
                f"service {svc.name} did not become healthy at {inst.health_url}: {why}\n"
                f"{ctx.indent(ctx.tail(log, 30))}"
            )
        ctx.say(
            f"  {ctx.DIM}service {svc.name} up on :{ports['port']} (health :{ports['health_port']}){ctx.RST}"
        )

    def _spawn(self, inst: Instance, mode: str) -> None:
        with open(inst.log, mode) as logf:
            logf.write(f"$ {shlex.join(inst.argv)}\n")
            logf.flush()
            try:
                inst.proc = subprocess.Popen(
                    inst.argv,
                    cwd=self.system.root,
                    env=inst.env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as e:
                raise HarnessError(
                    f"service {inst.name}: cannot start {inst.argv[0]!r}: {e}"
                ) from None

    def kill(self, name: str) -> int:
        """SIGKILL a running service's whole process group (chaos); returns its pid."""
        inst = self.instances.get(name)
        if inst is None or inst.proc is None:
            raise HarnessError(f"service {name} was not started")
        p = inst.proc
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        return p.pid

    def restart(self, name: str) -> None:
        """Start a killed service again with the same argv, ports, runtime.toml,
        and state directory, and wait for its health check. A service that
        does not come back is a failed verdict (ServiceError is not raised:
        the step, not the milestone, fails)."""
        inst = self.instances[name]
        with open(inst.log, "a") as f:
            f.write(f"\n--- SIGKILL; restarting at {time.strftime('%H:%M:%S')} ---\n")
        self._spawn(inst, "a")
        ok, why = web.wait_ok(inst.health_url, inst.svc.ready_timeout_s, inst.alive)
        if not ok:
            log = inst.log.read_text(errors="replace") if inst.log.is_file() else ""
            raise HarnessError(
                f"service {name} did not become healthy again at {inst.health_url}: {why}\n"
                f"{ctx.indent(ctx.tail(log, 30))}",
                code=1,
            )

    def stop(self) -> None:
        for inst in reversed(list(self.instances.values())):
            p = inst.proc
            if p is None or p.poll() is not None:
                continue
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            deadline = time.monotonic() + 5
            while p.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


class ServiceError(HarnessError):
    """A learner service failed to start: a failed verdict, not a harness error."""

    def __init__(self, msg: str):
        super().__init__(msg, code=1)


def run_build(
    system: System, log: Path, env: dict, timeout: float = 1800, lookup=None
) -> tuple[bool, str]:
    """[build].steps in order, in the learner repo; the first failure stops.
    Steps may name `{fixture:<path>}` (and, with a run's lookup, any run-level
    placeholder such as `{system}` or `{deploy.*}`)."""
    steps = system.build_steps
    if not steps:
        return True, ""
    fixtures = env.get("TINYLLM_FIXTURES")

    def fallback(name: str):
        if name.startswith("fixture:") and fixtures:
            return str(Path(fixtures) / name[len("fixture:") :])
        return None

    look = placeholders.chain(lookup, fallback) if lookup else fallback
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        for raw in steps:
            argv = placeholders.expand_argv(raw, look, "[build].steps")
            f.write(f"$ {shlex.join(argv)}\n")
            rc, out = ctx.run(argv, cwd=system.root, env=env, timeout=timeout)
            f.write(out + f"\n(exit {rc})\n")
            if rc != 0:
                return (
                    False,
                    f"build step `{shlex.join(argv)}` exited {rc}:\n{ctx.indent(ctx.tail(out, 30))}",
                )
    return True, ""
