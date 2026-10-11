"""Drill specs, injectors, and the undo journal (DESIGN 5.10).

    course/drills/<name>/drill.toml
      id, title, requires, seeded, time_limit_min, slo_profile, symptom
      [[inject]]  kind, target, at ("start" | "+30s" | "loadgen+30s"), plus per-kind keys
      [loadgen]   argv  (for at = "loadgen+Ns"; default --rate 20 --target {deploy.gateway_url})
      [workload]  role, argv, env, after_inject, wait_rollout: a command `ss drill start`
                  starts in the background through [entry].<role> (before the
                  injections, or after them with after_inject = true, once
                  `kubectl rollout status <wait_rollout>` passes); undo = kill
      [detect]    alert, within_s
      [[resolve.check]]  promql + hold_s, rollout (kubectl rollout status), or suite
      [[resolve.run]]    role, argv, env, timeout_s, expect (a milestone matcher):
                         a command `ss drill end` runs as a resolve check
      [[doc]]            path + sections (+ label): a document graded like the postmortem
      [evidence.trace]   a trace matcher table (service, services, spans)
      [postmortem] path (with {date}), sections

Every injection appends one line to .ss/drills/<run>/journal.jsonl with the
kubectl argv that undoes it; `ss drill reset` replays them in reverse.
"""

from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, ctx, placeholders
from .kube import Kube

BUILT = (
    "pod-kill",
    "deploy-patch",
    "scale-zero",
    "resource-limit",
    "config-drift",
    "netem",
    "wal-quota",
    "clock-skew",
    "activity-poison",
    "tenant-flood",
    "git-branch",
    "contract-bump",
)
LATER: dict[str, str] = {}
# Injectors that act on the cluster: a drill that uses one needs the safety gate.
CLUSTER = {
    "pod-kill",
    "deploy-patch",
    "scale-zero",
    "resource-limit",
    "config-drift",
    "netem",
    "wal-quota",
    "activity-poison",
}
# The two SLO window profiles of otel/slo.schema.json (DESIGN 2.11): pairs of
# (long window, short window, burn-rate factor). `drill` compresses `prod` 12x
# so a burn alert can fire within minutes.
SLO_PROFILES = {
    "prod": [("1h", "5m", 14.4), ("6h", "30m", 6.0)],
    "drill": [("5m", "25s", 14.4), ("30m", "150s", 6.0)],
}
AT = re.compile(r"^(start|(loadgen)?\+(\d+(?:\.\d+)?)s)$")


@dataclass
class Drill:
    name: str
    path: Path
    raw: dict

    @property
    def id(self) -> str:
        return str(self.raw.get("id", self.name))

    @property
    def title(self) -> str:
        return str(self.raw.get("title", ""))

    @property
    def injects(self) -> list[dict]:
        return list(self.raw.get("inject", []))

    @property
    def requires(self) -> list[str]:
        return list(self.raw.get("requires", []))

    @property
    def needs_cluster(self) -> bool:
        return any(i.get("kind") in CLUSTER for i in self.injects) or any(
            "rollout" in c for c in (self.raw.get("resolve") or {}).get("check", [])
        )

    @property
    def slo_profile(self) -> str | None:
        return self.raw.get("slo_profile")


def all_drills(course: Path) -> list[Drill]:
    d = course / "drills"
    out = []
    for p in sorted(d.glob("*/drill.toml")) if d.is_dir() else []:
        out.append(load_file(p))
    return out


def load_file(p: Path) -> Drill:
    try:
        raw = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(f"{p}: {e}") from None
    d = Drill(p.parent.name, p, raw)
    if d.slo_profile is not None and d.slo_profile not in SLO_PROFILES:
        raise HarnessError(
            f"{p}: slo_profile {d.slo_profile!r} is not one of {', '.join(SLO_PROFILES)}"
        )
    for i, inj in enumerate(d.injects):
        k = inj.get("kind")
        if k not in BUILT and k not in LATER:
            raise HarnessError(f"{p}: inject {i + 1}: unknown kind {k!r}")
        if not AT.match(str(inj.get("at", "start"))):
            raise HarnessError(
                f"{p}: inject {i + 1}: at {inj.get('at')!r} is not start, +Ns, or loadgen+Ns"
            )
    wl = raw.get("workload")
    if wl is not None:
        _check_command(p, "[workload]", wl)
        extra = sorted(
            set(wl) - {"role", "argv", "env", "after_inject", "wait_rollout"}
        )
        if extra:
            raise HarnessError(f"{p}: [workload]: unknown key(s) {extra}")
    from .matchers import MATCHERS

    for i, rr in enumerate((raw.get("resolve") or {}).get("run", [])):
        _check_command(p, f"resolve.run {i + 1}", rr)
        m = dict(rr.get("expect", {})).get("match", "exit-code")
        if m not in MATCHERS:
            raise HarnessError(f"{p}: resolve.run {i + 1}: unknown matcher {m!r}")
    return d


def _check_command(p: Path, what: str, t) -> None:
    if not isinstance(t, dict) or not t.get("role"):
        raise HarnessError(f"{p}: {what} needs `role` (an [entry] role)")
    if not isinstance(t.get("argv", []), list):
        raise HarnessError(f"{p}: {what}: argv must be a list")
    if not isinstance(t.get("env", {}), dict):
        raise HarnessError(f"{p}: {what}: env must be a table")


def find(course: Path, name: str) -> Drill:
    for d in all_drills(course):
        if name in (d.name, d.id):
            return d
    raise HarnessError(f"no drill {name!r} in {course / 'drills'} (ss drill list)")


# ---------------------------------------------------------------------------
# the journal


@dataclass
class Journal:
    dir: Path
    entries: list[dict] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.dir / "journal.jsonl"

    @classmethod
    def open(cls, d: Path) -> "Journal":
        j = cls(d)
        if j.path.is_file():
            j.entries = [
                json.loads(x) for x in j.path.read_text().splitlines() if x.strip()
            ]
        return j

    def append(self, entry: dict) -> None:
        entry = {"ts": time.time(), **entry}
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self.entries.append(entry)

    def pending_undos(self) -> list[dict]:
        done = {e["undone"] for e in self.entries if "undone" in e}
        return [
            e for e in self.entries if "n" in e and e.get("undo") and e["n"] not in done
        ]


def undo(k: Kube | None, j: Journal) -> list[str]:
    """Replay every pending undo in reverse order; returns what was undone.
    Actions: ["kill", pid], ["http", method, url, json], ["git-branch-remove",
    learner, scratch, branch], or a kubectl argv (needs the gate's Kube)."""
    from . import scratchcopy, web

    did = []
    for e in reversed(j.pending_undos()):
        for action in e["undo"]:
            if action[0] == "kill":
                try:
                    os.killpg(int(action[1]), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            elif action[0] == "http":
                r = web.request(
                    action[1], action[2], json_body=json.loads(action[3]), timeout=20
                )
                if not 200 <= r.status < 300:
                    raise HarnessError(
                        f"undo {action[1]} {action[2]}: HTTP {r.status} {r.error}"
                    )
            elif action[0] == "git-branch-remove":
                scratchcopy.remove(Path(action[1]), Path(action[2]), action[3])
            else:
                if k is None:
                    raise HarnessError(
                        "this undo needs the cluster (kubectl) and no safety gate passed"
                    )
                k.must(action)
        j.append({"undone": e["n"]})
        did.append(f"{e['kind']} {e.get('target', '')}".strip())
    return did


# ---------------------------------------------------------------------------
# injectors: each returns (what it did, undo actions)


def _kind_name(target: str) -> tuple[str, str]:
    kind, _, name = target.partition("/")
    if not name:
        raise HarnessError(
            f"target {target!r}: want <kind>/<name>, e.g. deploy/forge-gateway"
        )
    return kind, name


def _container(obj: dict, idx: int) -> dict:
    cs = obj.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    if idx >= len(cs):
        raise HarnessError(f"{obj.get('metadata', {}).get('name')}: no container {idx}")
    return cs[idx]


def _patch(k: Kube, target: str, ops: list[dict]) -> list[str]:
    return ["patch", target, "--type=json", "-p", json.dumps(ops)]


def ready_pods(k: Kube, target: str) -> list[str]:
    obj = k.json(["get", target])
    sel = obj.get("spec", {}).get("selector", {}).get("matchLabels") or {}
    if not sel:
        raise HarnessError(f"{target} has no spec.selector.matchLabels")
    pods = k.json(
        ["get", "pods", "-l", ",".join(f"{a}={b}" for a, b in sorted(sel.items()))]
    ).get("items", [])
    out = []
    for p in pods:
        conds = (p.get("status") or {}).get("conditions") or []
        if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds):
            out.append(p["metadata"]["name"])
    return sorted(out)


def _env_patch(
    k: Kube, target: str, idx: int, updates: dict[str, str]
) -> tuple[list, list]:
    """JSON-patch ops (and their undo) setting env vars on one container."""
    c = _container(k.json(["get", target]), idx)
    base = f"/spec/template/spec/containers/{idx}"
    old_env = c.get("env")
    env = [dict(x) for x in (old_env or [])]
    for name, value in updates.items():
        for x in env:
            if x.get("name") == name:
                x.clear()
                x.update({"name": name, "value": value})
                break
        else:
            env.append({"name": name, "value": value})
    undo = (
        {"op": "add", "path": f"{base}/env", "value": old_env}
        if old_env is not None
        else {"op": "remove", "path": f"{base}/env"}
    )
    return [{"op": "add", "path": f"{base}/env", "value": env}], [undo]


def inject(
    k: Kube | None, inj: dict, rng: random.Random, lookup, extra: dict | None = None
) -> tuple[str, list[list[str]], dict]:
    """`extra` carries what non-cluster injectors need: learner, course, reg,
    run_dir, env, drill name, the loadgen/ctl argv builder (`role`)."""
    kind = inj["kind"]
    if kind in LATER:
        raise HarnessError(f"the {kind} injector arrives with {LATER[kind]}")
    extra = extra or {}
    target = placeholders.expand(str(inj.get("target", "")), lookup, f"inject {kind}")
    idx = int(inj.get("container", 0))
    if kind in CLUSTER and k is None:
        raise HarnessError(f"the {kind} injector needs the cluster safety gate")
    if kind in (
        "netem",
        "wal-quota",
        "clock-skew",
        "activity-poison",
        "tenant-flood",
        "git-branch",
        "contract-bump",
    ):
        return _inject_more(k, kind, inj, target, idx, rng, lookup, extra)
    if kind == "pod-kill":
        pods = ready_pods(k, target)
        if not pods:
            raise HarnessError(f"pod-kill: {target} has no ready pod")
        pod = (
            rng.choice(pods)
            if inj.get("pick", "random-ready-replica") == "random-ready-replica"
            else pods[0]
        )
        k.must(["delete", "pod", pod, "--grace-period=0", "--force", "--wait=false"])
        return f"deleted pod {pod}", [], {"pod": pod}
    if kind == "scale-zero":
        old = int(k.json(["get", target]).get("spec", {}).get("replicas", 1))
        k.must(["scale", target, "--replicas=0"])
        return (
            f"scaled {target} from {old} to 0",
            [["scale", target, f"--replicas={old}"]],
            {"replicas": old},
        )
    if kind == "deploy-patch":
        c = _container(k.json(["get", target]), idx)
        base = f"/spec/template/spec/containers/{idx}"
        ops, undo_ops = [], []
        if "env" in inj:
            old_env = c.get("env")
            env = [dict(x) for x in (old_env or [])]
            for name, value in inj["env"].items():
                value = placeholders.expand(str(value), lookup, f"inject {kind}")
                for x in env:
                    if x.get("name") == name:
                        x.clear()
                        x.update({"name": name, "value": value})
                        break
                else:
                    env.append({"name": name, "value": value})
            ops.append({"op": "add", "path": f"{base}/env", "value": env})
            undo_ops.append(
                {"op": "add", "path": f"{base}/env", "value": old_env}
                if old_env is not None
                else {"op": "remove", "path": f"{base}/env"}
            )
        if "args" in inj:
            old_args = c.get("args")
            ops.append(
                {
                    "op": "add",
                    "path": f"{base}/args",
                    "value": [str(a) for a in inj["args"]],
                }
            )
            undo_ops.append(
                {"op": "add", "path": f"{base}/args", "value": old_args}
                if old_args is not None
                else {"op": "remove", "path": f"{base}/args"}
            )
        if not ops:
            raise HarnessError("deploy-patch needs `env` or `args`")
        k.must(_patch(k, target, ops))
        return (
            f"patched {target} ({', '.join(sorted(inj.get('env', {})) or ['args'])})",
            [_patch(k, target, undo_ops)],
            {},
        )
    if kind == "resource-limit":
        c = _container(k.json(["get", target]), idx)
        old = c.get("resources")
        new = json.loads(json.dumps(old or {}))
        new.setdefault("limits", {}).update(
            {a: str(b) for a, b in (inj.get("limits") or {}).items()}
        )
        path = f"/spec/template/spec/containers/{idx}/resources"
        k.must(_patch(k, target, [{"op": "add", "path": path, "value": new}]))
        undo_op = (
            {"op": "add", "path": path, "value": old}
            if old is not None
            else {"op": "remove", "path": path}
        )
        return (
            f"set limits {new['limits']} on {target}",
            [_patch(k, target, [undo_op])],
            {},
        )
    if kind == "config-drift":
        cm = str(inj.get("configmap", ""))
        if not cm:
            raise HarnessError("config-drift needs `configmap`")
        data = k.json(["get", f"configmap/{cm}"]).get("data") or {}
        new = {
            a: placeholders.expand(str(b), lookup, "config-drift")
            for a, b in (inj.get("set") or {}).items()
        }
        restore = {a: data.get(a) for a in new}  # None removes a key that was not there
        k.must(
            [
                "patch",
                f"configmap/{cm}",
                "--type=merge",
                "-p",
                json.dumps({"data": new}),
            ]
        )
        undo_actions = [
            [
                "patch",
                f"configmap/{cm}",
                "--type=merge",
                "-p",
                json.dumps({"data": restore}),
            ]
        ]
        if target:
            k.must(["rollout", "restart", target])
            undo_actions.append(["rollout", "restart", target])
        return (
            f"changed configmap/{cm} keys {sorted(new)}"
            + (f", restarted {target}" if target else ""),
            undo_actions,
            {},
        )
    raise HarnessError(f"unknown injector {kind!r}")


def _inject_more(k, kind, inj, target, idx, rng, lookup, extra):
    where = f"inject {kind}"
    if kind == "netem":
        pods = ready_pods(k, target)
        if not pods:
            raise HarnessError(f"netem: {target} has no ready pod")
        pod = rng.choice(pods)
        cname = _container(k.json(["get", target]), idx).get("name", "")
        image = str(inj.get("image", "nicolaka/netshoot:v0.13"))
        delay = str(inj.get("delay", "300ms"))
        loss = str(inj.get("loss", "5%"))
        base = [
            "debug",
            f"pod/{pod}",
            f"--image={image}",
            f"--target={cname}",
            "--profile=netadmin",
            "--",
        ]
        k.must(
            base
            + [
                "tc",
                "qdisc",
                "add",
                "dev",
                "eth0",
                "root",
                "netem",
                "delay",
                delay,
                "loss",
                loss,
            ]
        )
        return (
            f"netem delay {delay} loss {loss} on pod {pod}",
            [base + ["tc", "qdisc", "del", "dev", "eth0", "root"]],
            {"pod": pod},
        )
    if kind in ("wal-quota", "activity-poison"):
        if kind == "wal-quota":
            updates = {"TL_DURABLE__WAL_MAX_BYTES": str(inj.get("bytes", 1048576))}
        else:
            fp = str(inj.get("failpoint", "dur/activity/poison"))
            c = _container(k.json(["get", target]), idx)
            cur = next(
                (
                    e.get("value", "")
                    for e in c.get("env") or []
                    if e.get("name") == "TL_FAILPOINTS"
                ),
                "",
            )
            updates = {
                "TL_FAILPOINTS": ";".join(
                    x for x in (cur, f"{fp}={inj.get('action', 'panic')}") if x
                )
            }
        ops, undo_ops = _env_patch(k, target, idx, updates)
        k.must(_patch(k, target, ops))
        did = f"set {', '.join(f'{a}={b}' for a, b in updates.items())} on {target}"
        if kind == "wal-quota" and inj.get("burst_argv"):
            role = extra.get("role")
            argv = placeholders.expand_argv(
                role("ctl") + list(inj["burst_argv"]), lookup, where
            )
            rc, out = ctx.run(
                argv,
                cwd=extra.get("learner"),
                env=extra.get("env"),
                timeout=float(inj.get("burst_timeout_s", 300)),
            )
            did += f"; burst `{' '.join(argv)}` exited {rc}"
        return did, [_patch(k, target, undo_ops)], {}
    if kind == "clock-skew":
        from . import web

        url = placeholders.expand(
            str(inj.get("url", "{deploy.durable_debug_url}/debug/clock")), lookup, where
        )
        skew = float(inj.get("skew_s", 3600))
        r = web.request("POST", url, json_body={"skew_s": skew}, timeout=20)
        if not 200 <= r.status < 300:
            raise HarnessError(f"clock-skew: POST {url}: HTTP {r.status} {r.error}")
        return (
            f"skewed the durable clock by {skew:g}s",
            [["http", "POST", url, json.dumps({"skew_s": 0})]],
            {},
        )
    if kind == "tenant-flood":
        role = extra.get("role")
        tenant = str(inj.get("tenant", "tenant-noisy"))
        args = list(
            inj.get(
                "argv",
                [
                    "--tenant",
                    tenant,
                    "--rate",
                    str(inj.get("rate", 200)),
                    "--target",
                    "{deploy.gateway_url}",
                ],
            )
        )
        argv = placeholders.expand_argv(role("loadgen") + args, lookup, where)
        pid = start_background(
            argv,
            extra["learner"],
            extra["env"],
            Path(extra["run_dir"]) / f"flood-{tenant}.log",
        )
        return (
            f"flooding as {tenant}: {' '.join(argv)}",
            [["kill", str(pid)]],
            {"tenant": tenant},
        )
    from . import scratchcopy

    name = str(inj.get("branch") or extra.get("drill", "drill"))
    scratch = Path(extra["run_dir"]) / f"scratch-{name}"
    if kind == "git-branch":
        commits = [dict(c) for c in inj.get("commit", [])]
        if not commits:
            raise HarnessError(
                "git-branch needs [[inject.commit]] tables (seeded, run, or write)"
            )
    else:  # contract-bump: `ss contracts sync --to <rev>` on a branch
        to = placeholders.expand(str(inj.get("to", "")), lookup, where)
        if not to:
            raise HarnessError("contract-bump needs `to` (a tag or sha, e.g. kv/v2)")
        ss = str(Path(ctx.root()) / "practice" / "bin" / "ss")
        commits = [
            {
                "kind": "run",
                "argv": ["bash", ss, "contracts", "sync", "--to", to],
                "env": {"SS_COURSE_HOME": "{scratch}", "SS_ROOT": str(ctx.root())},
                "message": str(inj.get("message", f"chore(contracts): sync to {to}")),
            }
        ]
    built = scratchcopy.build(
        Path(extra["learner"]), scratch, name, commits, extra["course"], extra["reg"]
    )
    scratchcopy.publish(Path(extra["learner"]), built)
    return (
        f"created branch {built.branch} with {len(built.commits)} commit(s)",
        [["git-branch-remove", str(extra["learner"]), str(scratch), built.branch]],
        {"branch": built.branch, "commits": built.commits},
    )


def start_background(argv: list[str], cwd: Path, env: dict, log: Path) -> int:
    """Start a command in its own process group (undo = kill it); returns its pid."""
    with open(log, "w") as f:
        p = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return p.pid


# ---------------------------------------------------------------------------
# SLO profile (DESIGN 2.11)


_DUR = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}


def _seconds(d: str) -> float:
    """A Prometheus duration ("2m30s", "150s", "5m") in seconds."""
    return sum(float(n) * _DUR[u] for n, u in re.findall(r"(\d+)(ms|[smhdwy])", d))


def slo_errors(learner: Path, d: Drill) -> list[str]:
    """A drill under an SLO profile needs the learner's alert rules (files
    under deploy/observability/) to define the [detect] alert, written with
    the profile's short windows, so the alert can fire within the drill."""
    det = d.raw.get("detect")
    if not det:
        return []
    alert = str(det.get("alert", ""))
    obs = learner / "deploy" / "observability"
    files = (
        sorted(p for p in obs.rglob("*") if p.suffix in (".yaml", ".yml"))
        if obs.is_dir()
        else []
    )
    text = "\n".join(p.read_text(errors="replace") for p in files)
    errs = []
    if not files:
        errs.append(
            "no alert rules under deploy/observability/ (otel/slo.schema.json names the required alerts)"
        )
    elif not re.search(rf"\balert:\s*{re.escape(alert)}\b", text):
        errs.append(f"no rule defines alert {alert} (required by the drill's [detect])")
    windows = SLO_PROFILES[d.slo_profile]
    # PromQL range selectors: rate(x[5m]) and rate(x[25s]). Compared as
    # durations, so [150s] and [2m30s] are the same window.
    present = {
        _seconds(m.group(1))
        for m in re.finditer(r"\[((?:\d+(?:ms|[smhdwy]))+)(?::[^\]]*)?\]", text)
    }
    missing = [
        w
        for long_, short, _ in windows
        for w in (long_, short)
        if _seconds(w) not in present
    ]
    if files and missing:
        errs.append(
            f"the `{d.slo_profile}` profile windows {sorted(set(missing))} appear in no rule "
            f"(want {', '.join(f'{a}/{b} at {f:g}x' for a, b, f in windows)})"
        )
    return errs


# ---------------------------------------------------------------------------
# grading


def query_range(
    base: str, query: str, start: float, end: float, step: float
) -> tuple[list | None, str]:
    from . import web

    r = web.get(
        f"{base.rstrip('/')}/api/v1/query_range",
        {
            "query": query,
            "start": f"{start:.3f}",
            "end": f"{end:.3f}",
            "step": f"{step:g}",
        },
        timeout=20,
    )
    if r.status != 200:
        return None, r.error or f"HTTP {r.status}: {r.text[:200]}"
    try:
        return r.json()["data"]["result"] or [], ""
    except (ValueError, KeyError, TypeError):
        return None, f"unexpected Prometheus reply: {r.text[:200]}"


def true_times(series: list) -> list[float]:
    return sorted({float(v[0]) for s in series for v in s.get("values", [])})


def final_run_start(times: list[float], end: float, step: float) -> float | None:
    """Start of the continuous run of true samples that reaches `end`."""
    if not times or end - times[-1] > 1.5 * step:
        return None
    start = times[-1]
    for t in reversed(times[:-1]):
        if start - t > 1.5 * step:
            break
        start = t
    return start


def postmortem_errors(
    learner: Path, spec: dict, what: str = "postmortem"
) -> tuple[Path | None, list[str]]:
    pat = str(spec.get("path", ""))
    if not pat:
        return None, []
    glob = pat.replace("{date}", "*")
    hits = sorted(learner.glob(glob), key=lambda p: p.stat().st_mtime)
    if not hits:
        return None, [f"no {what} at {pat}"]
    p = hits[-1]
    text = p.read_text(errors="replace")
    heads = {
        m.group(1).strip().lower()
        for m in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M)
    }
    missing = [s for s in spec.get("sections", []) if s.lower() not in heads]
    return p, [
        f"{p.relative_to(learner)} lacks section(s): {', '.join(missing)}"
    ] if missing else []
