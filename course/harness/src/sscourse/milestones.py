"""Milestone specs: course/milestones/<MS-ID>.toml (DESIGN 5.7, D25).

    id       = "MS-P1"
    title    = "..."
    requires = ["rt.01", "L10.0"]      # module (or milestone) ids that must pass first
    pass     = 1
    ci       = "pr"
    includes = ["MS-L0"]               # part and component milestones a pass gate composes
    services = ["engine", "gateway"]   # [services.*] the steps need, unless a step says otherwise

    [[step]]
    name      = "stream through the gateway"
    run       = "tinyllm"              # an [entry] role; or `http = {...}`; or neither for
                                       # suite, promql, trace, git-log, ci-status, file-produced
    argv      = ["generate", "--prompt", "{fixture:...}"]
    http      = { method = "POST", url = "{gateway.api_base}/completions", json = {...}, auth = true }
    smoke     = true                   # rerun by every later pass gate and by --smoke
    ci        = "pr"                   # pr | nightly | local | kind
    services  = ["engine"]
    matrix    = { cache = ["none", "paged"] }
    vars      = { steps = "20000" }    # `{steps}`; smoke_vars = { steps = "200" } under --smoke
    expect    = { match = "tokens-equal", file = "course/fixtures/..." }
    timeout_s = 120
    sink      = "effects"              # start the effects sink for this step's services:
                                       # {effects.url}, TL_TEST_EFFECTS_URL in their env
    chaos     = { kill = ["durable", "worker"], every_s = "1..4", count = 20, seed = 1 }
                                       # seeded SIGKILL + restart while `run` runs (chaos.py)

A pass gate `MS-P<n>` also reruns the smoke steps of every earlier pass gate:
the spiral invariant (7.2) made executable.
"""

from __future__ import annotations

import itertools
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, chaos, ids
from .matchers import MATCHERS, NO_COMMAND

STEP_CI = ("pr", "nightly", "local", "kind")
SINKS = ("effects",)
GATE = re.compile(r"^MS-P(\d+)$")


@dataclass
class Step:
    name: str
    origin: str  # the milestone that declares it
    run: str | None = None
    argv: list[str] = field(default_factory=list)
    http: dict | None = None
    smoke: bool = False
    ci: str = "pr"
    services: list[str] = field(default_factory=list)
    matrix: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)
    timeout_s: float = 120.0
    queue: str = "default"
    model_dir: str | None = None
    env: dict = field(default_factory=dict)
    vars: dict = field(default_factory=dict)  # `{name}` placeholders of this step
    smoke_vars: dict = field(
        default_factory=dict
    )  # their values under --smoke (C1: 200 steps)
    sink: str | None = None  # "effects": the runner starts the effects sink
    chaos: chaos.Spec | None = None  # seeded SIGKILLs while the command runs

    @property
    def kind(self) -> bool:
        return self.ci == "kind"

    def qualified(self, target: str) -> str:
        return self.name if self.origin == target else f"{self.origin}/{self.name}"

    def variants(self) -> list[dict]:
        """The matrix expanded into one dict of placeholder values per run."""
        if not self.matrix:
            return [{}]
        keys = sorted(self.matrix)
        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*(self.matrix[k] for k in keys))
        ]


@dataclass
class Milestone:
    id: str
    title: str
    requires: list[str]
    pass_: int
    ci: str
    includes: list[str]
    services: list[str]
    steps: list[Step]
    path: Path


def path_of(course: Path, msid: str) -> Path:
    return course / "milestones" / f"{msid}.toml"


def load(course: Path, msid: str) -> Milestone:
    if not msid.startswith("MS-") or not ids.is_course_id(msid):
        raise HarnessError(f"{msid!r} is not a milestone id (MS-<name>, DESIGN 3.5)")
    p = path_of(course, msid)
    if not p.is_file():
        raise HarnessError(f"no milestone {msid}: {p} does not exist")
    try:
        raw = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(f"{p}: {e}") from None
    if raw.get("id", msid) != msid:
        raise HarnessError(f"{p}: id {raw.get('id')!r} does not match the file name")
    services = list(raw.get("services", []))
    steps, seen = [], set()
    for i, s in enumerate(raw.get("step", [])):
        name = s.get("name") or f"step {i + 1}"
        if name in seen:
            raise HarnessError(f"{p}: two steps are named {name!r}")
        seen.add(name)
        st = Step(
            name=name,
            origin=msid,
            run=s.get("run"),
            argv=list(s.get("argv", [])),
            http=s.get("http"),
            smoke=bool(s.get("smoke", False)),
            ci=s.get("ci", raw.get("ci", "pr")),
            services=list(s.get("services", services)),
            matrix=dict(s.get("matrix", {})),
            expect=dict(s.get("expect", {"match": "exit-code"})),
            timeout_s=float(s.get("timeout_s", 120)),
            queue=s.get("queue", "default"),
            model_dir=s.get("model_dir"),
            env=dict(s.get("env", {})),
            vars={k: str(v) for k, v in dict(s.get("vars", {})).items()},
            smoke_vars={k: str(v) for k, v in dict(s.get("smoke_vars", {})).items()},
            sink=s.get("sink"),
        )
        if "history" in s:
            raise HarnessError(
                f"{p} step {name!r}: `history` (a recorded client history) is not built yet"
            )
        if "chaos" in s:
            st.chaos = chaos.parse(s["chaos"], f"{p} step {name!r}")
        _check_step(p, st)
        steps.append(st)
    return Milestone(
        msid,
        raw.get("title", ""),
        list(raw.get("requires", [])),
        int(raw.get("pass", 0)),
        raw.get("ci", "pr"),
        list(raw.get("includes", [])),
        services,
        steps,
        p,
    )


def _check_step(p: Path, s: Step) -> None:
    where = f"{p} step {s.name!r}"
    if s.ci not in STEP_CI:
        raise HarnessError(f"{where}: ci {s.ci!r} is not one of {', '.join(STEP_CI)}")
    if s.run and s.http:
        raise HarnessError(f"{where}: give `run` or `http`, not both")
    if s.http is not None and not (isinstance(s.http, dict) and s.http.get("url")):
        raise HarnessError(f"{where}: `http` needs a `url`")
    m = s.expect.get("match", "exit-code")
    if m not in MATCHERS:
        raise HarnessError(f"{where}: unknown matcher {m!r}")
    if not s.run and not s.http and m not in NO_COMMAND:
        raise HarnessError(
            f"{where}: matcher {m} needs a command (`run`) or a request (`http`)"
        )
    for k, v in s.matrix.items():
        if not isinstance(v, list) or not v:
            raise HarnessError(f"{where}: matrix.{k} must be a non-empty list")
    extra = sorted(set(s.smoke_vars) - set(s.vars))
    if extra:
        raise HarnessError(f"{where}: smoke_vars {extra} override no `vars` entry")
    if s.sink is not None and s.sink not in SINKS:
        raise HarnessError(f"{where}: sink {s.sink!r} is not one of {', '.join(SINKS)}")
    if m == "effects" and s.sink != "effects":
        raise HarnessError(f'{where}: the effects matcher needs sink = "effects"')
    if s.chaos is not None and (not s.run or s.kind):
        raise HarnessError(
            f"{where}: chaos needs a local command (`run`) to run the kills against"
        )
    if m == "perf" and s.ci != "local":
        raise HarnessError(
            f'{where}: a perf step is ci = "local" (benchmarks never run in CI)'
        )


def all_ids(course: Path) -> list[str]:
    d = course / "milestones"
    return sorted(p.stem for p in d.glob("MS-*.toml")) if d.is_dir() else []


@dataclass
class Plan:
    target: Milestone
    steps: list[Step]
    requires: list[str]
    sources: list[str]  # every milestone contributing steps


def plan(course: Path, msid: str, smoke: bool) -> Plan:
    target = load(course, msid)
    steps: list[Step] = []
    requires: list[str] = []
    sources: list[str] = []
    seen: set[tuple[str, str]] = set()

    def add(ms: Milestone, only_smoke: bool, spiral: bool) -> None:
        if ms.id not in sources:
            sources.append(ms.id)
        for r in ms.requires:
            if r not in requires:
                requires.append(r)
        for s in ms.steps:
            if only_smoke and not s.smoke:
                continue
            # An earlier gate's nightly-tier smoke steps (MS-C1's training
            # run under MS-P9) are the nightly job's; a PR-tier gate that
            # reruns that gate's smoke steps leaves them to it.
            if spiral and target.ci == "pr" and s.ci == "nightly":
                continue
            if (s.origin, s.name) not in seen:
                seen.add((s.origin, s.name))
                steps.append(s)

    def visit(
        ms: Milestone, only_smoke: bool, stack: tuple[str, ...], spiral: bool = False
    ) -> None:
        if ms.id in stack:
            raise HarnessError(
                "milestone include cycle: " + " -> ".join(stack + (ms.id,))
            )
        add(ms, only_smoke, spiral)
        for inc in ms.includes:
            visit(load(course, inc), only_smoke, stack + (ms.id,), spiral)
        g = GATE.match(ms.id)
        if g:  # the spiral: rerun the smoke steps of every earlier pass gate
            for k in range(int(g.group(1))):
                gid = f"MS-P{k}"
                if path_of(course, gid).is_file() and gid not in stack:
                    visit(load(course, gid), True, stack + (ms.id,), True)

    visit(target, False, ())
    if smoke:
        steps = [s for s in steps if s.smoke and not s.kind]
    return Plan(target, steps, requires, sources)
