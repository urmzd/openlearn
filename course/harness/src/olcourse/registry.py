"""The module registry: course/modules/<ID>.toml (DESIGN 3.4).

`load(course)` parses and validates every file, `tsv(reg)` renders the
generated modules.tsv, and `invariants(reg)` returns the 3.4 rule violations
(mirror rule, pass order, call sites, ownership, upgrade chains).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import HarnessError, ids

TSV_COLUMNS = (
    "id",
    "kind",
    "lang",
    "pass",
    "chapter",
    "units",
    "deps",
    "used_by",
    "milestone",
)


@dataclass
class Module:
    id: str
    title: str
    kind: str
    lang: list[str]
    pass_: int
    chapter: str = ""
    contract: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    reading: list[str] = field(default_factory=list)
    used_by: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    upgrades: list[str] = field(default_factory=list)
    milestone: str = ""
    ci: str = "pr"
    tests: dict = field(default_factory=dict)
    fixtures: dict = field(default_factory=dict)
    learner_tests: dict | None = None
    # Learner-repo files or directories a practice module or drill grades
    # (primers/<id>/, deploy/, docs/adr/): hashed into its verdict's tree so
    # a pass goes stale when they change, and their presence counts as
    # started (DEVIATIONS I16).
    artifacts: list[str] = field(default_factory=list)
    # [bench]: a course perf budget relative to machine calibration (5.11)
    bench: dict = field(default_factory=dict)
    source: Path | None = None

    @property
    def owned(self) -> list[str]:
        """Every unit this module writes: its own plus the ones it takes over."""
        return self.units + self.upgrades

    @property
    def smoke(self) -> list[str]:
        return list(self.tests.get("smoke", []))

    @property
    def timeout_s(self) -> int:
        return int(self.tests.get("timeout_s", 120))

    @property
    def determinism_runs(self) -> int:
        return int(self.tests.get("determinism_runs", 2))

    def reading_ids(self) -> list[str]:
        return [r for r in self.reading if ids.is_module_id(r)]


@dataclass
class Registry:
    course: Path
    modules: dict[str, Module]

    def get(self, mid: str) -> Module:
        try:
            return self.modules[mid]
        except KeyError:
            raise HarnessError(
                f"no module {mid!r} in {self.course / 'modules'}"
            ) from None

    def ordered(self) -> list[Module]:
        """Pass order, and within it every module after its deps (so `ol check
        --all` never meets a dep it has not checked yet); ties by natural id."""
        base = sorted(self.modules.values(), key=lambda m: (m.pass_, _natural(m.id)))
        out: dict[str, Module] = {}

        def visit(m: Module, stack: tuple[str, ...]) -> None:
            if m.id in out or m.id in stack:
                return
            for d in sorted(
                (self.modules[x] for x in m.deps if x in self.modules),
                key=lambda x: (x.pass_, _natural(x.id)),
            ):
                visit(d, stack + (m.id,))
            out[m.id] = m

        for m in base:
            visit(m, ())
        return list(out.values())

    def closure(self, mid: str) -> list[str]:
        """Transitive deps of mid (excluding mid), dependencies first."""
        seen: dict[str, None] = {}

        def visit(x: str, stack: tuple[str, ...]) -> None:
            if x in stack:
                raise HarnessError("dependency cycle: " + " -> ".join(stack + (x,)))
            for d in self.get(x).deps:
                if d not in seen and d in self.modules:
                    visit(d, stack + (x,))
                    seen[d] = None

        visit(mid, ())
        return list(seen)

    def dependents(self, mid: str) -> list[str]:
        """Reverse transitive deps."""
        out: list[str] = []
        frontier = [mid]
        while frontier:
            x = frontier.pop()
            for m in self.modules.values():
                if x in m.deps and m.id not in out:
                    out.append(m.id)
                    frontier.append(m.id)
        return out

    # -- unit ownership ------------------------------------------------------

    def unit_chain(self, unit: str) -> list[str]:
        """Owners of a unit in order: the module listing it in `units`, then
        every module that `upgrades` it, ordered by pass."""
        first = [m.id for m in self.modules.values() if unit in m.units]
        ups = [m for m in self.modules.values() if unit in m.upgrades]
        ups.sort(key=lambda m: (m.pass_, _natural(m.id)))
        return first[:1] + [m.id for m in ups]

    def all_units(self) -> dict[str, list[str]]:
        units: dict[str, list[str]] = {}
        for m in self.modules.values():
            for u in m.owned:
                units.setdefault(u, [])
        return {u: self.unit_chain(u) for u in sorted(units)}

    def current_owner(self, unit: str) -> str:
        return self.unit_chain(unit)[-1]

    def superseded_by(self, mid: str) -> dict[str, str]:
        """Units mid owned that a later module took over: unit -> later id."""
        out = {}
        for u in self.get(mid).owned:
            chain = self.unit_chain(u)
            if mid in chain and chain.index(mid) < len(chain) - 1:
                out[u] = chain[chain.index(mid) + 1]
        return out


def _natural(mid: str) -> tuple:
    import re

    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", mid))


_LIST_FIELDS = (
    "lang",
    "contract",
    "deps",
    "reading",
    "used_by",
    "units",
    "upgrades",
    "artifacts",
)


def parse(path: Path) -> Module:
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise HarnessError(f"{path}: {e}") from None
    errs = []
    for req in ("id", "title", "kind", "lang", "pass"):
        if req not in raw:
            errs.append(f"missing `{req}`")
    if errs:
        raise HarnessError(f"{path}: " + "; ".join(errs))
    mid = raw["id"]
    for f in _LIST_FIELDS:
        v = raw.get(f, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            errs.append(f"`{f}` must be a list of strings")
    if errs:
        raise HarnessError(f"{path}: " + "; ".join(errs))
    try:
        lists = {f: ids.expand(raw.get(f, [])) for f in _LIST_FIELDS}
    except ValueError as e:
        raise HarnessError(f"{path}: {e}") from None
    m = Module(
        id=mid,
        title=raw["title"],
        kind=raw["kind"],
        lang=lists["lang"],
        pass_=raw["pass"],
        chapter=raw.get("chapter", ""),
        contract=lists["contract"],
        deps=lists["deps"],
        reading=lists["reading"],
        used_by=lists["used_by"],
        units=lists["units"],
        upgrades=lists["upgrades"],
        milestone=raw.get("milestone", ""),
        ci=raw.get("ci", "pr"),
        tests=raw.get("tests", {}),
        fixtures=raw.get("fixtures", {}),
        learner_tests=raw.get("learner_tests"),
        artifacts=lists["artifacts"],
        bench=dict(raw.get("bench", {})),
        source=path,
    )
    if path.stem != mid:
        errs.append(f"file name {path.name} does not match id {mid!r}")
    if not ids.is_module_id(mid):
        errs.append(f"id {mid!r} does not match the course id grammar (DESIGN 3.5)")
    if m.kind not in ids.MODULE_KINDS:
        errs.append(f"kind {m.kind!r} not in {ids.MODULE_KINDS}")
    bad = [x for x in m.lang if x not in ids.LANGS]
    if bad:
        errs.append(f"lang {bad} not in {ids.LANGS}")
    if not isinstance(m.pass_, int) or m.pass_ < 0:
        errs.append("`pass` must be a non-negative integer")
    if m.ci not in ids.CI_TIERS:
        errs.append(f"ci {m.ci!r} not in {ids.CI_TIERS}")
    bad = [a for a in m.artifacts if a.startswith("/") or ".." in a.split("/")]
    if bad:
        errs.append(f"artifacts {bad} must be relative paths inside the learner repo")
    if m.artifacts and m.kind not in ("practice", "drill"):
        errs.append(
            "`artifacts` is for practice modules and drills; build units go in `units`"
        )
    if m.bench:
        missing = [k for k in ("lang", "name", "metric", "budget") if k not in m.bench]
        if missing:
            errs.append(f"[bench] needs {', '.join(missing)}")
        elif m.bench["lang"] not in ids.CODE_LANGS:
            errs.append(f"[bench].lang {m.bench['lang']!r} not in {ids.CODE_LANGS}")
    if m.milestone and not m.milestone.startswith("MS-"):
        errs.append(f"milestone {m.milestone!r} is not an MS- id")
    if errs:
        raise HarnessError(f"{path}: " + "; ".join(errs))
    return m


def load(course: Path) -> Registry:
    mods: dict[str, Module] = {}
    d = course / "modules"
    for p in sorted(d.glob("*.toml")) if d.is_dir() else []:
        m = parse(p)
        mods[m.id] = m
    return Registry(course=course, modules=mods)


# ---------------------------------------------------------------------------
# modules.tsv


def tsv(reg: Registry) -> str:
    rows = [
        "# "
        + "\t".join(TSV_COLUMNS)
        + "\t(generated from modules/*.toml by `ol lint --fix-index`; do not edit)"
    ]
    for m in reg.ordered():
        rows.append(
            "\t".join(
                [
                    m.id,
                    m.kind,
                    ",".join(m.lang),
                    str(m.pass_),
                    m.chapter or "-",
                    ",".join(m.owned) or "-",
                    ",".join(m.deps) or "-",
                    ",".join(m.used_by) or "-",
                    m.milestone or "-",
                ]
            )
        )
    return "\n".join(rows) + "\n"


def tsv_current(reg: Registry) -> bool:
    p = reg.course / "modules.tsv"
    return p.is_file() and p.read_text() == tsv(reg)


def write_tsv(reg: Registry) -> Path:
    p = reg.course / "modules.tsv"
    p.write_text(tsv(reg))
    return p


# ---------------------------------------------------------------------------
# invariants (DESIGN 3.4, verify check 13; call site rule is check 9)


def invariants(reg: Registry) -> list[str]:
    errs: list[str] = []
    mods = reg.modules

    def known(mid: str, where: str, field_: str) -> bool:
        if mid not in mods:
            errs.append(f"{where}: `{field_}` names unknown module {mid}")
            return False
        return True

    for m in reg.ordered():
        for d in m.deps:
            if known(d, m.id, "deps") and mods[d].pass_ > m.pass_:
                errs.append(
                    f"{m.id} (pass {m.pass_}): dep {d} is taught later (pass {mods[d].pass_})"
                )
        for r in m.reading_ids():
            if known(r, m.id, "reading") and mods[r].pass_ > m.pass_:
                errs.append(
                    f"{m.id} (pass {m.pass_}): reading {r} is taught later (pass {mods[r].pass_})"
                )
        if m.id in m.deps:
            errs.append(f"{m.id}: depends on itself")
        for u in m.used_by:
            if not ids.is_module_id(u):
                errs.append(
                    f"{m.id}: used_by {u} is not a module id (milestones are not call sites)"
                )
            else:
                known(u, m.id, "used_by")

    # Mirror rule, with the upgrade inheritance exemption.
    for m in reg.ordered():
        if m.kind != "build":
            continue
        mirror = {x.id for x in mods.values() if x.kind != "side" and m.id in x.deps}
        inherited: set[str] = set()
        for u in m.upgrades:
            for prev in reg.unit_chain(u):
                if prev == m.id:
                    break
                inherited |= {
                    x.id for x in mods.values() if x.kind != "side" and prev in x.deps
                }
                inherited |= set(mods[prev].used_by) if prev in mods else set()
        inherited -= {m.id}
        declared = set(m.used_by)
        missing = sorted(mirror - declared)
        extra = sorted(declared - mirror - inherited)
        if missing:
            errs.append(
                f"{m.id}: used_by is missing {missing} (they list {m.id} in deps)"
            )
        if extra:
            errs.append(
                f"{m.id}: used_by lists {extra}, whose deps do not contain {m.id}"
            )
        for u in declared - inherited:
            if u in mods and mods[u].pass_ < m.pass_:
                errs.append(
                    f"{m.id} (pass {m.pass_}): call site {u} comes earlier (pass {mods[u].pass_})"
                )

    # Ownership: one current owner per unit; upgrades take over an existing unit.
    first: dict[str, list[str]] = {}
    for m in mods.values():
        for u in m.units:
            first.setdefault(u, []).append(m.id)
        if set(m.units) & set(m.upgrades):
            errs.append(f"{m.id}: a unit is in both units and upgrades")
    for u, owners in first.items():
        if len(owners) > 1:
            errs.append(
                f"unit {u} is listed in `units` of {sorted(owners)}; one owner, later ones use `upgrades`"
            )
    for m in mods.values():
        for u in m.upgrades:
            if u not in first:
                errs.append(f"{m.id}: upgrades {u}, which no module lists in `units`")
                continue
            chain = reg.unit_chain(u)
            passes = [mods[c].pass_ for c in chain]
            if passes != sorted(passes):
                errs.append(f"unit {u}: owner chain {chain} is not in pass order")

    # Cycles.
    for m in mods.values():
        try:
            reg.closure(m.id)
        except HarnessError as e:
            errs.append(str(e))
            break
    return sorted(set(errs), key=errs.index)


def call_site_errors(reg: Registry, m: Module) -> list[str]:
    """Verify check 9: a build module needs a module call site that is not a side quest."""
    if m.kind != "build":
        return []
    real = [u for u in m.used_by if u in reg.modules and reg.modules[u].kind != "side"]
    return (
        []
        if real
        else [
            f"{m.id}: build module has no call site (used_by needs a non-side module)"
        ]
    )
