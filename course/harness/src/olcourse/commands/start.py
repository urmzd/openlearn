"""ol start <ID>   stub the module's units into your repo (DESIGN 5.3)

Never overwrites a file that exists. Library manifests (python/pyproject.toml,
rust/Cargo.toml and crate manifests, go/go.mod) are copied from the
reference only when absent and are never rewritten afterwards. A Rust crate
root's `mod` declarations and a Go package's sibling units get stubs, so the
crate or package always compiles. For a unit the module takes over
(`upgrades`), the contract diff is printed and you edit your file in place."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from .. import EXIT_HARNESS, ctx, ledger, markers, solve, units
from ..session import open_session


def start_solve(s, m) -> int:
    """A solve set: write the answer template and an empty file per proof."""
    qs = solve.load_key(s.course, m.id)
    made: list[str] = []
    rel = f"solve/{m.id}.toml"
    _write(s.learner, rel, solve.template(qs, m.id), made)
    for q in qs:
        if q.type == "proof":
            _write(
                s.learner,
                f"solve/{m.id}/{q.qid}.md",
                f"# {m.id} {q.qid}\n\nWrite your proof here.\n",
                made,
            )
    ledger.event(s.learner, m.id, "start")
    ctx.say(f"{ctx.GRN}started{ctx.RST} {m.id}  {m.title}  ({m.kind}, pass {m.pass_})")
    for r in made:
        ctx.say(f"  wrote     {r}")
    probs = solve.solve_dir(s.course, m.id) / "problems.md"
    ctx.say(f"  problems  {probs.relative_to(s.course.parent)}")
    if m.chapter:
        ctx.say(f"  chapter   {m.chapter}")
    ctx.say(f"  check     ol check {m.id}  (answers in {rel}; proofs are self-graded)")
    return 0


def _write(learner: Path, rel: str, text: str | bytes, made: list[str]) -> None:
    p = learner / rel
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text if isinstance(text, bytes) else text.encode())
    made.append(rel)


def _manifests(course: Path, learner: Path, owned: list[str], made: list[str]) -> None:
    ref = course / "ref"
    langs = {markers.lang_of(u) for u in owned}
    if "python" in langs and (ref / "python" / "pyproject.toml").is_file():
        _write(
            learner,
            "python/pyproject.toml",
            (ref / "python" / "pyproject.toml").read_bytes(),
            made,
        )
    if "rust" in langs:
        for name in ("Cargo.toml", ".cargo/config.toml"):
            if (ref / "rust" / name).is_file():
                _write(
                    learner, f"rust/{name}", (ref / "rust" / name).read_bytes(), made
                )
        for u in owned:
            parts = u.split("/")
            if len(parts) >= 4 and parts[0] == "rust" and parts[1] == "crates":
                for name in ("Cargo.toml", "build.rs"):
                    src = ref / "rust" / "crates" / parts[2] / name
                    if src.is_file():
                        _write(
                            learner,
                            f"rust/crates/{parts[2]}/{name}",
                            src.read_bytes(),
                            made,
                        )
    if "go" in langs:
        for name in ("go.mod", "go.sum"):
            if (ref / "go" / name).is_file():
                _write(learner, f"go/{name}", (ref / "go" / name).read_bytes(), made)


def _glue(s, learner: Path, made: list[str]) -> None:
    """Stubs for units a present crate root or Go package needs to compile."""
    all_units = s.reg.all_units()
    want: set[str] = set()
    for root in (
        list((learner / "rust").rglob("*.rs")) if (learner / "rust").is_dir() else []
    ):
        if "target" in root.parts or root.name not in ("lib.rs", "mod.rs", "main.rs"):
            continue
        for m in re.finditer(
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", root.read_text(), re.M
        ):
            for cand in (
                root.parent / f"{m.group(1)}.rs",
                root.parent / m.group(1) / "mod.rs",
            ):
                want.add(cand.relative_to(learner).as_posix())
    for u in all_units:
        if markers.lang_of(u) == "go" and (learner / u).parent.is_dir():
            if any(p.suffix == ".go" for p in (learner / u).parent.iterdir()):
                want.add(u)
    for u in sorted(want):
        if u in all_units and not (learner / u).exists():
            _write(
                learner, u, units.stub_text(s.course, s.reg, u, all_units[u][-1]), made
            )


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return EXIT_HARNESS
    s = open_session()
    m = s.module(argv[0])
    if m.kind in ("solve", "proof"):
        return start_solve(s, m)
    made: list[str] = []
    kept: list[str] = []
    for u in m.units:
        if (s.learner / u).exists():
            kept.append(u)
        else:
            _write(s.learner, u, units.stub_text(s.course, s.reg, u, m.id), made)
    for u in m.upgrades:
        chain = s.reg.unit_chain(u)
        prev = chain[chain.index(m.id) - 1]
        if not (s.learner / u).exists():
            _write(s.learner, u, units.stub_text(s.course, s.reg, u, m.id), made)
            continue
        kept.append(u)
        a = units.stub_text(s.course, s.reg, u, prev).splitlines()
        b = units.stub_text(s.course, s.reg, u, m.id).splitlines()
        ctx.say(
            f"{ctx.YEL}{m.id} takes over {u} from {prev}{ctx.RST}; contract changes (edit your file in place):"
        )
        diff = list(
            difflib.unified_diff(a, b, f"{prev}/{u}", f"{m.id}/{u}", lineterm="")
        )
        ctx.say(ctx.indent("\n".join(diff) if diff else "(no signature changes)"))
    _manifests(s.course, s.learner, m.owned, made)
    _glue(s, s.learner, made)
    ledger.event(s.learner, m.id, "start")

    ctx.say(
        f"{ctx.GRN}started{ctx.RST} {m.id}  {m.title}  ({m.kind}, {', '.join(m.lang)}, pass {m.pass_})"
    )
    for rel in made:
        ctx.say(f"  wrote     {rel}")
    for rel in kept:
        ctx.say(f"  kept      {rel}  {ctx.DIM}(exists; never overwritten){ctx.RST}")
    if m.chapter:
        ctx.say(f"  chapter   {m.chapter}")
    for c in m.contract:
        ctx.say(
            f"  contract  {c}  {ctx.DIM}(your copy: contracts/{c.removeprefix('contracts/')}){ctx.RST}"
        )
    if m.deps:
        ctx.say(f"  needs     {', '.join(m.deps)}")
    if m.learner_tests:
        lt = m.learner_tests
        ctx.say(
            f"  your tests {lt.get('path')}  rung R{lt.get('rung')}, mutation threshold {lt.get('threshold')}"
        )
    ctx.say(f"  tests     ol tests {m.id}")
    if m.kind == "drill":
        from .. import drills

        name = next((d.name for d in drills.all_drills(s.course) if d.id == m.id), m.id)
        ctx.say(f"  run       ol drill start {name}, then ol drill end (graded there)")
    else:
        ctx.say(f"  check     ol check {m.id}")
    return 0
