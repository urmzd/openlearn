"""ol doctor [--pass N] [--json]   check the toolchain the course needs (DESIGN 5.3)

Required from Pass 0: git, uv, python >= 3.11, cc, make.
Required from Pass 1 (the tracer: Rust, Go, and kind): cargo, go, docker with
a running daemon, kubectl, kind, helm. From Pass 7: tilt (dep.04), and Docker
gets at least 6 CPUs and 12 GiB (2.13). Optional: protoc (lang.10 only; contracts ship generated code),
a Rust nightly (Miri, TSan), gh (ci-status on a GitHub remote).

The pass defaults to the highest pass among the modules you have started
(0 with no learner repo). Exit 0 when every required tool is present, 5 when
one is missing or too small."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys

from .. import EXIT_HARNESS, EXIT_PASS, ctx, learner, registry, tree

MIN_CPUS, MIN_MEM_GIB = 6, 12


def _version(cmd: list[str]) -> tuple[bool, str]:
    if shutil.which(cmd[0]) is None:
        return False, "not installed"
    rc, out = ctx.run(cmd, timeout=20)
    first = next((x.strip() for x in out.splitlines() if x.strip()), "")
    return rc == 0, first[
        :80
    ] if rc == 0 else f"`{' '.join(cmd)}` exited {rc}: {first[:80]}"


def checks(pass_: int) -> list[dict]:
    rows: list[dict] = []

    def add(name: str, need_from: int | None, ok: bool, detail: str) -> None:
        required = need_from is not None and pass_ >= need_from
        rows.append(
            {
                "tool": name,
                "ok": ok,
                "required": required,
                "from_pass": need_from,
                "detail": detail,
            }
        )

    py_ok = sys.version_info >= (3, 11)
    add(
        "python",
        0,
        py_ok,
        f"{sys.version.split()[0]}" + ("" if py_ok else " (need >= 3.11)"),
    )
    for name, from_pass, cmd in (
        ("uv", 0, ["uv", "--version"]),
        ("git", 0, ["git", "--version"]),
        ("cc", 0, ["cc", "--version"]),
        ("make", 0, ["make", "--version"]),
        ("cargo", 1, ["cargo", "--version"]),
        ("go", 1, ["go", "version"]),
    ):
        add(name, from_pass, *_version(cmd))

    dok, dver = _version(["docker", "version", "--format", "{{.Server.Version}}"])
    if not dok and shutil.which("docker"):
        dver = "the daemon is not running (start Docker Desktop or colima)"
    add("docker", 1, dok, dver)
    if dok:
        rc, out = ctx.run(
            ["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}}"], timeout=20
        )
        m = re.match(r"\s*(\d+)\s+(\d+)", out)
        if rc == 0 and m:
            cpus, gib = int(m.group(1)), int(m.group(2)) / 2**30
            big = cpus >= MIN_CPUS and gib >= MIN_MEM_GIB - 0.5
            add(
                "docker resources",
                7,
                big,
                f"{cpus} CPUs, {gib:.1f} GiB"
                + (
                    ""
                    if big
                    else f" (Pass 7 needs {MIN_CPUS} CPUs and {MIN_MEM_GIB} GiB)"
                ),
            )
        else:
            add("docker resources", 7, False, "docker info printed no CPU and memory")
    for name, cmd in (
        ("kubectl", ["kubectl", "version", "--client"]),
        ("kind", ["kind", "version"]),
        ("helm", ["helm", "version", "--short"]),
    ):
        add(name, 1, *_version(cmd))
    add("tilt", 7, *_version(["tilt", "version"]))
    add("protoc", None, *_version(["protoc", "--version"]))
    add("gh", None, *_version(["gh", "--version"]))
    if shutil.which("rustup"):
        rc, out = ctx.run(["rustup", "toolchain", "list"], timeout=20)
        night = [x.split()[0] for x in out.splitlines() if x.startswith("nightly")]
        add(
            "rust nightly",
            None,
            bool(night),
            night[0] if night else "no nightly toolchain (Miri, TSan)",
        )
    else:
        add("rust nightly", None, False, "rustup not installed")
    return rows


def current_pass() -> int:
    h = ctx.learner_home()
    if not (h / "system.toml").is_file():
        return 0
    try:
        ct = tree.resolve(h)
        reg = registry.load(ct.path)
    except Exception:  # doctor must report, not crash, on a broken setup
        return 0
    started = [
        m.pass_ for m in reg.modules.values() if learner.started(h, ct.path, reg, m.id)
    ]
    return max(started, default=0)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ol doctor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pass", dest="pass_", type=int)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    p = a.pass_ if a.pass_ is not None else current_pass()
    rows = checks(p)
    bad = [x for x in rows if x["required"] and not x["ok"]]
    if a.json:
        print(json.dumps({"pass": p, "ok": not bad, "checks": rows}, indent=1))
        return EXIT_HARNESS if bad else EXIT_PASS
    ctx.say(f"{ctx.BLD}ol doctor{ctx.RST}  (requirements for pass {p})")
    for x in rows:
        if x["ok"]:
            mark = f"{ctx.GRN}ok  {ctx.RST}"
        elif x["required"]:
            mark = f"{ctx.RED}FAIL{ctx.RST}"
        else:
            mark = f"{ctx.DIM}--  {ctx.RST}"
        when = "optional" if x["from_pass"] is None else f"pass {x['from_pass']}+"
        ctx.say(f"  {mark} {x['tool']:<17} {ctx.DIM}{when:<9}{ctx.RST} {x['detail']}")
    if bad:
        ctx.say(
            f"{ctx.RED}missing for pass {p}:{ctx.RST} {', '.join(x['tool'] for x in bad)}"
        )
        return EXIT_HARNESS
    ctx.say(f"{ctx.GRN}ready for pass {p}{ctx.RST}")
    return EXIT_PASS
