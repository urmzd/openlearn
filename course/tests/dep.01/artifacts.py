"""dep.01 artifact check: production Dockerfiles for your gateway and engine.

Run by `ol check dep.01` in your repo. Two tiers, in order:

  static   the dep.00 image rules still hold; every base image is pinned by
           digest; a HEALTHCHECK in exec form on the health port; the OCI labels;
           a static (CGO-free) gateway build; no secrets in the Dockerfiles, and
           a .dockerignore that keeps .env files and keys out of the context
  docker   both images build with VERSION and REVISION build args, carry the
           labels, run as a numeric non-root user, hold no secret in their
           history or environment, fit the size budget (twice the reference
           image), and, run the way the charts run them, report healthy and
           stream a completion engine -> gateway

There is no cluster tier: dep.02 and dep.03 put these images on kind.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import Ctx, Fail, check_dockerfile, dockerfile_stages, run  # noqa: E402

PARTS = {"engine": ("8000", "9464"), "gateway": ("8080", "9464")}
LABELS = [
    "org.opencontainers.image.title",
    "org.opencontainers.image.source",
    "org.opencontainers.image.version",
    "org.opencontainers.image.revision",
]
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
SECRET_LINE = re.compile(
    r"^\s*(ENV|ARG)\s+[A-Za-z0-9_]*(KEY|SECRET|PEPPER|TOKEN|PASSWORD|CREDENTIAL)[A-Za-z0-9_]*\s*=\s*\S+",
    re.I,
)
SECRET_TEXT = [
    (re.compile(r"\btl_[a-z2-7]{12}_[A-Za-z0-9]{32}\b"), "a tinyllm API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "a private key"),
    (re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"), "an AWS access key id"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "a GitHub token"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "an API secret key"),
]
VERSION = "0.0.0-check"


def course_tree() -> Path:
    return Path(os.environ.get("OL_COURSE_TREE") or Path(__file__).resolve().parents[2])


def dockerfile(c: Ctx, part: str) -> str:
    return c.require_file(f"deploy/docker/{part}.Dockerfile").read_text()


def final_lines(text: str) -> list[str]:
    stages = dockerfile_stages(text)
    return stages[-1]["lines"] if stages else []


def labels_in(lines: list[str]) -> set[str]:
    keys: set[str] = set()
    for line in lines:
        if line.split(None, 1)[0].upper() == "LABEL":
            keys |= set(re.findall(r"([A-Za-z0-9_.-]+)\s*=", line))
    return keys


def secret_hits(text: str) -> list[str]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if SECRET_LINE.search(line):
            out.append(
                f"line {i}: a secret-looking ENV or ARG with a value: {line.strip()[:70]}"
            )
        for pat, what in SECRET_TEXT:
            if pat.search(line):
                out.append(f"line {i}: {what}")
    return out


# -- static -------------------------------------------------------------------


def test_layout_present(c: Ctx) -> None:
    # WHY: the two Dockerfiles and the context filter are where dep.00 left
    #      them; MS-prod and dep.03 build these exact files.
    # KIND: unit
    # CHAPTER: dep.01 section 4, The artifact and its check
    for rel in (
        "deploy/docker/engine.Dockerfile",
        "deploy/docker/gateway.Dockerfile",
        ".dockerignore",
    ):
        c.require_file(rel)


def test_image_rules_still_hold(c: Ctx) -> None:
    # WHY: dep.00's rules stay true as the images grow: a build stage the
    #      image never ships, a numeric non-root USER, exec-form ENTRYPOINT, and
    #      the ports the charts target.
    # KIND: unit
    # CHAPTER: dep.01 section 2.1
    errs = []
    for part, ports in PARTS.items():
        for p in ports:
            errs += check_dockerfile(dockerfile(c, part), f"{part}.Dockerfile", port=p)
    if errs:
        raise Fail("\n".join(dict.fromkeys(errs)))


def test_base_images_pinned_by_digest(c: Ctx) -> None:
    # WHY: a tag such as golang:1.25-alpine is re-pushed with every patch
    #      release, so the same Dockerfile builds a different image next week;
    #      name@sha256:<digest> is content-addressed and cannot move. Keep the
    #      tag in front of the digest for readers.
    # KIND: unit
    # CHAPTER: dep.01 section 5, Pitfall 1
    errs = []
    for part in PARTS:
        stages = dockerfile_stages(dockerfile(c, part))
        names = {s["as"] for s in stages if s["as"]}
        for s in stages:
            if s["from"] in names or s["from"] == "scratch":
                continue
            if not DIGEST.search(s["from"]):
                errs.append(
                    f"{part}.Dockerfile: FROM {s['from']} is not pinned by digest (name:tag@sha256:...)"
                )
    if errs:
        raise Fail("\n".join(errs))


def test_healthcheck_in_exec_form(c: Ctx) -> None:
    # WHY: a HEALTHCHECK lets Docker (and compose, and `docker ps`) see a hung
    #      server that is still running. Exec form runs without /bin/sh, which
    #      a minimal base may not have; --interval and --timeout bound it; it
    #      asks the health port (9464), where /healthz lives.
    # KIND: unit
    # CHAPTER: dep.01 section 2.3
    errs = []
    for part in PARTS:
        hc = [
            l
            for l in final_lines(dockerfile(c, part))
            if l.split(None, 1)[0].upper() == "HEALTHCHECK"
        ]
        if len(hc) != 1:
            errs.append(
                f"{part}.Dockerfile: the final stage has {len(hc)} HEALTHCHECK instructions; want one"
            )
            continue
        line = hc[0]
        m = re.search(r"\bCMD\s+(.*)$", line)
        if not m or not m.group(1).lstrip().startswith("["):
            errs.append(
                f'{part}.Dockerfile: HEALTHCHECK must be CMD in exec form: CMD ["...", ...]'
            )
        for flag in ("--interval", "--timeout"):
            if flag not in line:
                errs.append(f"{part}.Dockerfile: HEALTHCHECK has no {flag}")
        if "9464" not in line or "/healthz" not in line:
            errs.append(
                f"{part}.Dockerfile: HEALTHCHECK must ask /healthz on the health port 9464"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_oci_labels_declared(c: Ctx) -> None:
    # WHY: an image found in a registry or a cluster months later must say what
    #      it is and where it came from: the org.opencontainers.image title,
    #      source, version, and revision (the git commit), the last two from
    #      build args so CI can fill them.
    # KIND: unit
    # CHAPTER: dep.01 section 2.4
    errs = []
    for part in PARTS:
        lines = final_lines(dockerfile(c, part))
        missing = [k for k in LABELS if k not in labels_in(lines)]
        if missing:
            errs.append(
                f"{part}.Dockerfile: the final stage lacks LABEL {', '.join(missing)}"
            )
        args = {
            l.split(None, 1)[1].split("=")[0].strip()
            for l in lines
            if l.split(None, 1)[0].upper() == "ARG"
        }
        for a in ("VERSION", "REVISION"):
            if a not in args:
                errs.append(
                    f"{part}.Dockerfile: the final stage declares no ARG {a} (an ARG before FROM is not visible after it)"
                )
    if errs:
        raise Fail("\n".join(errs))


def test_gateway_builds_a_static_binary(c: Ctx) -> None:
    # WHY: with CGO_ENABLED=0 the Go binary links nothing from the build
    #      stage's libc, so it runs on any small base. The usage ledger (gw.07)
    #      uses the pure-Go SQLite driver for exactly this reason.
    # KIND: unit
    # CHAPTER: dep.01 section 2.2
    text = dockerfile(c, "gateway")
    if not any(
        "CGO_ENABLED=0" in l and "go build" in l
        for s in dockerfile_stages(text)
        for l in s["lines"]
    ):
        raise Fail("gateway.Dockerfile: build the gateway with CGO_ENABLED=0 go build")


def test_no_secrets_in_dockerfiles_or_context(c: Ctx) -> None:
    # WHY: every ENV and ARG value is stored in the image's history, readable
    #      by anyone who can pull it; a secret belongs in a Kubernetes Secret
    #      at run time. The build context must also keep .env files and keys
    #      out, or a COPY . sweeps them into a layer.
    # KIND: unit
    # CHAPTER: dep.01 section 5, Pitfall 4
    errs = []
    for part in PARTS:
        errs += [f"{part}.Dockerfile: {h}" for h in secret_hits(dockerfile(c, part))]
    ignore = {
        ln.strip().strip("/")
        for ln in c.require_file(".dockerignore").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    }
    for pat in (".env", ".git"):
        if not ({pat, f"**/{pat}", f"{pat}*", f"**/{pat}*"} & ignore):
            errs.append(f".dockerignore does not exclude {pat}")
    if not any(x.endswith(".pem") or x.endswith(".key") for x in ignore):
        errs.append(".dockerignore does not exclude key files (*.pem)")
    if errs:
        raise Fail("\n".join(errs))


# -- docker -------------------------------------------------------------------


def revision(c: Ctx) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=c.root, capture_output=True, text=True
    )
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "check"


def images(c: Ctx) -> dict[str, str]:
    if "images" not in c.cache:
        c.need_docker()
        c.cache["images"] = None
        out = {}
        for part in PARTS:
            tag = f"ol-check/{c.system_name()}-{part}:dep01"
            c.sh(
                [
                    "docker",
                    "build",
                    "-q",
                    "-f",
                    f"deploy/docker/{part}.Dockerfile",
                    "--build-arg",
                    f"VERSION={VERSION}",
                    "--build-arg",
                    f"REVISION={revision(c)}",
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


def inspect(c: Ctx, tag: str) -> dict:
    return json.loads(c.sh(["docker", "image", "inspect", tag]).stdout)[0]


def test_images_build(c: Ctx) -> None:
    # WHY: both images build from the repo root with the pinned bases, the way
    #      CI builds them, with VERSION and REVISION passed as build args.
    # KIND: unit
    # CHAPTER: dep.01 section 4, The artifact and its check
    images(c)


def test_images_carry_the_labels(c: Ctx) -> None:
    # WHY: the labels must survive into the image config with the build args'
    #      values: revision is the commit this check built.
    # KIND: unit
    # CHAPTER: dep.01 section 2.4
    errs = []
    rev = revision(c)
    for part, tag in images(c).items():
        labels = (inspect(c, tag).get("Config") or {}).get("Labels") or {}
        for k in LABELS:
            if not labels.get(k):
                errs.append(f"{part}: label {k} is missing or empty")
        if labels.get("org.opencontainers.image.revision") != rev:
            errs.append(
                f"{part}: revision label is {labels.get('org.opencontainers.image.revision')!r}, want {rev!r} (from --build-arg REVISION)"
            )
        if labels.get("org.opencontainers.image.version") != VERSION:
            errs.append(
                f"{part}: version label is {labels.get('org.opencontainers.image.version')!r}, want {VERSION!r} (from --build-arg VERSION)"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_images_run_as_numeric_non_root(c: Ctx) -> None:
    # WHY: the kubelet enforces runAsNonRoot from the image config's numeric user.
    # KIND: boundary
    for part, tag in images(c).items():
        user = (inspect(c, tag).get("Config") or {}).get("User") or ""
        uid = user.split(":")[0]
        if not uid.isdigit() or uid == "0":
            raise Fail(
                f"the {part} image runs as {user or 'root'}; want a numeric uid other than 0"
            )


def test_no_secrets_in_image_history_or_env(c: Ctx) -> None:
    # WHY: `docker history --no-trunc` shows every build instruction with its
    #      arguments, and the image config keeps every ENV: a key passed as a
    #      build arg or baked into ENV ships to everyone who pulls the image.
    # KIND: unit
    # CHAPTER: dep.01 section 5, Pitfall 4
    errs = []
    for part, tag in images(c).items():
        hist = c.sh(
            ["docker", "history", "--no-trunc", "--format", "{{.CreatedBy}}", tag]
        ).stdout
        errs += [f"{part} history: {h}" for h in secret_hits(hist)]
        for e in (inspect(c, tag).get("Config") or {}).get("Env") or []:
            name, _, value = e.partition("=")
            if (
                re.search(r"(KEY|SECRET|PEPPER|TOKEN|PASSWORD|CREDENTIAL)", name, re.I)
                and value
            ):
                errs.append(f"{part} env: {name} has a value in the image")
    if errs:
        raise Fail("\n".join(errs))


def test_images_fit_the_size_budget(c: Ctx) -> None:
    # WHY: a single-stage build ships the compiler and the build cache (a Rust
    #      toolchain is over 800 MB); the budget is twice the reference image,
    #      measured as the flattened filesystem (`docker export`), which is the
    #      same number on every image store.
    # KIND: boundary
    # CHAPTER: dep.01 section 5, Pitfall 3
    base = Path(os.environ.get("TINYLLM_FIXTURES") or course_tree() / "fixtures")
    budget = json.loads((base / "dep.01" / "image-budget.json").read_text())
    errs = []
    for part, tag in images(c).items():
        cid = c.sh(["docker", "create", tag]).stdout.strip()
        c.cleanups.append(lambda cid=cid: c.sh(["docker", "rm", cid], check=False))
        p = subprocess.Popen(
            ["docker", "export", cid], stdout=subprocess.PIPE, start_new_session=True
        )
        size = 0
        assert p.stdout is not None
        for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
            size += len(chunk)
        p.wait(timeout=120)
        limit = int(budget["factor"] * budget["reference_bytes"][part])
        print(f"       {part}: {size / 2**20:.1f} MiB of {limit / 2**20:.1f} MiB")
        if size > limit:
            errs.append(
                f"the {part} image is {size / 2**20:.1f} MiB; the budget is {limit / 2**20:.1f} MiB "
                f"({budget['factor']}x the reference)"
            )
    if errs:
        raise Fail("\n".join(errs))


def dep00(c: Ctx):
    """dep.00's check module: its model training and streaming helpers."""
    if "dep00" not in c.cache:
        p = course_tree() / "tests" / "dep.00" / "artifacts.py"
        spec = importlib.util.spec_from_file_location("ol_dep00_artifacts", p)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        c.cache["dep00"] = mod
    return c.cache["dep00"]


def health(c: Ctx, name: str) -> str:
    r = c.sh(
        [
            "docker",
            "inspect",
            "-f",
            "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
            name,
        ],
        check=False,
    )
    return r.stdout.strip()


def test_containers_report_healthy_and_stream(c: Ctx) -> None:
    # WHY: the HEALTHCHECK must work in the image as built (a probe that needs
    #      curl on a base without curl reports unhealthy forever), and the two
    #      images, wired the way the charts wire them, still stream tokens.
    # KIND: conformance
    # CHAPTER: dep.01 section 2.3
    imgs = images(c)
    d0 = dep00(c)
    model = d0._model(c)
    tag = f"ol-dep01-{os.getpid()}"
    c.sh(["docker", "network", "create", tag])
    c.cleanups.append(lambda: c.sh(["docker", "network", "rm", tag], check=False))
    eng, gw = f"{tag}-engine", f"{tag}-gateway"
    c.cleanups.append(lambda: c.sh(["docker", "rm", "-f", eng, gw], check=False))
    key = d0.PROBE_KEY
    # --health-interval shortens the image's own interval for the test only.
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
            "--health-interval",
            "1s",
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
            f"TL_API_KEY={key}",
            "--health-interval",
            "1s",
            "-p",
            "127.0.0.1::8080",
            imgs["gateway"],
            "--port",
            "8080",
            "--health-port",
            "9464",
            "--upstream",
            "http://engine:8000",
        ]
    )
    deadline = time.monotonic() + 60
    states = {}
    while time.monotonic() < deadline:
        states = {n: health(c, n) for n in (eng, gw)}
        if all(s == "healthy" for s in states.values()) or any(
            s == "unhealthy" for s in states.values()
        ):
            break
        time.sleep(1)
    if not all(s == "healthy" for s in states.values()):
        logs = "\n".join(
            c.sh(
                ["docker", "inspect", "-f", "{{json .State.Health}}", n], check=False
            ).stdout[-400:]
            for n in (eng, gw)
        )
        raise Fail(
            f"health after start: engine {states.get(eng)}, gateway {states.get(gw)}\n{logs}"
        )
    port = int(
        c.sh(["docker", "port", gw, "8080/tcp"])
        .stdout.splitlines()[0]
        .rsplit(":", 1)[1]
    )
    code, data = d0._stream(
        c, f"http://127.0.0.1:{port}/v1/completions", key, max_tokens=8
    )
    if code != 200 or not data or data[-1] != "[DONE]":
        raise Fail(
            f"streaming through the gateway container: HTTP {code}, events {data[-2:]}"
        )


if __name__ == "__main__":
    raise SystemExit(run(globals()))
