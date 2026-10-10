"""A small runner for practice-module artifact checks (DESIGN 6.4, H15).

A practice module's `course/tests/<ID>/check` runs in the learner repo, and
its exit code is the verdict. The checks in this course are plain Python
files whose `test_*` functions carry the same WHY/KIND header as every other
course test, so `ss tests <ID>` lists them and `ss verify course` lints them.

    from _lib.practice import Ctx, Fail, run
    def test_dockerfile_exists(c: Ctx) -> None:
        # WHY: ...
        # KIND: unit
        c.require_file("Dockerfile")
    raise SystemExit(run(globals()))

Tests run in file order and share one `Ctx`, which caches expensive work
(a rendered chart, a built image). A test raises `Fail` (or AssertionError)
to fail and `Skip` to skip with a reason.

Tiers. Static tests always run. Tests that need Docker call `c.need_docker()`,
which fails when the daemon is unreachable (Docker is required from Pass 1).
Tests that need a cluster call `c.need_cluster(context)`: when the kube
context is missing or unreachable they fail, unless `SS_SMOKE=1` is set, in
which case they are skipped with the reason and the run is reported as
`smoke` (everything except the cluster was checked).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path


class Fail(AssertionError):
    """The artifact does not meet the check."""


class Skip(Exception):
    """The test cannot run here; the reason is printed."""


def smoke_mode() -> bool:
    return os.environ.get("SS_SMOKE", "") not in ("", "0")


class Ctx:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path.cwd()).resolve()
        self.cache: dict = {}
        self.skipped_cluster = False
        self.cleanups: list = []

    # -- files ---------------------------------------------------------------

    def path(self, rel: str) -> Path:
        return self.root / rel

    def require_file(self, rel: str) -> Path:
        p = self.path(rel)
        if not p.is_file():
            raise Fail(f"{rel} is missing")
        return p

    def system(self) -> dict:
        """The learner's system.toml, parsed (cached)."""
        if "system" not in self.cache:
            import tomllib

            p = self.require_file("system.toml")
            try:
                self.cache["system"] = tomllib.loads(p.read_text())
            except tomllib.TOMLDecodeError as e:
                raise Fail(f"system.toml: {e}") from None
        return self.cache["system"]

    def system_name(self) -> str:
        name = self.system().get("system", {}).get("name")
        if not name:
            raise Fail("system.toml has no [system].name")
        return str(name)

    # -- processes -----------------------------------------------------------

    def sh(
        self,
        argv: list[str],
        timeout: float = 120,
        check: bool = True,
        cwd: Path | None = None,
        env: dict | None = None,
        input: str | None = None,
    ) -> subprocess.CompletedProcess:
        tool = argv[0]
        if shutil.which(tool) is None:
            raise Fail(f"`{tool}` is not on PATH (see `ss doctor --pass 1`)")
        try:
            r = subprocess.run(
                argv,
                cwd=cwd or self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                input=input,
            )
        except subprocess.TimeoutExpired:
            raise Fail(
                f"`{' '.join(argv[:4])} ...` timed out after {timeout:.0f}s"
            ) from None
        if check and r.returncode != 0:
            raise Fail(
                f"`{' '.join(argv)}` exited {r.returncode}:\n{tail(r.stdout + r.stderr, 25)}"
            )
        return r

    def need_docker(self) -> None:
        if "docker_ok" not in self.cache:
            if shutil.which("docker") is None:
                self.cache["docker_ok"] = (
                    "docker is not installed (required from Pass 1, see `ss doctor --pass 1`)"
                )
            else:
                r = subprocess.run(
                    ["docker", "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                )
                self.cache["docker_ok"] = (
                    ""
                    if r.returncode == 0
                    else "the Docker daemon is not reachable: start Docker Desktop (or colima) and rerun"
                )
        if self.cache["docker_ok"]:
            raise Fail(self.cache["docker_ok"])

    def need_cluster(self, context: str) -> None:
        """Skip under SS_SMOKE=1 (`ss check --ci`: the cluster tier belongs to
        the milestones' kind steps, and a cluster that happens to exist may
        hold another pass's deployment); otherwise fail unless `context`
        names a reachable cluster."""
        if smoke_mode():
            self.skipped_cluster = True
            raise Skip(f"cluster tier skipped under SS_SMOKE=1 (context {context})")
        key = f"cluster:{context}"
        if key not in self.cache:
            why = ""
            if shutil.which("kubectl") is None:
                why = "kubectl is not installed"
            else:
                r = subprocess.run(
                    ["kubectl", "config", "get-contexts", "-o", "name"],
                    capture_output=True,
                    text=True,
                )
                if context not in r.stdout.split():
                    hint = (
                        " (kind is not installed: `brew install kind`)"
                        if shutil.which("kind") is None
                        else ""
                    )
                    why = f"kube context {context} does not exist{hint}"
                else:
                    r = subprocess.run(
                        ["kubectl", "--context", context, "get", "--raw", "/readyz"],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    if r.returncode != 0:
                        why = f"cluster {context} is not reachable: {tail(r.stderr, 3)}"
            self.cache[key] = why
        why = self.cache[key]
        if why:
            if smoke_mode():
                self.skipped_cluster = True
                raise Skip(f"cluster tier skipped under SS_SMOKE=1: {why}")
            raise Fail(
                f"{why}. Create the cluster and deploy (chapter section 4), "
                "or rerun with SS_SMOKE=1 to check everything except the cluster"
            )

    def kubectl_json(self, context: str, args: list[str]) -> dict:
        r = self.sh(["kubectl", "--context", context, *args, "-o", "json"], timeout=30)
        return json.loads(r.stdout)

    # -- YAML and Helm ---------------------------------------------------------

    def yaml_docs(self, text: str, where: str) -> list:
        try:
            import yaml
        except ImportError:
            raise Fail(
                "PyYAML is not importable: run this check through `ss check`"
            ) from None
        try:
            return [d for d in yaml.safe_load_all(text) if d is not None]
        except yaml.YAMLError as e:
            raise Fail(f"{where} is not valid YAML: {e}") from None

    def yaml_file(self, rel: str) -> list:
        return self.yaml_docs(self.require_file(rel).read_text(), rel)

    def helm_template(
        self,
        chart: str,
        release: str,
        sets: dict | None = None,
        namespace: str = "default",
    ) -> list:
        key = ("helm", chart, release, namespace, tuple(sorted((sets or {}).items())))
        if key not in self.cache:
            if not self.path(chart).joinpath("Chart.yaml").is_file():
                raise Fail(f"{chart}/Chart.yaml is missing")
            argv = ["helm", "template", release, chart, "--namespace", namespace]
            for k, v in (sets or {}).items():
                argv += ["--set-string", f"{k}={v}"]
            r = self.sh(argv, timeout=60)
            self.cache[key] = self.yaml_docs(
                r.stdout, f"`helm template {release} {chart}` output"
            )
        return self.cache[key]

    # -- HTTP ------------------------------------------------------------------

    def http(
        self,
        method: str,
        url: str,
        body: dict | None = None,
        headers: dict | None = None,
        timeout: float = 10,
    ) -> tuple[int, dict, bytes]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method, headers=dict(headers or {})
        )
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()
        except (urllib.error.URLError, OSError) as e:
            raise Fail(f"{method} {url}: {getattr(e, 'reason', e)}") from None

    def wait_http(self, url: str, timeout: float = 15) -> None:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return
                    last = f"HTTP {r.status}"
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, OSError) as e:
                last = str(getattr(e, "reason", e))
            time.sleep(0.2)
        raise Fail(f"{url} did not answer 200 within {timeout:.0f}s (last: {last})")


# ---------------------------------------------------------------------------
# Kubernetes object helpers


def objects(docs: list, kind: str) -> list[dict]:
    return [d for d in docs if isinstance(d, dict) and d.get("kind") == kind]


def one(docs: list, kind: str, where: str) -> dict:
    found = objects(docs, kind)
    if len(found) != 1:
        raise Fail(f"{where} must render exactly one {kind} (found {len(found)})")
    return found[0]


def pod_spec(obj: dict) -> dict:
    return ((obj.get("spec") or {}).get("template") or {}).get("spec") or {}


def containers(obj: dict) -> list[dict]:
    return list(pod_spec(obj).get("containers") or [])


def image_tag_problem(image: str) -> str:
    """'' when the image reference is pinned (a tag other than latest, or a digest)."""
    if "@sha256:" in image:
        return ""
    last = image.rsplit("/", 1)[-1]
    if ":" not in last:
        return f"{image} has no tag, which means :latest"
    if last.rsplit(":", 1)[1] == "latest":
        return f"{image} uses :latest"
    return ""


def dockerfile_stages(text: str) -> list[dict]:
    """Parse a Dockerfile into stages: {'from': image, 'as': name, 'user': last USER, 'expose': [...]}.
    Handles line continuations and comments; enough for the checks here."""
    lines: list[str] = []
    buf = ""
    for raw in text.splitlines():
        s = raw.strip()
        if not buf and (not s or s.startswith("#")):
            continue
        if s.endswith("\\"):
            buf += s[:-1] + " "
            continue
        lines.append(buf + s)
        buf = ""
    if buf:
        lines.append(buf)
    stages: list[dict] = []
    args: dict[str, str] = {}
    for line in lines:
        parts = line.split(None, 1)
        if not parts:
            continue
        ins, rest = parts[0].upper(), (parts[1] if len(parts) > 1 else "")
        if ins == "ARG" and not stages:
            k, _, v = rest.partition("=")
            args[k.strip()] = v.strip().strip('"')
        elif ins == "FROM":
            toks = [t for t in rest.split() if not t.startswith("--")]
            image = toks[0] if toks else ""
            for k, v in args.items():
                image = image.replace("${" + k + "}", v).replace("$" + k, v)
            name = toks[2] if len(toks) >= 3 and toks[1].lower() == "as" else ""
            stages.append(
                {"from": image, "as": name, "user": "", "expose": [], "lines": []}
            )
        elif stages:
            stages[-1]["lines"].append(line)
            if ins == "USER":
                stages[-1]["user"] = rest.strip()
            elif ins == "EXPOSE":
                stages[-1]["expose"] += [p.split("/")[0] for p in rest.split()]
    return stages


def check_dockerfile(text: str, where: str, port: str | None = None) -> list[str]:
    """The image rules of DESIGN 2.13 and dep.01 that a tracer image must already meet."""
    errs = []
    stages = dockerfile_stages(text)
    if not stages:
        return [f"{where}: no FROM instruction"]
    if len(stages) < 2:
        errs.append(
            f"{where}: one stage only; build in one stage and copy the binary into a small runtime stage"
        )
    names = {s["as"] for s in stages if s["as"]}
    for s in stages:
        if s["from"] in names or s["from"] == "scratch":
            continue
        why = image_tag_problem(s["from"])
        if why:
            errs.append(
                f"{where}: FROM {why}; pin a version tag (dep.01 later pins digests)"
            )
    final = stages[-1]
    user = final["user"]
    uid = user.split(":")[0]
    if not user:
        errs.append(f"{where}: the final stage has no USER, so it runs as root")
    elif uid in ("0", "root"):
        errs.append(f"{where}: the final stage runs as {user}")
    elif not uid.isdigit():
        errs.append(
            f"{where}: USER {user} is a name; use a numeric uid so Kubernetes runAsNonRoot can verify it"
        )
    if port and port not in final["expose"]:
        errs.append(f"{where}: the final stage does not EXPOSE {port}")
    for line in final["lines"]:
        ins = line.split(None, 1)[0].upper()
        if ins in ("CMD", "ENTRYPOINT") and not line.split(None, 1)[
            1
        ].lstrip().startswith("["):
            errs.append(
                f'{where}: `{line[:60]}` is shell form; use exec form ["..."] so your server is PID 1 and gets SIGTERM'
            )
    return errs


# ---------------------------------------------------------------------------
# the runner


def tail(s: str, n: int) -> str:
    lines = (s or "").rstrip().splitlines()
    return "\n".join(lines[-n:])


def run(namespace: dict, root: Path | None = None) -> int:
    """Run every test_* function of the calling module in definition order."""
    c = Ctx(root)
    tests = [
        (n, f) for n, f in namespace.items() if n.startswith("test_") and callable(f)
    ]
    title = (
        (namespace.get("__doc__") or "").strip().splitlines()[0]
        if namespace.get("__doc__")
        else ""
    )
    if title:
        print(title)
    failed = skipped = 0
    t_all = time.monotonic()
    try:
        for name, fn in tests:
            t0 = time.monotonic()
            try:
                fn(c)
                status, why = "PASS", ""
            except Skip as e:
                status, why = "SKIP", str(e)
                skipped += 1
            except AssertionError as e:
                status, why = "FAIL", str(e) or "assertion failed"
                failed += 1
            except (
                Exception
            ):  # a broken check is a failure with its traceback, never a pass
                status, why = (
                    "FAIL",
                    "error in the check itself:\n" + tail(traceback.format_exc(), 8),
                )
                failed += 1
            dt = time.monotonic() - t0
            print(f"  {status} {name}  ({dt:.1f}s)")
            if why:
                print("\n".join("       " + line for line in why.splitlines()))
            sys.stdout.flush()
    finally:
        for fn in reversed(c.cleanups):
            try:
                fn()
            except Exception:  # noqa: BLE001  cleanup is best effort
                pass
    passed = len(tests) - failed - skipped
    mode = " (smoke: cluster tier skipped)" if c.skipped_cluster and not failed else ""
    print(
        f"{passed} passed, {failed} failed, {skipped} skipped in {time.monotonic() - t_all:.1f}s{mode}"
    )
    return 1 if failed else 0
