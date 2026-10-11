"""Harness self-tests: drive practice/bin/ol against a throwaway course tree.

`fixtures/site/` is a miniature openlearn: a course/ tree with one sample
module per language (Python M90.*, C rt.90/rt.91, Rust ds.90/ds.91, Go
dur.90/dur.91, practice craft.90), its chapters, and a course path. Each test
copies it into a temp dir, adds the real frozen helpers (ol_test.h, tests/_lib),
commits it to a fresh git repo (so contracts/VERSION resolution is real), and
points ol at it and at a fresh learner repo through OL_* env vars.

Run:  uv run --project course/harness pytest practice/bin/tests
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
OL_BIN = REPO / "practice" / "bin" / "ol"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "site"
FAKES = Path(__file__).resolve().parent / "fakes"
EXTRAS = FIXTURE.parent / "extras"
MARKER = re.compile(r"^[^A-Za-z0-9]*SOLUTION-(BEGIN|END)\b.*$")
collect_ignore = [
    "fixtures",
    "fakes",
]  # the fixture course's own tests run only through ss


def git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout


class SS:
    def __init__(self, tmp: Path, site: Path):
        self.tmp = tmp
        self.site = site
        self.course = site / "course"
        self.learner = tmp / "learner"
        self.env = {
            **os.environ,
            "OL_COURSE_ROOT": str(self.course),
            "OL_COURSE_HOME": str(self.learner),
            "OL_CACHE": str(tmp / "cache"),
            "OL_SCRATCH": str(tmp / "scratch"),
            "OL_PATHS_DIR": str(site / "paths"),
            "OL_GO_RACE": "0",
            "EDITOR": "true",
            "CARGO_TERM_COLOR": "never",
        }
        self.env.pop("VIRTUAL_ENV", None)

    def __call__(
        self,
        *args: str,
        rc: int | None = None,
        env: dict | None = None,
        input: str | None = None,
        timeout: float = 120,
    ) -> subprocess.CompletedProcess:
        p = subprocess.run(
            ["bash", str(OL_BIN), *args],
            env={**self.env, **(env or {})},
            cwd=REPO,
            input=input,
            stdin=None if input is not None else subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        p.out = p.stdout + p.stderr  # type: ignore[attr-defined]
        if rc is not None:
            assert p.returncode == rc, (
                "ol {' '.join(args)} exited {p.returncode}, want {rc}\n{p.out}"
            )
        return p

    def init(self) -> None:
        self("course", "init", "--name", "forge", rc=0)

    def ref(self, unit: str, owner: str | None = None) -> Path:
        if owner:
            return self.course / "ref" / "history" / owner / unit
        return self.course / "ref" / unit

    def implement(self, unit: str, owner: str | None = None) -> None:
        """Write the reference (markers dropped) as if the learner solved it."""
        text = "".join(
            x
            for x in self.ref(unit, owner).read_text().splitlines(keepends=True)
            if not MARKER.match(x.rstrip("\n"))
        )
        dest = self.learner / unit
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)

    def verdicts(self, mid: str) -> list[dict]:
        p = self.learner / ".ol" / "verdicts.jsonl"
        return [
            e
            for e in map(json.loads, p.read_text().splitlines())
            if e.get("id") == mid and "result" in e
        ]

    def commit_site(self, msg: str) -> str:
        git(self.site, "add", "-A")
        git(self.site, "commit", "-qm", msg)
        return git(self.site, "rev-parse", "HEAD").strip()

    def install_system(self, deploy: dict | None = None) -> None:
        """Copy the fixture reference system (course/ref/entry + ref/system.toml)
        into the learner repo, as a learner who wrote their entry points would."""
        shutil.copytree(self.course / "ref" / "entry", self.learner, dirs_exist_ok=True)
        text = (self.course / "ref" / "system.toml").read_text()
        for k, v in (deploy or {}).items():
            text = re.sub(rf"^{k}\s*=.*$", f"{k} = {json.dumps(v)}", text, flags=re.M)
        (self.learner / "system.toml").write_text(text)
        self.commit_learner("feat: add the tracer entry points")

    def add_extras(self, *names: str) -> None:
        """Overlay fixtures/extras/<name>/ onto the site (extra modules a test
        needs), regenerate modules.tsv, and commit, before `init`."""
        for n in names:
            shutil.copytree(EXTRAS / n, self.site, dirs_exist_ok=True)
        self(
            "lint", "--fix-index"
        )  # rewrites modules.tsv even when a chapter lints dirty
        self.commit_site("extras: " + ", ".join(names))

    def commit_learner(self, msg: str) -> None:
        git(self.learner, "add", "-A")
        git(self.learner, "commit", "-qm", msg, "--allow-empty")

    def use_fakes(self, kube_state: dict | None = None) -> dict:
        """Put fake kubectl, kind, and gh first on PATH for every later ol call."""
        self.env["PATH"] = f"{FAKES}{os.pathsep}{self.env['PATH']}"
        self.env["FAKE_KUBE_STATE"] = str(self.tmp / "kube.json")
        self.env["FAKE_KUBE_LOG"] = str(self.tmp / "kube.log")
        self.env["FAKE_GH_LOG"] = str(self.tmp / "gh.log")
        st = kube_state or default_kube_state()
        (self.tmp / "kube.json").write_text(json.dumps(st, indent=1))
        (self.tmp / "kube.log").write_text("")
        return st

    def kube(self) -> dict:
        return json.loads((self.tmp / "kube.json").read_text())

    def kube_log(self) -> list[list[str]]:
        return [
            json.loads(x)
            for x in (self.tmp / "kube.log").read_text().splitlines()
            if x.strip()
        ]


def _deployment(name: str, app: str, replicas: int, env: list | None = None) -> dict:
    return {
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": app}},
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": app,
                            "env": env or [],
                            "args": ["--config", "/etc/tl/runtime.toml"],
                        }
                    ]
                }
            },
        },
    }


def _pod(name: str, app: str, ready: bool = True) -> dict:
    return {
        "metadata": {"name": name, "labels": {"app": app}},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


def default_kube_state() -> dict:
    return {
        "current_context": "kind-forge",
        "contexts": ["kind-forge", "minikube"],
        "namespaces": {
            "forge": {
                "deployments": {
                    "forge-gateway": _deployment(
                        "forge-gateway",
                        "gateway",
                        1,
                        [{"name": "TL_LOG", "value": "info"}],
                    ),
                    "forge-engine-decode": _deployment(
                        "forge-engine-decode", "decode", 2
                    ),
                    "forge-engine-prefill": _deployment(
                        "forge-engine-prefill", "prefill", 1
                    ),
                },
                "statefulsets": {},
                "configmaps": {"forge-runtime": {"data": {"route_policy": "affinity"}}},
                "pods": [
                    _pod("decode-a", "decode"),
                    _pod("decode-b", "decode"),
                    _pod("decode-c", "decode", ready=False),
                    _pod("prefill-a", "prefill"),
                    _pod("gateway-a", "gateway"),
                ],
            },
            "default": {},
        },
    }


class FakeHTTP:
    """An in-process HTTP server for Prometheus and Jaeger stand-ins. `routes`
    maps a path to a function(query dict) -> (status, JSON-able body)."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                try:
                    q = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    q = {"raw": raw.decode(errors="replace")}
                self._answer(urllib.parse.urlsplit(self.path).path, q)

            def do_GET(self):
                u = urllib.parse.urlsplit(self.path)
                q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
                self._answer(u.path, q)

            def _answer(self, path, q):
                outer.calls.append((path, q))
                fn = outer.routes.get(path)
                status, body = fn(q) if fn else (404, {"error": "no route"})
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_http():
    servers = []

    def make(routes: dict) -> FakeHTTP:
        s = FakeHTTP(routes)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


@pytest.fixture
def ss(tmp_path: Path) -> SS:
    site = tmp_path / "site"
    shutil.copytree(FIXTURE, site)
    for h in sorted((REPO / "course" / "contracts" / "c" / "include").glob("ol_*.h")):
        shutil.copy2(h, site / "course" / "contracts" / "c" / "include")
    shutil.copytree(
        REPO / "course" / "tests" / "_lib",
        site / "course" / "tests" / "_lib",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(REPO / "course" / "tests" / "conftest.py", site / "course" / "tests")
    shutil.copytree(REPO / "course" / "conformance", site / "course" / "conformance")
    shutil.copytree(
        REPO / "course" / "testkit",
        site / "course" / "testkit",
        ignore=shutil.ignore_patterns("__pycache__", "target"),
    )
    shutil.copytree(REPO / "course" / "rubrics", site / "course" / "rubrics")
    git(site, "init", "-q")
    s = SS(tmp_path, site)
    s.commit_site("fixture")
    return s
