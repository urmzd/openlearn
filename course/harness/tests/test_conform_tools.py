"""The tools.* conformance checks (L10.9), run directly against the fixture
v1 engine, since the flow test keeps them pending until L10.9 passes."""

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from olcourse import conform, web

FIX = Path(__file__).resolve().parents[3] / "practice" / "bin" / "tests" / "fixtures"


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def engine(tmp_path):
    serve = tmp_path / "serve"
    shutil.copytree(FIX / "site/course/ref/entry/serve", serve)
    shutil.copy2(
        FIX / "extras/conform-v1/course/ref/entry/serve/engine.py", serve / "engine.py"
    )
    procs = []

    def start(patch=None):
        if patch:
            p = serve / "engine.py"
            p.write_text(patch(p.read_text()))
        port, health = _port(), _port()
        cfg = tmp_path / "rt.toml"
        cfg.write_text(
            f'[engine]\nhttp_listen = "127.0.0.1:{port}"\nhealth_listen = "127.0.0.1:{health}"\n'
        )
        proc = subprocess.Popen(
            [sys.executable, str(serve / "engine.py"), "--config", str(cfg)], cwd=serve
        )
        procs.append(proc)
        ok, why = web.wait_ok(f"http://127.0.0.1:{health}/healthz", 10)
        assert ok, why
        spec = conform.load_spec(
            FIX / "extras/conform-v1/course/contracts/openapi/openai-subset.v1.yaml"
        )
        return conform.Client(
            base=f"http://127.0.0.1:{port}",
            tier="engine",
            spec=spec,
            health_base=f"http://127.0.0.1:{health}",
        )

    yield start
    for p in procs:
        p.terminate()
        p.wait()


def test_tool_checks_pass_on_a_conforming_engine(engine):
    c = engine()
    for name in ("tools_call", "tools_stream", "tools_choice"):
        assert conform.CHECKS[name](c, {}) == [], name


def test_tool_checks_catch_bad_arguments_and_ignored_choice(engine):
    c = engine(
        lambda t: t.replace(
            'args = json.dumps({"city": "Paris"})', "args = '{\"town\": 1}'"
        )
    )
    errs = conform.CHECKS["tools_call"](c, {})
    assert any("additional" in e or "city" in e for e in errs), errs
    time.sleep(0.1)


def test_tool_choice_none_must_not_call(engine):
    c = engine(lambda t: t.replace('if tools and choice != "none":', "if tools:"))
    errs = conform.CHECKS["tools_choice"](c, {})
    assert any("tool_choice none" in e for e in errs), errs
