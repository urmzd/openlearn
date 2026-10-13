"""The learner's local CI entry must propagate the course grader's verdict."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("verdict", [0, 1, 5])
@pytest.mark.parametrize("custom", [False, True])
def test_local_ci_runs_course_grader(tmp_path, verdict, custom):
    script = Path(__file__).resolve().parents[2] / "ref/entry/scripts/ci-local.sh"
    tools = tmp_path / "bin"
    tools.mkdir()
    for name, body in {
        "git": "echo 'feat: learner checkpoint'",
        "make": "exit 0",
        "uv": "exit 0",
    }.items():
        p = tools / name
        p.write_text(f"#!/bin/sh\n{body}\n")
        p.chmod(0o755)
    for rel in (".githooks/commit-msg", "primers/lang.02/greet"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/bin/sh\nexit 0\n")
        p.chmod(0o755)
    (tmp_path / "primers/lang.01").mkdir(parents=True)
    grader = tmp_path / (
        "custom grader/ol" if custom else ".ol/openlearn/practice/bin/ol"
    )
    grader.parent.mkdir(parents=True)
    grader.write_text(
        '#!/bin/sh\n[ "$OL_COURSE_HOME" = "$PWD" ] || exit 99\n'
        '[ "$*" = "check --all --ci" ] || exit 98\n'
        f"echo grader-ran\nexit {verdict}\n"
    )
    grader.chmod(0o755)
    env = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}")
    env.pop("OL_BIN", None)
    if custom:
        env["OL_BIN"] = str(grader)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        start_new_session=True,
    )
    assert result.returncode == verdict, result.stdout + result.stderr
    assert "grader-ran" in result.stdout
