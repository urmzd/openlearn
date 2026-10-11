"""ol doctor: required tools by pass, the Docker allocation from Pass 7, the
default pass from the learner's started modules, --json."""

import json
import os
import shutil


def _toolbox(tmp, extra: dict[str, str]):
    """A PATH holding only the real pass-0 tools plus scripted fakes."""
    d = tmp / "tools"
    d.mkdir(exist_ok=True)
    for t in ("uv", "cargo", "go", "git", "cc", "rustup"):
        p = shutil.which(t)
        if p and not (d / t).exists():
            (d / t).symlink_to(p)
    for name, body in extra.items():
        (d / name).write_text("#!/bin/sh\n" + body + "\n")
        (d / name).chmod(0o755)
    return f"{d}{os.pathsep}/usr/bin{os.pathsep}/bin"


def test_pass_zero_needs_only_the_compilers(ss, tmp_path):
    path = _toolbox(tmp_path, {})
    out = ss("doctor", "--pass", "0", rc=0, env={"PATH": path}).out
    assert "ready for pass 0" in out
    assert "docker" in out and "not installed" in out
    out = ss("doctor", "--pass", "1", rc=5, env={"PATH": path}).out
    # /usr/bin is on the PATH, and CI runners ship docker and kubectl there:
    # expect exactly the pass-1 tools this PATH lacks.
    missing = [
        t
        for t in ("docker", "kubectl", "kind", "helm")
        if not shutil.which(t, path=path)
    ]
    assert f"missing for pass 1: {', '.join(missing)}" in out


def test_docker_allocation_from_pass_seven(ss, tmp_path):
    fakes = {
        "docker": 'case "$1" in version) echo 27.3.1;; info) echo "4 8589934592";; esac',
        "kubectl": "echo 'Client Version: v1.31.0'",
        "kind": "echo kind v0.24.0",
        "helm": "echo v3.16.0",
        "tilt": "echo v0.33.0",
    }
    path = _toolbox(tmp_path, fakes)
    ss("doctor", "--pass", "1", rc=0, env={"PATH": path})
    out = ss("doctor", "--pass", "7", rc=5, env={"PATH": path}).out
    assert "4 CPUs, 8.0 GiB" in out and "missing for pass 7: docker resources" in out
    data = json.loads(
        ss("doctor", "--pass", "7", "--json", rc=5, env={"PATH": path}).stdout
    )
    row = next(r for r in data["checks"] if r["tool"] == "docker resources")
    assert data["ok"] is False and row["required"] and not row["ok"]
    # A stopped daemon is a failure with the reason, not a crash.
    path = _toolbox(
        tmp_path,
        {**fakes, "docker": "echo 'Cannot connect to the Docker daemon' >&2; exit 1"},
    )
    assert (
        "the daemon is not running"
        in ss("doctor", "--pass", "1", rc=5, env={"PATH": path}).out
    )


def test_default_pass_follows_started_modules(ss, tmp_path):
    path = _toolbox(tmp_path, {})
    ss.init()
    assert "requirements for pass 0" in ss("doctor", rc=0, env={"PATH": path}).out
    ss("start", "rt.90", rc=0)  # a pass 1 module
    assert "requirements for pass 1" in ss("doctor", rc=5, env={"PATH": path}).out
