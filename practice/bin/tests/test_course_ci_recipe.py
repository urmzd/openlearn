"""The learner CI recipe (DESIGN 5.13, craft.01), executed literally: the
learner repo clones openlearn at the sha its contracts/VERSION names into
.ol/openlearn and runs `ol check --all --ci` from there, with none of the
OL_* overrides a maintainer uses."""

import os
import shutil
import subprocess

from conftest import REPO, SS


def _site_with_harness(ss: SS) -> None:
    (ss.site / "practice" / "bin").mkdir(parents=True)
    shutil.copy2(REPO / "practice" / "bin" / "ol", ss.site / "practice" / "bin" / "ol")
    h = ss.site / "course" / "harness"
    h.mkdir()
    for f in ("pyproject.toml", "uv.lock"):
        shutil.copy2(REPO / "course" / "harness" / f, h / f)
    shutil.copytree(
        REPO / "course" / "harness" / "src",
        h / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    ss.commit_site("add the harness")


def _ci(ss: SS, recipe: str) -> subprocess.CompletedProcess:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "UV_OFFLINE": "1",
        "OL_CACHE": str(ss.tmp / "ci-cache"),
        "GOFLAGS": "-count=1",
    }
    # Offline uv must see the same cache and interpreters the job warmed;
    # CI (setup-uv) moves both off their defaults.
    for k in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON"):
        if k in os.environ:
            env[k] = os.environ[k]
    return subprocess.run(
        ["bash", "-c", recipe],
        cwd=ss.learner,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_recipe_runs_check_all_ci_from_a_clone(ss):
    _site_with_harness(ss)
    ss.init()
    recipe = ss("course", "ci", "--upstream", str(ss.site), rc=0).stdout
    assert "check --all --ci" in recipe and str(ss.site) in recipe
    p = _ci(ss, recipe)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "nothing started yet: trivially green" in p.stdout
    head = subprocess.run(
        ["git", "-C", str(ss.learner / ".ol/openlearn"), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert f'sha = "{head}"' in (ss.learner / "contracts/VERSION").read_text()

    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    ss.commit_learner("feat(M90.1): scale")
    p = _ci(ss, recipe)  # the second run reuses the clone
    assert p.returncode == 0, p.stdout + p.stderr
    assert "M90.1        pass" in p.stdout

    unit = ss.learner / "python/tinyllm/demo/scale.py"
    unit.write_text(unit.read_text().replace("k * x for", "k * x + 1 for"))
    p = _ci(ss, recipe)
    assert p.returncode == 1 and "M90.1        fail" in p.stdout
