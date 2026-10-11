"""lang.01 course tests: your uv project (primers/lang.01/pyproject.toml).

`check` already runs these tests inside that project, so if numpy imports at
all, the project works. This test says what is missing when it does not.
"""

import os
import tomllib
from pathlib import Path


def test_project_declares_numpy():
    # WHY: a uv project's dependencies are declared in pyproject.toml and
    #      pinned in uv.lock; numpy installed by hand into some other
    #      environment is invisible to `uv run` (and to your CI). The project
    #      must list numpy in [project].dependencies.
    # KIND: unit
    # CHAPTER: lang.01 section 4, step 1
    path = Path(os.environ["OL_PRIMER_DIR"]) / "pyproject.toml"
    doc = tomllib.loads(path.read_text())
    deps = doc.get("project", {}).get("dependencies", [])
    names = [d.split(";")[0].strip().lower() for d in deps]
    assert any(
        n == "numpy"
        or n.startswith(
            ("numpy=", "numpy>", "numpy<", "numpy~", "numpy!", "numpy[", "numpy ")
        )
        for n in names
    ), (
        f"[project].dependencies of {path} does not list numpy (uv add --project primers/lang.01 numpy): {deps}"
    )
