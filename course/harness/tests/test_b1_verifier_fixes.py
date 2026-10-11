"""Unit tests for the B1 verifier fixes (course/DEVIATIONS.md I28 to I33):
practice `artifacts` in the tree hash and the started rule, the verdict that
matches the current files, smoke-mode practice passes, and the stub's
include placement."""

from pathlib import Path

import pytest
from olcourse import HarnessError, learner, ledger, markers, registry

PRACTICE = (
    'kind = "practice"\nlang = ["c"]\npass = 0\nartifacts = ["primers/lang.90"]\n'
)
BUILD = 'kind = "build"\nlang = ["python"]\npass = 0\nunits = ["python/a.py"]\nused_by = ["craft.90"]\n'


def _course(tmp_path: Path) -> registry.Registry:
    mods = tmp_path / "course" / "modules"
    mods.mkdir(parents=True)
    (mods / "craft.90.toml").write_text('id = "craft.90"\ntitle = "t"\n' + PRACTICE)
    return registry.load(tmp_path / "course")


def _verdict(lr: Path, reg, mid: str, result: str, **extra) -> dict:
    return ledger.append(
        lr,
        {
            "id": mid,
            "result": result,
            "tree": ledger.tree_hash(lr, reg, mid),
            **extra,
        },
    )


def test_artifacts_are_parsed_and_checked(tmp_path):
    reg = _course(tmp_path)
    assert reg.get("craft.90").artifacts == ["primers/lang.90"]
    bad = tmp_path / "course" / "modules" / "craft.91.toml"
    bad.write_text(
        'id = "craft.91"\ntitle = "t"\n'
        + BUILD.replace("craft.90", "craft.92")
        + 'artifacts = ["../x"]\n'
    )
    with pytest.raises(HarnessError, match="artifacts"):
        registry.load(tmp_path / "course")


def test_artifact_presence_starts_a_practice_module(tmp_path):
    reg = _course(tmp_path)
    lr = tmp_path / "learner"
    lr.mkdir()
    assert not learner.started(lr, reg.course, reg, "craft.90")
    (lr / "primers" / "lang.90").mkdir(parents=True)
    (lr / "primers" / "lang.90" / "Makefile").write_text("all:\n")
    assert learner.started(lr, reg.course, reg, "craft.90")


def test_a_practice_pass_goes_stale_when_its_artifacts_change(tmp_path):
    reg = _course(tmp_path)
    lr = tmp_path / "learner"
    d = lr / "primers" / "lang.90"
    d.mkdir(parents=True)
    (d / "main.c").write_text("int main(void) { return 0; }\n")
    _verdict(lr, reg, "craft.90", "pass")
    assert learner.state(lr, reg.course, reg, "craft.90").status == "pass"
    # Build output and caches never make it stale.
    (d / "target").mkdir()
    (d / "target" / "out.o").write_bytes(b"\x00")
    assert learner.state(lr, reg.course, reg, "craft.90").status == "pass"
    (d / "main.c").write_text("int main(void) { return 1; }\n")
    assert learner.state(lr, reg.course, reg, "craft.90").status == "stale"


def test_a_reverted_edit_gets_its_pass_back(tmp_path):
    reg = _course(tmp_path)
    lr = tmp_path / "learner"
    d = lr / "primers" / "lang.90"
    d.mkdir(parents=True)
    good = "int main(void) { return 0; }\n"
    (d / "main.c").write_text(good)
    _verdict(lr, reg, "craft.90", "pass")
    (d / "main.c").write_text("broken\n")
    _verdict(lr, reg, "craft.90", "fail")
    assert ledger.fresh_pass(lr, reg, "craft.90") is None
    (d / "main.c").write_text(good)  # git checkout of the passing file
    assert ledger.fresh_pass(lr, reg, "craft.90") is not None
    assert learner.state(lr, reg.course, reg, "craft.90").status == "pass"


def test_a_smoke_pass_is_not_a_full_pass(tmp_path):
    reg = _course(tmp_path)
    lr = tmp_path / "learner"
    (lr / "primers" / "lang.90").mkdir(parents=True)
    (lr / "primers" / "lang.90" / "x").write_text("x")
    _verdict(lr, reg, "craft.90", "pass", mode="smoke")
    st = learner.state(lr, reg.course, reg, "craft.90")
    assert st.status == "smoke"
    assert ledger.fresh_pass(lr, reg, "craft.90") is not None
    assert ledger.fresh_pass(lr, reg, "craft.90", full_only=True) is None


def test_stub_includes_go_after_the_header_comment():
    src = (
        "/* primers/x.c: the header comment. */\n"
        '#include "x.h"\n\n'
        "int f(void) {\n    /* SOLUTION-BEGIN x */\n    return 1;\n    /* SOLUTION-END */\n}\n"
        "void *g(void) {\n    /* SOLUTION-BEGIN x */\n    return 0;\n    /* SOLUTION-END */\n}\n"
    )
    out = markers.stub(src, "x.c", want="x")
    first = out.splitlines()[0]
    assert first == "/* primers/x.c: the header comment. */", out
