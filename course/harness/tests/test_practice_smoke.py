"""Dependent artifact checks require an assembled reference learner."""

from types import SimpleNamespace

import pytest
from olcourse.commands import verify


@pytest.mark.parametrize("exit_code", [0, 7])
def test_practice_smoke_runs_in_reference_learner(tmp_path, monkeypatch, exit_code):
    course = tmp_path / "course"
    check = course / "tests" / "dep.07" / "check"
    check.parent.mkdir(parents=True)
    check.write_text(
        "#!/bin/sh\n"
        "[ -f system.toml ] || exit 91\n"
        '[ "$OL_SMOKE" = 1 ] || exit 92\n'
        '[ "$OL_COURSE_TREE" = "' + str(course) + '" ] || exit 93\n'
        f"echo artifact-check-ran\nexit {exit_code}\n"
    )
    check.chmod(0o755)
    lr = tmp_path / "reference"
    lr.mkdir()
    (lr / "system.toml").write_text('[system]\nname="test"\n')
    monkeypatch.setattr(verify, "reference_learner", lambda *args: (lr, ""))
    module = SimpleNamespace(kind="practice", timeout_s=5)
    reg = SimpleNamespace(get=lambda mid: module)
    report = verify.Report()
    verify.verify_smoke(reg, course, "dep.07", tmp_path / "absent-work", report)
    assert report.failures == (exit_code != 0)


def test_practice_smoke_reports_assembly_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        verify, "reference_learner", lambda *args: (None, "build failed")
    )
    reg = SimpleNamespace(get=lambda mid: SimpleNamespace(kind="practice"))
    report = verify.Report()
    verify.verify_smoke(reg, tmp_path, "dep.07", tmp_path, report)
    assert report.failures == 1
    assert "build failed" in capsys.readouterr().out
