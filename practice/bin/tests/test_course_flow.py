"""Dispatch, init, drift, reset/show/diff, catalog, --all, next, the path.tsv
check column, and contracts/VERSION tree resolution."""

import json


def test_dispatch_keeps_practice_kinds(ss):
    # Practice kinds never match COURSE_ID_RE, so they stay in bash.
    out = ss("list", "build", "c", rc=0).out
    assert "build/c" in out
    assert "unknown subcommand" not in ss("check", "build", "c", "02").out
    # Course ids and course-only verbs reach the harness.
    assert "no learner repo" in ss("status", rc=5).out
    assert "no learner repo" in ss("mutate", "M90.1", rc=5).out  # routed to the harness
    assert "no learner repo" in ss("milestone", "MS-P1", rc=5).out


def test_init_and_not_started(ss):
    ss.init()
    assert (ss.learner / "system.toml").is_file() and (ss.learner / ".git").is_dir()
    v = (ss.learner / "contracts/VERSION").read_text()
    assert 'semver = "0.1.0"' in v and 'content_hash = "sha256:' in v
    assert (
        "already holds a course repo"
        in ss("course", "init", "--name", "forge", rc=5).out
    )
    assert "not started" in ss("check", "M90.1", rc=2).out
    assert "no module 'M99.9'" in ss("check", "M99.9", rc=5).out


def test_contract_drift_then_sync(ss):
    ss.init()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    pyi = ss.learner / "contracts/py/tinyllm/demo/scale.pyi"
    pyi.write_text(pyi.read_text() + "def extra() -> None: ...\n")
    assert "contracts/ modified" in ss("check", "M90.1", rc=4).out
    ss("contracts", "sync", rc=0)
    ss("check", "M90.1", rc=0)
    # The unit must match its .pyi (an AST stand-in for stubtest).
    unit = ss.learner / "python/tinyllm/demo/scale.py"
    unit.write_text(
        unit.read_text().replace(
            "def scale(xs: list[float], k: float)", "def scale(xs, factor)"
        )
    )
    assert "differ from the contract" in ss("check", "M90.1", rc=4).out


def test_reset_show_diff_tests(ss):
    ss.init()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    assert "refusing" in ss("reset", "M90.1", rc=1).out
    assert "would spoil it" in ss("diff", "M90.1", rc=1).out
    unit = ss.learner / "python/tinyllm/demo/scale.py"
    unit.write_text(unit.read_text() + "# mine\n")
    ss("check", "M90.1", rc=0)
    assert (
        "+++ reference: python/tinyllm/demo/scale.py" in ss("diff", "M90.1", rc=0).out
    )
    cat = ss("tests", "M90.1", rc=0).out
    assert (
        "test_hand_example" in cat
        and "unit" in cat
        and "[smoke]" in cat
        and "why:" in cat
    )
    ss("reset", "M90.1", "--force", rc=0)
    assert (
        "NotImplementedError"
        in (ss.learner / "python/tinyllm/demo/scale.py").read_text()
    )
    assert "return scale(xs, 1.0 / total)" in ss("show", "M90.2", rc=0).out
    assert "spoiled" in (ss.learner / ".ol/verdicts.jsonl").read_text()


def test_check_all_ci(ss):
    ss.init()
    out = ss("check", "--all", "--ci", rc=0).out
    assert "nothing started yet" in out
    ss("start", "rt.90", rc=0)
    ss.implement("c/src/runtime/abi.c")
    ss("start", "M90.1", rc=0)
    out = ss("check", "--all", "--ci", rc=1).out
    assert "rt.90        pass" in out and "M90.1        fail" in out
    assert "forbids --ref-deps" in ss("check", "--all", "--ci", "--ref-deps", rc=5).out
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    ss("check", "--all", "--ci", rc=0)
    data = json.loads(
        ss("check", "M90.1", "--json", rc=0).stdout.strip().splitlines()[-1]
    )
    assert data["exit"] == 0 and data["verdict"]["result"] == "pass"


def test_check_all_reuses_passes_on_unchanged_files(ss):
    # --all reports a pass earned on exactly the current files (every owned
    # unit in the repo) from its verdict; any edit makes every module run
    # again, since a check builds more than the module's own files; --fresh
    # runs everything.
    ss.init()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    ss("start", "M90.2", rc=0)
    ss.implement("python/tinyllm/demo/norm.py")
    first = ss("check", "--all", "--ci", rc=0).out
    assert "unchanged since" not in first
    again = ss("check", "--all", "--ci", rc=0).out
    assert "M90.1        pass       unchanged since" in again
    assert "M90.2        pass       unchanged since" in again
    unit = ss.learner / "python/tinyllm/demo/norm.py"
    unit.write_text(unit.read_text() + "\n# touched\n")
    assert "unchanged since" not in ss("check", "--all", "--ci", rc=0).out
    assert "unchanged since" in ss("check", "--all", "--ci", rc=0).out
    # A non-CI pass never stands in for --ci, and --fresh reruns every module.
    assert "unchanged since" not in ss("check", "--all", "--ci", "--fresh", rc=0).out
    assert "unchanged since" in ss("check", "--all", rc=0).out


def test_next_and_learn_check_column(ss):
    ss.init()
    assert "next: stage 0" in ss("next", rc=0).out
    ss("learn", "course", "--done", "0", rc=0)
    assert "ol start M90.1" in ss("next", rc=0).out
    out = ss("learn", "course", "--done", "1", rc=1).out
    assert "not marked" in out and "needs module:M90.1" in out
    ss("learn", "course", "--done", "1", "--force", rc=0)
    assert "[~] 1" in ss("learn", "course", rc=0).out
    ss("learn", "course", "--undo", "1", rc=0)
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    # No verdict yet: marking runs the check, and marks on pass.
    ss("learn", "course", "--done", "1", rc=0)
    out = ss("learn", "course", rc=0).out
    assert "[x] 1" in out and "check:     module:M90.1" in out
    assert "next: stage 2" in ss("next", rc=0).out
    ss("learn", "--verify", rc=0)


def test_learn_verify_rejects_unknown_check(ss):
    tsv = ss.site / "paths/course/path.tsv"
    tsv.write_text(
        tsv.read_text() + "6\tBad\tchapters/m90-1-scale.md\tnever\tmodule:M99.9\n"
    )
    assert "no module M99.9" in ss("learn", "--verify", rc=1).out


def test_version_pins_the_course_tree(ss):
    ss.init()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    # Supersource moves on: a new commit changes M90.1's course test.
    test = ss.course / "tests/M90.1/test_scale.py"
    test.write_text(test.read_text().replace("[2.5, -5.0, 1.25]", "[2.5, -5.0, 1.0]"))
    ss.commit_site("harder test")
    # The learner's contracts/VERSION still names the old commit: its tree runs.
    ss("check", "M90.1", rc=0)
    assert any((ss.tmp / "cache" / "worktrees").iterdir())
    # After a sync the learner is on the new tree, and the new test applies.
    ss("contracts", "sync", rc=0)
    ss("check", "M90.1", rc=1)
