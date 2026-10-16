"""`ol verify course` and `ol lint` on the fixture: green as committed, and each
check catches the defect it exists for."""

import re

import pytest


def _line(out: str, n: int) -> str:
    """The first report line for check n."""
    for line in out.splitlines():
        if re.match(rf"^\s+(ok|FAIL|skip)\s+{n} ", line):
            return line
    return ""


def test_fixture_verifies_python_modules(ss):
    out = ss("verify", "course", "M90.2", "M90.3", rc=0).out
    for n in (1, 2, 3, 4, 5, 8, 9, 10, 13, 14):
        assert _line(out, n).lstrip().startswith("ok"), (n, out)


def test_fixture_verifies_c_and_rust_modules(ss):
    out = ss("verify", "course", "rt.91", "ds.90", rc=0).out
    for n in (1, 2, 3, 5, 8, 9, 10, 13, 14):
        assert _line(out, n).lstrip().startswith("ok"), (n, out)


@pytest.mark.parametrize("nested_paths", [False, True])
def test_fixture_verifies_globally(ss, nested_paths):
    if nested_paths:
        nested = ss.site / "content/courses/build-an-llm/paths"
        nested.parent.mkdir(parents=True)
        (ss.site / "paths").rename(nested)
        ss.env["OL_PATHS_DIR"] = str(nested)
    out = ss("verify", "course", "--global", rc=0).out
    assert (
        "fully stubbed reference tree compiles with every course test (c, go, python, rust)"
        in out
    )
    assert "paths: ol learn --verify" in out and "history snapshots match" in out


def test_verify_catches_wrong_marker_id(ss):
    demo = ss.course / "ref/c/src/runtime/demo.c"
    demo.write_text(
        demo.read_text().replace(
            "SOLUTION-BEGIN rt.91 */\n    return b ? b->data",
            "SOLUTION-BEGIN rt.90 */\n    return b ? b->data",
        )
    )
    out = ss("verify", "course", "rt.91", rc=1).out
    assert "marker id rt.90; every marker in this unit must carry rt.91" in out


def test_verify_catches_tests_that_pass_on_the_stub(ss):
    t = ss.course / "tests/M90.1/test_scale.py"
    t.write_text(
        "def test_hand_example():\n    # WHY: nothing\n    # KIND: unit\n    assert True\n\n\n"
        "def test_input_not_modified():\n    # WHY: nothing\n    # KIND: unit\n    assert True\n"
    )
    out = ss("verify", "course", "M90.1", rc=1).out
    assert "tests pass on the stub (python)" in out


def test_verify_catches_registry_annotation_and_helper_defects(ss):
    toml = ss.course / "modules/M90.1.toml"
    toml.write_text(
        toml.read_text().replace(
            'used_by   = ["M90.2", "M90.3"]', 'used_by   = ["M90.3"]'
        )
    )
    t = ss.course / "tests/M90.2/test_norm.py"
    t.write_text(
        t.read_text()
        .replace("    # KIND: boundary\n", "")
        .replace("import pytest\n", "import pytest\nimport numpy as np\n")
        + "\n\ndef test_extra():\n    # WHY: closeness without the frozen helper\n    # KIND: unit\n"
        "    assert np.allclose([1.0], [1.0])\n"
    )
    out = ss("verify", "course", "M90.1", "M90.2", rc=1).out
    assert "M90.1: used_by is missing ['M90.2']" in out
    assert "test_all_zero_raises: missing `KIND:`" in out
    assert "asserts closeness without _lib.close (D35: frozen helpers only)" in out
    assert "course test `test_extra` is not named in beat 4" in out


def test_verify_catches_seam_violation(ss):
    norm = ss.course / "ref/python/tinyllm/demo/norm.py"
    norm.write_text(
        norm.read_text().replace(
            "from tinyllm.demo.scale import scale",
            "from tinyllm.demo.scale import scale, _private",
        )
    )
    out = ss("verify", "course", "M90.2", rc=1).out
    assert (
        "uses tinyllm.demo.scale._private, which contracts/py/tinyllm/demo/scale.pyi does not declare"
        in out
    )


def test_lint_catches_chapter_defects(ss):
    ss("lint", rc=0)
    ch = ss.site / "chapters/m90-2-norm.md"
    text = ch.read_text()
    ch.write_text(
        text.replace("| Back | `M90.1` | scale a vector |\n", "").replace(
            "## 5. Pitfalls", "## 5. Pitfalls \u2014 traps"
        )
    )
    out = ss("lint", rc=1).out
    assert "beat 6 needs a Back row for M90.1" in out
    assert "em dash (U+2014)" in out
    ch.write_text(
        text.replace(
            "[`contracts/py/tinyllm/demo/norm.pyi`](../course/contracts/py/tinyllm/demo/norm.pyi)",
            "[`contracts/py/tinyllm/demo/norm.pyi`](../course/contracts/py/nope.pyi)",
        )
    )
    assert (
        "broken link ../course/contracts/py/nope.pyi" in ss("lint", "--links", rc=1).out
    )


def test_lint_flags_stale_modules_tsv_and_fixes_it(ss):
    toml = ss.course / "modules/craft.90.toml"
    toml.write_text(
        toml.read_text().replace(
            'title     = "Document the demo system"', 'title     = "Document it"'
        )
    )
    ch = ss.site / "chapters/craft-90-docs.md"
    ch.write_text(ch.read_text().replace("# Document the demo system", "# Document it"))
    ss("lint", rc=0)  # titles are not in modules.tsv (the H1 must follow the registry)
    toml.write_text(
        toml.read_text().replace('lang      = ["docs"]', 'lang      = ["none"]')
    )
    assert "modules.tsv is not current" in ss("lint", rc=1).out
    ss("lint", "--fix-index", rc=0)
    ss("lint", rc=0)


def test_verify_changed_maps_files_to_modules(ss):
    abi = ss.course / "ref/c/src/runtime/abi.c"
    abi.write_text(
        abi.read_text().replace(
            "/* c/src/runtime/abi.c (fixture rt.90) */",
            "/* c/src/runtime/abi.c (fixture rt.90, touched) */",
        )
    )
    ss.commit_site("touch rt.90 reference")
    out = ss("verify", "course", "--changed", "HEAD~1", rc=0).out
    assert "changed since HEAD~1: rt.90; dependents (smoke): rt.91" in out
    assert "smoke rt.91" in out
