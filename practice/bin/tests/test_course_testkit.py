"""The testkit reaches course tests in three languages through the overlay:
Python (sstestkit on PYTHONPATH), Go (the testkit module in go.work), and Rust
(tl-testkit a dependency of ol-tests)."""


def test_course_tests_import_the_testkit(ss):
    ss.add_extras("testkit")
    out = ss("verify", "course", "M90.1", "dur.90", "ds.90", timeout=600).out
    lines = [
        x
        for x in out.splitlines()
        if x.strip().startswith(("ok", "FAIL")) and " 1 " in x
    ]
    assert len(lines) == 3 and all(x.strip().startswith("ok") for x in lines), out
