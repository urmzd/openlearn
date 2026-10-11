"""Rust and Go modules: glue stubs keep crates and packages compiling, copy
farms with generated manifests, dependency smoke tests, KIND filters."""

import os
import subprocess


def test_rust_glue_stubs_compile_then_pass(ss, tmp_path):
    ss.init()
    out = ss("start", "ds.90", rc=0).out
    for f in ("src/lib.rs", "src/acc.rs", "src/mean.rs", "Cargo.toml"):
        assert f"rust/crates/tl-demo/{f}" in out
    assert (
        'todo!("ds.91")' in (ss.learner / "rust/crates/tl-demo/src/mean.rs").read_text()
    )
    r = subprocess.run(
        [
            "cargo",
            "build",
            "-q",
            "--manifest-path",
            str(ss.learner / "rust/Cargo.toml"),
        ],
        env={**os.environ, "CARGO_TARGET_DIR": str(tmp_path / "t")},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr  # the learner's own crate compiles with stubs
    assert "not yet implemented: ds.90" in ss("check", "ds.90", rc=1).out
    ss.implement("rust/crates/tl-demo/src/acc.rs")
    ss("check", "ds.90", rc=0)
    assert (
        "todo      ds.91" in ss("status", rc=0).out
    )  # the glue stub does not count as started


def test_rust_dep_smoke_and_kind_filter(ss):
    ss.init()
    ss("start", "ds.90", rc=0)
    ss("start", "ds.91", rc=0)
    ss.implement("rust/crates/tl-demo/src/acc.rs")
    ss.implement("rust/crates/tl-demo/src/mean.rs")
    ss("check", "ds.90", rc=0)
    out = ss("check", "ds.91", rc=0).out
    assert "ok   ds.90 smoke: hand_example" in out
    out = ss("check", "ds.91", "--kind", "boundary", "--no-cumulative", rc=0).out
    assert "smoke" not in out
    farm = ss.learner / ".ol/rust-farm"
    assert (farm / "ol-tests/tests/ds_91.rs").is_file() and "ol-tests" in (
        farm / "Cargo.toml"
    ).read_text()


def test_go_stub_builds_fails_then_passes(ss):
    ss.init()
    ss("start", "dur.90", rc=0)
    text = (ss.learner / "go/ds/demo/sum.go").read_text()
    assert 'import _ "math"' in text and 'panic("todo: dur.90")' in text
    r = subprocess.run(
        ["go", "build", "./..."],
        cwd=ss.learner / "go",
        capture_output=True,
        text=True,
        env={**os.environ, "GOWORK": "off", "GOFLAGS": ""},
    )
    assert r.returncode == 0, (
        r.stderr
    )  # mean.go was stubbed alongside, so the package compiles
    assert "todo: dur.90" in ss("check", "dur.90", rc=1).out
    ss.implement("go/ds/demo/sum.go")
    ss("check", "dur.90", rc=0)
    gomod = (ss.learner / ".ol/overlay/dur.90/go/go.mod").read_text()
    assert "replace" not in gomod  # go.work provides the contracts module


def test_go_dependent_smoke(ss):
    ss.init()
    for mid, unit in (
        ("dur.90", "go/ds/demo/sum.go"),
        ("dur.91", "go/ds/demo/mean.go"),
    ):
        ss("start", mid, rc=0)
        ss.implement(unit)
    ss("check", "dur.90", rc=0)
    out = ss("check", "dur.91", rc=0).out
    assert "ok   dur.90 smoke: TestHandExample" in out and "PASS dur.91" in out


def test_go_race_detector_path(ss):
    # `ol check` runs `go test -race` by default; the other tests turn it off for speed.
    ss.init()
    ss("start", "dur.90", rc=0)
    ss.implement("go/ds/demo/sum.go")
    assert "PASS dur.90" in ss("check", "dur.90", rc=0, env={"OL_GO_RACE": "1"}).out
