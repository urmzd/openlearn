"""ol export: refusals (dirty tree, incomplete), what is vendored and what is
never copied, STATUS.md, and the vendored tests running natively through the
generated glue in each language."""

import json
import subprocess

from conftest import git


def _pass(ss, mid, *units, owner=None):
    ss("start", mid, rc=0)
    for u in units:
        ss.implement(u, owner=owner)
    ss("check", mid, rc=0)


def _export(ss, *extra, rc=0):
    dest = ss.tmp / "export"
    p = ss("export", str(dest), "--json", *extra, rc=rc)
    return dest, (json.loads(p.stdout.strip().splitlines()[-1]) if rc == 0 else p.out)


def _native(dest, cmds):
    for c in cmds:
        p = subprocess.run(
            ["bash", "-c", c], cwd=dest, capture_output=True, text=True, timeout=300
        )
        assert p.returncode == 0, f"{c}\n{p.stdout}\n{p.stderr}"


def test_refusals(ss):
    ss.init()
    ss.commit_learner("chore: init")
    _pass(ss, "M90.1", "python/tinyllm/demo/scale.py", owner="M90.1")
    assert "uncommitted changes" in ss("export", str(ss.tmp / "x"), rc=1).out
    ss.commit_learner("feat(M90.1): scale")
    out = ss("export", str(ss.tmp / "x"), rc=1).out
    assert (
        "incomplete" in out
        and "--allow-incomplete" in out
        and not (ss.tmp / "x").exists()
    )
    (ss.tmp / "full").mkdir()
    (ss.tmp / "full" / "f").write_text("x")
    assert (
        "not empty"
        in ss("export", str(ss.tmp / "full"), "--allow-incomplete", rc=1).out
    )


def test_python_and_c_vendored_and_run_natively(ss):
    # These active tests read a course contract and the asset catalog through
    # OL_COURSE_TREE. Export must preserve those paths in the vendored tree.
    module = ss.course / "modules" / "M90.2.toml"
    text = module.read_text()
    text = text.replace(
        'contract  = ["contracts/py/tinyllm/demo/norm.pyi"]',
        'contract  = ["contracts/py/tinyllm/demo/norm.pyi", "contracts/formats/policy.v1.schema.json"]',
    )
    module.write_text(text)
    policy = ss.course / "contracts" / "formats" / "policy.v1.schema.json"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text('{"type":"object","properties":{"rules":{"type":"array"}}}\n')
    assets = ss.course / "fixtures" / "ASSETS.tsv"
    assets.write_text("asset\tpath\turl\trevision\tbytes\tsha256\tlicense\tsource\n")
    ss.commit_site("fixture: policy schema and assets catalog")
    ss.init()
    ss.install_system()
    _pass(ss, "M90.1", "python/tinyllm/demo/scale.py", owner="M90.1")
    _pass(ss, "rt.90", "c/src/runtime/abi.c")
    _pass(ss, "rt.91", "c/src/runtime/demo.c")
    _pass(ss, "M90.2", "python/tinyllm/demo/norm.py")
    ss("milestone", "MS-M90", rc=0)
    ss.commit_learner("feat: scale, abi, C sum, normalize")
    dest, res = _export(
        ss, "--allow-incomplete", "--remote", "git@github.com:me/forge.git"
    )
    v = dest / "third_party/openlearn"
    assert set(res["vendored"]) == {"M90.1", "M90.2", "rt.90", "rt.91"}
    for f in (
        "tests/M90.1/test_scale.py",
        "tests/M90.2/test_norm.py",
        "tests/rt.90/test_abi.c",
        "tests/_lib/close.py",
        "tests/conftest.py",
        "fixtures/M90.2/vectors.json",
        "fixtures/MANIFEST.tsv",
        "fixtures/ASSETS.tsv",
        "contracts/formats/policy.v1.schema.json",
        "VERSION",
        "NOTICE",
        "STATUS.md",
    ):
        assert (v / f).is_file(), f
    assert not (v / "tests/M90.3").exists()  # not passed, not vendored
    assert res["commands"]["python"][0].startswith("uv run --project python")
    assert not list(dest.rglob("SOLUTION-BEGIN*")) and not (dest / ".ol").exists()
    assert not any(
        "SOLUTION-BEGIN" in p.read_text(errors="ignore")
        for p in dest.rglob("*")
        if p.is_file() and ".git" not in p.parts
    )
    status = (v / "STATUS.md").read_text()
    assert (
        "| M90.1 | pass |" in status
        and "| M90.3 | incomplete |" in status
        and "- MS-M90: pass" in status
    )
    assert (
        git(dest, "remote", "get-url", "origin").strip()
        == "git@github.com:me/forge.git"
    )
    log = git(dest, "log", "--format=%s")
    assert (
        log.startswith("chore(tests): vendor openlearn course tests")
        and "feat: scale, abi, C sum" in log
    )
    assert git(dest, "status", "--porcelain").strip() == ""
    _native(dest, res["commands"]["python"] + res["commands"]["c"])


def test_go_and_rust_glue_run_natively(ss):
    ss.init()
    ss.commit_learner("chore: init")
    _pass(ss, "dur.90", "go/ds/demo/sum.go")
    _pass(
        ss, "ds.90", "rust/crates/tl-demo/src/lib.rs", "rust/crates/tl-demo/src/acc.rs"
    )
    ss.commit_learner("feat: go sum and rust kahan")
    dest, res = _export(ss, "--allow-incomplete")
    v = dest / "third_party/openlearn"
    assert (v / "go.work").is_file() and (v / "tests/go/dur_90/sum_test.go").is_file()
    assert (v / "rust/Cargo.toml").is_file() and (
        v / "rust/ol-tests/tests/ds_90.rs"
    ).is_file()
    assert "tl-demo" in (v / "rust/ol-tests/Cargo.toml").read_text()
    _native(
        dest,
        res["commands"]["go"]
        + [
            c.replace("cargo test", f"CARGO_TARGET_DIR={ss.tmp}/t cargo test -q")
            for c in res["commands"]["rust"]
        ],
    )
