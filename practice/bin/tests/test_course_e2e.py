"""ol verify course --e2e on the Python part of the fixture course (M90.1 and
its upgrade M90.3): the reference learner passes check --all --ci, the pr
milestones in --smoke mode, ol conform on both tiers, and ol export with its
vendored tests run natively. (Export glue for C, Rust, and Go is
covered by test_course_export.py; trimming keeps each test under 10 s.)"""

import shutil


def _python_only(ss, keep=("M90.1", "M90.3")):
    for p in (ss.course / "modules").glob("*.toml"):
        if p.stem not in keep:
            p.unlink()
    for d in ("c", "rust", "go"):
        shutil.rmtree(ss.course / "ref" / d)
    ss.commit_site("trimmed")


def test_e2e_reference_learner(ss):
    _python_only(ss)
    out = ss("verify", "course", "--e2e", rc=0).out
    for line in (
        "2 module(s) started, entry points, primers, docs, system.toml",
        "ok    e2e ol check --all --ci",
        "ol milestone MS-M90 --smoke",
        "ol milestone MS-P0 --smoke",
        "ol milestone MS-P1 --smoke",
        "ol conform openapi:v0 --target engine",
        "ol conform openapi:v0 --target gateway",
        "vendored python tests run natively",
        "ol verify course --e2e: ok",
    ):
        assert line in out, line


def test_e2e_catches_a_broken_entry_point(ss):
    _python_only(ss, keep=("M90.1",))
    eng = ss.course / "ref/entry/serve/engine.py"
    eng.write_text(eng.read_text().replace('"text/event-stream"', '"text/plain"'))
    ss.commit_site("break the reference engine")
    out = ss("verify", "course", "--e2e", rc=1).out
    assert (
        "FAIL     e2e ol conform openapi:v0 --target engine" in out
        or "FAIL  e2e ol conform openapi:v0 --target engine" in out
    )


def test_kind_mode_needs_the_cluster(ss):
    _python_only(ss)
    out = ss("verify", "course", "--kind", rc=1).out
    assert "--kind needs the reference cluster" in out


def test_verify_global_parses_milestones_and_their_fixtures(ss):
    out = ss("verify", "course", "--global", rc=0).out
    assert "milestone, drill, conformance, and parity spec(s) parse" in out
    ms = ss.course / "milestones/MS-M90.toml"
    ms.write_text(
        ms.read_text()
        .replace("greedy_{prompt}.json", "missing_{prompt}.json")
        .replace('requires = ["M90.1"]', 'requires = ["M99.9"]')
    )
    out = ss("verify", "course", "--global", rc=1).out
    assert (
        "fixture course/fixtures/MS-M90/missing_ab.json has no MANIFEST.tsv row" in out
    )
    assert "requires 'M99.9', which is not a module or milestone" in out
