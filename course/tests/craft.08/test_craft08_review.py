"""craft.08 course tests: code review of a seeded pull request.

Annotated exemplars (DESIGN 5.12). The subject is a pull request against your
system: a scratch copy of your repo whose go/gateway/limit/limit.go is gw.03's
course reference plus an idle-key eviction feature written with five seeded
defects (DESIGN 4.6, 5.10). `check` builds it and fetches it into your repo as
branch drill/craft-08; these tests then grade your findings in
docs/reviews/craft-08-pr-review.toml against the hidden defect list
(course/mutants/craft.08).

The first three tests prove the pull request itself: it rebuilds from the
current reference, the intended change passes every proof, and each seeded
defect fails its proof alone, so every defect you are graded on is a real
bug. Go runs in its own process group with a timeout; your files are never
edited (the branch lives in your repo, its commits never touch your tree).
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pr  # noqa: E402

LEARNER = Path(os.environ.get("OL_LEARNER_ROOT", ".")).resolve()


def review() -> dict:
    pr = _pr.build()
    p = LEARNER / pr.spec["review"]
    assert p.is_file(), (
        f"{pr.spec['review']} is missing: run `ol check craft.08` once to get the PR branch and the "
        "template, then review the PR (chapter section 4)"
    )
    try:
        return tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        raise AssertionError(f"{pr.spec['review']}: {e}") from None


def test_pr_rebuilds_from_the_reference():
    # WHY: the PR you review is rebuilt from the CURRENT gw.03 reference
    #      every time, so it must still apply cleanly, every seeded defect
    #      must change at least one line of the head, and each line the PR
    #      gets right on purpose (a decoy) must appear once and sit clear of
    #      every defect, or the grade would credit or blame the wrong line.
    # KIND: regression
    pr = _pr.build()
    assert pr.head[pr.unit] != pr.intended[pr.unit] != pr.reference
    assert pr.spec["test"] in pr.head, "pr.patch no longer creates the PR's test file"
    assert sorted(d["id"] for d in pr.defects) == sorted(_pr.PROOF_OF), (
        "every defect needs a proof"
    )
    lines = pr.head_lines()
    tol = int(pr.spec["grade"]["tolerance"])
    for d in pr.defects:
        assert d["lines"], f"{d['id']} changes no line of the head"
    for decoy in pr.spec["decoy"]:
        at = [i + 1 for i, x in enumerate(lines) if x.strip() == decoy["line"]]
        assert len(at) == 1, (
            f"decoy {decoy['line']!r} occurs {len(at)} times in the head"
        )
        for d in pr.defects:
            gap = min(abs(at[0] - x) for x in d["lines"])
            assert gap > 2 * tol, (
                f"decoy {decoy['line']!r} is {gap} line(s) from {d['id']}"
            )


def test_intended_pr_passes_every_proof():
    # WHY: a proof that fails on the CORRECT change would make every defect
    #      look real; the intended PR (reference plus the eviction feature
    #      with no defect) passes all five proofs and the PR's own test,
    #      under the race detector.
    # KIND: regression
    pr = _pr.build()
    rc, out = _pr.go_test(
        pr.intended[pr.unit], pr.intended[pr.spec["test"]], "Test", race=True
    )
    assert rc == 0, f"the intended PR fails go test -race:\n{_pr.tail(out)}"


def test_each_seeded_defect_is_a_real_bug():
    # WHY: you are graded on finding these five, so each must be a bug a
    #      program can show, not a style preference: the intended PR plus
    #      one defect fails the proof written for it.
    # KIND: fault
    # CATCHES: s01, s02, s03, s04, s05
    pr = _pr.build()
    for sid, (name, race) in sorted(_pr.PROOF_OF.items()):
        rc, out = _pr.go_test(pr.single[sid], None, f"^{name}$", race=race)
        assert rc not in (0, 124), (
            f"{sid}: {name} passes with the defect applied (or timed out):\n{_pr.tail(out)}"
        )
        assert "FAIL" in out and name in out, (
            f"{sid}: go test failed without failing {name}:\n{_pr.tail(out)}"
        )


def test_branch_is_the_course_pr():
    # WHY: you review the branch in your own repo; it must hold exactly the
    #      head the grade is computed on (check rebuilds it when the course
    #      changes it), with the reference commit first and the two PR
    #      commits after it.
    # KIND: smoke
    pr = _pr.build()
    br = _pr.branch(pr)
    have = _pr.branch_blob(LEARNER, pr)
    assert have is not None, (
        f"branch {br} is missing from your repo: rerun `ol check craft.08`"
    )
    assert have == _pr.blob_id(pr.head[pr.unit]), (
        f"{br} is not the current PR: rerun `ol check craft.08`"
    )
    rc, log = _pr.git(["log", "--format=%s", "-3", br], LEARNER)
    assert rc == 0, log
    subjects = log.splitlines()
    want = [c["message"].splitlines()[0] for c in reversed(pr.spec["commit"])]
    assert subjects == want, f"{br}: last three commits {subjects}, want {want}"


def test_review_is_well_formed():
    # WHY: a review a teammate can act on names the PR it read, says what to
    #      do next, and points every finding at a line of the PR head with a
    #      severity; a blocking or major finding says what to change. Ten
    #      serious findings at most: a review that blocks everything blocks
    #      nothing.
    # KIND: unit
    pr = _pr.build()
    rv = review()
    g = pr.spec["grade"]
    assert rv.get("pr") == _pr.branch(pr), (
        f"pr = {rv.get('pr')!r}, want {_pr.branch(pr)!r}"
    )
    assert rv.get("blob") == _pr.blob_id(pr.head[pr.unit]), (
        "blob does not match the PR head: the course changed the PR since you started; "
        "rerun `ol check craft.08`, review the new head, and copy its blob from a fresh template"
    )
    assert rv.get("verdict") in _pr.VERDICTS, (
        f"verdict {rv.get('verdict')!r} is not one of {', '.join(_pr.VERDICTS)}"
    )
    assert len(" ".join(str(rv.get("summary", "")).split())) >= 80, (
        "summary: at least 80 characters on what the PR does and what blocks it"
    )
    fs = _pr.findings(rv)
    assert fs, "no [[finding]] tables yet"
    n_lines = {
        pr.unit: len(pr.head_lines()),
        pr.spec["test"]: len(pr.head[pr.spec["test"]].splitlines()),
    }
    for i, f in enumerate(fs, 1):
        where = f"finding {i}"
        assert f["file"] in n_lines, (
            f"{where}: file {f['file']!r} is not in the PR ({', '.join(n_lines)})"
        )
        ln = f.get("line")
        assert isinstance(ln, int) and 1 <= ln <= n_lines[f["file"]], (
            f"{where}: line {ln!r} is not a line of {f['file']} at the PR head (1 to {n_lines[f['file']]})"
        )
        assert f.get("severity") in _pr.SEVERITIES, (
            f"{where}: severity {f.get('severity')!r} is not one of {', '.join(_pr.SEVERITIES)}"
        )
        assert f.get("category") in _pr.CATEGORIES, (
            f"{where}: category {f.get('category')!r} is not one of {', '.join(_pr.CATEGORIES)}"
        )
        assert len(" ".join(str(f.get("comment", "")).split())) >= 30, (
            f"{where}: comment shorter than 30 characters"
        )
        if f["severity"] in _pr.SERIOUS:
            assert len(" ".join(str(f.get("suggestion", "")).split())) >= 20, (
                f"{where}: a {f['severity']} finding needs a suggestion (at least 20 characters)"
            )
    serious = sum(f["severity"] in _pr.SERIOUS for f in fs)
    assert serious <= int(g["max_serious"]), (
        f"{serious} blocking or major findings; at most {g['max_serious']}"
    )


def test_verdict_requests_changes():
    # WHY: the PR carries defects that change what clients and operators
    #      observe; a verdict that lets it merge (approve, or comment without
    #      asking for changes) ships them.
    # KIND: unit
    # CATCHES: s02, s04
    rv = review()
    assert rv.get("verdict") == "request-changes", (
        f"verdict {rv.get('verdict')!r}: this PR must not merge as it is"
    )


def test_every_required_defect_is_found():
    # WHY: three of the five seeded defects change behavior a client, a
    #      test, or the race detector can observe on the common path; each
    #      needs a blocking or major finding on its lines (one line of slack
    #      either side).
    # KIND: unit
    # CATCHES: s01, s02, s04
    pr = _pr.build()
    gr = _pr.grade(review())
    n_req = sum(d["required"] for d in pr.defects)
    n_hit = sum(
        d["required"]
        and d["id"] in gr.found
        and gr.found[d["id"]].get("severity") in _pr.SERIOUS
        for d in pr.defects
    )
    if (
        n_hit != n_req
    ):  # a plain raise: pytest's assert introspection would print the hidden defect list
        raise AssertionError(
            f"{n_hit} of the {n_req} required defects have a blocking or major finding on their lines; "
            "a missed one, or one you marked minor, survived your review (no hint which: chapter section 2)"
        )


def test_most_defects_are_found():
    # WHY: the other two are quieter (an edge case, a cost) and still bugs;
    #      a careful review finds at least four of the five, at any
    #      severity.
    # KIND: unit
    # CATCHES: s03, s05
    pr = _pr.build()
    gr = _pr.grade(review())
    want, n_found, n_all = (
        int(pr.spec["grade"]["min_found"]),
        len(gr.found),
        len(pr.defects),
    )
    if n_found < want:
        raise AssertionError(
            f"{n_found} of {n_all} seeded defects found; want at least {want}"
        )


def test_no_serious_finding_blocks_a_correct_line():
    # WHY: a review costs the author time for every false alarm. Blocking
    #      or major findings on limit.go that hit no seeded defect are
    #      allowed once (you may see a real issue we did not plant), and
    #      never on a line the PR gets right on purpose.
    # KIND: unit
    pr = _pr.build()
    gr = _pr.grade(review())
    for f, text in gr.decoy_hits:
        raise AssertionError(
            f"finding at line {f['line']} ({f['severity']}) blocks `{text}`, which is correct as written; "
            "argue it in a question, or downgrade it to minor"
        )
    cap, n = int(pr.spec["grade"]["max_unmatched"]), len(gr.unmatched)
    if n > cap:
        raise AssertionError(
            f"{n} blocking or major findings on limit.go hit no seeded defect "
            f"(lines {', '.join(str(f['line']) for f in gr.unmatched)}); at most {cap}"
        )
