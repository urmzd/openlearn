"""craft.01 course tests: your commit-msg hook (.githooks/commit-msg).

Annotated exemplars (DESIGN 5.12). The hook is graded by what it accepts and
what it rejects: a table of messages, each with the reason it is in the table.
Its verdicts must agree with the pattern MS-P0's `git-log` matcher applies to
your HEAD, so a commit your hook lets through also passes the milestone.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(os.environ["OL_LEARNER"])
HOOK = REPO / ".githooks" / "commit-msg"

# The header pattern of course/harness matchers.CONVENTIONAL (MS-P0 git-log),
# copied here so this test reads on its own.
CONVENTIONAL = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([^()\s]+\))?!?: \S.*$"
)

ACCEPT = [
    "feat: count byte bigrams without a Python loop",  # the plain form
    "fix(make): rebuild both objects when greet.h changes",  # a scope in parentheses
    "feat!: drop the v0 completions endpoint",  # ! marks a breaking change
    "chore(deps)!: require numpy 2",  # scope and ! together
    "ci: lint every pushed commit, not only HEAD",  # every type of the list is allowed
    "revert: feat: count byte bigrams",  # a colon later in the description is fine
]

REJECT = [
    "Add bigram counter",  # no type at all
    "feat:count bigrams",  # the space after the colon is required
    "feat : count bigrams",  # no space before the colon
    "feature: count bigrams",  # not one of the allowed types
    "Feat: count bigrams",  # types are lowercase in this course (and in the MS-P0 matcher)
    "fix(): empty scope",  # a scope, when present, is not empty
    "fix(make files): spaces in scope",  # a scope is one word
    "feat: ",  # the description is not empty
    "feat:  two spaces",  # exactly one space, then a non-space character
    "wip",  # what most bad history looks like
]


def run_hook(tmp_path: Path, message: str) -> subprocess.CompletedProcess:
    f = tmp_path / "COMMIT_EDITMSG"
    f.write_text(message)
    return subprocess.run(
        [str(HOOK), str(f)], capture_output=True, text=True, timeout=10, cwd=REPO
    )


def test_hook_is_an_executable_script():
    # WHY: git runs hooks by exec'ing the file, exactly as CI does: it needs
    #      the execute bit and a #! line. A non-executable hook is silently
    #      skipped by git, so a broken hook looks like a passing one.
    # KIND: unit
    # CHAPTER: craft.01 section 4
    assert HOOK.is_file(), ".githooks/commit-msg is missing"
    assert os.access(HOOK, os.X_OK), "chmod +x .githooks/commit-msg"
    assert HOOK.read_bytes().startswith(b"#!"), (
        ".githooks/commit-msg must start with a #! line"
    )


def test_hook_accepts_the_worked_example(tmp_path):
    # WHY: section 3's message, header and body and footer, worked by hand:
    #      only the first line is the header; the body is free text.
    # KIND: unit
    # CHAPTER: craft.01 section 3
    msg = (
        "feat(make): rebuild objects when greet.h changes\n\n"
        "main.o and greet.o both include greet.h, so both depend on it.\n\n"
        "Refs: lang.02\n"
    )
    r = run_hook(tmp_path, msg)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("header", ACCEPT)
def test_hook_accepts_conventional_headers(tmp_path, header):
    # WHY: each accepted form in the table is one rule of the format; a hook
    #      that rejects valid history blocks real work and gets bypassed.
    # KIND: unit
    assert CONVENTIONAL.match(header), "test table out of sync with MS-P0"
    r = run_hook(tmp_path, header + "\n")
    assert r.returncode == 0, f"rejected {header!r}: {r.stderr}"


@pytest.mark.parametrize("header", REJECT)
def test_hook_rejects_malformed_headers(tmp_path, header):
    # WHY: each rejected form is a near miss a loose pattern lets through
    #      (missing space, unknown type, empty scope). One that slips past
    #      the hook fails MS-P0 on your HEAD instead, much later.
    # KIND: boundary
    # CHAPTER: craft.01 section 5, pitfall 1
    assert not CONVENTIONAL.match(header), "test table out of sync with MS-P0"
    r = run_hook(tmp_path, header + "\n")
    assert r.returncode != 0, f"accepted {header!r}"
    assert r.stderr.strip(), "say why on stderr: the person committing needs the rule"


def test_hook_reads_the_first_line_and_ignores_comments(tmp_path):
    # WHY: git hands the hook the raw editor file, comment lines included,
    #      and strips them only afterwards. The header is the first line that
    #      is neither blank nor a comment; anything later (body, footers) is
    #      free text and must not decide the verdict either way.
    # KIND: boundary
    # CHAPTER: craft.01 section 5, pitfall 2
    ok = "\nfix: keep empty rows at zero\n\nAdd a test.\n# Please enter the commit message for your changes.\n"
    assert run_hook(tmp_path, ok).returncode == 0
    bad = "# Please enter the commit message\nAdd a test\n\nfeat: hidden in the body\n"
    assert run_hook(tmp_path, bad).returncode != 0
    assert run_hook(tmp_path, "").returncode != 0, "an empty message is not a commit"


def test_head_commit_passes_your_hook(tmp_path):
    # WHY: the gate is only real once it has judged real history: your HEAD
    #      commit's message passes your own hook and MS-P0's git-log pattern.
    # KIND: conformance
    # CHAPTER: craft.01 section 4, step 4
    r = subprocess.run(
        ["git", "log", "-1", "--format=%B"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode == 0, f"no commit yet (git log failed): {r.stderr.strip()}"
    header = r.stdout.splitlines()[0] if r.stdout else ""
    assert CONVENTIONAL.match(header), f"HEAD is not a Conventional Commit: {header!r}"
    hook = run_hook(tmp_path, r.stdout)
    assert hook.returncode == 0, f"your hook rejects HEAD: {hook.stderr}"
