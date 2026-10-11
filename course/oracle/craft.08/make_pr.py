# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Maintainer generator for craft.08's seeded pull request.

craft.08 (code review) hands the learner a pull request against their own
system: a scratch copy of their repo whose `go/gateway/limit/limit.go` is the
course reference of gw.03 (markers dropped) plus a feature (evict idle keys)
written with five seeded defects (DESIGN 4.6, 5.10 `git-branch`). This script
writes the review material under course/mutants/craft.08/ from the CURRENT
gw.03 reference, so a change to that reference is one rerun away:

  pr.patch       reference -> the intended PR (correct): the eviction feature
                 in limit.go plus its test file idle_evict_test.go
  sNN.patch      intended PR -> intended PR with defect NN, one hunk each,
                 1 line of context, so the five apply in order without
                 touching each other (the PR head is all five)
  manifest.tsv   one row per defect; `line` is its first line in the PR head

and the reference review course/ref/docs/reviews/craft-08-pr-review.toml (one
finding per defect at its current line, the current head's blob), which
`ol verify course craft.08` grades in the reference learner.

Every edit below is an exact, unique string replacement on the reference, so
a drift in gw.03's reference fails loudly here instead of producing a
different PR. After writing, the script rebuilds the head by applying the
patches with `patch` (what the check does) and asserts it equals the head it
built in memory. course/tests/craft.08 then proves each defect is a real bug
(a Go test that passes on the intended PR and fails with that defect alone).

    uv run --script course/oracle/craft.08/make_pr.py      (from the repo root)
"""

from __future__ import annotations

import difflib
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COURSE = ROOT / "course"
OUT = COURSE / "mutants" / "craft.08"
UNIT = "go/gateway/limit/limit.go"
TEST = "go/gateway/limit/idle_evict_test.go"

sys.path.insert(0, str(COURSE / "harness" / "src"))
from olcourse import markers  # noqa: E402


def edit(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(
            f"{what}: the anchor occurs {n} times in the text (want exactly 1):\n{old}"
        )
    return text.replace(old, new)


# -- the intended pull request ------------------------------------------------

INTENDED = [
    (
        "package doc",
        "\npackage limit\n",
        "\n//\n"
        "// Keys that go quiet are evicted (sweep), so the bucket map holds the keys\n"
        "// active in the last IdleTTL instead of every key ever seen.\n"
        "package limit\n",
    ),
    (
        "keyState fields",
        "type keyState struct {\n\treq, tok bucket\n\tlim      Limits\n}\n",
        "type keyState struct {\n\treq, tok bucket\n\tlim      Limits\n"
        "\tseen     time.Time // the last Reserve for this key, admitted or not\n"
        "\topen     int       // reservations not yet settled or cancelled\n}\n",
    ),
    (
        "constants",
        "// Limiter holds every key's buckets. Safe for concurrent use.\n",
        "// IdleTTL is how long a key may go without a Reserve before sweep may drop\n"
        "// its buckets. A dropped key comes back full on its next request, so sweep\n"
        "// drops only keys whose buckets have refilled to capacity by then.\n"
        "const IdleTTL = 2 * time.Minute\n\n"
        "// sweepEvery is the sweep cadence: a sweep walks every key under the lock,\n"
        "// so it runs once per sweepEvery Reserve calls, not on each one.\n"
        "const sweepEvery = 256\n\n"
        "// Limiter holds every key's buckets. Safe for concurrent use.\n",
    ),
    (
        "Limiter.calls",
        "\tkeys  map[string]*keyState\n}\n",
        "\tkeys  map[string]*keyState\n\tcalls int // Reserve calls, for the sweep cadence\n}\n",
    ),
    (
        "Reserve: sweep cadence and seen",
        "\tl.mu.Lock()\n\tdefer l.mu.Unlock()\n\tst := l.state(key, lim, now)\n",
        "\tl.mu.Lock()\n\tdefer l.mu.Unlock()\n\tl.calls++\n\tif l.calls%sweepEvery == 0 {\n"
        "\t\tl.sweep(now)\n\t}\n\tst := l.state(key, lim, now)\n\tst.seen = now\n",
    ),
    (
        "Reserve: open++",
        "\treturn &reservation{l: l, key: key, cost: c, status: st.status()}, nil\n",
        "\tst.open++\n\treturn &reservation{l: l, key: key, cost: c, status: st.status()}, nil\n",
    ),
    (
        "sweep and Len",
        "type reservation struct {\n",
        "// sweep drops every key that has no open reservation, has been idle for at\n"
        "// least IdleTTL, and whose buckets have refilled to capacity: dropping it\n"
        "// then changes nothing a later request can observe. The caller holds l.mu.\n"
        "func (l *Limiter) sweep(now time.Time) {\n"
        "\tfor k, st := range l.keys {\n"
        "\t\tif st.open > 0 || now.Sub(st.seen) < IdleTTL {\n"
        "\t\t\tcontinue\n"
        "\t\t}\n"
        "\t\tst.req.refill(now)\n"
        "\t\tst.tok.refill(now)\n"
        "\t\tif st.req.level < st.req.capacity || st.tok.level < st.tok.capacity {\n"
        "\t\t\tcontinue // in debt or still refilling: dropping it would forgive that\n"
        "\t\t}\n"
        "\t\tdelete(l.keys, k)\n"
        "\t}\n"
        "}\n\n"
        "// Len is the number of keys the limiter holds (the size sweep bounds).\n"
        "func (l *Limiter) Len() int {\n"
        "\tl.mu.Lock()\n"
        "\tdefer l.mu.Unlock()\n"
        "\treturn len(l.keys)\n"
        "}\n\n"
        "type reservation struct {\n",
    ),
    (
        "adjust: open--",
        "\tst, ok := l.keys[key]\n\tif !ok {\n\t\treturn\n\t}\n\tst.req.refill(now)\n",
        "\tst, ok := l.keys[key]\n\tif !ok {\n\t\treturn\n\t}\n\tst.open--\n\tst.req.refill(now)\n",
    ),
]

PR_TEST = """package limit_test

import (
	"context"
	"testing"
	"time"

	"tinyllm/gateway/limit"
)

type fakeClock struct{ t time.Time }

func (f *fakeClock) Now() time.Time { return f.t }

// A key that settled its last request and then went quiet for longer than
// IdleTTL is gone after the next sweep; the busy key stays.
func TestIdleKeyIsEvicted(t *testing.T) {
	fc := &fakeClock{t: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
	l := limit.New(fc)
	r, err := l.Reserve(context.Background(), "quiet", limit.Limits{RPM: 60, TPM: 6000}, limit.Cost{Requests: 1, Tokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	r.Settle(limit.Cost{Requests: 1, Tokens: 10})
	fc.t = fc.t.Add(3 * time.Minute)
	for i := 0; i < 255; i++ { // calls 2 to 256: the 256th sweeps
		if _, err := l.Reserve(context.Background(), "busy", limit.Limits{}, limit.Cost{Requests: 1}); err != nil {
			t.Fatal(err)
		}
	}
	if n := l.Len(); n != 1 {
		t.Fatalf("Len() = %d after the sweep, want 1 (only the busy key)", n)
	}
}
"""

# -- the seeded defects (each against the intended PR) --------------------------
#
# s01  sweep forgives debt: the refill-and-full check is replaced by a comment
#      that reasons "two minutes idle means full", true only for a level >= 0
# s02  Len reads the map without the lock ("one word"): a data race
# s03  sweep ignores open reservations: a long stream's overrun is lost
# s04  RetryAfterSeconds "tidied" to round to nearest: 1.2 s says 1
# s05  sweep cadence `%` became `>=`: every call after the 256th walks every key

DEFECTS = {
    "s01": [
        (
            "\t\tst.req.refill(now)\n"
            "\t\tst.tok.refill(now)\n"
            "\t\tif st.req.level < st.req.capacity || st.tok.level < st.tok.capacity {\n"
            "\t\t\tcontinue // in debt or still refilling: dropping it would forgive that\n"
            "\t\t}\n"
            "\t\tdelete(l.keys, k)\n",
            "\t\t// Idle for IdleTTL means two minutes of refill, and a bucket refills\n"
            "\t\t// its whole capacity every minute, so its buckets are full by now.\n"
            "\t\tdelete(l.keys, k)\n",
        )
    ],
    "s02": [
        (
            "// Len is the number of keys the limiter holds (the size sweep bounds).\n"
            "func (l *Limiter) Len() int {\n"
            "\tl.mu.Lock()\n"
            "\tdefer l.mu.Unlock()\n"
            "\treturn len(l.keys)\n",
            "// Len is the number of keys the limiter holds (the size sweep bounds). A\n"
            "// map's length is one word, so reading it needs no lock.\n"
            "func (l *Limiter) Len() int {\n"
            "\treturn len(l.keys)\n",
        )
    ],
    "s03": [
        (
            "\t\tif st.open > 0 || now.Sub(st.seen) < IdleTTL {\n",
            "\t\tif now.Sub(st.seen) < IdleTTL {\n",
        )
    ],
    "s04": [
        (
            "\ts := int(math.Ceil(d.Seconds()))\n\tif s < 1 {\n\t\ts = 1\n\t}\n\treturn s\n",
            "\treturn max(1, int(d.Round(time.Second)/time.Second))\n",
        )
    ],
    "s05": [
        (
            "\tif l.calls%sweepEvery == 0 {\n",
            "\tif l.calls >= sweepEvery {\n",
        )
    ],
}

# id, required, category, severity floor, public text (shown once you pass), private note
ROWS = [
    (
        "s01",
        "y",
        "correctness",
        "sweep forgives a key's debt",
        "a key charged an overrun (level below 0) is dropped after IdleTTL and comes back full: two idle minutes refill only two capacities",
    ),
    (
        "s02",
        "y",
        "concurrency",
        "Len reads the map without the lock",
        "len(map) concurrent with Reserve's map writes is a data race (go test -race); a map length is not an atomic word in the Go memory model",
    ),
    (
        "s03",
        "n",
        "correctness",
        "sweep ignores open reservations",
        "a request still streaming after IdleTTL loses its key; its Settle finds no key and the overrun charge is dropped",
    ),
    (
        "s04",
        "y",
        "correctness",
        "Retry-After rounds to nearest",
        "a 1.2 s wait says Retry-After 1; the contract rounds up (at least 1), so a client that obeys it is rejected again",
    ),
    (
        "s05",
        "n",
        "performance",
        "sweep runs on every call after the 256th",
        "calls >= sweepEvery stays true: from call 256 on every Reserve walks every key under the global lock, O(keys) per request",
    ),
]

# The reference review (course/ref/docs/reviews/, copied into the reference
# learner by `ol verify course`): one finding per defect at the defect's first
# line, plus one on the PR's test file. Rewritten with the current lines and
# blob every run, so it always grades against the current head.
REF_FINDINGS = {
    "s01": (
        "blocking",
        "correctness",
        "Two idle minutes refill two capacities, not everything: a key that settled an overrun sits below zero "
        "(TPM 1000, reserve 300, settle 4000 leaves -3000; 150 s later it is still -500). Dropping it here hands it "
        "a full bucket on its next request, so the overrun is never paid. The comment's reasoning holds only for a "
        "level >= 0.",
        "Refill both buckets at `now` and drop the key only when both levels are back at capacity (continue "
        "otherwise). Add a test that settles an overrun, idles past IdleTTL, sweeps, and expects a rejection.",
    ),
    "s02": (
        "blocking",
        "concurrency",
        "Len reads l.keys without l.mu while Reserve inserts and sweep deletes under it. That is a data race in the "
        "Go memory model whatever the size of the value read; go test -race with one goroutine calling Len while "
        "others Reserve new keys reports it, and a metrics scrape calls Len exactly that way.",
        "Take l.mu in Len (Lock, defer Unlock) like every other method, or keep a counter updated under the lock "
        "and read it atomically if the gauge must not contend.",
    ),
    "s03": (
        "major",
        "correctness",
        "The guard no longer checks open reservations. A stream that outlives IdleTTL (a long completion) holds a "
        "reservation; its key can be swept mid-stream, and then adjust finds no key and drops the overrun charge "
        "of its Settle without a trace.",
        "Keep `st.open > 0` in the guard so keys with in-flight reservations are never dropped, and test it: sweep "
        "during an open reservation, then settle an overrun and expect the next request rejected.",
    ),
    "s04": (
        "blocking",
        "correctness",
        "This rounds to the nearest second; the contract (and the code it replaces) rounds up with a floor of 1. "
        "A 1.2 s wait now says Retry-After: 1, so a client that obeys it comes back early and is rejected again. "
        "It is also a behavior change hidden in a tidy-up line of an unrelated PR.",
        "Restore int(math.Ceil(d.Seconds())) with the floor of 1, and move clean-ups to their own PR.",
    ),
    "s05": (
        "major",
        "performance",
        "`l.calls >= sweepEvery` stays true after the 256th call, so from then on every Reserve walks every key "
        "under the global lock: O(keys) per request, the cost the cadence exists to amortize.",
        "Use `l.calls%sweepEvery == 0` as the description says (or reset l.calls when a sweep runs).",
    ),
}
REF_TEST_FINDING = (
    "func TestIdleKeyIsEvicted",
    "minor",
    "tests",
    "The new test covers only the happy path, a settled and full key. It misses the cases that make eviction risky: "
    "a key in debt, a key with an open reservation, and Len called while other goroutines Reserve.",
    "Add those three tests (the third under -race); they would have caught three of the issues above.",
)
REF_SUMMARY = (
    "The PR bounds the limiter's memory by sweeping idle keys on every 256th Reserve and adds Len for a keys gauge. "
    "The approach is right, but three changes break observable behavior: the sweep forgives overrun debt, Len races "
    "with Reserve, and the RetryAfterSeconds rewrite tells clients to retry too early. Two more need fixing before "
    "merge: keys with in-flight reservations are swept, and the cadence check sweeps on every call after the 256th. "
    "Requesting changes."
)


def toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def udiff(a: str, b: str, path: str, n: int, new: bool = False) -> str:
    lines = difflib.unified_diff(
        a.splitlines(keepends=True),
        b.splitlines(keepends=True),
        "/dev/null" if new else f"a/{path}",
        f"b/{path}",
        n=n,
    )
    return "".join(lines)


def changed_lines(a: str, b: str) -> list[int]:
    """1-based lines of b that differ from a (for a pure deletion, the line
    that now stands where the deleted block was)."""
    out: list[int] = []
    sm = difflib.SequenceMatcher(a=a.splitlines(), b=b.splitlines(), autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out += list(range(j1 + 1, j2 + 1)) if j2 > j1 else [j1 + 1]
    return out


def apply(patches: list[Path], files: dict[str, str]) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="ol-craft08-gen-") as d:
        for rel, text in files.items():
            p = Path(d) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        for patch in patches:
            r = subprocess.run(
                ["patch", "-s", "-p1", "-d", d, "-i", str(patch)],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                raise SystemExit(f"{patch.name} does not apply:\n{r.stdout}{r.stderr}")
        return {
            rel: (Path(d) / rel).read_text()
            for rel in (UNIT, TEST)
            if (Path(d) / rel).is_file()
        }


def main() -> int:
    ref = markers.drop_markers((COURSE / "ref" / UNIT).read_text())
    intended = ref
    for what, old, new in INTENDED:
        intended = edit(intended, old, new, f"intended: {what}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pr.patch").write_text(
        udiff(ref, intended, UNIT, 3) + udiff("", PR_TEST, TEST, 3, new=True)
    )
    head = intended
    for sid, edits in DEFECTS.items():
        one = intended
        for old, new in edits:
            one = edit(one, old, new, f"{sid} on the intended PR")
            head = edit(head, old, new, f"{sid} on the stacked head")
        (OUT / f"{sid}.patch").write_text(udiff(intended, one, UNIT, 1))
    # The head the check builds: ref + pr.patch + s01..s05 in order.
    built = apply(
        [OUT / "pr.patch"] + [OUT / f"{s}.patch" for s in DEFECTS], {UNIT: ref}
    )
    if built[UNIT] != head or built[TEST] != PR_TEST:
        raise SystemExit(
            "the patches do not rebuild the head this script built in memory"
        )
    rows = ["# mid\tunit\ttier\toperator\tline\trequired\tpublic\tprivate"]
    for sid, req, cat, public, private in ROWS:
        others = intended
        for s2, edits in DEFECTS.items():
            if s2 != sid:
                for old, new in edits:
                    others = edit(others, old, new, f"{s2} (line of {sid})")
        line = changed_lines(others, head)[0]
        rows.append(f"{sid}\t{UNIT}\tseeded\t{cat}\t{line}\t{req}\t{public}\t{private}")
    (OUT / "manifest.tsv").write_text("\n".join(rows) + "\n")
    hl = head.splitlines()
    import tomllib

    spec = tomllib.loads((OUT / "pr.toml").read_text())
    for d in spec.get("decoy", []):
        hits = [i + 1 for i, x in enumerate(hl) if x.strip() == d["line"]]
        if len(hits) != 1:
            raise SystemExit(
                f"decoy {d['line']!r} occurs {len(hits)} times in the head (want 1)"
            )
    blob = hashlib.sha1(b"blob %d\0" % len(head.encode()) + head.encode()).hexdigest()
    lines_of = {r.split("\t")[0]: int(r.split("\t")[4]) for r in rows[1:]}
    tl = PR_TEST.splitlines()
    test_line = next(
        i + 1 for i, x in enumerate(tl) if x.startswith(REF_TEST_FINDING[0])
    )
    out = [
        "# The reference review of craft.08's pull request (honor system, D34). Written by",
        "# course/oracle/craft.08/make_pr.py with the current head's lines and blob.",
        "",
        f'pr      = "drill/{spec["name"]}"',
        f'blob    = "{blob}"',
        'verdict = "request-changes"',
        f"summary = {toml_str(REF_SUMMARY)}",
    ]
    for sid, (sev, cat, comment, sugg) in REF_FINDINGS.items():
        out += [
            "",
            "[[finding]]",
            f'file       = "{UNIT}"',
            f"line       = {lines_of[sid]}",
            f'severity   = "{sev}"',
            f'category   = "{cat}"',
            f"comment    = {toml_str(comment)}",
            f"suggestion = {toml_str(sugg)}",
        ]
    _, sev, cat, comment, sugg = REF_TEST_FINDING
    out += [
        "",
        "[[finding]]",
        f'file       = "{TEST}"',
        f"line       = {test_line}",
        f'severity   = "{sev}"',
        f'category   = "{cat}"',
        f"comment    = {toml_str(comment)}",
        f"suggestion = {toml_str(sugg)}",
        "",
    ]
    ref_review = COURSE / "ref" / "docs" / "reviews" / "craft-08-pr-review.toml"
    ref_review.parent.mkdir(parents=True, exist_ok=True)
    ref_review.write_text("\n".join(out))
    print(
        f"wrote {OUT.relative_to(ROOT)}/: pr.patch, {', '.join(f'{s}.patch' for s in DEFECTS)}, manifest.tsv; and {ref_review.relative_to(ROOT)}"
    )
    print(
        f"PR head: {len(hl)} lines of {UNIT}, {len(PR_TEST.splitlines())} lines of {TEST}"
    )
    for r in rows[1:]:
        f = r.split("\t")
        print(f"  {f[0]} line {f[4]:>4}  required {f[5]}  {f[6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
