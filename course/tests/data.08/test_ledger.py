"""Course tests for data.08: licensing ledger verification and the datasheet
(corpus/ledger.py).

Rung R0 for these course tests (your own tests for this module are rung R3:
red then green, section 4 of the chapter). Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/data.08), and the chapter section it comes from.

Corpora are written with data.06's write_shards (and scrubbed with data.05,
tokenized with data.07 where the datasheet needs them); ledger rows follow
formats/ledger.schema.json, validated here with the stdlib subset validator
next to these tests (schema_lite.py).

The chapter's worked example (section 3):

    license                              permits
    CC-BY-4.0                            train, eval
    CC-BY-NC-4.0                         eval
    MIT OR CC-BY-NC-4.0                  train, eval   (OR: the union)
    MIT AND CC-BY-NC-4.0                 eval          (AND: the intersection)
    Apache-2.0 WITH LLVM-exception       unknown       -> exit 65

    stories (CC-BY-4.0, train+eval, 2 rows)    ok
    papers  (CC-BY-NC-4.0, claims train, 1 row) one problem: the claim
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from corpus.ledger import (
    ALLOWLIST,
    EX_DATAERR,
    LedgerError,
    check,
    datasheet,
    latest,
    permitted_uses,
    read_ledger,
    reconcile,
    row_errors,
    verify,
)
from corpus.pii import scrub
from corpus.shard import write_shards
from corpus.stage import Doc
from corpus.tokenize import tokenize_shards
from schema_lite import errors

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


def row(
    sid: str,
    lic: str = "CC-BY-4.0",
    uses=("train", "eval"),
    n_docs: int = 0,
    kept: int = 0,
    **kw,
) -> dict:
    r = {
        "source_id": sid,
        "url": f"https://example.org/{sid}.jsonl",
        "license_spdx": lic,
        "retrieved_at": "2026-10-09T12:00:00Z",
        "sha256": "ab" * 32,
        "n_docs": n_docs,
        "allowed_uses": list(uses),
        "pii_policy": "scrub",
        "filters_applied": [],
        "kept": kept,
        "dropped": 0,
        "notes": "",
    }
    r.update(kw)
    return r


def write_ledger(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def doc(i: str, lic: str = "CC-BY-4.0", text: str | None = None) -> Doc:
    return Doc(
        id=i,
        source_id=i.split(":")[0],
        text=text or f"text of {i}",
        meta={"url": "u", "license_spdx": lic, "lang": "en"},
    )


def shards(tmp: Path, docs: list[Doc], **kw) -> Path:
    out = tmp / "corpus"
    write_shards(docs, out, dataset="tiny", version="v1", shard_rows=2, **kw)
    return out


# --- the worked example ------------------------------------------------------------


def test_hand_example(tmp_path):
    # WHY: section 3 by hand: what each license expression permits, and
    #      the one problem in the two-source corpus: a non-commercial
    #      source claiming "train".
    # KIND: unit
    # CATCHES: s01, s02, s03, m02, m09
    # CHAPTER: data.08 section 3, Worked example by hand
    both, ev = frozenset({"train", "eval"}), frozenset({"eval"})
    assert permitted_uses("CC-BY-4.0") == both
    assert permitted_uses("CC-BY-NC-4.0") == ev
    assert permitted_uses("MIT OR CC-BY-NC-4.0") == both
    assert permitted_uses("MIT AND CC-BY-NC-4.0") == ev
    assert permitted_uses("Apache-2.0 WITH LLVM-exception") is None
    c = shards(
        tmp_path, [doc("stories:0"), doc("stories:1"), doc("papers:0", "CC-BY-NC-4.0")]
    )
    led = write_ledger(
        tmp_path / "LEDGER.jsonl",
        [
            row("stories", n_docs=2, kept=2),
            row("papers", "CC-BY-NC-4.0", ["train"], n_docs=1, kept=1),
        ],
    )
    problems = verify(led, c)
    assert len(problems) == 1 and "papers" in problems[0]


def test_expressions():
    # WHY: OR lets the licensee pick (union), AND imposes both
    #      (intersection), and AND binds tighter than OR, as in SPDX.
    #      Parentheses, WITH, lower-case operators, an empty string, or an
    #      id outside the allowlist are unknown: never guessed.
    # KIND: unit
    # CATCHES: s01, s02, s03, s04, m02
    # CHAPTER: data.08 section 2, Principles
    both, ev = frozenset({"train", "eval"}), frozenset({"eval"})
    assert permitted_uses("CC-BY-NC-4.0 OR CC-BY-ND-4.0 AND MIT") == ev
    assert permitted_uses("MIT OR CC-BY-NC-4.0 AND CC-BY-ND-4.0") == both
    assert permitted_uses("CC-BY-NC-4.0 AND MIT OR Apache-2.0") == both
    for unknown in [
        "",
        "GPL-3.0-only",
        "(MIT)",
        "MIT or Apache-2.0",
        "MIT OR GPL-3.0-only",
        "mit",
    ]:
        assert permitted_uses(unknown) is None, unknown
    assert permitted_uses("X-1", {"X-1": ["eval"]}) == ev
    assert ALLOWLIST["CC0-1.0"] == both and ALLOWLIST["CC-BY-NC-4.0"] == ev


def test_unknown_license_is_a_data_error(tmp_path):
    # WHY: an unknown or unlisted license is a data error: check() raises
    #      LedgerError with exit code 65 (EX_DATAERR), which the subprocess
    #      contract makes non-retryable, so CorpusBuild (data.09) and the
    #      release gate (dur.12) stop instead of retrying forever.
    # KIND: boundary
    # CATCHES: s05, m03
    # CHAPTER: data.08 section 5, Pitfalls, item 1
    c = shards(tmp_path, [doc("web:0", "GPL-3.0-only")])
    led = write_ledger(
        tmp_path / "LEDGER.jsonl", [row("web", "GPL-3.0-only", n_docs=1, kept=1)]
    )
    with pytest.raises(LedgerError) as e:
        check(led, c)
    assert e.value.exit_code == EX_DATAERR == 65
    assert (
        len(e.value.problems) == 1
        and "web" in e.value.problems[0]
        and "web" in str(e.value)
    )
    write_ledger(led, [row("web", "MIT", n_docs=1, kept=1)])
    shards(tmp_path, [doc("web:0", "MIT")])
    check(led, tmp_path / "corpus")


def test_rows_follow_the_schema(tmp_path):
    # WHY: the ledger is a contract too: row_errors must agree with
    #      formats/ledger.schema.json on a valid row and on each kind of
    #      violation (missing or extra key, malformed id, hash, or time, a
    #      negative or boolean count, an enum value outside its set).
    # KIND: conformance
    # CATCHES: s06, m04, m05, m06
    # CHAPTER: data.08 section 4, What the tests check
    schema = json.loads((CONTRACTS / "formats" / "ledger.schema.json").read_text())
    good = row("web", revoked=False)
    assert row_errors(good) == [] and errors(good, schema) == []
    multimodal = dict(
        good,
        modality="audio",
        biometric=True,
        consent="cc-licensed",
        attribution="fixture author",
        prohibited_uses=["speaker-identification"],
    )
    assert row_errors(multimodal) == [] and errors(multimodal, schema) == []
    for key, value in [
        ("modality", "video"),
        ("biometric", "false"),
        ("consent", "unknown"),
        ("attribution", 7),
        ("prohibited_uses", "speaker-identification"),
    ]:
        invalid = dict(multimodal, **{key: value})
        assert row_errors(invalid) and errors(invalid, schema)
    bad = [
        {k: v for k, v in good.items() if k != "notes"},
        {**good, "extra": 1},
        {**good, "source_id": "Web Site"},
        {**good, "sha256": "AB" * 32},
        {**good, "retrieved_at": "2026-10-09 12:00:00"},
        {**good, "kept": -1},
        {**good, "dropped": True},
        {**good, "allowed_uses": ["train", "resell"]},
        {**good, "pii_policy": "mask"},
        {**good, "filters_applied": "lang"},
        {**good, "revoked": "no"},
    ]
    for b in bad:
        assert errors(b, schema) != [], b
        assert row_errors(b) != [], b


def test_bad_rows_are_reported(tmp_path):
    # WHY: a malformed ledger row is a problem in the report, not a crash
    #      and not a pass: its source then has no valid row, so its shard
    #      rows do not trace either.
    # KIND: boundary
    # CATCHES: s07, s08, m06, m07, m08
    # CHAPTER: data.08 section 4, The interface
    c = shards(tmp_path, [doc("web:0")])
    led = write_ledger(
        tmp_path / "LEDGER.jsonl", [row("web", n_docs=1, kept=1, sha256="nope")]
    )
    problems = verify(led, c)
    assert len(problems) == 2 and all("web" in p or "row 1" in p for p in problems)


def test_every_shard_row_traces_to_its_source(tmp_path):
    # WHY: the design's conformance rule: every row of the shards names a
    #      source that has a ledger row, under the same license string. A
    #      row from an unlisted source, or relicensed on the way, fails.
    # KIND: conformance
    # CATCHES: s08, s09, m07
    # CHAPTER: data.08 section 5, Pitfalls, item 2
    c = shards(tmp_path, [doc("web:0"), doc("web:1", "MIT"), doc("ghost:0")])
    led = write_ledger(tmp_path / "LEDGER.jsonl", [row("web", n_docs=2, kept=2)])
    problems = verify(led, c)
    assert any("ghost" in p for p in problems)
    assert any("web" in p and "MIT" in p for p in problems)
    assert len(problems) == 2


def test_latest_row_wins(tmp_path):
    # WHY: the ledger is append-only; a later row for a source (a revocation
    #      from the data-incident drill, ops.08) overrides the earlier one.
    #      Reading the first row would keep training on revoked data.
    # KIND: unit
    # CATCHES: s10
    # CHAPTER: data.08 section 5, Pitfalls, item 3
    rows = [row("web", n_docs=1, kept=1), row("web", n_docs=1, kept=1, revoked=True)]
    assert latest(rows)["web"]["revoked"] is True
    c = shards(tmp_path, [doc("web:0")])
    led = write_ledger(tmp_path / "LEDGER.jsonl", rows)
    assert len(verify(led, c)) == 1 and "revoked" in verify(led, c)[0]
    write_ledger(led, list(reversed(rows)))
    assert verify(led, c) == []


def test_use_is_checked_only_where_rows_exist(tmp_path):
    # WHY: an eval-only source is fine in the ledger (eval suites), but
    #      not in a training corpus: it fails for use="train" only when
    #      shard rows come from it, and passes for use="eval". A source may
    #      also claim fewer uses than its license permits ("lean" claims
    #      train only), and then it fails for eval.
    # KIND: unit
    # CATCHES: s11, m01, m07, m09
    # CHAPTER: data.08 section 2, Principles
    led = write_ledger(
        tmp_path / "LEDGER.jsonl",
        [
            row("web", n_docs=1, kept=1),
            row("bench", "CC-BY-NC-4.0", ["eval"], n_docs=3),
        ],
    )
    assert verify(led, shards(tmp_path, [doc("web:0")])) == []
    write_ledger(
        led,
        [
            row("web", n_docs=1, kept=1),
            row("bench", "CC-BY-NC-4.0", ["eval"], n_docs=3, kept=1),
            row("lean", uses=["train"], n_docs=1, kept=1),
        ],
    )
    c = shards(tmp_path, [doc("web:0"), doc("bench:0", "CC-BY-NC-4.0"), doc("lean:0")])
    problems = verify(led, c)
    assert len(problems) == 1 and "bench" in problems[0]
    problems = verify(led, c, use="eval")
    assert len(problems) == 1 and "lean" in problems[0]


def test_kept_counts_and_reconcile(tmp_path):
    # WHY: fetch (data.01) records kept = n_docs; after filtering and dedup
    #      that is wrong, and verification compares kept with the shards.
    #      reconcile() appends one corrected row per source (kept = shard
    #      rows, dropped = n_docs - kept, the filters applied), leaving the
    #      earlier rows as history, after which verification passes.
    # KIND: unit
    # CATCHES: s10, s11, s12, s13, m07, m10, m11
    # CHAPTER: data.08 section 5, Pitfalls, item 4
    c = shards(tmp_path, [doc("web:0"), doc("web:3"), doc("books:1", "CC0-1.0")])
    first = [
        row("web", n_docs=5, kept=5),
        row("books", "CC0-1.0", n_docs=1, kept=1),
        row("idle", n_docs=4, kept=4),
    ]
    led = write_ledger(tmp_path / "LEDGER.jsonl", first)
    assert sorted(p.split(":")[0] for p in verify(led, c)) == [
        "source idle",
        "source web",
    ]
    added = reconcile(led, c, filters_applied=["lang", "gopher"])
    assert [
        (r["source_id"], r["kept"], r["dropped"], r["filters_applied"]) for r in added
    ] == [
        ("books", 1, 0, ["lang", "gopher"]),
        ("idle", 0, 4, ["lang", "gopher"]),
        ("web", 2, 3, ["lang", "gopher"]),
    ]
    rows = read_ledger(led)
    assert rows[:3] == first and rows[3:] == added
    assert verify(led, c) == []


def test_read_ledger(tmp_path):
    # WHY: one JSON object per line, blank lines allowed; anything else is
    #      a ValueError that names the line, never a silently skipped row.
    # KIND: boundary
    # CATCHES: m12
    # CHAPTER: data.08 section 4, The interface
    p = tmp_path / "L.jsonl"
    p.write_text(json.dumps(row("a")) + "\n\n" + json.dumps(row("b")) + "\n")
    assert [r["source_id"] for r in read_ledger(p)] == ["a", "b"]
    for bad in ["[1]\n", "{not json\n"]:
        p.write_text(json.dumps(row("a")) + "\n" + bad)
        with pytest.raises(ValueError):
            read_ledger(p)


def test_datasheet(tmp_path):
    # WHY: the datasheet (ethics.01, Gebru et al.) is generated from the
    #      same manifest and ledger the build used, so its numbers cannot
    #      drift from the data: the template's sections in order, every
    #      source with its license, the split sizes, PII counts, every
    #      drop count, and the token counts; your own prose goes in
    #      Motivation.
    # KIND: unit
    # CATCHES: s14, m13
    # CHAPTER: data.08 section 4, What the tests check
    raw = [
        doc("web:0", text="mail a@example.com"),
        doc("web:1", text="Mia naps."),
        doc("books:0", "CC0-1.0", "a story"),
    ]
    c = tmp_path / "corpus"
    write_shards(
        [scrub(d)[0] for d in raw],
        c,
        dataset="tiny",
        version="v1",
        shard_rows=2,
        val_permille=210,
        filters={"lang": 4},
        dedup={"exact_dropped": 2, "near_dropped": 1, "decontaminated": 3},
    )
    tokenize_shards(
        c / "_MANIFEST.json", None, tmp_path / "tokens", tokenizer_id="bytes"
    )
    led = write_ledger(
        tmp_path / "LEDGER.jsonl",
        [row("web", n_docs=2, kept=2), row("books", "CC0-1.0", n_docs=1, kept=1)],
    )
    text = datasheet(led, c, tmp_path / "tokens")
    sec: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            head = line
            sec[head] = ""
        else:
            sec[head] += line + "\n"
    assert list(sec) == [
        "# Datasheet: tiny v1",
        "## Motivation",
        "## Composition",
        "## Collection",
        "## Preprocessing",
        "## Uses",
        "## Distribution and maintenance",
    ]
    # "a story" is the only val document (sha256 mod 1000 = 203 < 210); bytes
    # tokenizer: 12 + 9 train tokens ("mail <EMAIL>", "Mia naps."), 7 val
    for s in ["3 documents", "2 train", "1 val", "emails 1"]:
        assert s in sec["## Composition"], s
    for s in [
        "web",
        "books",
        "CC0-1.0",
        "CC-BY-4.0",
        "https://example.org/web.jsonl",
        "ab" * 32,
    ]:
        assert s in sec["## Collection"], s
    for s in [
        "lang 4",
        "dropped 2",
        "dropped 1",
        "jaccard_threshold 0.8",
        "dropped 3",
        "13-gram",
        "bytes",
        "21 train tokens",
        "7 val tokens",
    ]:
        assert s in sec["## Preprocessing"], s
    for s in ["web", "books", "train"]:
        assert s in sec["## Uses"], s
    assert "<" in sec["## Motivation"]
