"""Model test suite for data.08 (red then green): written before the
verifier, from formats/ledger.schema.json, the license rules, and the
chapter's pitfalls. They import only names in contracts/py/corpus/
{ledger,shard,stage}.pyi."""

import json

import pytest
from corpus.ledger import (
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
from corpus.shard import write_shards
from corpus.stage import Doc

BOTH, EVAL = frozenset({"train", "eval"}), frozenset({"eval"})


def row(sid, lic="CC-BY-4.0", uses=("train", "eval"), n=1, kept=1, **kw):
    r = dict(
        source_id=sid,
        url="https://x/" + sid,
        license_spdx=lic,
        retrieved_at="2026-10-09T12:00:00Z",
        sha256="0" * 64,
        n_docs=n,
        allowed_uses=list(uses),
        pii_policy="scrub",
        filters_applied=[],
        kept=kept,
        dropped=0,
        notes="",
    )
    r.update(kw)
    return r


def setup(tmp, rows, docs):
    led = tmp / "LEDGER.jsonl"
    led.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = [
        Doc(
            id=i,
            source_id=i.split(":")[0],
            text=i,
            meta={"url": "u", "license_spdx": lic},
        )
        for i, lic in docs
    ]
    write_shards(ds, tmp / "c", dataset="d", version="v1", shard_rows=4)
    return led, tmp / "c"


def test_hand_example_expressions():
    assert permitted_uses("MIT OR CC-BY-NC-4.0") == BOTH
    assert permitted_uses("MIT AND CC-BY-NC-4.0") == EVAL
    assert permitted_uses("CC-BY-NC-4.0 OR CC-BY-ND-4.0 AND MIT") == EVAL
    for u in ["Apache-2.0 WITH LLVM-exception", "(MIT)", "MIT OR GPL-3.0-only"]:
        assert permitted_uses(u) is None


def test_unknown_license_exits_65(tmp_path):
    led, c = setup(tmp_path, [row("a", "GPL-3.0-only")], [("a:0", "GPL-3.0-only")])
    with pytest.raises(LedgerError) as e:
        check(led, c)
    assert e.value.exit_code == EX_DATAERR == 65


def test_clean_corpus_passes(tmp_path):
    led, c = setup(
        tmp_path, [row("a", kept=2, n=2)], [("a:0", "CC-BY-4.0"), ("a:1", "CC-BY-4.0")]
    )
    assert verify(led, c) == []
    check(led, c)


def test_untraceable_rows_fail(tmp_path):
    led, c = setup(tmp_path, [row("a")], [("a:0", "MIT"), ("b:0", "MIT")])
    problems = verify(led, c)
    assert any("b" in p for p in problems) and any("MIT" in p for p in problems)


def test_noncommercial_cannot_claim_train(tmp_path):
    led, c = setup(
        tmp_path, [row("a", "CC-BY-NC-4.0", ["train"])], [("a:0", "CC-BY-NC-4.0")]
    )
    assert len(verify(led, c)) == 1


def test_latest_row_and_revocation(tmp_path):
    led, c = setup(tmp_path, [row("a"), row("a", revoked=True)], [("a:0", "CC-BY-4.0")])
    assert latest(read_ledger(led))["a"]["revoked"] is True
    assert len(verify(led, c)) == 1


def test_schema_violations():
    assert row_errors(row("a")) == []
    for bad in [
        {**row("a"), "kept": True},
        {**row("a"), "zzz": 1},
        {k: v for k, v in row("a").items() if k != "url"},
        {**row("a"), "pii_policy": "x"},
    ]:
        assert row_errors(bad)


def test_reconcile_appends_corrected_rows(tmp_path):
    led, c = setup(
        tmp_path, [row("a", n=5, kept=5)], [("a:0", "CC-BY-4.0"), ("a:1", "CC-BY-4.0")]
    )
    assert verify(led, c)
    added = reconcile(led, c, filters_applied=["lang"])
    assert (added[0]["kept"], added[0]["dropped"], added[0]["filters_applied"]) == (
        2,
        3,
        ["lang"],
    )
    assert len(read_ledger(led)) == 2 and verify(led, c) == []


def test_read_ledger_rejects_non_objects(tmp_path):
    p = tmp_path / "L.jsonl"
    p.write_text("[]\n")
    with pytest.raises(ValueError):
        read_ledger(p)


def test_datasheet_sections(tmp_path):
    led, c = setup(tmp_path, [row("a")], [("a:0", "CC-BY-4.0")])
    heads = [x for x in datasheet(led, c).splitlines() if x.startswith("#")]
    assert (
        heads[0] == "# Datasheet: d v1"
        and heads[-1] == "## Distribution and maintenance"
        and len(heads) == 7
    )
