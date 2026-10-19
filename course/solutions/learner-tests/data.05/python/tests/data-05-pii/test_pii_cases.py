"""Model test suite for data.05 (red then green): written before the
scrubber, from the interface and the chapter's pitfalls. They import only
names in contracts/py/corpus/{pii,stage}.pyi."""

import json

import pytest
from corpus.pii import PiiSpan, audit_line, detect, luhn_ok, redact, scrub, scrub_stage
from corpus.stage import Doc

HAND = "Mail ana@example.com or call (212) 555-0123; card 4111 1111 1111 1111."


def doc(i, text, source="s", **meta):
    return Doc(id=i, source_id=source, text=text, meta=meta)


def found(text):
    return [(s.kind, text[s.start : s.end]) for s in detect(text)]


def test_hand_example():
    out, spans = scrub(doc("d", HAND))
    assert spans == [
        PiiSpan("email", 5, 20),
        PiiSpan("phone", 29, 43),
        PiiSpan("card", 50, 69),
    ]
    assert out.text == "Mail <EMAIL> or call <PHONE>; card <CARD>."
    assert out.meta["pii"] == {"email": 1, "phone": 1, "card": 1, "ip": 0, "key": 0}


def test_luhn_textbook_numbers():
    assert luhn_ok("79927398713") and not luhn_ok("79927398710")
    assert not luhn_ok("") and not luhn_ok("12a4")


def test_spans_point_into_the_original():
    text = "x a.long.address@example.com y 203.0.113.77"
    _, spans = scrub(doc("d", text))
    assert [text[s.start : s.end] for s in spans] == [
        "a.long.address@example.com",
        "203.0.113.77",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "isbn 9780306406156 and 978-0-306-40615-7",
        "react@18.3.1 and v1.2.3.4 and 1.2.3.4.5",
        "at 12:30:45 use std::vector and a[1::2]",
        "+1234567890123456789",
    ],
)
def test_lookalikes_stay(text):
    assert detect(text) == []


def test_every_kind_is_found():
    assert found("a@example.org") == [("email", "a@example.org")]
    assert found("call 212.555.0123") == [("phone", "212.555.0123")]
    assert found("card 4780306406151") == [("card", "4780306406151")]
    assert found("ip 2001:db8::1 up") == [("ip", "2001:db8::1")]
    assert found("ip 256.1.1.1 no") == []
    assert found("k=sk-" + "a" * 24) == [("key", "sk-" + "a" * 24)]
    assert found("xsk-" + "a" * 24) == []


def test_overlapping_candidates_give_one_span():
    assert found("to 1.2.3.4@example.com") == [("email", "1.2.3.4@example.com")]


def test_counts_and_meta_survive():
    out, _ = scrub(doc("d", "a@example.org b@example.org", url="u"))
    assert (
        out.meta["url"] == "u"
        and out.meta["pii"]["email"] == 2
        and out.meta["pii_redactions"] == 2
    )
    again, spans = scrub(out)
    assert spans == [] and again.meta["pii_redactions"] == 0


def test_redact_checks_spans():
    with pytest.raises(ValueError):
        redact("abcdef", [PiiSpan("ip", 3, 5), PiiSpan("ip", 0, 2)])


def test_policies_and_audit():
    seen = []
    docs = [
        doc("1", "a@example.org", "web"),
        doc("2", "b@example.org", "forum"),
        doc("3", "c@example.org", "books"),
    ]
    out = list(
        scrub_stage(
            docs,
            policy={"forum": "drop", "books": "none"},
            audit=lambda i, sp: seen.append(i),
        )
    )
    assert [d.text for d in out] == ["<EMAIL>", "c@example.org"]
    assert seen == ["1", "2"]
    assert json.loads(audit_line("1", detect("a@example.org"))) == {
        "id": "1",
        "spans": [{"kind": "email", "start": 0, "end": 13}],
    }
