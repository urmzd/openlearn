"""PII scrub with typed placeholders and audit spans (data.05).

Each detector is a candidate regex plus a validator: the regex finds text
that looks right, the validator throws out lookalikes (a card number must
pass Luhn and start like a card network's; an IPv6 candidate must parse).
Matches are replaced by a typed placeholder such as <EMAIL>, so a model
still learns that an address goes there, and the spans (kind and offsets,
never the matched text) are the audit trail.

Contract: contracts/py/corpus/pii.pyi.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence

from corpus.stage import Doc

KINDS = ("email", "phone", "card", "ip", "key")
PLACEHOLDERS = {
    "email": "<EMAIL>",
    "phone": "<PHONE>",
    "card": "<CARD>",
    "ip": "<IP>",
    "key": "<KEY>",
}
MANIFEST_KEYS = {
    "email": "emails",
    "phone": "phones",
    "card": "cards",
    "ip": "ips",
    "key": "keys",
}

_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}(?![\w-])"
)
_PHONE = re.compile(
    r"(?<![\w+])(?:"
    r"(?:\+1[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}"  # North American
    r"|\+\d{1,4}(?:[ .-]?\d{1,4}){2,6}"  # international, digits counted below
    r")(?!\w|-\d)"
)
_CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_IPV4 = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\w|\.\d)"
)
_IPV6 = re.compile(r"(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])")
_KEY = re.compile(
    r"(?<![\w-])(?:"
    r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{36}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|AIza[0-9A-Za-z_-]{35}"
    r"|tl_[a-z0-9]{4,32}_[A-Za-z0-9]{24,}"
    r")(?![\w-])"
)


@dataclass(frozen=True, slots=True)
class PiiSpan:
    """One redaction: [start, end) in the original text, never the text."""

    kind: str
    start: int
    end: int


def manifest_counts(counts: Optional[Mapping[str, int]]) -> dict[str, int]:
    """{MANIFEST_KEYS[k]: counts.get(k, 0)} for every kind, in KINDS order."""
    counts = counts or {}
    unknown = sorted(set(counts) - set(KINDS))
    if unknown:
        raise ValueError(f"unknown PII kind(s) {unknown}")
    return {MANIFEST_KEYS[k]: int(counts.get(k, 0)) for k in KINDS}


def luhn_ok(digits: str) -> bool:
    """The Luhn checksum over a string of decimal digits."""
    if not digits or not digits.isascii() or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _candidates(text: str) -> list[tuple[int, int, str]]:
    """(start, end, kind) of every validated candidate, overlaps included."""
    out: list[tuple[int, int, str]] = []
    for m in _EMAIL.finditer(text):
        out.append((m.start(), m.end(), "email"))
    for m in _PHONE.finditer(text):
        s = m.group()
        n = sum(c.isdigit() for c in s)
        if s.startswith("+") and not 8 <= n <= 15:
            continue
        out.append((m.start(), m.end(), "phone"))
    for m in _CARD.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if 13 <= len(digits) <= 19 and digits[0] in "23456" and luhn_ok(digits):
            out.append((m.start(), m.end(), "card"))
    for m in _IPV4.finditer(text):
        out.append((m.start(), m.end(), "ip"))
    for m in _IPV6.finditer(text):
        s = m.group()
        groups = [g for g in s.split(":") if g]
        if len(groups) < 2 or sum(len(g) for g in groups) < 4:
            continue
        try:
            ipaddress.IPv6Address(s)
        except ValueError:
            continue
        out.append((m.start(), m.end(), "ip"))
    for m in _KEY.finditer(text):
        out.append((m.start(), m.end(), "key"))
    return out


def detect(text: str) -> list[PiiSpan]:
    """Every PII span: sorted by start, non-overlapping; among overlapping
    candidates the earliest start wins, then the longer, then KINDS order."""
    order = {k: i for i, k in enumerate(KINDS)}
    cands = sorted(_candidates(text), key=lambda c: (c[0], -(c[1] - c[0]), order[c[2]]))
    spans: list[PiiSpan] = []
    end = 0
    for s, e, kind in cands:
        if s >= end:
            spans.append(PiiSpan(kind, s, e))
            end = e
    return spans


def redact(text: str, spans: Sequence[PiiSpan]) -> str:
    """text with each span replaced by its placeholder."""
    parts: list[str] = []
    pos = 0
    for sp in spans:
        if sp.start < pos or sp.end <= sp.start or sp.end > len(text):
            raise ValueError(
                f"span {sp} is unsorted, overlapping, empty, or outside the text"
            )
        parts.append(text[pos : sp.start])
        parts.append(PLACEHOLDERS[sp.kind])
        pos = sp.end
    parts.append(text[pos:])
    return "".join(parts)


def _with_counts(doc: Doc, text: str, spans: Sequence[PiiSpan]) -> Doc:
    """doc with text and the pii meta keys set from spans."""
    counts = {k: 0 for k in KINDS}
    for sp in spans:
        counts[sp.kind] += 1
    meta = {**doc.meta, "pii_redactions": len(spans), "pii": counts}
    return dataclasses.replace(doc, text=text, meta=meta)


def scrub(doc: Doc) -> tuple[Doc, list[PiiSpan]]:
    """(the redacted Doc, the spans in the original text)."""
    spans = detect(doc.text)
    return _with_counts(doc, redact(doc.text, spans), spans), spans


def audit_line(doc_id: str, spans: Sequence[PiiSpan]) -> str:
    """{"id": ..., "spans": [{"kind", "start", "end"}, ...]} on one line."""
    return json.dumps(
        {
            "id": doc_id,
            "spans": [{"kind": s.kind, "start": s.start, "end": s.end} for s in spans],
        },
        ensure_ascii=False,
    )


def scrub_stage(
    docs: Iterable[Doc],
    *,
    policy: Optional[Mapping[str, str]] = None,
    audit: Optional[Callable[[str, list[PiiSpan]], None]] = None,
) -> Iterator[Doc]:
    """Apply each source's pii_policy: scrub (default), drop, or none."""
    policy = policy or {}
    for doc in docs:
        mode = policy.get(doc.source_id, "scrub")
        if mode == "none":
            yield _with_counts(doc, doc.text, [])
            continue
        if mode not in ("scrub", "drop"):
            raise ValueError(
                f"unknown pii_policy {mode!r} for source {doc.source_id!r}"
            )
        spans = detect(doc.text)
        if spans and audit is not None:
            audit(doc.id, spans)
        if mode == "drop":
            if not spans:
                yield _with_counts(doc, doc.text, [])
            continue
        yield _with_counts(doc, redact(doc.text, spans), spans)
