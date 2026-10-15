# contracts/py/corpus/pii.pyi (data.05): PII scrub with typed placeholders
# and audit spans
# chapter: content/tracks/data-engineering/05-corpus-pipeline/05-pii-scrub.md
#
# Five detectors, each a candidate regex plus a validator. Offsets are
# Python str indexes (code points) into the text the detector was given.
#
#   email  local@domain.tld: local [A-Za-z0-9._%+-]+, dot-separated domain
#          labels [A-Za-z0-9-], a final label of 2+ letters; not preceded by
#          [\w.+-], not followed by [\w-]
#   phone  North American (+1 optional; (212) 555-0123, 212-555-0123,
#          212.555.0123, 212 555 0123), or international: "+" then 8 to 15
#          digits in total, optionally in groups split by single spaces,
#          dots, or dashes; not inside a longer run of digits or word chars
#   card   13 to 19 digits, optionally split by single spaces or dashes,
#          first digit 2 to 6 (card networks; ISBN-13 starts 978 or 979),
#          and Luhn-valid
#   ip     IPv4 dotted quad, each octet 0 to 255, not part of a longer
#          dotted run (so 1.2.3.4.5 and v1.2.3.4 are not addresses); IPv6
#          that ipaddress.IPv6Address accepts, with at least two non-empty
#          groups and four hex digits in total (so 1::2 in a slice and ::1
#          are not)
#   key    API-key shapes: sk-... (20+ chars after the prefix, sk-proj- and
#          sk-ant- included), AKIA + 16 [0-9A-Z], gh[pousr]_ + 36 alnum,
#          xox[abprs]- + 10+ [A-Za-z0-9-], AIza + 35 [0-9A-Za-z_-], and the
#          course's own tl_<id>_<secret> (id [a-z0-9]{4,32}, secret 24+
#          alnum, D19); not preceded or followed by [\w-]
#
# Overlaps resolve left to right: among candidates that overlap, the one
# that starts first wins, then the longer, then the earlier kind in KINDS.
# Placeholders never match a detector, so scrubbing twice changes nothing.
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence

from corpus.stage import Doc

KINDS: tuple[str, ...]  # ("email", "phone", "card", "ip", "key")
PLACEHOLDERS: dict[
    str, str
]  # {"email": "<EMAIL>", "phone": "<PHONE>", "card": "<CARD>", "ip": "<IP>", "key": "<KEY>"}
MANIFEST_KEYS: dict[
    str, str
]  # kind -> _MANIFEST.json pii key: {"email": "emails", "phone": "phones", "card": "cards", "ip": "ips", "key": "keys"}

@dataclass(frozen=True, slots=True)
class PiiSpan:
    """One redaction: [start, end) in the original text. It never holds the
    matched text, so an audit log of spans is not itself PII."""

    kind: str
    start: int
    end: int

def manifest_counts(counts: Optional[Mapping[str, int]]) -> dict[str, int]:
    """One document's meta["pii"] under the manifest's names: {MANIFEST_KEYS[k]:
    counts.get(k, 0)} for every kind in KINDS order; {} or None gives zeros.
    ValueError for a kind outside KINDS. data.06 sums these into the
    manifest's pii object."""

def luhn_ok(digits: str) -> bool:
    """The Luhn checksum of a string of decimal digits (no separators):
    from the rightmost digit, double every second digit, subtract 9 from a
    doubled digit above 9, and require the sum to be a multiple of 10.
    False for an empty string or a non-digit."""

def detect(text: str) -> list[PiiSpan]:
    """Every PII span of text: sorted by start, non-overlapping."""

def redact(text: str, spans: Sequence[PiiSpan]) -> str:
    """text with each span replaced by PLACEHOLDERS[span.kind]. ValueError
    when spans are unsorted, overlap, or fall outside text."""

def scrub(doc: Doc) -> tuple[Doc, list[PiiSpan]]:
    """(the redacted Doc, the spans in the original text). The new Doc keeps
    id, source_id, and every meta key, and sets meta["pii_redactions"] (the
    number of spans) and meta["pii"] ({kind: count} for every kind)."""

def audit_line(doc_id: str, spans: Sequence[PiiSpan]) -> str:
    """One JSON object on one line, no trailing newline:
    {"id": doc_id, "spans": [{"kind", "start", "end"}, ...]}."""

def scrub_stage(
    docs: Iterable[Doc],
    *,
    policy: Optional[Mapping[str, str]] = None,
    audit: Optional[Callable[[str, list[PiiSpan]], None]] = None,
) -> Iterator[Doc]:
    """A Stage. policy maps source_id to the ledger's pii_policy (default
    "scrub" for every source): "scrub" yields scrub(doc)[0]; "drop" drops a
    document with any span and yields the others with zero counts; "none"
    yields the document unchanged except for zero counts (no detection).
    audit(doc.id, spans) is called once for every scanned document with at
    least one span, before the drop decision. ValueError for another
    policy value."""
