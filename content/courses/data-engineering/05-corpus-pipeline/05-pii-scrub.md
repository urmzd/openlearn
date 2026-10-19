# PII scrub with typed placeholders and audit spans

## Overview

| | |
|---|---|
| **Module** | `data.05` · build · Python · Pass 3 · 3 to 4 h |
| **Solution** | [`python/corpus/pii.py`](../../../../course/solutions/python/corpus/pii.py): `luhn_ok`, `detect`, `redact`, `scrub`, `audit_line`, `scrub_stage` (a stage), `manifest_counts`, the `PiiSpan` record, and the tables `KINDS`, `PLACEHOLDERS`, `MANIFEST_KEYS` |
| **Contract** | [`course/contracts/py/corpus/pii.pyi`](../../../../course/contracts/py/corpus/pii.pyi) · the counts it feeds: [`formats/corpus-manifest.schema.json`](../../../../course/contracts/formats/corpus-manifest.schema.json) · the policy field: [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json) (`pii_policy`) |
| **Needs** | `data.02` `Doc` · `data.04` near dedup runs first · reading: `ethics.02` (the PII policy this stage enforces) |
| **Used by** | `data.06` sums the per-document counts into the manifest · `data.08` reports them in the datasheet · later: `gw.08` ports the detector list to Go for log redaction |
| **Optional depth** | Microsoft [Presidio](https://microsoft.github.io/presidio/) (analyzers and recognizers, free); Subramani et al., [*Detecting Personal Information in Training Corpora*](https://arxiv.org/abs/2307.03980) (free); ISO/IEC 7812-1 (card numbers and the Luhn check); RFC 5952 (IPv6 text form) |

## Key Takeaways

- A detector is a candidate regex plus a validator: the regex finds what looks right, the validator (Luhn plus a network prefix for cards, an address parser for IPv6) throws out lookalikes.
- Typed placeholders (`<EMAIL>`, `<CARD>`) keep the shape of the text, so the model still learns that an address goes there without learning anyone's address.
- The audit trail is the spans (kind and offsets into the original text), never the matched text: an audit log that copies the PII is a second copy of it.
- The policy is per source, from the ledger: scrub, drop the document, or skip detection for a source known to be clean.

## How to read this chapter

Sections 1 to 3 build the detectors (candidate regex plus validator), overlap resolution, typed placeholders, and audit spans from first principles and work one sentence by hand. Section 4 fixes the interface and lists the cases a correct solution must handle, including the model test suite. Section 5 walks through `pii.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

After `data.04` the corpus is deduplicated and decontaminated, and it is about to be written to shards that a model will read thousands of times. Some documents still hold an email address, a phone number, a card number pasted into a forum post, a server's IP address, or an API key someone committed by mistake. A language model memorizes rare strings that appear verbatim, and a generation can repeat them to anyone who asks the right prefix. The engine will serve that model through the gateway, so a key in the training data is a key in the API's output. This module replaces each match with a typed placeholder before the text is sharded, counts what it replaced per document and per kind for the manifest and the datasheet, and records where, without recording what.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T$ | a document's text, indexed by code point | `str` |
| $[s, e)$ | a span: the code points $s$ to $e - 1$ of $T$ | `PiiSpan(kind, start, end)` |
| kind | one of email, phone, card, ip, key | `str` |
| $d_1 \dots d_m$ | the decimal digits of a number, $d_m$ the rightmost | digits |
| $L$ | the Luhn sum (below) | `int` |

**PII here means five machine-findable kinds.** Email addresses, phone numbers, payment card numbers, IP addresses, and API keys. Names and street addresses are personal data too, but finding them needs a named-entity model with its own error rates; the datasheet (`data.08`) says plainly that they are not scrubbed. `ethics.02` is where the decision is made whether that is acceptable for this corpus.

**Candidate, then validate.** A regex alone is either too loose (every 16-digit number is a card) or too strict (misses real formats). Each detector here is a permissive regex followed by a check that the match is real:

- **email**: a local part of `[A-Za-z0-9._%+-]`, `@`, dot-separated labels, and a last label of at least two letters, so `react@18.3.1` (an npm version spec) is not an address and `root@localhost` (no top-level domain) is not either.
- **phone**: North American layouts `(212) 555-0123`, `212-555-0123`, `212.555.0123`, `+1 212 555 0123`, or `+` and 8 to 15 digits in groups (the E.164 limit). A bare 7-digit `555-0123` is too ambiguous to call a phone.
- **card**: 13 to 19 digits, single spaces or dashes allowed between them, **first digit 2 to 6** (the card networks' ranges), and a valid Luhn checksum.
- **ip**: an IPv4 dotted quad with every octet at most 255, not part of a longer dotted run (`1.2.3.4.5`, `v1.2.3.4` are versions); an IPv6 candidate that `ipaddress.IPv6Address` accepts, with at least two non-empty groups and four hex digits (so the Python slice `a[1::2]` and the loopback `::1`, which is not personal, stay).
- **key**: provider shapes (`sk-...`, `AKIA` + 16, `ghp_` + 36, `xoxb-...`, `AIza` + 35) and the gateway's own `tl_<id>_<secret>` (D19), never inside a longer word.

**The Luhn check.** Card numbers carry a check digit. Starting from the rightmost digit, double every second digit ($d_{m-1}, d_{m-3}, \dots$); when a doubled digit exceeds 9, subtract 9 (the same as adding its two digits); sum everything into $L$. The number is valid when $L \bmod 10 = 0$. A random number passes with probability $1/10$, which is why the network prefix matters: an ISBN-13 starts with 978 or 979 and a millisecond timestamp with 1, and about one in ten of them passes Luhn.

**Overlaps resolve left to right.** Detectors run independently, so matches overlap: `1.2.3.4@example.com` is an IPv4 and an email, `+4222222222222` is a phone and a card. Sort candidates by start, then longer first, then by the order of `KINDS`, and keep each candidate that starts at or after the end of the last kept one. The result is sorted and non-overlapping, which `redact` requires.

**Typed placeholders.** `redact` replaces each span by `<EMAIL>`, `<PHONE>`, `<CARD>`, `<IP>`, or `<KEY>`. A placeholder keeps the sentence grammatical and tells the model what kind of thing stood there. No placeholder matches any detector, so scrubbing twice changes nothing: a retried activity (`data.09`) is safe.

**Audit spans.** `scrub(doc)` returns the new document and the spans, with offsets into the **original** text. Offsets into the redacted text would shift after every placeholder of a different length and point at the wrong characters. `audit_line(id, spans)` writes `{"id", "spans": [{"kind", "start", "end"}]}`: enough to count, sample, and review a decision against the source later, and nothing that is itself personal data.

**Per-source policy.** The ledger row of every source carries `pii_policy` (`ethics.02` fills it): `scrub` (default) redacts; `drop` removes any document that has a span (for sources where PII signals something you do not want, such as a scraped contact page), keeping clean ones; `none` skips detection for a source known to be clean, such as the project's own synthetic data. Every yielded document carries `meta["pii_redactions"]` and `meta["pii"]` (a count per kind, zeros included), which `data.06` sums into the manifest under the names of `MANIFEST_KEYS` (`manifest_counts` does the renaming).

## 3. Worked example by hand

The text (70 code points):

```
Mail ana@example.com or call (212) 555-0123; card 4111 1111 1111 1111.
0    5              20       29            43     50                 69
```

**Candidates.** email `ana@example.com` at $[5, 20)$: local part `ana`, domain labels `example` and `com` (three letters). phone `(212) 555-0123` at $[29, 43)$: the North American layout with parentheses. card `4111 1111 1111 1111` at $[50, 69)$: 16 digits with spaces. Nothing overlaps.

**Luhn on 4111111111111111.** Number the digits from the right, starting at 1. The even positions (2, 4, ..., 16) are doubled: seven of them are `1` (doubled to 2) and position 16 is the leading `4` (doubled to 8), so $7 \cdot 2 + 8 = 22$. The eight odd positions are all `1`: 8. $L = 22 + 8 = 30$, a multiple of 10: valid, and the first digit 4 is a card network's. The card stands.

**Redact.** Replace right to left or rebuild left to right; either way the result is

```
Mail <EMAIL> or call <PHONE>; card <CARD>.
```

**Counts and audit.** `meta["pii_redactions"] = 3`, `meta["pii"] = {"email": 1, "phone": 1, "card": 1, "ip": 0, "key": 0}`, and the audit line is `{"id": "web:7", "spans": [{"kind": "email", "start": 5, "end": 20}, {"kind": "phone", "start": 29, "end": 43}, {"kind": "card", "start": 50, "end": 69}]}`: offsets into the line above, not into the redacted text, where the phone would start at 21.

## 4. The interface

```python
# python/corpus/pii.py (the full contract is contracts/py/corpus/pii.pyi)
KINDS: tuple[str, ...]                    # ("email", "phone", "card", "ip", "key")
PLACEHOLDERS: dict[str, str]              # kind -> "<EMAIL>", ...
MANIFEST_KEYS: dict[str, str]             # kind -> "emails", ...

@dataclass(frozen=True, slots=True)
class PiiSpan:
    kind: str
    start: int
    end: int

def luhn_ok(digits: str) -> bool
def detect(text: str) -> list[PiiSpan]                     # sorted, non-overlapping
def redact(text: str, spans: Sequence[PiiSpan]) -> str
def scrub(doc: Doc) -> tuple[Doc, list[PiiSpan]]
def audit_line(doc_id: str, spans: Sequence[PiiSpan]) -> str
def scrub_stage(docs, *, policy=None, audit=None) -> Iterator[Doc]   # a Stage
def manifest_counts(counts: Mapping[str, int] | None) -> dict[str, int]  # meta["pii"] under the manifest's names
```

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| The hand example | unit | the section 3 spans, Luhn sum, redacted text, and counts | the definitions are exactly the ones above |
| Luhn | unit | textbook valid and invalid numbers, odd lengths, empty and non-digit input | the validator that keeps cards from being any number |
| A labelled set of 396 sentences | conformance | recall at least 0.98 on email and card, precision at least 0.95 overall, span by span | the design's bar on real-looking text |
| Lookalikes | boundary | ISBNs, versions, dates, times, UUIDs, commit hashes, scopes, slices, C symbols: no span | technical text survives intact |
| Card network prefix | boundary | a Luhn-valid 978... is not a card, the same body behind a 4 is; Amex grouping | one random number in ten passes Luhn |
| Phones | unit | four North American layouts and international forms; 7 digits and 19 digits are not | the layouts people write |
| IPs | unit | IPv4 octet range and dotted runs; valid IPv6 forms; `::1`, slices, and times are not | addresses, not syntax |
| Keys | unit | every key shape whole; short and embedded lookalikes are not | the costliest leak in a code corpus |
| Overlaps | boundary | earliest start wins, then the longer; spans never overlap | `redact` gets a valid span list |
| Original offsets | unit | audit offsets slice the original text | the audit can be checked against the source |
| Bad spans | boundary | unsorted, overlapping, empty, out-of-range spans raise; the tables are as contracted | corruption fails loudly |
| Upstream meta | unit | id, source, `minhash_cluster` and other meta survive; per-kind counts, zeros included | `data.06` sums them |
| Scrubbing twice | property | a second scrub changes nothing and counts 0 | retried activities are safe |
| Policies | unit | scrub, drop, none per source; audit called before the drop decision; unknown policy raises | `ethics.02`'s policy is enforced |
| What the audit holds | unit | the audit line has kinds and offsets, not the address or the card | the log is not a second copy of the PII |
| Streaming | property | pulls one document at a time from an endless stream | the stage runs on any corpus size |

**The model test suite.** [`test_pii_cases.py`](../../../../course/solutions/learner-tests/data.05/python/tests/data-05-pii/test_pii_cases.py) covers the hand example, the Luhn textbook numbers, offsets into the original, lookalikes that must stay, each kind found once, overlapping candidates giving one span, counts and meta that survive a second scrub, `redact`'s checks, and the three policies with the audit hook. Each case pins down one pitfall of section 6; the offsets case is the one that catches pitfall 4.

## 5. The solution, step by step

### 5.1 From first principles

1. **What must never happen is a second copy of the PII.** So the output text holds placeholders, and the audit holds kinds and offsets. Offsets must index the original text, because that is the text a reviewer can compare against; offsets into the redacted text shift after every placeholder of a different length.
2. **A regex alone has the wrong error profile.** Loose enough for recall, it flags versions, ISBNs, and timestamps; strict enough for precision, it misses real formats. So each kind is a permissive candidate regex plus a validator that encodes what makes the thing real: Luhn and a network prefix for cards, an address parser for IPv6, the E.164 digit count for international phones.
3. **Boundaries matter as much as bodies.** Every regex has lookbehind and lookahead guards (`(?<![\w-])`, `(?![\w-])`, `(?!\w|\.\d)`) so a match is never the middle of a longer word, version, or dotted run.
4. **Detectors are independent, so their matches overlap, and `redact` needs a non-overlapping sorted list.** A single greedy sweep, earliest start first, then longer, then a fixed kind order, keeps each candidate that starts after the last kept one ends. The tie-break order makes the result deterministic.
5. **A retried activity must be harmless.** No placeholder matches any detector (`<EMAIL>` has no `@`), so `scrub(scrub(d))` equals `scrub(d)` with zero redactions.
6. **The policy belongs to the source, not the document.** `pii_policy` comes from the ledger, keyed by `source_id`; the stage looks it up per document and defaults to `scrub`. The audit hook is called before the drop decision, so dropped documents are also accounted for.
7. **Every document leaves with the same meta keys.** Counts per kind including zeros make the manifest sum a plain addition.

### 5.2 The code, annotated

The detectors ([`pii.py`](../../../../course/solutions/python/corpus/pii.py)). Each guard is there to reject a specific lookalike:

```python
_EMAIL = re.compile(
    # last label of 2+ letters: react@18.3.1 and root@localhost are not addresses
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}(?![\w-])"
)
_CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")   # 13-19 digits, single separators
_IPV4 = re.compile(
    # octets 0-255; not preceded by a word char or dot, not followed by ".digit": no versions
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\w|\.\d)"
)
_KEY = re.compile(
    r"(?<![\w-])(?:"
    r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{36}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|AIza[0-9A-Za-z_-]{35}"
    r"|tl_[a-z0-9]{4,32}_[A-Za-z0-9]{24,}"   # the gateway's own key format
    r")(?![\w-])"
)
```

Luhn, from the right:

```python
def luhn_ok(digits: str) -> bool:
    if not digits or not digits.isascii() or not digits.isdigit():
        return False                      # isascii: "١٢٣" (Arabic-Indic) is isdigit() but not a card
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:                    # every second digit counting from the right
            d *= 2
            if d > 9:
                d -= 9                    # same as summing the two digits of d
        total += d
    return total % 10 == 0
```

Candidates are validated before they compete:

```python
    for m in _PHONE.finditer(text):
        s = m.group()
        n = sum(c.isdigit() for c in s)
        if s.startswith("+") and not 8 <= n <= 15:   # E.164: at most 15 digits
            continue
        out.append((m.start(), m.end(), "phone"))
    for m in _CARD.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        # network prefix 2-6 AND Luhn: ISBN-13 (978...) and ms timestamps (1...) fail the prefix
        if 13 <= len(digits) <= 19 and digits[0] in "23456" and luhn_ok(digits):
            out.append((m.start(), m.end(), "card"))
    # ...
    for m in _IPV6.finditer(text):
        s = m.group()
        groups = [g for g in s.split(":") if g]
        if len(groups) < 2 or sum(len(g) for g in groups) < 4:
            continue                      # "::1" and slices like a[1::2] are not addresses
        try:
            ipaddress.IPv6Address(s)      # the parser is the validator
        except ValueError:
            continue
        out.append((m.start(), m.end(), "ip"))
```

Overlap resolution is one sort and one sweep:

```python
def detect(text: str) -> list[PiiSpan]:
    order = {k: i for i, k in enumerate(KINDS)}
    # earliest start, then longest, then KINDS order: a total, deterministic order
    cands = sorted(_candidates(text), key=lambda c: (c[0], -(c[1] - c[0]), order[c[2]]))
    spans: list[PiiSpan] = []
    end = 0
    for s, e, kind in cands:
        if s >= end:                      # starts after the last kept span ends
            spans.append(PiiSpan(kind, s, e))
            end = e
    return spans
```

`redact` rebuilds left to right and refuses a span list it cannot apply safely, rather than splicing garbage:

```python
def redact(text: str, spans: Sequence[PiiSpan]) -> str:
    parts: list[str] = []
    pos = 0
    for sp in spans:
        if sp.start < pos or sp.end <= sp.start or sp.end > len(text):
            raise ValueError(f"span {sp} is unsorted, overlapping, empty, or outside the text")
        parts.append(text[pos : sp.start])
        parts.append(PLACEHOLDERS[sp.kind])
        pos = sp.end
    parts.append(text[pos:])
    return "".join(parts)
```

Counts are added to the existing meta, never replacing it, and the stage applies the per-source policy:

```python
def _with_counts(doc: Doc, text: str, spans: Sequence[PiiSpan]) -> Doc:
    counts = {k: 0 for k in KINDS}        # zeros included: the manifest sum is plain addition
    for sp in spans:
        counts[sp.kind] += 1
    meta = {**doc.meta, "pii_redactions": len(spans), "pii": counts}   # minhash_cluster, license survive
    return dataclasses.replace(doc, text=text, meta=meta)


def scrub_stage(docs, *, policy=None, audit=None) -> Iterator[Doc]:
    policy = policy or {}
    for doc in docs:
        mode = policy.get(doc.source_id, "scrub")
        if mode == "none":
            yield _with_counts(doc, doc.text, [])      # not scanned, but same meta keys
            continue
        if mode not in ("scrub", "drop"):
            raise ValueError(f"unknown pii_policy {mode!r} for source {doc.source_id!r}")
        spans = detect(doc.text)
        if spans and audit is not None:
            audit(doc.id, spans)                       # id and spans only, never the text
        if mode == "drop":
            if not spans:
                yield _with_counts(doc, doc.text, [])  # clean documents of a drop source stay
            continue
        yield _with_counts(doc, redact(doc.text, spans), spans)
```

`audit_line` serializes exactly `{"id", "spans": [{"kind", "start", "end"}]}`, and `manifest_counts` renames per-kind counts to the manifest's plural keys, raising on an unknown kind.

Full solution: [`pii.py`](../../../../course/solutions/python/corpus/pii.py).

### 5.3 Why this solution

- **Cost.** Six regex scans per document, each linear in its length for these patterns, plus a sort of the candidates ($O(c \log c)$ for $c$ candidates, almost always tiny). The stage is streaming: $O(1)$ documents in memory.
- **A named-entity model** (Presidio with spaCy) would also find names and street addresses, but brings a model dependency, per-language behaviour, and error rates that need their own evaluation. The five machine-findable kinds have crisp definitions and validators; the datasheet states that names are not scrubbed.
- **Deleting matches or replacing with a fixed token** (`[REDACTED]`) loses what kind of thing stood there; typed placeholders keep the text grammatical and informative. Consistent fake values (BigCode's approach) are better still for code but need a mapping that is itself sensitive.
- **Logging the matched text** would make review easier and create a second copy of exactly what the stage exists to remove. Offsets into the original are enough to review against the source, which is access-controlled.
- **Accepted tradeoffs.** Some real phones in unusual layouts and 7-digit local numbers are missed (precision over recall on phones); the card bar is recall 0.98 on the labelled set, not 1.0.

### 5.4 Explain it back

1. Why is Luhn not enough to call a 16-digit number a card?
   <details><summary>Answer</summary>A random number passes Luhn with probability 1/10, so one ISBN-13 or millisecond timestamp in ten would be redacted. Real cards start with a network digit 2 to 6; ISBNs start with 9 and current timestamps with 1, so the prefix check removes them.</details>
2. Why are audit offsets taken in the original text?
   <details><summary>Answer</summary>`<EMAIL>` is 7 code points and `ana@example.com` is 15, so after the first placeholder every later offset in the redacted text is shifted. Offsets into the original point at the right characters and let a reviewer with access to the source verify each decision.</details>
3. Why does `detect` sort by `(start, -length, kind order)` and then keep greedily?
   <details><summary>Answer</summary>`redact` needs sorted, non-overlapping spans. Earliest start first means the sweep never skips past text; longer first means `1.2.3.4@example.com` becomes one email rather than an IP plus a dangling suffix; the kind order breaks exact ties so the same text always gives the same spans.</details>
4. Why is scrubbing idempotent, and why does that matter here?
   <details><summary>Answer</summary>No placeholder can match a detector (a placeholder holds no `@`, no digits, and no key prefix), so a second scrub finds nothing and changes nothing. The durable workflow in `data.09` may retry this activity on a partly processed batch, and a retry must not change the result.</details>
5. Why is the audit hook called before the `drop` decision?
   <details><summary>Answer</summary>Dropped documents contained PII too, and the account of what was found must include them; otherwise a `drop` source would look clean in the audit and the datasheet's counts would under-report.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a regex with no validator | version strings, IPv4-looking runs, and times become `<IP>`; npm specs become `<EMAIL>` | boundary guards, octet ranges, a 2-letter TLD, and `ipaddress.IPv6Address` |
| 2. Luhn alone for cards | ISBN-13s and timestamps that pass Luhn become `<CARD>` | requires `digits[0] in "23456"` as well |
| 3. keeping every candidate | two placeholders for one region; `redact` raises or splices | `detect` sorts and keeps non-overlapping spans greedily |
| 4. offsets into the redacted text | the audit points at the wrong characters after the first placeholder | spans come from `detect(doc.text)` on the original |
| 5. ignoring the source's policy | a `drop` source's documents are kept, a `none` source is scanned and changed | `policy.get(doc.source_id, "scrub")` per document |
| 6. handing the text to the audit hook | the audit log holds the PII it was meant to account for | `audit(doc.id, spans)`: id and spans only |
| 7. doubling from the left, or not subtracting 9 | half of the real cards fail Luhn | iterates `reversed(digits)` and subtracts 9 above 9 |
| 8. replacing `meta` instead of adding to it | `minhash_cluster` and the license vanish before the shards | `{**doc.meta, ...}` in `_with_counts` |

## 7. Where it's used next


| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.02` | `Doc`, and the stage shape `scrub_stage` follows |
| Back | `data.04` | the scrub runs on the near-dedup survivors and keeps their cluster tags |
| Back | `ethics.02` | the policy: which kinds, which placeholders, which sources scrub, drop, or skip |
| Forward | `data.06` | `pii_redactions` becomes a shard column; `meta["pii"]` sums into the manifest's `pii` object |
| Forward | `data.08` | the datasheet's Composition section reports the counts by kind |
| Forward | `gw.08` | the gateway redacts request logs with a Go port of these detectors (a port, not a call) |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `detect` | Microsoft Presidio | recognizer registry, context words that raise confidence, named-entity models for names and addresses, per-language recognizers | `presidio-analyzer/presidio_analyzer/predefined_recognizers/` |
| `scrub_stage` | datatrove `PIIFormatter`, BigCode's PII pipeline | corpus-scale replacement with consistent fake values, a trained key detector for code | datatrove `src/datatrove/pipeline/formatters/pii.py`; `bigcode-project/bigcode-dataset/pii` |
| key shapes | GitHub secret scanning, gitleaks | hundreds of provider patterns with entropy checks and verification against the provider | `gitleaks/config/gitleaks.toml` |
| `audit_line` | data protection impact assessments | the review process the audit spans feed: sampling, appeal, retention limits | `ethics.02` going further |
