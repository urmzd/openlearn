# PII scrub with typed placeholders and audit spans

After [near dedup and decontamination](04-near-dedup-minhash-lsh.md) the corpus is about to be written to shards that a model will read thousands of times. Some documents still hold an email address, a phone number, a card number pasted into a forum post, a server's IP address, or an API key someone committed by mistake. A language model memorizes rare strings that appear verbatim, and a generation can repeat them to anyone who asks with the right prefix. The engine will serve that model through the gateway, so a key in the training data is a key in the API's output.

This section builds the stage that replaces each match with a typed placeholder before the text is sharded, counts what it replaced per document and per kind for the manifest and the datasheet, and records where, without recording what. It enforces the policy set out in [Privacy and PII policy](../../ai-engineering/34-privacy-and-pii/01-privacy-and-pii-policy.md) in *AI Engineering*, and it uses the `Doc` type from [the filter stages](02-extract-normalize-and-quality-filters.md).

The solution is [`python/corpus/pii.py`](../../../../course/solutions/python/corpus/pii.py): `luhn_ok`, `detect`, `redact`, `scrub`, `audit_line`, `scrub_stage` (a stage), `manifest_counts`, the `PiiSpan` record, and the tables `KINDS`, `PLACEHOLDERS`, and `MANIFEST_KEYS`. Its contract is [`pii.pyi`](../../../../course/contracts/py/corpus/pii.pyi); the counts it feeds are defined in [`formats/corpus-manifest.schema.json`](../../../../course/contracts/formats/corpus-manifest.schema.json), and the policy field is `pii_policy` in [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json).

> **Learning objectives**
> - Build detectors as a permissive candidate regex plus a validator, and explain which lookalikes each validator rejects.
> - Compute a Luhn checksum by hand and explain why the network prefix check is also needed.
> - Resolve overlapping matches into a sorted, non-overlapping span list and redact with typed placeholders.
> - Record an audit trail of kinds and original offsets that never copies the PII.
> - Apply a per-source policy of scrub, drop, or none.

## What counts as PII here

The notation for this section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T$ | a document's text, indexed by code point | `str` |
| $[s, e)$ | a span: the code points $s$ to $e - 1$ of $T$ | `PiiSpan(kind, start, end)` |
| kind | one of email, phone, card, ip, key | `str` |
| $d_1 \dots d_m$ | the decimal digits of a number, $d_m$ the rightmost | digits |
| $L$ | the Luhn sum (below) | `int` |

PII here means five machine-findable kinds: email addresses, phone numbers, payment card numbers, IP addresses, and API keys. Names and street addresses are personal data too, but finding them needs a named-entity model with its own error rates, and [the datasheet](08-ledger-and-datasheet.md) says plainly that they are not scrubbed. The privacy policy chapter of *AI Engineering* is where the decision is made whether that is acceptable for this corpus.

## Candidate, then validate

A regex alone is either too loose (every 16-digit number is a card) or too strict (it misses real formats). Each detector here is therefore a permissive regex followed by a check that the match is real: the regex finds what looks right, and the validator throws out lookalikes.

An **email** has a local part of `[A-Za-z0-9._%+-]`, `@`, dot-separated labels, and a last label of at least two letters, so `react@18.3.1` (an npm version spec) is not an address and neither is `root@localhost` (no top-level domain). A **phone** uses one of the North American layouts `(212) 555-0123`, `212-555-0123`, `212.555.0123`, `+1 212 555 0123`, or `+` and 8 to 15 digits in groups (the E.164 limit); a bare 7-digit `555-0123` is too ambiguous to call a phone. A **card** is 13 to 19 digits, with single spaces or dashes allowed between them, a first digit from 2 to 6 (the card networks' ranges), and a valid Luhn checksum.

An **ip** is either an IPv4 dotted quad with every octet at most 255 that is not part of a longer dotted run (`1.2.3.4.5` and `v1.2.3.4` are versions), or an IPv6 candidate that `ipaddress.IPv6Address` accepts, with at least two non-empty groups and four hex digits, so the Python slice `a[1::2]` and the loopback `::1`, which is not personal, stay. A **key** matches a provider shape (`sk-...`, `AKIA` + 16, `ghp_` + 36, `xoxb-...`, `AIza` + 35) or the gateway's own `tl_<id>_<secret>` format, never inside a longer word.

### The Luhn check

Card numbers carry a check digit. Starting from the rightmost digit, double every second digit ($d_{m-1}, d_{m-3}, \dots$); when a doubled digit exceeds 9, subtract 9 (the same as adding its two digits); sum everything into $L$. The number is valid when $L \bmod 10 = 0$.

A random number passes with probability $1/10$, which is why the network prefix matters. An ISBN-13 starts with 978 or 979 and a millisecond timestamp with 1, and about one in ten of them passes Luhn.

## Overlaps, placeholders, and audit spans

Detectors run independently, so their matches overlap: `1.2.3.4@example.com` is an IPv4 and an email, and `+4222222222222` is a phone and a card. Overlaps resolve left to right. Sort candidates by start, then longer first, then by the order of `KINDS`, and keep each candidate that starts at or after the end of the last kept one. The result is sorted and non-overlapping, which `redact` requires.

`redact` replaces each span by `<EMAIL>`, `<PHONE>`, `<CARD>`, `<IP>`, or `<KEY>`. A typed placeholder keeps the shape of the text: the sentence stays grammatical and the model still learns that an address goes there, without learning anyone's address. No placeholder matches any detector, so scrubbing twice changes nothing, and a retried activity in [CorpusBuild](09-corpusbuild-as-a-durable-workflow.md) is safe.

`scrub(doc)` returns the new document and the spans, with offsets into the *original* text. Offsets into the redacted text would shift after every placeholder of a different length and point at the wrong characters. `audit_line(id, spans)` writes `{"id", "spans": [{"kind", "start", "end"}]}`: enough to count, sample, and review a decision against the source later, and nothing that is itself personal data. The audit trail is the spans, never the matched text, because an audit log that copies the PII is a second copy of it.

## Per-source policy

The last decision is what to do with a match, and it belongs to the source. The ledger row of every source carries `pii_policy`, filled according to the privacy policy. `scrub` (the default) redacts. `drop` removes any document that has a span, keeping clean ones; it suits sources where PII signals something you do not want, such as a scraped contact page. `none` skips detection for a source known to be clean, such as the project's own synthetic data.

Every yielded document carries `meta["pii_redactions"]` and `meta["pii"]` (a count per kind, zeros included), which [the shard writer](06-parquet-shards-and-manifest.md) sums into the manifest under the names of `MANIFEST_KEYS`; `manifest_counts` does the renaming.

## Working it by hand

Take this text of 70 code points, with offsets marked below it:

```
Mail ana@example.com or call (212) 555-0123; card 4111 1111 1111 1111.
0    5              20       29            43     50                 69
```

**Candidates.** The email `ana@example.com` sits at $[5, 20)$: local part `ana`, domain labels `example` and `com` (three letters). The phone `(212) 555-0123` sits at $[29, 43)$, the North American layout with parentheses. The card `4111 1111 1111 1111` sits at $[50, 69)$: 16 digits with spaces. Nothing overlaps.

**Luhn on 4111111111111111.** Number the digits from the right, starting at 1. The even positions (2, 4, ..., 16) are doubled: seven of them are `1` (doubled to 2) and position 16 is the leading `4` (doubled to 8), so $7 \cdot 2 + 8 = 22$. The eight odd positions are all `1`, giving 8. $L = 22 + 8 = 30$, a multiple of 10, so the number is valid, and the first digit 4 is a card network's. The card stands.

**Redact.** Replace right to left or rebuild left to right; either way the result is

```
Mail <EMAIL> or call <PHONE>; card <CARD>.
```

**Counts and audit.** `meta["pii_redactions"] = 3`, `meta["pii"] = {"email": 1, "phone": 1, "card": 1, "ip": 0, "key": 0}`, and the audit line is `{"id": "web:7", "spans": [{"kind": "email", "start": 5, "end": 20}, {"kind": "phone", "start": 29, "end": 43}, {"kind": "card", "start": 50, "end": 69}]}`. Those are offsets into the original line, not into the redacted text, where the phone would start at 21.

## Designing the interface

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

## Building it step by step

Seven requirements lead to the code.

1. **What must never happen is a second copy of the PII.** So the output text holds placeholders, and the audit holds kinds and offsets. Offsets must index the original text, because that is the text a reviewer can compare against; offsets into the redacted text shift after every placeholder of a different length.
2. **A regex alone has the wrong error profile.** Loose enough for recall, it flags versions, ISBNs, and timestamps; strict enough for precision, it misses real formats. So each kind is a permissive candidate regex plus a validator that encodes what makes the thing real: Luhn and a network prefix for cards, an address parser for IPv6, the E.164 digit count for international phones.
3. **Boundaries matter as much as bodies.** Every regex has lookbehind and lookahead guards (`(?<![\w-])`, `(?![\w-])`, `(?!\w|\.\d)`) so a match is never the middle of a longer word, version, or dotted run.
4. **Detectors are independent, so their matches overlap, and `redact` needs a non-overlapping sorted list.** A single greedy sweep, earliest start first, then longer, then a fixed kind order, keeps each candidate that starts after the last kept one ends. The tie-break order makes the result deterministic.
5. **A retried activity must be harmless.** No placeholder matches any detector (`<EMAIL>` has no `@`), so `scrub(scrub(d))` equals `scrub(d)` with zero redactions.
6. **The policy belongs to the source, not the document.** `pii_policy` comes from the ledger, keyed by `source_id`; the stage looks it up per document and defaults to `scrub`. The audit hook is called before the drop decision, so dropped documents are also accounted for.
7. **Every document leaves with the same meta keys.** Counts per kind including zeros make the manifest sum a plain addition.

### The detectors

In [`pii.py`](../../../../course/solutions/python/corpus/pii.py), each guard is there to reject a specific lookalike:

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

### Luhn, from the right

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

### Validating candidates

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

### Overlap resolution and redaction

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

### Counts and the stage

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

`audit_line` serializes exactly `{"id", "spans": [{"kind", "start", "end"}]}`, and `manifest_counts` renames per-kind counts to the manifest's plural keys, raising on an unknown kind. The full solution is [`pii.py`](../../../../course/solutions/python/corpus/pii.py).

### Why this design

The cost is six regex scans per document, each linear in its length for these patterns, plus a sort of the candidates ($O(c \log c)$ for $c$ candidates, almost always tiny). The stage is streaming: $O(1)$ documents in memory.

A named-entity model (Presidio with spaCy) would also find names and street addresses, but it brings a model dependency, per-language behaviour, and error rates that need their own evaluation. The five machine-findable kinds have crisp definitions and validators, and the datasheet states that names are not scrubbed. Deleting matches or replacing them with a fixed token (`[REDACTED]`) loses what kind of thing stood there; typed placeholders keep the text grammatical and informative. Consistent fake values (BigCode's approach) are better still for code but need a mapping that is itself sensitive. Logging the matched text would make review easier and create a second copy of exactly what the stage exists to remove; offsets into the original are enough to review against the source, which is access-controlled.

Two tradeoffs are accepted: some real phones in unusual layouts and 7-digit local numbers are missed (precision over recall on phones), and the card bar is recall 0.98 on the labelled set, not 1.0.

## What a correct scrub must handle

The cases cover the definitions, real-looking text, and the safety properties:

| Case | What must hold | Why it matters |
|---|---|---|
| The hand example | the spans, Luhn sum, redacted text, and counts above | the definitions are exactly the ones above |
| Luhn | textbook valid and invalid numbers, odd lengths, empty and non-digit input | the validator that keeps cards from being any number |
| A labelled set of 396 sentences | recall at least 0.98 on email and card, precision at least 0.95 overall, span by span | the design's bar on real-looking text |
| Lookalikes | ISBNs, versions, dates, times, UUIDs, commit hashes, scopes, slices, C symbols: no span | technical text survives intact |
| Card network prefix | a Luhn-valid 978... is not a card, the same body behind a 4 is; Amex grouping | one random number in ten passes Luhn |
| Phones | four North American layouts and international forms; 7 digits and 19 digits are not | the layouts people write |
| IPs | IPv4 octet range and dotted runs; valid IPv6 forms; `::1`, slices, and times are not | addresses, not syntax |
| Keys | every key shape whole; short and embedded lookalikes are not | the costliest leak in a code corpus |
| Overlaps | earliest start wins, then the longer; spans never overlap | `redact` gets a valid span list |
| Original offsets | audit offsets slice the original text | the audit can be checked against the source |
| Bad spans | unsorted, overlapping, empty, out-of-range spans raise; the tables are as contracted | corruption fails loudly |
| Upstream meta | id, source, `minhash_cluster` and other meta survive; per-kind counts, zeros included | the shard writer sums them |
| Scrubbing twice | a second scrub changes nothing and counts 0 | retried activities are safe |
| Policies | scrub, drop, none per source; audit called before the drop decision; unknown policy raises | the privacy policy is enforced |
| What the audit holds | kinds and offsets, not the address or the card | the log is not a second copy of the PII |
| Streaming | pulls one document at a time from an endless stream | the stage runs on any corpus size |

The reference suite [`test_pii_cases.py`](../../../../course/solutions/learner-tests/data.05/python/tests/data-05-pii/test_pii_cases.py) covers the hand example, the Luhn textbook numbers, offsets into the original, lookalikes that must stay, each kind found once, overlapping candidates giving one span, counts and meta that survive a second scrub, `redact`'s checks, and the three policies with the audit hook. Each case pins down one of the pitfalls below; the offsets case is the one that catches offsets taken in the redacted text.

## What can go wrong

The detection pitfalls are about lookalikes. A regex with no validator turns version strings, IPv4-looking runs, and times into `<IP>`, and npm specs into `<EMAIL>`; the solution uses boundary guards, octet ranges, a two-letter TLD, and `ipaddress.IPv6Address`. Luhn alone for cards turns ISBN-13s and timestamps that pass Luhn into `<CARD>`; the solution also requires `digits[0] in "23456"`. Doubling from the left, or not subtracting 9, makes half of the real cards fail Luhn; the solution iterates `reversed(digits)` and subtracts 9 above 9. Keeping every candidate produces two placeholders for one region, and `redact` raises or splices; `detect` sorts and keeps non-overlapping spans greedily.

> **Warning:** Two pitfalls defeat the point of the audit. Offsets into the redacted text make the audit point at the wrong characters after the first placeholder; spans must come from `detect(doc.text)` on the original. And handing the text to the audit hook puts the PII into the log that was meant to account for it; the hook receives `audit(doc.id, spans)`, id and spans only.

The last two are about the pipeline around the stage. Ignoring the source's policy keeps a `drop` source's documents and scans and changes a `none` source; the solution reads `policy.get(doc.source_id, "scrub")` per document. Replacing `meta` instead of adding to it makes `minhash_cluster` and the license vanish before the shards; `_with_counts` builds `{**doc.meta, ...}`.

## Where this leads

The scrub follows the stage shape of [the filter stages](02-extract-normalize-and-quality-filters.md), runs on the near-dedup survivors and keeps their cluster tags, and enforces the policy (which kinds, which placeholders, which sources scrub, drop, or skip) from [Privacy and PII policy](../../ai-engineering/34-privacy-and-pii/01-privacy-and-pii-policy.md). Next, [the shard writer](06-parquet-shards-and-manifest.md) turns `pii_redactions` into a shard column and sums `meta["pii"]` into the manifest's `pii` object, and [the datasheet](08-ledger-and-datasheet.md) reports the counts by kind in its Composition section. In *AI Engineering*, [Usage policy enforcement](../../ai-engineering/25-ai-gateway/06-usage-policy-enforcement.md) redacts the gateway's request logs with a Go port of these detectors (a port, not a call).

### Further reading

- Microsoft [Presidio](https://microsoft.github.io/presidio/) (`presidio-analyzer/presidio_analyzer/predefined_recognizers/`): a recognizer registry, context words that raise confidence, named-entity models for names and addresses, and per-language recognizers.
- Subramani et al., [*Detecting Personal Information in Training Corpora*](https://arxiv.org/abs/2307.03980).
- datatrove `PIIFormatter` (`src/datatrove/pipeline/formatters/pii.py`) and BigCode's PII pipeline (`bigcode-project/bigcode-dataset/pii`): corpus-scale replacement with consistent fake values, and a trained key detector for code.
- GitHub secret scanning and gitleaks (`gitleaks/config/gitleaks.toml`): hundreds of provider patterns with entropy checks and verification against the provider.
- Data protection impact assessments: the review process the audit spans feed (sampling, appeal, retention limits); see the further reading of the privacy policy chapter in *AI Engineering*.
- ISO/IEC 7812-1 (card numbers and the Luhn check); RFC 5952 (IPv6 text form).

## Key terms

- **Detector**: a permissive candidate regex plus a validator that rejects lookalikes.
- **Luhn check**: a mod-10 checksum over a card number's digits, doubling every second digit from the right.
- **Typed placeholder**: a token such as `<EMAIL>` that replaces a match and names its kind.
- **Audit span**: a record of kind and start and end offsets into the original text, never the matched text.
- **PII policy**: the per-source rule from the ledger: `scrub`, `drop`, or `none`.

## Summary

- A detector is a candidate regex plus a validator: the regex finds what looks right, and the validator (Luhn plus a network prefix for cards, an address parser for IPv6) throws out lookalikes.
- Typed placeholders (`<EMAIL>`, `<CARD>`) keep the shape of the text, so the model still learns that an address goes there without learning anyone's address.
- The audit trail is the spans (kind and offsets into the original text), never the matched text: an audit log that copies the PII is a second copy of it.
- The policy is per source, from the ledger: scrub, drop the document, or skip detection for a source known to be clean.
- Scrubbing is idempotent, so a retried activity is safe.

## Review questions

1. Why is Luhn not enough to call a 16-digit number a card?
   <details><summary>Answer</summary>A random number passes Luhn with probability 1/10, so one ISBN-13 or millisecond timestamp in ten would be redacted. Real cards start with a network digit 2 to 6; ISBNs start with 9 and current timestamps with 1, so the prefix check removes them.</details>
2. Why are audit offsets taken in the original text?
   <details><summary>Answer</summary>`<EMAIL>` is 7 code points and `ana@example.com` is 15, so after the first placeholder every later offset in the redacted text is shifted. Offsets into the original point at the right characters and let a reviewer with access to the source verify each decision.</details>
3. Why does `detect` sort by `(start, -length, kind order)` and then keep greedily?
   <details><summary>Answer</summary>`redact` needs sorted, non-overlapping spans. Earliest start first means the sweep never skips past text; longer first means `1.2.3.4@example.com` becomes one email rather than an IP plus a dangling suffix; the kind order breaks exact ties so the same text always gives the same spans.</details>
4. Why is scrubbing idempotent, and why does that matter here?
   <details><summary>Answer</summary>No placeholder can match a detector (a placeholder holds no `@`, no digits, and no key prefix), so a second scrub finds nothing and changes nothing. The durable `CorpusBuild` workflow may retry this activity on a partly processed batch, and a retry must not change the result.</details>
5. Why is the audit hook called before the `drop` decision?
   <details><summary>Answer</summary>Dropped documents contained PII too, and the account of what was found must include them; otherwise a `drop` source would look clean in the audit and the datasheet's counts would under-report.</details>
