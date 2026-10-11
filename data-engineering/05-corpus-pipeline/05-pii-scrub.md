<!-- ol:module data.05 -->
# PII scrub with typed placeholders and audit spans

## Overview

| | |
|---|---|
| **Module** | `data.05` · build · Python · Pass 3 · 3 to 4 h |
| **You build** | `python/corpus/pii.py`: `luhn_ok`, `detect`, `redact`, `scrub`, `audit_line`, `scrub_stage` (a stage), `manifest_counts`, the `PiiSpan` record, and the tables `KINDS`, `PLACEHOLDERS`, `MANIFEST_KEYS` |
| **Contract** | [`course/contracts/py/corpus/pii.pyi`](../../course/contracts/py/corpus/pii.pyi) · the counts it feeds: [`formats/corpus-manifest.schema.json`](../../course/contracts/formats/corpus-manifest.schema.json) · the policy field: [`formats/ledger.schema.json`](../../course/contracts/formats/ledger.schema.json) (`pii_policy`) |
| **Tests** | `course/tests/data.05/` (what they check: section 4); fixture `course/fixtures/data.05/labelled.jsonl` (396 sentences with labelled spans and lookalikes) |
| **Needs** | `data.02` `Doc` · `data.04` near dedup runs first (or `--ref-deps`) · reading: `ethics.02` (the PII policy this stage enforces) |
| **Used by** | `data.06` sums the per-document counts into the manifest · `data.08` reports them in the datasheet · later: `gw.08` ports the detector list to Go for log redaction |
| **Milestone** | `MS-corpus` |
| **Optional depth** | Microsoft [Presidio](https://microsoft.github.io/presidio/) (analyzers and recognizers, free); Subramani et al., [*Detecting Personal Information in Training Corpora*](https://arxiv.org/abs/2307.03980) (free); ISO/IEC 7812-1 (card numbers and the Luhn check); RFC 5952 (IPv6 text form) |

## Key Takeaways

- A detector is a candidate regex plus a validator: the regex finds what looks right, the validator (Luhn plus a network prefix for cards, an address parser for IPv6) throws out lookalikes (`test_lookalikes_are_not_pii`, `test_card_needs_a_network_prefix`).
- Typed placeholders (`<EMAIL>`, `<CARD>`) keep the shape of the text, so the model still learns that an address goes there without learning anyone's address (`test_hand_example`).
- The audit trail is the spans (kind and offsets into the original text), never the matched text: an audit log that copies the PII is a second copy of it (`test_audit_holds_spans_not_text`, `test_offsets_are_in_the_original_text`).
- The policy is per source, from the ledger: scrub, drop the document, or skip detection for a source known to be clean (`test_policies`).

## How to work this chapter

```bash
ol start data.05              # stubs pii.py into python/corpus/
ol tests data.05              # the course test catalog
ol tdd red data.05            # rung R3: your tests first, failing against the stubs
ol check data.05              # exit code is the verdict
ol check data.05 --ref-deps   # only if data.02 or data.04 is not passing yet
ol mutate data.05             # how many planted bugs your tests catch
ol diff  data.05              # after passing: your code against the reference
```

---

## 1. Why now

After `data.04` your corpus is deduplicated and decontaminated, and it is about to be written to shards that a model will read thousands of times. Some documents still hold an email address, a phone number, a card number pasted into a forum post, a server's IP address, or an API key someone committed by mistake. A language model memorizes rare strings that appear verbatim, and a generation can repeat them to anyone who asks the right prefix. Your engine will serve that model through your gateway, so a key in the training data is a key in your API's output. This module replaces each match with a typed placeholder before the text is sharded, counts what it replaced per document and per kind for the manifest and the datasheet, and records where, without recording what.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T$ | a document's text, indexed by code point | `str` |
| $[s, e)$ | a span: the code points $s$ to $e - 1$ of $T$ | `PiiSpan(kind, start, end)` |
| kind | one of email, phone, card, ip, key | `str` |
| $d_1 \dots d_m$ | the decimal digits of a number, $d_m$ the rightmost | digits |
| $L$ | the Luhn sum (below) | `int` |

**PII here means five machine-findable kinds.** Email addresses, phone numbers, payment card numbers, IP addresses, and API keys. Names and street addresses are personal data too, but finding them needs a named-entity model with its own error rates; the datasheet (`data.08`) says plainly that they are not scrubbed. `ethics.02` is where you decide whether that is acceptable for your corpus.

**Candidate, then validate.** A regex alone is either too loose (every 16-digit number is a card) or too strict (misses real formats). Each detector here is a permissive regex followed by a check that the match is real:

- **email**: a local part of `[A-Za-z0-9._%+-]`, `@`, dot-separated labels, and a last label of at least two letters, so `react@18.3.1` (an npm version spec) is not an address and `root@localhost` (no top-level domain) is not either.
- **phone**: North American layouts `(212) 555-0123`, `212-555-0123`, `212.555.0123`, `+1 212 555 0123`, or `+` and 8 to 15 digits in groups (the E.164 limit). A bare 7-digit `555-0123` is too ambiguous to call a phone.
- **card**: 13 to 19 digits, single spaces or dashes allowed between them, **first digit 2 to 6** (the card networks' ranges), and a valid Luhn checksum.
- **ip**: an IPv4 dotted quad with every octet at most 255, not part of a longer dotted run (`1.2.3.4.5`, `v1.2.3.4` are versions); an IPv6 candidate that `ipaddress.IPv6Address` accepts, with at least two non-empty groups and four hex digits (so the Python slice `a[1::2]` and the loopback `::1`, which is not personal, stay).
- **key**: provider shapes (`sk-...`, `AKIA` + 16, `ghp_` + 36, `xoxb-...`, `AIza` + 35) and your gateway's own `tl_<id>_<secret>` (D19), never inside a longer word.

**The Luhn check.** Card numbers carry a check digit. Starting from the rightmost digit, double every second digit ($d_{m-1}, d_{m-3}, \dots$); when a doubled digit exceeds 9, subtract 9 (the same as adding its two digits); sum everything into $L$. The number is valid when $L \bmod 10 = 0$. A random number passes with probability $1/10$, which is why the network prefix matters: an ISBN-13 starts with 978 or 979 and a millisecond timestamp with 1, and about one in ten of them passes Luhn.

**Overlaps resolve left to right.** Detectors run independently, so matches overlap: `1.2.3.4@example.com` is an IPv4 and an email, `+4222222222222` is a phone and a card. Sort candidates by start, then longer first, then by the order of `KINDS`, and keep each candidate that starts at or after the end of the last kept one. The result is sorted and non-overlapping, which `redact` requires.

**Typed placeholders.** `redact` replaces each span by `<EMAIL>`, `<PHONE>`, `<CARD>`, `<IP>`, or `<KEY>`. A placeholder keeps the sentence grammatical and tells the model what kind of thing stood there. No placeholder matches any detector, so scrubbing twice changes nothing: a retried activity (`data.09`) is safe.

**Audit spans.** `scrub(doc)` returns the new document and the spans, with offsets into the **original** text. Offsets into the redacted text would shift after every placeholder of a different length and point at the wrong characters. `audit_line(id, spans)` writes `{"id", "spans": [{"kind", "start", "end"}]}`: enough to count, sample, and review a decision against the source later, and nothing that is itself personal data.

**Per-source policy.** The ledger row of every source carries `pii_policy` (`ethics.02` fills it): `scrub` (default) redacts; `drop` removes any document that has a span (for sources where PII signals something you do not want, such as a scraped contact page), keeping clean ones; `none` skips detection for a source known to be clean, such as your own synthetic data. Every yielded document carries `meta["pii_redactions"]` and `meta["pii"]` (a count per kind, zeros included), which `data.06` sums into the manifest under the names of `MANIFEST_KEYS` (`manifest_counts` does the renaming).

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

This is `test_hand_example`; the same text drives `test_audit_holds_spans_not_text`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | the section 3 spans, Luhn sum, redacted text, and counts | you and the tests agree on the definitions |
| `test_luhn` | unit | textbook valid and invalid numbers, odd lengths, empty and non-digit input | the validator that keeps cards from being any number |
| `test_labelled_fixture` | conformance | recall at least 0.98 on email and card, precision at least 0.95 overall, span by span | the design's bar on a labelled set |
| `test_lookalikes_are_not_pii` | boundary | ISBNs, versions, dates, times, UUIDs, commit hashes, scopes, slices, C symbols: no span | technical text survives intact |
| `test_card_needs_a_network_prefix` | boundary | a Luhn-valid 978... is not a card, the same body behind a 4 is; Amex grouping | one random number in ten passes Luhn |
| `test_phones` | unit | four North American layouts and international forms; 7 digits and 19 digits are not | the layouts people write |
| `test_ips` | unit | IPv4 octet range and dotted runs; valid IPv6 forms; `::1`, slices, and times are not | addresses, not syntax |
| `test_keys` | unit | every key shape whole; short and embedded lookalikes are not | the costliest leak in a code corpus |
| `test_overlaps_resolve_left_to_right` | boundary | earliest start wins, then the longer; spans never overlap | `redact` gets a valid span list |
| `test_offsets_are_in_the_original_text` | unit | audit offsets slice the original text | the audit can be checked against the source |
| `test_redact_rejects_bad_spans` | boundary | unsorted, overlapping, empty, out-of-range spans raise; the tables are as contracted | corruption fails loudly |
| `test_scrub_keeps_the_document_and_upstream_meta` | unit | id, source, `minhash_cluster` and other meta survive; per-kind counts, zeros included | `data.06` sums them |
| `test_scrub_is_idempotent` | property | a second scrub changes nothing and counts 0 | retried activities are safe |
| `test_policies` | unit | scrub, drop, none per source; audit called before the drop decision; unknown policy raises | `ethics.02`'s policy is enforced |
| `test_audit_holds_spans_not_text` | unit | the audit line has kinds and offsets, not the address or the card | the log is not a second copy of the PII |
| `test_stage_streams` | property | pulls one document at a time from an endless stream | the stage runs on any corpus size |

**Your tests (rung R3, red then green).** Write them under `python/tests/data-05-pii/` against the stubs first (`ol tdd red data.05` must fail), then make them pass: the hand example, the Luhn textbook numbers, offsets into the original, lookalikes that must stay, each kind found once, overlapping candidates giving one span, counts and meta that survive a second scrub, `redact`'s checks, and the three policies with the audit hook. `ol mutate data.05` grades them: 0.70 of the mutants, including the one behind Pitfall 4.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a regex with no validator | version strings, IPv4-looking runs, and times become `<IP>`; npm specs become `<EMAIL>` | `test_lookalikes_are_not_pii`, `test_ips` (mutants `s04`, `s05`, `s07`) |
| 2. Luhn alone for cards | ISBN-13s and timestamps that pass Luhn become `<CARD>` | `test_card_needs_a_network_prefix` (mutant `s03`) |
| 3. keeping every candidate | two placeholders for one region; `redact` raises or splices | `test_overlaps_resolve_left_to_right` (mutant `s09`) |
| 4. offsets into the redacted text | the audit points at the wrong characters after the first placeholder | `test_offsets_are_in_the_original_text` (mutant `s10`) |
| 5. ignoring the source's policy | a `drop` source's documents are kept, a `none` source is scanned and changed | `test_policies` (mutants `s12`, `s13`) |
| 6. handing the text to the audit hook | the audit log holds the PII it was meant to account for | `test_policies` (mutant `s14`), `test_audit_holds_spans_not_text` |
| 7. doubling from the left, or not subtracting 9 | half of the real cards fail Luhn | `test_luhn`, `test_hand_example` (mutants `s01`, `s02`) |
| 8. replacing `meta` instead of adding to it | `minhash_cluster` and the license vanish before the shards | `test_scrub_keeps_the_document_and_upstream_meta` (mutant `s11`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.02` | `Doc`, and the stage shape `scrub_stage` follows |
| Back | `data.04` | the scrub runs on the near-dedup survivors and keeps their cluster tags |
| Back | `ethics.02` | the policy: which kinds, which placeholders, which sources scrub, drop, or skip |
| Forward | `data.06` | `pii_redactions` becomes a shard column; `meta["pii"]` sums into the manifest's `pii` object |
| Forward | `data.08` | the datasheet's Composition section reports the counts by kind |
| Forward | `gw.08` | the gateway redacts request logs with a Go port of these detectors (a port, not a call) |

If you skip this module, `ol check data.06` stops with `data.06 needs data.05`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `detect` | Microsoft Presidio | recognizer registry, context words that raise confidence, named-entity models for names and addresses, per-language recognizers | `presidio-analyzer/presidio_analyzer/predefined_recognizers/` |
| `scrub_stage` | datatrove `PIIFormatter`, BigCode's PII pipeline | corpus-scale replacement with consistent fake values, a trained key detector for code | datatrove `src/datatrove/pipeline/formatters/pii.py`; `bigcode-project/bigcode-dataset/pii` |
| key shapes | GitHub secret scanning, gitleaks | hundreds of provider patterns with entropy checks and verification against the provider | `gitleaks/config/gitleaks.toml` |
| `audit_line` | data protection impact assessments | the review process the audit spans feed: sampling, appeal, retention limits | `ethics.02` going further |
