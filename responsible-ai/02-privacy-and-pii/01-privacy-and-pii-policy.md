<!-- ol:module ethics.02 -->
# Privacy and PII policy

## Overview

| | |
|---|---|
| **Module** | `ethics.02` · practice · docs · Pass 3 · 3 to 5 h |
| **You build** | `docs/data/PII_POLICY.md` (your privacy policy, eight fixed sections), `docs/data/pii-policy.toml` (the kinds, placeholders, audit, and log rules a program checks), and `docs/data/pii-review.toml` (your answers to twelve review cases) |
| **Contract** | the kinds and placeholders are the PII scrub's: [`contracts/py/corpus/pii.pyi`](../../course/contracts/py/corpus/pii.pyi) (`KINDS`, `PLACEHOLDERS`); the per-source `pii_policy` is a field of [`formats/ledger.schema.json`](../../course/contracts/formats/ledger.schema.json); the file formats are in section 4 |
| **Tests** | `course/tests/ethics.02/check` runs `test_pii_policy.py` against your three files (what each test checks: section 4) |
| **Needs** | nothing to call · reading: `ethics.01` (a policy as prose plus checkable decisions) |
| **Used by** | no code call site (a policy): `data.05` implements the scrub it describes and is graded on recall and precision over the same lookalikes, and `gw.08` redacts gateway logs with the same kinds |
| **Milestone** | `MS-P3` |
| **Optional depth** | Carlini et al., *Extracting Training Data from Large Language Models* (2021, free); Lukas et al., *Analyzing Leakage of Personally Identifiable Information in Language Models* (2023, free); NIST Privacy Framework 1.0 (free); GDPR articles 5 and 17 (gdpr-info.eu, free) |

Like `ethics.01`, this is engineering, not legal advice: the course checks that your decisions are written down, consistent with what your pipeline does, and testable, not that they satisfy a particular law.

## Key Takeaways

- Language models memorize, and rare strings seen more than once (an email in a mirrored page, a key in a copied config file) are the easiest to extract. What enters the corpus can come out of the model (`test_data_flows_name_the_ways_in_and_out`).
- A policy names every flow: personal data enters through the corpus, prompts, and evaluation suites, and leaves through generations, logs, traces, and reports.
- Scrubbing replaces a value with a **typed placeholder** (`<EMAIL>`, `<CARD>`): the text keeps its shape for training and loses the value (`test_every_category_is_named_once_with_its_placeholder`).
- A detector is a classifier with a precision and a recall. The lookalikes (an ISBN, a number that fails the Luhn check, a trace id) decide its precision (`test_review_cases_are_classified`).
- An audit trail records what was replaced and where, never the value itself (`test_audit_never_stores_the_value`).

## How to work this chapter

```bash
ol start ethics.02            # records the start; you write docs/data/ from section 4
ol tests ethics.02            # read the test catalog first
ol check ethics.02            # exit code is the verdict
```

---

## 1. Why now

`data.01` is downloading text written by strangers, and some of it contains their email addresses, phone numbers, and, in copied configuration files, live API keys. In this pass `data.05` scrubs them, and in Pass 7 your gateway starts receiving prompts that contain your users' personal data and writing logs about them. Carlini et al. extracted hundreds of verbatim training sequences from GPT-2, among them names, phone numbers, and email addresses, simply by sampling and ranking; a scrub that runs after training is too late. Before any detector is written, the system needs a written answer to: which kinds of personal data do we look for, what do we replace them with, what do we keep as evidence that we did, how long do logs live, and how does a person get their data removed. `data.05` and `gw.08` then implement those answers, and the tests of both are written against the same lookalikes you classify here.

## 2. Principles

### 2.1 Personal data and secrets

**Personal data** is any information about an identified or identifiable person: a direct identifier (a name with an address, an email, a phone number, a card number) or an indirect one that can be tied to a person with other data (an IP address, which an internet provider can map to a subscriber). **Secrets** are not personal but are as dangerous: an API key grants whoever reads it the access of its owner. The course detects five kinds, the ones `data.05` implements (`KINDS` in `contracts/py/corpus/pii.pyi`):

| Kind | What `data.05` matches | Placeholder |
|---|---|---|
| `email` | `local@domain.tld` | `<EMAIL>` |
| `phone` | North American numbers, and `+` followed by 8 to 15 digits | `<PHONE>` |
| `card` | 13 to 19 digits, first digit 2 to 6, passing the Luhn check | `<CARD>` |
| `ip` | IPv4 dotted quads with octets 0 to 255, and IPv6 addresses | `<IP>` |
| `key` | known API-key shapes: `sk-...`, `AKIA` + 16, `ghp_...`, `xox?-...`, `AIza...`, and the course's own `tl_<id>_<secret>` | `<KEY>` |

Names are personal data too, but no rule-based detector finds them reliably; the policy says so instead of pretending.

### 2.2 Where personal data enters and leaves

| Flow | Direction | Control |
|---|---|---|
| the corpus (fetched sources) | in | the scrub before training (`data.05`) |
| prompts and uploaded documents | in | not logged; the gateway redacts what it does log (`gw.08`) |
| evaluation suites | in | written by you; no real personal data |
| generations | out | the scrub upstream; memorization is reduced by dedup (`data.03`, `data.04`) |
| logs | out | redaction with the same kinds, and a retention period |
| traces | out | lengths and ids only, never prompt or completion text |
| evaluation reports | out | quote model outputs: the same rules as logs |

A flow the policy does not name is a flow nobody controls.

### 2.3 What to do with a source

The ledger's `pii_policy` field (`formats/ledger.schema.json`) decides what `data.05` does to each source:

- `scrub`: replace every detected value by its placeholder and keep the document. "Write to <EMAIL> for the slides." still teaches the model how such a sentence goes.
- `drop`: remove every document with a detection. For a source dominated by personal data, a forum dump for example, the documents are not worth keeping.
- `none`: skip detection. Only for a named source someone checked and found free of personal data, such as your own synthetic fixtures; it is a claim, so it is never the default.

**Data minimization** comes before all three: the cheapest personal data to protect is the data you never fetch or log.

### 2.4 Detectors are classifiers

A detector decides, for each candidate string, "personal data" or not. Compare its decisions with labels:

| Symbol | Meaning | Type |
|---|---|---|
| $TP$ | true positives: personal data the detector flagged | count |
| $FP$ | false positives: harmless strings it flagged (an ISBN replaced by `<CARD>`) | count |
| $FN$ | false negatives: personal data it missed | count |
| precision $= TP / (TP + FP)$ | of what was flagged, the share that was personal data | in $[0, 1]$ |
| recall $= TP / (TP + FN)$ | of the personal data, the share that was flagged | in $[0, 1]$ |

Recall protects people; precision protects the corpus (every false positive destroys real text, and version strings, dates, and identifiers are common in technical writing). `data.05` is graded on both: recall of at least 0.98 on emails and cards, precision of at least 0.95, and no false positives on a list of lookalikes.

### 2.5 The Luhn check

Every payment card number ends in a check digit chosen so that the **Luhn sum** is a multiple of 10. With the digits $d_1 d_2 \dots d_L$ read **from the right** ($d_1$ is the check digit):

| Symbol | Meaning | Type |
|---|---|---|
| $d_i$ | the $i$-th digit from the right | integer in $0..9$ |
| $e_i$ | $d_i$ when $i$ is odd; $2 d_i$ when $i$ is even, minus 9 if that exceeds 9 | integer in $0..9$ |
| $S = \sum_i e_i$ | the Luhn sum | integer |

The number passes when $S \bmod 10 = 0$. A random 16-digit number passes with probability 1/10, so the check alone removes 90% of the false positives that "any 16 digits" produces.

### 2.6 Audit, logs, erasure

An **audit span** records one replacement: the document id, the kind, and the start and end offsets in the original text (`PiiSpan` in `data.05`). It never records the value: an audit log full of values is a second copy of the PII, in a file nobody thought to protect. Logs pass through the same detector, and they expire after a fixed number of days; a log kept forever becomes a corpus. An **erasure request** (GDPR article 17 is one legal basis) asks you to remove a person's data: the policy says how you find it (search raw parts and shards for the identifiers they give), what you rebuild (shards, token streams), and what happens to models trained on it.

## 3. Worked example by hand

**Luhn.** Check `79927398713`. From the right, double every second digit:

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $d_i$ | 3 | 1 | 7 | 8 | 9 | 3 | 7 | 2 | 9 | 9 | 7 |
| $e_i$ | 3 | 2 | 7 | 16 - 9 = 7 | 9 | 6 | 7 | 4 | 9 | 18 - 9 = 9 | 7 |

$S = 3 + 2 + 7 + 7 + 9 + 6 + 7 + 4 + 9 + 9 + 7 = 70$, a multiple of 10: the number passes. Change the last digit to 4 and $S = 71$: it fails.

**Precision of a naive detector.** Among the review cases of section 4, three strings hold 13 or more digits: the card `4539 1488 0343 6467` (Luhn sum 80), the order number `4539 1488 0343 6468` (Luhn sum 81), and the ISBN `978-0-306-40615-7` (13 digits, Luhn sum 51). A detector that flags "13 to 19 digits" as a card has $TP = 1$, $FP = 2$: precision $1/3$. Adding the Luhn check removes both false positives: precision $1/1$, recall unchanged. (`data.05` also requires a first digit from 2 to 6, which rules out every ISBN-13: they start with 978 or 979.)

**A key or not.** `trace_id=4bf92f3577b34da6a3ce929d0e0e4736` is 32 random hex digits, the shape of a W3C trace id: it identifies a request, grants nothing, and appears in every trace you will export, so flagging it would scrub your own observability. `AKIAIOSFODNN7EXAMPLE` is `AKIA` plus 16 upper-case letters and digits, the shape of an AWS access key id: a key. The difference is a known prefix, not the randomness.

## 4. The artifact and its check

Write three files in your repo.

**`docs/data/PII_POLICY.md`**: prose with these eight `##` sections (the check counts words, 8 to 30 per section) and no template placeholders left:

| Section | Answers |
|---|---|
| `## Scope` | which data, systems, and models the policy covers |
| `## Data flows` | every way personal data enters and leaves (2.2); name the corpus, prompts, logs, and traces |
| `## Categories and actions` | the five kinds, and when a source is scrubbed, dropped, or left alone |
| `## Placeholders` | what replaces each kind and why the text keeps its shape |
| `## Audit` | what an audit span records, and that it never records the value |
| `## Logs and retention` | what is logged, what is redacted, how long logs live |
| `## Erasure requests` | how a person gets their data removed, and what you rebuild |
| `## Owner and review` | who decides, and how often the policy is reviewed |

**`docs/data/pii-policy.toml`**:

```toml
version = 1
owner = "who decides"
reviewed = 2026-10-09                 # a TOML date, no quotes
default_source_policy = "scrub"       # scrub or drop: the ledger pii_policy of a new source

[[category]]                          # once for each of email, phone, card, ip, key
name = "email"
placeholder = "<EMAIL>"               # exactly data.05's placeholder for the kind
why = "One sentence, at least six words."

[audit]
store_values = false
fields = ["doc_id", "kind", "start", "end"]   # from doc_id, source_id, kind, start, end

[logs]
redact = ["email", "phone", "card", "ip", "key"]
retain_days = 30                      # whole days, 1 to 365
```

**`docs/data/pii-review.toml`**: one line per case, the value one of `email`, `phone`, `card`, `ip`, `key`, `none`:

| Case | String |
|---|---|
| `c01` | `Write to ana.lopez@example.org for the slides.` |
| `c02` | `Call me at +1 (415) 555-0132 after six.` |
| `c03` | `Card on file: 4539 1488 0343 6467` |
| `c04` | `Order number 4539 1488 0343 6468` |
| `c05` | `ISBN 978-0-306-40615-7` |
| `c06` | `Upgraded the engine to v2.10.3 last night.` |
| `c07` | `The request came from 203.0.113.42 at noon.` |
| `c08` | `Peer 2001:db8:85a3::8a2e:370:7334 dropped the stream.` |
| `c09` | `export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE` |
| `c10` | `Meeting moved to 2026-10-09 at 14:30.` |
| `c11` | `trace_id=4bf92f3577b34da6a3ce929d0e0e4736` |
| `c12` | `Authorization: Bearer tl_k7f3_9s8d7f6g5h4j3k2l1m0nq8w7` |

```toml
c01 = "email"
# ... c02 to c12
```

`ol check ethics.02` runs `course/tests/ethics.02/check` from your repo root; it refuses early if a file is missing.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_policy_has_every_section` | unit | the eight sections, each with enough words | every question has an answer |
| `test_policy_has_no_placeholders` | unit | no `TODO` or `<lower-case placeholder>` outside code | a template promises nothing |
| `test_data_flows_name_the_ways_in_and_out` | unit | the Data flows section names corpus, prompts, logs, traces | every flow has a control |
| `test_policy_toml_has_an_owner_and_a_review_date` | unit | `version = 1`, an owner, a past TOML date, no unknown keys | someone maintains it |
| `test_every_category_is_named_once_with_its_placeholder` | unit | the five kinds, once each, with data.05's placeholders and a reason | the policy matches what the shards contain |
| `test_new_sources_are_scrubbed_or_dropped` | boundary | `default_source_policy` is `scrub` or `drop` | "none" is a claim, never a default |
| `test_audit_never_stores_the_value` | boundary | `store_values = false`; fields include kind, start, end and nothing that could hold the value | the audit log is not a second copy |
| `test_logs_redact_every_category_and_expire` | boundary | every kind redacted; retention 1 to 365 days | logs do not become a corpus |
| `test_review_cases_are_classified` | unit | the twelve cases, lookalikes included | the precision data.05 is graded on |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. A data-flow list that stops at the corpus | prompts in logs and text in traces are nobody's job | `test_data_flows_name_the_ways_in_and_out` |
| 2. Placeholders in the policy that differ from what the scrub writes | an auditor cannot match the policy to the shards | `test_every_category_is_named_once_with_its_placeholder` |
| 3. `none` as the default source policy | every new source skips detection until someone notices | `test_new_sources_are_scrubbed_or_dropped` |
| 4. Logging the matched value "for debugging" | the audit log holds every email the corpus had | `test_audit_never_stores_the_value` |
| 5. Logs with no expiry | a growing archive of prompts and personal data | `test_logs_redact_every_category_and_expire` |
| 6. Treating any 16 digits as a card | order numbers and ISBNs replaced; precision of 1/3 on the review | `test_review_cases_are_classified` (cases c04, c05) |
| 7. Treating any long random hex as a secret | trace ids scrubbed from your own observability data | `test_review_cases_are_classified` (case c11) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ethics.01` | the same shape: policy in prose, decisions in data, a check that runs |
| Forward | `data.05` | the scrub with these kinds and placeholders, the per-source `pii_policy`, and audit spans without values; graded on recall, precision, and the lookalikes |
| Forward | `data.08` | the datasheet reports the redaction counts per kind |
| Forward | `gw.08` | gateway log redaction ports the detector list to Go |
| Forward | `ethics.03` | the model card states what personal data the training corpus may still contain |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the detector list | Microsoft Presidio | recognizers per entity with context words, named-entity models for names and places, configurable anonymizers | `microsoft/presidio` |
| the scrub policy | BigScience ROOTS and StarCoder PII pipelines | PII removal at corpus scale, with a trained detector (StarPII) for names, emails, keys, and passwords in code | Li et al., *StarCoder: may the source be with you!* (2023), the PII redaction section |
| memorization | differential privacy (DP-SGD), deduplication | provable bounds on what one training example can change; less memorization from fewer repeats | Abadi et al., "Deep Learning with Differential Privacy" (2016) |
