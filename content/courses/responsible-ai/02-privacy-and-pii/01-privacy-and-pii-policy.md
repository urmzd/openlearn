# Privacy and PII policy

The fetcher in [Async fetch with resume, checksums, and license capture](../../data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md) is downloading text written by strangers, and some of it contains their email addresses, phone numbers, and, in copied configuration files, live API keys. The scrub in [PII scrub with typed placeholders and audit spans](../../data-engineering/05-corpus-pipeline/05-pii-scrub.md) removes them, and later the gateway starts receiving prompts that contain its users' personal data and writing logs about them. Language models memorize, and rare strings seen more than once (an email in a mirrored page, a key in a copied config file) are the easiest to extract. Carlini et al. extracted hundreds of verbatim training sequences from GPT-2, among them names, phone numbers, and email addresses, simply by sampling and ranking. What enters the corpus can come out of the model, so a scrub that runs after training is too late.

Before any detector is written, then, the system needs a written answer to five questions: which kinds of personal data do we look for, what do we replace them with, what do we keep as evidence that we did, how long do logs live, and how does a person get their data removed. Privacy in an ML system is a data-flow property: personal data enters through the corpus and through user requests, and leaves through model outputs and logs. A policy names each flow and its control; the pipeline and the gateway enforce it; tests on labelled fixtures show it works.

This section writes that policy. It follows the shape of [Data licensing and the ledger](../01-data-licensing/01-data-licensing-and-the-ledger.md), a policy as prose plus checkable decisions, and like that section it is engineering, not legal advice: the solution makes sure the decisions are written down, consistent with what the pipeline does, and testable, not that they satisfy a particular law. The solution is three files: the reference policy [`course/solutions/docs/data/PII_POLICY.md`](../../../../course/solutions/docs/data/PII_POLICY.md) with eight fixed sections, the kinds, placeholders, audit, and log rules a program checks in [`course/solutions/docs/data/pii-policy.toml`](../../../../course/solutions/docs/data/pii-policy.toml), and the answers to twelve review cases in [`course/solutions/docs/data/pii-review.toml`](../../../../course/solutions/docs/data/pii-review.toml). The kinds and placeholders are the scrub's own, from the contract [`contracts/py/corpus/pii.pyi`](../../../../course/contracts/py/corpus/pii.pyi) (`KINDS`, `PLACEHOLDERS`); the per-source `pii_policy` is a field of [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json).

> **Learning objectives**
> - List every flow by which personal data enters and leaves an LLM system, and name the control on each.
> - Choose between scrubbing, dropping, and skipping detection for a source, and justify the default.
> - Compute a detector's precision and recall, and apply the Luhn check by hand.
> - Classify lookalike strings (ISBNs, version strings, trace ids) by a structural reason, not a judgment.
> - Write an audit, log-retention, and erasure policy that never creates a second copy of the data.

## Personal data and secrets

**Personal data** is any information about an identified or identifiable person: a direct identifier (a name with an address, an email, a phone number, a card number) or an indirect one that can be tied to a person with other data (an IP address, which an internet provider can map to a subscriber). **Secrets** are not personal but are as dangerous: an API key grants whoever reads it the access of its owner.

The system detects five kinds, the ones the scrub implements (`KINDS` in `contracts/py/corpus/pii.pyi`). Each has a match rule and a placeholder:

| Kind | What the scrub matches | Placeholder |
|---|---|---|
| `email` | `local@domain.tld` | `<EMAIL>` |
| `phone` | North American numbers, and `+` followed by 8 to 15 digits | `<PHONE>` |
| `card` | 13 to 19 digits, first digit 2 to 6, passing the Luhn check | `<CARD>` |
| `ip` | IPv4 dotted quads with octets 0 to 255, and IPv6 addresses | `<IP>` |
| `key` | known API-key shapes: `sk-...`, `AKIA` + 16, `ghp_...`, `xox?-...`, `AIza...`, and the system's own `tl_<id>_<secret>` | `<KEY>` |

Names are personal data too, but no rule-based detector finds them reliably; the policy says so instead of pretending.

## Where personal data enters and leaves

Knowing what to look for is half the problem. The other half is where to look, because personal data is only dangerous where it goes. The policy names every flow and the control on it:

| Flow | Direction | Control |
|---|---|---|
| the corpus (fetched sources) | in | the scrub before training |
| prompts and uploaded documents | in | not logged; the gateway redacts what it does log |
| evaluation suites | in | written by the project; no real personal data |
| generations | out | the scrub upstream; memorization is reduced by dedup |
| logs | out | redaction with the same kinds, and a retention period |
| traces | out | lengths and ids only, never prompt or completion text |
| evaluation reports | out | quote model outputs: the same rules as logs |

A flow the policy does not name is a flow nobody controls. The generations row deserves a note: duplicated sequences are memorized far more often, which is one more reason deduplication ([Exact dedup](../../data-engineering/05-corpus-pipeline/03-exact-dedup-bloom-and-sort-merge.md) and [near-duplicate dedup](../../data-engineering/05-corpus-pipeline/04-near-dedup-minhash-lsh.md)) runs before training. The gateway's redaction is built in [Usage policy enforcement](../../ai-platform-engineering/12-gateway/08-usage-policy-enforcement.md) in *AI Platform Engineering*, and it reuses the same detector list, so the same policy covers training data and traffic.

## What to do with a source

For the corpus, the control is decided per source. The ledger's `pii_policy` field (`formats/ledger.schema.json`) tells the scrub what to do with each one:

- `scrub`: replace every detected value by its placeholder and keep the document. "Write to <EMAIL> for the slides." still teaches the model how such a sentence goes.
- `drop`: remove every document with a detection. For a source dominated by personal data, a forum dump for example, the documents are not worth keeping.
- `none`: skip detection. Only for a named source someone checked and found free of personal data, such as the project's own synthetic fixtures; it is a claim, so it is never the default.

Scrubbing replaces a value with a **typed placeholder** (`<EMAIL>`, `<CARD>`): the text keeps its shape for training and loses the value. **Data minimization** comes before all three options: the cheapest personal data to protect is the data never fetched or logged. What you never collect or keep cannot leak or need erasing.

## Detectors are classifiers

Once a detector is replacing strings, it is making decisions, and decisions can be wrong in two directions. A detector decides, for each candidate string, "personal data" or not. Compare its decisions with labels:

| Symbol | Meaning | Type |
|---|---|---|
| $TP$ | true positives: personal data the detector flagged | count |
| $FP$ | false positives: harmless strings it flagged (an ISBN replaced by `<CARD>`) | count |
| $FN$ | false negatives: personal data it missed | count |
| precision $= TP / (TP + FP)$ | of what was flagged, the share that was personal data | in $[0, 1]$ |
| recall $= TP / (TP + FN)$ | of the personal data, the share that was flagged | in $[0, 1]$ |

Recall protects people; precision protects the corpus. Every false positive destroys real text, and version strings, dates, and identifiers are common in technical writing. The lookalikes (an ISBN, a number that fails the Luhn check, a trace id) decide a detector's precision. The scrub is held to both measures: recall of at least 0.98 on emails and cards, precision of at least 0.95, and no false positives on a list of lookalikes.

## The Luhn check

The single most useful precision tool for cards is a checksum. Every payment card number ends in a check digit chosen so that the **Luhn sum** is a multiple of 10. With the digits $d_1 d_2 \dots d_L$ read **from the right** ($d_1$ is the check digit), define:

| Symbol | Meaning | Type |
|---|---|---|
| $d_i$ | the $i$-th digit from the right | integer in $0..9$ |
| $e_i$ | $d_i$ when $i$ is odd; $2 d_i$ when $i$ is even, minus 9 if that exceeds 9 | integer in $0..9$ |
| $S = \sum_i e_i$ | the Luhn sum | integer |

The number passes when $S \bmod 10 = 0$. A random 16-digit number passes with probability 1/10, so the check alone removes 90% of the false positives that "any 16 digits" produces.

## Audit, logs, and erasure

Detection and replacement are not the end. You need evidence that the scrub happened, rules for the logs, and a path for a person who wants out.

An **audit span** records one replacement: the document id, the kind, and the start and end offsets in the original text (`PiiSpan` in the scrub). It never records the value: an audit log full of values is a second copy of the PII, in a file nobody thought to protect. Logs pass through the same detector, and they expire after a fixed number of days; a log kept forever becomes a corpus. An **erasure request** (GDPR article 17 is one legal basis) asks for a person's data to be removed. The policy says how it is found (search raw parts and shards for the identifiers they give), what is rebuilt (shards, token streams), and what happens to models trained on it.

## Working it by hand

Three exercises make the principles concrete: a Luhn check, the precision of a naive detector, and one key-or-not decision.

**Luhn.** Check `79927398713`. From the right, double every second digit:

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $d_i$ | 3 | 1 | 7 | 8 | 9 | 3 | 7 | 2 | 9 | 9 | 7 |
| $e_i$ | 3 | 2 | 7 | 16 - 9 = 7 | 9 | 6 | 7 | 4 | 9 | 18 - 9 = 9 | 7 |

$S = 3 + 2 + 7 + 7 + 9 + 6 + 7 + 4 + 9 + 9 + 7 = 70$, a multiple of 10: the number passes. Change the last digit to 4 and $S = 71$: it fails.

**Precision of a naive detector.** Among the review cases listed in the next section, three strings hold 13 or more digits: the card `4539 1488 0343 6467` (Luhn sum 80), the order number `4539 1488 0343 6468` (Luhn sum 81), and the ISBN `978-0-306-40615-7` (13 digits, Luhn sum 51). A detector that flags "13 to 19 digits" as a card has $TP = 1$, $FP = 2$: precision $1/3$. Adding the Luhn check removes both false positives: precision $1/1$, recall unchanged. (The scrub also requires a first digit from 2 to 6, which rules out every ISBN-13: they start with 978 or 979.)

**A key or not.** `trace_id=4bf92f3577b34da6a3ce929d0e0e4736` is 32 random hex digits, the shape of a W3C trace id: it identifies a request, grants nothing, and appears in every trace the system exports, so flagging it would scrub the system's own observability. `AKIAIOSFODNN7EXAMPLE` is `AKIA` plus 16 upper-case letters and digits, the shape of an AWS access key id: a key. The difference is a known prefix, not the randomness.

## The three files

The solution is three files: prose for people, a TOML file for programs, and a review that pins down the hard cases.

**`docs/data/PII_POLICY.md`** is prose with these eight `##` sections, each a few sentences long, and no template placeholders left:

| Section | Answers |
|---|---|
| `## Scope` | which data, systems, and models the policy covers |
| `## Data flows` | every way personal data enters and leaves; name the corpus, prompts, logs, and traces |
| `## Categories and actions` | the five kinds, and when a source is scrubbed, dropped, or left alone |
| `## Placeholders` | what replaces each kind and why the text keeps its shape |
| `## Audit` | what an audit span records, and that it never records the value |
| `## Logs and retention` | what is logged, what is redacted, how long logs live |
| `## Erasure requests` | how a person gets their data removed, and what is rebuilt |
| `## Owner and review` | who decides, and how often the policy is reviewed |

**`docs/data/pii-policy.toml`** has this format:

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

**`docs/data/pii-review.toml`** classifies twelve strings, one line per case, the value one of `email`, `phone`, `card`, `ip`, `key`, `none`. Try each one yourself before reading the answers below:

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

## Building the policy step by step

The reasoning runs from flows to files, one decision at a time.

**Start from the flows, not the detectors.** Personal data is only dangerous where it goes. List every way in (corpus, prompts, evaluation suites) and every way out (generations, logs, traces, reports), and give each a control. A detector with no flow to guard protects nothing; a flow with no control leaks.

**Fix a small, named vocabulary of kinds.** The scrub, the log redactor, the audit, and the datasheet must all speak about the same categories, so the policy uses exactly the five `KINDS` of the contract, each with the placeholder the scrub writes.

**Replace, do not delete.** A typed placeholder keeps the sentence and its grammar for training and tells a reader that something of a known kind was removed. Placeholders never match a detector, so scrubbing is idempotent.

**Default to scrubbing.** `none` asserts a source is clean, which someone must have checked; it can only be set per source, never as the default. `drop` is for sources dominated by personal data.

**Precision is part of the policy.** Every false positive destroys real text, and technical corpora are full of numbers. The review cases encode the decisions that keep precision high: cards must pass Luhn, versions are not addresses, random hex without a known prefix is not a key.

**Evidence without a second copy.** The audit records which kind was replaced where (document id and offsets), enough to count and to check recall, and never the value.

**Logs are a flow out, so they get the same detector and an expiry.** Every kind is redacted, prompts and completions are not logged at all, and logs are deleted after a fixed retention.

**Plan erasure before the first request.** Say how data is found (search raw parts and shards for the given identifiers), what is rebuilt (shards and token streams), what happens to trained models, and what record is kept (without the personal data).

### The machine-readable policy, annotated

```toml
version = 1
owner = "data owner"
reviewed = 2026-10-09
default_source_policy = "scrub"   # the ledger pii_policy a new source gets; never "none"

[[category]]
name = "email"
placeholder = "<EMAIL>"           # exactly the string data.05 writes
why = "Emails identify a person directly and are easy to extract from a model."

[[category]]
name = "card"
placeholder = "<CARD>"
why = "Payment card numbers are financial data; only Luhn-valid numbers count."

[[category]]
name = "key"
placeholder = "<KEY>"
why = "Credentials grant access; the source is reported so the key gets revoked."
# ... phone (<PHONE>) and ip (<IP>) in the same form: five kinds, once each

[audit]
store_values = false               # the audit log must not be a second copy of the PII
fields = ["doc_id", "kind", "start", "end"]   # where and what kind, never the text

[logs]
redact = ["email", "phone", "card", "ip", "key"]   # every kind, the same list as the scrub
retain_days = 30                   # a log kept forever becomes a corpus
```

Each `why` states the harm specific to the kind, which is also why the actions differ slightly: a key is not only hidden but reported, because an exposed credential must be revoked to stop being dangerous. The audit fields are positional on purpose: `start` and `end` index the original text, so a reviewer with access to the raw part can find the span, while the audit log alone reveals nothing. The full file is [`pii-policy.toml`](../../../../course/solutions/docs/data/pii-policy.toml).

### The review, annotated

```toml
c01 = "email"  # ana.lopez@example.org
c02 = "phone"  # +1 (415) 555-0132
c03 = "card"   # 4539 1488 0343 6467 passes Luhn: digit sum 80
c04 = "none"   # 4539 1488 0343 6468 fails Luhn: digit sum 81
c05 = "none"   # an ISBN-13 (978...), not a card
c06 = "none"   # a version string: three parts, not a dotted quad
c07 = "ip"     # 203.0.113.42 (a documentation range, still an address)
c08 = "ip"     # 2001:db8:85a3::8a2e:370:7334
c09 = "key"    # the AKIA prefix of an AWS access key id
c10 = "none"   # a date and a time
c11 = "none"   # a W3C trace id: random hex with no key prefix, not a secret
c12 = "key"    # a tl_<id>_<secret> gateway key
```

The six `none` answers are the lookalikes, and each has a structural reason rather than a judgment. Cases c03 and c04 differ in one digit and only the Luhn sum separates them (the hand-worked example computes both); c05 starts with 978, which no card does; c06 has three parts where an IPv4 address has four; c10 is a date and a time; c11 is the shape of a trace id the system itself exports. Case c07 is in a range reserved for documentation, but a detector cannot know which addresses are real, so it is an `ip`. Case c12 is the system's own gateway key format, `tl_<id>_<secret>`, which is why that shape is on the key list. The full file is [`pii-review.toml`](../../../../course/solutions/docs/data/pii-review.toml).

### The policy prose, annotated

The reference policy answers the eight sections. These are the decisions worth reading closely:

> **Data flows.** Personal data enters in three ways: through the corpus, through prompts and uploaded documents sent to the gateway, and through evaluation suites. It leaves in four ways: through generations, through logs written by the gateway and engines, through traces exported to the collector, and through evaluation reports that quote model inputs and outputs. The corpus is scrubbed before training; gateway logs are redacted before they are written; traces never carry prompt or completion text, only lengths and ids.

Every flow is named with its control, which is the first step of the reasoning above.

> **Logs and retention.** Logs are kept for 30 days and then deleted. Prompts and completions are not logged at all; request logs carry ids, token counts, and latencies.

This is data minimization in practice: the cheapest prompt to protect is the one never written to disk.

> **Erasure requests.** A person who asks for their data to be removed gets an answer within 30 days. We search the raw parts and shards for the identifiers they give us, remove the matching documents, and rebuild the affected shards and token streams. A model trained on the old shards is retrained or withdrawn at its next release, and the request is recorded with its date and outcome, without the personal data itself.

It has a deadline, a search procedure, a statement of what is rebuilt, what happens to models, and a record that does not itself hold the data. The full policy is [`PII_POLICY.md`](../../../../course/solutions/docs/data/PII_POLICY.md).

### Why this design

Rule-based kinds beat a learned detector here. Regexes plus checksums (Luhn) and known prefixes are auditable, fast, and testable against a fixed list of lookalikes; a trained detector (Presidio's NER, StarPII) finds names, which rules cannot, at the cost of opaque errors. The policy states openly that names are not detected rather than implying coverage it does not have.

Placeholders beat deletion or hashing. Deleting a value breaks sentences; hashing keeps a stable token that can still be linked across documents and, for small spaces like phone numbers, reversed by brute force. A typed placeholder removes the value and its linkability while keeping the text learnable.

Scrubbing before training beats filtering outputs. Extraction attacks recover memorized sequences from the weights, so a filter on generations is a second line of defence, not the first. The scrub plus deduplication reduces what the model can memorize at all.

The design accepts tradeoffs. Rule-based detection misses names and unusual formats (recall below 1), and the Luhn rule accepts about one in ten random digit strings as cards (some false positives remain). A 30-day retention loses old logs that might help a late investigation. Each is chosen because the alternative (more aggressive scrubbing, longer retention) costs more text or more risk than it saves.

## What can go wrong

A correct solution handles each of these cases, and each matters to something downstream:

| Case | Checks | Why it matters downstream |
|---|---|---|
| every policy section present | the eight sections, each with enough words | every question has an answer |
| no placeholders | no `TODO` or `<lower-case placeholder>` outside code | a template promises nothing |
| data flows named | the Data flows section names corpus, prompts, logs, traces | every flow has a control |
| an owner and a review date | `version = 1`, an owner, a past TOML date, no unknown keys | someone maintains it |
| every kind once, with its placeholder | the five kinds, once each, with the scrub's placeholders and a reason | the policy matches what the shards contain |
| new sources scrubbed or dropped | `default_source_policy` is `scrub` or `drop` | "none" is a claim, never a default |
| the audit never stores the value | `store_values = false`; fields include kind, start, end and nothing that could hold the value | the audit log is not a second copy |
| logs redacted and expiring | every kind redacted; retention 1 to 365 days | logs do not become a corpus |
| the review cases classified | the twelve cases, lookalikes included | the precision the scrub is held to |

The pitfalls behind those checks are easy to fall into. A data-flow list that stops at the corpus leaves prompts in logs and text in traces as nobody's job; the reference names three ways in and four ways out, each with its control. Placeholders in the policy that differ from what the scrub writes mean an auditor cannot match the policy to the shards; each `[[category]]` uses the scrub's exact placeholder. Making `none` the default source policy lets every new source skip detection until someone notices; the reference sets `default_source_policy = "scrub"`.

> **Warning:** Logging the matched value "for debugging" turns the audit log into a file holding every email the corpus had. Keep `store_values = false` and positional fields only.

Logs with no expiry grow into an archive of prompts and personal data; the reference redacts every kind, does not log prompts, and sets `retain_days = 30`. Treating any 16 digits as a card replaces order numbers and ISBNs and gives a precision of 1/3 on the review; cards must pass Luhn and start with 2 to 6, which is why c04 and c05 are `none`. Treating any long random hex as a secret scrubs trace ids from the system's own observability data; keys are matched by known prefixes, which is why c11 is `none`.

## Where this leads

The policy has no code of its own; others implement it. Looking back, it has the same shape as [Data licensing and the ledger](../01-data-licensing/01-data-licensing-and-the-ledger.md): policy in prose, decisions in data, a check that runs. Looking forward, [PII scrub with typed placeholders and audit spans](../../data-engineering/05-corpus-pipeline/05-pii-scrub.md) implements the scrub with these kinds and placeholders, the per-source `pii_policy`, and audit spans without values, and is measured on recall, precision, and the lookalikes classified here. The datasheet in [Licensing ledger verification and the datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md) reports the redaction counts per kind. Gateway log redaction in [Usage policy enforcement](../../ai-platform-engineering/12-gateway/08-usage-policy-enforcement.md) ports the detector list to Go. And the model card in [Datasheet and model card](../03-model-and-data-cards/01-datasheet-and-model-card.md) states what personal data the training corpus may still contain.

### Further reading

- Carlini et al., *Extracting Training Data from Large Language Models* (2021, free).
- Lukas et al., *Analyzing Leakage of Personally Identifiable Information in Language Models* (2023, free).
- NIST Privacy Framework 1.0 (free), and GDPR articles 5 and 17 (gdpr-info.eu, free).
- Microsoft Presidio (`microsoft/presidio`): the production equivalent of the detector list, with recognizers per entity using context words, named-entity models for names and places, and configurable anonymizers.
- Li et al., *StarCoder: may the source be with you!* (2023), the PII redaction section, and the BigScience ROOTS pipeline: PII removal at corpus scale, with a trained detector (StarPII) for names, emails, keys, and passwords in code.
- Abadi et al., "Deep Learning with Differential Privacy" (2016): DP-SGD gives provable bounds on what one training example can change; deduplication reduces memorization through fewer repeats.

## Key terms

- **Personal data**: information about an identified or identifiable person, directly (an email) or indirectly (an IP address).
- **Secret**: a credential, such as an API key, that grants its reader the access of its owner.
- **Typed placeholder**: a token such as `<EMAIL>` that replaces a detected value while keeping the text's shape.
- **Precision**: of the strings a detector flagged, the share that were personal data, $TP / (TP + FP)$.
- **Recall**: of the personal data present, the share the detector flagged, $TP / (TP + FN)$.
- **Luhn check**: the checksum every card number satisfies; the Luhn sum must be a multiple of 10.
- **Audit span**: a record of one replacement (document id, kind, offsets) that never holds the value.
- **Data minimization**: not collecting or keeping data you do not need, the cheapest privacy control.
- **Erasure request**: a person's request to have their data removed, with its search, rebuild, and model consequences.

## Summary

- Language models memorize, so what enters the corpus can come out of the model; scrub before training, redact before logging.
- A policy names every flow: personal data enters through the corpus, prompts, and evaluation suites, and leaves through generations, logs, traces, and reports.
- Scrubbing replaces a value with a typed placeholder, so the text keeps its shape for training and loses the value; new sources default to scrubbing.
- A detector is a classifier with a precision and a recall, and lookalikes (ISBNs, Luhn failures, trace ids) decide its precision.
- An audit trail records what was replaced and where, never the value; logs expire, and erasure is planned before the first request.

## Review questions

1. Why does the audit record offsets and a kind instead of the matched value, when the value would make debugging the detector easier?

   <details><summary>Answer</summary>

   An audit log holding every matched value is a second, complete copy of the personal data the scrub removed, usually in a file with weaker protection than the corpus. Offsets into the original text let someone with access to the raw part find each span, and the kind plus counts are enough to report redactions and measure recall on labelled fixtures, without the audit itself leaking anything.

   </details>

2. Cases c03 and c04 differ in one digit. Why does the solution call one a card and the other none, and what does that buy?

   <details><summary>Answer</summary>

   Only the first passes the Luhn check (sum 80 versus 81). A random digit string passes with probability 1/10, so requiring Luhn removes about 90% of the false positives that "13 to 19 digits" produces, at no cost in recall, because every real card number passes. On the review set it raises precision from 1/3 to 1.

   </details>

3. Why is `default_source_policy = "none"` never acceptable, even for a project with only clean sources today?

   <details><summary>Answer</summary>

   `none` is a claim that someone checked the source and found no personal data. As a default it would apply that claim to every future source nobody has checked, and they would skip detection until someone noticed. Setting `none` per source, with the check that justifies it, keeps the claim attached to evidence.

   </details>

4. Why is the trace id in c11 not a key, while `AKIAIOSFODNN7EXAMPLE` in c09 is?

   <details><summary>Answer</summary>

   Randomness does not make a string a secret; what grants access does. A W3C trace id identifies a request and grants nothing, and the system exports thousands of them, so flagging random hex would scrub its own observability data. `AKIA` followed by 16 characters is the documented shape of an AWS access key id, a credential. The detector keys on known prefixes, not entropy.

   </details>

5. Why must the gateway's log redactor use the same kinds as the corpus scrub?

   <details><summary>Answer</summary>

   Logs are a flow out of the system, and the gateway sees users' prompts, which can contain the same personal data as the corpus. If the two lists differed, a kind redacted in the corpus could appear in logs, and the policy's single category list would no longer describe what the system does. One list, written once in the policy, keeps both implementations and the audit consistent.

   </details>
