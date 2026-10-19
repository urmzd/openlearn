# Privacy and PII policy

Reference artifact for ethics.02. The machine-readable half is `docs/data/pii-policy.toml`; data.05 implements the corpus scrub and gw.08 the log redaction from the same category list.

## Scope

This policy covers personal data and secrets in everything the system stores or emits: the training corpus, the evaluation suites, requests to the gateway and engines, generated text, logs, traces, metrics labels, and the files under `/artifacts`. It applies to every model we train and every model we serve.

## Data flows

Personal data enters in three ways: through the corpus (crawled or downloaded text can contain emails, phone numbers, and keys), through prompts and uploaded documents sent to the gateway, and through evaluation suites. It leaves in four ways: through generations (a model can repeat what it memorized), through logs written by the gateway and engines, through traces exported to the collector, and through evaluation reports that quote model inputs and outputs. The corpus is scrubbed before training; gateway logs are redacted before they are written; traces never carry prompt or completion text, only lengths and ids.

## Categories and actions

We detect the five kinds data.05 implements: email addresses, phone numbers, payment card numbers (only those that pass the Luhn check, so order numbers and ISBNs survive), IP addresses (IPv4 and IPv6), and API keys of known shapes. What happens to a source is its ledger `pii_policy`: a new source is scrubbed (every detected value replaced, the document kept); a source dominated by personal data, such as a forum dump, is set to drop (any document with a detection is removed); only a source someone has checked and found free of personal data, such as our own synthetic fixtures, may be set to none. A leaked key is also reported to the owner of the source it came from, because a key must be revoked, not only hidden.

## Placeholders

Each scrubbed value becomes an upper-case token in angle brackets that names its kind: `<EMAIL>`, `<PHONE>`, `<CARD>`, `<IP>`, and `<KEY>`, exactly what data.05 writes. The placeholder keeps the sentence readable and its shape learnable, and tells a reader of a shard that something was removed and what kind of thing it was. A placeholder never matches a detector, so scrubbing twice changes nothing.

## Audit

Every replacement produces an audit span with the document id, the kind, and the start and end offsets in the original text. The value itself is never stored, not in the span, not in a log line, not in an error message. The spans let us count redactions per shard (the manifest's `pii` counts) and check the detector's recall on labelled fixtures.

## Logs and retention

Gateway and engine logs pass through the same detector list before they are written, so every kind above is redacted in logs as well. Logs are kept for 30 days and then deleted. Prompts and completions are not logged at all; request logs carry ids, token counts, and latencies.

## Erasure requests

A person who asks for their data to be removed gets an answer within 30 days. We search the raw parts and shards for the identifiers they give us, remove the matching documents, and rebuild the affected shards and token streams. A model trained on the old shards is retrained or withdrawn at its next release, and the request is recorded with its date and outcome, without the personal data itself.

## Owner and review

The data owner of this system maintains this policy and reviews it every quarter and whenever a new source, a new log, or a new kind of personal data is added.
