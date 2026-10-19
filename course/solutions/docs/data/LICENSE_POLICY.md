# Data licensing policy

Reference artifact for ethics.01. It records engineering decisions about provenance and is not legal advice; a change to the allowlist that widens what may be trained on goes to whoever gives this project legal review.

## Scope

This policy covers every source that enters the corpus pipeline: the files fetched by `corpus fetch` from the sources pinned in `specs/corpus/*.toml`, every document derived from them (raw parts, clean shards, token streams), and every model trained or evaluated on those documents. It does not cover third-party models we serve unchanged; their model cards say where they came from.

## Allowed uses

A source is used for one or both of two purposes. Training means its text shapes the weights of a model we release, so its license must allow us to make and share an adaptation. Evaluation means its text is only read to score a model and is never trained on; decontamination (data.04) keeps evaluation text out of training sets. A source may be allowed for evaluation and refused for training, never the other way round.

## Ledger

Every fetched source gets one row in `corpus/LEDGER.jsonl`, written at fetch time by data.01. The row records `source_id` (our name for the source), `url` (where it came from), `license_spdx` (the SPDX identifier from the source's license page, copied when we pinned it), `retrieved_at` (UTC time of the download), `sha256` (of the exact bytes received, so the license is tied to those bytes and not to whatever the URL serves later), `allowed_uses` (from the corpus config), and `pii_policy`. A row is never edited after the fact; a new version of a source is a new row with a new checksum.

## Allowlist

The decisions live in `docs/data/license-allowlist.toml`: `[[allow]]` entries with the uses they permit and the obligations they bring, and `[[refuse]]` entries with the reason. data.08 checks every ledger row against it and fails the build, with exit code 65, when a row's license is refused, missing from the list, or allowed only for a use the row claims beyond it.

## Obligations

Allowed licenses come with duties that travel with the data. Attribution licenses (CC BY, ODC-By) require crediting the source wherever we share the data or a dataset derived from it; the datasheet lists each source and its license for that reason. ShareAlike licenses (CC BY-SA, CDLA-Sharing) require that derived data we share keep the same license. Apache-2.0 and MIT require keeping the copyright and license notice with redistributed copies.

## Unknown and missing licenses

A source with no license, a license we cannot identify, or a license recorded as NOASSERTION is refused for every use. All rights are reserved by default, so the absence of a license is a refusal, not a permission. Such a source stays out of the corpus until someone finds its terms and adds an entry here.

## Revocation

When a source's license is withdrawn, found to be wrong, or a takedown request arrives, the ledger row is marked `revoked: true` and the source moves to `[[refuse]]`. Every shard and token stream derived from it is rebuilt without it, and any model trained on it is withdrawn or retrained before its next release. The data-incident drill (ops.08) rehearses exactly this path.

## Owner and review

The data owner of this system maintains this policy and the allowlist and reviews both every quarter and whenever a new source is pinned.
