# Data licensing and the ledger

Every training corpus is made of text somebody else wrote, and every piece of that text arrived under terms. The fetcher in [Async fetch with resume, checksums, and license capture](../../data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md) is about to download every source in the corpus config, and much later a workflow will train a model on the result and publish it ([ModelRelease: gates, canary, PromQL burn check, rollback](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) in *AI Platform Engineering*). Between those two moments somebody has to be able to answer, for every document in every shard: where did this come from, under what terms, and do those terms let us train on it and release what we trained?

That question cannot be answered after the fact. The Data Provenance Initiative audited over 1,800 popular fine-tuning datasets and found licenses missing for most of them on the hosting sites and miscategorized for a large share, usually as more permissive than the original authors granted. Once text is mixed into a corpus the answer cannot be reconstructed, so the policy has to exist before the first download, and it has to be data a build can check, not a paragraph in a README. Licensing for training data is a provenance problem: each byte of the corpus must trace back to a source, and each source must carry the terms under which it was obtained. Once provenance is data, compliance becomes a check that runs in CI.

This section shows how to make provenance auditable. It is not legal advice: whether a license permits training a model is decided by a lawyer for your situation. What the solution fixes is that every decision is written down, uses names a program can check, and is checked on every build. It needs nothing before it, because policy is written before the pipeline it governs. The result is two files, a prose policy, [`course/solutions/docs/data/LICENSE_POLICY.md`](../../../../course/solutions/docs/data/LICENSE_POLICY.md), with eight fixed sections, and the machine-readable decisions the pipeline enforces, [`course/solutions/docs/data/license-allowlist.toml`](../../../../course/solutions/docs/data/license-allowlist.toml). The ledger row they govern is defined by the contract [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json); the allowlist format is defined later in this section.

> **Learning objectives**
> - Explain why text without a license is not free to use, and place a license in its family by what it permits and what it asks.
> - Name a license by its exact SPDX identifier and read `OR` and `AND` license expressions.
> - Write an allowlist that decides each common license per use (`train`, `eval`) with its obligations, and refuses unknown licenses.
> - Describe what the ledger records for each source and why it binds the license to the sha256 of the fetched bytes.
> - Trace how a verification failure surfaces, and fix it in the right file.

## Why text has a license at all

Copyright gives the author of a text the exclusive right to copy it, adapt it, and distribute it, automatically and without registration. A **license** is the author's written permission to do some of those things under conditions. With no license, nothing is permitted beyond what the law allows anyway (fair use and text-and-data-mining exceptions vary by country and are exactly what you would need a lawyer for). So "no license" means "no", not "free".

Copyright is not the only thing that binds you. A website's **terms of service** can bind you by contract even for text that is not protected. And a dataset aggregated from many sources has one license per source: a dataset's license is a property of every source in it, not of the dataset. The dataset card's single license line is a claim about all of them that someone else made, and aggregators routinely relabel.

## License families

Most licenses you will meet on dataset cards fall into a handful of families, distinguished by what you may do and what the license asks in return:

| Family | Examples (SPDX ids) | You may | The license asks |
|---|---|---|---|
| Public domain dedication | `CC0-1.0`, `PDDL-1.0` | anything | nothing |
| Permissive | `MIT`, `Apache-2.0`, `BSD-3-Clause`, `CDLA-Permissive-2.0` | copy, adapt, redistribute | keep the copyright and license text (**notice**) |
| Attribution | `CC-BY-4.0`, `ODC-By-1.0` | copy, adapt, redistribute | credit the source (**attribution**) |
| ShareAlike | `CC-BY-SA-4.0`, `CDLA-Sharing-1.0`, `ODbL-1.0` | copy, adapt, redistribute | derived data you share keeps the same license (**share-alike**) |
| Non-commercial | `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0` | use for non-commercial purposes | no commercial use |
| No derivatives | `CC-BY-ND-4.0`, `CC-BY-NC-ND-4.0` | copy unchanged | no adaptations |
| None or unknown | `NOASSERTION` (nobody said), `NONE` (no license) | nothing | |

The hard question sits in the "adapt" column. Whether a trained model is an "adaptation" of its training text is unsettled. CDLA-Sharing-1.0 was written for this question and says that results computed from the data, a model among them, are not bound by its share-alike term; the Creative Commons licenses do not say. That is why the allowlist records **uses** separately: a license can be allowed for evaluation (reading text to score a model) and refused for training (shaping the weights of a model you release).

## Naming licenses exactly: SPDX

A program that checks licenses compares strings, so every license needs one spelling. The SPDX License List gives every common license a short, case-sensitive identifier: `CC-BY-4.0`, not "CC BY 4.0" or "cc-by-4.0"; `Apache-2.0`, not "Apache 2.0". A license SPDX does not list is written `LicenseRef-<name>`. The SPDX identifier is the one spelling the ledger, the allowlist, and the datasheet share.

SPDX also defines **expressions** that combine identifiers. `MIT OR Apache-2.0` means you may choose either, so you get the uses of both together. `CC-BY-4.0 AND MIT` means both apply, so you get only the uses both allow. The verifier in [Licensing ledger verification and the datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md) evaluates expressions against the allowlist; an identifier it does not find is refused.

## The rules that are not judgment calls

With families and names in place, you can write the allowlist: a policy in data, where each license is allowed for named uses with its obligations, or refused with a reason. Most of it is a judgment call. Four rules are not, because the rest of the system depends on them:

1. **Unknown is refused.** `NOASSERTION` is refused explicitly, and `NONE` can never be allowed.
2. **No NC or ND in training.** The model's weights are released under a license the project chooses; a non-commercial term forbids commercial use of what is built from the data and a no-derivatives term forbids adaptations, so neither can be in a training set. Evaluation use is a judgment call.
3. **Obligations are recorded.** Every allowed attribution license (`CC-BY*`, `ODC-By`) lists `attribution`; every ShareAlike license (`*-SA-*`, `CDLA-Sharing`) lists `share-alike`. The datasheet in [Datasheet and model card](../03-model-and-data-cards/01-datasheet-and-model-card.md) prints them from here.
4. **The system's own sources are allowed for training.** TinyStories, the corpus of the [first capstone](../../build-an-llm/capstones/01-tinystories.md) in *Build Your Own LLM System*, is `CDLA-Sharing-1.0`, and the repository's test fixtures are `Apache-2.0`. Refusing them is a valid policy, but then that corpus cannot be built.

## The ledger

The allowlist says what each license permits. The ledger says which license each source has. Every fetched source gets one row in `corpus/LEDGER.jsonl` (`formats/ledger.schema.json`), written by the fetcher at the moment the bytes, the URL, and the reviewed license are all in one place. Recording the license when you fetch is cheap; reconstructing it later is nearly impossible. Each field earns its place:

| Field | Records | Why it matters |
|---|---|---|
| `source_id` | the project's name for the source | shard rows point back to it |
| `url` | where the bytes came from | anyone can go and look |
| `license_spdx` | the license reviewed when the source was pinned | what the allowlist is checked against |
| `retrieved_at` | UTC time of the download | the terms in force then |
| `sha256` | digest of the exact bytes received | the license applies to these bytes, not to whatever the URL serves later |
| `allowed_uses` | `train`, `eval` as the config claims | the verifier checks the claim against the allowlist |
| `pii_policy`, `filters_applied`, `kept`, `dropped`, `notes` | what the pipeline did | the datasheet |

Rows are appended, never edited: a new version of a source is a new row with a new checksum, and a withdrawn source gets a row with `revoked: true`. Verification then checks that every shard row traces to a source id; an unknown or non-allowlisted license exits with code 65, which the subprocess activity contract treats as non-retryable, and the release workflow refuses to release a model whose data ledger does not verify.

## Revocation

Decisions can turn out wrong. A license can be misread, an author can ask for removal, a host can change its terms. The policy says in advance what happens: the source is refused from then on, every shard and token stream derived from it is rebuilt without it, and a model trained on it is retrained or withdrawn before its next release. The drill in [Drill: data incident](../../systems/05-incident-response-and-chaos/08-data-incident.md) from *Systems and Architecture* rehearses exactly this.

## Working it by hand

Before reading the reference files, decide a small corpus yourself. This one is different from the system's, with three sources:

| Source | License on the card | What a reviewer finds |
|---|---|---|
| `poems` | "Public Domain" | the site says the poems were dedicated with CC0 |
| `wiki-mini` | "CC BY-SA" | the footer says Creative Commons Attribution-ShareAlike 3.0 |
| `forum` | (none) | a scrape of a forum; the terms of service reserve all rights |

For `poems`, the SPDX identifier is `CC0-1.0`: allowed for `train` and `eval`, with no obligations. For `wiki-mini`, the identifier is `CC-BY-SA-3.0`, which carries attribution and ShareAlike. If the project decides a released model is not an adaptation under this license, it can allow `train` and `eval` with obligations `attribution` and `share-alike`; if unsure, it allows `eval` only and says so in `why`. This example chooses `eval`. For `forum`, there is no license and the terms are restrictive, so the ledger records `NOASSERTION`, which is refused.

The allowlist entries look like this:

```toml
[[allow]]
spdx = "CC0-1.0"
uses = ["train", "eval"]
obligations = []
why = "A public domain dedication: no conditions on use or redistribution."

[[allow]]
spdx = "CC-BY-SA-3.0"
uses = ["eval"]
obligations = ["attribution", "share-alike"]
why = "Evaluation only until we decide whether a released model is a ShareAlike adaptation."

[[refuse]]
spdx = "NOASSERTION"
why = "Nobody recorded a license, so all rights are reserved until someone finds the terms."
```

Now suppose the corpus config claims `allowed_uses = ["train", "eval"]` for `wiki-mini`. The verifier reads its ledger row, finds `CC-BY-SA-3.0` allowed for `eval` only, and fails verification with exit 65: the config claims a use the policy does not grant. The fix is in the config (`allowed_uses = ["eval"]`), not in the allowlist.

## The two files

That example shows the shape of the solution. It is two files, one for people and one for programs.

**`docs/data/LICENSE_POLICY.md`** is prose, with these eight `##` sections. Each must be long enough to answer its question (at least a couple of sentences), and no template placeholders (`TODO`, `<your text>`) may be left:

| Section | Answers |
|---|---|
| `## Scope` | which data and which models the policy covers |
| `## Allowed uses` | what `train` and `eval` mean for this project |
| `## Ledger` | what the ledger records; name the fields `source_id`, `license_spdx`, `retrieved_at`, `sha256`, `allowed_uses` |
| `## Allowlist` | where the decisions live and what happens when a row fails them |
| `## Obligations` | what attribution, share-alike, and notice require in practice |
| `## Unknown and missing licenses` | the default answer and why |
| `## Revocation` | what happens to shards and models when a license is withdrawn |
| `## Owner and review` | who decides, and how often the list is reviewed |

**`docs/data/license-allowlist.toml`** holds the decisions in this format:

```toml
version = 1
owner = "who decides"               # a person or a team
reviewed = 2026-10-09               # a TOML date, no quotes

[[allow]]                           # one per allowed license
spdx = "CC-BY-4.0"                  # an SPDX identifier, or LicenseRef-<name>
uses = ["train", "eval"]            # a non-empty subset of train, eval
obligations = ["attribution"]       # from: attribution, share-alike, notice, no-endorsement
why = "One sentence, at least six words."

[[refuse]]                          # one per refused license
spdx = "CC-BY-NC-4.0"
why = "One sentence, at least six words."
```

A complete allowlist also passes a review. It decides each of these seven licenses, which appear on dataset cards, as an `[[allow]]` or a `[[refuse]]`: `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-ND-4.0`, `ODC-By-1.0`, `MIT`, `NOASSERTION`. It adds `CDLA-Sharing-1.0` and `Apache-2.0` (the fourth fixed rule) and anything else the sources use.

## Building the policy step by step

Try to make each decision yourself before reading the reference's. The reasoning runs from first principles to the file.

**Decisions must be data.** The question "may this document be trained on?" is asked by a program on every build, so the answer has to be in a file a program parses: one TOML entry per license, keyed by its identifier. The prose policy explains the decisions; it does not hold them.

**The key must be exact.** A program compares strings, so the key is the SPDX identifier, case-sensitive, the same spelling the ledger and the datasheet use. "CC BY 4.0" from a dataset card is translated to `CC-BY-4.0` once, at review time.

**The default is refusal.** Without a license all rights are reserved, so a license missing from the list is refused, and `NOASSERTION` is refused explicitly so the decision is visible and has a reason.

**Training is the stricter use.** Training produces weights that are released under a license the project chooses. Non-commercial and no-derivatives terms restrict exactly what a release does, so they can never allow `train`. Evaluation only reads text to score a model, so a license can be allowed for `eval` and refused for `train`, never the reverse.

**Obligations follow the family.** Each allowed license carries the duties of its family: attribution for `CC-BY*` and `ODC-By`, share-alike for `*-SA-*` and `CDLA-Sharing`, notice for permissive code licenses. Recording them here is what lets the datasheet print them without anyone re-reading license texts.

**One decision per license, every common one decided.** An identifier in both tables makes the answer depend on read order; a common license in neither fails the build the first time a source uses it. The review list is the set that appears on dataset cards often enough to decide in advance.

**The license is bound to bytes, not to a URL.** The ledger records `sha256` of the exact bytes fetched, so the decision applies to those bytes even if the URL later serves something else, and rows are append-only so the history survives.

**Someone owns it.** A policy nobody reviews goes stale, so the file names an owner and a review date and the prose names a cadence and a trigger (a new source).

### The allowlist, annotated

Here is the reference allowlist. The comments are added for this walkthrough; the entries are the reference file's.

```toml
version = 1
owner = "data owner"
reviewed = 2026-10-09               # a TOML date: parsed, comparable, not free text

# Public domain and permissive: train and eval. Code licenses ask for the notice.
[[allow]]
spdx = "CC0-1.0"
uses = ["train", "eval"]
obligations = []                    # a dedication asks nothing
why = "A public domain dedication: no conditions on use or redistribution."

[[allow]]
spdx = "Apache-2.0"                 # rule 4: the course fixtures
uses = ["train", "eval"]
obligations = ["notice"]
why = "The course fixtures and much open code; keep the license notice with copies we share."

[[allow]]
spdx = "MIT"
uses = ["train", "eval"]
obligations = ["notice"]
why = "Permissive; keep the copyright and permission notice with copies we share."

# Attribution family: allowed, and the datasheet must credit them.
[[allow]]
spdx = "CC-BY-4.0"
uses = ["train", "eval"]
obligations = ["attribution"]
why = "Permissive with attribution; the datasheet credits every source under it."

[[allow]]
spdx = "ODC-By-1.0"
uses = ["train", "eval"]
obligations = ["attribution"]
why = "The open data attribution license; the datasheet credits the database."

# Share-alike: CDLA-Sharing says model results are not bound; CC BY-SA does not say.
[[allow]]
spdx = "CDLA-Sharing-1.0"           # rule 4: TinyStories, the C1 corpus
uses = ["train", "eval"]
obligations = ["share-alike"]
why = "TinyStories, the C1 corpus: data we share must stay under the same terms, model results are not restricted."

[[allow]]
spdx = "CC-BY-SA-4.0"
uses = ["eval"]                     # the unsettled case: the narrower use until decided
obligations = ["attribution", "share-alike"]
why = "Allowed for evaluation only until we decide whether a released model counts as an adaptation that must be ShareAlike."

# Refused: incompatible with releasing a model, or no license at all.
[[refuse]]
spdx = "CC-BY-NC-4.0"
why = "Non-commercial terms conflict with releasing the model under a license of our choosing."

[[refuse]]
spdx = "CC-BY-ND-4.0"
why = "No-derivatives terms forbid the adaptation that training produces."

[[refuse]]
spdx = "NOASSERTION"
why = "Nobody recorded a license, so all rights are reserved until someone finds the terms."
```

Read the two share-alike entries side by side. Both are ShareAlike, but CDLA-Sharing-1.0 states that results computed from the data (a model) are not bound by its share-alike term, so training is safe, while CC-BY-SA-4.0 is silent, so the solution takes the narrower use and writes down why. That is the pattern for any unsettled question: allow the use whose answer is known, and make the open question visible in `why`. Notice also what is missing: `NONE` has no entry at all. It is refused by the default, and it can never be allowed. The full file is [`license-allowlist.toml`](../../../../course/solutions/docs/data/license-allowlist.toml).

### The policy, annotated

The prose answers the eight questions. The sections that carry decisions read as follows in the reference:

> **Allowed uses.** Training means its text shapes the weights of a model we release, so its license must allow us to make and share an adaptation. Evaluation means its text is only read to score a model and is never trained on; decontamination (data.04) keeps evaluation text out of training sets. A source may be allowed for evaluation and refused for training, never the other way round.

This defines the two values of `uses` by what they do to the released artifact, which is what makes the no-NC-or-ND rule follow; and it names the mechanism that keeps an `eval`-only source out of training, the decontamination pass in [Near-duplicate dedup: MinHash, LSH, union-find, and decontamination](../../data-engineering/05-corpus-pipeline/04-near-dedup-minhash-lsh.md).

> **Ledger.** The row records `source_id` (our name for the source), `url` (where it came from), `license_spdx` (the SPDX identifier from the source's license page, copied when we pinned it), `retrieved_at` (UTC time of the download), `sha256` (of the exact bytes received, so the license is tied to those bytes and not to whatever the URL serves later), `allowed_uses` (from the corpus config), and `pii_policy`. A row is never edited after the fact; a new version of a source is a new row with a new checksum.

Every field is named with the reason it exists, so a reader of a ledger row knows where the license decision is recorded.

> **Allowlist.** data.08 checks every ledger row against it and fails the build, with exit code 65, when a row's license is refused, missing from the list, or allowed only for a use the row claims beyond it.

These are three failure cases, and the last matches the `wiki-mini` example: a config that claims `train` for an `eval`-only license fails, and the fix is in the config.

> **Unknown and missing licenses.** A source with no license, a license we cannot identify, or a license recorded as NOASSERTION is refused for every use. All rights are reserved by default, so the absence of a license is a refusal, not a permission.

> **Revocation.** The ledger row is marked `revoked: true` and the source moves to `[[refuse]]`. Every shard and token stream derived from it is rebuilt without it, and any model trained on it is withdrawn or retrained before its next release.

Revocation is planned before it is needed: the policy says what happens to data and to models, and the data-incident drill rehearses the path. The remaining sections (Scope, Obligations, Owner and review) say what is covered, translate each obligation into an action (credit in the datasheet, keep the license on shared derived data, keep notices with copies), and name the owner with a quarterly review plus a review whenever a source is pinned. The full policy is [`LICENSE_POLICY.md`](../../../../course/solutions/docs/data/LICENSE_POLICY.md).

### Why this design

The split into two files is deliberate. The TOML is what a build enforces in milliseconds; the prose is what a reviewer, an auditor, or a new maintainer reads. Keeping the decisions only in prose would make automated verification impossible; keeping them only in TOML would lose the reasons and the process (revocation, ownership).

An allow-list beats a deny-list because a deny-list fails open: every license nobody thought of is permitted. An allow-list fails closed, which matches the legal default that unlicensed text is not usable. Per-use decisions beat a single yes or no because a single boolean would force either refusing CC BY-SA text for evaluation (losing useful benchmarks) or allowing it for training (taking an unsettled legal risk); two uses capture the real distinction at almost no cost. And SPDX beats free-text names because free text is ambiguous ("Apache", "Apache License", "Apache 2"), and string matching on it either refuses legitimate sources or admits the wrong license. SPDX also gives expressions (`OR`, `AND`) a defined meaning.

The policy accepts tradeoffs. It is conservative: CC BY-SA text is not trained on even though some lawyers would allow it, and an unknown license blocks a build until a human acts. Both cost convenience; both are reversible by a recorded decision, while training on text that should not have been used is not reversible without retraining.

## What can go wrong

A correct solution has to handle a specific set of cases, each of which matters to something downstream:

| Case | Checks | Why it matters downstream |
|---|---|---|
| every policy section present | the eight sections, each with enough words | every question has an answer |
| no placeholders | no `TODO`, `TBD`, or `<lower-case placeholder>` outside code | a template is not a decision |
| the Ledger section names the fields | `source_id`, `license_spdx`, `retrieved_at`, `sha256`, `allowed_uses` | readers know where a decision is recorded |
| an owner and a review date | `version = 1`, an owner, a past TOML date, no unknown keys | someone maintains it |
| every entry complete | `uses`, `obligations`, and `why` in the vocabularies; no unknown keys | the verifier reads every field |
| every id an SPDX identifier | each id is on the SPDX list or `LicenseRef-...` | exact-match checks in the verifier |
| no license both allowed and refused | one decision per license | no order-dependent answers |
| the system's sources allowed for training | `CDLA-Sharing-1.0` and `Apache-2.0` allow `train` | the TinyStories corpus can be built |
| every review case decided | the seven review licenses are each allowed or refused | no undecided license slips in |
| no NC or ND in training | no `-NC`/`-ND` license allows `train` | the released model is the project's to license |
| unknown licenses refused | `NOASSERTION` refused, `NONE` never allowed | the default is no |
| obligations follow the license family | attribution and share-alike recorded where the family requires them | the datasheet credits and relicenses correctly |

The common mistakes are worth naming one by one, because each is easy to make in good faith.

> **Warning:** Copying the license as the card prints it ("CC BY 4.0") makes the verifier's exact match refuse a source you meant to allow. Every `spdx` value must be the exact SPDX identifier (`CC-BY-4.0`).

Treating "no license" as "free to use" lets a scraped source enter the corpus with no permission at all; the solution makes `NOASSERTION` an explicit `[[refuse]]`, and anything missing from the list is refused. Allowing `CC-BY-NC-*` for training because the project is a hobby means the released model inherits a restriction its license does not state; `CC-BY-NC-4.0` and `CC-BY-ND-4.0` are refused outright. Allowing a license without recording what it asks means the datasheet omits credits and ShareAlike data is redistributed under the wrong terms; every allow entry lists its family's obligations.

Three more mistakes are about completeness. Leaving a review case undecided means the first source under it fails the build at the worst time, or a permissive default lets it in; all seven review licenses appear in exactly one table. Listing a license in both tables makes the answer depend on which line a program reads last; there is one entry per identifier. A policy with no owner or review date is never updated after the first week; the reference has `owner`, `reviewed`, and an "Owner and review" section with a quarterly cadence. Finally, a Ledger section that never says what is recorded leaves nobody able to tell which field carries the license decision; the reference names every field and explains why `sha256` binds the license.

## Where this leads

The policy has no code of its own, but much of the system reads it. The fetcher in [Async fetch with resume, checksums, and license capture](../../data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md) writes the ledger row described here for every source it fetches, with the pinned license. The `{corpus} ledger verify` command in [Licensing ledger verification and the datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md) checks every row and every shard against the allowlist and exits 65 on a refused or unknown license. [Privacy and PII policy](../02-privacy-and-pii/01-privacy-and-pii-policy.md) applies the same structure to personal data: a policy in prose plus decisions a program checks. The datasheet in [Datasheet and model card](../03-model-and-data-cards/01-datasheet-and-model-card.md) lists every source, its license, and the obligations recorded here. And `ModelRelease` in [ModelRelease: gates, canary, PromQL burn check, rollback](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) refuses a model whose training sources are not licensed for `train`.

### Further reading

- Longpre et al., *The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI* (2023, free). Its Data Provenance Explorer and provenance cards (dataprovenance.org) are the production equivalent of the ledger, adding lineage across re-releases and license categories per dataset.
- The SPDX License List and the SPDX specification, annex D "License expressions" (spdx.org, free).
- Creative Commons, "About CC licenses" (free), and the Community Data License Agreement texts (cdla.dev, free).
- ScanCode Toolkit (`aboutcode-org/scancode-toolkit`), FOSSA, and ClearlyDefined: the production equivalents of the allowlist, adding license detection from text and curated license data per package and dataset.
- CommonCanvas (Gokaslan et al. 2023), Common Corpus, and the Common Pile (Kandpal et al. 2025): corpora built only from openly licensed or public domain text, with per-document licenses.

## Key terms

- **License**: the author's written permission to copy, adapt, or distribute a work under stated conditions.
- **Terms of service**: a contract with a website that can bind you independently of copyright.
- **SPDX identifier**: the short, case-sensitive, machine-readable name of a license, such as `CC-BY-4.0`.
- **License expression**: identifiers combined with `OR` (either may be chosen) or `AND` (both apply).
- **Allowlist**: the policy in data; each license allowed for named uses with its obligations, or refused with a reason.
- **Obligation**: a duty an allowed license imposes, such as attribution, share-alike, or notice.
- **Ledger**: the append-only record of every fetched source, binding its reviewed license to the sha256 of the bytes received.
- **Revocation**: withdrawing a source after the fact, which forces a rebuild of derived data and a retrain or withdrawal of models trained on it.

## Summary

- Text you did not write is somebody's, and without a license all rights are reserved; a dataset's license is a property of every source in it.
- A license is named by its SPDX identifier, the one spelling the ledger, the allowlist, and the datasheet share.
- The allowlist is a policy in data: per-use decisions with obligations, failing closed on anything unknown.
- Non-commercial and no-derivatives terms are incompatible with training a model you release, so they are refused for training.
- The ledger ties a license to the sha256 of the exact bytes fetched, so the decision cannot drift when the URL starts serving something else.
- This is engineering, not legal advice: the job is to make provenance auditable, and a lawyer decides what a license permits.

## Review questions

1. Why is `CC-BY-SA-4.0` allowed for `eval` but `CDLA-Sharing-1.0` allowed for `train`, when both are share-alike licenses?

   <details><summary>Answer</summary>

   Share-alike binds "adaptations" or derived data that is shared. CDLA-Sharing-1.0 says explicitly that results computed from the data, a trained model among them, are not bound by its share-alike term. The Creative Commons licenses do not say whether a model is an adaptation. The solution allows the use whose answer is known (evaluation) and records the open question in `why` until someone decides.

   </details>

2. Why does `NOASSERTION` get an explicit `[[refuse]]` entry when anything missing from the list is refused anyway?

   <details><summary>Answer</summary>

   `NOASSERTION` is a value the ledger will actually contain (it is what gets recorded when nobody stated a license), so the decision about it should be visible, with a reason a reader can check, rather than an accident of the default. It also makes the policy robust to a future change of the default.

   </details>

3. A corpus config claims `allowed_uses = ["train", "eval"]` for a source whose license is allowed only for `eval`. What happens, and where is the fix?

   <details><summary>Answer</summary>

   The verifier reads the ledger row, finds the license allowed for `eval` only, and fails verification with exit 65 because the row claims a use the policy does not grant. The fix is to change the config to `["eval"]`. Widening the allowlist to make the build pass would change the policy to fit the data, which is the opposite of what the policy is for.

   </details>

4. Why does the ledger record `sha256` of the fetched bytes instead of trusting the URL?

   <details><summary>Answer</summary>

   A license decision is about specific text. The same URL can serve a new version under different terms, or a different file entirely, later. The digest ties the reviewed license to the exact bytes that entered the corpus, so anyone can verify that the data trained on is the data reviewed, and a new version shows up as a new row with a new checksum.

   </details>

5. Why is a non-commercial license refused for training even if this project never sells anything?

   <details><summary>Answer</summary>

   The model is released under a license the project chooses, and downstream users may use it commercially. A non-commercial term on the training data would be a restriction the released model's license does not state, passed on to people who never saw it. Refusing NC and ND data for training keeps the released model the project's to license.

   </details>

6. Why is an allow-list safer than a deny-list for training data?

   <details><summary>Answer</summary>

   A deny-list fails open: any license nobody thought to list is permitted. An allow-list fails closed, refusing anything not explicitly decided, which matches the legal default that text without a license is not usable.

   </details>
