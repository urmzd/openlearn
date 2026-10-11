<!-- ol:module ethics.01 -->
# Data licensing and the ledger

## Overview

| | |
|---|---|
| **Module** | `ethics.01` · practice · docs · Pass 3 · 3 to 5 h |
| **You build** | `docs/data/LICENSE_POLICY.md` (your licensing policy, eight fixed sections) and `docs/data/license-allowlist.toml` (the machine-readable decisions `data.08` enforces) |
| **Contract** | the ledger row: [`formats/ledger.schema.json`](../../course/contracts/formats/ledger.schema.json); the allowlist format is defined in section 4 of this chapter |
| **Tests** | `course/tests/ethics.01/check` runs `test_license_policy.py` against your two files (what each test checks: section 4) |
| **Needs** | nothing: this is policy, written before the pipeline it governs |
| **Used by** | no code call site (a policy): `data.01` records a ledger row per source, `data.08` verifies every row and shard against your allowlist, `ethics.03` reports it in the datasheet, and `dur.12` refuses to release a model whose ledger does not verify |
| **Milestone** | `MS-P3` |
| **Optional depth** | Longpre et al., *The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI* (2023, free); the SPDX License List and the SPDX specification, annex D "License expressions" (spdx.org, free); Creative Commons, "About CC licenses" (free); Community Data License Agreement texts (cdla.dev, free) |

This chapter teaches you to make provenance auditable. It is not legal advice: whether a license permits training a model is decided by a lawyer for your situation. What the course fixes is that every decision is written down, uses names a program can check, and is checked on every build.

## Key Takeaways

- Text you did not write is somebody's, and without a license all rights are reserved: "no license" means "no", not "free" (`test_unknown_licenses_are_refused`).
- A license is named by its **SPDX identifier**, the one spelling the ledger, the allowlist, and the datasheet share (`test_every_id_is_an_spdx_identifier`).
- An allowlist is a policy in data: each license allowed for named uses with its obligations, or refused with a reason (`test_every_entry_is_complete`, `test_every_review_case_is_decided`).
- Non-commercial and no-derivatives terms are incompatible with training a model you release; the course refuses them for training (`test_non_commercial_and_no_derivatives_are_never_trained_on`).
- The ledger ties a license to the sha256 of the exact bytes you fetched, so the decision cannot drift when the URL starts serving something else (`test_ledger_section_names_the_fields`).

## How to work this chapter

```bash
ol start ethics.01            # records the start; you write docs/data/ from section 4
ol tests ethics.01            # read the test catalog first
ol check ethics.01            # exit code is the verdict
```

---

## 1. Why now

Your fetcher (`data.01`, this pass) is about to download every source in your corpus config, and in Pass 9 a workflow will train a model on the result and publish it (`dur.12`). Between the two, somebody has to be able to answer, for every document in every shard: where did this come from, under what terms, and do those terms let us train on it and release what we trained? The Data Provenance Initiative audited over 1,800 popular fine-tuning datasets and found licenses missing for most of them on the hosting sites and miscategorized for a large share, usually more permissive than the original authors granted. Once text is mixed into a corpus the answer cannot be reconstructed, so the policy has to exist before the first download, and it has to be data a build can check, not a paragraph in a README.

## 2. Principles

### 2.1 Why text has a license at all

Copyright gives the author of a text the exclusive right to copy it, adapt it, and distribute it, automatically and without registration. A **license** is the author's written permission to do some of those things under conditions. With no license, nothing is permitted beyond what the law allows anyway (fair use and text-and-data-mining exceptions vary by country and are exactly what you would need a lawyer for). Separately, a website's **terms of service** can bind you by contract even for text that is not protected. A dataset aggregated from many sources has one license per source; the dataset card's single license line is a claim about all of them that someone else made.

### 2.2 License families

| Family | Examples (SPDX ids) | You may | The license asks |
|---|---|---|---|
| Public domain dedication | `CC0-1.0`, `PDDL-1.0` | anything | nothing |
| Permissive | `MIT`, `Apache-2.0`, `BSD-3-Clause`, `CDLA-Permissive-2.0` | copy, adapt, redistribute | keep the copyright and license text (**notice**) |
| Attribution | `CC-BY-4.0`, `ODC-By-1.0` | copy, adapt, redistribute | credit the source (**attribution**) |
| ShareAlike | `CC-BY-SA-4.0`, `CDLA-Sharing-1.0`, `ODbL-1.0` | copy, adapt, redistribute | derived data you share keeps the same license (**share-alike**) |
| Non-commercial | `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0` | use for non-commercial purposes | no commercial use |
| No derivatives | `CC-BY-ND-4.0`, `CC-BY-NC-ND-4.0` | copy unchanged | no adaptations |
| None or unknown | `NOASSERTION` (nobody said), `NONE` (no license) | nothing | |

Whether a trained model is an "adaptation" of its training text is unsettled. CDLA-Sharing-1.0 was written for this question and says that results computed from the data, a model among them, are not bound by its share-alike term; the Creative Commons licenses do not say. That is why the allowlist records **uses** separately: a license can be allowed for evaluation (reading text to score a model) and refused for training (shaping the weights of a model you release).

### 2.3 SPDX identifiers and expressions

The SPDX License List gives every common license a short, case-sensitive identifier: `CC-BY-4.0`, not "CC BY 4.0" or "cc-by-4.0"; `Apache-2.0`, not "Apache 2.0". A license SPDX does not list is written `LicenseRef-<name>`. An **expression** combines identifiers: `MIT OR Apache-2.0` (you may choose either: the uses of both together) and `CC-BY-4.0 AND MIT` (both apply: only the uses both allow). `data.08` evaluates expressions against your allowlist; an identifier it does not find is refused.

### 2.4 The rules the course fixes

Your allowlist is your decision. Four rules are not, because the rest of the course depends on them:

1. **Unknown is refused.** `NOASSERTION` is refused explicitly, and `NONE` can never be allowed.
2. **No NC or ND in training.** Your model's weights are released (`dur.12`) under a license you choose; a non-commercial term forbids commercial use of what you build from the data and a no-derivatives term forbids adaptations, so neither can be in a training set. Evaluation use is your call.
3. **Obligations are recorded.** Every allowed attribution license (`CC-BY*`, `ODC-By`) lists `attribution`; every ShareAlike license (`*-SA-*`, `CDLA-Sharing`) lists `share-alike`. The datasheet (`ethics.03`) prints them from here.
4. **The course's own sources are allowed for training**: TinyStories, the C1 corpus, is `CDLA-Sharing-1.0`, and the course fixtures are `Apache-2.0`. Refusing them is a valid policy, but then the course corpus cannot be built.

### 2.5 The ledger

Every fetched source gets one row in `corpus/LEDGER.jsonl` (`formats/ledger.schema.json`), written by `data.01` at the moment the bytes, the URL, and the license you reviewed are all in one place:

| Field | Records | Why it matters |
|---|---|---|
| `source_id` | your name for the source | shard rows point back to it |
| `url` | where the bytes came from | anyone can go and look |
| `license_spdx` | the license you reviewed when you pinned the source | what the allowlist is checked against |
| `retrieved_at` | UTC time of the download | the terms in force then |
| `sha256` | digest of the exact bytes received | the license applies to these bytes, not to whatever the URL serves later |
| `allowed_uses` | `train`, `eval` as the config claims | `data.08` checks the claim against the allowlist |
| `pii_policy`, `filters_applied`, `kept`, `dropped`, `notes` | what the pipeline did | the datasheet |

Rows are appended, never edited: a new version of a source is a new row with a new checksum, and a withdrawn source gets a row with `revoked: true`.

### 2.6 Revocation

A license can turn out wrong, an author can ask for removal, a host can change its terms. The policy says in advance what happens: the source is refused from then on, every shard and token stream derived from it is rebuilt without it, and a model trained on it is retrained or withdrawn before its next release. The data-incident drill (`ops.08`) rehearses exactly this.

## 3. Worked example by hand

A different corpus than yours, to decide by hand. Three sources:

| Source | License on the card | What you find |
|---|---|---|
| `poems` | "Public Domain" | the site says the poems were dedicated with CC0 |
| `wiki-mini` | "CC BY-SA" | the footer says Creative Commons Attribution-ShareAlike 3.0 |
| `forum` | (none) | a scrape of a forum; the terms of service reserve all rights |

The decisions:

1. `poems`: the SPDX identifier is `CC0-1.0`. Allowed for `train` and `eval`, no obligations.
2. `wiki-mini`: `CC-BY-SA-3.0`. Attribution and ShareAlike. If you decide a released model is not an adaptation under this license, allow `train` and `eval` with obligations `attribution` and `share-alike`; if you are unsure, allow `eval` only and say so in `why`. This example chooses `eval`.
3. `forum`: no license and restrictive terms: the ledger records `NOASSERTION`, refused.

The allowlist entries:

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

Now suppose the corpus config claims `allowed_uses = ["train", "eval"]` for `wiki-mini`. `data.08` reads its ledger row, finds `CC-BY-SA-3.0` allowed for `eval` only, and fails verification with exit 65: the config claims a use the policy does not grant. The fix is in the config (`allowed_uses = ["eval"]`), not in the allowlist.

## 4. The artifact and its check

Write two files in your repo.

**`docs/data/LICENSE_POLICY.md`**: prose, with these eight `##` sections, each long enough to answer its question (the check counts words, from 8 to 25 per section), and no template placeholders (`TODO`, `<your text>`) left:

| Section | Answers |
|---|---|
| `## Scope` | which data and which models the policy covers |
| `## Allowed uses` | what `train` and `eval` mean for you |
| `## Ledger` | what the ledger records; name the fields `source_id`, `license_spdx`, `retrieved_at`, `sha256`, `allowed_uses` |
| `## Allowlist` | where the decisions live and what happens when a row fails them |
| `## Obligations` | what attribution, share-alike, and notice require of you in practice |
| `## Unknown and missing licenses` | the default answer and why |
| `## Revocation` | what happens to shards and models when a license is withdrawn |
| `## Owner and review` | who decides, and how often the list is reviewed |

**`docs/data/license-allowlist.toml`**:

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

**The review.** Decide each of these seven licenses, which you will meet on dataset cards, as an `[[allow]]` or a `[[refuse]]`: `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-ND-4.0`, `ODC-By-1.0`, `MIT`, `NOASSERTION`. Add `CDLA-Sharing-1.0` and `Apache-2.0` (rule 4 of section 2.4) and anything else your sources use.

`ol check ethics.01` runs `course/tests/ethics.01/check` from your repo root; it refuses early if either file is missing.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_policy_has_every_section` | unit | the eight sections, each with enough words | every question has an answer |
| `test_policy_has_no_placeholders` | unit | no `TODO`, `TBD`, or `<lower-case placeholder>` outside code | a template is not a decision |
| `test_ledger_section_names_the_fields` | unit | the Ledger section names the five fields | readers know where a decision is recorded |
| `test_allowlist_has_an_owner_and_a_review_date` | unit | `version = 1`, an owner, a past TOML date, no unknown keys | someone maintains it |
| `test_every_entry_is_complete` | unit | `uses`, `obligations`, and `why` in the vocabularies; no unknown keys | data.08 reads every field |
| `test_every_id_is_an_spdx_identifier` | boundary | each id is on the course's SPDX list or `LicenseRef-...` | exact-match checks in data.08 |
| `test_no_license_is_both_allowed_and_refused` | boundary | one decision per license | no order-dependent answers |
| `test_the_course_sources_are_allowed_for_training` | unit | `CDLA-Sharing-1.0` and `Apache-2.0` allow `train` | your corpus can be built |
| `test_every_review_case_is_decided` | unit | the seven review licenses are each allowed or refused | no undecided license slips in |
| `test_non_commercial_and_no_derivatives_are_never_trained_on` | boundary | no `-NC`/`-ND` license allows `train` | the released model is yours to license |
| `test_unknown_licenses_are_refused` | boundary | `NOASSERTION` refused, `NONE` never allowed | the default is no |
| `test_obligations_follow_the_license_family` | unit | attribution and share-alike recorded where the family requires them | the datasheet credits and relicenses correctly |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Copying the license as the card prints it ("CC BY 4.0") | data.08's exact match refuses a source you meant to allow | `test_every_id_is_an_spdx_identifier` |
| 2. Treating "no license" as "free to use" | a scraped source enters the corpus with no permission at all | `test_unknown_licenses_are_refused` |
| 3. Allowing `CC-BY-NC-*` for training because the project is a hobby | the released model inherits a restriction its license does not state | `test_non_commercial_and_no_derivatives_are_never_trained_on` |
| 4. Allowing a license without recording what it asks | the datasheet omits credits; ShareAlike data is redistributed under the wrong terms | `test_obligations_follow_the_license_family` |
| 5. Leaving a review case undecided | the first source under it fails the build at the worst time, or a permissive default lets it in | `test_every_review_case_is_decided` |
| 6. Listing a license in both tables | the answer depends on which line a program reads last | `test_no_license_is_both_allowed_and_refused` |
| 7. A policy with no owner or review date | the list is never updated after the first week | `test_allowlist_has_an_owner_and_a_review_date` |
| 8. A Ledger section that never says what is recorded | nobody can tell which field carries the license decision | `test_ledger_section_names_the_fields` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `data.01` | writes the ledger row of section 2.5 for every source it fetches, with the license you pinned |
| Forward | `data.08` | `{corpus} ledger verify` checks every row and every shard against your allowlist and exits 65 on a refused or unknown license |
| Forward | `ethics.02` | the same structure for personal data: a policy in prose plus decisions a program checks |
| Forward | `ethics.03` | the datasheet lists every source, its license, and the obligations recorded here |
| Forward | `dur.12` | `ModelRelease` refuses a model whose training sources are not licensed for `train` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the allowlist | ScanCode Toolkit, FOSSA, ClearlyDefined | license detection from text, curated license data per package and dataset | `aboutcode-org/scancode-toolkit` |
| the ledger | Data Provenance Initiative's Data Provenance Explorer and provenance cards | lineage across re-releases, license categories per dataset | dataprovenance.org |
| the review | CommonCanvas, Common Corpus, the Common Pile | corpora built only from openly licensed or public domain text, with per-document licenses | Gokaslan et al. 2023; Kandpal et al. 2025 |
