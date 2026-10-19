# Data licensing and the ledger

## Overview

| | |
|---|---|
| **Module** | `ethics.01` · practice · docs · Pass 3 · 1.5 to 2 h reading |
| **Solution** | the reference policy [`course/solutions/docs/data/LICENSE_POLICY.md`](../../../../course/solutions/docs/data/LICENSE_POLICY.md) (eight fixed sections) and the machine-readable decisions `data.08` enforces, [`course/solutions/docs/data/license-allowlist.toml`](../../../../course/solutions/docs/data/license-allowlist.toml) |
| **Contract** | the ledger row: [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json); the allowlist format is defined in section 4 of this chapter |
| **Needs** | nothing: this is policy, written before the pipeline it governs |
| **Used by** | no code call site (a policy): `data.01` records a ledger row per source, `data.08` verifies every row and shard against the allowlist, `ethics.03` reports it in the datasheet, and `dur.12` refuses to release a model whose ledger does not verify |
| **Optional depth** | Longpre et al., *The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI* (2023, free); the SPDX License List and the SPDX specification, annex D "License expressions" (spdx.org, free); Creative Commons, "About CC licenses" (free); Community Data License Agreement texts (cdla.dev, free) |

This chapter shows how to make provenance auditable. It is not legal advice: whether a license permits training a model is decided by a lawyer for your situation. What the solution fixes is that every decision is written down, uses names a program can check, and is checked on every build.

## Key Takeaways

- Text you did not write is somebody's, and without a license all rights are reserved: "no license" means "no", not "free".
- A license is named by its **SPDX identifier**, the one spelling the ledger, the allowlist, and the datasheet share.
- An allowlist is a policy in data: each license allowed for named uses with its obligations, or refused with a reason.
- Non-commercial and no-derivatives terms are incompatible with training a model you release; the solution refuses them for training.
- The ledger ties a license to the sha256 of the exact bytes you fetched, so the decision cannot drift when the URL starts serving something else.

## How to read this chapter

Sections 1 and 2 explain why text carries a license, how licenses group into families, and what the ledger records. Section 3 decides a small corpus by hand. Section 4 fixes the format of the two files, and section 5 walks through the reference policy and allowlist decision by decision and explains why each one is right. Try to make each decision yourself before reading the solution's.

---

## 1. Why now

The fetcher (`data.01`, this pass) is about to download every source in the corpus config, and in Pass 9 a workflow will train a model on the result and publish it (`dur.12`). Between the two, somebody has to be able to answer, for every document in every shard: where did this come from, under what terms, and do those terms let us train on it and release what we trained? The Data Provenance Initiative audited over 1,800 popular fine-tuning datasets and found licenses missing for most of them on the hosting sites and miscategorized for a large share, usually more permissive than the original authors granted. Once text is mixed into a corpus the answer cannot be reconstructed, so the policy has to exist before the first download, and it has to be data a build can check, not a paragraph in a README.

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

The SPDX License List gives every common license a short, case-sensitive identifier: `CC-BY-4.0`, not "CC BY 4.0" or "cc-by-4.0"; `Apache-2.0`, not "Apache 2.0". A license SPDX does not list is written `LicenseRef-<name>`. An **expression** combines identifiers: `MIT OR Apache-2.0` (you may choose either: the uses of both together) and `CC-BY-4.0 AND MIT` (both apply: only the uses both allow). `data.08` evaluates expressions against the allowlist; an identifier it does not find is refused.

### 2.4 The rules the course fixes

Most of the allowlist is a judgment call. Four rules are not, because the rest of the course depends on them:

1. **Unknown is refused.** `NOASSERTION` is refused explicitly, and `NONE` can never be allowed.
2. **No NC or ND in training.** The model's weights are released (`dur.12`) under a license the project chooses; a non-commercial term forbids commercial use of what is built from the data and a no-derivatives term forbids adaptations, so neither can be in a training set. Evaluation use is a judgment call.
3. **Obligations are recorded.** Every allowed attribution license (`CC-BY*`, `ODC-By`) lists `attribution`; every ShareAlike license (`*-SA-*`, `CDLA-Sharing`) lists `share-alike`. The datasheet (`ethics.03`) prints them from here.
4. **The course's own sources are allowed for training**: TinyStories, the C1 corpus, is `CDLA-Sharing-1.0`, and the course fixtures are `Apache-2.0`. Refusing them is a valid policy, but then the course corpus cannot be built.

### 2.5 The ledger

Every fetched source gets one row in `corpus/LEDGER.jsonl` (`formats/ledger.schema.json`), written by `data.01` at the moment the bytes, the URL, and the reviewed license are all in one place:

| Field | Records | Why it matters |
|---|---|---|
| `source_id` | the project's name for the source | shard rows point back to it |
| `url` | where the bytes came from | anyone can go and look |
| `license_spdx` | the license reviewed when the source was pinned | what the allowlist is checked against |
| `retrieved_at` | UTC time of the download | the terms in force then |
| `sha256` | digest of the exact bytes received | the license applies to these bytes, not to whatever the URL serves later |
| `allowed_uses` | `train`, `eval` as the config claims | `data.08` checks the claim against the allowlist |
| `pii_policy`, `filters_applied`, `kept`, `dropped`, `notes` | what the pipeline did | the datasheet |

Rows are appended, never edited: a new version of a source is a new row with a new checksum, and a withdrawn source gets a row with `revoked: true`.

### 2.6 Revocation

A license can turn out wrong, an author can ask for removal, a host can change its terms. The policy says in advance what happens: the source is refused from then on, every shard and token stream derived from it is rebuilt without it, and a model trained on it is retrained or withdrawn before its next release. The data-incident drill (`ops.08`) rehearses exactly this.

## 3. Worked example by hand

A different corpus than the course's, to decide by hand. Three sources:

| Source | License on the card | What a reviewer finds |
|---|---|---|
| `poems` | "Public Domain" | the site says the poems were dedicated with CC0 |
| `wiki-mini` | "CC BY-SA" | the footer says Creative Commons Attribution-ShareAlike 3.0 |
| `forum` | (none) | a scrape of a forum; the terms of service reserve all rights |

The decisions:

1. `poems`: the SPDX identifier is `CC0-1.0`. Allowed for `train` and `eval`, no obligations.
2. `wiki-mini`: `CC-BY-SA-3.0`. Attribution and ShareAlike. If the project decides a released model is not an adaptation under this license, allow `train` and `eval` with obligations `attribution` and `share-alike`; if unsure, allow `eval` only and say so in `why`. This example chooses `eval`.
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

## 4. The artifact

The solution is two files.

**`docs/data/LICENSE_POLICY.md`**: prose, with these eight `##` sections, each long enough to answer its question (at least a couple of sentences), and no template placeholders (`TODO`, `<your text>`) left:

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

**The review.** A complete allowlist decides each of these seven licenses, which appear on dataset cards, as an `[[allow]]` or a `[[refuse]]`: `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-ND-4.0`, `ODC-By-1.0`, `MIT`, `NOASSERTION`. It adds `CDLA-Sharing-1.0` and `Apache-2.0` (rule 4 of section 2.4) and anything else the sources use.

### What a correct solution must handle

| Case | Checks | Why it matters downstream |
|---|---|---|
| every policy section present | the eight sections, each with enough words | every question has an answer |
| no placeholders | no `TODO`, `TBD`, or `<lower-case placeholder>` outside code | a template is not a decision |
| the Ledger section names the fields | `source_id`, `license_spdx`, `retrieved_at`, `sha256`, `allowed_uses` | readers know where a decision is recorded |
| an owner and a review date | `version = 1`, an owner, a past TOML date, no unknown keys | someone maintains it |
| every entry complete | `uses`, `obligations`, and `why` in the vocabularies; no unknown keys | data.08 reads every field |
| every id an SPDX identifier | each id is on the SPDX list or `LicenseRef-...` | exact-match checks in data.08 |
| no license both allowed and refused | one decision per license | no order-dependent answers |
| the course sources allowed for training | `CDLA-Sharing-1.0` and `Apache-2.0` allow `train` | the course corpus can be built |
| every review case decided | the seven review licenses are each allowed or refused | no undecided license slips in |
| no NC or ND in training | no `-NC`/`-ND` license allows `train` | the released model is the project's to license |
| unknown licenses refused | `NOASSERTION` refused, `NONE` never allowed | the default is no |
| obligations follow the license family | attribution and share-alike recorded where the family requires them | the datasheet credits and relicenses correctly |

## 5. The solution, step by step

### From first principles

1. **Decisions must be data.** The question "may this document be trained on?" is asked by a program (`data.08`) on every build, so the answer has to be in a file a program parses: one TOML entry per license, keyed by its identifier. The prose policy explains the decisions; it does not hold them.
2. **The key must be exact.** A program compares strings, so the key is the SPDX identifier, case-sensitive, the same spelling the ledger and the datasheet use (section 2.3). "CC BY 4.0" from a dataset card is translated to `CC-BY-4.0` once, at review time.
3. **The default is refusal.** Without a license all rights are reserved (section 2.1), so a license missing from the list is refused, and `NOASSERTION` is refused explicitly so the decision is visible and has a reason.
4. **Training is the stricter use.** Training produces weights that are released under a license the project chooses (`dur.12`). Non-commercial and no-derivatives terms restrict exactly what a release does, so they can never allow `train`. Evaluation only reads text to score a model, so a license can be allowed for `eval` and refused for `train`, never the reverse.
5. **Obligations follow the family.** Each allowed license carries the duties of its family (section 2.2): attribution for `CC-BY*` and `ODC-By`, share-alike for `*-SA-*` and `CDLA-Sharing`, notice for permissive code licenses. Recording them here is what lets the datasheet (`ethics.03`) print them without anyone re-reading license texts.
6. **One decision per license, every common one decided.** An identifier in both tables makes the answer depend on read order; a common license in neither fails the build the first time a source uses it. The review list in section 4 is the set that appears on dataset cards often enough to decide in advance.
7. **The license is bound to bytes, not to a URL.** The ledger records `sha256` of the exact bytes fetched, so the decision applies to those bytes even if the URL later serves something else, and rows are append-only so the history survives.
8. **Someone owns it.** A policy nobody reviews goes stale, so the file names an owner and a review date and the prose names a cadence and a trigger (a new source).

### The allowlist, annotated

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

The comments above are added for this walkthrough; the entries are the reference file's. Read the two share-alike entries side by side: both are ShareAlike, but CDLA-Sharing-1.0 states that results computed from the data (a model) are not bound by its share-alike term, so training is safe, while CC-BY-SA-4.0 is silent, so the solution takes the narrower use and writes down why. That is the pattern for any unsettled question: allow the use whose answer is known, and make the open question visible in `why`. `NONE` has no entry at all: it is refused by the default, and it can never be allowed. Full file: [`license-allowlist.toml`](../../../../course/solutions/docs/data/license-allowlist.toml).

### The policy, annotated

The prose answers the eight questions of section 4. The sections that carry decisions, quoted from the reference:

> **Allowed uses.** Training means its text shapes the weights of a model we release, so its license must allow us to make and share an adaptation. Evaluation means its text is only read to score a model and is never trained on; decontamination (data.04) keeps evaluation text out of training sets. A source may be allowed for evaluation and refused for training, never the other way round.

This defines the two values of `uses` by what they do to the released artifact, which is what makes rule 2 follow; and it names the mechanism (`data.04`) that keeps an `eval`-only source out of training.

> **Ledger.** The row records `source_id` (our name for the source), `url` (where it came from), `license_spdx` (the SPDX identifier from the source's license page, copied when we pinned it), `retrieved_at` (UTC time of the download), `sha256` (of the exact bytes received, so the license is tied to those bytes and not to whatever the URL serves later), `allowed_uses` (from the corpus config), and `pii_policy`. A row is never edited after the fact; a new version of a source is a new row with a new checksum.

Every field is named with the reason it exists, so a reader of a ledger row knows where the license decision is recorded (step 7).

> **Allowlist.** data.08 checks every ledger row against it and fails the build, with exit code 65, when a row's license is refused, missing from the list, or allowed only for a use the row claims beyond it.

Three failure cases, matching section 3's `wiki-mini`: a config that claims `train` for an `eval`-only license fails, and the fix is in the config.

> **Unknown and missing licenses.** A source with no license, a license we cannot identify, or a license recorded as NOASSERTION is refused for every use. All rights are reserved by default, so the absence of a license is a refusal, not a permission.

> **Revocation.** The ledger row is marked `revoked: true` and the source moves to `[[refuse]]`. Every shard and token stream derived from it is rebuilt without it, and any model trained on it is withdrawn or retrained before its next release.

Revocation is planned before it is needed: the policy says what happens to data and to models, and the data-incident drill (`ops.08`) rehearses the path. The remaining sections (Scope, Obligations, Owner and review) say what is covered, translate each obligation into an action (credit in the datasheet, keep the license on shared derived data, keep notices with copies), and name the owner with a quarterly review plus a review whenever a source is pinned. Full policy: [`LICENSE_POLICY.md`](../../../../course/solutions/docs/data/LICENSE_POLICY.md).

### Why this solution

- **A two-file split.** The TOML is what a build enforces in milliseconds; the prose is what a reviewer, an auditor, or a new maintainer reads. Keeping the decisions only in prose would make `data.08` impossible; keeping them only in TOML would lose the reasons and the process (revocation, ownership).
- **Allow-list over deny-list.** A deny-list fails open: every license nobody thought of is permitted. An allow-list fails closed, which matches the legal default that unlicensed text is not usable.
- **Per-use decisions over a single yes or no.** A single boolean would force either refusing CC BY-SA text for evaluation (losing useful benchmarks) or allowing it for training (taking an unsettled legal risk). Two uses capture the real distinction at almost no cost.
- **SPDX over free-text names.** Free text is ambiguous ("Apache", "Apache License", "Apache 2"), and string matching on it either refuses legitimate sources or admits the wrong license. SPDX also gives expressions (`OR`, `AND`) a defined meaning.
- **Tradeoffs accepted.** The policy is conservative: CC BY-SA text is not trained on even though some lawyers would allow it, and an unknown license blocks a build until a human acts. Both cost convenience; both are reversible by a recorded decision, while training on text that should not have been used is not reversible without retraining.

### Explain it back

**1. Why is `CC-BY-SA-4.0` allowed for `eval` but `CDLA-Sharing-1.0` allowed for `train`, when both are share-alike licenses?**

<details><summary>Answer</summary>

Share-alike binds "adaptations" or derived data that is shared. CDLA-Sharing-1.0 says explicitly that results computed from the data, a trained model among them, are not bound by its share-alike term. The Creative Commons licenses do not say whether a model is an adaptation. The solution allows the use whose answer is known (evaluation) and records the open question in `why` until someone decides.

</details>

**2. Why does `NOASSERTION` get an explicit `[[refuse]]` entry when anything missing from the list is refused anyway?**

<details><summary>Answer</summary>

`NOASSERTION` is a value the ledger will actually contain (it is what gets recorded when nobody stated a license), so the decision about it should be visible, with a reason a reader can check, rather than an accident of the default. It also makes the policy robust to a future change of the default.

</details>

**3. A corpus config claims `allowed_uses = ["train", "eval"]` for a source whose license is allowed only for `eval`. What happens, and where is the fix?**

<details><summary>Answer</summary>

`data.08` reads the ledger row, finds the license allowed for `eval` only, and fails verification with exit 65 because the row claims a use the policy does not grant. The fix is to change the config to `["eval"]`. Widening the allowlist to make the build pass would change the policy to fit the data, which is the opposite of what the policy is for.

</details>

**4. Why does the ledger record `sha256` of the fetched bytes instead of trusting the URL?**

<details><summary>Answer</summary>

A license decision is about specific text. The same URL can serve a new version under different terms, or a different file entirely, later. The digest ties the reviewed license to the exact bytes that entered the corpus, so anyone can verify that the data trained on is the data reviewed, and a new version shows up as a new row with a new checksum.

</details>

**5. Why is a non-commercial license refused for training even if this project never sells anything?**

<details><summary>Answer</summary>

The model is released (`dur.12`) under a license the project chooses, and downstream users may use it commercially. A non-commercial term on the training data would be a restriction the released model's license does not state, passed on to people who never saw it. Refusing NC and ND data for training keeps the released model the project's to license.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. Copying the license as the card prints it ("CC BY 4.0") | data.08's exact match refuses a source you meant to allow | every `spdx` value is the exact SPDX identifier (`CC-BY-4.0`); step 2 |
| 2. Treating "no license" as "free to use" | a scraped source enters the corpus with no permission at all | `NOASSERTION` is an explicit `[[refuse]]`, and anything missing from the list is refused; step 3 |
| 3. Allowing `CC-BY-NC-*` for training because the project is a hobby | the released model inherits a restriction its license does not state | `CC-BY-NC-4.0` and `CC-BY-ND-4.0` are refused outright; step 4 |
| 4. Allowing a license without recording what it asks | the datasheet omits credits; ShareAlike data is redistributed under the wrong terms | every allow entry lists its family's obligations; step 5 |
| 5. Leaving a review case undecided | the first source under it fails the build at the worst time, or a permissive default lets it in | all seven review licenses appear in exactly one table; step 6 |
| 6. Listing a license in both tables | the answer depends on which line a program reads last | one entry per identifier; step 6 |
| 7. A policy with no owner or review date | the list is never updated after the first week | `owner`, `reviewed`, and the "Owner and review" section with a quarterly cadence; step 8 |
| 8. A Ledger section that never says what is recorded | nobody can tell which field carries the license decision | the Ledger section names every field and why `sha256` binds the license; step 7 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `data.01` | writes the ledger row of section 2.5 for every source it fetches, with the pinned license |
| Forward | `data.08` | `{corpus} ledger verify` checks every row and every shard against the allowlist and exits 65 on a refused or unknown license |
| Forward | `ethics.02` | the same structure for personal data: a policy in prose plus decisions a program checks |
| Forward | `ethics.03` | the datasheet lists every source, its license, and the obligations recorded here |
| Forward | `dur.12` | `ModelRelease` refuses a model whose training sources are not licensed for `train` |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the allowlist | ScanCode Toolkit, FOSSA, ClearlyDefined | license detection from text, curated license data per package and dataset | `aboutcode-org/scancode-toolkit` |
| the ledger | Data Provenance Initiative's Data Provenance Explorer and provenance cards | lineage across re-releases, license categories per dataset | dataprovenance.org |
| the review | CommonCanvas, Common Corpus, the Common Pile | corpora built only from openly licensed or public domain text, with per-document licenses | Gokaslan et al. 2023; Kandpal et al. 2025 |
