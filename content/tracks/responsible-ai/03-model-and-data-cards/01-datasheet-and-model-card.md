<!-- ol:module ethics.03 -->
# Datasheet and model card

## Overview

| | |
|---|---|
| **Module** | `ethics.03` · practice · docs · Pass 9 · 3 to 4 h |
| **You build** | `docs/MODEL_CARD.md` (your capstone model), `docs/DATASHEET.md` (its corpus), and `docs/models/smollm2-135m-instruct.md` (the third-party agent model, labelled as such) |
| **Contract** | the templates [`templates/MODEL_CARD.md`](../../../../course/contracts/templates/MODEL_CARD.md) and [`templates/DATASHEET.md`](../../../../course/contracts/templates/DATASHEET.md); the release layout in [`formats/checkpoint.md`](../../../../course/contracts/formats/checkpoint.md); the third-party card's sections are defined in section 4 of this chapter |
| **Tests** | `course/tests/ethics.03/check` runs `test_cards.py` against your three files (what each test checks: section 4) |
| **Needs** | `data.08` your `permitted_uses` (`python/corpus/ledger.py`) judges the datasheet's licenses, and `{corpus} datasheet` gives its counts · reading: `ethics.01` your license allowlist (the datasheet's sources are checked against it), `ethics.04` the safety and bias rows, `L6.7` the quality rows |
| **Used by** | no code call site (documents): `dur.12`'s `ModelRelease` refuses a release without `MODEL_CARD.md`, and `ethics.05` turns the card's out-of-scope list into gateway policy |
| **Milestone** | `MS-C1` (artifact 4: the capstone's model card and data ledger) |
| **Optional depth** | Mitchell et al., [*Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993) (FAT* 2019); Gebru et al., [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010) (CACM 2021); Hugging Face, [model card guide](https://huggingface.co/docs/hub/model-cards); Pushkarna et al., [*Data Cards*](https://arxiv.org/abs/2204.01075) (2022) |

A check can tell that a card is complete, specific, and consistent with your other records. It cannot tell that the card is true; that part is yours.

## Key Takeaways

- A model card is the model's interface documentation: what it is, what it is for and not for, how it was evaluated with what uncertainty, what it gets wrong, and what it learned from (`test_model_card_sections`, `test_intended_and_out_of_scope_uses`).
- Every number in a card carries its interval and the threshold it was held to, and the bias section cites measurements, not adjectives (`test_evaluation_table_has_intervals`, `test_limitations_cite_measurements`).
- A datasheet answers why the data exists, what is in it, where it came from, and what was done to it; every source's license must be one your ethics.01 allowlist permits for training (`test_datasheet_sources_are_allowed`).
- A model you serve but did not train gets its own card that says so, with the exact upstream revision you pinned (`test_third_party_model_labelled`).

## How to work this chapter

```bash
ol start ethics.03            # records the start; you write docs/ from section 4
ol tests ethics.03            # the checks, and why each exists
ol check ethics.03            # runs course/tests/ethics.03/check in your repo
```

---

## 1. Why now

Pass 9 ends with your first release: `dur.12` exports the capstone to `models/<id>/<version>/`, runs `EvalSuite`, and refuses to promote it without a `MODEL_CARD.md` next to the weights. Everything a reader needs to decide whether to use the model exists by now, scattered: the ledger and datasheet counts from `data.08`, the license decisions from `ethics.01`, the quality numbers from `L6.7`, the safety and bias rows from `ethics.04`, the training recipe from `L11.1`. Without a card, each downstream user (including you, in Pass 10, deciding whether the agent may call this model) rediscovers its limits in production. And Pass 10 adds a model you did not train, SmolLM2-135M-Instruct; a card that blurs which weights are yours and which are fetched misleads everyone who reads it. This module writes the three documents and checks them.

## 2. Principles

**The model card answers fixed questions.** Mitchell et al. proposed the sections the course template keeps: model details (who, what architecture, how trained, which release, under what license), intended use and out-of-scope uses, evaluation, bias and risks, and the data. The order matters: a reader deciding whether to use the model reads the first two sections and stops if the answer is no.

**Out of scope is the most important list.** It names the uses the model must not serve: for a 10M-parameter story model, any factual question, any advice, any user who is not part of the course. In Pass 10 `ethics.05` turns this list into rules of the gateway's usage policy, so write it as something a rule can enforce ("no user-facing traffic", "only the story-completion route"), not as a hope.

**Numbers carry their uncertainty.** A bits-per-byte figure without an interval cannot be compared with the next release's. The evaluation table writes every value as `value (lo, hi)`: the t interval of `M07.4` for a mean loss, the Wilson interval for a rate, the bootstrap interval for a bias gap (`ethics.04`), and the threshold the release gate held it to. A row that is reported but not gated says so, and why.

**Limitations cite measurements.** "The model may reflect biases in its training data" is true of every model and tells the reader nothing. Write what you measured, with its interval, and what it does and does not show: a gender gap of 1.5 nats whose interval excludes zero, which string length alone could explain (`ethics.04`, section 2), is a limitation a reader can act on.

**The datasheet answers why and how the data exists.** Gebru et al.'s questions, kept by the template: motivation, composition (what one document is, how many, which splits, what personal data), collection (every source with its URL, license, retrieval date, and sha256, from the ledger), preprocessing (each filter and how many documents it dropped, deduplication, decontamination, tokenization), uses, and distribution and maintenance (where the files live and what happens when a source's license is revoked: drill `ops.08`). `{corpus} datasheet` (data.08) fills the counts; the prose is yours.

**Licenses are consistent across records.** The datasheet's sources, the ledger's rows, and your `ethics.01` allowlist must agree: a source whose SPDX identifier your allowlist does not permit for `train` cannot be in a model you release. The check reads the allowlist and fails on any source license outside it.

**Third-party models are labelled.** The agent model is HuggingFaceTB's SmolLM2-135M-Instruct, fetched by `ol fetch` at the revision pinned in `course/fixtures/ASSETS.tsv`. Its card states plainly that you did not train it, names the upstream repository and revision, keeps the upstream license (Apache-2.0), describes the architecture from its own `config.json`, and reports only evaluations you ran; until Pass 10 runs them, it says they have not been run, rather than quoting numbers you did not reproduce.

**A card is versioned with the weights.** The release copies `docs/MODEL_CARD.md` into `models/<id>/<version>/` (formats/checkpoint.md); the title names the model id and version so the gate can tell a stale card from the current one.

## 3. Worked example by hand

The course's reference card describes a model you can rebuild in a second: a byte-level Kneser-Ney 4-gram (`L2.1`) trained on the first 90% of the tinyshakespeare fixture. Its numbers were measured by `course/oracle/ethics.03/reference_card_numbers.py` with the `ethics.04` suites. Three excerpts show the decisions a card makes.

The evaluation table, every value with its interval and its threshold:

```markdown
| Suite | Metric | Value (95% CI) | Release threshold |
|---|---|---|---|
| quality | bpb on the held-out 10% (first 20,000 bytes) | 2.395 (2.366, 2.424) | upper bound at most 2.6 |
| safety | toxicity-rate over 18 prompts | 0.000 (0.000, 0.176) | upper bound at most 0.2 |
| safety | refusal-rate over 8 prompts that should be refused | 0.000 (0.000, 0.324) | not gated: the model cannot follow or refuse instructions |
| bias | bias-gap:gender, 32 paired probes, nats | 1.503 (0.583, 2.430) | reported, not gated |
```

The toxicity rate of 0 out of 18 has a Wilson upper bound of 0.176: with 18 prompts, "no toxic output" bounds the rate only below about 18%. The refusal row is reported but not gated, and the threshold column says why.

The limitation the bias row supports, written as a measurement:

```markdown
- The gender gap of 1.503 nats (0.583, 2.430) favours the first term of each
  pair (he, the boy, grandpa, the man). Those terms are shorter by one byte
  on average, and this model spends about 1.66 nats per byte, so length alone
  predicts a gap of that size: the measurement does not separate a learned
  association from string length, and we report it as such.
```

The 1.66 nats per byte is the measured 2.395 bits per byte times $\ln 2$; the average length difference of the four pairs is $(1 + 1 + 0 + 2)/4 = 1$ byte.

One source in the datasheet's Collection section, with everything the ledger records:

```markdown
- Source `tinyshakespeare`: the file `data/tinyshakespeare/input.txt` of the
  repository github.com/karpathy/char-rnn at revision
  6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e, license MIT (the repository; the
  plays themselves are in the public domain), retrieved 2026-10-09,
  sha256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
```

MIT is in the reference allowlist with `uses = ["train", "eval"]`, so `test_datasheet_sources_are_allowed` passes. Your capstone's datasheet will list TinyStories (CDLA-Sharing-1.0) instead.

## 4. The artifact and its check

Write three files in your repo:

| File | From | Sections (`##`) |
|---|---|---|
| `docs/MODEL_CARD.md` | `contracts/templates/MODEL_CARD.md`, title `# Model card: <model_id> v<N>` | Model details · Intended use · Evaluation · Bias, risks, and limitations · Data |
| `docs/DATASHEET.md` | `contracts/templates/DATASHEET.md`, title `# Datasheet: <dataset> <version>` | Motivation · Composition · Collection · Preprocessing · Uses · Distribution and maintenance |
| `docs/models/smollm2-135m-instruct.md` | this table, title `# Third-party model card: SmolLM2-135M-Instruct` | Provenance · License · Intended use · Evaluation · Limitations |

In the model card, Model details keeps the template's six bullets (`- **Developer:** ...`, Architecture, Training, Release, Third-party components, License with an SPDX id); Intended use keeps `- **Primary uses:**` and `- **Out of scope:**`; Evaluation keeps the table `| Suite | Metric | Value (95% CI) | Release threshold |` with at least a quality row (bpb, perplexity, or loss), a `safety` row, and a `bias` row, each value written `value (lo, hi)`; Data links `[DATASHEET.md](DATASHEET.md)` and names the data's license. Delete the template's angle-bracket placeholders as you fill them in. `ol check ethics.03` runs `course/tests/ethics.03/check`, which needs your `ethics.01` allowlist at `docs/data/license-allowlist.toml` and your `data.08` ledger module: it runs in your `python/` project's environment and imports `permitted_uses` from `corpus.ledger`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_model_card_sections` | unit | the five sections with enough prose, and the title with the model id and version | the release gate finds the right card |
| `test_no_template_placeholders` | unit | no `<placeholder>` or TODO left in any of the three files | a copied template is not a card |
| `test_model_details_fields` | unit | the six detail bullets, an SPDX license | who, what, and under what terms |
| `test_intended_and_out_of_scope_uses` | unit | both lists, specific enough to enforce | `ethics.05` policy rules |
| `test_evaluation_table_has_intervals` | unit | quality, safety, and bias rows, each `value (lo, hi)` with a threshold | releases can be compared |
| `test_limitations_cite_measurements` | unit | a measured interval, bias and safety both discussed | limitations a reader can act on |
| `test_card_links_the_datasheet` | unit | a link to `DATASHEET.md` and the data's license | the model traces to its sources |
| `test_datasheet_sections` | unit | Gebru's six sections, personal data, deduplication, decontamination | the corpus is documented |
| `test_datasheet_sources_are_allowed` | unit | every source license is allowed for `train` by your allowlist, judged by your `data.08` `permitted_uses` | `dur.12`'s license gate |
| `test_third_party_model_labelled` | unit | upstream repository, pinned revision, Apache-2.0, "did not train" | honest provenance for the agent model |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. copying the template and filling half of it | `<you>` in a released card | `test_no_template_placeholders` |
| 2. numbers without intervals | two releases cannot be compared; a lucky run looks like progress | `test_evaluation_table_has_intervals` |
| 3. generic limitations ("may be biased") | nothing to act on, nothing to regress against | `test_limitations_cite_measurements` |
| 4. an out-of-scope list nobody can enforce | the usage policy has nothing to encode | `test_intended_and_out_of_scope_uses` |
| 5. a datasheet source under a license your allowlist refuses | the release gate (or a lawyer) stops you later | `test_datasheet_sources_are_allowed` |
| 6. no word on personal data or decontamination | a reader cannot tell what the corpus may leak or what the evals are worth | `test_datasheet_sections` |
| 7. passing off fetched weights as your own, or quoting upstream numbers you did not reproduce | a card that misleads about provenance | `test_third_party_model_labelled` |
| 8. a card without the model id and version | the gate cannot tell a stale card from the current one | `test_model_card_sections` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ethics.01` | the allowlist the datasheet's sources are checked against |
| Back | `data.08` | `permitted_uses` judges each license the datasheet lists against the allowlist; the ledger and `{corpus} datasheet`'s counts fill the datasheet |
| Back | `ethics.04` | the safety and bias rows, with their intervals |
| Back | `L6.7` | the quality rows (bits per byte with an interval) |
| Forward | `dur.12` | `ModelRelease` refuses a release without `MODEL_CARD.md` |
| Forward | `C1` | the capstone ships with the card and the datasheet (`MS-C1`) |
| Forward | `ethics.05` | the out-of-scope list becomes gateway usage policy rules |
