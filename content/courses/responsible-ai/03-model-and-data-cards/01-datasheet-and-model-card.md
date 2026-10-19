# Datasheet and model card

The first release of the system ends with a gate. `ModelRelease`, in [ModelRelease: gates, canary, burn check, and rollback](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) from *AI Platform Engineering*, exports the capstone to `models/<id>/<version>/`, runs `EvalSuite`, and refuses to promote it without a `MODEL_CARD.md` next to the weights. Everything a reader needs to decide whether to use the model exists by now, but it is scattered: the ledger and datasheet counts from [Licensing ledger verification and the datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md), the license decisions from [Data licensing and the ledger](../01-data-licensing/01-data-licensing-and-the-ledger.md), the quality numbers from [The evaluation harness and the model zoo](../../build-an-llm/p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md), the safety and bias rows from [Bias and safety evals](../04-bias-and-safety-evals/01-bias-and-safety-evals.md), and the training recipe from [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](../../build-an-llm/p11-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md).

Documentation is how a model's limits travel with it. Without a card, each downstream user, including the project itself when it later decides whether the agent may call this model, rediscovers the model's failure modes in production; with one, the limits are stated, measured, and versioned next to the weights. And the system soon adds a model the project did not train, SmolLM2-135M-Instruct. A card that blurs which weights were trained here and which were fetched misleads everyone who reads it.

This section writes three documents and the checks on them: the reference model card [`course/solutions/docs/MODEL_CARD.md`](../../../../course/solutions/docs/MODEL_CARD.md) (`shakespeare-kn4 v0`), its corpus's datasheet [`course/solutions/docs/DATASHEET.md`](../../../../course/solutions/docs/DATASHEET.md), and the third-party card [`course/solutions/docs/models/smollm2-135m-instruct.md`](../../../../course/solutions/docs/models/smollm2-135m-instruct.md), labelled as such. The first two start from the templates [`templates/MODEL_CARD.md`](../../../../course/contracts/templates/MODEL_CARD.md) and [`templates/DATASHEET.md`](../../../../course/contracts/templates/DATASHEET.md), and the release layout is in [`formats/checkpoint.md`](../../../../course/contracts/formats/checkpoint.md). The datasheet's licenses are judged by `permitted_uses` in [`python/corpus/ledger.py`](../../../../course/solutions/python/corpus/ledger.py). Keep one limit in mind throughout: a check can tell that a card is complete, specific, and consistent with the other records. It cannot tell that the card is true; that part is the author's.

> **Learning objectives**
> - Write a model card that leads with what the model is for and not for, with an out-of-scope list a gateway rule can enforce.
> - Report every evaluation number with its interval and the threshold it was held to, and explain what a Wilson upper bound says about a zero rate.
> - Write limitations that start from measurements and name their confounds.
> - Write a datasheet that is the ledger in prose, with licenses consistent with the allowlist.
> - Label a third-party model honestly, with a pinned upstream revision and only evaluations the project ran.

## What a model card is for

A model card is the model's interface documentation: what it is, what it is for and not for, how it was evaluated with what uncertainty, what it gets wrong, and what it learned from. Mitchell et al. proposed a set of fixed questions (model details, intended use, factors, metrics, evaluation data, training data, quantitative analyses, ethical considerations, caveats), and the system's template keeps the essential ones: model details (who, what architecture, how trained, which release, under what license), intended use and out-of-scope uses, evaluation, bias and risks, and the data. The order matters. A reader deciding whether to use the model reads the first two sections and stops if the answer is no.

**Out of scope is the most important list.** It names the uses the model must not serve: for a 10M-parameter story model, any factual question, any advice, any user who is not part of the project. Later, [Usage policy at the gateway](../05-usage-policy/01-usage-policy-at-the-gateway.md) turns this list into rules of the gateway's usage policy, so it is written as something a rule can enforce ("no user-facing traffic", "only the story-completion route"), not as a hope.

**Numbers carry their uncertainty.** A bits-per-byte figure without an interval cannot be compared with the next release's. The evaluation table writes every value as `value (lo, hi)`: the t interval of [LLN, CLT, confidence intervals, bootstrap](../../mathematics-for-computing/07-probability-statistics/04-lln-clt-confidence-intervals-bootstrap.md) in *Mathematics for Computing* for a mean loss, the Wilson interval for a rate, the bootstrap interval for a bias gap, and the threshold the release gate held it to. A row that is reported but not gated says so, and why.

**Limitations cite measurements.** "The model may reflect biases in its training data" is true of every model and tells the reader nothing. Write what was measured, with its interval, and what it does and does not show. A gender gap of 1.5 nats whose interval excludes zero, which string length alone could explain, is a limitation a reader can act on; the bias section cites measurements, not adjectives.

## What a datasheet is for

The model card points at its data; the datasheet describes it. A datasheet answers why the data exists, what is in it, where it came from, and what was done to it. Gebru et al.'s questions, kept by the template, are motivation; composition (what one document is, how many, which splits, what personal data); collection (every source with its URL, license, retrieval date, and sha256, from the ledger); preprocessing (each filter and how many documents it dropped, deduplication, decontamination, tokenization); uses; and distribution and maintenance (where the files live and what happens when a source's license is revoked, the procedure rehearsed in [A data incident](../../systems/05-incident-response-and-chaos/08-data-incident.md)). The `{corpus} datasheet` command fills the counts; the prose is the author's.

**Licenses are consistent across records.** The datasheet's sources, the ledger's rows, and the licensing allowlist must agree: a source whose SPDX identifier the allowlist does not permit for `train` cannot be in a released model.

## Third-party models and versioning

Two more rules keep cards honest over time. **Third-party models are labelled.** The agent model is HuggingFaceTB's SmolLM2-135M-Instruct, fetched at a pinned upstream revision and checked against a recorded checksum. Its card states plainly that the project did not train it, names the upstream repository and revision, keeps the upstream license (Apache-2.0), describes the architecture from its own `config.json`, and reports only evaluations the project ran; until the system runs them, it says they have not been run, rather than quoting numbers nobody reproduced.

**A card is versioned with the weights.** The release copies `docs/MODEL_CARD.md` into `models/<id>/<version>/` (`formats/checkpoint.md`); the title names the model id and version so the gate can tell a stale card from the current one. Cards are generated where possible and checked always: corpus statistics and the license summary come from the ledger, eval tables from the `EvalSuite` report, and `ModelRelease` checks that the card exists and that its eval section matches the release candidate.

## Working it by hand

The reference card describes a model that rebuilds in a second: a byte-level Kneser-Ney 4-gram, from [n-gram language model with interpolated modified Kneser-Ney](../../build-an-llm/p02-statistical-lm/01-ngram-kneser-ney.md), trained on the first 90% of the tinyshakespeare fixture. Its numbers were measured at seed 0 with the suites from [Bias and safety evals](../04-bias-and-safety-evals/01-bias-and-safety-evals.md). Three excerpts show the decisions a card makes. Try to write each one's key sentence yourself before reading it.

First, the evaluation table, every value with its interval and its threshold:

```markdown
| Suite | Metric | Value (95% CI) | Release threshold |
|---|---|---|---|
| quality | bpb on the held-out 10% (first 20,000 bytes) | 2.395 (2.366, 2.424) | upper bound at most 2.6 |
| safety | toxicity-rate over 18 prompts | 0.000 (0.000, 0.176) | upper bound at most 0.2 |
| safety | refusal-rate over 8 prompts that should be refused | 0.000 (0.000, 0.324) | not gated: the model cannot follow or refuse instructions |
| bias | bias-gap:gender, 32 paired probes, nats | 1.503 (0.583, 2.430) | reported, not gated |
```

The toxicity rate of 0 out of 18 has a Wilson upper bound of 0.176: with 18 prompts, "no toxic output" bounds the rate only below about 18%. The refusal row is reported but not gated, and the threshold column says why.

Second, the limitation the bias row supports, written as a measurement:

```markdown
- The gender gap of 1.503 nats (0.583, 2.430) favours the first term of each
  pair (he, the boy, grandpa, the man). Those terms are shorter by one byte
  on average, and this model spends about 1.66 nats per byte, so length alone
  predicts a gap of that size: the measurement does not separate a learned
  association from string length, and we report it as such.
```

The 1.66 nats per byte is the measured 2.395 bits per byte times $\ln 2$; the average length difference of the four pairs is $(1 + 1 + 0 + 2)/4 = 1$ byte.

Third, one source in the datasheet's Collection section, with everything the ledger records:

```markdown
- Source `tinyshakespeare`: the file `data/tinyshakespeare/input.txt` of the
  repository github.com/karpathy/char-rnn at revision
  6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e, license MIT (the repository; the
  plays themselves are in the public domain), retrieved 2026-10-09,
  sha256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
```

MIT is in the reference allowlist with `uses = ["train", "eval"]`, so the source is allowed for training. The capstone's datasheet lists TinyStories (CDLA-Sharing-1.0) instead.

## The three documents

With those decisions in hand, the shape of the solution is fixed. It is three files:

| File | From | Sections (`##`) |
|---|---|---|
| `docs/MODEL_CARD.md` | `contracts/templates/MODEL_CARD.md`, title `# Model card: <model_id> v<N>` | Model details · Intended use · Evaluation · Bias, risks, and limitations · Data |
| `docs/DATASHEET.md` | `contracts/templates/DATASHEET.md`, title `# Datasheet: <dataset> <version>` | Motivation · Composition · Collection · Preprocessing · Uses · Distribution and maintenance |
| `docs/models/smollm2-135m-instruct.md` | this table, title `# Third-party model card: SmolLM2-135M-Instruct` | Provenance · License · Intended use · Evaluation · Limitations |

In the model card, Model details keeps the template's six bullets (`- **Developer:** ...`, Architecture, Training, Release, Third-party components, License with an SPDX id). Intended use keeps `- **Primary uses:**` and `- **Out of scope:**`. Evaluation keeps the table `| Suite | Metric | Value (95% CI) | Release threshold |` with at least a quality row (bpb, perplexity, or loss), a `safety` row, and a `bias` row, each value written `value (lo, hi)`. Data links `[DATASHEET.md](DATASHEET.md)` and names the data's license. No angle-bracket placeholder of the template survives. Every source license in the datasheet is judged by `permitted_uses` against the allowlist at `docs/data/license-allowlist.toml`.

## Writing the documents step by step

Seven principles drive every sentence of the reference documents.

**Write for the reader who must decide.** A card is read by someone choosing whether to use the model, so it leads with what the model is and what it is for and not for; a reader who sees "out of scope: any user-facing text generation" can stop there.

**Make the out-of-scope list enforceable.** The usage policy turns it into gateway rules, so each item is a route or a traffic class ("no user traffic"), not a sentiment.

**Every number with its interval and its gate.** Without an interval, two releases cannot be compared and a lucky seed looks like progress. Without a threshold column, a reader cannot tell a gated number from a decorative one. Rows that are reported but not gated say why.

**Limitations follow from the measurements.** Each bullet starts from a number in the evaluation table and says what it does and does not show, including the confound that could explain it.

**The datasheet is the ledger in prose.** Every source appears with URL, revision, license, retrieval date, and sha256, exactly the ledger fields, and every preprocessing step with how much it removed. Its licenses must be ones the allowlist permits for `train`.

**Provenance is never blurred.** A model the project did not train gets its own card that says so in the first sentence, pins the upstream revision and checksum, keeps the upstream license, and reports only evaluations the project ran.

**A card is versioned with the weights.** The title names the model id and version, and the release copies the card into the version directory, so a stale card is detectable.

### The model card, annotated

The reference card describes `shakespeare-kn4 v0`, the 4-gram of the worked example. The title encodes versioning:

```markdown
# Model card: shakespeare-kn4 v0
```

Model details answers who, what, how, which release, what else is inside, and under what terms; the third-party bullet is where the card points away from itself:

```markdown
- **Architecture:** a byte-level interpolated Kneser-Ney 4-gram language model
  (`tl_arch = ngram`, `L2.1`): vocabulary of 256 byte values, context of 3
  bytes, modified discounts estimated from the counts, no neural parameters.
- **Release:** 2026-10-09, `models/shakespeare-kn4/v0/`, written by
  `NGramLM.save` as one safetensors file.
- **Third-party components:** none. The agent model this system also serves,
  SmolLM2-135M-Instruct, has its own card in
  [models/smollm2-135m-instruct.md](models/smollm2-135m-instruct.md).
- **License:** Apache-2.0.
```

Intended use comes next, with an out-of-scope list a rule can enforce:

```markdown
- **Primary uses:** a teaching baseline for the model zoo and the release
  pipeline: it shows what bits per byte a model with no learned
  representations reaches, and it exercises every release gate end to end.
- **Out of scope:** any user-facing text generation. It produces plausible
  letter sequences without meaning, follows no instructions, and cannot
  refuse anything; the gateway's usage policy (ethics.05) must route no user
  traffic to it.
```

The evaluation table extends the worked example's excerpt with three more rows (over-refusal, the age gap, stereotype preference) and is followed by a paragraph that states how each interval was computed and how much to trust the scorer:

```markdown
The bpb interval is a t interval over per-byte negative log-likelihoods; the
rates carry Wilson intervals; the gaps carry percentile bootstrap intervals
over paired differences (1,000 resamples). [...] The lexicon scorer
behind the toxicity rate has ROC-AUC 0.752 on the 30 labelled texts of the
ethics.04 fixture, so a rate of zero here is weak evidence on its own.
```

That last sentence applies the interval principle to the instrument itself: a rate is only as good as the classifier that produced it. The limitations then walk the table row by row: the gender gap with its length confound, the age gap whose interval includes zero, a stereotype preference of 0.500 with an interval too wide to rule anything out at 12 pairs, safety rates of zero that say nothing about the training text, and the plain limits of 1 MB of early modern drama. The full card is [`MODEL_CARD.md`](../../../../course/solutions/docs/MODEL_CARD.md).

### The datasheet, annotated

```markdown
## Composition

- One document: 1,115,394 bytes of plain ASCII text in 40,001 lines (32,777
  of them non-empty) of dialogue from Shakespeare's plays, each speech headed
  by the speaker's name.
- Splits by position: the first 90% (1,003,854 bytes) for training, the
  last 10% (111,540 bytes) for evaluation. A position split leaks style
  across the boundary but no lines are duplicated between the halves.
- Personal data: none about living people. The data.05 PII detector finds
  no span in the file [...]
```

Composition says what a document is, how many there are, how the splits were made and what that choice costs, and what personal data was looked for and found. Preprocessing records every step, including the ones deliberately not taken and why:

```markdown
- Exact deduplication by line would drop 7,056 of the 32,777 non-empty lines
  (21.5%), mostly repeated speaker headings such as "ROMEO:"; they are kept
  because they are the structure of the text. No near deduplication.
- Decontamination: none of the 130 ethics.04 probe sentences and red-team
  prompts occurs in the text (exact substring search).
```

A reader can now judge what the evaluation numbers are worth: the probes did not leak into training. Uses states the allowlist decision (MIT allows `train` and `eval`) and what to avoid; Distribution and maintenance says where the file lives and applies the data-incident revocation procedure if the upstream license changed. The full datasheet is [`DATASHEET.md`](../../../../course/solutions/docs/DATASHEET.md).

### The third-party card, annotated

```markdown
# Third-party model card: SmolLM2-135M-Instruct

We did not train or fine-tune this model. It is third-party weights,
fetched at a pinned revision and served by our engine for the Pass 10 agent.

## Provenance

- **Upstream:** Hugging Face model `HuggingFaceTB/SmolLM2-135M-Instruct`,
  revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, by the Hugging Face
  SmolLM team; an instruction-tuned member of the SmolLM2 family.
- **Architecture (from its config.json):** `LlamaForCausalLM`, 30 layers,
  hidden size 576, 9 attention heads with 3 KV heads, MLP width 1536,
  vocabulary 49,152, context 8,192, tied embeddings, bfloat16 weights;
  134,515,008 parameters by M05.1's param_count.

## Evaluation

- Not yet evaluated by this system. [...] Upstream evaluations exist but
  were not reproduced by us, so we do not quote them.
```

The first sentence settles provenance. The revision is a full commit hash, not a branch name, because a branch moves; the reference card also records the `model.safetensors` sha256. The architecture comes from the model's own `config.json` and the parameter count from `param_count` in [Counting: parameters, FLOPs, KV bytes, and memory plans](../../mathematics-for-computing/05-discrete-math-1/01-counting-params-flops-kv-bytes.md), so both are verified facts rather than copied claims. The Evaluation section is honest about absence: no numbers rather than unreproduced ones. The full card is [`smollm2-135m-instruct.md`](../../../../course/solutions/docs/models/smollm2-135m-instruct.md).

### Why this design

Templates beat free-form documents because fixed sections make cards comparable across releases and let a program confirm completeness; the prose inside each section still carries the judgment. Intervals beat point estimates: with 18 prompts, "0 toxic outputs" bounds the rate only below 17.6%, and a point estimate of 0 would claim far more than was measured. The interval is the honest size of the evidence. Measured limitations beat generic disclaimers because a disclaimer true of every model cannot regress, while a measured gap with an interval can be tracked release over release and can trigger action.

The design accepts tradeoffs. A check can confirm that a card is complete, specific, and consistent with the ledger and allowlist, but not that it is true; the cards therefore cite where each number came from so a reader can reproduce it. Reporting a confound (string length) instead of claiming a bias finding makes the card less dramatic and more accurate.

## What can go wrong

A correct solution handles each of these cases:

| Case | Checks | Why it matters downstream |
|---|---|---|
| model card sections and title | the five sections with enough prose, and the title with the model id and version | the release gate finds the right card |
| no template placeholders | no `<placeholder>` or TODO left in any of the three files | a copied template is not a card |
| model details fields | the six detail bullets, an SPDX license | who, what, and under what terms |
| intended and out-of-scope uses | both lists, specific enough to enforce | the usage policy's rules |
| evaluation table with intervals | quality, safety, and bias rows, each `value (lo, hi)` with a threshold | releases can be compared |
| limitations cite measurements | a measured interval, bias and safety both discussed | limitations a reader can act on |
| the card links the datasheet | a link to `DATASHEET.md` and the data's license | the model traces to its sources |
| datasheet sections | Gebru's six sections, personal data, deduplication, decontamination | the corpus is documented |
| datasheet sources allowed | every source license is allowed for `train` by the allowlist, judged by `permitted_uses` | the release workflow's license gate |
| third-party model labelled | upstream repository, pinned revision, Apache-2.0, "did not train" | honest provenance for the agent model |

> **Warning:** The most common failure is copying the template and filling half of it, which ships a released card containing `<you>`. Every template section must be filled, with no placeholder left.

The other pitfalls mirror the principles. Numbers without intervals mean two releases cannot be compared and a lucky run looks like progress; every row is `value (lo, hi)` with a threshold, and the interval methods are stated. Generic limitations ("may be biased") give nothing to act on and nothing to regress against; each limitation starts from a measured row and names its confound. An out-of-scope list nobody can enforce leaves the usage policy with nothing to encode; the reference says "any user-facing text generation" and "route no user traffic to it".

On the datasheet side, a source under a license the allowlist refuses means the release gate (or a lawyer) stops the release later; the only source here is MIT, allowed for `train`. A datasheet with no word on personal data or decontamination leaves a reader unable to tell what the corpus may leak or what the evals are worth; Composition reports the PII scan and Preprocessing reports the probe search. Passing off fetched weights as the project's own, or quoting upstream numbers nobody reproduced, produces a card that misleads about provenance; the third-party card's first sentence, its pinned revision, and its refusal to quote unreproduced numbers prevent that. And a card without the model id and version leaves the gate unable to tell a stale card from the current one; the reference title is `# Model card: shakespeare-kn4 v0`.

## Where this leads

The documents have no code of their own, but they connect much of the system. Looking back, the datasheet's sources are checked against the allowlist from [Data licensing and the ledger](../01-data-licensing/01-data-licensing-and-the-ledger.md); `permitted_uses` and the `{corpus} datasheet` counts from [Licensing ledger verification and the datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md) fill and judge the datasheet; the safety and bias rows, with their intervals, come from [Bias and safety evals](../04-bias-and-safety-evals/01-bias-and-safety-evals.md); and the quality rows (bits per byte with an interval) come from [The evaluation harness and the model zoo](../../build-an-llm/p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md).

Looking forward, `ModelRelease` in [ModelRelease: gates, canary, burn check, and rollback](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) refuses a release without `MODEL_CARD.md`; the [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) ships with the card, the datasheet, and its data ledger; and [Usage policy at the gateway](../05-usage-policy/01-usage-policy-at-the-gateway.md) turns the out-of-scope list into gateway usage policy rules.

### Further reading

- Mitchell et al., [*Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993) (FAT* 2019).
- Gebru et al., [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010) (CACM 2021).
- Hugging Face, [model card guide](https://huggingface.co/docs/hub/model-cards).
- Pushkarna et al., [*Data Cards*](https://arxiv.org/abs/2204.01075) (2022).

## Key terms

- **Model card**: the model's interface documentation: details, intended and out-of-scope use, evaluation with uncertainty, limitations, and data.
- **Datasheet**: the dataset's documentation: motivation, composition, collection, preprocessing, uses, distribution and maintenance.
- **Out-of-scope list**: the uses a model must not serve, written so a gateway rule can enforce them.
- **Wilson interval**: a confidence interval for a rate that stays meaningful at zero successes, such as 0 of 18 with upper bound 0.176.
- **Confound**: an alternative explanation for a measurement, such as string length for a log-probability gap.
- **Third-party model card**: a card for weights the project did not train, pinning the upstream revision and reporting only evaluations the project ran.

## Summary

- A model card is the model's interface documentation, ordered so a reader can stop at intended and out-of-scope use.
- Every number carries its interval and the threshold it was held to; rows reported but not gated say why.
- Limitations start from measurements and name their confounds, rather than generic disclaimers.
- The datasheet is the ledger in prose, and every source's license must be one the allowlist permits for training.
- A model you serve but did not train gets its own card that says so, with the exact pinned upstream revision.
- Cards are generated where possible, checked always, and versioned with the weights so the release gate can find the right one.

## Review questions

1. Why does the toxicity row report 0.000 with an interval up to 0.176 rather than simply 0?

   <details><summary>Answer</summary>

   Zero toxic outputs in 18 prompts is consistent with a true rate anywhere from 0 to about 18%: the Wilson interval for 0 of 18 has upper bound 0.176. Reporting 0 alone would claim the model never produces toxic text, which 18 samples cannot show. The interval also makes the gate meaningful: the release threshold is on the upper bound (at most 0.2), so a larger suite is needed to certify a lower rate.

   </details>

2. The gender gap's interval excludes zero. Why does the card not call it a learned gender bias?

   <details><summary>Answer</summary>

   The first term of each pair is on average one byte shorter, and the model spends about 1.66 nats per byte (2.395 bits per byte times $\ln 2$), so the length difference alone predicts a gap of about the measured 1.5 nats. The measurement cannot separate a learned association from string length, and the card says exactly that, which is what a reader needs to act on it.

   </details>

3. Why is the refusal-rate row reported but not gated?

   <details><summary>Answer</summary>

   A byte-level 4-gram cannot follow or refuse instructions, so its refusal rate is 0 by construction and says nothing about safety; gating on it would fail every release for a property the model cannot have. Reporting it keeps the suite's shape consistent across models, and the threshold column states why it is not gated.

   </details>

4. Why does the third-party card refuse to quote upstream evaluation numbers?

   <details><summary>Answer</summary>

   The card reports what this system measured. Upstream numbers come from a different harness, prompts and settings, and the project has not reproduced them; quoting them would present someone else's claims as the project's evidence. The card says the evaluations have not been run yet and that the rows will be added with intervals when the system runs them.

   </details>

5. How do the datasheet and the licensing allowlist constrain each other?

   <details><summary>Answer</summary>

   Every source the datasheet lists must have a license the allowlist permits for `train`, judged by `permitted_uses`. tinyshakespeare is MIT, allowed for `train` and `eval`, so it can be in the released model. A source under a refused license would block the release at the release workflow's gate, so the inconsistency is caught before publication rather than by a reader or a lawyer later.

   </details>
