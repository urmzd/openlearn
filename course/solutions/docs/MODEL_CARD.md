# Model card: shakespeare-kn4 v0

This is the course's reference example of `ethics.03`: a model small enough
to train and evaluate in seconds, carded the way the capstone (`C1`) will be.
Every number below was measured by `course/oracle/ethics.03/reference_card_numbers.py`
at seed 0.

## Model details

- **Developer:** the course maintainers, as part of the reference system forge.
- **Architecture:** a byte-level interpolated Kneser-Ney 4-gram language model
  (`tl_arch = ngram`, `L2.1`): vocabulary of 256 byte values, context of 3
  bytes, modified discounts estimated from the counts, no neural parameters.
- **Training:** the first 90% (1,003,854 bytes) of the tinyshakespeare text
  described in [DATASHEET.md](DATASHEET.md), one counting pass, seed 0; no
  precision choices apply because the model is counts.
- **Release:** 2026-10-09, `models/shakespeare-kn4/v0/`, written by
  `NGramLM.save` as one safetensors file.
- **Third-party components:** none. The agent model this system also serves,
  SmolLM2-135M-Instruct, has its own card in
  [models/smollm2-135m-instruct.md](models/smollm2-135m-instruct.md).
- **License:** Apache-2.0.

## Intended use

- **Primary uses:** a teaching baseline for the model zoo and the release
  pipeline: it shows what bits per byte a model with no learned
  representations reaches, and it exercises every release gate end to end.
- **Out of scope:** any user-facing text generation. It produces plausible
  letter sequences without meaning, follows no instructions, and cannot
  refuse anything; the gateway's usage policy (ethics.05) must route no user
  traffic to it.

## Evaluation

| Suite | Metric | Value (95% CI) | Release threshold |
|---|---|---|---|
| quality | bpb on the held-out 10% (first 20,000 bytes) | 2.395 (2.366, 2.424) | upper bound at most 2.6 |
| safety | toxicity-rate over 18 prompts | 0.000 (0.000, 0.176) | upper bound at most 0.2 |
| safety | refusal-rate over 8 prompts that should be refused | 0.000 (0.000, 0.324) | not gated: the model cannot follow or refuse instructions |
| safety | over-refusal-rate over 10 benign prompts | 0.000 (0.000, 0.278) | upper bound at most 0.3 |
| bias | bias-gap:gender, 32 paired probes, nats | 1.503 (0.583, 2.430) | reported, not gated |
| bias | bias-gap:age, 12 paired probes, nats | -1.282 (-2.449, 0.061) | reported, not gated |
| bias | stereotype-preference over 12 pairs | 0.500 (0.254, 0.746) | upper bound at most 0.8 |

The bpb interval is a t interval over per-byte negative log-likelihoods; the
rates carry Wilson intervals; the gaps carry percentile bootstrap intervals
over paired differences (1,000 resamples). The safety suite sampled 80 bytes
per prompt from the model with one seeded PCG32 stream. The lexicon scorer
behind the toxicity rate has ROC-AUC 0.752 on the 30 labelled texts of the
ethics.04 fixture, so a rate of zero here is weak evidence on its own.

## Bias, risks, and limitations

- The gender gap of 1.503 nats (0.583, 2.430) favours the first term of each
  pair (he, the boy, grandpa, the man). Those terms are shorter by one byte
  on average, and this model spends about 1.66 nats per byte, so length alone
  predicts a gap of that size: the measurement does not separate a learned
  association from string length, and we report it as such.
- The age gap of -1.282 nats (-2.449, 0.061) favours the longer "old" terms;
  its interval includes zero.
- Stereotype preference is 0.500 (0.254, 0.746): no measurable preference,
  with an interval too wide to rule one out at 12 pairs.
- Safety rates are all zero because the model emits Shakespeare-like letter
  strings, not answers; that says nothing about harmful content in its
  training text, which includes violence and archaic insults.
- Limitations of size and data: 1 MB of early modern English drama, so it
  knows no modern vocabulary, no other language, and nothing after 1616.

## Data

The training and evaluation text is tinyshakespeare, documented in
[DATASHEET.md](DATASHEET.md): one source, a public domain text distributed
in the karpathy/char-rnn repository under MIT, which our allowlist permits
for train and eval. The text has no personal data about living people: the
data.05 PII detector finds no span in it. The evaluation split is the last 10% of
the same file, a different stretch of the plays, and no ethics.04 probe
appears in it.
