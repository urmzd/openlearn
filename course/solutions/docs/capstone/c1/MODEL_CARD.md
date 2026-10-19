# Model card: c1-smoke v1

## Model details

- **Developer:** forge, the course's reference system.
- **Architecture:** llama, 4 layers, hidden size 128, 4 heads, 2 KV heads
  (GQA), SwiGLU MLP of 344, vocabulary 512 (byte-level BPE), context 128,
  tied embeddings, 791,680 parameters.
- **Training:** the course's synthetic stories (see the data ledger), 200
  steps of 16 x 128 tokens (409,600 tokens), fp32, AdamW with a WSD
  schedule, seed 0, train spec `specs/c1/smoke.json`.
- **Release:** the smoke tier of the C1 capstone, `models/c1-smoke/v1/`.
- **Third-party components:** none; the tokenizer and weights are trained
  here.
- **License:** Apache-2.0.

## Intended use

- **Primary uses:** a smoke test of the capstone pipeline: data, training,
  evaluation, release, and serving run end to end in minutes.
- **Out of scope:** any real use. The model writes template stories about
  animals and children and nothing else.

## Evaluation

| Suite | Metric | Value (95% CI) | Release threshold |
|---|---|---|---|
| quality | held-out bpb | 0.500 (0.449, 0.597) | at most 0.6 |
| zoo | bpb against KN-4, NPLM, LSTM, GPT | 0.500 vs 0.646, 0.602, 0.496, 0.943 | none |

The numbers come from `docs/capstone/c1/report.json` (32 held-out
documents, seed 0, bootstrap over documents).

## Bias, risks, and limitations

- The training text is synthetic and repetitive; the model has seen no
  real people, places, or harmful content, and it has no refusal behavior
  of its own.
- At this size and training length the scaling fit is flat (alpha 0.004):
  the smoke tier proves the pipeline, not the model.

## Data

One source, `course-stories` (Apache-2.0), allowed for train and eval; no
PII by construction. The held-out documents are the last 32 of the file
and are never trained on.
