# Model card: c2-reference-smoke v1

## Model details

- **Developer:** TinyLLM course reference learner.
- **Base checkpoint:** `artifacts/c1/model`.
- **Training objective:** SFT, as recorded in `artifacts/c2/run.json`.
- **Dataset revision:** `post-training-fixtures-v1`; the manifest records the training and held-out fixture paths, seed, tokenizer, template, and optimizer.
- **Artifact status:** provenance smoke artifact only. It does not include trained weights or a measured improvement claim.

## Intended use

- **Primary use:** demonstrate the post-training artifact and provenance fields expected by the C2 workflow.
- **Out of scope:** deployment or claims about general language quality, safety, or alignment.

## Evaluation

The workflow pairs the candidate and C1 base on `course/fixtures/post-training/heldout.jsonl`, using paired reward differences, a 95% paired bootstrap interval, and a two-sided paired permutation test. This reference smoke artifact records the evaluation plan, not measured results.

## Limitations

The small course fixture does not establish a reliable improvement or broad behavior change. The manifest makes no measured held-out improvement or general safety claim.
