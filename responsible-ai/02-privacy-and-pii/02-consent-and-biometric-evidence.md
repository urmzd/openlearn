<!-- ol:module ethics.08 -->
# Consent and biometric evidence in model cards

## Overview

| | |
|---|---|
| **Module** | `ethics.08` · practice · python and docs · Pass 12 |
| **You build** | A consent policy, multimodal source ledger, runnable evidence program, recorded-prediction report and model-card supplement |
| **Contract** | Section 4 specifies the artifacts and evidence functions; the ledger extends `formats/ledger.schema.json` |
| **Tests** | `course/tests/ethics.08/check` recomputes reports and runs actual tiny image and Whisper towers |
| **Needs** | `L13.5`, `L14.3`, `L14.4`, `L14.1`, `L14.5`, `data.08`; reading: `ethics.02`, `ethics.03` |
| **Used by** | A terminal artifact practice; future multimodal release automation will consume these records |
| **Milestone** | `MS-P12` |

## Key Takeaways

- A license and a consent decision are separate records. Never infer permission for identification from access to a recording.
- An aggregate WER weights utterances by reference words, not equally.
- A tiny model's teacher-forced next-token accuracy is a diagnostic, not transcription quality.
- A model card must preserve the provenance and all groups of its underlying report.

## How to work this chapter

```bash
ol start ethics.08
ol tests ethics.08
PYTHONPATH=python python primers/ethics.08/evidence.py --root .
ol check ethics.08
```

## 1. Why now

The vision and speech towers now accept encoded media. A tensor-level test alone
cannot establish that an uploaded image passed through the right normalization
or that stereo audio passed through channel mixing and resampling. Before making
quality claims, run those complete paths and document exactly what was measured.
Consent records govern the media used to obtain those measurements.

This practice can run before the transcription and VLM policies exist. It reports
recorded transcript predictions separately from teacher-forced Whisper scores.
That distinction keeps a useful foundation check from masquerading as an ASR
benchmark. Production transcription and demographic gap analysis remain later work.

## 2. Principles

The ledger adds optional fields for legacy text compatibility: modality,
biometric, consent, attribution and prohibited_uses. This practice requires all
five on every evaluated source. The consent categories are none, public-domain,
cc-licensed and explicit-donation. A biometric source with none is rejected;
a CC-licensed source needs attribution. Voice is conservatively biometric.
Both speaker-identification and face-recognition must be prohibited.

Validate every historical row's shape, then apply permission checks to the latest
row per source. A later revocation overrides an older grant. A corrected later
record can supersede a revocation, but the record must represent an actual new
permission decision, not an automatic retry.

For each transcript, count substitutions, deletions and insertions after basic
normalization. Divide pooled edits by pooled reference words, including when
reporting each group. Preserve a digest of the complete recorded prediction data
so a card cannot silently refer to another run. A digest establishes identity,
not that the predictions are truthful or that their source was authorized.

The image diagnostic runs prompt ensembling and the raw-image tower before
computing top-1 accuracy. The speech diagnostic encodes a WAV, scores a supplied
prefix, and reports next-token accuracy and negative log probability. Neither
function changes model weights or claims to train a model.

## 3. Worked example

The reference practice uses four synthetic two-word transcripts. Their edit
counts are 0, 1, 2 and 0, so overall WER is 3/8 = 0.375. The first two belong to
clear and have WER 1/4 = 0.25; the last two belong to noisy and have WER 2/4 = 0.5.
These are authored group names, not measurements of real noise or human groups.
`test_artifact_numbers_are_recomputed` verifies the report and card from the
recorded predictions. With reference lengths 3 and 1 and one error in each,
pooled WER is 2/4, not the mean of 1/3 and 1.

## 4. Build and verify the evidence

Write these files in your learner repository:

| Artifact | Required contents |
|---|---|
| `primers/ethics.08/evidence.py` | transcript_report, image_diagnostic, speech_diagnostic, validate_consent, validate_card, and a main entry point accepting --root |
| `docs/data/CONSENT_POLICY.md` | Biometric data, Prohibited uses, Source terms, Retention and deletion sections; name every ledger source URL |
| `docs/data/mm-ledger.jsonl` | Complete ledger rows plus the five multimodal fields; no invented permissions |
| `evals/asr-recorded/predictions.json` | List of unique id, group, reference and hypothesis string records; hypotheses may be empty |
| `evals/asr-recorded/summary.json` | Recomputed suite, count, WER, CER, per-group counts and WER, predictions_sha256 |
| `docs/models/multimodal.md` | Multimodal data, Multimodal evaluation, Limitations; one JSON block with n, predictions_sha256 and wer mapping containing overall and every group |

The summary suite is asr-recorded. The canonical prediction digest is SHA-256 of
UTF-8 JSON with sorted keys, compact separators, and ensure_ascii=False. Card WER
values may be rounded to four decimal places, with at most 0.00005 absolute error.
The evidence program checks consent before writing the report. Its tower helpers
accept loaded learner models and return measurements; do not substitute constant
scores. The reference artifacts use synthetic strings and make no production
quality claim. Use actual model predictions and source terms for your own card.

| Test | What it proves |
|---|---|
| `test_artifact_numbers_are_recomputed` | The worked example's report, digest and card agree; stale evidence fails |
| `test_hand_example_pooled_error_rates` | Word weighting, normalization, empty hypotheses and unique records |
| `test_bad_consent_cannot_produce_evidence` | Invalid, unlicensed, revoked and identity-permitting records fail |
| `test_append_only_revocation_uses_latest_record` | Current permission follows the latest source record |
| `test_actual_image_towers_feed_zero_shot_evidence` | Real pixel and prompt towers feed the reported accuracy |
| `test_actual_whisper_logits_feed_named_diagnostic` | Real audio-conditioned logits feed a correctly labelled diagnostic |
| `test_report_command_checks_consent_before_writing` | The command regenerates evidence and refuses revoked data |

## 5. Pitfalls

| Mistake | Consequence | Test |
|---|---|---|
| Average utterance WERs | Short utterances dominate | `test_hand_example_pooled_error_rates` |
| Read the first consent record | Revocations disappear | `test_append_only_revocation_uses_latest_record` |
| Copy card numbers manually | Results drift without changing the card | `test_artifact_numbers_are_recomputed` |
| Call teacher-forced scores WER | Readers infer transcription quality that was never measured | `test_actual_whisper_logits_feed_named_diagnostic` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `L13.5` | Prompt classifier and complete image-tower inference |
| Back | `L14.3` | WAV-to-cross-attention encoding and next-token logits |
| Back | `L14.5` | Normalization, pooled WER, CER and zero-shot accuracy |
| Back | `data.08` | Ledger shapes, latest records and licensed uses |
| Back | `ethics.02` | Privacy policy structure and retention decisions |
| Back | `ethics.03` | Evidence-backed model cards |

Later release automation can consume the artifacts after adding its own
provenance, consent and quality gates. This practice does not implement that
release workflow or the unbuilt transcription and VLM policies.

| Back | `L14.4` | Generate transcripts before computing WER and CER. |
| Back | `L14.1` | Decode and resample consent-bound WAV evidence. |

`transcribed_report` validates the current consent ledger and binds every WAV
byte string to its source SHA-256 before calling the model. It returns generated
predictions plus a recomputed `asr-generated` report; it does not relabel the
teacher-forced diagnostic as WER. A changed or revoked recording fails before
inference. The synthetic tests establish plumbing, not trained model quality.

| Test | Kind | Contract |
|---|---|---|
| `test_generated_transcripts_bind_audio_consent_before_inference` | conformance | Actual transcription supplies metric inputs; changed and revoked audio never reach the model. |
