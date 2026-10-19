# Multimodal evidence card

## Multimodal data
Four authored symbolic transcript pairs are CC0 fixtures. No human voices,
faces, speaker identities or accent groups occur in this data.

## Multimodal evaluation
The following numbers are recomputed from recorded predictions by the evidence
program. They demonstrate the arithmetic and report provenance, not a trained
speech recognizer's quality. WER is pooled over words; each group has two items.

```json
{
  "n": 4,
  "predictions_sha256": "6d9e809ebec9cc03d7e60487dee1af23ff90d37b07284c39039c5f20d72def2e",
  "wer": {
    "overall": 0.375,
    "clear": 0.25,
    "noisy": 0.5
  }
}
```

## Limitations
Synthetic clear/noisy labels cannot establish demographic fairness or real
noise robustness. Tower diagnostics use tiny fixtures and teacher-forced token
scores, not generated transcripts. No production quality claim is made.
