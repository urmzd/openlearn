# Multimodal consent policy

## Biometric data
Voice and face inputs are handled as biometric data. We collect neither
speaker identities nor face templates. Authored transcript strings contain
no human recordings and are recorded as text, not biometric audio.

## Prohibited uses
All sources prohibit speaker-identification and face-recognition. A model
embedding must never be repurposed as an identity database or lookup key.

## Source terms
local:evals/asr-recorded/predictions.json is authored synthetic CC0 data.
No Common Voice files are distributed. Anyone using that optional source
must record its dataset terms, consent, attribution, and redistribution limits
before processing it. This fixture does not authorize that use.

## Retention and deletion
Keep only aggregate evaluation reports in published cards. Remove raw media,
encoder caches and derived artifacts when a source is revoked. Record the
revocation in the append-only ledger and exclude the source on the next run.
