# Consent and biometric evidence in model cards

## Overview

| | |
|---|---|
| **Module** | `ethics.08` · practice · python and docs · Pass 12 |
| **Solution** | the evidence program [`course/solutions/primers/ethics.08/evidence.py`](../../../../course/solutions/primers/ethics.08/evidence.py) (`transcript_report`, `transcribed_report`, `image_diagnostic`, `speech_diagnostic`, `validate_consent`, `validate_card`, `main`); the consent policy [`CONSENT_POLICY.md`](../../../../course/solutions/docs/data/CONSENT_POLICY.md); the multimodal ledger [`mm-ledger.jsonl`](../../../../course/solutions/docs/data/mm-ledger.jsonl); the recorded predictions and report in [`entry/evals/asr-recorded/`](../../../../course/solutions/entry/evals/asr-recorded/); and the model-card supplement [`multimodal.md`](../../../../course/solutions/docs/models/multimodal.md) |
| **Contract** | Section 4 specifies the artifacts and evidence functions; the ledger extends [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json) |
| **Needs** | `L13.5`, `L14.3`, `L14.4`, `L14.1`, `L14.5`, `data.08`; reading: `ethics.02`, `ethics.03` |
| **Used by** | A terminal artifact practice; future multimodal release automation will consume these records |

## Key Takeaways

- A license and a consent decision are separate records. Never infer permission for identification from access to a recording.
- An aggregate WER weights utterances by reference words, not equally.
- A tiny model's teacher-forced next-token accuracy is a diagnostic, not transcription quality.
- A model card must preserve the provenance and all groups of its underlying report.

## How to read this chapter

Sections 1 to 3 explain what consent records add to a license, how pooled error rates are computed, and work the reference numbers by hand. Section 4 fixes the artifacts and what each must contain. Section 5 walks through the evidence program function by function and explains why each check is where it is. Try to predict, for each check, which bad record or stale card it exists to stop.

The solution's entry point regenerates the report from the recorded predictions, after checking consent:

```bash
PYTHONPATH=python python primers/ethics.08/evidence.py --root .
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

The reference practice uses four synthetic two-word transcripts:

| id | group | reference | hypothesis | edits |
|---|---|---|---|---|
| 1 | clear | `a b` | `a b` | 0 |
| 2 | clear | `c d` | `c x` | 1 (substitution) |
| 3 | noisy | `e f` | (empty) | 2 (deletions) |
| 4 | noisy | `g h` | `g h` | 0 |

Their edit counts are 0, 1, 2 and 0, so overall WER is 3/8 = 0.375. The first two
belong to clear and have WER 1/4 = 0.25; the last two belong to noisy and have WER
2/4 = 0.5. These are authored group names, not measurements of real noise or human
groups. The recorded report and the card must both carry exactly these numbers,
recomputed from the predictions. With reference lengths 3 and 1 and one error in
each, pooled WER is 2/4, not the mean of 1/3 and 1.

## 4. The evidence artifacts

The solution consists of these files:

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
accept loaded models and return measurements; they never substitute constant
scores. The reference artifacts use synthetic strings and make no production
quality claim. A real card needs actual model predictions and source terms.

`transcribed_report` validates the current consent ledger and binds every WAV
byte string to its source SHA-256 before calling the model. It returns generated
predictions plus a recomputed `asr-generated` report; it does not relabel the
teacher-forced diagnostic as WER. A changed or revoked recording fails before
inference.

### What a correct solution must handle

| Case | What it proves |
|---|---|
| the artifact numbers recomputed | The worked example's report, digest and card agree; stale evidence fails |
| the hand example's pooled error rates | Word weighting, normalization, empty hypotheses and unique records |
| bad consent cannot produce evidence | Invalid, unlicensed, revoked and identity-permitting records fail |
| append-only revocation uses the latest record | Current permission follows the latest source record |
| actual image towers feed zero-shot evidence | Real pixel and prompt towers feed the reported accuracy |
| actual Whisper logits feed a named diagnostic | Real audio-conditioned logits feed a correctly labelled diagnostic |
| the report command checks consent before writing | The command regenerates evidence and refuses revoked data |
| generated transcripts bind audio consent before inference | Actual transcription supplies metric inputs; changed and revoked audio never reach the model |

## 5. The solution, step by step

### From first principles

1. **Permission comes before measurement.** A number computed from media nobody was allowed to use should not exist, so every path that produces evidence (`main`, `transcribed_report`) calls `validate_consent` first and writes nothing if it fails.
2. **Shape is checked on every row; permission on the latest.** The ledger is append-only (as in `ethics.01`), so a malformed historical row is still a bug, but only the latest row per source states the current decision. A revocation appended later therefore wins over an earlier grant.
3. **License and consent are separate questions.** The license says whether the source may be used for `eval`; consent says whether a person agreed to their voice or face being used. Both must pass, and voice is always treated as biometric.
4. **Identity uses are always prohibited.** Every row must list `speaker-identification` and `face-recognition` in `prohibited_uses`, so no embedding built from these sources can be turned into a lookup of who someone is.
5. **Pool edits, do not average rates.** WER is total edits over total reference words, overall and within each group (section 3). Averaging per-utterance rates lets a one-word utterance count as much as a fifty-word one.
6. **Bind the card to the run by a digest.** The report stores a SHA-256 of the canonical JSON of the predictions, and the card must repeat it; a card whose numbers came from another run fails. The digest proves identity, not truth.
7. **Name each measurement for what it is.** A tiny Whisper's teacher-forced next-token accuracy is reported as `teacher_forced_next_token`, never as WER; only generated transcripts produce an `asr-generated` WER.
8. **Bind media bytes to their consent record before inference.** `transcribed_report` hashes each WAV and compares with the ledger's `sha256` for its source, for all rows, before running the model on any.

### The code, annotated

```python
def validate_consent(rows: list[dict]) -> None:
    """Validate all ledger records, then enforce current source consent."""
    if not rows:
        raise ValueError("empty source ledger")
    for row in rows:                                   # step 2: every row, including history
        problems = row_errors(row)                     # the ledger schema (data.08)
        if problems:
            raise ValueError("; ".join(problems))
        if any(k not in row for k in ("modality", "biometric", "consent",
                                      "attribution", "prohibited_uses")):
            raise ValueError("missing multimodal source fields")
    for row in latest(rows).values():                  # step 2: the current decision per source
        allowed = permitted_uses(row["license_spdx"])  # step 3: the license half
        if (allowed is None
                or not set(row["allowed_uses"]) <= allowed
                or "eval" not in row["allowed_uses"]):
            raise ValueError("source is not licensed for evaluation")
        if row.get("revoked"):
            raise ValueError("source consent revoked")
        if row["modality"] == "audio" and not row["biometric"]:
            raise ValueError("voice must be recorded as biometric")
        if row["biometric"] and row["consent"] == "none":   # step 3: the consent half
            raise ValueError("biometric source lacks consent")
        if row["consent"] == "cc-licensed" and not row["attribution"].strip():
            raise ValueError("CC-licensed source requires attribution")
        if not {"speaker-identification", "face-recognition"} <= set(row["prohibited_uses"]):
            raise ValueError("identity uses must be prohibited")   # step 4
```

`latest` (from `corpus.ledger`) maps each `source_id` to its last row, so a revocation appended after a grant is the row checked. `permitted_uses` evaluates the SPDX expression against the `ethics.01` allowlist and returns `None` for any unknown identifier, which is a refusal. The checks are ordered from the cheapest structural ones to the permission ones, and each raises with a message naming the rule broken.

```python
def transcript_report(rows: list[dict]) -> dict:
    """Pooled WER, CER and per-group WER; retain a digest, not the texts."""
    # ... every row needs non-empty string id, group, reference, and a string hypothesis
    if any(r["group"] == "overall" for r in rows):
        raise ValueError("overall is reserved for the pooled metric")  # the card's key
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate prediction id")    # a duplicate would be counted twice
    refs = [basic_normalize(r["reference"]) for r in rows]
    hyps = [basic_normalize(r["hypothesis"]) for r in rows]
    groups = {}
    for group in sorted({r["group"] for r in rows}):
        indices = [i for i, row in enumerate(rows) if row["group"] == group]
        groups[group] = {
            "n": len(indices),
            "wer": wer([refs[i] for i in indices], [hyps[i] for i in indices]),  # pooled within the group
        }
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()                                      # step 6: canonical JSON, then SHA-256
    return {
        "suite": "asr-recorded",
        "n": len(rows),
        "wer": wer(refs, hyps),                        # step 5: sum of edits / sum of ref words
        "cer": cer(refs, hyps),
        "groups": groups,
        "predictions_sha256": digest,
    }
```

`wer` from `tinyllm.eval.mm` (`L14.5`) sums substitutions, deletions and insertions across all pairs and divides by the summed reference lengths, which is exactly the pooling of step 5: on section 3's data, $(0 + 1 + 2 + 0)/(2 + 2 + 2 + 2) = 3/8$. The digest uses sorted keys, compact separators and `ensure_ascii=False` so that it depends only on the content, not on how a file happened to be formatted; it is computed from the raw rows, so normalization choices do not change the run's identity. The report keeps the digest, not the texts.

```python
def transcribed_report(model, tok, rows, ledger, opts) -> dict:
    validate_consent(ledger)                           # step 1
    sources = latest(ledger)
    # ... no rows raises
    # Validate every binding before executing a model on any source.
    for row in rows:
        # ... each row needs WAV bytes
        source = sources.get(row.get("source_id"))
        if source is None or source["modality"] != "audio":
            raise ValueError("audio source absent from the consent ledger")
        if hashlib.sha256(row["audio"]).hexdigest() != source["sha256"]:
            raise ValueError("audio bytes differ from the consent record")   # step 8
    predictions = []
    for row in rows:                                   # only now does the model run
        audio = to_mono_16k(read_wav(row["audio"]), FeatureConfig(feature_size=model.config.n_mels))
        segments = transcribe(model, tok, audio, opts) # real generation (L14.4)
        # ... copy id, group, reference, source_id; join segment texts as the hypothesis;
        #     record audio_sha256
    report = transcript_report(predictions)
    report["suite"] = "asr-generated"                  # step 7: a different suite name
    return {"predictions": predictions, "report": report}
```

Two loops, not one: if the third recording had changed, a single loop would already have run the model on the first two. Validating every binding first guarantees that changed or revoked audio never reaches the model at all.

```python
def speech_diagnostic(model, wavs, prefixes, targets) -> dict:
    """Teacher-forced next-token accuracy and NLL, explicitly NOT ASR WER."""
    # ... non-empty, equal lengths
    correct, losses = 0, []
    for audio, prefix, target in zip(wavs, prefixes, targets):
        # ... target must be an int inside the vocabulary
        logits = np.asarray(model.decoder(prefix, model.encode_wav(audio)).data, dtype=np.float64)
        # ... logits must be 2-D and finite
        row = logits[-1]                               # the prediction after the given prefix
        shifted = row - row.max()                      # stable log-softmax
        losses.append(float(np.log(np.exp(shifted).sum()) - shifted[target]))
        correct += int(np.argmax(row) == target)
    return {"metric": "teacher_forced_next_token", "n": len(wavs),
            "accuracy": correct / len(wavs), "nll": math.fsum(losses) / len(losses)}
```

The metric name is part of the return value, so a report cannot present this as transcription quality without visibly renaming it. The audio goes through the real `encode_wav` path (decode, resample, log-mel, encoder), which is the point of the diagnostic. `image_diagnostic` is the vision counterpart: it builds a prompt-ensembled zero-shot classifier with `zero_shot_classifier`, embeds raw images through `encode_images` with the real processor, refuses non-finite scores, and reports `zero_shot_accuracy` at top-1.

```python
def validate_card(policy: str, card: str, summary: dict) -> None:
    # ... the policy has the four sections (at least 8 words each) and names both identity uses
    # ... the card has Multimodal data, Multimodal evaluation, Limitations
    blocks = re.findall(r"```json\s*\n(.*?)\n```", card, re.S)
    if len(blocks) != 1:
        raise ValueError("card needs exactly one JSON evidence block")
    stated = json.loads(blocks[0])
    if (stated.get("predictions_sha256") != summary["predictions_sha256"]
            or stated.get("n") != summary["n"]):
        raise ValueError("card evidence provenance differs")     # step 6
    expected = {"overall": summary["wer"], **{g: v["wer"] for g, v in summary["groups"].items()}}
    if set(stated.get("wer", {})) != set(expected):
        raise ValueError("card must report every group and overall WER")   # no dropped group
    for group, value in expected.items():
        reported = stated["wer"][group]
        if (type(reported) not in (float, int) or not math.isfinite(reported)
                or abs(reported - value) > 0.00005):
            raise ValueError("card WER differs from evidence: " + group)   # 4-dp rounding allowed
```

Exactly one JSON block keeps the card unambiguous: a reader and a program read the same numbers. The set comparison of group keys means a card cannot quietly omit the worst group. The tolerance $0.00005$ is half of the fourth decimal place, so honest rounding passes and a changed number does not.

`main` reads the ledger, calls `validate_consent`, computes `transcript_report` from `evals/asr-recorded/predictions.json`, and writes `summary.json` with sorted keys and `allow_nan=False`, so a NaN can never be published as a number.

Full solution: [`evidence.py`](../../../../course/solutions/primers/ethics.08/evidence.py).

### The documents

The reference ledger has one row: `synthetic-transcripts`, `CC0-1.0`, `allowed_uses = ["eval"]`, `modality = "text"`, `biometric = false`, `consent = "public-domain"`, both identity uses prohibited, and a note that the clear and noisy labels are synthetic. Because the data is authored text, not recorded speech, it is honestly recorded as non-biometric; a real voice source would be `audio` and therefore biometric.

The consent policy's four sections each state a decision: voice and faces are biometric and no identities or templates are collected; every source prohibits speaker identification and face recognition; the only source is the synthetic CC0 file, and an optional real dataset (Common Voice) is explicitly not authorized by this fixture; and on revocation, raw media, encoder caches and derived artifacts are removed, the revocation is appended to the ledger, and the next run excludes the source.

The card reports the section 3 numbers in one JSON block with the digest of the recorded predictions, and its Limitations section says what the numbers cannot show: synthetic group labels establish neither demographic fairness nor noise robustness, and tower diagnostics are teacher-forced scores, not transcripts.

### Why this solution

- **Evidence regenerated, not typed.** Copying numbers into a card by hand lets them drift from the data; regenerating the summary and checking the card against it by digest makes a stale card fail.
- **Fail before running.** Checking consent and byte bindings before any inference costs a hash per file and guarantees that no measurement exists for unauthorized media, which is cheaper than deleting results afterwards.
- **Pooled over averaged WER.** Pooling matches how WER is defined in speech evaluation and is robust to utterance-length mix; averaging is easy to compute but answers a different question.
- **Tradeoffs accepted.** The digest proves which predictions a card refers to, not that they came from the claimed model; the report states that limit rather than overclaiming. The reference data is synthetic, so the artifacts demonstrate the plumbing and the arithmetic, not model quality.

### Explain it back

**1. A source was granted in an early ledger row and revoked in a later one. Which row does `validate_consent` use, and why does it still check the early row's shape?**

<details><summary>Answer</summary>

Permission uses `latest(rows)`, the last row per source, so the revocation wins and validation fails with "source consent revoked". Every row's shape is still checked because the ledger is an append-only history that must stay well formed; a malformed old row is a bug even if it no longer decides permission.

</details>

**2. Why is the pooled WER of references of lengths 3 and 1, each with one error, $2/4$ and not $(1/3 + 1)/2$?**

<details><summary>Answer</summary>

WER is edits per reference word over the whole set: two edits over four reference words is $1/2$. Averaging per-utterance rates gives $2/3$ and lets the one-word utterance weigh as much as the three-word one, so a suite with many short utterances would dominate the score.

</details>

**3. Why does `transcribed_report` loop over the rows twice?**

<details><summary>Answer</summary>

The first loop validates every row's binding (the source is in the ledger, is audio, and its bytes hash to the recorded `sha256`) before any inference. With a single loop, a changed or revoked recording late in the list would be detected only after the model had already processed earlier ones, and some measurements would exist for a batch that should have been refused.

</details>

**4. What does the `predictions_sha256` in the card prove, and what does it not prove?**

<details><summary>Answer</summary>

It proves the card's numbers refer to exactly this set of recorded predictions: any change to an id, a group, a reference or a hypothesis changes the digest, and `validate_card` fails. It does not prove the predictions are what a model really output, nor that the media behind them were authorized; those come from the generation path and the consent ledger.

</details>

**5. Why does `speech_diagnostic` return `metric: "teacher_forced_next_token"` instead of a WER?**

<details><summary>Answer</summary>

Teacher forcing feeds the true prefix and scores only the next token, which says nothing about how well the model transcribes when it must generate its own tokens. Labelling it as WER would let readers infer transcription quality that was never measured. Naming the metric in the result keeps the claim tied to what was computed.

</details>

## 6. Pitfalls

| Mistake | Consequence | How the solution avoids it |
|---|---|---|
| Average utterance WERs | Short utterances dominate | `wer` pools edits over reference words, overall and per group; step 5 |
| Read the first consent record | Revocations disappear | permission is checked on `latest(rows)`; step 2 |
| Copy card numbers manually | Results drift without changing the card | `validate_card` compares the digest, `n`, and every group's WER with the regenerated summary; step 6 |
| Call teacher-forced scores WER | Readers infer transcription quality that was never measured | `speech_diagnostic` names its metric; only `transcribed_report` produces an `asr-generated` WER; step 7 |
| Run the model, then check the bytes | Unauthorized or changed audio is processed | every binding is checked before the first inference; step 8 |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `L13.5` | Prompt classifier and complete image-tower inference |
| Back | `L14.1` | Decode and resample consent-bound WAV evidence |
| Back | `L14.3` | WAV-to-cross-attention encoding and next-token logits |
| Back | `L14.4` | Generate transcripts before computing WER and CER |
| Back | `L14.5` | Normalization, pooled WER, CER and zero-shot accuracy |
| Back | `data.08` | Ledger shapes, latest records and licensed uses |
| Back | `ethics.02` | Privacy policy structure and retention decisions |
| Back | `ethics.03` | Evidence-backed model cards |

Later release automation can consume the artifacts after adding its own
provenance, consent and quality gates. This practice does not implement that
release workflow or the unbuilt VLM policy.
