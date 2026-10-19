"""Consent and model-card evidence (ethics.08).

Run `python primers/ethics.08/evidence.py --root .` to regenerate the
transcript report from recorded predictions. The tower diagnostics are
callable for a learner's loaded models; they never label teacher-forced
accuracy as transcription WER or random weights as a trained model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from corpus.ledger import latest, permitted_uses, row_errors
from tinyllm.eval.mm import basic_normalize, cer, wer, zero_shot_accuracy
from tinyllm.speech.decoder import Whisper
from tinyllm.speech.decode import DecodeOptions, transcribe
from tinyllm.io.wav import read_wav
from tinyllm.speech.frontend import FeatureConfig, to_mono_16k
from tinyllm.vision.contrastive import DualEncoder, zero_shot_classifier


def transcript_report(rows: list[dict]) -> dict:
    """Pooled WER, CER and per-group WER; retain a digest, not the texts."""
    if not rows or any(
        not isinstance(r, dict)
        or any(
            not isinstance(r.get(k), str) or not r[k].strip()
            for k in ("id", "group", "reference")
        )
        or not isinstance(r.get("hypothesis"), str)
        for r in rows
    ):
        raise ValueError("predictions need id, group, reference and hypothesis")
    if any(r["group"] == "overall" for r in rows):
        raise ValueError("overall is reserved for the pooled metric")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate prediction id")
    refs = [basic_normalize(r["reference"]) for r in rows]
    hyps = [basic_normalize(r["hypothesis"]) for r in rows]
    groups = {}
    for group in sorted({r["group"] for r in rows}):
        indices = [i for i, row in enumerate(rows) if row["group"] == group]
        groups[group] = {
            "n": len(indices),
            "wer": wer([refs[i] for i in indices], [hyps[i] for i in indices]),
        }
    digest = hashlib.sha256(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    return {
        "suite": "asr-recorded",
        "n": len(rows),
        "wer": wer(refs, hyps),
        "cer": cer(refs, hyps),
        "groups": groups,
        "predictions_sha256": digest,
    }


def transcribed_report(
    model: Whisper, tok, rows: list[dict], ledger: list[dict], opts: DecodeOptions
) -> dict:
    """Generate ASR evidence from consent-bound WAV bytes, then compute WER/CER."""
    validate_consent(ledger)
    sources = latest(ledger)
    if not rows:
        raise ValueError("no audio examples")
    # Validate every binding before executing a model on any source.
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("audio"), bytes):
            raise ValueError("each audio example needs WAV bytes")
        source = sources.get(row.get("source_id"))
        if source is None or source["modality"] != "audio":
            raise ValueError("audio source absent from the consent ledger")
        if hashlib.sha256(row["audio"]).hexdigest() != source["sha256"]:
            raise ValueError("audio bytes differ from the consent record")
    predictions = []
    for row in rows:
        audio = to_mono_16k(
            read_wav(row["audio"]), FeatureConfig(feature_size=model.config.n_mels)
        )
        segments = transcribe(model, tok, audio, opts)
        predictions.append(
            {key: row[key] for key in ("id", "group", "reference", "source_id")}
        )
        predictions[-1]["hypothesis"] = " ".join(segment.text for segment in segments)
        predictions[-1]["audio_sha256"] = hashlib.sha256(row["audio"]).hexdigest()
    report = transcript_report(predictions)
    report["suite"] = "asr-generated"
    return {"predictions": predictions, "report": report}


def image_diagnostic(
    model: DualEncoder, tok, images, processor, classes, templates, gold
) -> dict:
    """Actual raw-image and prompt towers, followed by top-1 accuracy."""
    if (
        not classes
        or not templates
        or any(not isinstance(s, str) or not s for s in classes)
    ):
        raise ValueError("nonempty classes and templates required")
    weights = zero_shot_classifier(model, tok, classes, templates)
    embeddings = model.encode_images(images, processor).data
    scores = embeddings @ weights.T
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite image scores")
    accuracy = zero_shot_accuracy(scores, np.asarray(gold), k=1)
    return {
        "metric": "zero_shot_accuracy",
        "n": len(images),
        "value": accuracy,
        "predictions": np.argmax(scores, axis=1).tolist(),
    }


def speech_diagnostic(model: Whisper, wavs, prefixes, targets) -> dict:
    """Teacher-forced next-token accuracy and NLL, explicitly NOT ASR WER."""
    if not wavs or not (len(wavs) == len(prefixes) == len(targets)):
        raise ValueError(
            "nonempty equally sized audio, prefix and target lists required"
        )
    correct, losses = 0, []
    for audio, prefix, target in zip(wavs, prefixes, targets):
        if type(target) is not int or not 0 <= target < model.config.vocab:
            raise ValueError("target outside vocabulary")
        logits = np.asarray(
            model.decoder(prefix, model.encode_wav(audio)).data, dtype=np.float64
        )
        if logits.ndim != 2 or not np.isfinite(logits).all():
            raise ValueError("expected finite single-example logits")
        row = logits[-1]
        shifted = row - row.max()
        losses.append(float(np.log(np.exp(shifted).sum()) - shifted[target]))
        correct += int(np.argmax(row) == target)
    return {
        "metric": "teacher_forced_next_token",
        "n": len(wavs),
        "accuracy": correct / len(wavs),
        "nll": math.fsum(losses) / len(losses),
    }


def validate_consent(rows: list[dict]) -> None:
    """Validate all ledger records, then enforce current source consent."""
    if not rows:
        raise ValueError("empty source ledger")
    for row in rows:
        problems = row_errors(row)
        if problems:
            raise ValueError("; ".join(problems))
        if any(
            k not in row
            for k in (
                "modality",
                "biometric",
                "consent",
                "attribution",
                "prohibited_uses",
            )
        ):
            raise ValueError("missing multimodal source fields")
    for row in latest(rows).values():
        allowed = permitted_uses(row["license_spdx"])
        if (
            allowed is None
            or not set(row["allowed_uses"]) <= allowed
            or "eval" not in row["allowed_uses"]
        ):
            raise ValueError("source is not licensed for evaluation")
        if row.get("revoked"):
            raise ValueError("source consent revoked")
        if row["modality"] == "audio" and not row["biometric"]:
            raise ValueError("voice must be recorded as biometric")
        if row["biometric"] and row["consent"] == "none":
            raise ValueError("biometric source lacks consent")
        if row["consent"] == "cc-licensed" and not row["attribution"].strip():
            raise ValueError("CC-licensed source requires attribution")
        if not {"speaker-identification", "face-recognition"} <= set(
            row["prohibited_uses"]
        ):
            raise ValueError("identity uses must be prohibited")


def validate_card(policy: str, card: str, summary: dict) -> None:
    """Check sections and a JSON evidence block with values rounded to 4 dp."""
    for heading in (
        "Biometric data",
        "Prohibited uses",
        "Source terms",
        "Retention and deletion",
    ):
        match = re.search(
            r"^## " + re.escape(heading) + r"\s*\n(.+?)(?=^## |\Z)", policy, re.M | re.S
        )
        if match is None or len(match[1].split()) < 8:
            raise ValueError("missing consent policy section: " + heading)
    for phrase in ("speaker-identification", "face-recognition"):
        if phrase not in policy:
            raise ValueError("policy must prohibit " + phrase)
    for heading in ("Multimodal data", "Multimodal evaluation", "Limitations"):
        if not re.search(r"^## " + re.escape(heading) + r"\s*$", card, re.M):
            raise ValueError("missing model-card section: " + heading)
    blocks = re.findall(r"```json\s*\n(.*?)\n```", card, re.S)
    if len(blocks) != 1:
        raise ValueError("card needs exactly one JSON evidence block")
    stated = json.loads(blocks[0])
    if (
        stated.get("predictions_sha256") != summary["predictions_sha256"]
        or stated.get("n") != summary["n"]
    ):
        raise ValueError("card evidence provenance differs")
    expected = {
        "overall": summary["wer"],
        **{g: v["wer"] for g, v in summary["groups"].items()},
    }
    if set(stated.get("wer", {})) != set(expected):
        raise ValueError("card must report every group and overall WER")
    for group, value in expected.items():
        reported = stated["wer"][group]
        if (
            type(reported) not in (float, int)
            or not math.isfinite(reported)
            or abs(reported - value) > 0.00005
        ):
            raise ValueError("card WER differs from evidence: " + group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root
    ledger = [
        json.loads(line)
        for line in (root / "docs/data/mm-ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    validate_consent(ledger)
    rows = json.loads((root / "evals/asr-recorded/predictions.json").read_text())
    report = transcript_report(rows)
    out = root / "evals/asr-recorded/summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
