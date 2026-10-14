"""The model card must be backed by runnable, consent-checked evidence."""

import copy
import importlib.util
import io
import json
import math
import os
import wave
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from tinyllm.speech.decoder import Whisper
from tinyllm.vision.contrastive import DualEncoder, TextEncoder, zero_shot_classifier
from tinyllm.vision.preprocess import ProcessorConfig, preprocess
from tinyllm.vision.vit import ViT, ViTConfig
from tinyllm.autograd.tensor import Tensor
from PIL import Image

ROOT = Path(os.environ.get("OL_REPO", "."))
FIX = Path(os.environ["TINYLLM_FIXTURES"])


def evidence():
    spec = importlib.util.spec_from_file_location(
        "learner_evidence", ROOT / "primers/ethics.08/evidence.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def records():
    return [
        json.loads(x)
        for x in (ROOT / "docs/data/mm-ledger.jsonl").read_text().splitlines()
        if x.strip()
    ]


def test_artifact_numbers_are_recomputed():
    # WHY: A plausible handwritten card must not pass with stale or invented metrics.
    # KIND: conformance
    e = evidence()
    rows = json.loads((ROOT / "evals/asr-recorded/predictions.json").read_text())
    report = e.transcript_report(rows)
    assert report == json.loads((ROOT / "evals/asr-recorded/summary.json").read_text())
    policy = (ROOT / "docs/data/CONSENT_POLICY.md").read_text()
    card = (ROOT / "docs/models/multimodal.md").read_text()
    e.validate_consent(records())
    e.validate_card(policy, card, report)
    for row in records():
        assert row["url"] in policy, "every source needs its own terms in the policy"
    bad = copy.deepcopy(report)
    bad["wer"] += 0.01
    with pytest.raises(ValueError, match="WER"):
        e.validate_card(policy, card, bad)
    bad = copy.deepcopy(report)
    bad["predictions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        e.validate_card(policy, card, bad)


def test_hand_example_pooled_error_rates():
    # WHY: Pooled WER is not the mean of utterance error rates.
    # KIND: unit, smoke
    rows = [
        {
            "id": "a",
            "group": "x",
            "reference": "ONE two three",
            "hypothesis": "one two",
        },
        {"id": "b", "group": "y", "reference": "four", "hypothesis": ""},
    ]
    report = evidence().transcript_report(rows)
    assert report["wer"] == 0.5
    assert report["groups"]["x"]["wer"] == 1 / 3
    assert report["groups"]["y"]["wer"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        evidence().transcript_report(rows + rows)
    with pytest.raises(ValueError):
        evidence().transcript_report([])
    with pytest.raises(ValueError, match="reserved"):
        evidence().transcript_report([dict(rows[0], group="overall")])


@pytest.mark.parametrize(
    "change",
    [
        {"modality": "audio", "biometric": True, "consent": "none"},
        {"modality": "audio", "biometric": False},
        {"consent": "unknown"},
        {"biometric": "false"},
        {"prohibited_uses": []},
        {"revoked": True},
        {"allowed_uses": ["train"]},
        {"license_spdx": "Unknown-License"},
        {"consent": "cc-licensed", "attribution": ""},
    ],
)
def test_bad_consent_cannot_produce_evidence(change):
    # WHY: Permission errors fail closed before writing an evaluation report.
    # KIND: boundary
    rows = copy.deepcopy(records())
    rows[-1].update(change)
    with pytest.raises(ValueError):
        evidence().validate_consent(rows)


def test_append_only_revocation_uses_latest_record():
    # WHY: An old grant of permission must not hide a later revocation.
    # KIND: property
    rows = copy.deepcopy(records())
    revoked = dict(rows[-1], revoked=True)
    with pytest.raises(ValueError, match="revoked"):
        evidence().validate_consent(rows + [revoked])
    evidence().validate_consent(rows + [revoked, rows[-1]])


def test_actual_image_towers_feed_zero_shot_evidence():
    # WHY: Metrics over fabricated scores do not validate the raw-image model path.
    # KIND: conformance
    vision = ViT(
        ViTConfig(
            image_size=4,
            patch_size=2,
            n_channels=3,
            d=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
        )
    )
    model = DualEncoder(
        vision, TextEncoder(12, 8, 1, 2, 16, 6), 6, "clip", math.log(10)
    )
    cfg = ProcessorConfig(size={"height": 4, "width": 4}, do_center_crop=False)
    images = []
    for color in [(230, 20, 40), (10, 90, 240)]:
        buf = io.BytesIO()
        Image.new("RGB", (6, 5), color).save(buf, format="PNG")
        images.append(buf.getvalue())
    tok = lambda text: [1, 2] if text == "red" else [3, 4]
    weights = zero_shot_classifier(model, tok, ["red", "blue"], ["{}"])
    pixels = np.concatenate([x.pixel_values for x in preprocess(images, cfg)])
    scores = model.encode_image(Tensor(pixels)).data @ weights.T
    gold = np.argmax(scores, axis=1)
    out = evidence().image_diagnostic(
        model, tok, images, cfg, ["red", "blue"], ["{}"], gold
    )
    assert out["value"] == 1
    assert out["predictions"] == gold.tolist()
    wrong = evidence().image_diagnostic(
        model, tok, images, cfg, ["red", "blue"], ["{}"], 1 - gold
    )
    assert wrong["value"] == 0


def test_actual_whisper_logits_feed_named_diagnostic():
    # WHY: Teacher-forced accuracy must use audio and must not be labelled ASR WER.
    # KIND: conformance
    model = Whisper.from_pretrained(str(FIX / "L14.2/tiny-whisper-128"))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.arange(400, dtype="<i2").tobytes())
    wav = buf.getvalue()
    row = model.decoder([1, 2], model.encode_wav(wav)).data[-1].astype(np.float64)
    target = int(np.argmax(row))
    out = evidence().speech_diagnostic(model, [wav], [[1, 2]], [target])
    assert out["metric"] == "teacher_forced_next_token"
    assert out["accuracy"] == 1
    assert "wer" not in out
    assert_close(
        out["nll"], float(np.log(np.exp(row - row.max()).sum())), atol=1e-7, rtol=0
    )
    wrong = evidence().speech_diagnostic(
        model, [wav], [[1, 2]], [(target + 1) % model.config.vocab]
    )
    assert wrong["accuracy"] == 0
    with pytest.raises(ValueError):
        evidence().speech_diagnostic(model, [wav], [[1, 2]], [-1])


def test_report_command_checks_consent_before_writing(tmp_path, monkeypatch):
    # WHY: A denied source must not yield a fresh report through the CLI entry point.
    # KIND: fault
    import shutil
    import sys

    for path in ["docs/data/mm-ledger.jsonl", "evals/asr-recorded/predictions.json"]:
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
    monkeypatch.setattr(sys, "argv", ["evidence.py", "--root", str(tmp_path)])
    e = evidence()
    e.main()
    out = tmp_path / "evals/asr-recorded/summary.json"
    assert json.loads(out.read_text()) == json.loads(
        (ROOT / "evals/asr-recorded/summary.json").read_text()
    )
    out.unlink()
    rows = records()
    rows[-1]["revoked"] = True
    (tmp_path / "docs/data/mm-ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    with pytest.raises(ValueError, match="revoked"):
        e.main()
    assert not out.exists()
