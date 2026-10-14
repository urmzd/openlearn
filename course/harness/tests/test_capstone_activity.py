"""Reference capstone training must resume without changing weights or loss."""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def capstone(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "course/ref/python"))
    import tinyllm

    monkeypatch.setattr(
        tinyllm,
        "__path__",
        [
            str(root / "course/ref/python/tinyllm"),
            str(root / "course/ref/entry/python/tinyllm"),
        ],
    )
    from tinyllm import cli_c1

    return cli_c1


@pytest.mark.parametrize("precision,recompute", [("fp32", False), ("bf16", True)])
@pytest.mark.parametrize("file_tokenizer", [False, True])
def test_llama_activity_resume_is_exact(
    tmp_path, capstone, precision, recompute, monkeypatch, capsys, file_tokenizer
):
    from tinyllm.cli_activity import check_train_spec
    from tinyllm.io.activity import Cancelled
    from tinyllm.io.checkpoint import load_checkpoint, verify_step_dir

    train = capstone.train
    base = tmp_path
    spec = {
        "name": "mini",
        "model": {
            "tl_arch": "llama",
            "tl_tokenizer": "bytes",
            "vocab_size": 256,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128,
            "tie_word_embeddings": True,
            "tl_format": 1,
        },
        "data": {"train": ["train.bin"], "val": ["train.bin"]},
        "optimizer": {"name": "adamw", "lr": 0.003, "betas": [0.9, 0.95]},
        "schedule": {
            "name": "wsd",
            "warmup_steps": 1,
            "decay_frac": 0.5,
            "min_lr_ratio": 0.1,
        },
        "precision": {"dtype": "fp32"},
        "batch": {"micro_batch": 2, "accum": 2, "seq_len": 8},
        "steps": 4,
        "ckpt_every": 2,
        "seed": 3,
    }
    spec["precision"] = {"dtype": precision, "recompute": recompute}
    if file_tokenizer:
        spec["model"]["tl_tokenizer"] = "file"
        spec["tokenizer"] = {"path": "tokenizer.json"}
    assert not check_train_spec(spec), check_train_spec(spec)

    class Act:
        key = "mini"
        resume = None
        cancelled = False

        def __init__(self, path, stop=None):
            self.artifacts = str(path)
            self.stop = stop
            path.mkdir()
            if file_tokenizer:
                from tinyllm.tok.bpe import BPETokenizer

                BPETokenizer.train(["Once upon a time"], 256).save(str(path))
            h = np.zeros(256, dtype="<i4")
            ids = np.tile(np.arange(16, dtype="<u2"), 20)
            h[:4] = [20240520, 1, len(ids), 256]
            (path / "train.bin").write_bytes(h.tobytes() + ids.tobytes())

        def step(self, step, *args):
            self.cancelled = step == self.stop

        def checkpoint(self, *args):
            pass

    full = Act(base / "full")
    train(full, spec)
    part = Act(base / "part", 2)
    try:
        train(part, spec)
    except Cancelled:
        pass
    else:
        raise AssertionError("cancellation not raised")
    part.stop = None
    part.cancelled = False
    train(part, spec)
    a = load_checkpoint(str(base / "full/runs/mini/ckpt"))
    b = load_checkpoint(str(base / "part/runs/mini/ckpt"))
    assert a.step == b.step == 4
    assert a.extra["tokens_seen"] == 128
    assert a.model.keys() == b.model.keys()
    for k in a.model:
        np.testing.assert_array_equal(a.model[k], b.model[k], err_msg=k)
    assert json.loads((base / "full/runs/mini/final.json").read_text()) == json.loads(
        (base / "part/runs/mini/final.json").read_text()
    )
    assert not verify_step_dir(b.path)

    from tinyllm.cli_c1_eval import eval_ppl, quality

    metric = eval_ppl(str(base / "part/runs/mini/ckpt"), str(base / "part/train.bin"))
    assert metric["tokens"] == 319
    assert np.isfinite(metric["ppl"]) and metric["ppl"] > 1
    assert metric["bpb"] == pytest.approx(np.log2(metric["ppl"]))

    from types import SimpleNamespace

    from tinyllm.cli_activity import eval_main

    eval_spec = {
        "name": "mini-eval",
        "suites": ["ppl"],
        "subjects": [{"id": "trained", "model": "runs/mini/ckpt/step-000004"}],
        "seed": 3,
    }
    activity = SimpleNamespace(
        artifacts=str(base / "part"),
        key="eval",
        load_spec=lambda check: eval_spec,
        check_cancel=lambda: None,
        metric=lambda *args: None,
    )
    outputs = eval_main(activity)
    summary = json.loads((base / "part" / outputs[0]).read_text())
    assert summary["metrics"]["trained"]["nll"]["mean"] == pytest.approx(
        np.log(metric["ppl"])
    )
    root = Path(__file__).resolve().parents[3]
    suite = base / "primers/craft.22/evals.py"
    suite.parent.mkdir(parents=True)
    suite.write_text((root / "course/ref/primers/craft.22/evals.py").read_text())
    monkeypatch.chdir(base)
    result = quality(str(base / "part/runs/mini/ckpt"), 0, 2)
    assert result == quality(str(base / "part/runs/mini/ckpt"), 0, 2)
    assert result["samples"] == 2
    assert 0 <= result["repeat8"] <= 1
    assert 0 <= result["distinct1"] <= 1

    from tinyllm.cli_activity import intercept
    from tinyllm.io.activity import Activity

    cli_root = base / "cli"
    cli_root.mkdir()
    (cli_root / "train.bin").write_bytes((base / "full/train.bin").read_bytes())
    if file_tokenizer:
        (cli_root / "tokenizer.json").write_bytes(
            (base / "full/tokenizer.json").read_bytes()
        )
        assert (Path(b.path) / "tokenizer.json").is_file()
        manifest = json.loads((Path(b.path) / "MANIFEST.json").read_text())
        assert "tokenizer.json" in {row["name"] for row in manifest["files"]}
    monkeypatch.setenv("TL_ARTIFACTS", str(cli_root))
    monkeypatch.setenv("TL_IDEMPOTENCY_KEY", "cli-run")
    monkeypatch.setattr(Activity, "install_signal_handlers", lambda self: None)
    args = [
        "train",
        "--spec",
        str(base / "full/runs/mini/spec.json"),
        "--progress",
        str(cli_root / "progress.jsonl"),
    ]
    capsys.readouterr()
    assert intercept(args) == 0
    first = capsys.readouterr().out
    assert intercept(args) == 0
    assert capsys.readouterr().out == first
    summary = json.loads(first)
    assert summary["steps"] == 4 and np.isfinite(summary["final_loss"])
    assert (cli_root / summary["ckpt"] / "model.safetensors").is_file()


def test_latest_cannot_escape_checkpoint_root(tmp_path, capstone):
    from tinyllm.cli_modern import UsageError, model_dir

    (tmp_path / "LATEST").write_text("../other")
    with pytest.raises(UsageError, match="checkpoint step"):
        model_dir(str(tmp_path))


def test_preparation_records_clean_disjoint_inputs(tmp_path, capstone):
    from corpus.ledger import verify
    from corpus.shard import read_shards
    from tinyllm.cli_c1 import train_shards
    from tinyllm.cli_c1_data import prepare, stage_inputs
    from tinyllm.io.activity import SpecError

    root = Path(__file__).resolve().parents[3]
    spec = json.loads((root / "course/ref/entry/specs/c1/smoke.json").read_text())
    spec["model"]["vocab_size"] = 256
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    source = tmp_path / "source.jsonl"
    texts = [
        "Training example: contact ana@example.com for the story.",
        "Another training story about a frog.",
        "Held-out story about the moon.",
    ]
    source.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    result = prepare(spec_path, source, tmp_path / "out", 1, "CC0-1.0", seed=4)
    assert result["train_docs"] == 2 and result["val_docs"] == 1
    assert json.loads((tmp_path / "out/train-spec.json").read_text())["seed"] == 4
    train = list(read_shards(tmp_path / "out/corpus/train"))
    val = list(read_shards(tmp_path / "out/corpus/val"))
    assert len(train) == 2 and len(val) == 1
    assert "<EMAIL>" in train[0].text and "ana@example.com" not in train[0].text
    assert val[0].text == texts[-1]
    for split, use in [("train", "train"), ("val", "eval")]:
        folder = tmp_path / "out/corpus" / split
        assert not verify(folder / "LEDGER.jsonl", folder, use=use)
    manifest = tmp_path / "out" / result["train"]
    staged = stage_inputs(spec_path, tmp_path / "out", tmp_path / "staged", seed=2)
    assert staged["train_docs"] == 2 and staged["val_tokens"] > 1
    assert json.loads((tmp_path / "staged/train-spec.json").read_text())["seed"] == 2
    assert (tmp_path / "staged" / result["tokenizer"]).read_bytes() == (
        tmp_path / "out" / result["tokenizer"]
    ).read_bytes()
    shards = train_shards([manifest], 256)
    assert len(shards) == 1
    Path(shards[0]).write_bytes(Path(shards[0]).read_bytes()[:-1])
    with pytest.raises(SpecError, match="checksum mismatch"):
        train_shards([manifest], 256)
    with pytest.raises(ValueError, match="must be empty"):
        prepare(spec_path, source, tmp_path / "out", 1, "CC0-1.0")
    source.write_text(
        json.dumps({"text": texts[0]}) + "\n" + json.dumps({"text": texts[0]}) + "\n"
    )
    with pytest.raises(ValueError, match="identical documents"):
        prepare(spec_path, source, tmp_path / "duplicate", 1, "CC0-1.0")
    assert not (tmp_path / "duplicate").exists()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {"steps": "many"},
        {"model": []},
        {"precision": None},
        {"data": {"train": "file.bin"}},
        {"optimizer": {"name": "adamw", "lr": None}},
    ],
)
def test_training_spec_errors_are_reported(capstone, bad):
    from tinyllm.cli_activity import check_train_spec

    root = Path(__file__).resolve().parents[3]
    spec = json.loads((root / "course/ref/entry/specs/c1/smoke.json").read_text())
    if isinstance(bad, dict):
        spec.update(bad)
    else:
        spec = bad
    assert check_train_spec(spec)
