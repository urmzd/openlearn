"""Prepare explicitly supplied capstone text into auditable token streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def stage_inputs(
    spec_path: Path, inputs: Path, out: Path, seed: int | None = None
) -> dict:
    """Stage the full capstone's existing CorpusBuild outputs without resplitting."""
    from tinyllm.cli_c1 import train_shards
    from tinyllm.io.tokens import read_tokens_header
    from tinyllm.tok.bpe import BPETokenizer

    spec = json.loads(spec_path.read_text())
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        spec["seed"] = seed
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"output directory must be empty: {out}")
    inputs = inputs.resolve()

    def source(name):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"input path must be relative: {name}")
        path = (inputs / name).resolve()
        if not path.is_relative_to(inputs) or path == inputs:
            raise ValueError(f"input path escapes its artifact root: {name}")
        if not path.is_file():
            raise ValueError(
                f"missing prepared input {path}: run CorpusBuild and tokenizer preparation first"
            )
        return path

    vocab = spec["model"]["vocab_size"]
    tokenizer = source(spec["tokenizer"]["path"])
    if BPETokenizer.from_hf_json(str(tokenizer)).vocab_size != vocab:
        raise ValueError("prepared tokenizer vocabulary differs from the model")
    files = {tokenizer}
    docs = 0
    for name in spec["data"]["train"]:
        manifest = source(name)
        parsed = json.loads(manifest.read_text())
        if (
            parsed["tokenizer_sha256"]
            != hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        ):
            raise ValueError("training manifest names a different tokenizer")
        files.add(manifest)
        files.update(Path(path) for path in train_shards([manifest], vocab))
        for row in parsed["files"]:
            if row["split"] == "train":
                continue
            path = source((manifest.parent / row["name"]).relative_to(inputs))
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if (
                digest != row["sha256"]
                or read_tokens_header(str(path))["n_tokens"] != row["n_tokens"]
            ):
                raise ValueError(
                    f"prepared validation shard differs from its manifest: {path}"
                )
            files.add(path)
        docs += sum(row["n_docs"] for row in parsed["files"] if row["split"] == "train")
    val_tokens = 0
    for name in spec["data"]["val"]:
        path = source(name)
        header = read_tokens_header(str(path))
        if header["vocab_size"] != vocab:
            raise ValueError("validation vocabulary differs from the model")
        val_tokens += header["n_tokens"]
        files.add(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        for path in files:
            target = stage / path.relative_to(inputs)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        (stage / "train-spec.json").write_text(json.dumps(spec, sort_keys=True) + "\n")
        result = {
            "train_docs": docs,
            "val_tokens": val_tokens,
            "vocab_size": vocab,
            "source": "prepared CorpusBuild artifacts",
        }
        (stage / "INPUTS.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        if out.exists():
            out.rmdir()
        os.rename(stage, out)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def prepare(
    spec_path: Path,
    source: Path,
    out: Path,
    holdout: int,
    license_spdx: str,
    seed: int | None = None,
) -> dict:
    from corpus.ledger import permitted_uses, verify
    from corpus.pii import scrub
    from corpus.shard import write_shards
    from corpus.stage import Doc
    from corpus.tokenize import read_bin, tokenize_shards, write_bin
    from tinyllm.tok.bpe import BPETokenizer

    spec = json.loads(spec_path.read_text())
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        spec["seed"] = seed
    raw = source.read_bytes()
    texts = [json.loads(line)["text"] for line in raw.splitlines() if line.strip()]
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("every source row must contain nonempty text")
    if not 0 < holdout < len(texts):
        raise ValueError("holdout must leave nonempty training and validation sets")
    uses = permitted_uses(license_spdx)
    if uses is None or not {"train", "eval"}.issubset(uses):
        raise ValueError(f"{license_spdx} does not permit both training and evaluation")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"output directory must be empty: {out}")
    if spec["model"].get("tl_tokenizer") != "file":
        raise ValueError("capstone preparation requires a file tokenizer")
    if len(spec["data"]["train"]) != 1 or len(spec["data"].get("val", [])) != 1:
        raise ValueError(
            "preparation expects one training manifest and one validation stream"
        )

    def relative(name):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError(
                f"artifact path must stay inside the output directory: {name}"
            )
        return path

    tok_rel = relative(spec["tokenizer"]["path"])
    train_rel = relative(spec["data"]["train"][0])
    val_rel = relative(spec["data"]["val"][0])
    if train_rel.name != "_MANIFEST.json":
        raise ValueError("data.train must name a token _MANIFEST.json")
    source_sha = hashlib.sha256(raw).hexdigest()
    stamp = datetime.fromtimestamp(source.stat().st_mtime, UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    split_at = len(texts) - holdout
    groups = {}
    for split, rows in (("train", texts[:split_at]), ("val", texts[split_at:])):
        groups[split] = [
            scrub(
                Doc(
                    f"{split}:{i:08d}",
                    split,
                    text,
                    {
                        "url": source.resolve().as_uri(),
                        "license_spdx": license_spdx,
                        "lang": "en",
                    },
                )
            )[0]
            for i, text in enumerate(rows)
        ]
    if {doc.text for doc in groups["train"]} & {doc.text for doc in groups["val"]}:
        raise ValueError(
            "training and validation contain identical documents after cleaning"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        tokenizer = BPETokenizer.train(
            (doc.text for doc in groups["train"]), int(spec["model"]["vocab_size"])
        )
        if tokenizer.vocab_size != spec["model"]["vocab_size"]:
            raise ValueError(
                "training corpus cannot produce the requested tokenizer vocabulary"
            )
        tokenizer.save(str(stage / tok_rel.parent))
        if tok_rel.name != "tokenizer.json":
            os.rename(stage / tok_rel.parent / "tokenizer.json", stage / tok_rel)
        spec_bytes = (json.dumps(spec, sort_keys=True) + "\n").encode()
        (stage / "train-spec.json").write_bytes(spec_bytes)
        config_sha = hashlib.sha256(spec_bytes).hexdigest()
        for split, docs in groups.items():
            corpus = stage / "corpus" / split
            write_shards(
                docs,
                corpus,
                dataset=spec["name"],
                version="v1",
                shard_rows=1024,
                val_permille=0 if split == "train" else 1000,
                ledger_ref=f"corpus/{split}/LEDGER.jsonl",
                config_sha256=config_sha,
            )
            ledger = {
                "source_id": split,
                "url": source.resolve().as_uri(),
                "license_spdx": license_spdx,
                "retrieved_at": stamp,
                "sha256": source_sha,
                "n_docs": len(texts),
                "allowed_uses": ["train", "eval"],
                "pii_policy": "scrub",
                "filters_applied": ["positional_split", "pii_scrub"],
                "kept": len(docs),
                "dropped": len(texts) - len(docs),
                "notes": f"last {holdout} documents held out; no tokenizer fitting on validation",
            }
            (corpus / "LEDGER.jsonl").write_text(
                json.dumps(ledger, sort_keys=True) + "\n"
            )
            errors = verify(
                corpus / "LEDGER.jsonl",
                corpus,
                use="train" if split == "train" else "eval",
            )
            if errors:
                raise ValueError("; ".join(errors))
            destination = (
                stage / train_rel.parent
                if split == "train"
                else stage / "validation-tokens"
            )
            manifest = tokenize_shards(
                corpus / "_MANIFEST.json",
                stage / tok_rel,
                destination,
                tokenizer_id=spec["tokenizer"]["id"],
            )
            if split == "val":
                import numpy as np

                arrays = [
                    read_bin(destination / row["name"])[1]
                    for row in manifest.files
                    if row["split"] == "val"
                ]
                (stage / val_rel).parent.mkdir(parents=True, exist_ok=True)
                write_bin(stage / val_rel, np.concatenate(arrays), tokenizer.vocab_size)
        from tinyllm.io.tokens import read_tokens_header

        result = {
            "source_sha256": source_sha,
            "train_docs": split_at,
            "val_docs": holdout,
            "val_tokens": read_tokens_header(str(stage / val_rel))["n_tokens"],
            "vocab_size": tokenizer.vocab_size,
            "train": str(train_rel),
            "val": str(val_rel),
            "tokenizer": str(tok_rel),
        }
        (stage / "INPUTS.json").write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n"
        )
        if out.exists():
            out.rmdir()
        os.rename(stage, out)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def intercept(argv: list[str]) -> int | None:
    if argv[:2] != ["capstone", "prepare"]:
        return None
    parser = argparse.ArgumentParser(prog="tinyllm capstone prepare")
    parser.add_argument("--spec", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", type=Path)
    sources.add_argument("--fixture", type=Path)
    sources.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=0)
    parser.add_argument("--license", default="")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv[2:])
    try:
        if args.inputs is not None:
            result = stage_inputs(args.spec, args.inputs, args.out, args.seed)
            print(json.dumps(result), flush=True)
            return 0
        source = args.source
        if args.fixture is not None:
            fixture_root = Path(os.environ["TINYLLM_FIXTURES"]).resolve()
            source = (fixture_root / args.fixture).resolve()
            if not source.is_relative_to(fixture_root):
                raise ValueError("fixture must stay inside TINYLLM_FIXTURES")
        result = prepare(
            args.spec, source, args.out, args.holdout, args.license, args.seed
        )
    except (ValueError, OSError, KeyError) as error:
        print(f"tinyllm capstone prepare: {error}", file=sys.stderr)
        return 65
    print(json.dumps(result), flush=True)
    return 0
