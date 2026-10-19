"""WAV transcription through the learner's Whisper model and decoding policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def intercept(argv: list[str]) -> int | None:
    if argv[:2] == ["eval", "asr"]:
        return evaluate(argv[2:])
    if argv and argv[0] == "features":
        return features(argv[1:])
    if argv and argv[0] == "logits" and "--audio-features" in argv:
        return logits(argv[1:])
    if not argv or argv[0] != "transcribe":
        return None
    if "--logits-trace" in argv:
        return replay(argv[1:])
    parser = argparse.ArgumentParser(prog="tinyllm transcribe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language")
    parser.add_argument(
        "--task", choices=("transcribe", "translate"), default="transcribe"
    )
    parser.add_argument(
        "--timestamps", action=argparse.BooleanOptionalAction, default=True
    )
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--greedy", action="store_true")
    search.add_argument("--beam", type=int, default=0)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--patience", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=224)
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--detect-language", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv[1:])
    try:
        from tinyllm.io.wav import read_wav
        from tinyllm.speech.decode import DecodeOptions, detect_language, transcribe
        from tinyllm.speech.decoder import Whisper
        from tinyllm.speech.frontend import (
            FeatureConfig,
            pad_or_trim,
            to_mono_16k,
            whisper_log_mel,
        )
        from tinyllm.speech.tokenizer import WhisperTokenizer

        opts = DecodeOptions(
            task=args.task,
            language=args.language,
            timestamps=args.timestamps,
            beam_size=args.beam,
            best_of=args.best_of,
            patience=args.patience,
            seed=args.seed,
            max_tokens=args.max_tokens,
            condition_on_previous_text=args.condition_on_previous_text,
        )
        if args.greedy or args.temperature is not None:
            from dataclasses import replace

            if args.greedy and args.temperature not in (None, 0):
                raise ValueError("--greedy conflicts with a nonzero temperature")
            opts = replace(
                opts, temperatures=(0.0 if args.greedy else args.temperature,)
            )
        opts.validate()
        model = Whisper.from_pretrained(args.model)
        tok = WhisperTokenizer.from_pretrained(args.model, language=args.language)
        if model.config.vocab != tok.bpe.vocab_size:
            raise ValueError("tokenizer and model vocabulary sizes differ")
        cfg = FeatureConfig(feature_size=model.config.n_mels)
        audio = to_mono_16k(read_wav(Path(args.audio).read_bytes()), cfg)
        if args.detect_language:
            if not len(audio):
                raise ValueError("cannot detect a language in empty audio")
            mel = whisper_log_mel(
                pad_or_trim(audio, max(400, model.config.n_audio_ctx * 320)), cfg
            )
            language, probabilities = detect_language(model, mel, tok)
            result = {"language": language, "probabilities": probabilities}
        else:
            segments = transcribe(model, tok, audio, opts)
            result = {
                "text": " ".join(s.text for s in segments),
                "segments": [asdict(s) for s in segments],
                "ids": [token for s in segments for token in s.tokens],
            }
        encoded = json.dumps(
            result, ensure_ascii=False, allow_nan=False, sort_keys=True
        )
        if args.out:
            Path(args.out).write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"transcribe: {error}", file=sys.stderr)
        return 2


def replay(argv: list[str]) -> int:
    """Run the same long-form policy on recorded model outputs, without weights."""
    parser = argparse.ArgumentParser(prog="tinyllm transcribe --logits-trace")
    parser.add_argument("--logits-trace", required=True)
    parser.add_argument("--opts", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        from tinyllm.speech.decode import DecodeOptions, _transcribe_windows
        from tinyllm.speech.tokenizer import WhisperTokenizer
        from tinyllm.speech.trace import RecordedLogits

        options_path = Path(args.opts)
        document = json.loads(options_path.read_text())
        allowed = {"tokenizer", "decode", "duration", "window_frames", "context_size"}
        if not isinstance(document, dict) or set(document) - allowed:
            raise ValueError("unknown recorded-logit options")
        if not {"tokenizer", "duration", "window_frames"} <= set(document):
            raise ValueError(
                "trace options require tokenizer, duration and window_frames"
            )
        tokenizer_path = Path(document["tokenizer"])
        if not tokenizer_path.is_absolute():
            tokenizer_path = options_path.parent / tokenizer_path
        opts = DecodeOptions(**document.get("decode", {}))
        opts.validate()
        tok = WhisperTokenizer.from_pretrained(
            str(tokenizer_path), language=opts.language
        )
        trace = RecordedLogits(
            args.logits_trace, context_size=document.get("context_size", 448)
        )
        if trace.vocab_size != tok.bpe.vocab_size:
            raise ValueError("trace and tokenizer vocabulary sizes differ")
        frames = document["window_frames"]
        if type(frames) is not int or frames < 1:
            raise ValueError("window_frames must be a positive integer")
        segments = _transcribe_windows(
            trace.at_seek, tok, float(document["duration"]), frames, opts
        )
        result = {
            "text": " ".join(s.text for s in segments),
            "segments": [asdict(s) for s in segments],
            "ids": [token for s in segments for token in s.tokens],
        }
        encoded = json.dumps(
            result, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        if args.out:
            Path(args.out).write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"transcribe: {error}", file=sys.stderr)
        return 2


def logits(argv: list[str]) -> int:
    """Teacher-forced Whisper logits for independently recorded feature batches."""
    parser = argparse.ArgumentParser(prog="tinyllm logits")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-features", required=True)
    parser.add_argument("--feature-key", default="mel")
    parser.add_argument("--decoder-prompts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        import numpy as np
        from tinyllm.speech.decoder import Whisper

        with np.load(args.audio_features, allow_pickle=False) as archive:
            features = archive[args.feature_key]
        if (
            features.ndim != 3
            or features.dtype.kind != "f"
            or not np.isfinite(features).all()
        ):
            raise ValueError(
                "audio features must be finite floating [batch, mel, frames]"
            )
        prompts = []
        for line in Path(args.decoder_prompts).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            ids = row.get("ids") if isinstance(row, dict) else None
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(i) is not int or i < 0 for i in ids)
            ):
                raise ValueError(
                    "each decoder prompt must contain nonnegative integer ids"
                )
            prompts.append(ids)
        if len(prompts) != features.shape[0] or not prompts:
            raise ValueError("one decoder prompt is required per feature batch row")
        if len({len(ids) for ids in prompts}) != 1:
            raise ValueError("decoder prompts must have equal lengths")
        model = Whisper.from_pretrained(args.model)
        ids = np.asarray(prompts, dtype=np.int64)
        if np.any(ids >= model.config.vocab):
            raise ValueError("decoder prompt id exceeds model vocabulary")
        values = model(features, ids).data
        # An open file preserves the requested path, including non-.npz suffixes.
        with Path(args.out).open("wb") as output:
            np.savez(output, logits=values)
        print(json.dumps({"out": args.out, "prompts": len(prompts)}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        print(f"logits: {error}", file=sys.stderr)
        return 2


def features(argv: list[str]) -> int:
    """Extract deterministic Whisper features from a WAV or sorted WAV directory."""
    parser = argparse.ArgumentParser(prog="tinyllm features")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--kind", choices=["whisper-log-mel"], required=True)
    parser.add_argument("--n-mels", type=int, choices=[80, 128], default=80)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        import numpy as np
        from tinyllm.io.wav import read_wav
        from tinyllm.speech.frontend import (
            FeatureConfig,
            pad_or_trim,
            to_mono_16k,
            whisper_log_mel,
        )

        source = Path(args.audio)
        paths = sorted(source.glob("*.wav")) if source.is_dir() else [source]
        if not paths:
            raise ValueError("audio directory contains no WAV files")
        cfg = FeatureConfig(feature_size=args.n_mels)
        arrays = {}
        for path in paths:
            samples = to_mono_16k(read_wav(path.read_bytes()), cfg)
            arrays[f"{path.stem}.f{args.n_mels}"] = whisper_log_mel(
                pad_or_trim(samples, cfg.n_samples), cfg
            )
        with Path(args.out).open("wb") as output:
            np.savez_compressed(output, **arrays)
        print(json.dumps({"out": args.out, "clips": len(paths), "n_mels": args.n_mels}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"features: {error}", file=sys.stderr)
        return 2


def evaluate(argv: list[str]) -> int:
    """Score a generated transcript against a UTF-8 reference transcript."""
    if "--data" in argv:
        return evaluate_corpus(argv)
    parser = argparse.ArgumentParser(prog="tinyllm eval asr")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args(argv)
    try:
        from tinyllm.eval.mm import basic_normalize, cer, wer, wer_stats

        prediction = json.loads(Path(args.pred).read_text(encoding="utf-8"))
        if isinstance(prediction, dict):
            segments = prediction.get("segments")
            text = prediction.get("text")
        elif isinstance(prediction, list):
            segments, text = prediction, None
        else:
            raise ValueError("prediction must be a transcript object or segment list")
        if segments is not None:
            if not isinstance(segments, list) or any(
                not isinstance(s, dict) or not isinstance(s.get("text"), str)
                for s in segments
            ):
                raise ValueError("each segment must contain text")
            text = " ".join(s["text"] for s in segments)
        if not isinstance(text, str):
            raise ValueError("prediction must contain text or segments")
        refs = [basic_normalize(Path(args.ref).read_text(encoding="utf-8"))]
        hyps = [basic_normalize(text)]
        stats = wer_stats(refs, hyps).sum(axis=0)
        result = {
            "wer": wer(refs, hyps),
            "cer": cer(refs, hyps),
            "substitutions": int(stats[0]),
            "deletions": int(stats[1]),
            "insertions": int(stats[2]),
            "reference_words": int(stats[3]),
            "normalizer": "whisper-basic",
        }
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"eval asr: {error}", file=sys.stderr)
        return 2


def evaluate_corpus(argv: list[str]) -> int:
    """Transcribe a JSONL WAV corpus once and bootstrap utterance-level WER."""
    parser = argparse.ArgumentParser(prog="tinyllm eval asr")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--language")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        from tinyllm.eval.mm import basic_normalize, bootstrap_ci, cer, wer, wer_stats
        from tinyllm.io.wav import read_wav
        from tinyllm.speech.decode import DecodeOptions, transcribe
        from tinyllm.speech.decoder import Whisper
        from tinyllm.speech.frontend import FeatureConfig, to_mono_16k
        from tinyllm.speech.tokenizer import WhisperTokenizer

        if args.bootstrap < 1:
            raise ValueError("bootstrap count must be positive")
        manifest = Path(args.data)
        rows, refs = [], []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("audio"), str)
                or not isinstance(row.get("text"), str)
            ):
                raise ValueError("each corpus row needs audio and text strings")
            ref = basic_normalize(row["text"])
            if not ref.split():
                raise ValueError("each corpus reference must contain normalized words")
            path = manifest.parent / row["audio"]
            if not path.is_file():
                raise ValueError(f"corpus audio missing: {path}")
            rows.append(path)
            refs.append(ref)
        if not rows:
            raise ValueError("corpus must contain at least one row")
        model = Whisper.from_pretrained(args.model)
        tok = WhisperTokenizer.from_pretrained(args.model, language=args.language)
        if model.config.vocab != tok.bpe.vocab_size:
            raise ValueError("tokenizer and model vocabulary sizes differ")
        cfg = FeatureConfig(feature_size=model.config.n_mels)
        opts = DecodeOptions(language=args.language, seed=args.seed)
        opts.validate()
        hyps = []
        for path in rows:
            audio = to_mono_16k(read_wav(path.read_bytes()), cfg)
            segments = transcribe(model, tok, audio, opts)
            hyps.append(basic_normalize(" ".join(s.text for s in segments)))
        stats = wer_stats(refs, hyps)
        _, low, high = bootstrap_ci(
            stats,
            lambda sample: float(sample[:, :3].sum()) / float(sample[:, 3].sum()),
            n=args.bootstrap,
            seed=args.seed,
        )
        result = {
            "wer": wer(refs, hyps),
            "cer": cer(refs, hyps),
            "wer_ci": [low, high],
            "items": len(rows),
            "reference_words": int(stats[:, 3].sum()),
            "normalizer": "whisper-basic",
            "seed": args.seed,
            "bootstrap": args.bootstrap,
        }
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"eval asr: {error}", file=sys.stderr)
        return 2
