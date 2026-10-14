"""WAV transcription through the learner's Whisper model and decoding policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def intercept(argv: list[str]) -> int | None:
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
