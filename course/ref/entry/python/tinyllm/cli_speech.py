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
