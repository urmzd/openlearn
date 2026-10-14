"""Whisper controls over the learner's byte-level BPE tokenizer (L14.4)."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Sequence

from tinyllm.tok.bpe import BPETokenizer
from tinyllm.tok.bytes_unicode import bytes_to_unicode


class WhisperTokenizer:
    def __init__(
        self,
        bpe: BPETokenizer,
        *,
        multilingual: bool = True,
        language: str | None = None,
        suppress_tokens: Sequence[int] | None = None,
    ) -> None:
        # SOLUTION-BEGIN L14.4
        self.bpe = bpe
        self.language = language
        controls = {
            "eot": "endoftext",
            "sot": "startoftranscript",
            "sot_prev": "startofprev",
            "sot_lm": "startoflm",
            "transcribe": "transcribe",
            "translate": "translate",
            "no_speech": "nospeech",
            "no_timestamps": "notimestamps",
            "timestamp_begin": "0.00",
        }
        for name, label in controls.items():
            value = bpe.token_to_id(f"<|{label}|>")
            if name == "no_speech" and value is None:
                value = bpe.token_to_id("<|nocaptions|>")
            if value is None:
                raise ValueError(f"missing Whisper control token: {label}")
            setattr(self, name, value)
        if (
            not self.eot < self.sot < self.timestamp_begin
            or self.no_timestamps != self.timestamp_begin - 1
        ):
            raise ValueError("invalid Whisper control-token ordering")
        # Timestamp arithmetic is a wire contract, not a tokenizer heuristic.
        for i in range(1501):
            if bpe.token_to_id(f"<|{i / 50:.2f}|>") != self.timestamp_begin + i:
                raise ValueError(
                    "Whisper timestamps must be contiguous at 20 ms spacing"
                )
        if bpe.vocab_size != self.timestamp_begin + 1501:
            raise ValueError("timestamp tokens must occupy the vocabulary suffix")
        if any(
            bpe.token_to_id(symbol) is None for symbol in bytes_to_unicode().values()
        ):
            raise ValueError("Whisper BPE must represent every byte")
        all_tokens = {**bpe.vocab, **bpe.added}
        languages = (
            sorted(
                (
                    (i, token[2:-2])
                    for token, i in all_tokens.items()
                    if re.fullmatch(r"<\|[a-z]{2,3}\|>", token)
                ),
                key=lambda pair: pair[0],
            )
            if multilingual
            else []
        )
        self.all_language_tokens = tuple(i for i, _ in languages)
        self.all_language_codes = tuple(code for _, code in languages)
        if multilingual and not languages:
            raise ValueError("multilingual tokenizer has no language controls")
        if language is not None and language not in (
            self.all_language_codes if multilingual else ("en",)
        ):
            raise ValueError(f"unsupported language: {language}")
        if suppress_tokens is not None:
            if any(
                type(t) is not int or not 0 <= t < bpe.vocab_size
                for t in suppress_tokens
            ):
                raise ValueError("invalid suppression token id")
            self.non_speech_tokens = tuple(sorted(set(suppress_tokens)))
        else:
            self.non_speech_tokens = self._non_speech()
        # SOLUTION-END

    def encode(self, text: str) -> list[int]:
        # SOLUTION-BEGIN L14.4
        # Text supplied as previous context is ordinary text, never executable
        # tokenizer control syntax. Reject explicit controls instead of injecting.
        if any(token in text for token in self.bpe.added):
            raise ValueError("ordinary transcript text contains a special token")
        return self.bpe.encode(text)
        # SOLUTION-END

    def decode(self, ids: Sequence[int]) -> str:
        # SOLUTION-BEGIN L14.4
        return self.bpe.decode([int(t) for t in ids if t < self.timestamp_begin])
        # SOLUTION-END

    def _non_speech(self) -> tuple[int, ...]:
        # SOLUTION-BEGIN L14.4
        # Speaker-label punctuation and music annotations, preserving normal
        # sentence punctuation. Musical UTF-8 prefixes may be multi-token.
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols.extend(
            [
                "<<",
                ">>",
                "<<<",
                ">>>",
                "--",
                "---",
                "-(",
                "-[",
                "('",
                '("',
                "((",
                "))",
                "(((",
                ")))",
                "[[",
                "]]",
                "{{",
                "}}",
                "♪♪",
                "♪♪♪",
            ]
        )
        music = set("♩♪♫♬♭♮♯")
        blocked = {self.encode(" -")[0], self.encode(" '")[0]}
        for symbol in symbols + sorted(music):
            for text in (symbol, " " + symbol):
                ids = self.encode(text)
                if len(ids) == 1 or symbol in music:
                    blocked.add(ids[0])
        return tuple(sorted(blocked))
        # SOLUTION-END

    @classmethod
    def from_pretrained(
        cls, directory: str, *, language: str | None = None
    ) -> WhisperTokenizer:
        # SOLUTION-BEGIN L14.4
        root = Path(directory)
        doc = json.loads((root / "tokenizer.json").read_text())
        if doc.get("version") != "1.0" or doc.get("normalizer") is not None:
            raise ValueError(
                "Whisper requires an unnormalized tokenizer.json version 1.0"
            )
        model = doc.get("model", {})
        if (
            model.get("type") != "BPE"
            or model.get("dropout") is not None
            or model.get("unk_token") is not None
        ):
            raise ValueError(
                "Whisper requires deterministic byte BPE without unknown tokens"
            )
        for key in (
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "fuse_unk",
            "byte_fallback",
            "ignore_merges",
        ):
            if model.get(key) not in (None, "", False):
                raise ValueError(f"unsupported Whisper BPE setting: {key}")
        if (doc.get("decoder") or {}).get("type") != "ByteLevel":
            raise ValueError("Whisper requires a byte-level decoder")
        pre = doc.get("pre_tokenizer") or {}
        if pre.get("type") != "ByteLevel" or pre.get("add_prefix_space", False):
            raise ValueError(
                "Whisper requires ByteLevel without automatic prefix space"
            )
        post = doc.get("post_processor") or {}
        if post.get("type") not in (None, "TemplateProcessing", "ByteLevel"):
            raise ValueError("unsupported tokenizer post-processor")
        # Templates are deliberately not executed: sot_sequence supplies task
        # and language controls, and generated tokens never need an added EOT.
        merges = []
        for entry in model.get("merges", []):
            pair = entry.split(" ") if isinstance(entry, str) else entry
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(x, str) for x in pair)
            ):
                raise ValueError("invalid BPE merge")
            merges.append(tuple(pair))
        added = []
        for entry in doc.get("added_tokens", []):
            if any(
                entry.get(key, False) for key in ("single_word", "lstrip", "rstrip")
            ):
                raise ValueError("Whisper controls cannot strip or match partial words")
            added.append((entry["content"], entry["id"], entry.get("special", False)))
        bpe = BPETokenizer(model["vocab"], merges, added, pre)
        generation_path = root / "generation_config.json"
        generation = (
            json.loads(generation_path.read_text()) if generation_path.exists() else {}
        )
        multilingual = generation.get("is_multilingual", bpe.vocab_size != 51864)
        if type(multilingual) is not bool:
            raise ValueError("is_multilingual must be a boolean")
        return cls(
            bpe,
            multilingual=multilingual,
            language=language,
            suppress_tokens=generation.get("suppress_tokens"),
        )
        # SOLUTION-END
