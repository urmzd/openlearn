"""Build milestone greedy/beam-5 traces with pinned upstream search algorithms.

The learner is never imported. A small vocabulary is embedded in the actual
Whisper vocabulary so the CLI can use its real tokenizer and control tokens.
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import warnings
import zlib

import numpy as np
import torch
from tokenizers import Tokenizer


def generate(root: Path):
    spec = importlib.util.spec_from_file_location(
        "decode_oracle", "course/oracle/L14.4/decode_openai.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    algorithms = module.algorithms()
    tokenizer = Tokenizer.from_file(
        "course/fixtures/L14.4/whisper-tokenizer/tokenizer.json"
    )
    eot = tokenizer.token_to_id("<|endoftext|>")
    prompt = [
        tokenizer.token_to_id(x)
        for x in (
            "<|startoftranscript|>",
            "<|en|>",
            "<|transcribe|>",
            "<|notimestamps|>",
        )
    ]
    vocabulary = [65, 66, 67, 68, 69, 70, eot]
    root.mkdir(parents=True, exist_ok=True)
    for beam in (0, 5):
        name = "greedy" if beam == 0 else "beam5"
        rows = {}
        width = beam or 1
        decoder = (
            algorithms["BeamSearchDecoder"](
                beam,
                6,
                type(
                    "Inference", (), {"rearrange_kv_cache": lambda self, indices: None}
                )(),
                1.0,
            )
            if beam
            else algorithms["GreedyDecoder"](0, 6)
        )
        # Upstream only needs generated tokens, with a harmless start marker.
        tokens = torch.tensor([[7]] * width)
        scores = torch.zeros(width)
        for _ in range(6):
            batch = []
            for prefix in tokens.tolist():
                generated = prefix[1:]
                seed = int.from_bytes(
                    hashlib.sha256(bytes(generated)).digest()[:8], "little"
                )
                row = np.random.default_rng(seed).normal(size=7).astype(np.float32)
                row[6] += len(generated) * 1.2 - 3
                if not generated:
                    # Greedy selects 0, but 1 leads to a much stronger complete
                    # path. Beam-5 must explore and rank that alternative.
                    row = np.array([3, 2.9, -18, -19, -20, -21, -100], dtype=np.float32)
                elif generated[0] == 1:
                    row = np.array([-20, -21, -22, -23, -24, -25, 3], dtype=np.float32)
                rows[tuple(prompt + [vocabulary[i] for i in generated])] = row.copy()
                batch.append(row)
            tokens, done = decoder.update(tokens, torch.tensor(np.stack(batch)), scores)
            if done:
                break
        grouped, totals = decoder.finalize(tokens[None], scores[None])
        candidates = []
        for candidate in grouped[0]:
            ids = candidate.tolist()[1:]
            candidates.append(ids[: ids.index(6)] if 6 in ids else ids)
        selected = algorithms["MaximumLikelihoodRanker"](None).rank(
            [[torch.tensor(ids) for ids in candidates]], totals
        )[0]
        expected = [vocabulary[i] for i in candidates[selected]]
        rows[(prompt[0],)] = np.zeros(7, dtype=np.float32)
        offsets, prefixes, values = [0], [], []
        for prefix, row in sorted(rows.items()):
            prefixes.extend(prefix)
            offsets.append(len(prefixes))
            values.append(row)
        np.savez_compressed(
            root / f"{name}.npz",
            prefix_offsets=np.array(offsets),
            prefix_ids=np.array(prefixes),
            token_ids=np.tile(vocabulary, (len(values), 1)),
            values=np.stack(values),
            vocab_size=np.array(tokenizer.get_vocab_size()),
        )
        options = {
            "tokenizer": "../L14.4/whisper-tokenizer",
            "duration": 0.1,
            "window_frames": 10,
            "decode": {
                "language": "en",
                "timestamps": False,
                "beam_size": beam,
                "max_tokens": 6,
                "temperatures": [0.0],
                "suppress_tokens": [],
                "suppress_blank": False,
            },
        }
        (root / f"{name}.json").write_text(json.dumps(options, indent=2) + "\n")
        (root / f"{name}-expected.json").write_text(
            json.dumps({"ids": expected, "text": tokenizer.decode(expected)}) + "\n"
        )


def fallback_case(module):
    """Let upstream transcribe decide retries and prompt resets for repetition."""
    calls, windows = [], []
    tok = SimpleNamespace(
        eot=5,
        timestamp_begin=10,
        decode=lambda ids: "".join({1: "b", 2: "c"}.get(t, "") for t in ids),
    )

    def pad(values, length):
        return torch.nn.functional.pad(
            values[:, :length], (0, max(0, length - values.shape[-1]))
        )

    namespace = dict(
        torch=torch,
        np=np,
        warnings=warnings,
        tqdm=SimpleNamespace(tqdm=module.Progress),
        N_FRAMES=10,
        N_SAMPLES=1600,
        HOP_LENGTH=160,
        SAMPLE_RATE=16000,
        FRAMES_PER_SECOND=100,
        exact_div=lambda a, b: a // b,
        log_mel_spectrogram=lambda *a, **kw: torch.arange(30).float()[None],
        pad_or_trim=pad,
        get_tokenizer=lambda *a, **kw: tok,
        DecodingOptions=lambda **kw: SimpleNamespace(**kw),
    )
    exec(module.upstream(), namespace)

    def decode(segment, options):
        seek = int(segment[0, 0])
        tokens = [1] * 40 if seek == 0 else [2]
        text = tok.decode(tokens).encode()
        result = dict(
            tokens=tokens,
            temperature=options.temperature,
            avg_logprob=0.0,
            compression_ratio=len(text) / len(zlib.compress(text)),
            no_speech_prob=0.0,
        )
        calls.append(
            dict(
                seek=seek, temperature=options.temperature, prompt=list(options.prompt)
            )
        )
        windows.append(dict(seek=seek, result=result))
        return SimpleNamespace(**result)

    model = SimpleNamespace(
        device=torch.device("cpu"),
        is_multilingual=False,
        num_languages=0,
        dims=SimpleNamespace(n_mels=80, n_audio_ctx=5, n_text_ctx=448),
        decode=decode,
    )
    result = namespace["transcribe"](
        model,
        np.zeros(3200),
        verbose=None,
        fp16=False,
        language="en",
        temperature=(0.0, 0.6),
    )
    assert [call["temperature"] for call in calls] == [0.0, 0.6, 0.0]
    assert calls[-1]["prompt"] == []
    return dict(
        name="fallback-reset",
        duration=0.2,
        window_frames=10,
        condition_on_previous_text=True,
        timestamps=False,
        temperatures=[0.0, 0.6],
        calls=calls,
        windows=windows,
        segments=result["segments"],
    )


def generate_longform(root: Path):
    spec = importlib.util.spec_from_file_location(
        "transcribe_oracle", "course/oracle/L14.4/transcribe_openai.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = Tokenizer.from_file(
        "course/fixtures/L14.4/whisper-tokenizer/tokenizer.json"
    )
    controls = {
        name: tokenizer.token_to_id(f"<|{name}|>")
        for name in (
            "startoftranscript",
            "startofprev",
            "en",
            "transcribe",
            "endoftext",
            "nocaptions",
            "notimestamps",
        )
    }
    timestamp = tokenizer.token_to_id("<|0.00|>")

    def remap(token):
        return (
            timestamp + token - 10
            if token >= 10
            else {1: 65, 2: 66, 3: 67, 5: controls["endoftext"]}[token]
        )

    for case in module.generate()["cases"] + [fallback_case(module)]:
        if case["name"] not in (
            "conditioned",
            "partial-tail",
            "unconditioned",
            "fallback-reset",
        ):
            continue
        name = "longform-" + case["name"]
        rows = {}
        for call, window in zip(case["calls"], case["windows"]):
            seek = window["seek"]
            prefix = [
                controls[key] for key in ("startoftranscript", "en", "transcribe")
            ]
            if case.get("timestamps") is False:
                prefix.append(controls["notimestamps"])
            if call["prompt"]:
                prefix = (
                    [controls["startofprev"]]
                    + [remap(t) for t in call["prompt"][-223:]]
                    + prefix
                )
            rows[(seek, (controls["startoftranscript"],))] = (
                [controls["nocaptions"], 70],
                [-100.0, 0.0],
            )
            for token in [remap(t) for t in window["result"]["tokens"]] + [
                controls["endoftext"]
            ]:
                rows[(seek, tuple(prefix))] = ([token, 70], [0.0, -100.0])
                prefix.append(token)
        offsets, prefixes, seeks, ids, values = [0], [], [], [], []
        for (seek, prefix), (row_ids, row_values) in sorted(rows.items()):
            prefixes.extend(prefix)
            offsets.append(len(prefixes))
            seeks.append(seek)
            ids.append(row_ids)
            values.append(row_values)
        np.savez_compressed(
            root / f"{name}.npz",
            prefix_offsets=np.array(offsets),
            prefix_ids=np.array(prefixes),
            seeks=np.array(seeks),
            token_ids=np.array(ids),
            values=np.array(values, dtype=np.float32),
            vocab_size=np.array(tokenizer.get_vocab_size()),
        )
        options = {
            "tokenizer": "../L14.4/whisper-tokenizer",
            "duration": case["duration"],
            "window_frames": case["window_frames"],
            "decode": {
                "language": "en",
                "temperatures": [0.0],
                "condition_on_previous_text": case["condition_on_previous_text"],
            },
        }
        expected = {
            "ids": [
                remap(t) for segment in case["segments"] for t in segment["tokens"]
            ],
            "text": " ".join(
                tokenizer.decode([remap(t) for t in segment["tokens"] if t < 5]).strip()
                for segment in case["segments"]
            ),
        }
        if case["name"] == "fallback-reset":
            options["decode"].update(
                timestamps=False, temperatures=case["temperatures"], best_of=1
            )
            expected["temperatures"] = [
                segment["temperature"] for segment in case["segments"]
            ]
        (root / f"{name}.json").write_text(json.dumps(options, indent=2) + "\n")
        (root / f"{name}-expected.json").write_text(json.dumps(expected) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bootstrap-missing", action="store_true")
    args = parser.parse_args()
    root = Path("course/fixtures/MS-L14")
    if not args.check:
        generate(root)
        generate_longform(root)
        return
    with TemporaryDirectory(prefix="whisper-recorded-oracle-") as temporary:
        generated = Path(temporary)
        generate(generated)
        generate_longform(generated)
        for path in sorted(generated.iterdir()):
            committed = root / path.name
            if args.bootstrap_missing and not committed.exists():
                root.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, committed)
            if path.suffix == ".npz":
                with (
                    np.load(path, allow_pickle=False) as actual,
                    np.load(committed, allow_pickle=False) as expected,
                ):
                    if set(actual.files) != set(expected.files):
                        raise ValueError(f"trace array keys drift: {path.name}")
                    for key in actual.files:
                        if actual[key].dtype != expected[
                            key
                        ].dtype or not np.array_equal(actual[key], expected[key]):
                            raise ValueError(f"trace array drift: {path.name}:{key}")
            elif json.loads(path.read_text()) != json.loads(committed.read_text()):
                raise ValueError(f"trace JSON drift: {path.name}")
    print("Independent recorded-search fixtures match committed values")


if __name__ == "__main__":
    main()
