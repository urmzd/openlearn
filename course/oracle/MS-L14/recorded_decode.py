"""Build milestone greedy/beam-5 traces with pinned upstream search algorithms.

The learner is never imported. A small vocabulary is embedded in the actual
Whisper vocabulary so the CLI can use its real tokenizer and control tokens.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path("course/fixtures/MS-L14")
    if not args.check:
        generate(root)
        return
    with TemporaryDirectory(prefix="whisper-recorded-oracle-") as temporary:
        generated = Path(temporary)
        generate(generated)
        for path in sorted(generated.iterdir()):
            committed = root / path.name
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
