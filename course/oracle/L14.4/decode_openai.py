"""Independent timestamp/beam fixtures from pinned official OpenAI Whisper.

Only upstream algorithm classes execute here. No learner/reference import is
permitted. CPU torch and numpy versions are pinned by oracle-drift.yml.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F

REVISION = "86098128c0b4f24f0e2aa2994de830614b474227"
SOURCE_SHA256 = "94084fedc5af74cabf8e1883a85552700057081f00ea669ad49435ecb29039db"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/openai/whisper/{REVISION}/whisper/decoding.py"
)


def algorithms():
    with urlopen(SOURCE_URL, timeout=30) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("upstream source fingerprint changed")
    names = {
        "TokenDecoder",
        "GreedyDecoder",
        "BeamSearchDecoder",
        "LogitFilter",
        "ApplyTimestampRules",
        "SequenceRanker",
        "MaximumLikelihoodRanker",
    }
    classes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if {node.name for node in classes} != names:
        raise ValueError("upstream oracle classes are missing")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + classes,
        type_ignores=[],
    )
    namespace = {"torch": torch, "np": np, "F": F}
    exec(compile(ast.fix_missing_locations(module), SOURCE_URL, "exec"), namespace)
    return namespace


def generate():
    torch.set_num_threads(1)
    oracle = algorithms()
    tok = SimpleNamespace(eot=5, no_timestamps=9, timestamp_begin=10)
    timestamp_cases = []
    prefixes = [[], [10], [10, 1], [10, 1, 12], [10, 1, 12, 12], [10, 1, 12, 13, 2]]
    for seed in range(4):
        for generated in prefixes:
            row = np.random.default_rng(seed).normal(size=16).astype(np.float32)
            output = torch.tensor(row[None].copy())
            oracle["ApplyTimestampRules"](tok, 3, seed).apply(
                output, torch.tensor([[6, 3, 7] + generated])
            )
            timestamp_cases.append(
                {
                    "prefix": [6, 3, 7] + generated,
                    "sample_begin": 3,
                    "max_initial_index": seed,
                    "logits": row.tolist(),
                    "allowed": torch.isfinite(output[0]).nonzero().flatten().tolist(),
                }
            )
    searches = []
    for beam, patience in [(0, 1.0), (2, 1.0), (2, 2.0), (3, 1.0)]:
        rows = {}
        initial = [6, 3, 7, 9]
        width = beam or 1
        decoder = (
            oracle["BeamSearchDecoder"](
                beam,
                5,
                SimpleNamespace(rearrange_kv_cache=lambda indices: None),
                patience,
            )
            if beam
            else oracle["GreedyDecoder"](0, 5)
        )
        tokens = torch.tensor([initial] * width)
        scores = torch.zeros(width)
        for _ in range(6):
            batch = []
            for prefix in tokens.tolist():
                generated = prefix[len(initial) :]
                digest = hashlib.sha256(bytes(generated)).digest()
                seed = int.from_bytes(digest[:8], "little")
                row = np.full(18, -100, dtype=np.float32)
                row[:6] = np.random.default_rng(seed).normal(size=6)
                row[5] += len(generated) * 1.2 - 3
                rows[",".join(map(str, prefix))] = row.tolist()
                row[6:] = -np.inf
                batch.append(row)
            tokens, done = decoder.update(tokens, torch.tensor(np.stack(batch)), scores)
            if done:
                break
        grouped, sums = decoder.finalize(tokens[None], scores[None])
        candidates = []
        for candidate in grouped[0]:
            ids = candidate.tolist()[len(initial) :]
            candidates.append(ids[: ids.index(5)] if 5 in ids else ids)
        ranker = oracle["MaximumLikelihoodRanker"](None)
        chosen = ranker.rank([[torch.tensor(ids) for ids in candidates]], sums)[0]
        selected = candidates[chosen]
        searches.append(
            {
                "beam": beam,
                "patience": patience,
                "max_tokens": 6,
                "rows": rows,
                "tokens": selected,
                "avg_logprob": sums[0][chosen] / (len(selected) + 1),
            }
        )
    return {
        "upstream_revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "timestamps": timestamp_cases,
        "searches": searches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = generate()
    path = Path("course/fixtures/L14.4/decode_openai.json")
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if json.loads(path.read_text()) != result:
            raise ValueError("independent Whisper decoding fixture drift")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    print(
        f"{path}\t{len(encoded.encode())}\t{hashlib.sha256(encoded.encode()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
