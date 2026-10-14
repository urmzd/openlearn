"""Run pinned OpenAI long-form policy over scripted decoded audio windows.

Model scores and features are inputs to this oracle, not learner outputs. The
upstream transcribe function itself makes fallback, seek and context decisions.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen
import warnings

import numpy as np
import torch

REVISION = "86098128c0b4f24f0e2aa2994de830614b474227"
SHA256 = "176f5ab30e961bec2a7872f5701a1f5bf3cbfc77946e50512f0f0cf00a1ceacf"
URL = (
    f"https://raw.githubusercontent.com/openai/whisper/{REVISION}/whisper/transcribe.py"
)


class Progress:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, n):
        pass


def upstream():
    with urlopen(URL, timeout=30) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise ValueError("upstream transcribe fingerprint changed")
    body = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "transcribe"
    ]
    if len(body) != 1:
        raise ValueError("upstream transcribe missing")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + body,
        type_ignores=[],
    )
    return compile(ast.fix_missing_locations(module), URL, "exec")


def generate():
    code = upstream()
    cases = []
    for name in (
        "conditioned",
        "partial-tail",
        "hot-fallback",
        "silence-override",
        "unconditioned",
    ):
        frames = 10 if name == "partial-tail" else 20
        calls, windows = [], []
        tok = SimpleNamespace(
            eot=5,
            timestamp_begin=10,
            decode=lambda ids: "".join(
                {1: "a", 2: "b", 3: "c"}.get(t, "") for t in ids
            ),
        )

        def pad(values, length):
            return torch.nn.functional.pad(
                values[:, :length], (0, max(0, length - values.shape[-1]))
            )

        namespace = dict(
            torch=torch,
            np=np,
            warnings=warnings,
            tqdm=SimpleNamespace(tqdm=Progress),
            N_FRAMES=10,
            N_SAMPLES=1600,
            HOP_LENGTH=160,
            SAMPLE_RATE=16000,
            FRAMES_PER_SECOND=100,
            exact_div=lambda a, b: a // b,
            log_mel_spectrogram=lambda audio, n_mels, padding: torch.arange(
                frames + 10
            ).float()[None],
            pad_or_trim=pad,
            get_tokenizer=lambda *args, **kwargs: tok,
            DecodingOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        exec(code, namespace)

        def decode(segment, opts):
            seek = int(segment[0, 0])
            temperature = opts.temperature
            calls.append(
                dict(seek=seek, temperature=temperature, prompt=list(opts.prompt))
            )
            tokens = [10, 1 if seek == 0 else 2, 15]
            avg, ratio, silence = 0.0, 0.1, 0.0
            if name == "partial-tail":
                tokens = [10, 1, 12, 12, 2] if seek == 0 else [10, 2, 13]
            if name == "hot-fallback" and seek == 0 and temperature < 0.6:
                ratio = 3.0
            if name == "silence-override":
                silence = 0.9
                if seek == 0:
                    avg, tokens = -2.0, []
            result = dict(
                tokens=tokens,
                temperature=temperature,
                avg_logprob=avg,
                compression_ratio=ratio,
                no_speech_prob=silence,
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
            np.zeros(frames * 160),
            verbose=None,
            fp16=False,
            language="en",
            condition_on_previous_text=name != "unconditioned",
        )
        cases.append(
            dict(
                name=name,
                duration=frames / 100,
                window_frames=10,
                condition_on_previous_text=name != "unconditioned",
                windows=windows,
                calls=calls,
                segments=result["segments"],
            )
        )
    return dict(upstream_revision=REVISION, source_sha256=SHA256, cases=cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = generate()
    path = Path("course/fixtures/L14.4/transcribe_openai.json")
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if json.loads(path.read_text()) != result:
            raise ValueError("independent long-form policy drift")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    print(
        f"{path}\t{len(encoded.encode())}\t{hashlib.sha256(encoded.encode()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
