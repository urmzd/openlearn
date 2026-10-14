"""Decoding policy boundaries and real cached Whisper prefix queries."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _lib.close import assert_close
from tinyllm.speech.decode import (
    WhisperLogitsSource,
    apply_timestamp_rules,
    language_probs,
    sot_sequence,
)
from tinyllm.speech.decoder import Whisper


def tokenizer():
    return SimpleNamespace(
        sot=6,
        eot=5,
        no_timestamps=9,
        timestamp_begin=10,
        transcribe=7,
        translate=8,
        language="en",
        all_language_codes=("en", "fr"),
        all_language_tokens=(3, 4),
    )


def test_start_controls():
    # WHY: Language selection must not silently reuse a tokenizer's old task.
    # KIND: boundary
    tok = tokenizer()
    assert sot_sequence(tok, "fr", "translate", False) == [6, 4, 8, 9]
    assert sot_sequence(tok, None, "transcribe", True) == [6, 3, 7]
    with pytest.raises(ValueError):
        sot_sequence(tok, "xx", "transcribe", True)
    with pytest.raises(ValueError):
        sot_sequence(tok, "en", "invalid", True)
    tok.all_language_codes = tok.all_language_tokens = ()
    assert sot_sequence(tok, None, "transcribe", False) == [6, 9]
    with pytest.raises(ValueError):
        sot_sequence(tok, "fr", "transcribe", True)


def test_initial_timestamp_bound_and_source_ownership():
    # WHY: Text cannot precede the initial timestamp, even with a dominant logit.
    # KIND: boundary
    row = np.arange(16, dtype=float)
    row[1] = 100
    before = row.copy()
    masked = apply_timestamp_rules(row, [6, 3, 7], 3, tokenizer(), 2)
    assert np.array_equal(row, before)
    assert np.flatnonzero(np.isfinite(masked)).tolist() == [10, 11, 12]


@pytest.mark.parametrize(
    "tokens,allowed",
    [
        ([10], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ([10, 1, 12], [5, 6, 7, 8, 12, 13, 14, 15]),
        ([10, 1, 12, 12], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ([10, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]),
    ],
)
def test_timestamp_pairs_and_progress(tokens, allowed):
    # WHY: Endpoints advance, but adjacent segments may share a boundary.
    # KIND: boundary
    row = np.zeros(16)
    row[:10] = 10  # disable the separate probability-mass forcing condition
    result = apply_timestamp_rules(row, [6] + tokens, 1, tokenizer(), 5)
    assert np.flatnonzero(np.isfinite(result)).tolist() == allowed


def test_timestamp_mass_forces_time_even_when_each_token_is_weaker():
    # WHY: Comparing argmaxes misses a diffuse but dominant timestamp mass.
    # KIND: property
    row = np.zeros(16)
    row[1] = 1
    result = apply_timestamp_rules(row, [6, 10, 1], 1, tokenizer(), 5)
    assert np.isneginf(result[:11]).all()
    assert np.isfinite(result[11:]).all()


def test_no_legal_timestamp_is_an_error():
    # WHY: An all-masked row must not turn into NaNs and an arbitrary token.
    # KIND: boundary
    with pytest.raises(ValueError, match="no legal token"):
        apply_timestamp_rules(np.full(16, -np.inf), [6], 1, tokenizer(), 2)


def test_language_distribution_excludes_dominant_text():
    # WHY: Detection renormalizes among language tokens, not the vocabulary.
    # KIND: property
    class Source:
        def step(self, prefix):
            assert prefix == [6]
            return np.array([100, 0, 0, 0, np.log(3), 0])

    language, probs = language_probs(Source(), tokenizer())
    assert language == "fr"
    assert_close(
        np.array(list(probs.values())), np.array([0.25, 0.75]), atol=1e-12, rtol=0
    )


def test_cached_source_extends_repeats_rewinds_and_switches_branches():
    # WHY: Beam branches cannot inherit another branch's self-attention state.
    # KIND: property
    fix = Path(os.environ["TINYLLM_FIXTURES"])
    model = Whisper.from_pretrained(str(fix / "L14.2/tiny-whisper-128"))
    mel = np.load(fix / "L14.2/encoder_hf.npz")["small.mel"][:1]
    xkv = model.encode(mel)
    source = WhisperLogitsSource(model, xkv)
    for prefix in ([1, 2], [1, 2, 3], [1, 2, 3], [1, 4, 3], [1], [1, 8]):
        expected = model.decoder(np.array(prefix), xkv).data[-1]
        row = source.step(prefix)
        assert_close(row, expected, atol=1e-5, rtol=0)
        row[:] = 0  # callers may mask their returned copy
        assert_close(source.step(prefix), expected, atol=1e-5, rtol=0)
        assert source.cache.seq_len() == len(prefix)
    with pytest.raises(ValueError):
        source.step([1.5])
    assert source.cache.seq_len() == 2


def search_tokenizer():
    tok = tokenizer()
    tok.sot_prev = 16
    tok.non_speech_tokens = (4,)
    tok.encode = lambda text: [0]
    tok.decode = lambda ids: "".join(
        {0: " ", 1: "a", 2: "b", 3: "c", 4: "!"}[i] for i in ids
    )
    return tok


class ScriptedSource:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def step(self, prefix):
        self.calls.append(tuple(prefix))
        # Controls end in no_timestamps in these search tests.
        generated = tuple(prefix[prefix.index(9) + 1 :])
        row = np.full(18, -np.inf)
        for token, value in self.rows[generated].items():
            row[token] = value
        return row


def test_greedy_and_beam_one_match_with_control_suppression():
    # WHY: Beam size one must not explore a different path than greedy.
    # KIND: property
    from _lib.pcg32 import PCG32
    from tinyllm.speech.decode import DecodeOptions, decode_window

    rows = {(): {0: 20, 1: 0, 6: 30, 4: 40}, (1,): {5: 0}}
    for beam in (0, 1):
        result = decode_window(
            ScriptedSource(rows),
            search_tokenizer(),
            [],
            DecodeOptions(timestamps=False, beam_size=beam),
            0,
            PCG32(0),
        )
        assert result.tokens == [1]
        assert result.text == "a"
        assert result.avg_logprob == 0


def test_beam_explores_lower_first_token_and_ranks_complete_paths():
    # WHY: Choosing only the locally largest logit is not beam search.
    # KIND: golden
    from _lib.pcg32 import PCG32
    from tinyllm.speech.decode import DecodeOptions, decode_window

    rows = {(): {1: 0, 2: -0.1}, (1,): {5: 0, 3: -0.01}, (2,): {5: 0}, (1, 3): {5: 0}}
    result = decode_window(
        ScriptedSource(rows),
        search_tokenizer(),
        [],
        DecodeOptions(timestamps=False, beam_size=2, max_tokens=2),
        0,
        PCG32(0),
    )
    assert result.tokens == [2]
    assert result.text == "b"


def test_sampling_is_reproducible_and_scores_untempered_probability():
    # WHY: Temperature changes draws, not the average-logprob fallback statistic.
    # KIND: property
    from _lib.pcg32 import PCG32
    from tinyllm.speech.decode import DecodeOptions, decode_window

    rows = {(): {1: 0, 2: -1}, (1,): {5: 0}, (2,): {5: 0}}
    opts = DecodeOptions(timestamps=False, best_of=3)
    a = decode_window(ScriptedSource(rows), search_tokenizer(), [], opts, 0.5, PCG32(7))
    b = decode_window(ScriptedSource(rows), search_tokenizer(), [], opts, 0.5, PCG32(7))
    assert a == b
    expected = (-np.log1p(np.exp(-1)) - (a.tokens[0] == 2)) / 2
    assert_close(np.array(a.avg_logprob), np.array(expected), atol=1e-12, rtol=0)


def test_prompt_context_and_generation_limit():
    # WHY: Previous text conditions the model but never appears as new output.
    # KIND: boundary
    from _lib.pcg32 import PCG32
    from tinyllm.speech.decode import DecodeOptions, decode_window

    source = ScriptedSource({(): {1: 0}})
    result = decode_window(
        source,
        search_tokenizer(),
        [2, 3],
        DecodeOptions(timestamps=False, max_tokens=1),
        0,
        PCG32(0),
    )
    assert source.calls[0] == (16, 2, 3, 6, 3, 7, 9)
    assert result.tokens == [1]
    assert result.avg_logprob == 0


@pytest.mark.parametrize(
    "change",
    [
        {"beam_size": -1},
        {"max_tokens": 0},
        {"best_of": 1.5},
        {"patience": float("nan")},
        {"temperatures": ()},
        {"no_speech_threshold": 2},
    ],
)
def test_invalid_search_options_fail_before_model_calls(change):
    # WHY: Malformed requests must not mutate decoder state.
    # KIND: boundary
    from _lib.pcg32 import PCG32
    from tinyllm.speech.decode import DecodeOptions, decode_window

    source = ScriptedSource({})
    with pytest.raises(ValueError):
        decode_window(
            source, search_tokenizer(), [], DecodeOptions(**change), 0, PCG32(0)
        )
    assert source.calls == []


def window_result(tokens, *, temperature=0.0, avg_logprob=0.0, ratio=0.1, silence=0.0):
    from tinyllm.speech.decode import Segment

    return Segment(0, 0, 0, 0, "a", tokens, temperature, avg_logprob, ratio, silence)


def test_fallback_retries_quality_but_stops_for_probable_silence(monkeypatch):
    # WHY: Silence must not trigger six expensive retries and hallucinated text.
    # KIND: boundary
    from _lib.pcg32 import PCG32
    from tinyllm.speech import decode

    calls = []

    def window(src, tok, prompt, opts, temperature, rng):
        calls.append(temperature)
        return window_result(
            [1], temperature=temperature, ratio=3 if temperature == 0 else 1
        )

    monkeypatch.setattr(decode, "decode_window", window)
    result = decode.decode_with_fallback(
        None, None, [], decode.DecodeOptions(), PCG32(0)
    )
    assert calls == [0.0, 0.2]
    assert result.temperature == 0.2
    calls.clear()

    def silent(src, tok, prompt, opts, temperature, rng):
        calls.append(temperature)
        return window_result([], avg_logprob=-2, silence=0.9, ratio=3)

    monkeypatch.setattr(decode, "decode_window", silent)
    decode.decode_with_fallback(None, None, [], decode.DecodeOptions(), PCG32(0))
    assert calls == [0.0]


def test_longform_seeks_last_complete_boundary_and_reuses_context(monkeypatch):
    # WHY: Incomplete trailing speech must be retried from its prior boundary.
    # KIND: golden
    from tinyllm.speech import decode

    seeks, prompts = [], []

    def factory(seek):
        seeks.append(seek)
        return seek

    def window(source, tok, prompt, opts, rng):
        prompts.append(list(prompt))
        if source == 0:
            return window_result([10, 1, 12, 12, 2])
        return window_result([10, 2, 13])

    monkeypatch.setattr(decode, "decode_with_fallback", window)
    segments = decode._transcribe_windows(
        factory, search_tokenizer(), 0.1, 10, decode.DecodeOptions(language="en")
    )
    assert seeks == [0, 4]
    assert prompts == [[], [10, 1, 12]]
    assert [(s.id, s.seek, s.text) for s in segments] == [(0, 0, "a"), (1, 4, "b")]
    assert_close(
        np.array([(s.start, s.end) for s in segments]),
        np.array([[0, 0.04], [0.04, 0.1]]),
        atol=1e-12,
        rtol=0,
    )


def test_longform_clears_hot_context_and_bounds_final_segment(monkeypatch):
    # WHY: Fallback hallucinations must not condition subsequent windows.
    # KIND: boundary
    from tinyllm.speech import decode

    prompts = []

    def window(source, tok, prompt, opts, rng):
        prompts.append(list(prompt))
        return window_result([10, 1, 25], temperature=0.6)

    monkeypatch.setattr(decode, "decode_with_fallback", window)
    segments = decode._transcribe_windows(
        lambda seek: seek,
        search_tokenizer(),
        0.15,
        10,
        decode.DecodeOptions(language="en"),
    )
    assert prompts == [[], []]
    assert len(segments) == 2
    assert_close(
        np.array([segments[0].end, segments[1].end]),
        np.array([0.1, 0.15]),
        atol=1e-12,
        rtol=0,
    )


def test_longform_skips_silence_but_preserves_confident_speech(monkeypatch):
    # WHY: High no-speech probability alone cannot veto confident transcription.
    # KIND: boundary
    from tinyllm.speech import decode

    seeks = []

    def window(source, tok, prompt, opts, rng):
        seeks.append(source)
        return window_result(
            [10, 1, 15], avg_logprob=-2 if source == 0 else 0, silence=0.9
        )

    monkeypatch.setattr(decode, "decode_with_fallback", window)
    segments = decode._transcribe_windows(
        lambda seek: seek,
        search_tokenizer(),
        0.2,
        10,
        decode.DecodeOptions(language="en"),
    )
    assert seeks == [0, 10]
    assert len(segments) == 1
    assert segments[0].seek == 10


def test_zero_timestamp_does_not_stall_seek(monkeypatch):
    # WHY: A zero-duration or truncated decode must always advance the window.
    # KIND: boundary
    from tinyllm.speech import decode

    seeks = []

    def window(source, tok, prompt, opts, rng):
        seeks.append(source)
        return window_result([10, 1, 10, 10])

    monkeypatch.setattr(decode, "decode_with_fallback", window)
    assert (
        decode._transcribe_windows(
            lambda seek: seek,
            search_tokenizer(),
            0.2,
            10,
            decode.DecodeOptions(language="en"),
        )
        == []
    )
    assert seeks == [0, 10]


def test_transcribe_real_model_encodes_each_window_once(monkeypatch):
    # WHY: The long-form path must actually connect audio features and Whisper.
    # KIND: property
    from tinyllm.speech.decode import DecodeOptions, transcribe

    fix = Path(os.environ["TINYLLM_FIXTURES"])
    model = Whisper.from_pretrained(str(fix / "L14.2/tiny-whisper-128"))
    encode = model.encode
    shapes = []

    def counted(mel):
        shapes.append(mel.shape)
        return encode(mel)

    monkeypatch.setattr(model, "encode", counted)
    samples = 0.1 * np.sin(np.arange(5120) * (2 * np.pi * 440 / 16000))
    segments = transcribe(
        model,
        search_tokenizer(),
        samples,
        DecodeOptions(
            language="en",
            timestamps=False,
            max_tokens=2,
            temperatures=(0.0, 0.2),
            logprob_threshold=-100,
            compression_ratio_threshold=100,
        ),
    )
    assert shapes == [(1, 128, 16), (1, 128, 16)]
    assert segments
    assert all(0 <= s.start < s.end <= 0.32 for s in segments)
    assert [s.id for s in segments] == list(range(len(segments)))


def test_independent_openai_timestamp_and_search_fixture():
    # WHY: Internal consistency cannot detect a shared search-policy mistake.
    # KIND: golden
    import json
    from _lib.pcg32 import PCG32
    from tinyllm.speech.decode import DecodeOptions, decode_window

    fixture = json.loads(
        (Path(os.environ["TINYLLM_FIXTURES"]) / "L14.4/decode_openai.json").read_text()
    )
    for case in fixture["timestamps"]:
        row = apply_timestamp_rules(
            np.array(case["logits"]),
            case["prefix"],
            case["sample_begin"],
            tokenizer(),
            case["max_initial_index"],
        )
        assert np.flatnonzero(np.isfinite(row)).tolist() == case["allowed"]
    for case in fixture["searches"]:

        class Recorded:
            def step(self, prefix):
                return np.array(case["rows"][",".join(map(str, prefix))])

        result = decode_window(
            Recorded(),
            search_tokenizer(),
            [],
            DecodeOptions(
                timestamps=False,
                beam_size=case["beam"],
                patience=case["patience"],
                max_tokens=case["max_tokens"],
                suppress_tokens=(),
                suppress_blank=False,
            ),
            0,
            PCG32(0),
        )
        assert result.tokens == case["tokens"]
        assert_close(
            np.array(result.avg_logprob),
            np.array(case["avg_logprob"]),
            atol=1e-6,
            rtol=0,
        )


def test_independent_openai_longform_policy_fixture(monkeypatch):
    # WHY: Seek, fallback and prompt decisions need an oracle outside this implementation.
    # KIND: golden
    import json
    from dataclasses import asdict
    from tinyllm.speech import decode

    root = Path(os.environ["TINYLLM_FIXTURES"]) / "L14.4"
    fixture = json.loads((root / "transcribe_openai.json").read_text())
    for case in fixture["cases"]:
        calls = []
        windows = iter(case["windows"])

        def recorded_window(source, tok, prompt, opts, temperature, rng):
            expected = next(windows)
            assert source == expected["seek"]
            assert temperature == expected["result"]["temperature"]
            calls.append(
                {"seek": source, "temperature": temperature, "prompt": list(prompt)}
            )
            return decode.Segment(
                id=0, seek=source, start=0, end=0, text="", **expected["result"]
            )

        monkeypatch.setattr(decode, "decode_window", recorded_window)
        segments = decode._transcribe_windows(
            lambda seek: seek,
            search_tokenizer(),
            case["duration"],
            case["window_frames"],
            decode.DecodeOptions(
                language="en",
                condition_on_previous_text=case["condition_on_previous_text"],
            ),
        )
        assert calls == case["calls"], case["name"]
        assert [asdict(segment) for segment in segments] == case["segments"], case[
            "name"
        ]
        assert next(windows, None) is None
