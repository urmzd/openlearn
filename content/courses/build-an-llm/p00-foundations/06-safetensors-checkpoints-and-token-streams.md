<!-- ol:module L0.6 -->
# Safetensors for every dtype, atomic checkpoints, and the token stream

## Overview

| | |
|---|---|
| **Module** | `L0.6` · build · Python · Pass 2 · 5 to 7 h |
| **You build** | you take over `python/tinyllm/io/safetensors.py` from `L0.0` and add F16, BF16, F8_E4M3, I8, U8, I32 plus `read_header`; `python/tinyllm/io/checkpoint.py`: `save_checkpoint`, `load_checkpoint`, `verify_step_dir`; `python/tinyllm/io/tokens.py`: `read_tokens_header`, `open_tokens`, `TokenStream` |
| **Contract** | [`course/contracts/py/tinyllm/io/safetensors.pyi`](../../../../course/contracts/py/tinyllm/io/safetensors.pyi) · [`course/contracts/py/tinyllm/io/checkpoint.pyi`](../../../../course/contracts/py/tinyllm/io/checkpoint.pyi) · [`course/contracts/py/tinyllm/io/tokens.pyi`](../../../../course/contracts/py/tinyllm/io/tokens.pyi) · formats: [`safetensors.md`](../../../../course/contracts/formats/safetensors.md), [`checkpoint.md`](../../../../course/contracts/formats/checkpoint.md), [`tokens-bin.md`](../../../../course/contracts/formats/tokens-bin.md) |
| **Tests** | `course/tests/L0.6/` (what they check: section 4), golden files from the pinned `safetensors` library in `course/fixtures/L0.6/safetensors/`; `L0.0`'s safetensors tests keep running as your regression suite |
| **Needs** | `M09.1` the BF16 bit converters · `L0.1`, `L0.2`, `L0.4` the model the checkpoint tests train · `M10.2` SGD and `M10.3` AdamW (optimizer state to save) · `M06.3` the PCG32 whose state the token cursor carries · `L0.5` its `bigram.py` (`L0.0`'s tests, your regression suite, exercise it) · reading: `L0.0` (or `--ref-deps`) |
| **Used by** | `L10.0` your engine reads the file this writer produces (inherited from `L0.0` with `safetensors.py`) · `L2.1` the n-gram model saves and loads its tables with `save_safetensors` and `load_safetensors` · later: `L2.2` reads token streams, `L7.9` loads BF16 HF weights, `L10.1` memory-maps the same layout, the capstone trainer and `dur.09` resume from these checkpoints · later: `L3.6`, `L4.1`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5`, `L6.7`  · `L14.3` Whisper decoding |
| **Milestone** | `MS-L0` (step 4: a run killed after step 60 resumes from step 50 and ends bitwise equal to an uninterrupted one) |
| **Optional depth** | the safetensors README and `safetensors/src/tensor.rs`; Micikevicius et al., "FP8 Formats for Deep Learning" (2022); Pillai et al., "All File Systems Are Not Created Equal" (OSDI 2014) on crash consistency |

## Key Takeaways

- One safetensors writer for seven dtypes, byte-identical to the reference library: tensors laid out by dtype first, then by name (`test_matches_library_bytes`, `test_dtype_order_then_name`).
- BF16 and F8_E4M3 have no numpy type, so they travel as float32 values rounded to nearest, ties to even, or as raw bits written untouched; F8_E4M3 has no infinity, so a value beyond 448 is an error, not a saturation (`test_bf16_rounds_to_nearest_even`, `test_f8_rounds_to_nearest_even`, `test_f8_rejects_out_of_range`).
- A checkpoint is written into a `.tmp` directory, every file fsynced, the manifest last, then renamed into place, then LATEST replaced by a rename: a crash at any instant leaves the old checkpoint or the new one (`test_crash_at_every_write_keeps_a_valid_checkpoint`).
- Loading checks every file against the manifest and falls back to the newest older checkpoint that verifies (`test_load_skips_incomplete_and_corrupt`).
- The token stream's cursor (shard, offset, generator state) is its whole position: N batches, save, restore, M batches equals N + M batches (`test_cursor_restore_is_bitwise`, `test_resume_is_bitwise`).

## How to work this chapter

```bash
ol start L0.6              # stubs checkpoint.py and tokens.py; safetensors.py is yours: ol start prints its contract diff
ol tests L0.6              # read the test catalog first: rung R0
ol check L0.6              # also reruns L0.0's smoke tests against your safetensors.py
ol check L0.6 --ref-deps   # only if a dependency is not passing yet
ol diff  L0.6              # after passing: your code against the reference
```

Your CLI's `train bigram --method autograd` gains the token-stream form `MS-L0` runs: `--data shard.bin --max-steps N --batch B --seq-len T --ckpt-every K [--resume]` reads windows through `TokenStream`, writes `save_checkpoint(<out>/ckpt, ...)` every K steps with `extra = {"tokens_seen", "data_cursor", "lr", "config_sha256", "git_sha"}` and the cursor's generator state as `rng_state`, and with `--resume` restores model, optimizer, step, and cursor from `load_checkpoint(<out>/ckpt)`. It evaluates the failpoint `train/after-step` after every step (the kata of `craft.03` parses it) and ends with the `final.json` line `MS-L0` compares byte for byte.

---

## 1. Why now

Three things in your system cannot happen yet. Your safetensors writer (`L0.0`) knows only F32, but SmolLM2's weights (`L7.9`) are BF16 and quantized weights (`M09.4`, `L8.5`) are F8_E4M3 and U8. Your training runs keep their state in memory: `MS-L0`'s fourth step kills one after step 60, and today it would restart from zero, because nothing on disk says where it was, what AdamW's moments were, or which window comes next. And the bigram trains on a byte string read whole into memory; the capstone's TinyStories tokens are hundreds of megabytes in llm.c's `.bin` format. This module writes every dtype, saves a training run so that a crash at any instant leaves a usable checkpoint, and reads token shards through a memory map with a cursor that resumes exactly.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $b_{31} \dots b_0$ | the 32 bits of a float32: sign, 8 exponent bits, 23 mantissa bits | |
| $\mathrm{BF16}(x)$ | the top 16 bits of float32 $x$ after rounding | `uint16` |
| $s, e, m$ | F8_E4M3 fields: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits | ints |
| $v(c)$ | the value of an F8_E4M3 code $c$ | float |
| $T$ | `seq_len`, tokens per training window input | int |
| $B$ | windows per batch | int |
| $n_k$ | tokens in shard $k$ | int |
| $\phi$ | the phase of a pass, the offset of its first window | int in $[0, \min(T, n - T))$ |
| $o$ | the cursor offset: where the next window starts | int |
| `step-` $nnnnnn$ | a checkpoint directory, six digits of the optimizer step | path |

**Many dtypes, one layout.** A safetensors file is unchanged from `L0.0`: an 8-byte header length, compact JSON, the tensor bytes back to back. What changes is the order when dtypes mix. The pinned library sorts tensors by dtype first, larger alignment first (F32, I32, BF16, F16, F8_E4M3, I8, U8 among the course dtypes), then by name as UTF-8 bytes (writer rule 1 of `formats/safetensors.md`). Name order alone gives the same file only when every tensor has one dtype, which is why `L0.0`'s goldens never saw the difference. A dtype comes from the array (float32 F32, float16 F16, int8 I8, uint8 U8, int32 I32) unless `dtypes={name: ...}` says otherwise; float64 is never converted silently, because a float64 checkpoint read as F32 by an engine is a different model.

**BF16 is the top half of a float32.** BF16 keeps float32's sign and 8-bit exponent and only 7 of its 23 mantissa bits, so its bits are $b_{31} \dots b_{16}$ of the float32. Dropping the low 16 bits by truncation rounds every weight toward zero; the rule is round to nearest, ties to even, which `M09.1`'s `f32_to_bf16_bits` implements by adding `0x7FFF` plus the lowest kept bit before shifting. Decoding is exact: shift the 16 bits back to the top of a float32. So the reader returns BF16 tensors as float32 arrays, and writing those values back as BF16 reproduces the file.

**F8_E4M3 is a table of 256 codes.** The "fn" variant (finite, NaN only) has bias 7 and no infinity:

| $e$ | value |
|---|---|
| 0 (subnormal) | $(-1)^s \cdot \frac{m}{8} \cdot 2^{-6}$ |
| 1 to 15 | $(-1)^s \cdot (1 + \frac{m}{8}) \cdot 2^{e - 7}$, except $e = 15, m = 7$ |
| 15 with $m = 7$ | NaN (`0x7F`, `0xFF`) |

The largest value is $e = 15, m = 6$: $1.75 \cdot 2^8 = 448$. The smallest positive is $2^{-9}$ (code `0x01`). Unlike IEEE formats, exponent 15 is not reserved for infinity, which is the habit to unlearn. To encode, round to the nearest code, ties to the even one (its last mantissa bit is 0; consecutive codes alternate that bit, across exponents too). A value beyond 448 has no code to round to: raise, so the missing scale is found (`M09.4` adds the scale); NaN encodes as `0x7F`.

**Read the header without the data.** `read_header` checks all five reader rules from the header and the file size alone (rule 5, the tiling, needs only the offsets and the size) and returns each tensor's dtype and shape. It is what tells BF16 from F32 after `load_safetensors` decoded both to float32, and it is how a memory-mapping reader (`L10.1`) checks a file before trusting its offsets.

**What a checkpoint holds.** `formats/checkpoint.md`: `model.safetensors` (the `state_dict`, F32), `optimizer.safetensors` (each optimizer array named `<parameter>.<state key>`, for AdamW `fc.weight.exp_avg` and `fc.weight.exp_avg_sq`, for SGD `fc.weight.momentum_buffer`, with the rest of the optimizer's state, its step count and hyperparameters, as JSON in the metadata key `state`), `trainer_state.json` (step, tokens seen, the data cursor, the generator state as 16 hex digits each so no JSON reader rounds a 64-bit integer, the learning rate, the config's sha256, the git sha), `config.json` when there is one, and `MANIFEST.json`: every other file with its sha256 and size, sorted by name.

**Writing atomically.** A crash can stop a program between any two system calls, and until `fsync` returns, written bytes may exist only in the page cache. The protocol:

1. write every file into `step-<n>.tmp/` and `fsync` each;
2. write `MANIFEST.json` last, `fsync` it, `fsync` the `.tmp` directory;
3. `rename` the directory to `step-<n>` and `fsync` the parent (a rename is atomic: the directory appears whole or not at all);
4. write `LATEST.tmp` with `step-<n>\n`, `fsync`, `rename` it to `LATEST`, `fsync` the parent;
5. only then delete stale `.tmp` directories and, with `keep`, old steps.

So at every instant each `step-<n>` directory is complete and `LATEST` names one of them. Deleting old steps before step 4 would leave nothing to resume from if the save then failed.

**Loading defensively.** `load_checkpoint(dir)` starts from the directory `LATEST` names and checks it against its manifest: every listed file present with its size and sha256, and no file the manifest does not list. A directory that fails is skipped for the newest older one that passes. Directories newer than `LATEST` are ignored: they were never declared done. With no `LATEST`, the newest complete directory wins; with none complete, `FileNotFoundError`.

**The token stream.** A shard is a `formats/tokens-bin.md` file: a 1024-byte header of 256 little-endian int32 (`20240520`, version, $n$, vocab size), then $n$ ids as `uint16` (version 1) or `uint32` (version 2). `open_tokens` validates the header and the file size and maps the ids with `np.memmap`, read-only, so a 100M-token shard costs nothing until a window is read. A window is $T + 1$ consecutive ids: inputs are the first $T$, targets the last $T$ (the next token of each input). Windows of one pass start $T$ apart, so each window's last id is the next one's first and every token after the phase is a target exactly once. A pass over shard $k$ starts at a random phase $\phi$ = `rng.below(min(T, n_k - T))` (the bound keeps one window in range on a short shard), so window boundaries move from pass to pass, and runs until the next window would pass the end; then the next shard (wrapping) starts a new pass with a new phase. The generator is used only at the start of a pass.

**The cursor is the whole position.** `cursor()` returns the shard index, the offset of the **next** window, and the generator's state. Restoring all three into a stream over the same shards continues bit for bit; without the generator state the next pass draws a different phase and the resumed run diverges silently.

## 3. Worked example by hand

**A BF16 file.** The format page's tensor $w = [[1, 2], [3, 4]]$ with metadata `{"format": "tinyllm"}`, written as BF16.

1. $1.0$ is float32 `0x3F800000`; its top half is `0x3F80`, stored little-endian as `80 3f`. Likewise $2.0 \to$ `0x4000` (`00 40`), $3.0 \to$ `0x4040` (`40 40`), $4.0 \to$ `0x4080` (`80 40`). All four are exact: their low 16 bits are zero.
2. The header `{"__metadata__":{"format":"tinyllm"},"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}` is 93 bytes (the F32 example's, with `BF16` one byte longer and `8` one shorter than `16`); three spaces pad it to $N = 96$.
3. The file: `60 00 00 00 00 00 00 00`, the 96 header bytes, then `80 3f 00 40 40 40 80 40`: $8 + 96 + 8 = 112$ bytes.

**Rounding.** $1 + 2^{-8}$ (float32 `0x3F808000`) sits exactly halfway between BF16 `0x3F80` ($1$) and `0x3F81` ($1 + 2^{-7}$): the tie goes to the even code, `0x3F80`. In F8_E4M3, $1.0$ is $e = 7, m = 0$, code `0x38`, and $1.125$ is `0x39`; $1.0625$ is halfway and becomes `0x38`. $250$ lies between $240$ (`0x77`) and $256$ (`0x78`), above their midpoint $248$, so it becomes `0x78`.

**Token windows.** One shard holding the ids $10, 11, \dots, 19$ ($n = 10$), $T = 3$, $B = 2$, and phase draws that come out 1, then 0. Each draw is `below(min(3, 10 - 3)) = below(3)`.

1. Pass 1 starts at offset 1. Windows at offsets 1 and 4: ids $[11, 12, 13, 14]$ and $[14, 15, 16, 17]$. Batch 1 is inputs $[[11, 12, 13], [14, 15, 16]]$, targets $[[12, 13, 14], [15, 16, 17]]$.
2. The next window would start at 7 and need ids up to index 10, past the end. Pass 2 starts at phase 0: windows at 0 and 3, inputs $[[10, 11, 12], [13, 14, 15]]$.
3. The cursor is now shard 0, offset 6 (the next window), plus the generator state.

**The header.** The ids $[1, 2, 3]$ as version 1, vocabulary 256: `88 d8 34 01` ($20240520$), `01 00 00 00`, `03 00 00 00`, `00 01 00 00`, 1008 zero bytes, then `01 00 02 00 03 00`: 1030 bytes.

**A checkpoint.** A `Linear(2, 1)` trained two AdamW steps and saved at step 7: `step-000007/` holds `model.safetensors` (`weight`, `bias`), `optimizer.safetensors` (`weight.exp_avg`, `weight.exp_avg_sq`, `bias.exp_avg`, `bias.exp_avg_sq`), `trainer_state.json`, `MANIFEST.json`; `LATEST` holds `step-000007` and a newline.

These are `test_worked_example_bf16_bytes`, `test_f8_rounds_to_nearest_even`, `test_hand_example_token_windows`, `test_header_worked_example`, and `test_checkpoint_roundtrip`.

## 4. The interface

```python
# python/tinyllm/io/safetensors.py (taken over from L0.0; v0 calls keep working)
DTYPE_SIZES: dict[str, int]; E4M3_MAX = 448.0
def save_safetensors(path, tensors, meta, dtypes=None) -> None
def read_header(path) -> tuple[dict[str, tuple[str, tuple[int, ...]]], dict[str, str]]
def load_safetensors(path) -> tuple[dict[str, NDArray], dict[str, str]]

# python/tinyllm/io/checkpoint.py
class Checkpoint: path, step, model, opt, rng_state, extra
def step_name(step) -> str
def save_checkpoint(dir, model, opt, step, rng_state, extra, keep=None) -> str
def verify_step_dir(path) -> list[str]
def load_checkpoint(dir, step=None) -> Checkpoint

# python/tinyllm/io/tokens.py
def read_tokens_header(path) -> dict[str, int]; def open_tokens(path) -> NDArray
class TokenStream:
    def __init__(self, shards, seq_len, batch, rng, vocab_size=None)
    def next_batch(self) -> tuple[NDArray, NDArray]; def cursor(self) -> dict; def restore(self, cursor) -> None
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_worked_example_bf16_bytes` | unit | section 3's 112-byte file, byte for byte | you and the test agree on BF16 |
| `test_matches_library_bytes` | golden | five library files, every dtype, written from float32 values | any reader loads your files |
| `test_raw_bits_are_written_as_is` | unit | uint16 BF16 bits and uint8 F8 codes go to disk untouched | re-saving HF or quantized weights |
| `test_bf16_rounds_to_nearest_even` | boundary | the ties of section 3 and a below-half case | mixed precision (`L11.1`) |
| `test_f8_e4m3_codes` | unit | zero, $2^{-9}$, $2^{-6}$, 1, 448, NaN, $-0$, $-448$ | fp8 weights (`M09.4`) |
| `test_f8_rounds_to_nearest_even` | boundary | section 3's F8 ties, a subnormal tie, $-0$ | the same codes as torch |
| `test_f8_rejects_out_of_range` | boundary | 449 and infinity are errors; NaN is `0x7F` | a missing scale fails loudly |
| `test_dtype_order_then_name` | unit | F32, I32, U8 laid out in that order whatever the names | writer rule 1 |
| `test_save_rejects_bad_dtypes` | boundary | float64, int64, unknown names, stray `dtypes` keys, BF16 from float16 | no silent conversion |
| `test_load_decodes_every_dtype` | golden | values, numpy types, shapes, writable arrays, signed zeros | `L7.9` reads BF16 weights |
| `test_read_header_checks_without_reading` | unit | dtypes and shapes; trailing bytes and a huge length rejected | memory-mapped loading (`L10.1`) |
| `test_load_rejects_unknown_dtypes` | boundary | F64, BOOL, `bf16`, a wrong byte count | reader rule 4 |
| `test_raw_codes_roundtrip` | property | bits to values to bits gives the same file for every code | decode and encode agree |
| `test_v0_calls_still_work` | regression | the Pass 1 calls and the F32 file | your CLI and engine still work |
| `test_checkpoint_roundtrip` | unit | section 3's directory, `LATEST`, and every value loaded back bit for bit | resuming |
| `test_layout_matches_the_formats` | conformance | trainer state keys and hex, sorted manifest with true hashes, tensor names, unlisted files | `dur.09` and `dur.12` read these files |
| `test_config_and_defaults` | unit | `config.json` written and hashed; defaults for missing fields | a minimal caller is still schema-valid |
| `test_sgd_momentum_names` | unit | SGD state stored by parameter name and loaded back | any optimizer checkpoints |
| `test_resume_is_bitwise` | property | 3 steps, save, load into a fresh model, 3 steps equals 6 | `MS-L0` step 4 |
| `test_crash_at_every_write_keeps_a_valid_checkpoint` | fault | a crash at each fsync or rename leaves complete directories and a valid `LATEST`; at least 8 fsyncs | the atomic protocol |
| `test_load_skips_incomplete_and_corrupt` | fault | a flipped byte, a missing manifest, a stale `LATEST`, nothing valid | a crash or bit rot cannot load garbage |
| `test_newer_than_latest_is_ignored` | boundary | `LATEST` governs; without it the newest complete wins | the rename order's meaning |
| `test_keep_and_stale_tmp` | unit | `keep=2` leaves two steps; a stale `.tmp` is removed | disk use of long runs |
| `test_rejects_bad_state` | boundary | unknown extra keys, an even increment, a negative cursor, a short git sha, a negative step; nothing written | schema-valid state or nothing |
| `test_hand_example_token_windows` | unit | section 3's windows, cursor, and `below(3)` draws | you and the test agree on the stream |
| `test_header_worked_example` | unit | the 1030-byte file of `tokens-bin.md` | the llm.c layout |
| `test_version_2_reads_uint32` | unit | uint32 ids above 65535 | vocabularies above 64k |
| `test_open_tokens_maps_the_file` | unit | a read-only `np.memmap` | shards larger than memory |
| `test_rejects_bad_files` | boundary | wrong magic or version, size mismatch, out-of-vocabulary id, short file | bad data fails before training |
| `test_windows_tile_each_pass` | property | windows $T$ apart, passes end where the next window would not fit, shards in order | every token is a target once per pass |
| `test_cursor_restore_is_bitwise` | property | 7 batches, cursor, a new stream restored, 9 more equals one stream | `MS-L0` step 4 |
| `test_targets_are_inputs_shifted` | property | `targets[:, t] == inputs[:, t + 1]`, fresh int64 arrays | next-token prediction |
| `test_stream_rejects_bad_args` | boundary | no shards, a bare string, $T$ or $B < 1$, too short, vocabulary mismatch; a shard with one window | errors where the cause is visible |
| `test_restore_rejects_bad_cursor` | boundary | shard or offset out of range; offset at the end starts a new pass | a cursor from another run |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. laying a mixed-dtype file out by name | single-dtype files match the library, mixed ones do not | `test_dtype_order_then_name`, `test_matches_library_bytes` (mutant `s01`) |
| 2. truncating to BF16 or F8, or breaking ties upward | every weight biased; bytes differ from torch's | `test_bf16_rounds_to_nearest_even` (mutant `s02`), `test_f8_rounds_to_nearest_even` (mutant `s04`) |
| 3. saturating F8_E4M3 at 448 | out-of-range weights clip silently, the model degrades | `test_f8_rejects_out_of_range` (mutant `s07`) |
| 4. decoding F8_E4M3 subnormals as normals, or exponent 15 as NaN | small weights come out wrong; 256 to 448 become NaN | `test_f8_e4m3_codes` (mutants `s05`, `s06`) |
| 5. loose dtype checks: by kind only, or case-insensitive names | float64 slips in as F32; a file another reader rejects loads here | `test_save_rejects_bad_dtypes` (mutant `s09`), `test_load_rejects_unknown_dtypes` (mutant `s11`) |
| 6. writing straight into `step-<n>`, skipping fsync, or replacing `LATEST` before the rename | a crash leaves a half-written directory or a `LATEST` naming nothing | `test_crash_at_every_write_keeps_a_valid_checkpoint` (mutants `s14`, `s15`, `s16`) |
| 7. a loader that trusts `LATEST` | a corrupt newest checkpoint loads and the run trains on garbage | `test_load_skips_incomplete_and_corrupt` (mutant `s17`) |
| 8. trusting the `.bin` header | version 2 read as uint16, trailing bytes, ids beyond the vocabulary | `test_version_2_reads_uint32` (mutant `s31`), `test_rejects_bad_files` (mutants `s33`, `s34`) |
| 9. windows $T + 1$ apart, or targets equal to inputs | boundary tokens are never targets; the model learns the identity | `test_hand_example_token_windows` (mutant `s25`), `test_targets_are_inputs_shifted` (mutant `s26`) |
| 10. a cursor without the generator state | a resumed run draws a different phase at its next pass and diverges | `test_cursor_restore_is_bitwise` (mutant `s29`) |
| 11. the phase from `below(T)` on a short shard | a window past the end of the shard | `test_stream_rejects_bad_args` (mutant `s24`) |
| 12. a cursor naming the window just read | a resumed run repeats one window | `test_hand_example_token_windows` (mutant `s30`) |

## 6. Where it's used next
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L2.2` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |
| Forward | `L4.1` | Registered call site uses this module. |
| Forward | `L5.5` | Registered call site uses this module. |
| Forward | `L6.1` | Registered call site uses this module. |
| Forward | `L6.2` | Registered call site uses this module. |
| Forward | `L6.3` | Registered call site uses this module. |
| Forward | `L6.5` | Registered call site uses this module. |
| Forward | `L6.7` | Registered call site uses this module. |
| Forward | `L7.9` | Registered call site uses this module. |
| Forward | `dur.09` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.1` | `f32_to_bf16_bits` and `bf16_bits_to_f32` for BF16 |
| Back | `L0.1` | the tensors of the model the checkpoint tests train |
| Back | `L0.2` | `F.mean` in the checkpoint tests' loss |
| Back | `L0.4` | a checkpoint is `state_dict()`; resume is `load_state_dict()` |
| Back | `M10.2` | SGD's momentum buffers, saved by parameter name |
| Back | `M10.3` | AdamW's moments and step count, saved and restored |
| Back | `M06.3` | `PCG32.below` draws each pass's phase; `state()` and `set_state()` move with the cursor |
| Back | `L0.5` | `L0.0`'s suite runs as your regression for `safetensors.py` and also covers `bigram.py`, which `L0.5` owns |
| Forward | `L10.0` | your engine reads the `model.safetensors` this writer produces |
| Forward | `L2.1` | the n-gram model writes and reads its count tables through `save_safetensors` and `load_safetensors` |

Later passes build on it without changing it: `L2.2` reads `TokenStream` windows, `L7.9` loads BF16 SmolLM2 weights, `L10.1` memory-maps the same layout from Rust, and the capstone trainer and `dur.09` resume from these checkpoints.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `save_safetensors` | Hugging Face `safetensors` | zero-copy memory-mapped loading, lazy slicing of one tensor, sharded files with an index | `safetensors/src/tensor.rs` |
| F8_E4M3 table | `ml_dtypes`, `torch.float8_e4m3fn` | E5M2, scaled matmuls on Hopper, MX block-scaled formats | `ml_dtypes/_src/float8.h` |
| atomic checkpoints | PyTorch Distributed Checkpoint, Orbax | sharded saves from many ranks, async writes off the training thread, a commit marker per save | `torch/distributed/checkpoint/`, `orbax/checkpoint/` |
| `TokenStream` | llm.c `DataLoader`, nanotron, MosaicML `streaming` | sharding across ranks, shuffled shard order, deterministic resumption across world sizes | llm.c `llmc/dataloader.h`, `streaming/base/dataset.py` |
