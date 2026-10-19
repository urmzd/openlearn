# Safetensors for every dtype, atomic checkpoints, and the token stream

## Overview

| | |
|---|---|
| **Module** | `L0.6` · build · Python · Pass 2 · 3 to 4 h |
| **Solution** | [`safetensors.py`](../../../../course/solutions/python/tinyllm/io/safetensors.py), taken over from `L0.0` (earlier version in [`history/L0.0/`](../../../../course/solutions/history/L0.0/)): F16, BF16, F8_E4M3, I8, U8, I32 plus `read_header`, `f32_to_e4m3`, `e4m3_to_f32` · [`checkpoint.py`](../../../../course/solutions/python/tinyllm/io/checkpoint.py): `save_checkpoint`, `load_checkpoint`, `verify_step_dir` · [`tokens.py`](../../../../course/solutions/python/tinyllm/io/tokens.py): `read_tokens_header`, `open_tokens`, `TokenStream` · the `.bin` form of `train bigram` in [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py) |
| **Contract** | [`course/contracts/py/tinyllm/io/safetensors.pyi`](../../../../course/contracts/py/tinyllm/io/safetensors.pyi) · [`course/contracts/py/tinyllm/io/checkpoint.pyi`](../../../../course/contracts/py/tinyllm/io/checkpoint.pyi) · [`course/contracts/py/tinyllm/io/tokens.pyi`](../../../../course/contracts/py/tinyllm/io/tokens.pyi) · formats: [`safetensors.md`](../../../../course/contracts/formats/safetensors.md), [`checkpoint.md`](../../../../course/contracts/formats/checkpoint.md), [`tokens-bin.md`](../../../../course/contracts/formats/tokens-bin.md) |
| **Needs** | `M09.1` the BF16 bit converters · `L0.1`, `L0.2`, `L0.4` the model a checkpoint holds · `M10.2` SGD and `M10.3` AdamW (optimizer state to save) · `M06.3` the PCG32 whose state the token cursor carries · `L0.5` its `bigram.py` · reading: `L0.0` |
| **Used by** | `L10.0` the engine reads the file this writer produces (inherited from `L0.0` with `safetensors.py`) · `L2.1` the n-gram model saves and loads its tables with `save_safetensors` and `load_safetensors` · later: `L2.2` reads token streams, `L7.9` loads BF16 HF weights, `L10.1` memory-maps the same layout, the capstone trainer and `dur.09` resume from these checkpoints; the Part 0 end-to-end run kills a training job after step 60, resumes it from the step-50 checkpoint, and ends bitwise equal to an uninterrupted run · later: `L3.6`, `L4.1`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5`, `L6.7`  · `L14.3` Whisper decoding |
| **Optional depth** | the safetensors README and `safetensors/src/tensor.rs`; Micikevicius et al., "FP8 Formats for Deep Learning" (2022); Pillai et al., "All File Systems Are Not Created Equal" (OSDI 2014) on crash consistency |

## Key Takeaways

- One safetensors writer for seven dtypes, byte-identical to the reference library: tensors laid out by dtype first, then by name.
- BF16 and F8_E4M3 have no numpy type, so they travel as float32 values rounded to nearest, ties to even, or as raw bits written untouched; F8_E4M3 has no infinity, so a value beyond 448 is an error, not a saturation.
- A checkpoint is written into a `.tmp` directory, every file fsynced, the manifest last, then renamed into place, then LATEST replaced by a rename: a crash at any instant leaves the old checkpoint or the new one.
- Loading checks every file against the manifest and falls back to the newest older checkpoint that verifies.
- The token stream's cursor (shard, offset, generator state) is its whole position: N batches, save, restore, M batches equals N + M batches.

## How to read this chapter

Sections 1 to 3 build the three formats from first principles (dtype codes, the atomic checkpoint protocol, the token stream) and work each by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through `safetensors.py`, `checkpoint.py`, and `tokens.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The solution's entry point uses all three: `train bigram --method autograd --data shard.bin --max-steps N --batch B --seq-len T --ckpt-every K [--resume]` in [`cli_train.py`](../../../../course/solutions/entry/python/tinyllm/cli_train.py) reads windows through `TokenStream`, writes `save_checkpoint(<out>/ckpt, ...)` every K steps with `extra = {"tokens_seen", "data_cursor", "lr", "config_sha256", "git_sha", "config"}` and the cursor's generator state as `rng_state`, and with `--resume` restores model, optimizer, step, and cursor from `load_checkpoint(<out>/ckpt)`. After every step it evaluates the failpoint `train/after-step` (the fault-injection kata of `craft.03` uses it to kill the run), and it ends by writing `final.json`: step, tokens seen, the loss and its exact hex, the cursor, and a hash of the parameters, so an interrupted-and-resumed run can be compared with an uninterrupted one byte for byte.

---

## 1. Why now

Three things in the system cannot happen yet. The safetensors writer of `L0.0` knows only F32, but SmolLM2's weights (`L7.9`) are BF16 and quantized weights (`M09.4`, `L8.5`) are F8_E4M3 and U8. Training runs keep their state in memory: kill one after step 60 and today it would restart from zero, because nothing on disk says where it was, what AdamW's moments were, or which window comes next. And the bigram trains on a byte string read whole into memory; the capstone's TinyStories tokens are hundreds of megabytes in llm.c's `.bin` format. This module writes every dtype, saves a training run so that a crash at any instant leaves a usable checkpoint, and reads token shards through a memory map with a cursor that resumes exactly.

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

These are the first cases a correct solution must reproduce.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the BF16 worked example | unit | section 3's 112-byte file, byte for byte | everyone agrees on BF16 |
| bytes match the library | golden | five library files, every dtype, written from float32 values | any reader loads these files |
| raw bits are written as is | unit | uint16 BF16 bits and uint8 F8 codes go to disk untouched | re-saving HF or quantized weights |
| BF16 rounds to nearest even | boundary | the ties of section 3 and a below-half case | mixed precision (`L11.1`) |
| F8_E4M3 codes | unit | zero, $2^{-9}$, $2^{-6}$, 1, 448, NaN, $-0$, $-448$ | fp8 weights (`M09.4`) |
| F8 rounds to nearest even | boundary | section 3's F8 ties, a subnormal tie, $-0$ | the same codes as torch |
| F8 rejects out of range | boundary | 449 and infinity are errors; NaN is `0x7F` | a missing scale fails loudly |
| dtype order, then name | unit | F32, I32, U8 laid out in that order whatever the names | writer rule 1 |
| bad dtypes on save | boundary | float64, int64, unknown names, stray `dtypes` keys, BF16 from float16 | no silent conversion |
| load decodes every dtype | golden | values, numpy types, shapes, writable arrays, signed zeros | `L7.9` reads BF16 weights |
| `read_header` checks without reading | unit | dtypes and shapes; trailing bytes and a huge length rejected | memory-mapped loading (`L10.1`) |
| unknown dtypes on load | boundary | F64, BOOL, `bf16`, a wrong byte count | reader rule 4 |
| raw codes round trip | property | bits to values to bits gives the same file for every code | decode and encode agree |
| v0 calls still work | regression | the Pass 1 calls and the F32 file | the CLI and engine still work |
| checkpoint round trip | unit | section 3's directory, `LATEST`, and every value loaded back bit for bit | resuming |
| layout matches the formats | conformance | trainer state keys and hex, sorted manifest with true hashes, tensor names, unlisted files | `dur.09` and `dur.12` read these files |
| config and defaults | unit | `config.json` written and hashed; defaults for missing fields | a minimal caller is still schema-valid |
| SGD momentum names | unit | SGD state stored by parameter name and loaded back | any optimizer checkpoints |
| resume is bitwise | property | 3 steps, save, load into a fresh model, 3 steps equals 6 | the kill-and-resume run of Part 0 |
| a crash at every write | fault | a crash at each fsync or rename leaves complete directories and a valid `LATEST`; at least 8 fsyncs | the atomic protocol |
| incomplete and corrupt checkpoints | fault | a flipped byte, a missing manifest, a stale `LATEST`, nothing valid | a crash or bit rot cannot load garbage |
| newer than `LATEST` is ignored | boundary | `LATEST` governs; without it the newest complete wins | the rename order's meaning |
| `keep` and stale `.tmp` | unit | `keep=2` leaves two steps; a stale `.tmp` is removed | disk use of long runs |
| bad state | boundary | unknown extra keys, an even increment, a negative cursor, a short git sha, a negative step; nothing written | schema-valid state or nothing |
| the token-window worked example | unit | section 3's windows, cursor, and `below(3)` draws | everyone agrees on the stream |
| the `.bin` header worked example | unit | the 1030-byte file of `tokens-bin.md` | the llm.c layout |
| version 2 reads uint32 | unit | uint32 ids above 65535 | vocabularies above 64k |
| `open_tokens` maps the file | unit | a read-only `np.memmap` | shards larger than memory |
| bad `.bin` files | boundary | wrong magic or version, size mismatch, out-of-vocabulary id, short file | bad data fails before training |
| windows tile each pass | property | windows $T$ apart, passes end where the next window would not fit, shards in order | every token is a target once per pass |
| cursor restore is bitwise | property | 7 batches, cursor, a new stream restored, 9 more equals one stream | the kill-and-resume run of Part 0 |
| targets are inputs shifted | property | `targets[:, t] == inputs[:, t + 1]`, fresh int64 arrays | next-token prediction |
| bad stream arguments | boundary | no shards, a bare string, $T$ or $B < 1$, too short, vocabulary mismatch; a shard with one window | errors where the cause is visible |
| bad cursor | boundary | shard or offset out of range; offset at the end starts a new pass | a cursor from another run |

## 5. The solution, step by step

### From first principles

1. **Files must be byte-identical to the library's**, or golden hashes, manifests, and other readers disagree. So the writer copies the library's canonical order exactly: sort by dtype rank, then by name as UTF-8 bytes.
2. **Lossy dtypes need one rounding rule.** Round to nearest, ties to even, is the IEEE default and what torch uses; anything else biases every weight. BF16 delegates to `M09.1`. F8_E4M3 has only 256 codes, so the simplest correct encoder is a table of their values and a nearest-neighbour search: there is no bit trick to get wrong.
3. **A value with no code is an error.** F8_E4M3 has no infinity; saturating at 448 would silently clip weights. Raising forces the caller to add the scale `M09.4` introduces.
4. **Atomicity comes from rename, durability from fsync.** A rename of a directory is atomic on POSIX file systems; data is durable only after `fsync`. So: write everything somewhere invisible, fsync, publish with one rename, fsync the parent, and only then publish the pointer (`LATEST`) by the same write-then-rename trick. Deleting anything comes strictly after.
5. **Completeness must be checkable after the fact.** The manifest is written last and lists every other file's size and hash. A directory with a valid manifest that matches the disk is complete; anything else is skipped, and the loader falls back to an older step.
6. **The stream's position must be a small, serializable value.** Shards are read in order and windows are a fixed stride $T$ apart, so position is (shard, offset of the next window). The only randomness, the phase of each pass, comes from one generator, so its state completes the cursor.

### The code, annotated

**F8_E4M3 encoding.** The 256-entry table is built once from the format's definition; the positive half, codes `0x00` to `0x7E`, is sorted by value, so encoding is a binary search.

```python
def _e4m3_table() -> NDArray:
    out = np.empty(256, dtype=np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 0xF, c & 0x7
        if e == 0xF and m == 0x7:
            v = math.nan                      # the only NaN pattern; e = 15 is otherwise normal
        elif e == 0:
            v = (m / 8.0) * 2.0**-6  # subnormal: no implicit 1, exponent fixed at 1 - bias
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        out[c] = -v if s else v
    return out

def f32_to_e4m3(x: NDArray) -> NDArray:
    f = np.asarray(x, dtype=np.float32)
    a = np.abs(f).astype(np.float64)
    finite = ~np.isnan(f)
    if np.any(a[finite] > E4M3_MAX):          # includes +-inf: no code to round to
        raise ValueError(f"F8_E4M3 holds |x| <= {E4M3_MAX:g}; got {float(a[finite].max()):g} (scale it first)")
    pos = _e4m3()[:0x7F].astype(np.float64)  # codes 0x00..0x7E: 0, ..., 448, increasing
    hi = np.clip(np.searchsorted(pos, a, side="left"), 0, len(pos) - 1)
    lo = np.clip(hi - 1, 0, None)
    d_lo, d_hi = a - pos[lo], pos[hi] - a
    # Nearest code; on an exact tie the even one (its last mantissa bit is 0,
    # and consecutive codes alternate that bit, across exponents too).
    pick = np.where(d_lo < d_hi, lo, np.where(d_hi < d_lo, hi, np.where(lo % 2 == 0, lo, hi)))
    code = pick.astype(np.uint8) | np.where(np.signbit(f), 0x80, 0).astype(np.uint8)  # signbit keeps -0
    return np.where(finite, code, np.uint8(0x7F)).astype(np.uint8)
```

On section 3's values: $1.0625$ lies between `0x38` ($1.0$) and `0x39` ($1.125$) with equal distances, and `0x38` is even, so it wins. $250$ has `d_lo` $= 10$ to $240$ and `d_hi` $= 6$ to $256$, so `hi`, code `0x78`.

**Encoding one tensor.** Every dtype has an explicit list of accepted numpy types; nothing is converted by kind.

```python
def _encode(name: str, arr: NDArray, dtype: Optional[str]) -> tuple[str, bytes]:
    k = arr.dtype.newbyteorder("=")  # a big-endian float32 is still a float32
    if dtype is None:
        dtype = _INFER.get(k)        # float32, float16, int8, uint8, int32 only: float64 is refused
        if dtype is None:
            raise ValueError(...)
    if dtype not in DTYPE_SIZES:     # exact, case-sensitive names: "bf16" is not "BF16"
        raise ValueError(...)
    if dtype == "BF16":
        if k == np.uint16:
            bits = arr  # raw bits, written as is
        elif k == np.float32:
            bits = f32_to_bf16_bits(arr)     # M09.1: round to nearest, ties to even
        else:
            raise ValueError(...)            # float16 -> BF16 would round twice
        out = np.asarray(bits, dtype=np.uint16)
    elif dtype == "F8_E4M3":
        out = arr if k == np.uint8 else f32_to_e4m3(arr)   # (after the same type check)
    # ... F16 from float16 or float32; F32, I8, U8, I32 only from their exact type
    # Little-endian, row-major, whatever the array's byte order or memory layout.
    return dtype, np.asarray(out).astype(_STORE[dtype], copy=False).tobytes(order="C")
```

**The canonical layout.** The sort key is the whole of writer rule 1.

```python
_ORDER = ["F32", "I32", "BF16", "F16", "F8_E4M3", "I8", "U8"]   # larger alignment first

def save_safetensors(path, tensors, meta, dtypes=None) -> None:
    # ... validate meta (str -> str) and that dtypes names only tensors being written
    encoded = []
    for name in tensors:
        arr = np.asarray(tensors[name])
        dt, data = _encode(name, arr, dtypes.get(name))
        encoded.append((_ORDER.index(dt), name.encode("utf-8"), name, dt, arr.shape, data))
    # Writer rule 1: by dtype (larger alignment first), then by name as UTF-8 bytes.
    encoded.sort(key=lambda e: (e[0], e[1]))
    # ... header: __metadata__ first (keys sorted), then tensors with running data_offsets
    text = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (-len(text) % 8)          # pad to a multiple of 8: section 3's 93 -> 96
    with open(path, "wb") as f:
        f.write(len(text).to_bytes(8, "little"))
        f.write(text)
        for e in encoded:
            f.write(e[5])
```

Sorting by UTF-8 bytes is the library's rule (Rust compares `String`s bytewise); since UTF-8 preserves code point order, it also agrees with sorting the Python strings, but the key states the rule rather than relying on that fact.

**Reading a header without the data.** `_parse` checks the five reader rules from the header bytes and the file size; `read_header` feeds it only the header.

```python
def read_header(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) == 8:
            n = int.from_bytes(head, "little")
            # Read the header only when rule 1 holds: a corrupt length must
            # not make the reader allocate gigabytes.
            if n <= MAX_HEADER and 8 + n <= size:
                head += f.read(n)
    _, entries, meta = _parse(path, head, size)
    return {name: (dt, tuple(shape)) for _, _, name, dt, shape in entries}, meta
```

Rule 5 inside `_parse` sorts entries by `begin` and requires each to start where the last ended and the last to end exactly at the file size, which is how trailing bytes, gaps, and overlaps are all caught with offsets alone. `load_safetensors` decodes BF16 and F8_E4M3 back to float32 and returns writable native-order copies.

**The atomic save.** The numbered comments are the protocol of section 2.

```python
def save_checkpoint(dir, model, opt, step, rng_state, extra, keep=None) -> str:
    root = Path(dir)
    root.mkdir(parents=True, exist_ok=True)
    name = step_name(step)                              # "step-000007"
    final, tmp = root / name, root / (name + ".tmp")
    names = [n for n, _ in model.named_parameters()]
    skel, opt_arrays = _flatten_opt(opt.state_dict(), names)   # arrays keyed "<param>.<state key>"
    # ... config bytes; _trainer_state validates extra *before* anything is written
    state = _trainer_state(step, rng_state, extra, opt, config_bytes)
    if tmp.exists():
        shutil.rmtree(tmp)  # a crashed earlier attempt at this step
    tmp.mkdir()
    # 1. Every file into the .tmp directory, each fsynced.
    save_safetensors(str(tmp / "model.safetensors"), model.state_dict(), {"format": "tinyllm"})
    _fsync_path(tmp / "model.safetensors")
    save_safetensors(str(tmp / "optimizer.safetensors"), opt_arrays, {..., "state": json.dumps(skel, ...)})
    _fsync_path(tmp / "optimizer.safetensors")
    # ... config.json if any; trainer_state.json, both through _write (write, flush, fsync)
    # 2. The manifest last: its presence (and agreement) means "complete".
    files = sorted(p.name for p in tmp.iterdir())
    manifest = {
        "kind": "checkpoint",
        "files": [{"name": f, "sha256": _sha256(tmp / f), "bytes": (tmp / f).stat().st_size} for f in files],
    }
    _write(tmp / "MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode())
    _fsync_path(tmp)                                    # the directory's entries, too
    # 3. Into place by rename: a directory appears whole or not at all.
    if final.exists():
        shutil.rmtree(final)  # re-saving a step replaces it
    os.rename(tmp, final)
    _fsync_path(root)                                   # make the rename itself durable
    # 4. LATEST by write-then-rename, so a reader never sees half a name.
    _write(root / "LATEST.tmp", (name + "\n").encode())
    os.rename(root / "LATEST.tmp", root / "LATEST")
    _fsync_path(root)
    # 5. Housekeeping only after the new checkpoint is durable.
    for p in root.iterdir():
        if p.name.endswith(".tmp") and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    if keep is not None:
        # ... delete all but the newest `keep` step directories
    return str(final)
```

Count the fsyncs for a checkpoint without `config.json`: model, optimizer, trainer state, manifest, the `.tmp` directory, the root after the directory rename, `LATEST.tmp`, the root after the `LATEST` rename. That is the "at least 8" a crash test can interrupt one at a time. The generator state goes into `trainer_state.json` as 16 hex digits each (`_rng_json`), because a JSON reader that parses numbers as doubles would round a 64-bit integer.

**Verification and fallback.** `verify_step_dir` returns a list of problems (empty means valid), so a caller can report why a directory was skipped.

```python
def verify_step_dir(path: str) -> list[str]:
    d = Path(path)
    man = d / "MANIFEST.json"
    if not man.is_file():
        return [f"{d.name}: no MANIFEST.json (incomplete)"]   # crashed before step 2 finished
    # ... parse, require kind == "checkpoint" and the three REQUIRED_FILES listed
    on_disk = {p.name for p in d.iterdir() if p.name != "MANIFEST.json"}
    for f in sorted(on_disk - set(listed)):
        errs.append(f"{d.name}: {f} is not in MANIFEST.json")  # unlisted files are suspect too
    for f, r in sorted(listed.items()):
        p = d / f
        if not p.is_file():
            errs.append(f"{d.name}: {f} is missing")
        elif p.stat().st_size != r.get("bytes") or _sha256(p) != r.get("sha256"):
            errs.append(f"{d.name}: {f} does not match its sha256 or size")   # bit rot, torn write
    return errs

def load_checkpoint(dir: str, step: Optional[int] = None) -> Checkpoint:
    root = Path(dir)
    # ... an explicit step must verify or ValueError
    dirs = sorted(_step_dirs(root), key=lambda x: -x[0])      # newest first
    latest = root / "LATEST"
    if latest.is_file():
        mm = _STEP_DIR.match(latest.read_text().strip())
        if mm:
            # Start from what LATEST names; newer directories were never
            # declared complete. Older ones are the fallback.
            dirs = [(n, p) for n, p in dirs if n <= int(mm.group(1))]
    for _, p in dirs:
        if not verify_step_dir(str(p)):
            return _read(p)
    raise FileNotFoundError(f"{root}: no checkpoint directory verifies")
```

**The token stream.** The header is validated against the file size before anything is mapped; ids are checked against the vocabulary once, at open time.

```python
def open_tokens(path: str) -> NDArray:
    h = read_tokens_header(path)           # magic, version 1 or 2, size == 1024 + n * width
    # ...
    ids = np.memmap(path, dtype=_WIDTH[h["version"]], mode="r", offset=HEADER_BYTES, shape=(h["n_tokens"],))
    if h["vocab_size"]:
        # The header promises every id is below vocab_size: check it once,
        # here, not as an out-of-range embedding lookup mid-training.
        top = int(ids.max())
        if top >= h["vocab_size"]:
            raise ValueError(...)
    return ids

class TokenStream:
    # __init__: validates arguments, opens every shard, requires T + 1 tokens per shard,
    # then self.shard = 0 and self.offset = self._phase(0)

    def _phase(self, shard: int) -> int:
        starts = self.ids[shard].size - self.seq_len  # valid window starts: 0 .. n - T - 1
        return int(self.rng.below(min(self.seq_len, starts)))   # short shard: stay in range

    def next_batch(self) -> tuple[NDArray, NDArray]:
        T = self.seq_len
        rows = np.empty((self.batch, T + 1), dtype=np.int64)
        for b in range(self.batch):
            if self.offset + T + 1 > self.ids[self.shard].size:
                # This pass is done: the next shard (wrapping), at a new phase.
                self.shard = (self.shard + 1) % len(self.ids)
                self.offset = self._phase(self.shard)
            rows[b] = self.ids[self.shard][self.offset : self.offset + T + 1]
            self.offset += T  # windows share their boundary token: target of one, input of the next
        return rows[:, :-1].copy(), rows[:, 1:].copy()   # inputs, targets shifted by one

    def cursor(self) -> dict:
        # offset already points at the *next* window, so a restore never repeats one
        return {"shard": self.shard, "offset": self.offset, "rng": tuple(self.rng.state())}

    def restore(self, cursor: dict) -> None:
        # ... shard in range, 0 <= offset <= shard size (at the end: the next call starts a pass)
        self.rng.set_state(tuple(cursor["rng"]))
        self.shard, self.offset = shard, offset
```

Trace section 3: the constructor draws phase 1. Batch 1 reads offsets 1 and 4 and leaves `offset = 7`. Batch 2 sees $7 + 4 > 10$, moves to shard 0 again with phase 0, reads offsets 0 and 3, and leaves `offset = 6`: the cursor of section 3.

Full solution: [`safetensors.py`](../../../../course/solutions/python/tinyllm/io/safetensors.py) · [`checkpoint.py`](../../../../course/solutions/python/tinyllm/io/checkpoint.py) (also `_flatten_opt`, `_trainer_state`, `_read`) · [`tokens.py`](../../../../course/solutions/python/tinyllm/io/tokens.py).

### Why this solution

- **Cost.** Writing a safetensors file is one pass over the bytes plus a sort of $k$ tensor entries. F8 encoding is $O(n \log 256)$ by binary search. A checkpoint costs two hashes of every file (once for the manifest, once on load) and eight fsyncs; for training runs that save every few hundred steps, that is negligible next to the steps. The token stream maps shards and copies only $B(T + 1)$ ids per batch.
- **Alternatives.** Bit manipulation for F8 rounding is faster but has many edge cases (subnormals, the missing infinity, ties across exponents) where a table cannot be wrong. Writing checkpoints in place with a "done" flag file still needs fsync ordering and leaves partial directories visible. Pickling the optimizer state is shorter and ties checkpoints to Python and to class layout; safetensors plus JSON is readable from Rust and Go. Random windows at any offset (not a stride) cover tokens unevenly and make the cursor depend on every draw, not just one per pass.
- **Accepted tradeoffs.** Loading reads whole files into memory (the memory-mapped reader is `L10.1`'s job). The checkpoint saves model weights as F32 regardless of training precision. Housekeeping failures are ignored (`ignore_errors=True`): a leftover old step costs disk, never correctness.

### Explain it back

1. Why does `f32_to_e4m3` raise for 449 instead of returning code `0x7E` (448)?
   <details><summary>Answer</summary>
   F8_E4M3 has no infinity and `0x7E` means exactly 448. Returning it would clip the weight silently, and a model with clipped outliers degrades without any error. The value needs a scale (`M09.4`) to fit; raising is how the missing scale is found.
   </details>
2. Why does `LATEST` get its own write-then-rename, after the directory rename?
   <details><summary>Answer</summary>
   Writing `LATEST` in place could leave a torn or empty file after a crash. Renaming `LATEST.tmp` over it replaces the name atomically. Doing it after the directory rename (and its fsync) guarantees `LATEST` never names a directory that does not exist yet.
   </details>
3. Why does `load_checkpoint` ignore step directories newer than the one `LATEST` names?
   <details><summary>Answer</summary>
   A directory newer than `LATEST` either never finished being declared done or was written by a run that crashed between step 3 and step 4 of the protocol. `LATEST` is the commit point; only what it covers counts. Without `LATEST`, the newest directory that verifies is used.
   </details>
4. Why is the phase drawn with `below(min(T, n - T))` and not `below(T)`?
   <details><summary>Answer</summary>
   A window needs $T + 1$ ids, so the last valid start is $n - T - 1$. On a shard shorter than $2T$, a phase up to $T - 1$ could be past that start and the first window would run off the end. Limiting the bound to $n - T$ keeps at least one window in range.
   </details>
5. What exactly goes wrong if the cursor omits the generator state?
   <details><summary>Answer</summary>
   The resumed run continues correctly until the current pass ends. At the next pass, it draws a phase from a generator in a different state than the uninterrupted run would have, so window boundaries shift and every later batch differs. The divergence is silent; only a bitwise comparison of the two runs shows it.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. laying a mixed-dtype file out by name | single-dtype files match the library, mixed ones do not | sorts by `(_ORDER.index(dtype), name bytes)` |
| 2. truncating to BF16 or F8, or breaking ties upward | every weight biased; bytes differ from torch's | `f32_to_bf16_bits` rounds to nearest even; `f32_to_e4m3` picks the even code on an exact tie |
| 3. saturating F8_E4M3 at 448 | out-of-range weights clip silently, the model degrades | raises `ValueError` for any finite $\lvert x \rvert > 448$ and for infinity |
| 4. decoding F8_E4M3 subnormals as normals, or exponent 15 as NaN | small weights come out wrong; 256 to 448 become NaN | `_e4m3_table` has a separate $e = 0$ branch and treats only $e = 15, m = 7$ as NaN |
| 5. loose dtype checks: by kind only, or case-insensitive names | float64 slips in as F32; a file another reader rejects loads here | `_INFER` and each dtype branch accept exact numpy types; `DTYPE_SIZES` membership is exact |
| 6. writing straight into `step-<n>`, skipping fsync, or replacing `LATEST` before the rename | a crash leaves a half-written directory or a `LATEST` naming nothing | writes into `.tmp`, fsyncs every file and directory, renames, then replaces `LATEST` by rename |
| 7. a loader that trusts `LATEST` | a corrupt newest checkpoint loads and the run trains on garbage | `verify_step_dir` on every candidate, newest first, falling back to older steps |
| 8. trusting the `.bin` header | version 2 read as uint16, trailing bytes, ids beyond the vocabulary | width from the version, exact file size check, one `max()` against `vocab_size` |
| 9. windows $T + 1$ apart, or targets equal to inputs | boundary tokens are never targets; the model learns the identity | `offset += T`; returns `rows[:, :-1]` and `rows[:, 1:]` |
| 10. a cursor without the generator state | a resumed run draws a different phase at its next pass and diverges | `cursor()` includes `rng.state()`, `restore` sets it |
| 11. the phase from `below(T)` on a short shard | a window past the end of the shard | `_phase` uses `below(min(T, n - T))` |
| 12. a cursor naming the window just read | a resumed run repeats one window | `offset` is advanced past each window as it is read |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.1` | `f32_to_bf16_bits` and `bf16_bits_to_f32` for BF16 |
| Back | `L0.1` | the tensors of the model a checkpoint holds |
| Back | `L0.2` | the ops of the training step a checkpoint resumes |
| Back | `L0.4` | a checkpoint is `state_dict()`; resume is `load_state_dict()` |
| Back | `M10.2` | SGD's momentum buffers, saved by parameter name |
| Back | `M10.3` | AdamW's moments and step count, saved and restored |
| Back | `M06.3` | `PCG32.below` draws each pass's phase; `state()` and `set_state()` move with the cursor |
| Back | `L0.5` | `L0.0`'s behaviour of `safetensors.py` and `bigram.py` (which `L0.5` owns) is kept as the regression baseline |
| Forward | `L10.0` | the engine reads the `model.safetensors` this writer produces |
| Forward | `L2.1` | the n-gram model writes and reads its count tables through `save_safetensors` and `load_safetensors` |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L2.2` | `save_nplm`/`load_nplm` write and read the weights with `save_safetensors`/`load_safetensors` |
| Forward | `L3.6` | the RNN LM checkpoint is a safetensors file |
| Forward | `L4.1` | the seq2seq checkpoint is a safetensors file |
| Forward | `L5.5` | the transformer checkpoint is a safetensors file |
| Forward | `L6.1` | GPT checkpoints are safetensors files |
| Forward | `L6.2` | BERT checkpoints are safetensors files |
| Forward | `L6.3` | ELECTRA checkpoints are safetensors files |
| Forward | `L6.5` | the task heads save and load with safetensors |
| Forward | `L6.7` | the zoo reads evaluation tokens with `open_tokens` and every checkpoint with `load_safetensors` |
| Forward | `L7.9` | HF Llama weights are read with `load_safetensors` |
| Forward | `dur.09` | the training activity calls `verify_step_dir` before it reports a checkpoint as complete |

Later passes build on it without changing it: `L2.2` reads `TokenStream` windows, `L7.9` loads BF16 SmolLM2 weights, `L10.1` memory-maps the same layout from Rust, and the capstone trainer and `dur.09` resume from these checkpoints.

## Going further

| Piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `save_safetensors` | Hugging Face `safetensors` | zero-copy memory-mapped loading, lazy slicing of one tensor, sharded files with an index | `safetensors/src/tensor.rs` |
| F8_E4M3 table | `ml_dtypes`, `torch.float8_e4m3fn` | E5M2, scaled matmuls on Hopper, MX block-scaled formats | `ml_dtypes/_src/float8.h` |
| atomic checkpoints | PyTorch Distributed Checkpoint, Orbax | sharded saves from many ranks, async writes off the training thread, a commit marker per save | `torch/distributed/checkpoint/`, `orbax/checkpoint/` |
| `TokenStream` | llm.c `DataLoader`, nanotron, MosaicML `streaming` | sharding across ranks, shuffled shard order, deterministic resumption across world sizes | llm.c `llmc/dataloader.h`, `streaming/base/dataset.py` |
