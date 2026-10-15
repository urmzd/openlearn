<!-- ol:module L13.4 -->
# Dynamic resolution and tiling

## Overview

| | |
|---|---|
| **Module** | `L13.4` · build · Python · Pass 12 · 4 to 6 h |
| **You build** | `python/tinyllm/vision/tiling.py`: `TilingConfig`, `select_best_resolution`, `fit_size`, `anyres_tiles`, `unpad_shape` (LLaVA-NeXT AnyRes); `idefics3_resize_size`, `idefics3_split`, `idefics3_prompt` (Idefics3 / SmolVLM); `pixel_shuffle`, `pixel_unshuffle`; `image_token_count` |
| **Contract** | [`course/contracts/py/tinyllm/vision/tiling.pyi`](../../../../course/contracts/py/tinyllm/vision/tiling.pyi) |
| **Tests** | `course/tests/L13.4/test_tiling.py` (what they check: section 4), golden values in `course/fixtures/L13.4/tiling_hf.npz` from `course/oracle/L13.4/tiling_hf.py` (transformers 5.19.0 `Idefics3ImageProcessorPil`, `Idefics3Processor`, `LlavaNextImageProcessorPil`, `LlavaNextProcessor`, `select_best_resolution`, `Idefics3Connector`, on Pillow 12.3.0) |
| **Needs** | `M12.3` `resize` (or `--ref-deps`). Reading: `M12.2` HWC pixels, `M05.1` counting tokens and bytes |
| **Used by** | `L13.6` (runs the split, reports token counts) · `L13.7` (expands image placeholders) · later `L15.1` (Rust) and `gw.10` (Go `ImageTokens`) re-implement `image_token_count` against the same fixture table |
| **Milestone** | `MS-L13` step 3 (preprocessing to spec: token counts exact) |
| **Optional depth** | Liu et al., "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge" (2024); Laurençon et al., "Building and better understanding vision-language models" (2024), for Idefics3; Marafioti et al., "SmolVLM: Redefining small and efficient multimodal models" (2025); Shi et al., "Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network" (2016), for pixel shuffle |

## Key Takeaways

- Idefics3 / SmolVLM splitting is integer arithmetic on the size: longest edge to 2048 (other edge rounded up to even), each edge grown to a multiple of 512, a `rows x cols` grid of tiles plus one global image, so a $1000 \times 1500$ photo costs $(3 \cdot 4 + 1) \cdot 64 = 832$ tokens (`test_hand_example`, `test_token_table_matches_hf`).
- LLaVA-NeXT AnyRes picks the candidate resolution with the most effective pixels, then the least waste, then the earliest; the padding it adds is later cut from the feature grid, with one `image_newline` per remaining row (`test_select_best_resolution_ties`, `test_unpad_shape_and_fit_size`).
- The tiles equal Hugging Face's PIL processors byte for byte, because every resize is `M12.3`'s Pillow-exact `resize` (`test_idefics3_tiles_match_hf`, `test_anyres_tiles_match_hf`).
- Pixel shuffle moves an $r \times r$ block of tokens into the channels of one token: $r^2$ times fewer tokens, $r^2$ times wider, and exactly invertible (`test_pixel_shuffle_matches_hf_and_inverts`).
- `image_token_count` needs only $(h, w)$ and the config, never the pixels, and a larger copy of an image never costs fewer tokens (`test_tokens_monotone_in_area`).

## How to work this chapter

```bash
ol start L13.4              # stubs python/tinyllm/vision/tiling.py into your repo
ol tests L13.4              # read the test catalog first: rung R0, you write no tests here
ol check L13.4              # exit code is the verdict
ol check L13.4 --ref-deps   # only if your M12.3 is not passing yet
ol diff  L13.4              # after passing: your code against the reference
```

---

## 1. Why now

`L13.3`'s tower was trained at one square size. Feed it a receipt 1000 pixels high and 4000 wide squeezed to $512 \times 512$ and the width shrinks by a factor of $0.128$: a 12-pixel-wide character becomes 1.5 pixels and the text is gone before the model sees it. Current VLMs keep the pixels and pay in tokens: they cut a large image into tower-sized tiles and add one small view of the whole. That makes the number of image tokens a function of the image's size, and three other components need that number before any pixel is decoded: `L13.7` must emit exactly that many placeholders, the engine (`L15.1`) reserves context and encoder budget with it, and the gateway (`gw.10`) meters and rate-limits with it. One wrong rounding and the placeholder count disagrees with the feature count, which is a hard failure, not a quality loss. This module is the two recipes in use today, LLaVA-NeXT AnyRes and Idefics3 / SmolVLM image splitting, plus the table all four implementations are proven against.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $(h, w)$ | original image size, height first | ints |
| $a = w/h$ | aspect ratio | float |
| $T$ | tile side (`tile`): 512 for SmolVLM, 336 for LLaVA-NeXT | int |
| $E$ | `longest_edge` (Idefics3) | int |
| $(H, W)$ | a candidate resolution (a pinpoint), multiples of $T$ | ints |
| $g = T/p$ | feature rows per tile side, $p$ = `patch_size` | int |
| $n_\text{seq}$ | tokens per tile after the connector (`image_seq_len`) | int |

Images are `uint8` HWC arrays (`M12.2`). Sizes and pinpoints are written (height, width) throughout.

### 2.1 What a tile costs

A tower at $T \times T$ with patch $p$ turns any input into $g^2$ features, however many pixels the original had. Tiling makes the cost proportional to area instead: $k$ tiles cost $k$ towers' worth of features, and one extra downsized view carries the layout that tiles lose at their seams. The rest of the module is deciding $k$ from $(h, w)$, with the exact rounding of the published processors.

### 2.2 Idefics3 / SmolVLM: grow, cut, tag

**Step 1, longest edge.** Scale the longer edge to $E$ and keep the aspect ratio. If $w \ge h$: $W_1 = E$, $H_1 = \text{int}(E / a)$, then $H_1$ is rounded up to even. Otherwise mirrored. If the longer edge still exceeds 4096 it is set to 4096 and the other edge to $\text{int}$ of the scaled value.

**Step 2, grow to the tile grid.** With $a_1 = W_1 / H_1$ and $w \ge h$:
$$W' = \lceil W_1 / T \rceil\, T, \qquad H' = \lceil \text{int}(W' / a_1) / T \rceil\, T,$$
mirrored when the height is longer. The height is recomputed from the grown width $W'$, not carried over from $H_1$. That matters: a $505 \times 1000$ image at $E = 1000$ becomes $506 \times 1000$; the width grows to 1024, so the height becomes $\text{int}(1024 \cdot 506/1000) = 518$, which needs two rows of 512, not one. `idefics3_resize_size` returns $(H', W')$, or $(T, T)$ when `split` is off.

**Step 3, cut.** Both resizes are really performed, with Lanczos: to $(H_1, W_1)$, then to $(H', W')$. If $H'$ or $W'$ exceeds $T$, crop `rows` $= H'/T$ by `cols` $= W'/T$ tiles in row-major order and append the global image: the grown image squeezed to $T \times T$. Otherwise (and when `split` is off) there are no tiles, `(rows, cols) = (0, 0)`, and the squeezed image is the only one.

**The prompt.** `idefics3_prompt` writes the text that replaces one image. Each tile is `<fake_token_around_image>`, a tag `<row_R_col_C>` (1-based), then `<image>` repeated $n_\text{seq}$ times; a `\n` follows each row of tiles; then one more `\n`, the fake token, `<global-img>`, the run of `<image>`, and a closing fake token. For 1 row, 2 columns, $n_\text{seq} = 1$:

```text
<fake_token_around_image><row_1_col_1><image><fake_token_around_image><row_1_col_2><image>\n\n<fake_token_around_image><global-img><image><fake_token_around_image>
```

With no tiles it is just the last group. The row and column tags are how the language model learns where each tile sat.

### 2.3 LLaVA-NeXT AnyRes: choose, fit, pad, unpad

**Choose.** A config lists a few allowed resolutions (pinpoints). For a candidate $(H, W)$ the image fits inside it at scale $s = \min(W/w,\ H/h)$, covering $\text{int}(ws) \cdot \text{int}(hs)$ pixels. Upscaling adds no information, so the *effective* resolution is capped at the original:
$$\text{eff} = \min(\text{int}(ws) \cdot \text{int}(hs),\ wh), \qquad \text{waste} = HW - \text{eff}.$$
`select_best_resolution` takes the largest eff, breaks ties by the smallest waste, and on an exact tie keeps the earlier candidate.

**Fit and pad.** `fit_size` gives the size inside $(H, W)$: the tighter edge takes the target exactly and the other is $\min(\lceil \text{edge} \cdot s \rceil, \text{target})$. The image is resized to it with bicubic (Pillow's, $a = -0.5$) and centered on a zero canvas of $H \times W$; when the leftover is odd, the extra pixel goes to the bottom or the right. The canvas is cut into $H/T$ by $W/T$ tiles, row-major. The base image is the *original* squeezed to $T \times T$ with bicubic. `anyres_tiles` returns `(tiles, base, (rows, cols))`.

**Unpad.** The tiles' features form one grid of $G_h = (H/T)\,g$ rows by $G_w = (W/T)\,g$ columns, and some of its rows or columns are pure padding. `unpad_shape((h, w), (G_h, G_w))` computes what is left. If $w/h > G_w/G_h$ the image is wider than the grid, so it fills the width and is padded above and below: the content is $m = \text{int}(\text{round}(h\,G_w / w,\ 7))$ rows, the padding on one side is $q = \lfloor (G_h - m)/2 \rfloor$, and $G_h - 2q$ rows are kept with every column. Otherwise the same on the columns. Note it is $G_h - 2q$, not $m$: with an odd leftover the two differ by one, and both sides are trimmed by the same $q$. The language model then reads the base image's features, then the kept grid row by row with one learned `image_newline` vector after each row.

### 2.4 Token counts

`image_token_count(h, w, cfg, detail)` counts image placeholder tokens (feature rows). `detail="auto"` follows `cfg.split`, `"high"` splits, `"low"` does not.

| `kind` | split | no split |
|---|---|---|
| `idefics3` | $(\text{rows} \cdot \text{cols} + 1)\, n_\text{seq}$ if step 3 produced tiles, else $n_\text{seq}$ | $n_\text{seq}$ |
| `llava_next` | $g^2 + \texttt{base\_extra} + u_h u_w + u_h$, with $(u_h, u_w)$ the unpadded grid | $g^2 + \texttt{base\_extra} + 1$ |
| `single` | $n_\text{seq}$ | $n_\text{seq}$ |

In the `llava_next` row, $u_h u_w$ are the grid cells and $u_h$ the newlines, one per row; the unsplit case has one newline. Wrapper text (fake tokens, row tags, `<global-img>`, the `\n` characters) is not counted, because how many tokens it becomes depends on the tokenizer. Sizes below 1 and unknown `detail` values raise `ValueError`. The fixture holds 40 sizes (tiny, huge, extreme aspect ratios) with the processors' answers; the Rust and Go ports must reproduce the same table.

### 2.5 Pixel shuffle

A $512 \times 512$ tile at patch 16 is a $32 \times 32$ grid, 1024 tokens. Idefics3's connector trades length for width before the language model sees them. Given $x$ of shape $[N, S^2, D]$, a row-major $S \times S$ grid, and a factor $r$ dividing $S$, output token $(i, j)$ of the $(S/r) \times (S/r)$ grid is the concatenation of the block
$$y_{i,j} = [\,x_{ir+a,\ jr+b}\,]_{a = 0..r-1,\ b = 0..r-1},$$
with the row offset $a$ outer and the column offset $b$ inner: each block row contributes $rD$ channels. The shape becomes $[N, S^2/r^2, D r^2]$, so $r = 4$ turns 1024 tokens into the 64 of SmolVLM-256M. Every input entry appears in the output exactly once, so the map is a permutation of entries and `pixel_unshuffle` inverts it exactly. The order is not free to choose: the projector after it was trained on Hugging Face's.

## 3. Worked example by hand

**SmolVLM** ($E = 2048$, $T = 512$, $n_\text{seq} = 64$) on a photo with $h = 1000$, $w = 1500$.

1. $a = 1.5$, $w \ge h$: $W_1 = 2048$, $H_1 = \text{int}(2048/1.5) = \text{int}(1365.33) = 1365$, odd, so $H_1 = 1366$. Neither edge exceeds 4096.
2. $W' = \lceil 2048/512 \rceil \cdot 512 = 2048$. $a_1 = 2048/1366$, so $W'/a_1 \approx 1366$ and $H' = \lceil 1366/512 \rceil \cdot 512 = 3 \cdot 512 = 1536$ (2.67 rounds up to 3).
3. `rows` $= 1536/512 = 3$, `cols` $= 2048/512 = 4$: 12 tiles plus the global image.
4. Tokens: $(12 + 1) \cdot 64 = 832$. With `detail="low"`: 64.

**LLaVA-NeXT** ($T = 336$, $p = 14$, $g = 24$) on $h = 500$, $w = 1000$ with pinpoints $(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)$; $wh = 500000$.

| $(H, W)$ | $s$ | $\text{int}(ws) \times \text{int}(hs)$ | eff | waste |
|---|---|---|---|---|
| $(336, 672)$ | $0.672$ | $672 \times 336$ | 225792 | 0 |
| $(672, 336)$ | $0.336$ | $336 \times 168$ | 56448 | 169344 |
| $(672, 672)$ | $0.672$ | $672 \times 336$ | 225792 | 225792 |
| $(1008, 336)$ | $0.336$ | $336 \times 168$ | 56448 | 282240 |
| $(336, 1008)$ | $0.672$ | $672 \times 336$ | 225792 | 112896 |

Three candidates tie on eff; $(336, 672)$ wins on waste. The feature grid is $(336/336 \cdot 24,\ 672/336 \cdot 24) = (24, 48)$; $w/h = 2$ is not greater than $48/24 = 2$, so the column branch runs: $m = 1000 \cdot 24/500 = 48$, $q = 0$, nothing to remove. Tokens: $576 + 24 \cdot 48 + 24 = 1752$; with `detail="low"`, $577$ (`test_hand_example`, `test_unpad_shape_and_fit_size`).

For a case that does unpad: a $(480, 640)$ image on a $(48, 48)$ grid has $w/h = 1.33 > 1$, so $m = 480 \cdot 48/640 = 36$ rows, $q = 6$, and $(36, 48)$ is kept.

## 4. The interface

```python
@dataclass
class TilingConfig:
    kind: Literal["idefics3", "llava_next", "single"]
    tile: int
    longest_edge: int = 0; image_seq_len: int = 0; pinpoints: tuple = ()
    patch_size: int = 14; base_extra: int = 0; split: bool = True
def select_best_resolution(hw, pinpoints) -> tuple[int, int]: ...
def fit_size(hw, target) -> tuple[int, int]: ...
def anyres_tiles(img, pinpoints, tile) -> tuple[list[NDArray], NDArray, tuple[int, int]]: ...   # tiles, base, (rows, cols)
def unpad_shape(orig_hw, cur_hw) -> tuple[int, int]: ...
def idefics3_resize_size(hw, longest_edge, tile, split=True) -> tuple[int, int]: ...
def idefics3_split(img, longest_edge, tile, split=True) -> tuple[list[NDArray], NDArray, tuple[int, int]]: ...  # tiles, global, (rows, cols)
def idefics3_prompt(rows, cols, image_seq_len, image_token="<image>",
                    fake_token="<fake_token_around_image>", global_tag="<global-img>") -> str: ...
def pixel_shuffle(x, r) -> NDArray: ...       # [N, S*S, D] -> [N, (S/r)^2, D r^2]
def pixel_unshuffle(y, r) -> NDArray: ...
def image_token_count(h, w, cfg, detail: Literal["low", "high", "auto"] = "auto") -> int: ...
```

Tiles, base, and global images are `uint8 [tile, tile, C]`. `anyres_tiles` raises `ValueError` unless every pinpoint is a multiple of `tile`; both splitters raise it for an image that is not `uint8` HWC; `pixel_shuffle` raises it unless the token count is a square $S^2$ and $r$ divides $S$.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3: $(1536, 2048)$, 832 tokens, 64 at `low`, best resolution $(336, 672)$ | you and the test agree on the recipe |
| `test_token_table_matches_hf` | golden | 40 sizes: resized size, Idefics3 tokens, best resolution, LLaVA-NeXT tokens, all exact | the parity table of `L15.1` and `gw.10` |
| `test_prompt_strings_match_hf` | golden | the prompt for grids $0 \times 0$, $1 \times 2$, $2 \times 3$ equals `Idefics3Processor`'s | `L13.7`'s placeholder expansion |
| `test_idefics3_tiles_match_hf` | golden | three images at $E = 48$, $T = 16$: grid and every tile and global pixel equal | `L13.6` preprocessing, `MS-L13` step 3 |
| `test_anyres_tiles_match_hf` | golden | two images: base and tiles equal HF's pixel for pixel | LLaVA-NeXT inputs in `L13.7` |
| `test_pixel_shuffle_matches_hf_and_inverts` | golden | HF's connector order on `[2, 16, 6]` with $r = 2$; unshuffle restores the input, also for $r = 3$ | the projector's trained input layout |
| `test_grow_to_tiles_uses_the_grown_width` | unit | $505 \times 1000$ and its transpose at $E = 1000$ give $(1024, 1024)$, 50 tokens at $n_\text{seq} = 10$; no split gives $(512, 512)$ | section 2.2's step 2 |
| `test_unpad_shape_and_fit_size` | unit | `fit_size` and `unpad_shape` cases, 1752 and 577 tokens, a `single` config | newlines and unpadding agree with the features |
| `test_select_best_resolution_ties` | unit | eff first, then waste, then the earlier candidate | deterministic choice across ports |
| `test_tokens_monotone_in_area` | property | five aspect ratios scaled by 16 to 512: counts never decrease, both processors | rate limits and encoder budgets |
| `test_rejects_bad_inputs` | boundary | zero size, unknown `detail`, empty pinpoints, bad shuffle factor, non-multiple pinpoints, float image | caller bugs fail loudly |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. an exact tie takes the later candidate | a transposed grid for square images | `test_select_best_resolution_ties`, `test_hand_example` (mutant `s01`) |
| 2. the height grown from the old height, not the grown width | one row of tiles too few | `test_grow_to_tiles_uses_the_grown_width`, `test_idefics3_tiles_match_hf` (mutant `s02`) |
| 3. one newline per column instead of per row | counts right only for square grids | `test_token_table_matches_hf`, `test_unpad_shape_and_fit_size` (mutant `s03`) |
| 4. the odd padding pixel on top or left | every tile shifted by one pixel | `test_anyres_tiles_match_hf` (mutant `s04`) |
| 5. pixel shuffle without the last transpose | right shape, tokens in column-major order | `test_pixel_shuffle_matches_hf_and_inverts` (mutant `s05`) |
| 6. no newline after each row of tiles | the model cannot tell where a row ends | `test_prompt_strings_match_hf` (mutant `s06`) |
| 7. the longest-edge resize keeps an odd edge | 1365 rows instead of 1366; pixels and sometimes the grid differ | `test_idefics3_tiles_match_hf`, `test_hand_example` (mutant `s07`) |
| 8. unpadding removes one side's padding only | too many rows kept, too many tokens | `test_unpad_shape_and_fit_size`, `test_token_table_matches_hf` (mutant `s08`) |
| accepting a zero-size image | a token count for nothing | `test_rejects_bad_inputs` (mutant `m01`) |
| accepting an unknown `detail` | a typo silently means "do not split" | `test_rejects_bad_inputs` (mutant `m02`) |
| accepting pinpoints that are not tile multiples | a ragged last tile | `test_rejects_bad_inputs` (mutant `m03`) |

One more without a mutant: sizes are (height, width) here, but Pillow's `Image.resize` and many configs write (width, height). Swap them and every aspect ratio above $1$ takes the wrong branch.

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M12.3` | `resize` in its Pillow-exact mode: Lanczos for Idefics3, bicubic for AnyRes |
| Back | `M12.2` | HWC `uint8` pixels and row-major grids (reading) |
| Back | `M05.1` | counting tokens against a budget (reading) |
| Forward | `L13.6` | the image processor calls the splitters and builds a `TilingConfig` from its `ProcessorConfig` |
| Forward | `L13.7` | expands each image placeholder with `idefics3_prompt` or the AnyRes count, unpads features, applies `pixel_shuffle` in the connector |
| Forward | `L15.1` | the Rust engine re-implements `image_token_count` and is checked on the 40-size table |
| Forward | `gw.10` | the Go gateway's `ImageTokens` meters requests with the same table |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `idefics3_split`, `idefics3_prompt` | HF `Idefics3ImageProcessor`, `SmolVLMProcessor` | batching with pixel masks, torch-backed fast processors, video frames | `transformers/models/idefics3/image_processing_idefics3.py` |
| `anyres_tiles`, `unpad_shape` | HF `LlavaNextImageProcessor`, LLaVA-OneVision | larger pinpoint sets, bilinear pooling of tile features, video | `transformers/models/llava_next/` |
| `pixel_shuffle` | `torch.nn.PixelUnshuffle`, InternVL's connector | the same space-to-depth trade on NCHW tensors | Shi et al. (2016) |
| tiling as a whole | Qwen2-VL naive dynamic resolution, SigLIP 2 NaFlex | no tiles: one variable-length patch sequence at native aspect ratio | `transformers/models/qwen2_vl/image_processing_qwen2_vl.py` |
| `image_token_count` | OpenAI and Anthropic image token formulas | published billing rules that a gateway mirrors before dispatch | the providers' vision pricing documentation |
