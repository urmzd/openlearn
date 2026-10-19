# Dynamic resolution and tiling

The tower from [Vision transformer encoder](03-vision-transformer.md) was trained at one square size. Feed it a receipt 1000 pixels high and 4000 wide squeezed to $512 \times 512$ and the width shrinks by a factor of $0.128$: a 12-pixel-wide character becomes 1.5 pixels and the text is gone before the model sees it. Current vision-language models keep the pixels and pay in tokens instead. They cut a large image into tower-sized tiles and add one small view of the whole.

That makes the number of image tokens a function of the image's size, and several other components need that number before any pixel is decoded. The planned vision-language model must emit exactly that many placeholders, a serving engine reserves context and encoder budget with it, and a gateway meters and rate-limits with it. One wrong rounding and the placeholder count disagrees with the feature count, which is a hard failure, not a quality loss. This section builds the two recipes in use today, LLaVA-NeXT AnyRes and Idefics3 / SmolVLM image splitting, plus a 40-size reference table that every implementation (this Python one, and planned Rust and Go ports) is proven against.

You will need the Pillow-exact `resize` from [Resampling and interpolation](../../mathematics-for-computing/12-signals-and-images/03-resampling-and-interpolation.md) in *Mathematics for Computing*. The HWC pixel layout of [Image tensors and layouts](../../mathematics-for-computing/12-signals-and-images/02-image-tensors-and-layouts.md) and the token budgeting of [Counting: params, FLOPs, KV bytes, memory plans](../../mathematics-for-computing/05-discrete-math-1/01-counting-params-flops-kv-bytes.md) are useful background. The full solution is [`tiling.py`](../../../../course/solutions/python/tinyllm/vision/tiling.py) and its contract is [`course/contracts/py/tinyllm/vision/tiling.pyi`](../../../../course/contracts/py/tinyllm/vision/tiling.pyi).

> **Learning objectives**
> - Compute the Idefics3 / SmolVLM tile grid and token count for any image size, including the published rounding.
> - Choose a LLaVA-NeXT AnyRes resolution, fit and pad the image, and unpad the feature grid.
> - Count image placeholder tokens from $(h, w)$ and a config alone, without decoding pixels.
> - Implement pixel shuffle in Hugging Face's order and invert it exactly.

## Notation

These symbols appear throughout. Images are `uint8` HWC arrays, and sizes and pinpoints are written (height, width).

| Symbol | Meaning | Type / shape |
|---|---|---|
| $(h, w)$ | original image size, height first | ints |
| $a = w/h$ | aspect ratio | float |
| $T$ | tile side (`tile`): 512 for SmolVLM, 336 for LLaVA-NeXT | int |
| $E$ | `longest_edge` (Idefics3) | int |
| $(H, W)$ | a candidate resolution (a pinpoint), multiples of $T$ | ints |
| $g = T/p$ | feature rows per tile side, $p$ = `patch_size` | int |
| $n_\text{seq}$ | tokens per tile after the connector (`image_seq_len`) | int |

## What a tile costs

A tower at $T \times T$ with patch $p$ turns any input into $g^2$ features, however many pixels the original had. Tiling makes the cost proportional to area instead: $k$ tiles cost $k$ towers' worth of features, and one extra downsized view carries the layout that tiles lose at their seams. The rest of this section is about deciding $k$ from $(h, w)$, with the exact rounding of the published processors.

## Idefics3 and SmolVLM: grow, cut, tag

The Idefics3 recipe has three steps. First, scale the longer edge to $E$ and keep the aspect ratio. If $w \ge h$: $W_1 = E$, $H_1 = \text{int}(E / a)$, then $H_1$ is rounded up to even. Otherwise mirrored. If the longer edge still exceeds 4096 it is set to 4096 and the other edge to $\text{int}$ of the scaled value.

Second, grow to the tile grid. With $a_1 = W_1 / H_1$ and $w \ge h$:
$$W' = \lceil W_1 / T \rceil\, T, \qquad H' = \lceil \text{int}(W' / a_1) / T \rceil\, T,$$
mirrored when the height is longer. The height is recomputed from the grown width $W'$, not carried over from $H_1$, and that matters. A $505 \times 1000$ image at $E = 1000$ becomes $506 \times 1000$; the width grows to 1024, so the height becomes $\text{int}(1024 \cdot 506/1000) = 518$, which needs two rows of 512, not one. `idefics3_resize_size` returns $(H', W')$, or $(T, T)$ when `split` is off.

Third, cut. Both resizes are really performed, with Lanczos: to $(H_1, W_1)$, then to $(H', W')$. If $H'$ or $W'$ exceeds $T$, crop `rows` $= H'/T$ by `cols` $= W'/T$ tiles in row-major order and append the global image: the grown image squeezed to $T \times T$. Otherwise (and when `split` is off) there are no tiles, `(rows, cols) = (0, 0)`, and the squeezed image is the only one.

The language model also needs to know where each tile sat, which is the job of the prompt. `idefics3_prompt` writes the text that replaces one image. Each tile is `<fake_token_around_image>`, a tag `<row_R_col_C>` (1-based), then `<image>` repeated $n_\text{seq}$ times; a `\n` follows each row of tiles; then one more `\n`, the fake token, `<global-img>`, the run of `<image>`, and a closing fake token. For 1 row, 2 columns, $n_\text{seq} = 1$:

```text
<fake_token_around_image><row_1_col_1><image><fake_token_around_image><row_1_col_2><image>\n\n<fake_token_around_image><global-img><image><fake_token_around_image>
```

With no tiles it is just the last group. The row and column tags are how the language model learns where each tile sat.

## LLaVA-NeXT AnyRes: choose, fit, pad, unpad

AnyRes takes a different approach: instead of growing the image to a grid, it picks the best of a few fixed grids. A config lists the allowed resolutions (pinpoints). For a candidate $(H, W)$ the image fits inside it at scale $s = \min(W/w,\ H/h)$, covering $\text{int}(ws) \cdot \text{int}(hs)$ pixels. Upscaling adds no information, so the *effective* resolution is capped at the original:
$$\text{eff} = \min(\text{int}(ws) \cdot \text{int}(hs),\ wh), \qquad \text{waste} = HW - \text{eff}.$$
`select_best_resolution` takes the largest eff, breaks ties by the smallest waste, and on an exact tie keeps the earlier candidate.

Once the resolution is chosen, `fit_size` gives the size inside $(H, W)$: the tighter edge takes the target exactly and the other is $\min(\lceil \text{edge} \cdot s \rceil, \text{target})$. The image is resized to it with bicubic (Pillow's, $a = -0.5$) and centered on a zero canvas of $H \times W$; when the leftover is odd, the extra pixel goes to the bottom or the right. The canvas is cut into $H/T$ by $W/T$ tiles, row-major. The base image is the *original* squeezed to $T \times T$ with bicubic. `anyres_tiles` returns `(tiles, base, (rows, cols))`.

The padding added to the canvas has to come out again on the feature side. The tiles' features form one grid of $G_h = (H/T)\,g$ rows by $G_w = (W/T)\,g$ columns, and some of its rows or columns are pure padding. `unpad_shape((h, w), (G_h, G_w))` computes what is left. If $w/h > G_w/G_h$ the image is wider than the grid, so it fills the width and is padded above and below: the content is $m = \text{int}(\text{round}(h\,G_w / w,\ 7))$ rows, the padding on one side is $q = \lfloor (G_h - m)/2 \rfloor$, and $G_h - 2q$ rows are kept with every column. Otherwise the same on the columns. Note it is $G_h - 2q$, not $m$: with an odd leftover the two differ by one, and both sides are trimmed by the same $q$. The language model then reads the base image's features, then the kept grid row by row with one learned `image_newline` vector after each row.

## Token counts

With both recipes defined, the token count follows. `image_token_count(h, w, cfg, detail)` counts image placeholder tokens (feature rows). `detail="auto"` follows `cfg.split`, `"high"` splits, `"low"` does not. The formulas per kind are:

| `kind` | split | no split |
|---|---|---|
| `idefics3` | $(\text{rows} \cdot \text{cols} + 1)\, n_\text{seq}$ if the cut step produced tiles, else $n_\text{seq}$ | $n_\text{seq}$ |
| `llava_next` | $g^2 + \texttt{base\_extra} + u_h u_w + u_h$, with $(u_h, u_w)$ the unpadded grid | $g^2 + \texttt{base\_extra} + 1$ |
| `single` | $n_\text{seq}$ | $n_\text{seq}$ |

In the `llava_next` row, $u_h u_w$ are the grid cells and $u_h$ the newlines, one per row; the unsplit case has one newline. Wrapper text (fake tokens, row tags, `<global-img>`, the `\n` characters) is not counted, because how many tokens it becomes depends on the tokenizer. Sizes below 1 and unknown `detail` values raise `ValueError`. A table of 40 sizes (tiny, huge, extreme aspect ratios) with the processors' answers is the parity reference that any port must reproduce.

## Pixel shuffle

Tiling multiplies the token count, so connectors claw some of it back. A $512 \times 512$ tile at patch 16 is a $32 \times 32$ grid, 1024 tokens. Idefics3's connector trades length for width before the language model sees them. Given $x$ of shape $[N, S^2, D]$, a row-major $S \times S$ grid, and a factor $r$ dividing $S$, output token $(i, j)$ of the $(S/r) \times (S/r)$ grid is the concatenation of the block
$$y_{i,j} = [\,x_{ir+a,\ jr+b}\,]_{a = 0..r-1,\ b = 0..r-1},$$
with the row offset $a$ outer and the column offset $b$ inner: each block row contributes $rD$ channels. The shape becomes $[N, S^2/r^2, D r^2]$, so $r = 4$ turns 1024 tokens into the 64 of SmolVLM-256M. Every input entry appears in the output exactly once, so the map is a permutation of entries and `pixel_unshuffle` inverts it exactly. The order is not free to choose: the projector after it was trained on Hugging Face's.

## Working it by hand

Run SmolVLM ($E = 2048$, $T = 512$, $n_\text{seq} = 64$) on a photo with $h = 1000$, $w = 1500$.

1. $a = 1.5$, $w \ge h$: $W_1 = 2048$, $H_1 = \text{int}(2048/1.5) = \text{int}(1365.33) = 1365$, odd, so $H_1 = 1366$. Neither edge exceeds 4096.
2. $W' = \lceil 2048/512 \rceil \cdot 512 = 2048$. $a_1 = 2048/1366$, so $W'/a_1 \approx 1366$ and $H' = \lceil 1366/512 \rceil \cdot 512 = 3 \cdot 512 = 1536$ (2.67 rounds up to 3).
3. `rows` $= 1536/512 = 3$, `cols` $= 2048/512 = 4$: 12 tiles plus the global image.
4. Tokens: $(12 + 1) \cdot 64 = 832$. With `detail="low"`: 64.

Now run LLaVA-NeXT ($T = 336$, $p = 14$, $g = 24$) on $h = 500$, $w = 1000$ with pinpoints $(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)$; $wh = 500000$. Each candidate scores as follows.

| $(H, W)$ | $s$ | $\text{int}(ws) \times \text{int}(hs)$ | eff | waste |
|---|---|---|---|---|
| $(336, 672)$ | $0.672$ | $672 \times 336$ | 225792 | 0 |
| $(672, 336)$ | $0.336$ | $336 \times 168$ | 56448 | 169344 |
| $(672, 672)$ | $0.672$ | $672 \times 336$ | 225792 | 225792 |
| $(1008, 336)$ | $0.336$ | $336 \times 168$ | 56448 | 282240 |
| $(336, 1008)$ | $0.672$ | $672 \times 336$ | 225792 | 112896 |

Three candidates tie on eff; $(336, 672)$ wins on waste. The feature grid is $(336/336 \cdot 24,\ 672/336 \cdot 24) = (24, 48)$; $w/h = 2$ is not greater than $48/24 = 2$, so the column branch runs: $m = 1000 \cdot 24/500 = 48$, $q = 0$, nothing to remove. Tokens: $576 + 24 \cdot 48 + 24 = 1752$; with `detail="low"`, $577$.

For a case that does unpad: a $(480, 640)$ image on a $(48, 48)$ grid has $w/h = 1.33 > 1$, so $m = 480 \cdot 48/640 = 36$ rows, $q = 6$, and $(36, 48)$ is kept.

## Designing the API

The interface splits size arithmetic from pixel work:

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

## Building it step by step

Five principles drive the implementation.

1. **The token count must be a pure function of $(h, w)$ and the config.** The engine and the gateway need it before any pixel is decoded, and the language model needs exactly as many placeholders as there are features. So the solution separates *size arithmetic* (`select_best_resolution`, `fit_size`, `unpad_shape`, `idefics3_resize_size`) from *pixel work* (`anyres_tiles`, `idefics3_split`), and `image_token_count` calls only the arithmetic.
2. **The arithmetic must reproduce the published processors' rounding, including their quirks.** `int` truncation, rounding an edge up to even, recomputing the height from the grown width, `round(..., 7)` before `int` in unpadding: each is copied because changing any one changes some image's count, which then disagrees with the model.
3. **The pixels must be identical too.** Every resize goes through the Pillow-exact `resize` with the processor's kernel (Lanczos for Idefics3, bicubic for AnyRes), and crops are plain slices in row-major order.
4. **Choosing a resolution is a lexicographic maximum.** Most effective pixels, then least waste, then first. Writing the comparison as "strictly better" makes the third rule automatic.
5. **Pixel shuffle is a reshape-and-transpose permutation.** Any reshape sequence that groups an $r \times r$ block gives the right shape; only Hugging Face's sequence gives the order the projector was trained on, and the inverse is the same steps backwards.

AnyRes selection comes first:

```python
def select_best_resolution(hw, pinpoints):
    h, w = (int(v) for v in hw)
    # ... raise on a size below 1 or no candidates
    best, best_eff, best_waste = None, 0, math.inf
    for ph, pw in pinpoints:
        scale = min(pw / w, ph / h)                  # fit inside, aspect kept
        dw, dh = int(w * scale), int(h * scale)
        eff = min(dw * dh, w * h)                    # upscaling adds no information
        waste = pw * ph - eff
        # Strictly better only: an exact tie keeps the earlier candidate.
        if eff > best_eff or (eff == best_eff and waste < best_waste):
            best, best_eff, best_waste = (int(ph), int(pw)), eff, waste
    return best
```

Then fitting, padding, and cropping. The tighter edge is exact, the other rounded up and capped; the odd leftover pixel goes after:

```python
def fit_size(hw, target):
    h, w = hw
    th, tw = target
    sw, sh = tw / w, th / h
    if sw < sh:
        return min(math.ceil(h * sw), th), tw        # width is the tight edge
    return th, min(math.ceil(w * sh), tw)

# in anyres_tiles:
th, tw = select_best_resolution((h, w), pinpoints)
nh, nw = fit_size((h, w), (th, tw))
fitted = resize(a, (nh, nw), kernel)                 # Pillow bicubic
py, ry = divmod(th - nh, 2)
px, rx = divmod(tw - nw, 2)
canvas = np.pad(fitted, ((py, py + ry), (px, px + rx), (0, 0)))   # centered; extra pixel bottom/right
rows, cols = th // tile, tw // tile
tiles = [canvas[i * tile : (i + 1) * tile, j * tile : (j + 1) * tile].copy() for i in range(rows) for j in range(cols)]
base = resize(a, (tile, tile), kernel)               # the original, not the canvas
```

Unpadding trims the same $q$ from both sides:

```python
def unpad_shape(orig_hw, cur_hw):
    h, w = orig_hw
    ch, cw = cur_hw
    if w / h > cw / ch:                              # wider than the grid: padded above and below
        new_h = int(round(h * (cw / w), 7))          # round to 7 places first, as the processor does
        pad = (ch - new_h) // 2
        return ch - 2 * pad, cw
    new_w = int(round(w * (ch / h), 7))
    pad = (cw - new_w) // 2
    return ch, cw - 2 * pad
```

For Idefics3, `_longest_edge_size` is the first step (edge to `max_len`, other edge rounded up to even, then the 4096 cap), and `idefics3_resize_size` is the second:

```python
def idefics3_resize_size(hw, longest_edge, tile, split=True):
    h, w = (int(v) for v in hw)
    # ... raise on a size below 1
    if not split:
        return tile, tile
    if longest_edge:
        h, w = _longest_edge_size(h, w, longest_edge)
    aspect = w / h
    if w >= h:
        w = math.ceil(w / tile) * tile
        h = math.ceil(int(w / aspect) / tile) * tile   # from the grown w, not the old h
    else:
        h = math.ceil(h / tile) * tile
        w = math.ceil(int(h * aspect) / tile) * tile
    return h, w
```

`idefics3_split` performs the same two resizes on the pixels (Lanczos), crops `rows x cols` tiles row-major when either edge exceeds `tile`, and returns the grown image squeezed to `tile x tile` as the global image. `idefics3_prompt` writes the prompt string:

```python
run = image_token * image_seq_len
if rows == 0 and cols == 0:
    return fake_token + global_tag + run + fake_token     # unsplit: only the global group
out = ""
for r in range(rows):
    for c in range(cols):
        out += fake_token + f"<row_{r + 1}_col_{c + 1}>" + run   # 1-based tags
    out += "\n"                                           # one newline per row of tiles
return out + "\n" + fake_token + global_tag + run + fake_token
```

Pixel shuffle follows `Idefics3Connector` op for op:

```python
n, seq, d = a.shape
s = math.isqrt(seq)
# ... raise unless seq = s*s and r divides s
y = a.reshape(n, s, s, d).reshape(n, s, s // r, d * r)          # merge r adjacent columns into channels
y = y.transpose(0, 2, 1, 3).reshape(n, s // r, s // r, d * r * r)  # swap axes, merge r adjacent rows
return y.transpose(0, 2, 1, 3).reshape(n, seq // (r * r), d * r * r)  # swap back: row-major blocks
```

`pixel_unshuffle` applies the inverse reshapes and transposes in reverse order, so the round trip is the identity on every entry.

Finally, the token count uses only the arithmetic:

```python
split = cfg.split if detail == "auto" else detail == "high"
if cfg.kind == "single":
    return cfg.image_seq_len
if cfg.kind == "idefics3":
    if not split:
        return cfg.image_seq_len
    bh, bw = idefics3_resize_size((h, w), cfg.longest_edge, cfg.tile)
    if bh <= cfg.tile and bw <= cfg.tile:
        return cfg.image_seq_len                               # no tiles: the global image only
    return ((bh // cfg.tile) * (bw // cfg.tile) + 1) * cfg.image_seq_len   # tiles plus the global image
if cfg.kind == "llava_next":
    side = cfg.tile // cfg.patch_size                          # g
    base = side * side + cfg.base_extra
    if not split:
        return base + 1                                        # one newline
    th, tw = select_best_resolution((h, w), cfg.pinpoints)
    uh, uw = unpad_shape((h, w), (th // cfg.tile * side, tw // cfg.tile * side))
    return base + uh * uw + uh                                 # cells plus one newline per row
```

The full solution is [`tiling.py`](../../../../course/solutions/python/tinyllm/vision/tiling.py).

### Why this design

`image_token_count` is $O(\text{pinpoints})$ arithmetic, cheap enough for a gateway to run on every request. The splitters cost a few resizes, $O(hw)$ each, plus copies of the tiles. A count computed by actually splitting would need the decoded image and would be too slow for metering, and sharing the size functions between `image_token_count` and the splitters guarantees the count and the tiles always agree.

Bug-for-bug parity is deliberate. A "nicer" rounding (keep the height, skip the even rounding) would give a different count for some sizes, and a mismatch between placeholders and features is a hard failure in the model. The 40-size table and any ports depend on the exact published arithmetic.

Tiling itself is a tradeoff. It reuses a fixed-size tower unchanged, at the cost of seams and a global view to compensate. Qwen2-VL's variable-length sequence avoids tiles but needs 2D RoPE and a tower trained for it, as in [2D RoPE and M-RoPE](02-rope-2d-and-m-rope.md) and [Vision transformer encoder](03-vision-transformer.md).

## What can go wrong

The test suite pins exact values against the Hugging Face processors.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example | unit, smoke | the hand-worked $(1536, 2048)$, 832 tokens, 64 at `low`, best resolution $(336, 672)$ | the recipe is pinned to exact values |
| token table matches HF | golden | 40 sizes: resized size, Idefics3 tokens, best resolution, LLaVA-NeXT tokens, all exact | the parity table for the planned Rust and Go ports |
| prompt strings match HF | golden | the prompt for grids $0 \times 0$, $1 \times 2$, $2 \times 3$ equals `Idefics3Processor`'s | placeholder expansion in a VLM |
| Idefics3 tiles match HF | golden | three images at $E = 48$, $T = 16$: grid and every tile and global pixel equal | preprocessing to spec |
| AnyRes tiles match HF | golden | two images: base and tiles equal HF's pixel for pixel | LLaVA-NeXT inputs |
| pixel shuffle matches HF and inverts | golden | HF's connector order on `[2, 16, 6]` with $r = 2$; unshuffle restores the input, also for $r = 3$ | the projector's trained input layout |
| growing to tiles uses the grown width | unit | $505 \times 1000$ and its transpose at $E = 1000$ give $(1024, 1024)$, 50 tokens at $n_\text{seq} = 10$; no split gives $(512, 512)$ | the grow step |
| `unpad_shape` and `fit_size` | unit | `fit_size` and `unpad_shape` cases, 1752 and 577 tokens, a `single` config | newlines and unpadding agree with the features |
| best-resolution ties | unit | eff first, then waste, then the earlier candidate | deterministic choice across ports |
| tokens monotone in area | property | five aspect ratios scaled by 16 to 512: counts never decrease, both processors | rate limits and encoder budgets |
| bad inputs rejected | boundary | zero size, unknown `detail`, empty pinpoints, bad shuffle factor, non-multiple pinpoints, float image | caller bugs fail loudly |

Most pitfalls are a single rounding or ordering decision gone the other way. An exact tie that takes the later candidate gives a transposed grid for square images; `select_best_resolution` replaces only on strictly better eff or waste. Growing the height from the old height instead of the grown width gives one row of tiles too few; `idefics3_resize_size` computes `h` from the grown `w` and the aspect. A longest-edge resize that keeps an odd edge gives 1365 rows instead of 1366, so the pixels and sometimes the grid differ; `_longest_edge_size` adds `h % 2`. Putting the odd padding pixel on top or left shifts every tile by one pixel; `np.pad` puts the `divmod` remainder after, `(py, py + ry)`. And unpadding that removes only one side's padding keeps too many rows and too many tokens; `unpad_shape` keeps `ch - 2 * pad`.

Newlines are another source of trouble. One newline per column instead of per row gives counts that are right only for square grids, so `image_token_count` adds `uh`, the kept rows. Forgetting the newline after each row of tiles in the prompt means the model cannot tell where a row ends; `idefics3_prompt` appends `"\n"` after each row. Pixel shuffle without its last transpose has the right shape but puts tokens in column-major order, which is why `pixel_shuffle` follows `Idefics3Connector` op for op.

Validation catches the rest. A zero-size image would get a token count for nothing, and an unknown `detail` would silently mean "do not split", so `image_token_count` raises for both. Pinpoints that are not tile multiples would give a ragged last tile, so `anyres_tiles` raises.

> **Warning:** Sizes are (height, width) here, but Pillow's `Image.resize` and many configs write (width, height). Swap them and every aspect ratio above 1 takes the wrong branch.

## Where this leads

[Image preprocessing to the specification](06-image-preprocessing.md) calls these splitters from its image processor and builds a `TilingConfig` from its `ProcessorConfig`. The planned vision-language model section will expand each image placeholder with `idefics3_prompt` or the AnyRes count, unpad features, and apply `pixel_shuffle` in the connector. Planned Rust engine and Go gateway ports will re-implement `image_token_count` and be checked on the same 40-size table, the gateway using it to meter requests.

In production, Hugging Face's `Idefics3ImageProcessor` and `SmolVLMProcessor` add batching with pixel masks, torch-backed fast processors, and video frames (`transformers/models/idefics3/image_processing_idefics3.py`). `LlavaNextImageProcessor` and LLaVA-OneVision add larger pinpoint sets, bilinear pooling of tile features, and video (`transformers/models/llava_next/`). `torch.nn.PixelUnshuffle` and InternVL's connector make the same space-to-depth trade on NCHW tensors. Qwen2-VL's naive dynamic resolution and SigLIP 2 NaFlex drop tiles for one variable-length patch sequence at native aspect ratio (`transformers/models/qwen2_vl/image_processing_qwen2_vl.py`). And OpenAI and Anthropic publish image token formulas as billing rules that a gateway mirrors before dispatch; see the providers' vision pricing documentation.

### Further reading

- Liu et al., "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge" (2024).
- Laurençon et al., "Building and better understanding vision-language models" (2024), for Idefics3.
- Marafioti et al., "SmolVLM: Redefining small and efficient multimodal models" (2025).
- Shi et al., "Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network" (2016), for pixel shuffle.

## Key terms

- **tile**: a tower-sized crop of a large image, encoded independently.
- **global image**: the whole image squeezed to one tile, which carries the layout tiles lose at their seams.
- **pinpoint**: one of the candidate resolutions AnyRes may choose, a multiple of the tile size.
- **effective resolution**: the number of original pixels a candidate preserves, capped at the original area.
- **unpadding**: removing feature rows or columns that correspond only to canvas padding.
- **pixel shuffle**: a permutation that moves an $r \times r$ block of tokens into the channels of one token.

## Summary

- Idefics3 / SmolVLM splitting is integer arithmetic on the size: longest edge to 2048 (other edge rounded up to even), each edge grown to a multiple of 512, a `rows x cols` grid of tiles plus one global image, so a $1000 \times 1500$ photo costs $(3 \cdot 4 + 1) \cdot 64 = 832$ tokens.
- LLaVA-NeXT AnyRes picks the candidate resolution with the most effective pixels, then the least waste, then the earliest; the padding it adds is later cut from the feature grid, with one `image_newline` per remaining row.
- The tiles equal Hugging Face's PIL processors byte for byte, because every resize is the Pillow-exact `resize`.
- Pixel shuffle moves an $r \times r$ block of tokens into the channels of one token: $r^2$ times fewer tokens, $r^2$ times wider, and exactly invertible.
- `image_token_count` needs only $(h, w)$ and the config, never the pixels, and a larger copy of an image never costs fewer tokens.

## Review questions

1. Why does a $505 \times 1000$ image at $E = 1000$ need two rows of 512 tiles?

   <details><summary>Answer</summary>

   The longest-edge step makes it $506 \times 1000$ (505 is odd). The grow step grows the width to 1024 and recomputes the height from it: $\text{int}(1024 \cdot 506/1000) = 518 > 512$, so two rows. Growing the old height 506 directly would give one row, and the count would disagree with the processor.

   </details>

2. In the LLaVA-NeXT example, why does $(336, 672)$ beat $(336, 1008)$ although both have the same effective resolution?

   <details><summary>Answer</summary>

   Both fit the image at scale 0.672 to $672 \times 336$ pixels, eff 225792. The tie breaks on waste: $336 \cdot 672 - 225792 = 0$ against $336 \cdot 1008 - 225792 = 112896$.

   </details>

3. Why does `unpad_shape` keep $G_h - 2q$ rows and not the content height $m$?

   <details><summary>Answer</summary>

   That is what the processor does: it trims the same $q = \lfloor (G_h - m)/2 \rfloor$ from both sides. With an odd leftover, $G_h - 2q = m + 1$. Keeping $m$ would drop one row of features the model expects and the count would be off by $u_w + 1$.

   </details>

4. Why is a larger copy of an image never cheaper, and how far does that argument go?

   <details><summary>Answer</summary>

   For Idefics3 with `longest_edge` set, the first step rescales every copy to the same longest edge, so copies of one aspect ratio land on the same grid; the original size matters only through the rounding of the short edge. For LLaVA-NeXT, the fitted size inside each candidate depends only on the aspect ratio, while the cap $wh$ grows with the copy, so every candidate's effective resolution can only rise and the chosen grid tends to grow. These are arguments for the common case, not a proof for every rounding; the monotonicity is checked on five aspect ratios scaled from 16 to 512 pixels, and it is what lets a gateway set rate limits by size.

   </details>

5. What would go wrong with a pixel shuffle that gives the right shape in a different order?

   <details><summary>Answer</summary>

   Each output token would hold the same $r^2$ input entries in a different channel arrangement, or entries from a different block. The shape and even the round trip could be fine, but the trained projector after it would read channels it was not trained on, and the features would be garbage.

   </details>
