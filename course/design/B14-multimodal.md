# Supersource Course: B14 Multimodal (vision and speech)

> **Merged** into [`course/DESIGN.md`](../DESIGN.md) on 2026-10-10 (D41 to D52, Pass 12 rows in every section; no FFI: no `tinyllm_rs`, `tl-py`, `tl-sys`, or `kernels = "c"`, and L15.8 is retired). DESIGN.md is the spec; this draft is kept for history only.

| | |
|---|---|
| **Status** | Draft v1, proposed section for `course/DESIGN.md`; not merged (a build workflow is reading DESIGN.md) |
| **Date** | 2026-10-09 |
| **Branch** | `feat/course` |
| **Scope** | Pass 12 (`course-p12-multimodal`): signal and image math, vision models, VLM, Whisper-style ASR, multimodal data, engine encoder cache and encode/prefill/decode disaggregation, API v2.1 content parts and transcriptions, gateway limits, routing and metering, deploy, SLOs, drills, evals, ethics |
| **Scope change** | Kernels are useful, not mandatory. The **core path** runs vision and speech compute on **candle** in the Rust engine (`candle-core`, `candle-nn`; Metal on macOS, CPU in CI), with numpy references in Python. C kernels for im2col conv2d, the conv1d stem, patch embedding, resize, and STFT + mel are **optional** modules (`L13.8`, `L13.9`, `L14.6`) reached through an engine backend switch (`[engine.mm] kernels = "candle" \| "c"`, wired by the optional `L15.8`). Every core milestone passes without them |

Contents:

1. Overview, decisions, prerequisites, pass placement
2. New and changed contracts
3. Module catalog
4. Solve set
5. Milestones
6. Fixtures, assets, oracle generators, licenses
7. B14 batch entry
8. Open questions

House rules from DESIGN.md apply unchanged: ids per 3.5, registry per 3.4, six-beat chapters per 6, tests under 10 s with numpy-only Python course tests, oracles from maintainer-only `course/oracle/` scripts, large assets through `ss fetch`, every part ending in a milestone run through the learner's own entry points, `--smoke` for kind-only tiers, no U+2014 in prose.

---

## 1. Overview

### 1.1 What the learner builds in Pass 12

After Pass 11 the learner owns a `v1.0.0` text platform. Pass 12 makes it see and hear, end to end, without breaking the text path:

1. **Math** (`M12`, Python): convolution as a linear map, image layouts, resampling, DFT and FFT, windows and STFT, mel filterbanks and log-mel. These are the numpy reference semantics for everything below.
2. **Vision models** (`L13`, Python): conv layers through im2col, 2D RoPE and M-RoPE, a ViT encoder, dynamic resolution and tiling with pixel shuffle, CLIP and SigLIP, image preprocessing to a written spec, and a LLaVA/Idefics3-style VLM that loads **SmolVLM-256M-Instruct** and matches HF logits.
3. **Speech models** (`L14`, Python): the exact Whisper log-mel frontend, a Whisper encoder and decoder that load **whisper-tiny** and match HF, Whisper decoding (special tokens, timestamps, greedy, beam, temperature fallback, long-form), and ASR plus multimodal eval metrics.
4. **Serving** (`L15`, Rust): media preprocessing in the engine, an encoder cache keyed by content hash, prefix-cache keys that carry image hashes, encoder-aware scheduling, encode/prefill/decode disaggregation (EPD), Whisper serving, and `/v1/audio/transcriptions`. Vision towers and Whisper run on candle; the text LLM keeps running on the learner's C kernels. Optional C kernels (conv via im2col, the conv1d stem, patch embedding, resize, STFT + mel) plug in behind `[engine.mm] kernels = "c"` for learners who want the kernel view; core milestones never need them.
5. **Control plane and ops**: gateway media validation, limits, metering, and pool routing (Go); a durable multimodal corpus workflow; encoder and ASR pools as their own Deployments with separate autoscaling; SLOs for encode time and real-time factor; two new drills.
6. **Data, evals, ethics**: image-text and audio-transcript pipelines with perceptual-hash dedup, EXIF stripping, face redaction, speaker-safe splits, consent in the ledger; WER/CER, VQA, captioning, zero-shot accuracy; accent WER gaps; model-card additions.

The pass gate tags `v1.1.0`: an additive minor release (API `2.1.0`, proto minors, formats minors), which is itself a craft.11/craft.12 exercise in shipping additive change behind unchanged contracts.

### 1.2 Decisions for B14

Binding for this batch, in the D-table style of DESIGN 1.5. Ids `E1` onward avoid colliding with `D1` to `D38`.

| # | Decision | Rejected alternative(s) | Reason |
|---|---|---|---|
| E1 | **Kernels are optional in Pass 12.** Conv, STFT, mel, resize, and patch embedding are numpy reference code in `M12` and `L13`/`L14` (core). C versions are optional modules `L13.8` (conv2d via im2col, conv1d stem, patch embed), `L13.9` (resize), `L14.6` (FFT, STFT power, mel, Whisper log), each parity-tested against the learner's Python, with a real call site through the engine switch `[engine.mm] kernels = "c"` wired by optional `L15.8`. New `tinyllm.h` units are additive and always covered by stub objects (DESIGN 2.4), so a library without them loads; the engine refuses `kernels = "c"` at startup when any of them is a stub | mandatory C kernels (first sketch); no kernels at all (first scope change) | Kernels teach the memory layout and FLOP story behind conv and STFT, but the Part 9 kernel lessons already gate the course; making them optional keeps Pass 12 tractable while every optional kernel still runs in the learner's system |
| E2 | **The engine runs vision towers, projectors, and Whisper on candle** (`candle-core`, `candle-nn`), device `cpu` in tests and CI, `metal` locally on macOS. `candle-transformers` and `candle-examples` are **forbidden** by `allowed-deps.toml`. Preprocessing and the four kernel-shaped ops (resize, log-mel, patch embed, conv1d stem) go through one seam, `MmKernels` in `tl-engine/src/mm/kernels.rs`, whose core implementation is candle (plus `rustfft` and `fast_image_resize`) and whose optional implementation is the learner's C | mandatory course C kernels; `candle-transformers` models; ONNX Runtime | candle gives tensors and devices, not models: the learner still writes every layer. This amends D7 for multimodal only: the text LLM graph stays on the learner's C kernels |
| E3 | **Encoder determinism by construction**: the engine never batches tiles or frames of different media items in one encoder call; an item's embeddings depend only on its bytes, the model, and `preprocess_rev`. Whisper decoder steps are not batched across requests in v1 | cross-request encoder batching | candle CPU matmul is not batch-invariant (2.4 is a C-kernel property). One item per call keeps engine output equal to Python and makes the encoder cache sound. Cross-request batching is "Going further" |
| E4 | **One media hash everywhere**: `media_hash = fnv1a64(kind_u8 ‖ raw_bytes)` over the decoded `data:` URL payload or uploaded file bytes. The gateway (Go) and the engine (Rust) compute the same value | perceptual or pixel hashes for cache keys | Cheap, hand-implementable (FNV-1a from M06.3), identical across languages; re-encoded duplicates are a data-pipeline concern (pHash), not a serving one |
| E5 | **Prefix-cache keys via hash ids**: image placeholder tokens are hashed as virtual ids derived from `media_hash`; text-only sequences hash exactly as before | a new `tl_kv_block_hash_mm` in C; upgrading `rt.04` | No C change, no `rt.04` upgrade, the radix tree and `tl_kv_block_hash` work unchanged, and text prefix hits stay bitwise identical (spiral invariant) |
| E6 | **Media decode is a dependency, not a module**: PNG and baseline/progressive JPEG decode via `zune-png`/`zune-jpeg` inside a new crate `tl-media`, exposed to Python through `tinyllm_rs.decode_image`. Everything after decode (EXIF orientation, resize, tiling, normalize) is learner-built | learner-written JPEG decoder (core); Pillow in learner Python | One decode path for Python and Rust makes preprocessing parity a question of the learner's code only; a JPEG decoder is `sq.jpeg-decoder` |
| E7 | **Audio input is WAV only** (RIFF PCM s16le, s24le, f32le, `WAVE_FORMAT_EXTENSIBLE`, mono or stereo). The learner writes the parser in Python and Rust. Other containers return 400 `unsupported_media_type` | mp3/flac/ogg via crates or ffmpeg | WAV parsing is small and teachable; codecs add nothing to the system |
| E8 | **Preprocessing parity is split from model parity.** Model parity is tested on oracle `pixel_values` and `input_features`; preprocessing parity is tested separately (pixels within 1 uint8 LSB of PIL, log-mel within 1e-4 of HF) | one end-to-end tolerance | PIL rounds between separable passes; isolating it keeps logit tolerances tight |
| E9 | **API is additive v2.1** (`openai-subset.v2.yaml`, `info.version: 2.1.0`): `image_url` content parts with `data:` URLs, `POST /v1/audio/transcriptions`, `usage.prompt_tokens_details.image_tokens`. Engines accept only `data:` URLs and the internal `tl-media:` scheme; remote `https:` URLs are fetched only by the gateway, behind an allowlist and the SSRF guard | engine-side URL fetch; a v3 major | No migration needed; SSRF stays at one trust boundary |
| E10 | **Pass 12 modules that touch Pass 7 to 11 code do it through `upgrades`** (3.4 call-site inheritance): `forward.rs`, `openai.rs`, `template.rs`, `sched.rs`, `http.rs`, `control.rs`, `go/gateway/{server,route,ledger,limit,policy}/`, `go/loadgen`, `go/workflows/{eval_suite,model_release}.go` | new side units with no caller | The registry forbids a Pass 12 unit being "used by" an earlier-pass module; upgrades are how the spiral reaches old code |
| E11 | **Agents and audio-in LLMs stay side quests** (`sq.vision-agent`, `sq.audio-llm`, `sq.streaming-asr`) | `ag.13` core | No Pass 12 module would call them, and an earlier-pass caller violates the pass-order invariant |

### 1.3 Prerequisites

Pass 12 starts after `MS-P11`. The direct prerequisites, by module id:

| Area | Prerequisite modules | Why |
|---|---|---|
| Math | M00.2, M00.3 (rotations, frequency ladders), M03.1 to M03.5 (matmul, maps, SVD), M06.3 (PCG32, FNV-1a), M07.4 (bootstrap), M08.1 to M08.3 (VJPs), M09.1 to M09.3 (floats, stable numerics, error bounds), M11.1 (CE, KL) | conv as a matrix, DFT as a unitary matrix, InfoNCE as cross-entropy, CIs on WER |
| Autograd and nn | L0.1 to L0.6 (tensor, ops, losses, modules, checkpoints) | conv and ViT backward, freezing |
| Tokenizers | L1.2, L1.5 (BPE Python and Rust) | Whisper and SmolVLM tokenizers load through `tokenizer.json` |
| Transformer | L4.5 (BLEU, chrF, edit distance), L5.1 to L5.4 (SDPA, masks, MHA, sinusoidal PE), L6.6 (LoRA), L6.7 (zoo) | ViT and Whisper blocks, cross-attention, caption metrics, zoo rows |
| Modern block | L7.1 to L7.5, L7.9 (RMSNorm, RoPE, GQA, Llama loader, downloader) | SmolLM2 inside SmolVLM; RoPE generalizes to 2D and M-RoPE |
| Inference | L8.1, L8.2, L8.4, L8.5 (sampler, KV cache, radix prefix cache, beam from L4) | Whisper decoding, cache keys |
| Engine | L10.1 to L10.7, L10.9 (runner, scheduler, chunked prefill, block manager, server, disaggregation, metrics, templates) | every L15 module upgrades or calls them |
| Data | data.01 to data.09, ds.08 | multimodal pipelines reuse fetch, ledger, MinHash, union-find, PII scrub, `CorpusBuild` patterns |
| Durable | dur.06, dur.09, dur.11, dur.12, ag.12 | `MmCorpusBuild`, multimodal `EvalSuite` and `ModelRelease` gates |
| Gateway | gw.01 to gw.08, craft.14 (v2 routing in `go/gateway/server/`, per-version ledger) | gw.09 to gw.11 upgrade these units |
| Ops | dep.03, dep.06, obs.03, obs.04, ops.01, ops.09, load.01, load.02 | encoder Deployments, KEDA, SLOs, drill framework |
| Ethics and craft | ethics.01 to ethics.05, craft.11, craft.12, craft.17 | ledger, PII, model card, bias evals, releases, threat model |

### 1.4 Pass 12 in the spiral

Row to append to DESIGN 7.2:

| Pass | Path | Weeks at 10 to 12 h | System after the pass (all learner-built) | Gate |
|---|---|---|---|---|
| P12 | course-p12-multimodal | 15 | `v1.1.0`: signal and image math in numpy; ViT, CLIP/SigLIP, SmolVLM-256M and whisper-tiny loaded and matching HF in Python; engine on candle for encoders with an encoder cache, image-aware prefix caching, encoder budgets, EPD, and `/v1/audio/transcriptions`; gateway media limits, metering, and pool routing; multimodal corpora with pHash dedup, EXIF stripping, face redaction, consent; encode and ASR pools with their own autoscaling, SLOs, and drills; accent WER gaps in the model card | MS-P12 = MS-L13 + MS-L14 + MS-mm-corpus + MS-L15 + MS-multimodal-prod |

`MS-multimodal-prod` is the gate's production component (kind, nightly); `MS-P12` composes it with the part milestones and reruns the smoke steps of `MS-P0` to `MS-P11` (DESIGN 7.2), so every text-only check from earlier passes stays green.

Stage list to append to DESIGN 7.4 (`paths/course-p12-multimodal/path.tsv`):

| Pass | Ordered stages |
|---|---|
| P12 | lang.12 · M12.1 · M12.2 · M12.3 · M12.4 · M12.5 · M12.6 · S-M12 · L13.1 · L13.2 · L13.3 · L13.4 · L13.5 · L13.6 · L13.7 · MS-L13 · L14.1 · L14.2 · L14.3 · L14.4 · L14.5 · MS-L14 · ethics.08 · data.12 · data.10 · data.11 · data.13 · MS-mm-corpus · lang.13 · ds.10 · L15.1 · L15.2 · L15.3 · L15.4 · L15.5 · L15.6 · L15.7 · MS-L15 · gw.09 · gw.10 · gw.11 · load.03 · ethics.07 · dur.13 · dep.08 · obs.06 · drill ops.13 · drill ops.14 · craft.24 · review.04 · MS-multimodal-prod · MS-P12 (optional, skippable, reachable after their deps: L13.8 · L13.9 after L13.7 · L14.6 after L14.5 · L15.8 after L15.7 · MS-mm-kernels; side quests sq.audio-llm · sq.streaming-asr · sq.vision-agent · sq.jpeg-decoder · sq.video-frames) |

Just-in-time math row to append to DESIGN 7.5:

| Spine part | Math that must be green first |
|---|---|
| L13 | M12.1, M12.2, M12.3; S-M12 items on convolution, layouts, interpolation, contrastive losses |
| L14 | M12.4, M12.5, M12.6; S-M12 items on sampling, DFT, STFT, mel, edit distance |

### 1.5 Chapter homes

Rows to append to DESIGN 3.3:

| Layer | Chapter location |
|---|---|
| `M12` | `math/12-signals-and-images/NN-<slug>.md` (new topic: no existing topic owns sampling, DFT, or interpolation) |
| `L13` | `ml/08-tinyllm/p13-vision/NN-<slug>.md` |
| `L14` | `ml/08-tinyllm/p14-speech/NN-<slug>.md` |
| `L15` | `ml/08-tinyllm/p15-multimodal-serving/NN-<slug>.md` |
| `ds.10` | `algorithms/16-systems-data-structures/` |
| `data.10` to `data.13` | `data-engineering/05-corpus-pipeline/` (chapters 10 to 13) |
| `dur.13` | `ai-platform-engineering/05-durable-orchestration-and-workers/` |
| `gw.09` to `gw.11` | `ai-platform-engineering/12-gateway/` |
| `load.03` | `ml/08-tinyllm/p10-serving/` |
| `dep.08`, `obs.06`, `ops.13`, `ops.14` | `infrastructure/01-containers-kubernetes/`, `systems/04-observability/`, `systems/05-incident-response-and-chaos/` |
| `ethics.07`, `ethics.08` | `responsible-ai/04-bias-and-safety-evals/`, `responsible-ai/02-privacy-and-pii/` |
| `craft.24`, `review.04` | `software-craftsmanship/11-security/`, `systems/01-system-design/` |
| `lang.12`, `lang.13` | `software-craftsmanship/12-language-and-tool-primers/{12-media-formats,13-candle-tensors}.md` |
| Guide | `paths/course-p12-multimodal/` (README, `path.tsv`, `milestone.md`); `paths/course/path.tsv` gains `@course-p12-multimodal` |

Existing reading becomes depth, not duplicated content (DESIGN 8.2 style):

| Existing file | Fate in B14 |
|---|---|
| `ml/05-foundation-models/README.md` section 4 (modalities) | stays as the topic overview; cited by the `p13`, `p14` part READMEs as "why now"; its fusion-pattern bullets point to L13.5 (dual encoder) and L13.7 (projector into the LLM) |
| `ml/06-neural-architectures/code/conv2d.c` | worked example for M12.1 beat 3 (valid cross-correlation, output size formula) and the naive baseline that the optional L13.8 im2col kernel is benchmarked against |
| `ml/06-neural-architectures/code/conv2d.cu` | stays in `sq.cnns`; `sq.cuda-kernels` gains a CUDA variant of the optional L13.8 conv |
| `ml/06-neural-architectures/code/lenet.rs`, `alexnet.py` | `sq.cnns` reading; L13.1 beat 6 "Going further" cites them; LeNet becomes the L13.1 learning test architecture in numpy |

---

## 2. New and changed contracts

All paths are relative to `course/contracts/`. Every change is additive (minor) unless marked. `contracts/VERSION` moves to the next minor; learners pick it up with `ss contracts sync`.

### 2.1 Components (rows to amend in DESIGN 2.1)

| # | Component | Lang | Learner path | Pass 12 responsibilities | Built in |
|---|---|---|---|---|---|
| 1 | `tinyllm` | Python | `python/tinyllm/` | `sig/` and `image/` (M12), `vision/` (L13), `speech/` (L14), `io/{wav,hf_vlm,hf_whisper}.py`, `eval/{mm,mm_bias}.py` | M12, L13, L14, ethics.07 |
| 2 | `corpus` | Python | `python/corpus/` | `image_pairs.py`, `audio_pairs.py`, `faces.py`, `png.py` | data.10 to data.12 |
| 3 | `libtinyllm` | C | `c/src/kernels/{conv,image,audio}.c` | optional multimodal kernels behind `tinyllm/{conv,image,audio}.h` | L13.8, L13.9, L14.6 (optional) |
| 18 | `tl-media` (new crate) | Rust | `rust/crates/tl-media/` | image decode (zune), EXIF parse and orientation, WAV parse; used by `tl-engine` and `tl-py` | L13.6, L15.1 |
| 4 | `tl-ds` | Rust | `tl-ds/src/lru_bytes.rs` | byte-budgeted LRU with pins | ds.10 |
| 6 | `tl-py` | Rust | `tl-py/src/media.rs` | `tinyllm_rs.decode_image`, `tinyllm_rs.read_exif` | L13.6 |
| 8 | `tl-engine` | Rust | `tl-engine/src/mm/*.rs`, `tl-engine/src/asr/*.rs` | candle device, media preprocessing, vision towers and connectors, merge, encoder cache, hash ids, encoder budget, EPD, Whisper runner and decoding | L15.1 to L15.6 |
| 9 | `tl-serve` | Rust | `tl-serve/src/{audio,encoder_service}.rs` | content parts, transcription endpoint (multipart, srt, vtt), `tl.encoder.v1` services | L15.3, L15.5, L15.7 |
| 10 | gateway | Go | `go/gateway/media/` | media parsing, limits, remote fetch with SSRF guard, metering, VLM/ASR/encode pools, EPD orchestration | gw.09 to gw.11 |
| 12 | durable workflows | Go | `go/workflows/mm_corpus_build.go` | `MmCorpusBuild`; multimodal `EvalSuite` and `ModelRelease` gates | data.13, dur.13 |
| 14 | loadgen | Go | `go/loadgen/` | multimodal workload mix, multipart sender, RTF report | load.03 |

### 2.2 C ABI

Additive, optional units (E1). `TL_ABI_VERSION` stays 1: new symbols are a minor addition, and the harness's stub object covers each unit until its module is started (every function returns `TL_EUNSUPPORTED` with `tl_last_error = "unimplemented: <ID>"`), so a library without them still loads in ctypes and links in `tl-sys`. The `abi/` suite gains layout checks for `tl_conv2d_shape`, `tl_resize_cfg`, and `tl_stft_cfg`, and `nm` checks for the new `tl_` symbols. All functions follow DESIGN 2.4: caller-owned buffers, `int64_t` dims, row-major, scratch from `tl_arena`, threads from `tl_pool`, reentrant.

```c
/* tinyllm.h gains (after elementwise.h): */
#include "tinyllm/conv.h"        /* L13.8, optional */
#include "tinyllm/image.h"       /* L13.9, optional */
#include "tinyllm/audio.h"       /* L14.6, optional */

/* tinyllm/conv.h  (L13.8): conv as im2col + tl_matmul_f32, so it inherits L9.1 batch invariance per image */
typedef struct { int64_t N, C, H, W;            /* input, NCHW */
                 int64_t K, R, S;               /* output channels, kernel height, width (weights KCRS, C/groups per group) */
                 int64_t stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups; } tl_conv2d_shape;
int64_t   tl_conv_out_size(int64_t in, int64_t k, int64_t stride, int64_t pad, int64_t dil);   /* floor((in + 2p - d(k-1) - 1)/s) + 1; <= 0 invalid */
size_t    tl_im2col_bytes(const tl_conv2d_shape *s);                                          /* (C/groups * R * S) * (P * Q) * sizeof(float) */
tl_status tl_im2col_f32(const float *x, const tl_conv2d_shape *s, int64_t n, int64_t group,
                        float *cols);                   /* rows ordered (c, r, s), columns (p, q); zero padding */
tl_status tl_conv2d_f32(const float *x, const float *w, const float *bias /* nullable [K] */, float *y,
                        const tl_conv2d_shape *s, tl_arena *scratch, tl_pool *tp);            /* y NKPQ */
tl_status tl_conv1d_f32(const float *x, const float *w, const float *bias /* nullable */, float *y,
                        int64_t N, int64_t C, int64_t L, int64_t K, int64_t R,
                        int64_t stride, int64_t pad, int64_t dil,
                        tl_arena *scratch, tl_pool *tp);                                      /* x NCL, w KCR, y NKL'; Whisper stem: R 3, pad 1, stride 1 then 2 */
tl_status tl_patch_embed_f32(const float *img, const float *w, const float *bias, float *out,
                             int64_t N, int64_t C, int64_t H, int64_t W, int64_t P, int64_t D,
                             int layout /* 0 NCHW, 1 NHWC */, tl_pool *tp);                   /* out [N, (H/P)*(W/P), D]; H or W not divisible by P: TL_ESHAPE */

/* tinyllm/image.h  (L13.9) */
enum { TL_RESAMPLE_NEAREST = 0, TL_RESAMPLE_BILINEAR = 1, TL_RESAMPLE_BICUBIC = 2, TL_RESAMPLE_LANCZOS3 = 3 };
typedef struct { int32_t kernel;      /* TL_RESAMPLE_* (int32_t, never a C enum across the ABI) */
                 int32_t antialias;   /* 1: support scaled by max(1, in/out) */
                 int32_t pil_round;   /* 1: round half up and clip to u8 between the horizontal and vertical passes */
                 float   cubic_a;     /* -0.5 (PIL) */ } tl_resize_cfg;
tl_status tl_resize_u8(const uint8_t *src, int64_t h, int64_t w, int64_t c, uint8_t *dst, int64_t oh, int64_t ow,
                       const tl_resize_cfg *cfg, tl_arena *scratch, tl_pool *tp);            /* HWC interleaved, formats/image-preprocess.md */
tl_status tl_resize_f32(const float *src, int64_t c, int64_t h, int64_t w, float *dst, int64_t oh, int64_t ow,
                        const tl_resize_cfg *cfg, tl_arena *scratch, tl_pool *tp);           /* CHW planar */
tl_status tl_rescale_normalize_u8(const uint8_t *src_hwc, int64_t h, int64_t w, int64_t c, float scale,
                                  const float *mean, const float *std, float *dst_chw);

/* tinyllm/audio.h  (L14.6) */
typedef struct tl_fft_plan tl_fft_plan;
tl_status tl_fft_plan_create(int64_t n, tl_fft_plan **out);     /* n factors over {2, 3, 4, 5}; else TL_EUNSUPPORTED; allocates via the hook */
void      tl_fft_plan_destroy(tl_fft_plan *p);
tl_status tl_rfft_f32(const tl_fft_plan *p, const float *x, float *re, float *im);   /* n real in, n/2 + 1 complex out */
typedef struct { int64_t n_fft, hop; int32_t center; int32_t pad_mode; /* 0 reflect, 1 constant */
                 int32_t drop_last; } tl_stft_cfg;
int64_t   tl_stft_frames(int64_t n_samples, const tl_stft_cfg *c);
tl_status tl_stft_power_f32(const float *x, int64_t n_samples, const float *window /* [n_fft] */,
                            const tl_stft_cfg *c, const tl_fft_plan *p,
                            float *power /* [n_fft/2 + 1, frames] */, tl_arena *scratch, tl_pool *tp);
tl_status tl_mel_f32(const float *filters /* [n_mels, n_freq] */, const float *power, float *mel,
                     int64_t n_mels, int64_t n_freq, int64_t frames, tl_pool *tp);    /* tl_matmul_f32 underneath */
tl_status tl_log_mel_whisper_f32(float *mel, int64_t n);                             /* in place: log10(max(x, 1e-10)); max(x, max - 8); (x + 4) / 4 */
```

The filterbank and window are inputs, built by the caller from the M12.6 and M12.5 formulas (the engine builds them in Rust from the same spec), so the kernel owns only the compute. C uses libm `log10f`; no other libm call is needed.

### 2.3 Python contracts (`contracts/py/tinyllm/**.pyi`)

`NDArray` is `numpy.ndarray`; `Tensor` is the L0.1 autograd tensor. Images are `uint8[H, W, C]` (HWC, RGB) until normalized; `pixel_values` are `float32[N, C, H, W]`.

```python
# tinyllm/sig/conv.pyi  (M12.1)
def conv_out_size(n: int, k: int, stride: int = 1, pad: int = 0, dilation: int = 1) -> int   # floor((n + 2p - d(k-1) - 1)/s) + 1
def conv1d_direct(x: NDArray, w: NDArray, b: NDArray | None, stride=1, pad=0, dilation=1, groups=1) -> NDArray  # x [N,C,L], w [K,C/g,R]
def conv2d_direct(x: NDArray, w: NDArray, b: NDArray | None, stride=(1,1), pad=(0,0), dilation=(1,1), groups=1) -> NDArray  # NCHW, KCRS
def correlate_vs_convolve(x: NDArray, w: NDArray) -> tuple[NDArray, NDArray]   # cross-correlation and true convolution (flipped kernel)
def im2col(x: NDArray, kh: int, kw: int, stride=(1,1), pad=(0,0), dilation=(1,1)) -> NDArray    # [N, C*kh*kw, P*Q], rows ordered (c, r, s)
def col2im(cols: NDArray, x_shape: tuple[int,int,int,int], kh: int, kw: int, stride=(1,1), pad=(0,0), dilation=(1,1)) -> NDArray  # adjoint of im2col (sums overlaps)
def conv_matrix(in_shape: tuple[int,int], w: NDArray, stride=(1,1), pad=(0,0)) -> NDArray        # doubly block Toeplitz T with vec(y) = T vec(x)

# tinyllm/image/layout.pyi  (M12.2)
Layout = Literal['NCHW', 'NHWC']
def to_layout(x: NDArray, src: Layout, dst: Layout) -> NDArray            # returns a contiguous copy
def strides_of(shape: tuple[int, ...], layout: Layout, itemsize: int) -> tuple[int, ...]
def rescale_normalize(img_u8_hwc: NDArray, scale: float, mean: Sequence[float], std: Sequence[float]) -> NDArray  # -> f32 CHW
def to_grayscale_pil(img_u8_hwc: NDArray) -> NDArray                      # L = (R*299 + G*587 + B*114 + 500) // 1000, PIL "L" rule
def patchify(x_chw: NDArray, p: int) -> NDArray                           # [C,H,W] -> [(H/p)*(W/p), C*p*p], row-major patches, (c, i, j) inner order
def unpatchify(patches: NDArray, c: int, h: int, w: int, p: int) -> NDArray

# tinyllm/sig/resample.pyi  (M12.3)
Kernel = Literal['nearest', 'bilinear', 'bicubic', 'lanczos3']
def kernel_weights(in_size: int, out_size: int, kernel: Kernel, antialias: bool = True, cubic_a: float = -0.5) -> tuple[NDArray, NDArray]  # start idx [out], weights [out, taps], rows sum to 1
def resize(img: NDArray, size: tuple[int, int], kernel: Kernel, antialias: bool = True,
           mode: Literal['pil', 'float'] = 'pil', cubic_a: float = -0.5) -> NDArray   # 'pil': u8 HWC, horizontal pass then vertical, round+clip between passes
def resize_planar_f32(x_chw: NDArray, size: tuple[int, int], kernel: Kernel, align_corners: bool = False, antialias: bool = False) -> NDArray  # position-table interpolation
def resample_audio(x: NDArray, sr_in: int, sr_out: int, lowpass_filter_width: int = 6, rolloff: float = 0.99) -> NDArray  # torchaudio sinc_interp_hann semantics

# tinyllm/sig/fft.pyi  (M12.4)
def dft_matrix(n: int, unitary: bool = False) -> NDArray                  # complex128 [n, n]
def fft(x: NDArray) -> NDArray                                            # mixed radix over factors {2, 3, 4, 5}; ValueError otherwise
def ifft(X: NDArray) -> NDArray
def rfft(x: NDArray) -> NDArray                                           # n real -> n//2 + 1 complex, via one complex FFT of size n/2 (n even)
def irfft(X: NDArray, n: int) -> NDArray
def fft_convolve(a: NDArray, b: NDArray) -> NDArray                       # linear convolution via zero-padded FFT
def dct2_ortho(x: NDArray, axis: int = -1) -> NDArray                     # DCT-II, norm='ortho' (scipy semantics), via FFT; used by pHash

# tinyllm/sig/stft.pyi  (M12.5)
def hann(n: int, periodic: bool = True) -> NDArray
def frame_count(n_samples: int, n_fft: int, hop: int, center: bool = True) -> int     # center: 1 + n // hop
def stft(x: NDArray, n_fft: int, hop: int, window: NDArray, center: bool = True,
         pad_mode: Literal['reflect', 'constant'] = 'reflect') -> NDArray # complex [n_fft//2 + 1, frames]
def istft(S: NDArray, n_fft: int, hop: int, window: NDArray, length: int) -> NDArray   # WOLA; exact under COLA
def power(S: NDArray) -> NDArray                                          # |S|^2

# tinyllm/sig/mel.pyi  (M12.6)
def hz_to_mel(f: NDArray, scale: Literal['slaney', 'htk'] = 'slaney') -> NDArray
def mel_to_hz(m: NDArray, scale: Literal['slaney', 'htk'] = 'slaney') -> NDArray
def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None,
                   scale: Literal['slaney', 'htk'] = 'slaney', norm: Literal['slaney', None] = 'slaney') -> NDArray   # [n_mels, n_fft//2 + 1]
def power_to_db(p: NDArray, ref: float = 1.0, amin: float = 1e-10, top_db: float | None = 80.0) -> NDArray
def log_mel_whisper(power_spec: NDArray, filters: NDArray) -> NDArray     # log10(max(mel, 1e-10)); max(x, x.max() - 8); (x + 4) / 4
```

```python
# tinyllm/vision/conv.pyi  (L13.1)
class Conv2d(Module):
    def __init__(self, c_in: int, c_out: int, k: int | tuple[int,int], stride=1, pad=0, dilation=1, groups=1, bias=True, rng: PCG32 | None = None)
    def forward(self, x: Tensor) -> Tensor                                # NCHW; im2col + matmul; backward via col2im
class Conv1d(Module):
    def __init__(self, c_in: int, c_out: int, k: int, stride=1, pad=0, dilation=1, groups=1, bias=True, rng: PCG32 | None = None)
def max_pool2d(x: Tensor, k: int, stride: int | None = None) -> Tensor   # argmax routing in backward, ties to the lowest flat index
def avg_pool2d(x: Tensor, k: int, stride: int | None = None) -> Tensor
class LeNet5(Module): def __init__(self, n_classes: int = 10, in_hw: int = 8, rng: PCG32 | None = None)

# tinyllm/vision/rope2d.pyi  (L13.2)
def rope2d_cos_sin(grid_hw: tuple[int, int], d_rot: int, base: float = 10000.0) -> tuple[NDArray, NDArray]   # half the rotary dims for rows, half for cols
def mrope_position_ids(input_ids: NDArray, image_grid_thw: NDArray, image_token_id: int,
                       spatial_merge: int = 2) -> NDArray                  # [3, T]: (t, h, w); text continues at max(prev) + 1
def apply_mrope(x: Tensor, pos3: NDArray, inv_freq: NDArray, mrope_section: Sequence[int]) -> Tensor   # sections of the rotary half take t, h, w positions

# tinyllm/vision/vit.pyi  (L13.3)
@dataclass
class ViTConfig: image_size: int; patch_size: int; n_channels: int; d: int; n_layers: int; n_heads: int; d_ff: int
                 pos: Literal['learned', 'sincos2d', 'rope2d']; cls_token: bool; pre_norm: bool; norm: Literal['layernorm']
                 act: Literal['gelu', 'gelu_tanh', 'quick_gelu']; ln_eps: float; post_layernorm: bool
class PatchEmbed(Module): def __init__(self, cfg: ViTConfig)               # Conv2d(k=p, stride=p) == patchify + Linear
class ViT(Module):
    def __init__(self, cfg: ViTConfig)
    def forward(self, pixel_values: Tensor, patch_mask: NDArray | None = None,
                output_hidden_states: bool = False) -> tuple[Tensor, list[Tensor]]    # last hidden [N, T, d], all layers
def interpolate_pos_embed(table: NDArray, old_grid: tuple[int,int], new_grid: tuple[int,int]) -> NDArray   # bicubic via M12.3, CLS row kept
def bucketed_position_ids(patch_mask: NDArray, n_side: int) -> NDArray      # NaViT-style fractional-coordinate buckets (Idefics3)

# tinyllm/vision/tiling.pyi  (L13.4)
def select_best_resolution(hw: tuple[int,int], pinpoints: Sequence[tuple[int,int]]) -> tuple[int,int]   # LLaVA-NeXT: max effective res, then min waste
def anyres_tiles(img: NDArray, pinpoints, tile: int) -> tuple[list[NDArray], NDArray, tuple[int,int]]    # tiles, base image, grid
def idefics3_split(img: NDArray, longest_edge: int, tile: int) -> tuple[list[NDArray], NDArray, tuple[int,int]]   # tiles, global image, (rows, cols)
def pixel_shuffle(x: NDArray, r: int) -> NDArray                         # [N, S*S, D] -> [N, (S/r)^2, D*r*r]
def pixel_unshuffle(y: NDArray, r: int) -> NDArray
def image_token_count(h: int, w: int, proc: "ProcessorConfig", detail: Literal['low','high','auto'] = 'auto') -> int  # placeholders, incl. wrappers

# tinyllm/vision/contrastive.pyi  (L13.5)
def clip_loss(img_emb: Tensor, txt_emb: Tensor, logit_scale: Tensor) -> Tensor        # symmetric InfoNCE; scale = exp(t) clamped at 100
def siglip_loss(img_emb: Tensor, txt_emb: Tensor, t: Tensor, b: Tensor) -> Tensor     # -mean over all pairs of log sigmoid(z_ij * (t*cos_ij + b)); z = +1 diag, -1 off
class DualEncoder(Module):
    def __init__(self, vision: ViT, text: Module, d_embed: int, loss: Literal['clip', 'siglip'], t_init: float, b_init: float = -10.0)
    def encode_image(self, pixel_values: Tensor) -> Tensor; def encode_text(self, ids: NDArray) -> Tensor   # L2-normalized
def zero_shot_classifier(model: DualEncoder, tok, class_names: Sequence[str], templates: Sequence[str]) -> NDArray   # mean of normalized template embeddings
def zero_shot_predict(model: DualEncoder, pixel_values: Tensor, W: NDArray) -> NDArray

# tinyllm/vision/preprocess.pyi  (L13.6); formats/image-preprocess.md
@dataclass
class ProcessorConfig: do_resize: bool; size: dict; resample: Kernel; do_rescale: bool; rescale_factor: float
                       do_normalize: bool; image_mean: tuple[float,...]; image_std: tuple[float,...]
                       do_image_splitting: bool; max_image_size: dict; tile_size: int; scale_factor: int; image_seq_len: int
    @classmethod
    def from_hf(cls, preprocessor_config_json: str) -> "ProcessorConfig"
@dataclass
class ImageInputs: pixel_values: NDArray; pixel_mask: NDArray; grid: tuple[int,int]; n_tokens: int; media_hash: int
def decode_image(data: bytes) -> NDArray                                  # tinyllm_rs.decode_image + EXIF orientation applied
def preprocess(images: Sequence[bytes], cfg: ProcessorConfig, detail: Literal['low','high','auto'] = 'auto') -> list[ImageInputs]
def media_hash(kind: Literal['image', 'audio'], data: bytes) -> int        # fnv1a64(kind_u8 || data), kind 1 image, 2 audio (E4)

# tinyllm/vision/vlm.pyi  (L13.7)
@dataclass
class VLMConfig: arch: Literal['idefics3', 'llava', 'llava_next', 'qwen2vl']; vision: ViTConfig; text: LlamaConfig
                 projector: Literal['linear', 'mlp2x_gelu']; scale_factor: int; image_token_id: int
                 vision_feature_layer: int; select: Literal['default', 'full']
class VisionLanguageModel(Module):
    def __init__(self, cfg: VLMConfig)
    def encode_images(self, inputs: Sequence[ImageInputs]) -> list[Tensor]          # projected features, one [n_tokens_i, d_text] per image
    def merge(self, ids: NDArray, feats: Sequence[Tensor]) -> Tensor                  # scatter rows at image_token_id positions in order
    def forward(self, ids: NDArray, images: Sequence[ImageInputs], positions: NDArray | None = None, cache=None) -> Tensor
    @classmethod
    def from_pretrained(cls, dir: str) -> "VisionLanguageModel"                     # HF key map in io/hf_vlm.py
def render_mm_chat(messages: Sequence[dict], template: str, proc: ProcessorConfig, images: Sequence[ImageInputs]) -> tuple[list[int], NDArray]  # ids, assistant mask
@dataclass
class FreezeStage: name: str; trainable: Sequence[str]; steps: int; lr: float     # glob patterns over parameter names
def apply_stage(model: Module, stage: FreezeStage) -> list[str]                    # returns trainable names; others requires_grad=False, no optimizer state
```

```python
# tinyllm/io/wav.pyi  (L14.1)
@dataclass
class Wav: samples: NDArray; sr: int; channels: int; fmt: Literal['s16', 's24', 'f32']
def read_wav(data: bytes) -> Wav                                          # RIFF/WAVE, fmt and data chunks, WAVE_FORMAT_EXTENSIBLE; ValueError otherwise
def write_wav(samples: NDArray, sr: int) -> bytes                         # s16le mono

# tinyllm/speech/frontend.pyi  (L14.1); formats/mel-spec.md
@dataclass
class FeatureConfig: sampling_rate: int = 16000; n_fft: int = 400; hop_length: int = 160; feature_size: int = 80
                     chunk_length: int = 30; padding_value: float = 0.0
    @classmethod
    def from_hf(cls, preprocessor_config_json: str) -> "FeatureConfig"
def to_mono_16k(w: Wav, cfg: FeatureConfig) -> NDArray                     # mean over channels, then resample_audio
def pad_or_trim(x: NDArray, n_samples: int = 480000) -> NDArray
def whisper_log_mel(x: NDArray, cfg: FeatureConfig) -> NDArray             # f32 [n_mels, 3000]: hann(400, periodic), center reflect, drop last frame

# tinyllm/speech/encoder.pyi  (L14.2)
@dataclass
class WhisperConfig: n_mels: int; n_audio_ctx: int; d: int; n_heads: int; enc_layers: int; dec_layers: int
                     n_text_ctx: int; vocab: int
class WhisperEncoder(Module):
    def __init__(self, cfg: WhisperConfig)                                 # conv(k3,p1) GELU, conv(k3,s2,p1) GELU, + sinusoids, pre-LN blocks, ln_post
    def forward(self, mel: Tensor) -> Tensor                               # [N, 80, 3000] -> [N, 1500, d]
def whisper_sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> NDArray   # concat(sin, cos), not interleaved

# tinyllm/speech/decoder.pyi  (L14.3)
class CrossKV: k: list[NDArray]; v: list[NDArray]                          # per layer, computed once per audio window
class WhisperDecoder(Module):
    def __init__(self, cfg: WhisperConfig)
    def cross_kv(self, audio: Tensor) -> CrossKV
    def forward(self, ids: NDArray, xkv: CrossKV, cache: "KVCache | None" = None) -> Tensor   # logits via tied embedding
class Whisper(Module):
    @classmethod
    def from_pretrained(cls, dir: str) -> "Whisper"                        # HF key map in io/hf_whisper.py

# tinyllm/speech/decode.pyi  (L14.4); spec/asr-decoding.md
@dataclass
class DecodeOptions: task: Literal['transcribe', 'translate'] = 'transcribe'; language: str | None = None
                     timestamps: bool = True; beam_size: int = 0; best_of: int = 5; patience: float = 1.0
                     temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
                     compression_ratio_threshold: float = 2.4; logprob_threshold: float = -1.0; no_speech_threshold: float = 0.6
                     condition_on_previous_text: bool = True; max_initial_timestamp: float = 1.0; seed: int = 0
class LogitsSource(Protocol):                                             # the model, or recorded logits in parity tests
    def step(self, prefix: Sequence[int]) -> NDArray
@dataclass
class Segment: id: int; seek: int; start: float; end: float; text: str; tokens: list[int]
               temperature: float; avg_logprob: float; compression_ratio: float; no_speech_prob: float
def sot_sequence(tok, language: str | None, task: str, timestamps: bool) -> list[int]
def detect_language(model, mel_window: NDArray, tok) -> tuple[str, dict[str, float]]
def apply_timestamp_rules(logits: NDArray, prefix: Sequence[int], sample_begin: int, tok, max_initial_index: int) -> NDArray
def decode_window(src: LogitsSource, tok, prompt: Sequence[int], opts: DecodeOptions, temperature: float, rng: PCG32) -> Segment-like
def transcribe(model: Whisper, tok, audio_16k: NDArray, opts: DecodeOptions) -> list[Segment]   # long-form seek loop with fallback

# tinyllm/eval/mm.pyi  (L14.5)
def basic_normalize(text: str) -> str                                    # spec/asr-decoding.md normalizer (lowercase, bracket and symbol removal, whitespace)
def edit_ops(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int, int]   # S, D, I, N with lowest-cost alignment, ties S < D < I
def wer(refs: Sequence[str], hyps: Sequence[str]) -> float                # corpus-level sum(S+D+I)/sum(N)
def cer(refs: Sequence[str], hyps: Sequence[str]) -> float
def vqa_accuracy(preds: Sequence[str], answers: Sequence[Sequence[str]]) -> float   # min(1, matches/3) over normalized answers
def caption_scores(preds: Sequence[str], refs: Sequence[Sequence[str]]) -> dict[str, float]   # BLEU-4, chrF (L4.5), attribute recall
def zero_shot_accuracy(pred: NDArray, gold: NDArray, k: int = 1) -> float
def bootstrap_ci(per_item: NDArray, stat: Callable, n: int = 1000, seed: int = 0) -> tuple[float, float, float]
```

```python
# contracts/py/tinyllm_rs.pyi additions (L13.6)
def decode_image(data: bytes) -> tuple[int, int, int, bytes]     # (h, w, c, u8 HWC RGB); EXIF orientation NOT applied; raises ValueError
def read_exif(data: bytes) -> dict[str, object]                  # orientation, has_gps, tags present; raw APP1 parse in tl-media
def image_header(data: bytes) -> tuple[str, int, int]            # format, width, height without full decode
```

### 2.4 Rust contracts (`rust/tl-contracts`)

```rust
// tl-media (L13.6, L15.1)
pub enum ImageFormat { Png, Jpeg }
pub struct DecodedImage { pub h: u32, pub w: u32, pub c: u8, pub data: Vec<u8> }        // HWC RGB u8
pub fn image_header(bytes: &[u8]) -> Result<(ImageFormat, u32, u32), MediaError>;       // no pixel decode
pub fn decode_image(bytes: &[u8], max_pixels: u64) -> Result<DecodedImage, MediaError>; // checks header before decode
pub fn exif_orientation(bytes: &[u8]) -> Result<Option<u8>, MediaError>;                // 1..=8
pub struct Wav { pub samples: Vec<f32>, pub sr: u32, pub channels: u16 }
pub fn read_wav(bytes: &[u8], max_seconds: f32) -> Result<Wav, MediaError>;

// tl-engine/src/mm/media.rs (L15.1)
pub struct MediaItem { pub kind: MediaKind /* Image | Audio */, pub bytes: Arc<[u8]>, pub hash: u64, pub detail: Detail }
pub fn media_hash(kind: MediaKind, bytes: &[u8]) -> u64;                                 // E4
pub struct ImagePrep { pub pixel_values: Vec<f32>, pub shape: [usize; 4], pub pixel_mask: Vec<u8>, pub grid: (u32, u32), pub n_tokens: u32 }
pub trait Preprocessor: Send + Sync {
    fn rev(&self) -> u32;                                                                // preprocess_rev, part of every cache key
    fn image(&self, item: &MediaItem) -> Result<ImagePrep, MediaError>;
    fn audio(&self, item: &MediaItem) -> Result<Vec<Vec<f32>>, MediaError>;            // one [n_mels * 3000] log-mel per 30 s window
    fn image_token_count(&self, h: u32, w: u32, detail: Detail) -> u32;                 // must equal the Python and Go counts
}

// tl-engine/src/mm/kernels.rs (L15.1 core: candle impl; optional L15.8 upgrades it to add the C impl)
pub enum KernelBackend { Candle, C }                                                     // [engine.mm].kernels
pub trait MmKernels: Send + Sync {
    fn resize_u8(&self, src: &[u8], hwc: (usize, usize, usize), out_hw: (usize, usize), cfg: &ResizeCfg) -> Result<Vec<u8>, KernelError>;
    fn log_mel(&self, audio_16k: &[f32], mel: &MelSpec) -> Result<Vec<f32>, KernelError>;          // [n_mels * 3000] per window
    fn patch_embed(&self, pix: &[f32], shape: [usize; 4], w: &[f32], b: &[f32], p: usize, d: usize) -> Result<Vec<f32>, KernelError>;
    fn conv1d(&self, x: &[f32], shape: [usize; 3], w: &[f32], b: &[f32], k: usize, stride: usize, pad: usize) -> Result<Vec<f32>, KernelError>;
}
pub fn kernels(backend: KernelBackend, tp: Option<&tl_sys::Pool>) -> Result<Box<dyn MmKernels>, KernelError>;   // C: probes each unit, stub => Err at startup
// The candle towers call patch_embed and conv1d through this seam on the host (cpu) path; on metal the backend must be Candle.

// tl-engine/src/mm/encoder_cache.rs, hash_ids.rs (L15.2), over tl-ds::LruBytes (ds.10)
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct EncoderKey { pub model: Arc<str>, pub preprocess_rev: u32, pub media_hash: u64, pub detail: Detail }
pub struct EncoderEntry { pub n_tokens: u32, pub d: u32, pub data: Arc<[f32]> }       // projected features, row-major
impl EncoderCache {
    pub fn new(max_bytes: usize) -> Self;
    pub fn get_pinned(&mut self, k: &EncoderKey) -> Option<(Arc<EncoderEntry>, PinGuard)>;  // pinned while a request references it
    pub fn insert(&mut self, k: EncoderKey, e: EncoderEntry) -> Result<(), CacheFull>;      // evicts unpinned LRU; never a pinned entry
    pub fn stats(&self) -> EncoderCacheStats;                                             // bytes, entries, hits, misses, evictions
}
pub fn hash_ids(model_ids: &[u32], spans: &[MmSpan]) -> Vec<u32>;                        // formats/kv-block.md "Multimodal hash ids"

// tl-engine/src/mm/vision.rs, merge.rs (L15.3), on candle (E2)
pub struct CandleCtx { pub device: candle_core::Device, pub dtype: candle_core::DType }  // cpu | metal; f32 in tests
pub trait VisionEncoder: Send {
    fn load(dir: &Path, ctx: &CandleCtx) -> anyhow::Result<Self> where Self: Sized;      // VarBuilder::from_mmaped_safetensors
    fn encode(&mut self, prep: &ImagePrep) -> anyhow::Result<EncoderEntry>;              // tower + pixel shuffle + connector, one item per call (E3)
}
pub struct MmSpan { pub start: u32, pub len: u32, pub media_hash: u64 }                  // placeholder run in the prompt
pub struct ForwardBatch { /* existing fields */ pub mm: Vec<(SeqIdx, MmSpan, Arc<EncoderEntry>)> }   // upgrade of forward.rs: rows replace embeddings after lookup

// tl-engine/src/sched.rs (L15.4, upgrades L10.2)
pub struct SchedulerConfig { /* existing */ pub max_encoder_tokens_per_step: u32, pub encoder_cache_bytes: usize }
pub struct ScheduleOutput { /* existing */ pub encode: Vec<(RequestId, EncoderKey)> }   // encoder work admitted this step, within budget

// tl-engine/src/mm/epd.rs (L15.5)
#[async_trait] pub trait EmbeddingTransport: Send + Sync {
    async fn has(&self, target: &str, keys: &[EncoderKey]) -> Result<Vec<bool>, TransferError>;
    async fn push(&self, target: &str, key: &EncoderKey, e: &EncoderEntry) -> Result<EmbedAck, TransferError>;
}

// tl-engine/src/asr/{whisper,decode}.rs (L15.6), on candle
pub struct WhisperRunner { /* candle encoder + decoder, contiguous self-KV and cross-KV per request */ }
impl WhisperRunner {
    pub fn load(dir: &Path, ctx: &CandleCtx) -> anyhow::Result<Self>;
    pub fn encode(&mut self, mel: &[f32]) -> anyhow::Result<AudioCtx>;                  // [1500, d] plus cross-KV
    pub fn step(&mut self, a: &AudioCtx, prefix: &[u32], kv: &mut SelfKv) -> anyhow::Result<Vec<f32>>;  // logits
}
pub trait LogitsSource { fn step(&mut self, prefix: &[u32]) -> Vec<f32>; }               // same seam as Python, for parity on recorded logits
pub fn transcribe(src: &mut dyn LogitsSource, tok: &dyn Tokenizer, windows: &[Vec<f32>], opts: &DecodeOptions, rng: &mut Pcg32) -> Vec<Segment>;
```

### 2.5 Go contracts (`contracts/go`)

```go
// go/gateway/media (gw.09)
type Kind uint8 // KindImage = 1, KindAudio = 2
type Part struct { Kind Kind; Bytes []byte; Hash uint64; Format string; W, H int; Seconds float64; Detail string }
type Limits struct { MaxRequestBytes, MaxImages, MaxImagePixels, MaxAudioBytes int64; MaxAudioSeconds float64 }
func Hash(k Kind, b []byte) uint64                               // E4, parity with Rust and Python
func ParseChat(body []byte, l Limits, f Fetcher) (*ChatMedia, error)   // decodes data: URLs, header-only dimension check via image.DecodeConfig
func ParseTranscription(r *http.Request, l Limits) (*AudioUpload, error) // multipart under http.MaxBytesReader; WAV header duration check
type Fetcher interface { Fetch(ctx context.Context, url string) ([]byte, error) }       // allowlist + SSRF guard; disabled by default
func ImageTokens(w, h int, p ProcessorConfig, detail string) int  // Go port of L13.4 image_token_count (re-implemented, parity fixture)

// go/gateway/limit (gw.10 upgrades gw.03)
type Cost struct { Requests, Tokens int; AudioMs int64 }        // AudioMs: new bucket "audio seconds per minute"

// go/gateway/route (gw.11 upgrades gw.05)
type Pool string // "text" | "vlm" | "asr" | "encode"
type EpdPlan struct { Encoders []Target; Prefill, Decode Target; Media []media.Part }
```

### 2.6 File formats (`formats/`)

| Artifact | Format |
|---|---|
| `formats/image-preprocess.md` (new) | The preprocessing spec every implementation follows: decode (E6), EXIF orientation 1 to 8 applied before anything else, RGBA composited on white, grayscale and palette expanded to RGB; resize kernels `nearest`, `bilinear`, `bicubic` (a = -0.5, the PIL value; torch uses -0.75), `lanczos3`; pixel-center convention `src = (dst + 0.5) * scale - 0.5`; antialias support `support * max(1, in/out)`; weights normalized per output pixel; PIL mode: horizontal pass, round half up and clip to u8, vertical pass, round and clip; then `x * rescale_factor`, then `(x - mean) / std` in f32, CHW. Tiling: LLaVA-NeXT AnyRes and Idefics3 splitting as in 2.3; **token count**: `image_token_count` per processor is normative and fixture-tested in Python, Rust, and Go. `preprocess_rev` (u32) increments on any change and is part of every encoder-cache key |
| `formats/preprocessor-config.schema.json` (new) | subset of HF `preprocessor_config.json` / `processor_config.json` that loads unchanged for SmolVLM-256M-Instruct, LLaVA, CLIP, SigLIP, and Whisper; unknown keys ignored, unsupported values (for example `resample` 4 box) rejected |
| `formats/mel-spec.md` (new) | Whisper features: 16 kHz mono f32, pad or trim to 480000 samples (30 s) per window, periodic Hann(400), `n_fft = 400` (factors 2^4 · 5^2, so FFTs are mixed radix), hop 160, `center = True` with reflect padding, 201 bins, drop the last frame (3000 frames), power spectrum, Slaney mel filterbank with Slaney normalization (fmin 0, fmax 8000) for 80 or 128 mels, `log10(max(m, 1e-10))`, `max(x, max(x) - 8)`, `(x + 4) / 4`. Long-form: features over the whole file, windows of 3000 frames at `seek`. Mono downmix is the channel mean; resampling per `resample_audio` (torchaudio `sinc_interp_hann`, width 6, rolloff 0.99) |
| `spec/asr-decoding.md` (new) | SOT sequence, special tokens, language detection, timestamp rules, suppress lists, `best_of` sampling with PCG32 per `spec/sampling.md` (one `uniform_f64` per token, seed per window = SplitMix64(seed, window index)), beam search (log-prob sum, length-normalized by token count, ties to the lowest token id, `patience`), temperature fallback (compression ratio `len(utf8) / len(zlib.compress(utf8, 9))`, average log-prob, no-speech probability at the SOT position), long-form seek and prompt conditioning (`<|startofprev|>` + up to 223 previous tokens; dropped when the last temperature exceeded 0.5), segment output, and the `basic` text normalizer for WER |
| `formats/encoder-entry.md` (new) | In memory: `EncoderEntry` (2.4). **Export envelope** used by `PushEmbeddings` (the embedding transfer wire format): header `{magic "TLEM", u16 version = 1, u16 dtype (TL_F32 or TL_F16 codes from 2.4), u64 media_hash, u32 preprocess_rev, u32 n_tokens, u32 d, u8 kind, u8 detail, u16 model_id_len, model_id utf-8}`, payload `[n_tokens][d]` row-major little-endian, then `u32 crc32c` over everything before it. Bad magic, CRC, version, or model id: rejected (`FAILED_PRECONDITION`) |
| `formats/kv-block.md` (minor) | new section **Multimodal hash ids**: the hashed token stream replaces the placeholder at offset `j` of an image span with `0x8000_0000 \| ((h >> (31 * (j & 1))) & 0x7FFF_FFFF)` where `h = media_hash`; all other ids are unchanged, so text-only blocks hash exactly as in v1. Model ids are always below `2^31`. KV payload formats v1 and v2 are unchanged |
| `formats/config.schema.json` (minor) | `tl_arch` adds `vit, clip, siglip, idefics3, llava, llava_next, qwen2vl, whisper`; `tl_modalities` in `{text, image, audio}`; the engine serves `bigram`, `llama`, `idefics3`, `whisper` (`llava` optional through the zoo) |
| `formats/corpus-shard.md` (minor) | image-text shards: `id string, image_png binary, width int32, height int32, caption string, source_id string, url string, license_spdx string, phash fixed_size_binary(8), dhash fixed_size_binary(8), faces_redacted int32, clip_score float32, near_dup_cluster int64, split string`. Audio shards: `id string, audio_wav binary (16 kHz mono s16le), duration_s float32, transcript string, lang string, speaker_key fixed_size_binary(16) (HMAC of the source speaker id; never the id), consent string, source_id string, license_spdx string, pii_redactions int32, near_dup_cluster int64, split string`. No document, image, or speaker crosses splits |
| `formats/ledger.schema.json` (minor) | adds `modality: text\|image\|audio`, `biometric: bool`, `consent: none\|public-domain\|cc-licensed\|explicit-donation`, `attribution: string`, `prohibited_uses: [string]` (for example `speaker-identification`). `ModelRelease` refuses `biometric: true` sources whose `consent` is `none` |
| `formats/usage.v2.sql` (minor of the per-version ledger from craft.14) | adds `images int, image_tokens int, audio_seconds real, modality text` |
| `formats/policy.v1.schema.json` (minor) | `match.modalities: [image, audio]`, `limits{max_images, max_image_pixels, max_audio_seconds}`, action `deny` with reason `media_policy` |
| `formats/eval-case.schema.json` (minor) | `input` may carry `media: [{kind, path, sha256}]`; suites `asr`, `vqa`, `caption`, `zeroshot`, `mm_bias` |
| `templates/MODEL_CARD.md` (minor) | sections "Modalities and input limits", "Per-group performance" (WER per accent with CIs), "Prohibited uses" (speaker identification, face recognition), "Third-party components" (SmolVLM-256M, whisper-tiny) |
| `spec/cli-roles.md` (minor) | verbs `{tinyllm} features`, `transcribe`, `vlm generate`, `logits --mm-prompts`, `train {lenet,clip,vlm}`, `eval {zeroshot,vqa,caption,asr}`, `preprocess image`; `{corpus} mm build`; `{ctl} chat --image`, `{ctl} transcribe`. Every generating verb ends with the JSON line `{"ids": [...], "text": ...}` |

### 2.7 OpenAPI: `openapi/openai-subset.v2.yaml` to `info.version: 2.1.0`

Engine and gateway tiers. v1 (sunset in P11) gains nothing.

```yaml
components:
  schemas:
    ChatMessage:
      properties:
        content:
          oneOf:
            - type: string
            - type: array
              minItems: 1
              items: { $ref: '#/components/schemas/ContentPart' }
    ContentPart:
      oneOf:
        - $ref: '#/components/schemas/TextPart'
        - $ref: '#/components/schemas/ImageUrlPart'
        - $ref: '#/components/schemas/InputAudioPart'      # 422 unsupported_parameter unless sq.audio-llm serves the model
      discriminator: { propertyName: type }
    TextPart:
      type: object
      required: [type, text]
      properties: { type: { const: text }, text: { type: string } }
    ImageUrlPart:
      type: object
      required: [type, image_url]
      properties:
        type: { const: image_url }
        image_url:
          type: object
          required: [url]
          properties:
            url:
              type: string
              description: >
                data:image/png;base64,... or data:image/jpeg;base64,... (engine and gateway);
                https://... only at the gateway with [gateway.media.remote_fetch] enabled and the host allowlisted;
                tl-media:<16 hex> is internal (EPD) and is rejected from clients with 400.
            detail: { type: string, enum: [auto, low, high], default: auto }   # low: global image only, no tiles
    InputAudioPart:
      type: object
      required: [type, input_audio]
      properties:
        type: { const: input_audio }
        input_audio:
          type: object
          required: [data, format]
          properties: { data: { type: string, contentEncoding: base64 }, format: { type: string, enum: [wav] } }
    PromptTokensDetails:
      properties:
        cached_tokens: { type: integer }
        image_tokens: { type: integer, description: placeholder tokens counted in prompt_tokens }
    TranscriptionRequest:                                   # multipart/form-data
      type: object
      required: [file, model]
      properties:
        file: { type: string, format: binary, description: RIFF WAV (E7) }
        model: { type: string }
        language: { type: string, description: ISO-639-1; omitted means detect }
        prompt: { type: string, description: previous-text conditioning, at most 223 tokens kept }
        response_format: { type: string, enum: [json, text, srt, verbose_json, vtt], default: json }
        temperature: { type: number, minimum: 0, maximum: 1, default: 0, description: 0 enables the fallback ladder }
        timestamp_granularities[]: { type: array, items: { enum: [segment] } }   # word: 422 unsupported_parameter
        stream: { type: boolean, default: false }
    Transcription:                                          # json
      type: object
      required: [text, usage]
      properties:
        text: { type: string }
        usage: { type: object, required: [type, seconds], properties: { type: { const: duration }, seconds: { type: number } } }
    TranscriptionVerbose:
      type: object
      required: [task, language, duration, text, segments, usage]
      properties:
        task: { const: transcribe }
        language: { type: string }
        duration: { type: number }
        text: { type: string }
        segments:
          type: array
          items:
            type: object
            required: [id, seek, start, end, text, tokens, temperature, avg_logprob, compression_ratio, no_speech_prob]
paths:
  /v1/audio/transcriptions:
    post:
      requestBody: { required: true, content: { multipart/form-data: { schema: { $ref: '#/components/schemas/TranscriptionRequest' } } } }
      responses:
        '200':
          content:
            application/json: { schema: { oneOf: [ { $ref: '#/components/schemas/Transcription' }, { $ref: '#/components/schemas/TranscriptionVerbose' } ] } }
            text/plain: {}                                   # text, srt, vtt
            text/event-stream: {}                            # stream=true: events transcript.text.delta {delta}, transcript.text.done {text, usage}
        '400': { $ref: '#/components/responses/Error' }
        '413': { $ref: '#/components/responses/Error' }
        '422': { $ref: '#/components/responses/Error' }
```

New error rows (shape unchanged):

| Status | `type` / `code` | When |
|---|---|---|
| 400 | `invalid_request_error` / `invalid_media` | undecodable image or WAV, bad base64, `tl-media:` from a client |
| 400 | `invalid_request_error` / `unsupported_media_type` | not PNG/JPEG, not RIFF WAV |
| 413 | `invalid_request_error` / `media_too_large` | request bytes, image bytes, pixels (header-checked before decode), image count, or audio seconds over limits |
| 422 | `invalid_request_error` / `unsupported_parameter` | `image_url` to a text-only model, `timestamp_granularities[]=word`, remote URL with fetch disabled (`param: messages[i].content[j].image_url.url`) |
| 451 | `policy_error` / `usage_policy` | a `media_policy` rule |

Conformance cases (`course/conformance/openapi/cases/`, tier, `requires`):

| Case | Tier | Asserts |
|---|---|---|
| `mm.image.data_url` | engine | `requires = ["L15.3"]`: PNG and JPEG data URLs, greedy output stable across 2 calls |
| `mm.image.detail_low` | engine | `detail: low` gives `image_tokens` equal to one global tile's count |
| `mm.usage.image_tokens` | both | `usage.prompt_tokens_details.image_tokens` equals `image_token_count` from the fixture table; `prompt_tokens` includes them |
| `mm.cache.same_image` | engine | `requires = ["L15.2"]`: repeated request reports `cached_tokens >= image_tokens`; a different image with the same text reports `cached_tokens < first image span start + 64` (no false hit) |
| `mm.errors.*` | both | 400, 413, 422 rows above; pixel bomb (a 30000 x 30000 PNG header in under 1 KiB) is rejected without allocating the pixels (RSS probe on `/metrics`) |
| `mm.stream.equals_nonstream` | engine | the text-path invariant with an image |
| `audio.json`, `audio.verbose_json`, `audio.srt`, `audio.vtt`, `audio.text` | both | `requires = ["L15.7"]`: schemas; SRT `HH:MM:SS,mmm` and VTT `HH:MM:SS.mmm` timestamps; segments monotone |
| `audio.stream` | engine | `transcript.text.delta` concatenation equals `transcript.text.done.text` |
| `audio.errors.*` | both | mp3 bytes give `unsupported_media_type`; 31-minute WAV gives 413 |
| `mm.remote_url` | gateway | `requires = ["gw.09"]`: non-allowlisted, private, link-local, and redirect-to-private URLs refused; allowlisted fixture host inlined |
| `mm.internal_scheme` | gateway | client-sent `tl-media:` refused; `X-TL-MM-Handle` stripped |

### 2.8 gRPC (`proto/tl/*/v1`)

New `tl/encoder/v1/encoder.proto` (D6 pattern: dedup by hash, push missing, release on abort):

```proto
syntax = "proto3";
package tl.encoder.v1;
service EncoderService {                                   // served by role=encode engines on :50051 next to EngineControl
  rpc Encode(EncodeRequest) returns (EncodeResponse);      // preprocess + encode each item, push to target, return refs
}
service EmbeddingTransfer {                                // served by unified and prefill engines on :50053 ({emb_port} locally)
  rpc HasEmbeddings(HasEmbeddingsRequest) returns (HasEmbeddingsResponse);
  rpc PushEmbeddings(stream EmbeddingChunk) returns (EmbeddingAck);
  rpc Release(ReleaseEmbeddingsRequest) returns (ReleaseEmbeddingsResponse);
}
message MediaInput { uint32 kind = 1; bytes data = 2; uint64 media_hash = 3; string detail = 4; }  // data <= 4 MiB per item, larger items stream via chunks
message EncodeRequest { string request_id = 1; string model = 2; repeated MediaInput media = 3;
  string target = 4;                                        // host:port of the target's EmbeddingTransfer
  int64 deadline_unix_ms = 5; }
message MediaRef { uint64 media_hash = 1; uint32 kind = 2; uint32 preprocess_rev = 3; uint32 n_tokens = 4; string detail = 5; }
message EncodeResponse { string handle_id = 1; repeated MediaRef refs = 2; uint32 encoded = 3; uint32 deduped = 4; double encode_ms = 5; }
message HasEmbeddingsRequest { string model = 1; repeated MediaRef refs = 2; }
message HasEmbeddingsResponse { repeated bool present = 1; }
message EmbeddingChunk { string handle_id = 1; MediaRef ref = 2; uint32 chunk_index = 3; uint32 n_chunks = 4;
  bytes payload = 5;                                        // formats/encoder-entry.md envelope, split at 4 MiB
  uint32 crc32c = 6; }
message EmbeddingAck { string handle_id = 1; uint32 received = 2; uint32 deduped = 3; }
message ReleaseEmbeddingsRequest { string handle_id = 1; }  message ReleaseEmbeddingsResponse {}
```

Minor additions to existing protos:

```proto
// tl/engine/v1/engine.proto
message PrefillRequest { /* 1..7 unchanged */ repeated tl.encoder.v1.MediaRef media = 8; string mm_handle = 9; }
message InfoResponse   { /* 1..6 unchanged */ repeated string modalities = 7; uint32 preprocess_rev = 8; }
// tl/control/v1/control.proto
message WorkerStatus   { /* 1..12 unchanged; role adds "encode" */ repeated string modalities = 13;
  int32 encoder_queue_depth = 14; int64 encoder_cache_free_bytes = 15; string emb_address = 16; }
```

**EPD flow.** The gateway parses media (gw.09), picks encoder E (affinity on the first `media_hash`, bounded by `encoder_queue_depth`) and target T (unified, or prefill P then decode D per 2.7). It rewrites each image part to `tl-media:<hex16>` and calls `E.Encode(target = T.emb_address)`. E asks T `HasEmbeddings`, encodes only missing items (its own encoder cache first), pushes envelopes, and returns refs. The gateway then sends the rewritten chat request to T with `X-TL-MM-Handle: <handle_id>` (internal, stripped from clients). T resolves `tl-media:` refs from its encoder cache, pinned for the request. A miss at T returns internal 409 `mm_missing` and the gateway retries once through the unified path that encodes locally. On abort the gateway calls `Release`. Encoder loss before first byte: retry another E, then local encode (`[gateway.media].epd_fallback = "local"`), counted in `tl.gateway.epd.fallbacks`.

### 2.9 Runtime configuration (`config/runtime.schema.json`, minor)

```toml
[engine]
role = "unified"                                  # unified | prefill | decode | encode
emb_listen = ":50053"                             # EmbeddingTransfer (unified, prefill)

[engine.mm]
device = "cpu"                                    # cpu | metal (local only; CI and tests are cpu)
kernels = "candle"                                # candle | c (optional L13.8, L13.9, L14.6 via L15.8; cpu only; startup fails if a C unit is a stub)
dtype = "f32"                                     # f32 | f16 (f16 only on metal)
max_images_per_request = 8
max_image_bytes = 20971520
max_image_pixels = 33554432                       # checked from the header before decode
encoder_cache_bytes = 536870912
max_encoder_tokens_per_step = 2048                # encoder budget per scheduler step (L15.4)
preprocess_threads = 2
embedding_transfer_dtype = "f16"                  # envelope dtype

[engine.asr]
max_audio_seconds = 1800
beam_size = 5
best_of = 5
patience = 1.0
temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
compression_ratio_threshold = 2.4
logprob_threshold = -1.0
no_speech_threshold = 0.6
condition_on_previous_text = true
max_initial_timestamp = 1.0

[gateway.media]
max_request_bytes = 26214400
max_images = 8
max_image_pixels = 33554432
max_audio_bytes = 26214400
max_audio_seconds = 1800
epd_fallback = "local"                            # local | fail
remote_fetch = { enabled = false, allow_hosts = [], timeout_ms = 3000, max_bytes = 20971520 }

[[gateway.routes]]
model = "smolvlm-256m"
targets = ["vlm"]
epd = { encode_pool = "encode" }
[[gateway.routes]]
model = "whisper-tiny"
targets = ["asr"]
```

`TL_ENGINE_MM__DEVICE=metal` style overrides follow DESIGN 2.12 for scalars (nested tables use `__` per level: `TL_ENGINE__MM__DEVICE`; the loader conformance cases in `config/` gain one nested case).

### 2.10 Ports and Kubernetes objects (rows to add to DESIGN 2.13)

| Component | Helm release (chart) | Kind | Replicas on kind | Ports |
|---|---|---|---|---|
| engine, VLM pool | `<system>-engine-vlm` (`<system>-engine`, `role: unified`, SmolVLM model dir) | Deployment + KEDA `ScaledObject` on `tl_engine_queue_depth{model="smolvlm-256m"}` | 1 (max 2) | 8000, 50051, 50052, **50053** emb, 9464 |
| engine, encode pool | `<system>-engine-encode` (`role: encode`) | Deployment + KEDA `ScaledObject` on `tl_engine_encoder_queue_depth`; PDB `minAvailable: 1` | 2 (min 1, max 4) | 50051 (EncoderService, EngineControl), 9464 |
| engine, ASR pool | `<system>-engine-asr` (`role: unified`, whisper-tiny) | Deployment + KEDA `ScaledObject` on `tl_asr_audio_seconds_queued` | 1 (max 3) | 8000, 50051, 9464 |

`contracts/helm/engine.values.schema.json` (minor): `role` enum adds `encode`; `mm.device` (`cpu` only on kind, since the kind node is Linux), `mm.encoderCacheBytes`, `asr.*`, `autoscaling.keda{metric, threshold, min, max}`. Policy tests from dep.03 apply unchanged.

### 2.11 Observability (`otel/semconv.md`, `otel/metrics.yaml`, `otel/slo.schema.json`; minor)

| Span | Kind | Emitted by | Required attributes |
|---|---|---|---|
| `gateway.media` | INTERNAL | gateway | `tl.media.items, tl.media.bytes, tl.media.decision` (`allow`, `reject:<code>`) |
| `tl.encoder.v1.EncoderService/Encode` | CLIENT/SERVER | gateway, encode engine | `rpc.system=grpc` |
| `engine.preprocess`, `engine.encode` | INTERNAL | engine | `tl.mm.modality, tl.mm.items, tl.mm.tokens, tl.mm.cache_hit, tl.mm.device` |
| `encoder.transfer` | CLIENT/SERVER | encode, target | `tl.mm.bytes, tl.mm.deduped` |
| `asr.window` | INTERNAL | engine | `tl.asr.seek, tl.asr.temperature, tl.asr.fallbacks, tl.asr.no_speech` |

| Metric | Type, unit | Use |
|---|---|---|
| `tl.engine.encode.duration{modality}` | histogram, s | **encode time** SLO |
| `tl.engine.encoder_cache.{hits,misses,evictions}`, `tl.engine.encoder_cache.bytes` | counter, gauge | cache efficiency |
| `tl.engine.encoder.queue.depth` | gauge | encode pool autoscaling |
| `tl.asr.rtf` | histogram, 1 | **real-time factor** SLO: processing seconds / audio seconds |
| `tl.asr.audio_seconds.queued` | gauge | ASR pool autoscaling |
| `tl.gateway.media.rejections{reason}`, `tl.gateway.media.bytes{tenant}` | counter | ops.14 |
| `tl.gateway.epd.fallbacks` | counter | ops.13 |
| `tl.usage.image_tokens{tenant}`, `tl.usage.audio_seconds{tenant}` | counter | metering cross-check |

SLOs (course defaults, scaled by `ss bench --calibrate --in-cluster`; required alert names added to `otel/slo.schema.json`):

| SLO | Default | Alerts |
|---|---|---|
| Encode p95 per image (SmolVLM, one global tile, cpu) | < 400 ms | `EncodeLatencyBurnFast`, `EncodeLatencyBurnSlow` |
| Image-request TTFT p95 | < 1.5 s at 1 rps | `ImageTTFTBurnFast`, `ImageTTFTBurnSlow` |
| ASR RTF p95 (whisper-tiny, greedy, cpu) | < 0.3 | `AsrRtfBurnFast`, `AsrRtfBurnSlow` |
| Text TTFT p95 under mixed load | unchanged from DESIGN 2.11 | existing |

### 2.12 Learner repo additions (DESIGN 2.15)

```
python/tinyllm/
  sig/        conv, resample, fft, stft, mel                      # M12.1, M12.3 to M12.6
  image/      layout                                              # M12.2
  vision/     conv, rope2d, vit, tiling, contrastive, preprocess, vlm   # L13.1 to L13.7
  speech/     frontend, encoder, decoder, decode                  # L14.1 to L14.4
  io/         wav, hf_vlm, hf_whisper                             # L14.1, L13.7, L14.3
  eval/       mm, mm_bias                                         # L14.5, ethics.07
python/corpus/ image_pairs, audio_pairs, faces, png               # data.10 to data.12
c/src/kernels/          conv.c image.c audio.c                    # optional: L13.8, L13.9, L14.6 (headers in contracts, 2.2)
rust/crates/tl-sys/     src/mm.rs                                 # optional: L15.8 (extern decls + RAII for the three headers)
rust/crates/tl-media/   src/{lib,decode,exif,wav}.rs              # L13.6 (decode, exif), L15.1 (wav)
rust/crates/tl-ds/      src/lru_bytes.rs                          # ds.10
rust/crates/tl-py/      src/media.rs                              # L13.6
rust/crates/tl-engine/  src/mm/{media,image,audio,kernels,encoder_cache,hash_ids,vision,merge,epd}.rs
                        src/asr/{whisper,decode}.rs
rust/crates/tl-serve/   src/{audio,encoder_service}.rs
go/gateway/media/
go/workflows/mm_corpus_build.go
deploy/helm/<system>-engine/values-{vlm,encode,asr}.yaml           # (learner)
deploy/observability/dashboards/multimodal.json, rules/multimodal.yaml   # (learner)
docs/CONSENT_POLICY.md, docs/adr/*-epd.md                          # (learner)
```

### 2.13 `allowed-deps.toml` additions

```toml
[rust.tl-media]
zune-png = "0.4"
zune-jpeg = "0.4"
[rust.tl-engine]
candle-core = { version = "0.9", features = [] }       # "metal" enabled by the learner on macOS through a target-specific feature
candle-nn = "0.9"
rustfft = "6"                                          # FFT for the engine log-mel; STFT framing, window, mel, log are learner code
fast_image_resize = "5"                                # core resize in the engine; PIL parity is checked, see Q3
rubato = "0.16"                                        # non-16 kHz audio only
flate2 = { version = "1", default-features = false, features = ["zlib"] }   # compression ratio, zlib backend to match Python zlib
# forbidden for every component: candle-transformers, candle-examples, image (decoders beyond zune), ort, tch
[go.gateway]
# stdlib image/png, image/jpeg for DecodeConfig; no new modules
```

Versions are requirements at authoring time; exact pins come from the learner's lock files (DESIGN allowed-deps header).

### 2.14 Interface boundary rows (DESIGN 2.3)

| From | To | Mechanism | Contract | Conformance suite |
|---|---|---|---|---|
| Python | Rust | PyO3 `tinyllm_rs.decode_image` | `py/tinyllm_rs.pyi` | `parity/image.decode` |
| Python preprocessing | Rust preprocessing | spec | `formats/image-preprocess.md`, `formats/mel-spec.md` | `parity/image.preprocess`, `parity/audio.logmel`, `parity/image.tokens` (Python, Rust, Go) |
| Python decoding | Rust decoding | spec + recorded logits | `spec/asr-decoding.md` | `parity/asr.decode` |
| gateway | encode engine | gRPC | `proto/tl/encoder/v1/encoder.proto` | `encoder-grpc/` |
| encode engine | target engine | gRPC | `EmbeddingTransfer` + `formats/encoder-entry.md` | `encoder-grpc/`, `parity/embed.wire.v1` |
| gateway, engine | media hash | spec | E4 in `formats/encoder-entry.md` | `parity/media.hash` (Python, Rust, Go) |

---

## 3. Module catalog

Columns follow DESIGN 4.0. Tests codes: U unit/boundary, G gradcheck, O golden, E differential, I property, S statistical, L learning, C conformance, F fault, B bench. All modules are `pass = 12`. "Core" unless marked.

### 3.1 M12 Signals and images (`math/12-signals-and-images/`, Python)

| ID | Module | Core | Lang | Path | Prereqs | Interface | Call sites | Tests |
|---|---|---|---|---|---|---|---|---|
| M12.1 | Convolution as a linear map: cross-correlation vs convolution, output size with padding, stride, dilation, groups; doubly block Toeplitz matrix; im2col and its adjoint col2im; the gradient of a conv is a conv | core | Py | `tinyllm/sig/conv.py` | M03.2, M03.3, M08.2, reading: `ml/06-neural-architectures/code/conv2d.c` | 2.3 `sig/conv.pyi` | L13.1 (im2col, col2im), L14.2 (`conv_out_size` for the stem) | O (torch `conv1d`/`conv2d` and `F.unfold`/`F.fold` incl. groups and dilation), E (direct == im2col @ W == `conv_matrix` @ vec), I (adjoint: `<im2col x, y> == <x, col2im y>`; flipping the kernel turns correlation into convolution) |
| M12.2 | Image tensors and layouts: NCHW, NHWC, channels-last strides, contiguity, u8 to f32 rescale and normalize, PIL grayscale, patchify as a reshape | core | Py | `tinyllm/image/layout.py` | M03.1, lang.01 | `image/layout.pyi` | L13.3 (patchify), L13.6 (rescale_normalize), data.10 (grayscale for hashes) | U (strides table, hand example 2x3x4), O (PIL `convert("L")` on fixtures, exact), I (layout roundtrip; `unpatchify(patchify(x)) == x`) |
| M12.3 | Resampling and interpolation: sampling theorem in 2D, nearest, bilinear, bicubic (a = -0.5 vs -0.75), Lanczos3, antialias as a low-pass, separable passes and PIL rounding; 1D band-limited audio resampling | core | Py | `tinyllm/sig/resample.py` | M12.2, M02.1, reading: M12.5 (aliasing, as concept) | `sig/resample.pyi` | L13.6 (resize), L13.3 (`interpolate_pos_embed`), L14.1 (`resample_audio`), data.10 (hash thumbnails) | O (PIL `resize` per kernel: at most 1 LSB on 100% of pixels and exact on >= 99%; torchvision antialias float mode atol 1e-4; torchaudio `resample` atol 1e-5), I (weights sum to 1; upsample then downsample of a band-limited signal within bound; identity at equal size) |
| M12.4 | DFT and FFT: the DFT matrix, unitarity, convolution theorem, radix-2 Cooley-Tukey, mixed radix for `n = 400`, real FFT packing, DCT-II via FFT | core | Py | `tinyllm/sig/fft.py` | M00.2 (Euler), M03.3, M09.3 | `sig/fft.pyi` | M12.5 (stft), data.10 (`dct2_ortho` in pHash) | O (numpy `fft` and scipy `dct(norm='ortho')` within the frozen f64 bound), E (FFT == `dft_matrix @ x`; `fft_convolve` == `np.convolve`), I (Parseval; `ifft(fft(x)) == x`; unsupported prime factor raises) |
| M12.5 | Sampling, Nyquist, windows, STFT: aliasing, periodic vs symmetric Hann, COLA, center reflect padding, frame counts, power spectrum, ISTFT | core | Py | `tinyllm/sig/stft.py` | M12.4 | `sig/stft.pyi` | L14.1 (Whisper frontend) | O (`torch.stft(center=True, pad_mode='reflect', window=hann_window(400, periodic=True))` complex output atol 1e-5), I (COLA reconstruction error < 1e-6 at hop n/4; a tone above Nyquist folds to the predicted bin), U (frame count table incl. Whisper's 3001 then drop last) |
| M12.6 | Mel scale and filterbanks: HTK vs Slaney mel, Slaney area normalization, decibels and `power_to_db`, Whisper's log10 clamp and 8-unit dynamic range | core | Py | `tinyllm/sig/mel.py` | M00.1 (logs), M12.5 | `sig/mel.pyi` | L14.1, data.11 (mel dHash for audio near-dup) | O (`librosa.filters.mel(sr=16000, n_fft=400, n_mels=80/128)` atol 1e-7; `librosa.power_to_db`), I (each filter is a triangle with the declared edges; Slaney rows have unit area in Hz), U (dB of 1, 10, 1e-12 with `amin`) |

### 3.2 L13 Vision (`ml/08-tinyllm/p13-vision/`, Python)

| ID | Module | Core | Lang | Path | Prereqs | Interface | Call sites | Tests |
|---|---|---|---|---|---|---|---|---|
| L13.1 | Conv layers and pooling on autograd: `Conv2d`, `Conv1d` through im2col with col2im backward, max/avg pool, LeNet-5 | core | Py | `tinyllm/vision/conv.py` | M12.1, L0.2, L0.4, M06.3 | `vision/conv.pyi` | L13.3 (`PatchEmbed` is a stride-p conv), L14.2 (conv stem) | G (frozen gradcheck on x, W, b for stride/pad/dilation/groups), O (torch fwd/bwd fixture), I (pool backward routes to argmax, ties lowest index), L (LeNet-5 on `digits` 8x8: test accuracy >= calibrated threshold in 400 steps, about 0.97) |
| L13.2 | 2D RoPE and M-RoPE: axial 2D rotary for vision patches, Qwen2-VL M-RoPE with `mrope_section`, multimodal position ids (text resumes after the image's max position) | core | Py | `tinyllm/vision/rope2d.py` | L7.3, M00.2 | `vision/rope2d.pyi` | L13.3 (`pos='rope2d'`), L13.7 (`qwen2vl` arch) | O (HF Qwen2-VL `get_rope_index` and `apply_multimodal_rotary_pos_emb` on a tiny config), I (per axis, scores depend only on the coordinate difference; M-RoPE equals 1D RoPE when t = h = w), G |
| L13.3 | Vision transformer encoder: patch embedding, CLS token, learned / 2D sincos / 2D RoPE positions, pre-LN blocks with LayerNorm and GELU, position-table interpolation, padded patches with bucketed positions (Idefics3), hidden-state taps for `vision_feature_layer` | core | Py | `tinyllm/vision/vit.py` | L13.1, L13.2, L5.1, L5.3, M12.2, M12.3 | `vision/vit.pyi` | L13.5 (image tower), L13.7 (vision tower) | O (tiny HF `ViTModel`, `CLIPVisionModel`, `SiglipVisionModel`, `Idefics3VisionTransformer` hidden states atol 1e-5), E (PatchEmbed as conv == patchify + Linear), I (without positions, outputs permute with patches; padded patches never change unpadded outputs) |
| L13.4 | Dynamic resolution and tiling: LLaVA-NeXT AnyRes (`select_best_resolution`, base image, unpad, `image_newline`), Idefics3/SmolVLM splitting (longest edge, 512 tiles, global image, row and column marker tokens), pixel shuffle, image token counts | core | Py | `tinyllm/vision/tiling.py` | M12.2, M12.3, M05.1 | `vision/tiling.pyi` | L13.6 (splitting), L13.7 (placeholder expansion), L15.1 (Rust port proven against it), gw.10 (Go `ImageTokens` re-implemented, parity fixture) | O (HF `select_best_resolution`, Idefics3 tile grids and prompt strings for 40 fixture sizes), I (pixel shuffle invertible; `image_token_count` == placeholders emitted by `render_mm_chat`; tokens monotone in area within a processor) |
| L13.5 | Contrastive image-text: CLIP symmetric InfoNCE with learned temperature (clamp at 100), SigLIP sigmoid loss with learned t and b (init log 10 and -10), dual encoder with a tiny text tower, zero-shot classifiers from prompt templates | core | Py | `tinyllm/vision/contrastive.py` | L13.3, L5.3, M11.1, L0.3 | `vision/contrastive.pyi` | data.10 (CLIP-score filter), ethics.07 (zero-shot disparity probe) | G, O (HF `clip_loss` and `SiglipModel` loss on fixture embeddings), I (CLIP loss symmetric in modalities; SigLIP defined at batch 1; temperature gradient sign), L (tiny dual encoder on `mm-shapes`, 600 steps: zero-shot accuracy over 24 color-shape classes >= calibrated threshold) |
| L13.6 | Image preprocessing to the spec: decode through `tinyllm_rs` (`tl-media` crate: zune PNG/JPEG, EXIF parse), orientation, RGB conversion, resize, splitting, rescale, normalize, `preprocessor_config.json` loading, media hash | core | Py, Rust | `tinyllm/vision/preprocess.py`, `rust/crates/tl-media/src/{lib,decode,exif}.rs`, `rust/crates/tl-py/src/media.rs` | M12.2, M12.3, L13.4, M06.3 (FNV-1a), L1.5 (`tl-py` crate root), lang.12 | `vision/preprocess.pyi`, `tinyllm_rs.pyi` | L13.7, L15.1 (Rust port and `tl-media` reuse), data.10 (decode, orientation) | O (HF `CLIPImageProcessor`, `SiglipImageProcessor`, `Idefics3ImageProcessor`: u8 resized pixels within 1 LSB, `pixel_values` atol `1/(255*std)`), U (EXIF orientations 1 to 8 on fixture JPEGs; CMYK and 16-bit PNG rejected), F (truncated files, header pixel bomb rejected before decode), I (media hash stable; preprocessing idempotent on already-sized inputs) |
| L13.7 | Vision-language model: vision tower + projector (`linear` after pixel shuffle, `mlp2x_gelu`) + Llama LM, placeholder expansion in the chat template, embedding merge, freezing schedule (projector-only then full or LoRA), HF weight loading for SmolVLM-256M-Instruct, LLaVA, Qwen2-VL tiny | core | Py | `tinyllm/vision/vlm.py`, `tinyllm/io/hf_vlm.py` | L13.3, L13.4, L13.6, L13.2, L7.9, L6.6, L1.5, M10.3 | `vision/vlm.pyi` | L15.3 (engine logits proven against it), dur.13 (`vqa` and `caption` suites run `{tinyllm}`) | O (tiny random `Idefics3ForConditionalGeneration`, `LlavaForConditionalGeneration`, `LlavaNextForConditionalGeneration`, `Qwen2VLForConditionalGeneration` logits atol 1e-5 on oracle `pixel_values`), E (merge == HF `inputs_embeds`), I (stage 1: frozen params bitwise unchanged and no optimizer state; placeholders == feature rows), L (tiny VLM on `mm-shapes`: align 200 steps then SFT 300 steps, VQA accuracy >= calibrated) |
| L13.8 | Conv kernels in C: im2col + `tl_matmul_f32` conv2d (groups, stride, padding, dilation), conv1d for the Whisper stem, fused patch embedding (NCHW and NHWC) | **optional** (skippable) | C | `c/src/kernels/conv.c` | M12.1, L13.1, L9.1, rt.02, rt.03, reading: `ml/06-neural-architectures/code/conv2d.c` | 2.2 `tinyllm/conv.h` | L15.8 (engine `kernels = "c"`: SmolVLM patch embed, Whisper stem) | E (vs the learner's Python `conv2d_direct`, `Conv1d`, `PatchEmbed` through ctypes, within the frozen dot-product bound, and vs the torch fixture), I (per image, output bitwise equal at N = 1 and N = 7; im2col row order matches M12.1), F (alloc fail-after-n), B (>= 5x the naive `conv2d.c` loop at 3x224x224, k 16, s 16) |
| L13.9 | Resize kernels in C: separable nearest, bilinear, bicubic (a = -0.5), Lanczos3 with antialias and PIL rounding; planar f32 resize; fused rescale + normalize to CHW | **optional** (skippable) | C | `c/src/kernels/image.c` | M12.2, M12.3, L13.6, rt.02, rt.03 | 2.2 `tinyllm/image.h` | L15.8 (engine `kernels = "c"` image preprocessing) | E (u8 output bit-exact with the learner's Python `resize(mode='pil')` and within 1 LSB of PIL on the M12 fixtures; f32 path within 1e-5), I (weights sum to 1; identity at equal size), F (huge dims overflow-checked, `TL_EINVAL`) |

### 3.3 L14 Speech (`ml/08-tinyllm/p14-speech/`, Python)

| ID | Module | Core | Lang | Path | Prereqs | Interface | Call sites | Tests |
|---|---|---|---|---|---|---|---|---|
| L14.1 | Audio frontend: WAV parsing (RIFF chunks, s16/s24/f32, extensible), mono downmix, resample to 16 kHz, pad or trim to 30 s, exact Whisper log-mel (80 or 128 mels) | core | Py | `tinyllm/io/wav.py`, `tinyllm/speech/frontend.py` | M12.3, M12.5, M12.6, lang.12 | `io/wav.pyi`, `speech/frontend.pyi` | L14.2, L15.1 (Rust port proven against it), data.11 | O (HF `WhisperFeatureExtractor` atol 1e-4 on fixture clips, 80 and 128 mels), U (fixture WAVs: s24, f32, stereo, `LIST` chunks, odd chunk padding), F (truncated data chunk, absurd sizes), I (silence gives the floor value everywhere) |
| L14.2 | Whisper encoder: conv stem (k3 p1, then k3 s2 p1, GELU), fixed sinusoids (sin then cos halves), pre-LN blocks, `ln_post` | core | Py | `tinyllm/speech/encoder.py` | L13.1, M12.1, L5.1, L5.3, L5.4 | `speech/encoder.pyi` | L14.3 (Whisper composes it), L15.6 (Rust encoder proven against it) | O (tiny random `WhisperModel` encoder outputs atol 1e-5; `whisper_sinusoids` vs HF), I (3000 frames give 1500 positions; output length formula from M12.1) |
| L14.3 | Whisper decoder and weights: learned positions, causal self-attention with KV cache, cross-attention with cross-KV computed once per window, tied output; HF loader for `openai/whisper-tiny` | core | Py | `tinyllm/speech/decoder.py`, `tinyllm/io/hf_whisper.py` | L14.2, L8.2, L7.9 (downloader, safetensors), L0.6 | `speech/decoder.pyi` | L14.4, L15.6 | O (tiny random `WhisperForConditionalGeneration` logits atol 1e-5; whisper-tiny encoder hidden atol 1e-3 nightly), E (cached decode == full recompute; cross-KV reuse == recompute), I (`param_count` == HF) |
| L14.4 | Whisper decoding: tokenizer specials, SOT sequence, language detection, task tokens, timestamp tokens and rules, suppress lists, greedy, beam with patience, temperature fallback (compression ratio, avg log-prob, no-speech), long-form seek with prompt conditioning, segments | core | Py | `tinyllm/speech/decode.py` | L14.3, L8.1, L4.4 (beam), M06.3, L1.2 | `speech/decode.pyi`, `spec/asr-decoding.md` | L15.6 (Rust port proven on recorded logits), dur.13 (`asr` suite runs `{tinyllm} transcribe`) | O (HF `generate` and `openai-whisper` `transcribe` token ids on margin-filtered clips, incl. timestamps), E (beam 1 == greedy; scripted `LogitsSource` traces give fixture tokens), I (timestamps paired and monotone; no text before the first timestamp; segment ends within audio), U (fallback triggers on the repetitive fixture at the first temperature, not on the clean one) |
| L14.5 | Multimodal eval metrics: Levenshtein alignment (S, D, I), WER and CER, the `basic` normalizer, bootstrap CIs, VQA accuracy, caption BLEU-4/chrF and attribute recall, zero-shot top-k | core | Py | `tinyllm/eval/mm.py` | L4.5, M07.4, M05.2 | `eval/mm.pyi` | ethics.07 (per-group WER gaps), dur.13 (multimodal `EvalSuite` rows) | O (`jiwer` WER/CER and alignments; sacrebleu for captions), I (WER(x, x) = 0; WER of an empty hypothesis is 1; CI contains the point estimate), U (alignment ties: S before D before I) |
| L14.6 | Audio kernels in C: mixed-radix real FFT plans (2, 3, 4, 5), STFT power with center reflect padding and drop-last, mel projection via `tl_matmul_f32`, Whisper log scaling | **optional** (skippable) | C | `c/src/kernels/audio.c` | M12.4, M12.5, M12.6, L14.1, L9.1, rt.02, rt.03 | 2.2 `tinyllm/audio.h` | L15.8 (engine `kernels = "c"` log-mel) | E (log-mel vs the learner's Python `whisper_log_mel` within 1e-5 and vs HF within 1e-4; `tl_rfft_f32` vs `sig.fft.rfft` within the f32 FFT bound `c * log2(n) * eps32 * norm(x)`), I (Parseval; frame count formula), F (plan create under alloc failure), B (30 s log-mel under 20 ms at calibration) |

### 3.4 L15 Multimodal serving (`ml/08-tinyllm/p15-multimodal-serving/`, Rust on candle)

| ID | Module | Core | Lang | Path | Prereqs | Interface | Call sites | Tests |
|---|---|---|---|---|---|---|---|---|
| L15.1 | Media preprocessing in the engine: media items and hash (E4), image path (decode via `tl-media`, orientation, `fast_image_resize`, tiling port, normalize), audio path (WAV parse in `tl-media`, `rubato` for non-16 kHz, STFT framing over `rustfft`, mel, log), token counts | core | Rust | `tl-engine/src/mm/{media,image,audio,kernels}.rs`, `tl-media/src/wav.rs` | L13.4, L13.6, L14.1, lang.13 | 2.4 `Preprocessor`, `MmKernels` (candle impl) | L15.2 (hash), L15.3, L15.6 | E (pixel_values vs the learner's Python within `1/(255*std)` and the oracle fixture; log-mel vs Python within 1e-4; token counts exact; media hash equal to Python and Go), F (header pixel bomb, truncated JPEG, zero-length WAV, 31-minute WAV: errors map to the 2.7 codes without panics), B (preprocess p95 vs calibration) |
| L15.2 | Encoder cache and image-aware prefix keys: byte-budgeted LRU with pins (over ds.10), keys `(model, preprocess_rev, media_hash, detail)`, hash ids for placeholder spans | core | Rust | `tl-engine/src/mm/{encoder_cache,hash_ids}.rs` | ds.10, L15.1, L8.4, L10.4 | `EncoderCache`, `hash_ids` | L15.3, L15.4, L15.5 | I (a pinned entry is never evicted; bytes never exceed the budget; text-only `hash_ids` == model ids), E (block hashes for text-only prompts bitwise equal to L10.4's; two different images with equal text never share a block past the first image token), C (`formats/kv-block.md` multimodal hash vectors) |
| L15.3 | VLM serving: candle vision tower, pixel shuffle, connector; merge into the Llama forward on C kernels; chat content parts and image placeholders in the template; `image_tokens` usage; `tl_arch = idefics3` | core | Rust | new `tl-engine/src/mm/{vision,merge}.rs`; upgrades `tl-engine/src/forward.rs` (L10.1), `tl-serve/src/openai.rs` (craft.14), `tl-serve/src/template.rs` (L10.5) | L15.1, L15.2, L13.7, L10.1, L10.5, L10.9, craft.14, lang.13 | `VisionEncoder`, `ForwardBatch.mm` | inherits the call sites of the upgraded units (serve loop, L10.6 to L10.9, gw.04, ag.01); L15.4, L15.5 | E (engine logits on `tiny-idefics3` vs the learner's Python within 1e-4 on cpu and within the HF fixture tolerance; SmolVLM nightly max abs 1e-3, top-5 equal), C (`mm.*` engine cases), I (an image item's embeddings are identical alone or with other requests in flight, E3) |
| L15.4 | Encoder-aware scheduling: encoder work items with a per-step token budget, admission accounting for image tokens, encoder cache pins across preemption (no re-encode), audio windows as schedulable work | core | Rust | upgrades `tl-engine/src/sched.rs` (L10.2) | L15.2, L15.3, L10.2, L10.3 | `SchedulerConfig.max_encoder_tokens_per_step`, `ScheduleOutput.encode` | inherited (L10.3 to L10.6, serve loop); L15.6 | I (fake-model simulation: encoder budget never exceeded; every request finishes; no block or pin leak; a preempted request never re-encodes), E (outputs identical with budgets 64 and 4096), F (KV exhaustion with image requests forces preemption, outputs unchanged) |
| L15.5 | Encode/prefill/decode disaggregation: `role = encode`, `tl.encoder.v1` services, embedding envelope, `HasEmbeddings` dedup, `tl-media:` refs, `X-TL-MM-Handle`, release on abort, heartbeat fields | core | Rust | new `tl-engine/src/mm/epd.rs`, `tl-serve/src/encoder_service.rs`; upgrades `tl-serve/src/control.rs` (L10.6) | L15.2, L15.3, L10.6, lang.10 | `EmbeddingTransport`, `encoder.proto` | gw.11 (calls `Encode` over gRPC); inherited from `control.rs` | E (EPD output == unified output, greedy and seeded), I (bytes moved = missing items x `n_tokens * d * 2` plus headers), F (chaos-proxy reset mid-push: target releases, gateway retries; CRC flip rejected; wrong `preprocess_rev` refused with `FAILED_PRECONDITION`) |
| L15.6 | Whisper serving: candle encoder and decoder, per-request cross-KV and self-KV, Rust port of the L14.4 decoding (greedy, beam, fallback, timestamps, long-form) behind the `LogitsSource` seam | core | Rust | `tl-engine/src/asr/{whisper,decode}.rs` | L15.1, L15.4, L14.3, L14.4, L10.1 (Rust PCG32), lang.13 | `WhisperRunner`, `transcribe` | L15.7 | E (`parity/asr.decode`: the same recorded logits give the same ids, segments, and fallback temperatures as the learner's Python; tiny-whisper logits vs Python within 1e-4; whisper-tiny greedy transcripts equal to Python under the near-tie rule, nightly), I (compression ratios on fixture texts equal Python's to 1e-9 with the zlib backend), B (RTF vs calibration) |
| L15.7 | Transcription endpoint: multipart parsing, `response_format` json, text, verbose_json, srt, vtt; `stream` SSE events; usage in seconds; routing of `/v1/audio/transcriptions` | core | Rust | new `tl-serve/src/audio.rs`; upgrades `tl-serve/src/http.rs` (L10.5) | L15.6, L10.5, lang.05 | `openai-subset.v2.yaml` 2.1.0 | inherited from `http.rs` (gw.04, load.01 over HTTP); gw.09, gw.10, load.03 reach it over HTTP | C (`audio.*` cases plus the official `openai` client `audio.transcriptions.create`), F (disconnect mid-stream frees the request within one window; oversize part gives 413 before buffering it all), U (SRT and VTT formatting of 0, 59.999, 3600.5 s) |
| L15.8 | C kernel backend for the engine: `tl-sys` bindings for `conv.h`, `image.h`, `audio.h`; `MmKernels` C implementation; `[engine.mm] kernels = "c"` with a startup probe that refuses stub units | **optional** (skippable; requires L13.8, L13.9, L14.6) | Rust | new `tl-sys/src/mm.rs`; upgrades `tl-engine/src/mm/kernels.rs` (L15.1) | L13.8, L13.9, L14.6, L15.1, L10.1 | `MmKernels`, `KernelBackend::C` | inherited from `kernels.rs` (L15.1's call sites: L15.3 vision tower, L15.6 Whisper stem and log-mel, L15.1 image path) | E (engine with `kernels = "c"` vs `kernels = "candle"`: u8 resize stage bit-exact, log-mel within 1e-5, tiny-idefics3 and SmolVLM logits within 1e-4, greedy output equal under the near-tie rule), U (startup with a stubbed unit fails with a message naming the module), F (RAII frees exactly once, counting allocator) |

### 3.5 Systems, data, operations, and practices

| ID | Title | Kind | Lang | Path | Prereqs | Interface or check | Call sites | Tests | Lights up at |
|---|---|---|---|---|---|---|---|---|---|
| lang.12 | Media formats from the bytes up: RIFF/WAV chunks, PNG chunks and CRC, JPEG markers and APP1/EXIF, base64 and `data:` URLs, `multipart/form-data` | practice | Py, Rust, Go | `primers/lang.12/` | lang.05 | a WAV chunk walker, a PNG chunk lister, a multipart encoder checked against fixtures | L13.6, L14.1, gw.09 (first use) | exercise checks | `{tinyllm} features` on a fixture WAV |
| lang.13 | candle: tensors, devices (cpu, metal), dtypes, `VarBuilder` over safetensors, `matmul`, `softmax_last_dim`, broadcasting, contiguity | practice | Rust | `primers/lang.13/` | lang.04 | a two-layer MLP in candle matching a numpy fixture | L15.1, L15.3, L15.6 | exercise checks | `cargo test` in the primer |
| ds.10 | Byte-budgeted LRU with pinned entries (`LruBytes<K, V: Weighted>`), O(1) touch, evict-until-fits skipping pins | build | Rust | `rust/crates/tl-ds/src/lru_bytes.rs` | ds.05, ds.07 | `insert(k, v) -> Result<Vec<(K,V)>, Full>`, `get`, `pin`, `unpin`, `bytes()` | L15.2 encoder cache | U, I (model-based vs a naive LRU: bytes <= budget; pinned never evicted; eviction order), Miri | `tl.engine.encoder_cache.evictions` under the ops.14 flood |
| data.10 | Image-text pair pipeline: fetch with license capture, decode and orient, size and aspect filters, EXIF stripping by re-encoding to PNG with the learner's writer, pHash (32x32 DCT, 8x8 low band, median) and dHash (9x8 gradients), near-dup by Hamming LSH + union-find, caption cleanup, CLIP-score filter, face redaction stage | build | Py | `corpus/image_pairs.py`, `corpus/png.py` | data.01, data.04, data.05, data.08, L13.5, L13.6, M12.3, M12.4, data.12 | `image_pairs(cfg) -> Stage`; `png_encode(u8_hwc) -> bytes` (stdlib zlib, no ancillary chunks) | data.13 | O (`imagehash` pHash/dHash on fixtures, exact bits), C (shard schema), I (output PNGs carry no `eXIf`, `tEXt`, `iTXt`, `zTXt`; planted near-dups (rescale, JPEG q60, 5% crop) clustered, planted flips not; deterministic hash across worker counts), F (corrupt image quarantined, not retried forever) | `{corpus} mm build --only images` |
| data.11 | Audio-transcript pipeline: WAV parse, mono 16 kHz, duration and clipping filters, energy VAD on log-mel, transcript normalization and PII scrub (data.05), exact dedup on PCM hash, near-dup by mel dHash windows + MinHash (data.04), speaker-disjoint splits via a keyed speaker hash, consent column | build | Py | `corpus/audio_pairs.py` | data.04, data.05, data.08, L14.1, L14.5 (normalizer), M12.6 | `audio_pairs(cfg) -> Stage` | data.13 | C (shard schema), I (no speaker key in two splits; no raw speaker id in any output; durations within bounds; deterministic), S (near-dup recall >= 0.95 on planted re-encodes and gain changes), U (transcript PII placeholders) | `{corpus} mm build --only audio` |
| data.12 | Face and biometric handling for images: Viola-Jones evaluation of OpenCV's pretrained frontal cascade (integral images, Haar features, stage thresholds, scale pyramid, neighbor grouping), redaction by triple box blur or drop per source policy | build | Py | `corpus/faces.py` | M12.2, ethics.02, data.05 | `detect_faces(gray, cascade, scale=1.1, min_neighbors=3, min_size=24) -> list[Box]`; `redact(img, boxes, mode)` | data.10 | O (OpenCV `detectMultiScale` boxes on fixtures: recall >= 0.9 at IoU 0.5, precision >= 0.8), I (redacted regions differ from the original by at least the blur bound; nothing outside boxes changes), U (integral image hand example) | `faces_redacted` counts in the manifest |
| data.13 | `MmCorpusBuild` durable workflow over the data.10 to data.12 stages, idempotent per shard, compensation on cancel | build | Go | `go/workflows/mm_corpus_build.go` | data.09, dur.06, dur.09 | Go workflow; activities `{corpus} run --stage {images,faces,audio,...}` | dur.13 (multimodal `ModelRelease` waits on its manifest and ledger) | F (`KillLoop` on worker and server: manifest hash equals a clean run; `effects` shows each shard written once) | `{ctl} data build-mm` |
| dur.13 | Multimodal `EvalSuite` and `ModelRelease`: suites `asr`, `vqa`, `caption`, `zeroshot`, `mm_bias`; gates on WER ceiling, accent-gap ceiling, consent for biometric sources, model-card multimodal sections | build | Go | upgrades `go/workflows/eval_suite.go` (ag.12), `go/workflows/model_release.go` (dur.12) | dur.12, ag.12, data.13, L14.5, ethics.07, ethics.08 | gate config in `formats/eval-spec.schema.json` (minor) | inherited (dur.12, ops.08, MS-C2 path) | C (replay of recorded multimodal `EvalSuite` and `ModelRelease` histories), F (a biometric source with `consent: none` fails before export, non-retryable; an accent gap over the ceiling blocks promotion) | `{ctl} release --model whisper-tiny --version v1` |
| gw.09 | Media validation and limits: content-part parsing, `data:` URL decode, header-only pixel checks, multipart under `MaxBytesReader`, WAV duration from the header, remote fetch behind allowlist + SSRF guard, modality policy rules, `tl-media:` and `X-TL-MM-Handle` stripping | build | Go | new `go/gateway/media/`; upgrades `go/gateway/server/` (craft.14) and `go/gateway/policy/` (gw.08) | gw.08, craft.14, ag.06 (SSRF rules, re-implemented), lang.12 | 2.5 `media` package; chain `requestid -> otel -> recover -> authn -> media -> policy -> ratelimit -> cache -> route -> proxy -> meter` | inherited (gw.01 chain); gw.10, gw.11 | C (`mm.errors.*`, `mm.remote_url`, `mm.internal_scheme`), E (`media.Hash` == Python and Rust on fixtures), F (`go test -fuzz` 60 s on data URL, multipart, WAV header parsers; gateway RSS stays under its limit during a bomb flood) | 413 on an oversized image |
| gw.10 | Multimodal metering and limits: Go `ImageTokens`, image tokens in TPM reservations, audio-seconds-per-minute bucket, ledger columns (`usage.v2.sql`), usage API `group_by=modality` | build | Go | upgrades `go/gateway/limit/` (gw.03), `go/gateway/ledger/` (craft.14) | gw.09, gw.07, gw.03, L13.4 (formula, re-implemented) | `Cost{AudioMs}`; ledger rows | inherited (gw.01 chain, ag.04 `query_usage`) | E (`ImageTokens` == the Python fixture table for 40 sizes and 3 details), I (ledger sums == engine-reported image tokens and audio seconds over 1k mixed requests), U (audio bucket under a fake clock) | `{ctl} usage --tenant acme --group-by modality` |
| gw.11 | Pool routing and EPD orchestration: capability pools (`text`, `vlm`, `asr`, `encode`), media-hash affinity bounded by encoder queue depth, Encode then prefill/decode, fallback to local encode before first byte, `Release` on abort | build | Go | upgrades `go/gateway/route/` (gw.05) | gw.09, L15.5, ds.09 | `Router.Route` with `EpdPlan` | inherited (gw.01 chain, dur.12 canary) | E (EPD == unified output, greedy), F (encode pod killed before first byte: retried or local fallback, never a spliced stream; target miss gives one retry), C (VLM model never routed to the text pool; ASR only to `asr`) | `route_policy=affinity` with two encoders |
| load.03 | Multimodal workload mix: `workload.toml` mixes of text, image (size classes, repeat ratio for cache hits), and audio (durations); multipart sender; RTF and encode-time reporting | build | Go | upgrades `go/loadgen` (load.01) | load.01, gw.09 | `Mix{Next() Request}`; report adds `encode_ms`, `rtf`, `image_tokens` | inherited (load.02, drills, MS-prod) | U (mix proportions chi-square at a fixed seed), C (report schema minor), F (no coordinated omission with slow ASR responses) | `{loadgen} --workload mm-mix.toml` |
| dep.08 | Encoder and ASR pools as their own Deployments: values files for `vlm`, `encode`, `asr`; KEDA `ScaledObject`s on encoder queue depth and queued audio seconds; PDBs; resource requests sized from calibration | practice | YAML | `deploy/helm/<system>-engine/values-{vlm,encode,asr}.yaml` | dep.06, dep.03, L15.5 | dep.03 policy over `helm template`; `values.schema.json` minor; a burst of 50 image requests scales `encode` up and back down on kind; ASR scaling independent of encode | | artifact checks | `kubectl get hpa` during a burst |
| obs.06 | Multimodal SLOs and dashboards: encode time, image TTFT, ASR RTF burn-rate rules (`prod` and `drill` profiles), `multimodal.json` panels (encode heatmap by modality, encoder cache hit ratio, queue depths per pool, RTF heatmap, media rejections by reason, EPD fallbacks, image tokens per tenant) | practice | YAML | `deploy/observability/{rules/multimodal.yaml,dashboards/multimodal.json}` | obs.03, obs.04, L15.3, L15.6 | `promtool test rules` with synthetic series; every panel query names a contract metric; one trace spans `gateway.media` to `engine.decode` through `encoder.transfer` | | artifact checks | Grafana during a mixed load run |
| ops.13 | Drill `encoder-pool-loss`: `scale-zero` on the encode Deployment under mixed load | drill | ops | `docs/postmortems/` | MS-L15, dep.08, obs.06 | detect `EncodeLatencyBurnFast` or `ImageTTFTBurnFast` within 300 s; pass: text TTFT p95 within SLO throughout, image requests served by local fallback or 503 `no_capacity` with `Retry-After` (never hangs past the deadline), encode pool restored, postmortem | | drill | `ss drill start encoder-pool-loss` |
| ops.14 | Drill `image-flood`: one tenant floods oversized images, pixel bombs, and 20 MB data URLs (`tenant-flood` injector with the `media-flood` profile) | drill | ops | `docs/postmortems/` | gw.09, gw.10, obs.06 | detect via `tl.gateway.media.rejections` panel and image TTFT burn; pass: bombs rejected at the gateway before decode, gateway memory below its limit, other tenants' TTFT within SLO after per-tenant image limits are tuned, postmortem | | drill | `ss drill start image-flood` |
| ethics.07 | Multimodal bias evals: per-accent WER with two-sample bootstrap CIs on gaps vs overall (`cv-accent-mini`), zero-shot accuracy disparity across image source domains, counterfactual caption probes (occupation x gendered term) on fixed images | build | Py | `tinyllm/eval/mm_bias.py` | L14.5, L13.5, ethics.04, M07.4 | `accent_wer_gaps(...) -> list[GroupGap]`, `zeroshot_disparity(...)`, `caption_counterfactuals(...)` | dur.13 (`mm_bias` suite rows gate release) | O (gap values on fixture predictions), S (a planted 5-point gap detected, a null gap's CI covers 0 at the nominal rate over 200 seeds) | `{tinyllm} eval --suite mm_bias` |
| ethics.08 | Consent and biometric data: `CONSENT_POLICY.md` (voice and face are biometric; no speaker identification, no face recognition; Common Voice terms), ledger `consent`/`biometric`/`prohibited_uses` filled for every multimodal source, model card multimodal sections with real numbers | practice | docs | `docs/CONSENT_POLICY.md`, `DATASHEET.md`, `MODEL_CARD.md` | ethics.02, ethics.03, data.08 | sections lint; ledger validates; model-card WER per accent equals `evals/<suite>/summary.json` within rounding | | artifact checks | `ModelRelease` gate |
| craft.24 | Untrusted media: threat model update (new boundary: media bytes from clients and remote hosts), fuzz targets for the Rust WAV and image-header paths (`cargo fuzz`, nightly only) and the Go parsers, decompression-bomb tests | practice | Rust, Go, docs | `docs/THREAT_MODEL.md`, `rust/crates/tl-media/fuzz/`, `go/gateway/media/*_test.go` | craft.17, gw.09, L15.1 | STRIDE rows for media paths; fuzzers run 60 s each without findings; a seeded crash mutant (`s01`, unchecked chunk length) is found by the learner's fuzzer within budget | | artifact checks | `go test -fuzz` |
| review.04 | Design review: the multimodal platform (EPD trade-offs, encoder cache sizing, candle vs own kernels, determinism rule E3) | proof (rubric) | docs | `docs/reviews/multimodal.md` | MS-L15 | `rubrics/design-review.md` | | self | P12 gate |

### 3.6 Side quests

| ID | Content |
|---|---|
| `sq.audio-llm` | audio encoder (the learner's Whisper encoder) + frame stacking (k = 4) + MLP projector into SmolLM2-135M, trained projector-only on a synthetic spoken-digits task; served through `input_audio` parts (Ultravox / Qwen2-Audio pattern) |
| `sq.streaming-asr` | chunked streaming over the transcription endpoint with the local-agreement policy (whisper-streaming), latency vs WER table |
| `sq.vision-agent` | a `describe_image` tool in the agent SDK that calls the gateway with an `image_url` part; MS-agent style eval on fixture screenshots |
| `sq.jpeg-decoder` | baseline JPEG decoder (Huffman, dequantization, IDCT via M12.4's DCT, chroma upsampling) proven bit-exact against `zune-jpeg` on fixtures |
| `sq.video-frames` | frame sampling and M-RoPE temporal positions for short clips |
| `sq.cnns` (existing) | unchanged; now cites L13.1 as the course version of conv layers |
| `sq.cuda-kernels` (existing) | adds an im2col conv variant from `conv2d.cu`; still never in CI |

### 3.7 Counts

| Layer | Core | Optional | Python | Rust | Go | Other |
|---|---|---|---|---|---|---|
| M12 | 6 | | 6 | | | |
| Solve | 1 set (S-M12), 72 items, 4 rubric | | | | | SymPy |
| L13 | 7 | 2 (L13.8, L13.9, C) | 7 (L13.6 also Rust) | 1 (L13.6) | | 2 C (optional) |
| L14 | 5 | 1 (L14.6, C) | 5 | | | 1 C (optional) |
| L15 | 7 | 1 (L15.8) | | 7 core + 1 optional | | |
| ds, data, dur, gw, load | 10 | | 3 (data.10 to data.12) | 1 (ds.10) | 6 (data.13, dur.13, gw.09 to gw.11, load.03) | |
| dep, obs, ops | 4 | | | | | YAML, drills |
| lang, ethics, craft, review | 6 | | 1 (ethics.07) | | | practice, docs |
| Side quests | | 5 new ids | | | | |
| **Total** | **45 core modules + S-M12** | **4 optional** | 22 | 9 core + 1 optional | 6 | 3 optional C |

C units in Pass 12 are optional only (E1): L13.8, L13.9, L14.6, wired into the engine by optional L15.8. No core module lists them in `deps`, so `ss check` on every core module and every core milestone runs without them; their stub objects keep `libtinyllm` loadable.

---

## 4. Solve set

`S-M12` (Pass 12, unsplit; `course/solve/S-M12/`). Answer kinds per DESIGN 5.5.

| Topic | Items | Example item |
|---|---|---|
| Convolution output sizes, padding, stride, dilation, receptive fields | 8 | output length of Whisper's stem for 3000 frames (`number`) |
| Correlation vs convolution, Toeplitz form, adjoint and gradient of a conv | 8 | the 4x9 matrix of a 1D conv, `n = 6`, `k = 3`, stride 1 (`matrix`) |
| Conv and ViT FLOP and parameter counts (im2col sizes, patch tokens, pixel shuffle dims) | 6 | SmolVLM tokens per 512 tile after shuffle with r = 4 (`number`) |
| Layouts and strides | 4 | NHWC strides in bytes for `[2, 224, 224, 3]` f32 (`vector`) |
| Sampling, Nyquist, aliasing (1D and 2D) | 8 | the alias frequency of 9 kHz sampled at 16 kHz (`number`) |
| Interpolation kernels, antialias support, bicubic `a` | 6 | bicubic weight at distance 1.5 with a = -0.5 (`number`, exact) |
| DFT properties, convolution theorem, Parseval | 8 | DFT of a shifted impulse (`expr`) |
| FFT operation counts, mixed radix factorization of 400 | 4 | complex multiplies for radix-2 n = 512 (`number`) |
| Windows, COLA, STFT frame counts and bin frequencies | 6 | bin index of 1 kHz at n_fft 400, 16 kHz (`number`) |
| Mel, log, decibels, Whisper scaling | 6 | Slaney mel of 1000 Hz (`number`); dB of a 1e-6 power ratio (`number`) |
| Contrastive losses: InfoNCE as cross-entropy, the `log N` MI bound, SigLIP gradient in t and b, temperature effect | 6 (2 proofs) | prove InfoNCE <= log N - I(X;Y) lower-bounds MI (`proof`) |
| Edit distance and WER: alignment by DP, WER > 1 cases, CER vs WER | 2 (1 proof) | WER of a fixture pair (`number`); prove WER is not a metric (`proof`) |
| **Total** | **72 (4 rubric)** | |

The DESIGN 4.2 totals become 13 sets in 21 parts, 731 items (68 rubric).

---

## 5. Milestones

All specs are `course/milestones/<MS-ID>.toml` in the DESIGN 5.7 format. New matcher (harness work, section 7): **`npz-close`**: compares named arrays in a produced `.npz` with a fixture under `{atol, rtol, max_lsb, top_k_equal}`. New placeholder: `{emb_port}`. Every step lists its CI tier; PR CI runs `--smoke` (tiny assets), nightly runs real weights.

### 5.1 MS-L13 (vision)

`requires = ["L13.1", ..., "L13.7", "M12.1", "M12.2", "M12.3"]`, `pass = 12`.

| # | Step | Learner command (role placeholders per DESIGN 2.16) | Assertion | CI |
|---|---|---|---|---|
| 1 | LeNet learns | `{tinyllm} train lenet --data {fixture:digits/digits.npz} --steps 400 --seed 0` | `json-last-line`: `test_acc >= calibrated` | pr, smoke |
| 2 | Contrastive learns | `{tinyllm} train clip --data {asset:mm-shapes} --cfg {fixture:configs/clip-tiny.json} --loss siglip --steps 600 --seed 0 --out {out}/clip` then `{tinyllm} eval zeroshot --model {out}/clip --data {asset:mm-shapes/test.jsonl}` | `zeroshot_acc >= calibrated`; rerun with `--loss clip` reports both | pr, smoke |
| 3 | Preprocessing to spec | `{tinyllm} preprocess image --processor {asset:tiny-mm-models/tiny-idefics3} --images {fixture:L13.6/images} --out {out}/pv.npz` | `npz-close` vs `course/fixtures/L13.6/idefics3_pixel_values.npz`: `max_lsb = 1` before normalize, `atol = 0.0079` after; token counts exact | pr, smoke |
| 4 | Tiny VLM parity | `{tinyllm} logits --model {asset:tiny-mm-models/tiny-idefics3} --mm-prompts {fixture:L13.7/tiny_prompts.jsonl} --pixel-values {fixture:L13.7/tiny_pixel_values.npz} --out {out}/logits.npz` (repeat for `tiny-llava-next`, `tiny-qwen2vl`) | `npz-close` atol 1e-5 | pr, smoke |
| 5 | Pull SmolVLM | `{tinyllm} pull HuggingFaceTB/SmolVLM-256M-Instruct --revision {asset-rev:smolvlm-256m-instruct}` | `file-produced`: safetensors sha256 equals `ASSETS.tsv` | nightly |
| 6 | SmolVLM logits (model parity) | `{tinyllm} logits --model {models}/SmolVLM-256M-Instruct --mm-prompts {fixture:smolvlm-parity/prompts.jsonl} --pixel-values {fixture:smolvlm-parity/pixel_values.npz} --out {out}/logits.npz` | `npz-close`: max abs <= 1e-3, `top_k_equal = 5` | nightly |
| 7 | SmolVLM end to end | `{tinyllm} vlm generate --model {models}/SmolVLM-256M-Instruct --image {fixture:smolvlm-parity/img-{i}.png} --prompt-file {fixture:smolvlm-parity/prompt-{i}.txt} --greedy --max-tokens 32`, matrix `i = 0..3` | `tokens-equal` vs `course/fixtures/smolvlm-parity/greedy_{i}.json` (margin-filtered pairs; own preprocessing) | nightly |
| 8 | Freezing schedule | `{tinyllm} train vlm --cfg {fixture:configs/vlm-tiny.json} --data {asset:mm-shapes} --stage align --steps 200 --out {out}/vlm` then `--stage sft --steps 300 --resume {out}/vlm` then `{tinyllm} eval vqa --model {out}/vlm --data {asset:mm-shapes/vqa-test.jsonl}` | stage 1 last line `frozen_params_changed == 0`; `vqa_acc >= calibrated` | pr (`--smoke`: 60 + 90 steps, looser threshold), nightly full |
| 9 | Param count | `{tinyllm} info --model {models}/SmolVLM-256M-Instruct` | `params` == HF `num_parameters()` from the fixture | nightly |

### 5.2 MS-L14 (speech)

`requires = ["L14.1", ..., "L14.5", "M12.4", "M12.5", "M12.6"]`.

| # | Step | Learner command | Assertion | CI |
|---|---|---|---|---|
| 1 | Log-mel exact | `{tinyllm} features --audio {fixture:L14.1/clips} --kind whisper-log-mel --n-mels {n} --out {out}/mel.npz`, matrix `n = [80, 128]` | `npz-close` atol 1e-4 vs HF `WhisperFeatureExtractor` | pr, smoke |
| 2 | Tiny Whisper parity | `{tinyllm} logits --model {asset:tiny-mm-models/tiny-whisper} --audio-features {fixture:L14.3/tiny_features.npz} --decoder-prompts {fixture:L14.3/tiny_prompts.jsonl} --out {out}/logits.npz` | `npz-close` atol 1e-5 | pr, smoke |
| 3 | Decoding on recorded logits | `{tinyllm} transcribe --logits-trace {fixture:L14.4/traces.npz} --opts {fixture:L14.4/opts.json}` | `tokens-equal` per trace (greedy, beam 5, fallback, long-form seek) vs `course/fixtures/L14.4/expected.json` | pr, smoke |
| 4 | Pull whisper-tiny | `{tinyllm} pull openai/whisper-tiny --revision {asset-rev:whisper-tiny}` | sha256 | nightly |
| 5 | Greedy with timestamps | `{tinyllm} transcribe --model {models}/whisper-tiny --audio {fixture:whisper-parity/clip-{i}.wav} --greedy --timestamps --language en`, matrix `i = 0..5` | `tokens-equal` (timestamp tokens included) vs HF `generate(return_timestamps=True)` ids, margin-filtered clips | nightly |
| 6 | Language detection | `{tinyllm} transcribe --model {models}/whisper-tiny --audio {fixture:whisper-parity/lang-{l}.wav} --detect-language` | `json-last-line`: `language == "{l}"` for 3 fixture languages | nightly |
| 7 | Long-form with fallback | `{tinyllm} transcribe --model {models}/whisper-tiny --audio {asset:asr-mini/longform-90s.wav} --beam 5 --out {out}/segments.json` then `{tinyllm} eval asr --pred {out}/segments.json --ref {asset:asr-mini/longform-90s.txt}` | `json-schema` on segments; `wer <= ref + 0.01` (ref from the oracle run of `openai-whisper`, same normalizer) | nightly |
| 8 | Corpus WER | `{tinyllm} eval asr --model {models}/whisper-tiny --data {asset:asr-mini/manifest.jsonl}` | `wer <= ref + 0.01`; `wer_ci` present | nightly |

### 5.3 MS-mm-corpus

| # | Step | Learner command | Assertion | CI |
|---|---|---|---|---|
| 1 | Build | `{corpus} mm build --config {fixture:mm-corpus/small-mm.toml} --workers {w}`, matrix `w = [1, 4]` | `file-produced`: image and audio shards and `_MANIFEST.json` pass `conformance/formats/`; output hash equal across worker counts | pr, smoke |
| 2 | Dedup and privacy | `{corpus} mm audit --manifest {out:build}/_MANIFEST.json` | `json-last-line`: `near_dup_recall >= 0.95`, `false_drops == 0`, `exif_chunks == 0`, `faces_redacted >= {fixture count}`, `speakers_crossing_splits == 0`, `raw_speaker_ids == 0` | pr, smoke |
| 3 | Ledger | `{corpus} ledger verify` | exit 0; every multimodal row has `modality`, `consent`, `biometric` | pr, smoke |
| 4 | Durable | `{ctl} data build-mm --dataset mm-small --version v1` under `KillLoop` | manifest hash equals step 1 | nightly |

### 5.4 MS-L15 (multimodal serving, local processes)

Services from `system.toml`: `engine` instances `vlm` (tiny-idefics3 in PR, SmolVLM nightly), `encode` (`role = encode`), `asr` (tiny-whisper in PR, whisper-tiny nightly), `text` (the Pass 7 engine), and `gateway`; the runner allocates `{port}`, `{grpc_port}`, `{emb_port}`.

| # | Step | Command | Assertion | CI |
|---|---|---|---|---|
| 1 | Engine conformance | `ss conform openapi:v2 --target engine --cases 'mm.*,audio.*'` against `vlm` and `asr` | 100% | pr, smoke |
| 2 | Engine equals Python | `{ctl} chat --base {vlm.api_base} --image {fixture:L13.7/img-0.png} --prompt-file ... --greedy --max-tokens 32 --json` | `tokens-equal` vs `{tinyllm} vlm generate` ids for the same inputs (near-tie rule) | pr, smoke |
| 3 | Cache correctness | `{ctl} chat ... --image A` twice, then `--image B` with the same text | second A: `cached_tokens >= image_tokens`; B: no hit past the image span start; `tl_engine_encoder_cache_hits_total` increments by 1 | pr, smoke |
| 4 | Encoder budget | `{loadgen} --target {vlm.api_base} --workload {fixture:load/images-burst.toml} --duration 30s` with `TL_ENGINE__MM__MAX_ENCODER_TOKENS_PER_STEP=128` | error rate 0; text requests in the mix keep `tpot_p95` within 1.2x of the text-only run (a `perf` step, `ci = "local"`); KV and pins at baseline afterwards | pr (functional), local (perf) |
| 5 | EPD | `{gateway}` with `epd = { encode_pool = "encode" }`; `{ctl} chat --base {gateway.api_base} ... --greedy --json` | ids equal to step 2; one trace has `gateway.media`, `Encode`, `encoder.transfer`, `engine.prefill` | pr, smoke |
| 6 | EPD fault | kill `encode` mid-run (runner `signal = "KILL"`) | in-flight image requests complete via local fallback or fail before first byte with 503; `tl_gateway_epd_fallbacks_total > 0`; no spliced streams | pr |
| 7 | Transcription | `{ctl} transcribe --base {gateway.api_base} --file {fixture:L14.1/clips/en-0.wav} --format verbose_json` | text equals `{tinyllm} transcribe --greedy` text; `json-schema` verbose; `--format srt` parses | pr, smoke |
| 8 | Mixed load | `{loadgen} --target {gateway.api_base} --workload {fixture:load/mm-mix.toml} --rate 4 --duration 60s` | `error_rate == 0`; report has `encode_ms`, `rtf`, `image_tokens`; ledger image tokens and audio seconds equal engine usage | pr (30 s), nightly (60 s) |
| 9 | Real weights | steps 2, 5, 7 on SmolVLM-256M and whisper-tiny | as above | nightly |

### 5.5 MS-multimodal-prod (kind; production component of the P12 gate)

| # | Step | Assertion | CI |
|---|---|---|---|
| 1 | Deploy | `helm upgrade --install` of gateway, `engine-text`, `engine-vlm`, `engine-encode` (2), `engine-asr` from the learner's values files; rollouts healthy; dep.03 policy passes on every release | kind (nightly) |
| 2 | Conformance through the NodePort | `ss conform openapi:v2 --target gateway --base {deploy.gateway_url} --cases 'mm.*,audio.*'` and the full v2 smoke set | kind |
| 3 | Traces | `trace` matcher: `gateway.media` > `gateway.route` > `Encode` > `encoder.transfer` > `engine.encode`/`engine.prefill` > `engine.decode` for one image request; `asr.window` spans for one transcription | kind |
| 4 | Metrics and rules | `promscrape`: every 2.11 metric present with contract names; `multimodal.yaml` rules loaded with the required alert names | kind |
| 5 | SLOs under load | `{loadgen} --target {deploy.gateway_url} --workload mm-mix.toml --rate {calibrated}` for 10 min; `promql` with `hold_s = 300`: encode p95, image TTFT p95, ASR RTF p95, text TTFT p95 within their SLOs | kind |
| 6 | Separate autoscaling | image burst: `encode` replicas rise to >= 2 and return to min within 5 min after; `asr` replicas unchanged; then an audio burst scales `asr` only | kind |
| 7 | Drills | `ss drill start encoder-pool-loss --seed {n}` and `image-flood` with the scripted responder: detected and resolved | kind (rotation) |
| 8 | Release gates | `{ctl} release --model smolvlm-256m --version v1` and `--model whisper-tiny` through `ModelRelease`: multimodal suites, accent-gap ceiling, consent gate, model card sections | kind |
| 9 | Tag | `git-log` and a `v1.1.0` tag; changelog lists API 2.1.0 and proto minors | local |

### 5.6 MS-mm-kernels (optional, not part of the gate)

`requires = ["L13.8", "L13.9", "L14.6", "L15.8"]`, `optional = true`. Recorded in `STATUS.md` when earned; `MS-P12` never requires it.

| # | Step | Command | Assertion | CI |
|---|---|---|---|---|
| 1 | Kernel parity | `ss parity conv2d conv1d patch_embed resize stft.mel` | C vs the learner's Python and vs oracle fixtures within the 3.4 bounds | pr, smoke |
| 2 | Engine switch, vision | engine `vlm` started twice from the same template with `TL_ENGINE__MM__KERNELS=candle` and `=c`; `{ctl} chat --image {fixture:L13.7/img-0.png} ... --greedy --json` against each | `tokens-equal` between the two runs and against `{tinyllm} vlm generate` (near-tie rule) | pr, smoke |
| 3 | Engine switch, speech | `{ctl} transcribe --file {fixture:L14.1/clips/en-0.wav}` against `asr` with `kernels = "c"` | text equals the `candle` run | pr, smoke |
| 4 | Stub refusal | start `vlm` with `kernels = "c"` and `--ref-deps` disabled while one unit is a stub (scratch copy) | the engine exits non-zero naming the unstarted module | pr |
| 5 | Speed | `{tinyllm} bench mm --kernels candle,c` (cpu) | reports `speedup_c` for preprocess and patch embed (a `perf` step, `ci = "local"`, no bound) | local |

**MS-P12** (pass gate) = MS-L13 + MS-L14 + MS-mm-corpus + MS-L15 + MS-multimodal-prod, and reruns the smoke steps of MS-P0 to MS-P11 (text paths, tracer bigram, durable kill loop, agent with `faketool`). Milestone catalog row for DESIGN 5.7:

| Milestone | Pass | Pass condition | CI |
|---|---|---|---|
| MS-L13, MS-L14 | P12 | conv and contrastive learning; preprocessing within 1 LSB; tiny VLM and tiny Whisper logits; SmolVLM logits 1e-3 and greedy exact; log-mel 1e-4; whisper-tiny greedy with timestamps exact; WER within 1 point of the oracle | pr (tiny, `--smoke`), nightly (real weights) |
| MS-mm-corpus | P12 | shards, dedup, EXIF, faces, speaker-disjoint splits, consent ledger; durable under kills | pr, nightly |
| MS-L15 | P12 | engine conformance, engine == Python, cache correctness, encoder budget, EPD and its fault, transcription, mixed load and metering | pr (tiny), nightly (real), local (perf) |
| MS-multimodal-prod | P12 | kind deploy with separate pools and autoscaling, traces, SLOs, drills ops.13 and ops.14, release gates, `v1.1.0` | nightly (kind) |
| MS-mm-kernels | P12 (optional) | C conv, resize, STFT + mel parity; engine output identical with `kernels = "c"` and `"candle"`; stub refusal | pr (when started), local (perf) |

---

## 6. Fixtures, assets, oracle generators, licenses

### 6.1 Committed fixtures (`course/fixtures/`)

| Fixture | Generator | Contents | Size |
|---|---|---|---|
| `M12/*` | `gen_signal_fixtures.py` | conv fwd/bwd and unfold cases, PIL resize outputs on 6 small images per kernel, torchaudio resample, numpy FFT/DCT, torch STFT, librosa mel filters | about 1.5 MiB |
| `L13.6/images/`, `L13.6/*_pixel_values.npz` | `gen_image_fixtures.py` | 12 small PNG/JPEG (EXIF orientations 1 to 8, CMYK, 16-bit, truncated, pixel-bomb header), CLIP/SigLIP/Idefics3 processor outputs, 40-size token-count table | about 1.5 MiB |
| `L13.2`, `L13.4`, `L13.5`, `L13.7` | `gen_mrope_fixtures.py`, `gen_tiling_fixtures.py`, `gen_contrastive_fixtures.py` | M-RoPE position ids, tile grids and prompt strings, loss values, tiny prompts and pixel values | about 0.5 MiB |
| `L14.1/clips/`, `L14.4/traces.npz`, `whisper-parity/` (ids only) | `gen_whisper_parity.py` | 6 clips of at most 4 s at 16 kHz s16 (public-domain speech from LibriVox-derived LibriSpeech is CC BY 4.0, so these use CC0 Common Voice clips), log-mel for 2 clips (the rest regenerated within the test from WAV), recorded decoder logits for 8 traces as top-64 sparse rows, expected ids | about 1.5 MiB |
| `smolvlm-parity/` | `gen_smolvlm_parity.py` | 4 CC0 images (256 px), prompts, `pixel_values` for the global tile only (f16 stored, compared after upcast), last-position logits top-256 sparse, greedy ids | about 1 MiB |
| `mm-corpus/small-mm.toml`, `load/*.toml`, `configs/{clip-tiny,vlm-tiny}.json` | hand-written | configs | < 0.1 MiB |
| `L13.8`, `L13.9`, `L14.6` | reuse the `M12/*` goldens; `gen_signal_fixtures.py` adds batch-invariance and groups cases | conv shapes incl. SmolVLM patch embed (3x512x512, p 16) slices, Whisper stem on 1 s | about 0.3 MiB |
| **B14 total** | | | **about 6.3 MiB** |

DESIGN 9 allocates the full 50 MiB to B1 to B13. B14 needs the cap raised to **56 MiB** (Q1), or 6 MiB moved from committed `ops-torch` into a fetched asset.

### 6.2 Fetched assets (`course/fixtures/ASSETS.tsv`)

| Asset | Source at pinned revision | Size | License (to verify, Q2) |
|---|---|---|---|
| `smolvlm-256m-instruct` | `HuggingFaceTB/SmolVLM-256M-Instruct` | about 0.5 GB | Apache-2.0 per the model card; verify the card and the SigLIP and SmolLM2 component terms |
| `whisper-tiny` | `openai/whisper-tiny` | about 150 MB | openai/whisper code and weights MIT; the HF card lists Apache-2.0; record which governs |
| `tiny-mm-models` | `course/oracle/gen_tiny_mm_models.py`, published as a supersource release asset | about 8 MB | course (Apache-2.0); random weights |
| `mm-shapes` | `course/oracle/gen_mm_shapes.py` (procedural, PCG32-seeded) | about 6 MB | CC0 (course-generated) |
| `mm-pairs-mini` | 200 pairs sampled from PD12M plus planted variants (rescale, JPEG q60, 5% crop, flip, GPS EXIF) and 20 public-domain portraits | about 8 MB | images public domain or CC0 per item; PD12M metadata CDLA-Permissive-2.0; verify per-item |
| `asr-mini` | 40 LibriSpeech test-clean utterances as 16 kHz WAV plus one 90 s concatenation with gaps | about 10 MB | CC BY 4.0, attribution recorded in the ledger and model card |
| `cv-accent-mini` | Mozilla Common Voice English, 5 accent groups x 30 validated clips | about 12 MB | CC0; Common Voice terms forbid re-identifying speakers (ethics.08) |
| `haarcascade-frontalface` | OpenCV `data/haarcascades/haarcascade_frontalface_default.xml` | 0.9 MB | Intel License Agreement (BSD-3-style); verify |
| `siglip-base-patch16-224` (optional, nightly) | `google/siglip-base-patch16-224` | about 0.8 GB | Apache-2.0; only for an optional real zero-shot step |

### 6.3 Oracle generators (`course/oracle/`, maintainer only)

New pinned dependencies: `torchaudio`, `torchvision`, `pillow`, `librosa`, `openai-whisper`, `jiwer`, `imagehash`, `opencv-python-headless`, `soundfile`.

| Script | Produces |
|---|---|
| `gen_signal_fixtures.py` | M12 goldens (torch conv/unfold/fold, PIL resize, torchvision antialias, torchaudio resample, numpy/scipy FFT and DCT, torch STFT, librosa mel and dB) |
| `gen_image_fixtures.py` | processor outputs for CLIP, SigLIP, Idefics3, LLaVA-NeXT; EXIF fixture JPEGs; token-count table; bomb headers |
| `gen_tiling_fixtures.py`, `gen_mrope_fixtures.py`, `gen_contrastive_fixtures.py` | L13.2, L13.4, L13.5 goldens |
| `gen_tiny_mm_models.py` | `tiny-mm-models`: random tiny `ViTModel`, `CLIPModel`, `SiglipModel`, `Idefics3ForConditionalGeneration`, `LlavaForConditionalGeneration`, `LlavaNextForConditionalGeneration`, `Qwen2VLForConditionalGeneration`, `WhisperForConditionalGeneration` with configs, safetensors, logits |
| `gen_smolvlm_parity.py` | margin-filtered (image, prompt) pairs (top-2 margin >= 1e-3 at every greedy step), pixel values, logits, greedy ids, param count |
| `gen_whisper_parity.py` | `WhisperFeatureExtractor` features; HF `generate` ids with timestamps; `openai-whisper` `transcribe` segments for long-form and fallback; recorded logits traces; margin-filtered clips |
| `gen_mm_shapes.py` | procedural images (32x32 and 128x128), captions, VQA questions, zero-shot labels; seeded by `spec/pcg32.md` so the learner can audit it |
| `gen_mm_pairs.py` | `mm-pairs-mini` sampling and planted variants; `imagehash` and OpenCV cascade oracle labels |
| `gen_asr_eval.py` | `asr-mini`, `cv-accent-mini` manifests; `jiwer` WER and per-group oracle values |

`oracle-drift` (weekly) covers all of them; integer outputs (token ids, hash bits, tile grids, token counts) compare exactly, floats within the DESIGN 5.11 tolerances.

---

## 7. B14 batch entry (DESIGN section 9 format)

| Batch | Harness work | Modules (reference + tests + mutants + chapter + registry row) | Paths, new topic READMEs, committed fixture budget | Verification |
|---|---|---|---|---|
| **B14 Multimodal: vision and speech** | registry: pass 12 and the `L15` part (`L1[0-9]` already matches the id regex; no regex change); Rust farm builds with `candle-core`/`candle-nn` (cpu feature set in CI, `metal` only when the learner enables it locally), shared warm `CARGO_TARGET_DIR` sized for candle; `allowed-deps.toml` additions and the forbidden list (`candle-transformers`, `candle-examples`); new crate `tl-media` in the contract pre-check and the PyO3 build (`tinyllm_rs.decode_image`); matcher `npz-close`; placeholder `{emb_port}` and runner `signal` steps for EPD faults; `ss conform --cases` filter and multipart request support in the OpenAPI runner (`openai-core` validation of `multipart/form-data`, official `openai` client `audio.transcriptions`); conformance suite `encoder-grpc/`; parity suites `image.decode`, `image.preprocess`, `image.tokens`, `audio.logmel`, `asr.decode`, `media.hash`, `embed.wire.v1`; generated code for `tl/encoder/v1` in `tl-proto` and `contracts/go/gen`; C overlay and stub objects for the optional `conv.h`, `image.h`, `audio.h` units (two builds, sanitizers, counting allocator, as in B8); the `tl-sys` crate root contract declares `mm`, stubbed until L15.8 starts; ctypes parity suites `conv2d`, `conv1d`, `patch_embed`, `resize`, `stft.mel`; registry support for `optional = true` milestones; drills `encoder-pool-loss` (`scale-zero`) and `image-flood` (`tenant-flood` with a `media-flood` loadgen profile) with scripted responders; `hf-real` extended with `smolvlm-256m-instruct` and `whisper-tiny` (cached by sha); `kind-e2e` rotation adds ops.13, ops.14 and the `MS-multimodal-prod` steps; fixture cap raised per Q1 | lang.12, lang.13; M12.1 to M12.6, S-M12; L13.1 to L13.7; L14.1 to L14.5; L15.1 to L15.7; optional L13.8, L13.9, L14.6, L15.8 and MS-mm-kernels (authored last; the core verification never depends on them); ds.10; data.10 to data.13; dur.13; gw.09 to gw.11; load.03; dep.08; obs.06; ops.13, ops.14; ethics.07, ethics.08; craft.24; review.04; MS-L13, MS-L14, MS-mm-corpus, MS-L15, MS-multimodal-prod, MS-P12; side quests sq.audio-llm, sq.streaming-asr, sq.vision-agent, sq.jpeg-decoder, sq.video-frames | `paths/course-p12-multimodal/` (README, `path.tsv`, `milestone.md`); `paths/course/path.tsv` gains `@course-p12-multimodal`; new topic README `math/12-signals-and-images/`; new part READMEs `ml/08-tinyllm/{p13-vision,p14-speech,p15-multimodal-serving}/`; chapter indexes in `data-engineering/05-corpus-pipeline/`, `ai-platform-engineering/12-gateway/`, `responsible-ai/{02-privacy-and-pii,04-bias-and-safety-evals}/`; `ml/05-foundation-models/README.md` section 4 gains pointers to L13.5, L13.7, L14; fixtures 6 MiB | `ss lint && ss learn --verify && ss verify course --changed origin/main && ss parity && ss conform openapi:v2 --target engine --cases 'mm.*,audio.*' && ss milestone MS-P12 --smoke` (nightly: `hf-real` for MS-L13/MS-L14/MS-L15 real-weight steps, `kind-e2e` for MS-multimodal-prod) |

**Learner time.** About **15 weeks** at 10 to 12 h per week: primers 0.5, M12 and S-M12 2, L13 4, L14 3, data and ethics 1.5, L15 3, gateway, load, deploy, observability, drills, review 1. The course total moves from about 67 to about 82 weeks. The optional kernel track (L13.8, L13.9, L14.6, L15.8) adds about 2 weeks for learners who take it. Planning estimate only (DESIGN Q12).

**Authoring order inside B14**: M12 and S-M12, then L13.1 to L13.7 and MS-L13 (Python only, unblocks fixtures), L14 and MS-L14, data and ethics with MS-mm-corpus, then lang.13, ds.10, L15 with MS-L15, then gw, load, dep, obs, ops, craft, review with MS-multimodal-prod, and last the optional kernels with MS-mm-kernels. Each module follows the per-module authoring rule of DESIGN 9 (reference passes, stub fails, mutants killed, chapter lints, every authored `used_by` imports it).

**Rows to update when merged**: DESIGN 2.1, 2.3, 2.6, 2.7, 2.9, 2.10 (`asr-decoding.md`), 2.11, 2.12, 2.13, 2.15, 3.3, 4.2 totals, 4.7, 4.8, 5.7 catalog, 5.8 parity table, 5.14 CI jobs, 7.1 tree, 7.2, 7.3 (new rows: `Preprocessor` behind `preprocess_rev`; API v2.1; engine roles add `encode`), 7.4, 7.5, 9 (budget sum), 10.

---

## 8. Open questions

| # | Question | Current default |
|---|---|---|
| Q1 | The 50 MiB committed-fixture cap is fully allocated by B1 to B13. Raise it to 56 MiB for B14, or move about 6 MiB of existing goldens (`ops-torch`) to a fetched asset? | raise to 56 MiB |
| Q2 | Licenses to verify and record in `ASSETS.tsv` before authoring: SmolVLM-256M-Instruct (Apache-2.0 per card; component terms), whisper-tiny (MIT in openai/whisper vs Apache-2.0 on the HF card), PD12M per-item status and metadata license, LibriSpeech CC BY 4.0 attribution text, Common Voice CC0 plus its no-re-identification terms, the OpenCV Haar cascade (Intel License), the public-domain portraits | blocking for B14 |
| Q3 | `fast_image_resize` is not documented as PIL bit-compatible. If the core engine path misses the 1 LSB bound against PIL, port the learner's Python resize to plain Rust in L15.1, or relax the core bound to 2 LSB (the optional C path L13.9 is bit-exact with the Python reference either way)? | test early in B14; fall back to a plain Rust port of M12.3 |
| Q13 | Optional kernels run on the host only. With `device = "metal"`, should `kernels = "c"` be rejected (current rule) or force a host round trip for the four ops? | reject the combination at startup |
| Q14 | The registry has optional build modules (L11.2, L11.3) but no optional milestones. Is `optional = true` on `MS-mm-kernels` the right mechanism, or should it be a side-quest milestone outside `MS-*` pass gates? | `optional = true`, excluded from every pass gate |
| Q4 | candle CPU matmul is not batch-invariant, so E3 forbids cross-request encoder batching and cross-request Whisper decoder batching. Is the throughput cost acceptable for the RTF and encode SLO defaults, or should Whisper decoding batch with a looser "greedy equal under near-tie" check? | no cross-request batching in v1 |
| Q5 | Metal results differ from CPU beyond the f32 tolerances. Should any milestone step run on Metal, or is Metal purely a local `perf` concern? | Metal only in `ci = "local"` perf steps |
| Q6 | SmolVLM's exact `Idefics3ImageProcessor` behavior for the global image (padding vs distortion) and the newer `SmolVLMForConditionalGeneration` class names depend on the pinned transformers version. Pin which version, and does the loader accept both key layouts? | pin at authoring; accept both layouts |
| Q7 | Should the audio-in LLM (`sq.audio-llm`) and streaming ASR (`sq.streaming-asr`) become core in a later pass once a call site exists (for example a voice agent)? | side quests |
| Q8 | E7 limits audio to WAV. Is that acceptable for the OpenAI-compatible endpoint (official clients often send mp3/m4a), or should the gateway transcode with an allowed dependency? | WAV only; 400 `unsupported_media_type` otherwise |
| Q9 | Vision bias evals have no person images (consent, E-ethics). Are domain-disparity and counterfactual caption probes enough for ethics.07, or should a consented, licensed face dataset be sourced? | probes only; no face datasets |
| Q10 | Pass 12 follows the `v1.0.0` gate. Should multimodal instead be an optional track after P11 (not counted in "the course"), given 15 added weeks? | core pass 12 |
| Q11 | `hf-real` nightly grows by about 0.65 GB of weights and SmolVLM numpy inference time. Split it into `hf-real-text` and `hf-real-mm` jobs with separate 20 min budgets? | split |
| Q12 | Learner Rust preprocessing (L15.1) reuses `tl-media` from L13.6, and Python decodes through `tinyllm_rs`. Does that coupling (Python preprocessing tests need the Rust farm and PyO3 build) fit the "Python is the semantic source of truth" principle, or should Python decode with a pure-Python PNG reader for tests and use `tinyllm_rs` only for JPEG? | `tinyllm_rs` for both |
