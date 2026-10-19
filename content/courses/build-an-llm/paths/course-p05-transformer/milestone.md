# What the System Does After Pass 5

After Pass 5 the system has the transformer and everything built on it. The encoder-decoder transformer learns addition, and a controlled run shows pre-LN training where post-LN does not. GPT, BERT, ELECTRA, and LoRA run through the CLI and are scored together by the model zoo. And the modern decoder loads Hugging Face checkpoints, including SmolLM2-135M, and matches them. Everything from earlier passes keeps working. This page gives the commands the finished solution runs, what each produces, and why each piece is needed.

The transformer is the chapter [The transformer (2017)](../../06-transformer-2017/README.md): [scaled dot-product attention](../../06-transformer-2017/01-scaled-dot-product-attention.md), [masks](../../06-transformer-2017/02-masks.md), [multi-head attention](../../06-transformer-2017/03-multi-head-attention.md), [positional encodings](../../06-transformer-2017/04-positional-encodings.md), and [the encoder-decoder transformer](../../06-transformer-2017/05-encoder-decoder-transformer.md). The objectives come from [Objectives and adaptation](../../07-objectives/README.md): [GPT](../../07-objectives/01-gpt-decoder-only.md), [BERT](../../07-objectives/02-bert-masked-lm.md), [ELECTRA](../../07-objectives/03-electra-replaced-token-detection.md), [fine-tuning heads](../../07-objectives/05-fine-tuning-heads-and-the-linear-head-export.md), [LoRA](../../07-objectives/06-lora-pissa-and-merge.md), and [the evaluation harness and model zoo](../../07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md). The modern decoder is [The modern decoder block](../../08-modern-block/README.md), from [pre-LN and RMSNorm](../../08-modern-block/01-pre-ln-rmsnorm.md) through [mixture of experts](../../08-modern-block/08-mixture-of-experts.md) to [the Llama-family model](../../08-modern-block/09-llama-family-model.md). Underneath sit the mathematics of attention and objectives, memory arithmetic in [Counting params, FLOPs, and KV bytes](../../../mathematics-for-computing/06-discrete-math-1/01-counting-params-flops-kv-bytes.md), and oracle-style testing in [Oracles, golden and differential tests](../../../software-craftsmanship/03-testing-mentality/03-oracles-golden-and-gradcheck.md); the full list is in the [pass opening](README.md).

## The transformer learns addition, and pre-LN trains where post-LN does not

The encoder-decoder transformer is trained and decoded through the system's own entry point. It learns three-digit addition to an exact-match rate of 0.98 with beam search, and a LayerNorm-placement demonstration shows why the modern decoder block uses pre-LN: with no warmup, a deep post-LN stack does not train, while the same stack in pre-LN does.

The task, with 4000 training pairs and 500 unseen test pairs, writes every number least significant digit first. For a = 123 and b = 456, the source is "321+654" and the target is "9750", which is 579 reversed and zero-padded to 4 digits. Reversing the digits lets the model emit each digit after the carry that determines it, which is why a small transformer can learn the task.

Why this matters: addition has an exact answer, so the transformer's correctness (its masks, positions, and attention) shows directly in the exact-match rate. The post-LN versus pre-LN run is the evidence for the block structure every later model uses.

The solution defines the Pass 5 transformer verbs of the `tinyllm` role under "Verbs of later passes" in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md). Every verb keeps the rules of that page: exit 2 on a usage error, and the last stdout line is one JSON object.

```text
{tinyllm} train transformer --task <pairs.tsv> --cfg <config.json> --out <dir>
              [--steps S] [--seed S] [--norm post|pre] [--warmup W] [--lr X]
              [--factor F] [--smoothing E] [--batch B]
    Reads source<TAB>target pairs; builds the CharTokenizer over every
    source and target with the specials <pad> <bos> <eos>; builds
    TransformerConfig(V, V, ...) from the config keys d_model, n_heads,
    d_ff, n_enc, n_dec, dropout, norm, tie_embeddings, max_len with
    rng=PCG32(S).substream("init"); trains with the transformer's fit on
    targets wrapped as <bos> target <eos>, for `steps` updates of `batch`
    pairs, shuffled by PCG32(S).substream("shuffle"), at the Noam rate
    factor * noam(step + 1, d_model, warmup) and label smoothing
    `smoothing`. The flags override the config's keys. --warmup 0 means no
    warmup: a constant rate, given with --lr (exit 2 without it). Writes
    the model directory (save_transformer) and tinyllm_char.json into <dir>.
    A non-finite loss stops training (that run diverged).
    Final line: {"out", "arch": "transformer", "norm", "warmup", "steps",
     "first_loss" (mean of the first 10 updates), "final_loss" (mean of the
     last 20), "loss_ratio" (final / first), "diverged" (1 when the loss
     went non-finite or loss_ratio > 0.8, else 0), "params"}, plus
     "post_ln_no_warmup_diverged" (true or false) when --norm post and
     --warmup 0.

{tinyllm} translate --model <transformer dir> --in <pairs.tsv> [--beam 4] [--max-len 8]
    The transformer's translate (using the generic beam search) on every
    source; exact match of the decoded string against the target with the
    sequence metrics' metric_ci (1000 bootstrap resamples, PCG32(0)). The
    same verb on a seq2seq directory is Pass 4's. Final line: {"model",
     "arch": "transformer", "beam", "n", "em_transformer", "em", "em_lo",
     "em_hi", "em_greedy"}.
```

## GPT, BERT, ELECTRA, and LoRA, scored by the model zoo

The objectives and the evaluation harness run through the system's own entry point. A GPT is trained on the byte corpus to the reference's calibrated validation loss. A BERT and an ELECTRA are pretrained on the same in-domain text, and each is then fine-tuned with LoRA (r = 8) into a sentiment classifier that trains under 5% of its parameters and reaches the reference's calibrated accuracy. A perplexity is reported with its confidence interval, and the model zoo scores every family trained so far in one report.

The data are stand-ins: Pass 3's byte corpus for GPT, and a synthetic SST-2-shaped set (split, sentence, label; 1600 train and 400 validation rows) for the classifiers, whose train sentences also serve as a token shard for BERT and ELECTRA. Small `gpt-tiny` and `bert-tiny` configs keep each run to minutes.

Why this matters: the three objectives (next token, masked token, and replaced-token detection) are the ways a transformer is pretrained. LoRA shows that adapting a pretrained model can touch under 5% of its weights. And the zoo puts every model on one report, so later passes compare like with like.

The solution defines the Pass 5 objective verbs of the `tinyllm` role in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), with the same rules. They use the byte tokenizer (decision D32) with four control bytes as specials: 0 `[PAD]`, 1 `[CLS]`, 2 `[SEP]`, 3 `[MASK]`. Seeds work the same way throughout: `PCG32(S).substream("init")` builds a model, `.substream("shuffle")` draws windows and batches, and `.substream("sample")` masks and samples.

```text
{tinyllm} train gpt --cfg <config.json> --data <train.bin> --val <val.bin> --out <dir> [--steps S] [--seed S]
    GPTConfig(vocab 256, n_ctx, d_model, n_heads, n_layers, d_ff, dropout
    from the config); the TokenStream(seq_len n_ctx, batch); fit_gpt for S
    updates (default: the config's steps) at lr with warmup; save_gpt
    (tokenizer bytes). val_loss is the harness's eval_ppl on val.bin with
    ctx n_ctx and stride n_ctx / 2: the mean NLL in nats.
    Final line: {"out", "arch": "gpt", "steps", "train_loss" (mean of the
    last 20 updates), "val_loss", "val_bpb", "params"}.

{tinyllm} train bert --cfg <config.json> --data <train.bin> --out <dir> [--steps S] [--seed S]
    BertForMLM(BertConfig(vocab 256, max_len, d_model, n_heads, n_layers,
    d_ff)); each update draws `batch` windows of `window` bytes at
    shuffle.below(n - window), wraps each as [CLS] window [SEP], masks with
    mlm_mask (p, the specials never picked, mask id 3), and takes one AdamW
    step (lr, weight_decay). save_bert (tokenizer bytes).
    Final line: {"out", "arch": "bert", "steps", "mlm_loss" (mean of the
    last 20), "params"}.

{tinyllm} train electra --cfg <config.json> --data <train.bin> --out <dir> [--steps S] [--seed S]
    ELECTRA(generator: gen_layers layers of gen_d_ff, discriminator: the
    config; tied embeddings), the same windows, electra_step with
    lam = lambda; save_electra(mask_id 3, specials 0 to 3). Final line:
    {"out", "arch": "electra", "steps", "loss", "disc_loss", "params"}.

{tinyllm} finetune classify --base <bert or electra dir> --data <split.tsv> --lora r=8[,alpha=16]
              --out <dir> [--steps 300] [--batch 16] [--lr 0.003] [--pool mean] [--seed S]
    The base's encoder (BERT's, or ELECTRA's discriminator body) under a
    SequenceClassifier (2 classes), lora_classifier (LoRA adapters on the
    attention queries and values, the head trainable), train_classifier on
    the train rows encoded [CLS] bytes [SEP], then merge_lora and
    save_classifier (arch = the base's). Accuracy on the val rows with the
    Wilson interval. Final line: {"out", "arch", "steps", "train_loss",
    "acc", "acc_lo", "acc_hi", "n", "trainable_frac" (LoRA's, before
    merging)}.

{tinyllm} eval ppl --model <gpt dir> --data <tokens.bin> [--ctx C] [--stride S]
    The harness's eval_ppl (default ctx n_ctx, stride ctx / 2), one byte per
    token. Final line: {"model", "arch", "ctx", "stride", "tokens", "ppl",
    "ppl_lo", "ppl_hi", "bpb", "bpb_lo", "bpb_hi", "nll_mean"}.

{tinyllm} zoo add --manifest <zoo.json> --id <id> --dir <dir> --task <name> --data <file> [--set key=json ...]
    Adds (or replaces) one entry of the zoo manifest
    (contracts/py/tinyllm/eval/zoo.pyi), with absolute paths.
    Final line: {"manifest", "models", "added"}.

{tinyllm} eval --suite zoo --manifest <zoo.json> [--out <report.json>] [--seed S]
    The harness's run_zoo with PCG32(S).substream("sample"); writes the
    formats/eval-results.schema.json report (default: zoo-report.json next
    to the manifest) and prints one line per row. Final line: {"suite",
    "report", "rows", "ok", "errors", "skipped"}.
```

The AdamW step is the one from [Adam and AdamW](../../../mathematics-for-computing/11-optimization/03-adam-and-adamw.md), and the Wilson interval comes from [LLN, CLT, confidence intervals, and the bootstrap](../../../mathematics-for-computing/08-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md).

## The modern decoder loads Hugging Face checkpoints and matches them

The modern decoder runs through the system's own entry point. A Llama-family checkpoint, exactly as `transformers` saves it, loads into the Llama-family model built from the modern-block sections. It reports Hugging Face's parameter count, gives Hugging Face's next-token logits, and decodes Hugging Face's greedy continuation token for token. A tiny random two-layer model shows this in seconds; the same commands on SmolLM2-135M-Instruct, pulled with the solution's own downloader, match Hugging Face as well.

The solution defines the Pass 5 Llama verbs of the `tinyllm` role in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), with the same rules. `--model` names a Llama directory or its `config.json`. A Llama directory holds `config.json` (per `formats/config.schema.json`: `tl_arch` `"llama"`, or a plain Hugging Face `model_type` of `llama`, `mistral`, or `qwen2`) and `model.safetensors`, or shards with `model.safetensors.index.json`. Its tokenizer is the byte tokenizer when `config.json` says `tl_tokenizer` `"bytes"` (decision D32); otherwise it is `tokenizer.json` read with `BPETokenizer.from_hf_json` from [Byte-level BPE](../../02-tokenizers/02-byte-level-bpe.md).

```text
{tinyllm} pull <owner/name> [--revision R] [--cache-dir DIR] [--files a,b,...]
    hf_download of the repository's files (default: config.json,
    generation_config.json, model.safetensors, tokenizer.json,
    tokenizer_config.json, special_tokens_map.json) into
    DIR/<owner>--<name>/<R> (DIR defaults to $TINYLLM_CACHE/models).
    Final line: {"dir": "<that directory>", "files": [...]}

{tinyllm} info --model <llama dir>
    Final line: {"params": LlamaForCausalLM.param_count(), "arch": "llama",
    "layers", "attention", "vocab", "kv_bytes_per_token_bf16"}.

{tinyllm} logits --model <llama dir> --prompt <text> [--prefix-ids a,b,...]
    The Pass 1 verb (spec/cli-roles.md) on a Llama directory: the float32
    next-token logits after the prompt's ids (then the prefix ids). Final
    line: {"logits": [<vocab floats>]} and nothing else on stdout (the
    numeric matcher reads every number printed).

{tinyllm} logits --model <llama dir> --prompts <file.jsonl> --out <file.npz>
    One {"prompt": ...} or {"ids": [...]} per line; the npz holds logits0,
    logits1, ... (float32 [vocab]). Final line: {"out", "prompts"}.

{tinyllm} generate --model <llama dir> (--prompt <text> | --prompt-file <file>)
              [--max-tokens N] [--greedy | --temperature T] [--seed S] [--out FILE]
    Decodes N tokens after each prompt (one per line of --prompt-file)
    through a KV cache (the GQA section's ConcatKVCache hook; Pass 6
    brings your own cache). --greedy: the largest logit, ties to the
    lowest id. Final line: {"ids": [the generated ids of every prompt,
    concatenated in order], "text": ..., "per_prompt": [[...], ...] (with
    --prompt-file)}.
```

The `kv_bytes_per_token_bf16` figure is computed as in [Counting params, FLOPs, and KV bytes](../../../mathematics-for-computing/06-discrete-math-1/01-counting-params-flops-kv-bytes.md). The cache hook is from [Grouped-query attention](../../08-modern-block/05-gqa-attention.md), and [The KV cache, incremental decoding, and generate](../../09-inference/02-kv-cache-and-generate.md) replaces it in Pass 6.

The tiny model, `tiny-llama-2l`, is a random `LlamaForCausalLM` (2 layers, BF16 weights, byte tokenizer) saved by `transformers` 5.19.0; the expected logits and greedy ids are Hugging Face's float32 forward pass. Every prompt's greedy continuation keeps a top-2 logit margin of at least 1e-3 at all 32 steps, so floating-point differences cannot flip a near tie, and the token-for-token comparison is meaningful. The full-size model is SmolLM2-135M-Instruct at a pinned revision.

Why this matters: matching a published checkpoint exactly is the proof that every modern component (RMSNorm, RoPE, grouped-query attention, SwiGLU, and the loader) is right. From here on the engine can serve real weights, and [Pass 6](../course-p06-inference-and-kernels/README.md) makes decoding them fast.
