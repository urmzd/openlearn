# What the system does after Pass 5

**Pass result**: transformer, GPT/BERT/ELECTRA/LoRA, the model zoo, modern block; SmolLM2-135M loads and matches HF.

Each capability below is built from the listed modules. The sections that follow give the commands the finished solution runs, what each one produces, and why each piece is needed. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| The transformer learns addition, and pre-LN trains where post-LN does not. | `L5.1`, `L5.2`, `L5.3`, `L5.4`, `L5.5` |
| GPT, BERT, ELECTRA, and LoRA through the CLI, scored by the model zoo. | `L6.1`, `L6.2`, `L6.3`, `L6.5`, `L6.6`, `L6.7` |
| The modern decoder loads Hugging Face checkpoints and matches them. | `L7.1`, `L7.2`, `L7.3`, `L7.4`, `L7.5`, `L7.6`, `L7.7`, `L7.8`, `L7.9` |
| Transformers: the 2017 model, its objectives, and a modern decoder that matches SmolLM2. | `M01.4`, `M07.5`, `M07.7`, `S-M07d`, `L5.1`, `L5.2`, `L5.3`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.6`, `L6.5`, `L6.7`, `M05.1`, `L7.1`, `L7.2`, `L7.3`, `L7.4`, `L7.5`, `L7.6`, `L7.7`, `L7.8`, `L7.9`, `craft.05` |


## Details

## The transformer learns addition, and pre-LN trains where post-LN does not

The encoder-decoder Transformer (L5.1 to L5.5), trained and decoded through
the system's own entry point: it learns three-digit addition to an exact-match rate
of 0.98 with beam search, and the LayerNorm-placement demo shows why Part 7
uses pre-LN: with no warmup, a deep post-LN stack does not train while the
same stack in pre-LN does.

The task (course/fixtures/small-corpora/add3.tsv, 4000 pairs, and
add3-test.tsv, 500 unseen pairs, from course/oracle/MS-L5/add3.py) writes
every number least significant digit first: a = 123, b = 456 is the source
"321+654" and the target "9750" (579 reversed, zero padded to 4 digits).
DEVIATIONS B72-01 says why it is add3, not the catalog's add5.

The solution defines the Pass 5 transformer verbs of the `tinyllm` role
([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes"). Every verb keeps the rules of
that page: exit 2 on a usage error, the last stdout line is one JSON object.

  {tinyllm} train transformer --task <pairs.tsv> --cfg <config.json> --out <dir>
                [--steps S] [--seed S] [--norm post|pre] [--warmup W] [--lr X]
                [--factor F] [--smoothing E] [--batch B]
      Reads `source<TAB>target` pairs; builds L1.1's CharTokenizer over every
      source and target with the specials <pad> <bos> <eos>; builds
      TransformerConfig(V, V, ...) from the config keys d_model, n_heads,
      d_ff, n_enc, n_dec, dropout, norm, tie_embeddings, max_len with
      rng=PCG32(S).substream("init"); trains with L5.5's fit on targets
      wrapped as <bos> target <eos>, for `steps` updates of `batch` pairs,
      shuffled by PCG32(S).substream("shuffle"), at the Noam rate
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
      L5.5's translate (L4.4's beam search) on every source; exact match of
      the decoded string against the target with L4.5's metric_ci (1000
      bootstrap resamples, PCG32(0)). The same verb on a seq2seq directory is
      MS-L4's. Final line: {"model", "arch": "transformer", "beam", "n",
       "em_transformer", "em", "em_lo", "em_hi", "em_greedy"}.

## GPT, BERT, ELECTRA, and LoRA through the CLI, scored by the model zoo

The objectives and the evaluation harness, run through its own entry
point: a GPT trained on the byte corpus to the reference's calibrated
validation loss; a BERT and an ELECTRA pretrained on the same in-domain
text, each then fine-tuned with LoRA (r = 8) into a sentiment classifier
that trains under 5% of its parameters and reaches the reference's
calibrated accuracy; a perplexity with its confidence interval; and the
model zoo scoring every family trained here in one report.

Data (stand-ins, DEVIATIONS B73-05 and B73-06): the byte corpus of MS-L2
(course/fixtures/MS-L2/{train,val}.bin) for GPT; the synthetic SST-2-shaped
set course/fixtures/small-corpora/sst2-2k.tsv (split, sentence, label;
1600 train, 400 val) for the classifiers, and its train sentences as a
token shard, course/fixtures/MS-L6/sst2-train.bin, for BERT and ELECTRA.
Configs: course/fixtures/configs/{gpt-tiny,bert-tiny}.json.

The solution defines the Pass 5 objective verbs of the `tinyllm` role
([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes"). Every verb keeps the rules of
that page: exit 2 on a usage error, the last stdout line is one JSON object.
The byte tokenizer (D32) with four control bytes as specials: 0 [PAD],
1 [CLS], 2 [SEP], 3 [MASK]. Seeds: PCG32(S).substream("init") builds a
model, .substream("shuffle") draws windows and batches, .substream("sample")
masks and samples.

  {tinyllm} train gpt --cfg <config.json> --data <train.bin> --val <val.bin> --out <dir> [--steps S] [--seed S]
      GPTConfig(vocab 256, n_ctx, d_model, n_heads, n_layers, d_ff,
      dropout from the config); L0.6's TokenStream(seq_len n_ctx, batch);
      L6.1's fit_gpt for S updates (default: the config's steps) at lr with
      warmup; save_gpt (tokenizer bytes). val_loss is L6.7's eval_ppl on
      val.bin with ctx n_ctx and stride n_ctx / 2: the mean NLL in nats.
      Final line: {"out", "arch": "gpt", "steps", "train_loss" (mean of the
      last 20 updates), "val_loss", "val_bpb", "params"}.

  {tinyllm} train bert --cfg <config.json> --data <train.bin> --out <dir> [--steps S] [--seed S]
      BertForMLM(BertConfig(vocab 256, max_len, d_model, n_heads, n_layers,
      d_ff)); each update draws `batch` windows of `window` bytes at
      shuffle.below(n - window), wraps each as [CLS] window [SEP], masks with
      L6.2's mlm_mask (p, the specials never picked, mask id 3), and takes
      one M10.3 AdamW step (lr, weight_decay). save_bert (tokenizer bytes).
      Final line: {"out", "arch": "bert", "steps", "mlm_loss" (mean of the
      last 20), "params"}.

  {tinyllm} train electra --cfg <config.json> --data <train.bin> --out <dir> [--steps S] [--seed S]
      L6.3's ELECTRA(generator: gen_layers layers of gen_d_ff, discriminator:
      the config; tied embeddings), the same windows, electra_step with
      lam = lambda; save_electra(mask_id 3, specials 0 to 3). Final line:
      {"out", "arch": "electra", "steps", "loss", "disc_loss", "params"}.

  {tinyllm} finetune classify --base <bert or electra dir> --data <split.tsv> --lora r=8[,alpha=16]
                --out <dir> [--steps 300] [--batch 16] [--lr 0.003] [--pool mean] [--seed S]
      The base's encoder (BERT's, or ELECTRA's discriminator body) under an
      L6.5 SequenceClassifier (2 classes), L6.5's lora_classifier (L6.6
      adapters on the attention queries and values, the head trainable),
      train_classifier on the train rows encoded [CLS] bytes [SEP], then
      merge_lora and save_classifier (arch = the base's). Accuracy on the
      val rows with M07.4's Wilson interval. Final line: {"out", "arch",
      "steps", "train_loss", "acc", "acc_lo", "acc_hi", "n",
      "trainable_frac" (L6.6's, before merging)}.

  {tinyllm} eval ppl --model <gpt dir> --data <tokens.bin> [--ctx C] [--stride S]
      L6.7's eval_ppl (default ctx n_ctx, stride ctx / 2), one byte per
      token. Final line: {"model", "arch", "ctx", "stride", "tokens", "ppl",
      "ppl_lo", "ppl_hi", "bpb", "bpb_lo", "bpb_hi", "nll_mean"}.

  {tinyllm} zoo add --manifest <zoo.json> --id <id> --dir <dir> --task <name> --data <file> [--set key=json ...]
      Adds (or replaces) one entry of the zoo manifest (contracts/py/tinyllm/eval/zoo.pyi),
      with absolute paths. Final line: {"manifest", "models", "added"}.

  {tinyllm} eval --suite zoo --manifest <zoo.json> [--out <report.json>] [--seed S]
      L6.7's run_zoo with PCG32(S).substream("sample"); writes the
      formats/eval-results.schema.json report (default: zoo-report.json next
      to the manifest) and prints one line per row. Final line: {"suite",
      "report", "rows", "ok", "errors", "skipped"}.

## The modern decoder loads Hugging Face checkpoints and matches them

The modern decoder, run through the system's own entry point: a Llama-family
checkpoint exactly as transformers saves it loads into the L7.9 model
(built from L7.1 to L7.8), reports Hugging Face's parameter count, gives
HF's next-token logits, and decodes HF's greedy continuation token for
token. PR CI runs it on the committed tiny model; the nightly run pulls
SmolLM2-135M-Instruct with its own downloader and checks it against HF.

The solution defines the Pass 5 Llama verbs of the `tinyllm` role
([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes"). Every verb keeps the rules of
that page: exit 2 on a usage error, the last stdout line is one JSON object.
--model names a Llama directory, or its config.json (the committed tiny
model is referenced by its config.json, a fixture file with a MANIFEST row).
A "Llama directory" holds config.json (formats/config.schema.json: tl_arch
"llama", or a plain HF model_type llama, mistral, or qwen2) and
model.safetensors (or shards with model.safetensors.index.json). Its
tokenizer is the byte tokenizer when config.json says tl_tokenizer "bytes"
(D32), else tokenizer.json read with the L1.2 BPETokenizer.from_hf_json.

  {tinyllm} pull <owner/name> [--revision R] [--cache-dir DIR] [--files a,b,...]
      L7.9's hf_download of the repository's files (default: config.json,
      generation_config.json, model.safetensors, tokenizer.json,
      tokenizer_config.json, special_tokens_map.json) into
      DIR/<owner>--<name>/<R> (DIR defaults to $TINYLLM_CACHE/models).
      Final line: {"dir": "<that directory>", "files": [...]}

  {tinyllm} info --model <llama dir>
      Final line: {"params": LlamaForCausalLM.param_count(), "arch": "llama",
      "layers", "attention", "vocab", "kv_bytes_per_token_bf16"} (M05.1).

  {tinyllm} logits --model <llama dir> --prompt <text> [--prefix-ids a,b,...]
      The Pass 1 verb (spec/cli-roles.md) on a Llama directory: the
      float32 next-token logits after the prompt's ids (then the prefix
      ids). Final line: {"logits": [<vocab floats>]} and nothing else on
      stdout (the numeric matcher reads every number printed).
  {tinyllm} logits --model <llama dir> --prompts <file.jsonl> --out <file.npz>
      One {"prompt": ...} or {"ids": [...]} per line; the npz holds
      logits0, logits1, ... (float32 [vocab]). Final line: {"out", "prompts"}.

  {tinyllm} generate --model <llama dir> (--prompt <text> | --prompt-file <file>)
                [--max-tokens N] [--greedy | --temperature T] [--seed S] [--out FILE]
      Decodes N tokens after each prompt (one per line of --prompt-file)
      through a KV cache (L7.5's ConcatKVCache hook; L8.2 brings yours).
      --greedy: the largest logit, ties to the lowest id. Final line:
      {"ids": [the generated ids of every prompt, concatenated in order],
       "text": ..., "per_prompt": [[...], ...] (with --prompt-file)}.

Fixtures (course/oracle/L7.9/llama_hf.py, course/oracle/MS-L7/smollm2_parity.py):
tiny-llama-2l is a random LlamaForCausalLM (2 layers, BF16 weights, byte
tokenizer) saved by transformers 5.19.0; its logits and greedy ids are HF's
float32 forward. Every prompt's greedy continuation keeps a top-2 logit
margin >= 1e-3 at all 32 steps (DESIGN 5.7 near-tie rule), and the expected
files carry that margin per step. The nightly model is
SmolLM2-135M-Instruct at the revision pinned in course/fixtures/ASSETS.tsv,
not the base SmolLM2-135M of DESIGN 4.3 (DEVIATIONS B75-07).

## Transformers: the 2017 model, its objectives, and a modern decoder that matches SmolLM2

Pass 5 builds the transformer three times over: the 2017 encoder-decoder
(MS-L5), the objectives and the model zoo (MS-L6), and the modern decoder
that loads SmolLM2 (MS-L7). The gate is those three plus the spiral
invariant: the smoke steps of every earlier pass gate rerun, so the engine
still streams the Pass 1 bigram and the earlier models still train.

`requires` is every Pass 5 stage of DESIGN 7.4 (the optional L6.4 is not
required); each needs a fresh pass of the own. craft.05 grades the oracle
tests (rung R5) with its own artifact check.