# What the system does after Pass 3

**Pass result**: corpus pipeline v1 with ledger and decontamination, BPE, WordPiece, and Unigram in Python, BPE in Rust, n-gram/NPLM/word2vec run through {tinyllm}; the engine keeps serving the bigram.

Each capability below is built from the listed modules. The sections that follow give the commands the finished solution runs, what each one produces, and why each piece is needed. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| The tokenizers match GPT-2 and SmolLM2 exactly, in Python and in Rust. | `L1.1`, `L1.2`, `L1.3`, `L1.4`, `L1.5`, `L1.6`, `ds.05`, `ds.06`, `craft.04` |
| The corpus pipeline cleans, shards, licenses, and tokenizes a corpus deterministically. | `data.01`, `data.02`, `data.03`, `ds.08`, `data.04`, `data.05`, `data.06`, `data.07`, `data.08`, `L0.6` |
| The n-gram and neural language models, trained and scored through the CLI. | `L2.1`, `L2.2`, `L2.3` |
| The foundations the three capabilities rest on: Unicode, hashing, information theory, probability, linear algebra, data ethics, and the Rust primer. | `M05.2`, `M06.2`, `S-M06b`, `M11.2`, `M07.1`, `M07.2`, `S-M07b`, `ethics.01`, `ethics.02`, `ds.08`, `lang.08`, `M03.5`, `M03.6`, `S-M03b`, `M11.4`, `S-M11b` |


## Details

## The tokenizers match GPT-2 and SmolLM2 exactly, in Python and in Rust

The tokenizers, run through their own entry points: the Python BPE (L1.2)
trains a vocabulary and encodes exactly like Hugging Face on GPT-2's and
SmolLM2's tokenizer.json; Rust tl-tok (L1.5) independently reads the same
files and produces the same ids.

The solution defines the Pass 3 tokenizer verbs of the `tinyllm` and `tl-tok`
roles ([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes"). Every verb keeps the
rules of that page: exit 2 on a usage error, the last stdout line is one
JSON object. A texts file ending in .jsonl is JSON lines with a string
"text" (other keys are ignored); any other file is one text per line.

  {tinyllm} tok train --algo bpe --vocab N --in <text file> --out <dir> [--min-freq F] [--special S]...
      Trains the BPETokenizer (L1.2) on the lines of the file and saves
      <dir>/tokenizer.json (the formats/tokenizer.md BPE subset). Final line:
      {"out": "<dir>", "algo": "bpe", "vocab_size": N, "merges": <int>, "bytes_per_token": <float>}
      with bytes_per_token measured on the training file.

  {tinyllm} tok encode --tokenizer <tokenizer.json> --in <texts.jsonl>
      Encodes every text with the Python BPE. Final line:
      {"ids": [every id of every text, in order], "texts": <int>, "tokens": <int>, "bytes_per_token": <float>}

  {tl-tok} encode --tokenizer <tokenizer.json> --in <texts.jsonl>
      Encodes every text with the independent Rust BPE. Its output is
      identical to the Python command's ids on the same texts.

  {tl-tok} bench --tokenizer <tokenizer.json> --in <texts.jsonl> [--threads T] [--repeat R]
      Final line includes {"tokens_per_s": <float>, "batch_speedup": t_serial / t_batch}.

The training corpus is a synthetic stand-in for the TinyStories 5 MB slice,
and the vocabulary is 512. The encoded texts are 2,000 hard cases (emoji
sequences, CJK, white-space runs, contractions in every case, Unicode
digits), and the expected ids come from Hugging Face tokenizers. Why it is
needed: every later model's ids come from these tokenizers, and a single
id that differs from Hugging Face makes a published checkpoint read the
wrong embedding row. The Rust tokenizer is independent so the engine can
tokenize without Python, and `bench` shows why batching matters.

## The corpus pipeline cleans, shards, licenses, and tokenizes a corpus deterministically

The `corpus` CLI (data.01 to data.08 composed: fetch, filter, exact and
near dedup, decontamination, PII scrub, shards, ledger, tokens) cleans the
small story corpus into shards, a manifest, a reconciled ledger, and .bin
token streams that pass the format contracts; the output hash is the same
for 1 and 4 workers in two separate runs; the decontamination count is
recorded; an unlicensed source stops the build with exit 65.

The solution defines the Pass 3 verbs of the `corpus` role ([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md),
"Verbs of later passes"). Every verb keeps the rules of that page: exit 2
on a usage error, the last stdout line is one JSON object.

  Paths. Every output goes under $TL_ARTIFACTS (the caller sets it): corpus/
  (raw/, LEDGER.jsonl, <dataset>/<version>/ with the shards and
  _MANIFEST.json) and tokens/<tokenizer_id>/<dataset>/ (formats/
  corpus-shard.md, formats/tokens-bin.md). In a config, a source url
  without a scheme and a decontam.protected path that is relative resolve
  against the config file's directory. data.01's fetch speaks HTTP only,
  so serve such sources on a loopback port while it runs (the reference
  uses http.server in a thread) and write the documents' urls back as the
  config gave them ("sources/stories.jsonl#12"): the port must not reach
  the shards.

  {corpus} run --config <toml> [--until <stage>] [--workers N]
      Stages fetch, filter (length, lang, gopher, repetition), dedup_exact,
      dedup_near (and decontamination), pii, shard, tokenize; --until stops
      after one (default tokenize). After shard: reconcile the ledger
      (data.08) and verify it; a failed verification exits 65. Final line,
      also written to $TL_ARTIFACTS/corpus/RUN-<dataset>-<version>.json:
      {"dataset", "version", "stage", "n_docs", "n_shards", "filters": {...},
       "exact_dropped", "near_dropped", "decontaminated", "pii_redactions",
       "train_tokens", "val_tokens", "output_sha256"}
      output_sha256 = sha256 of the lines "<path>\t<sha256 of the file>\n",
      sorted by path, for every file under corpus/<dataset>/<version>/ and
      tokens/<tokenizer_id>/<dataset>/, paths relative to $TL_ARTIFACTS.
      No time, absolute path, port, or worker count may appear in the line.

  {corpus} ledger verify --config <toml>
      data.08's check() on corpus/LEDGER.jsonl and the config's corpus;
      exit 0, or 65 with the problems on stderr.

  {corpus} datasheet --config <toml>
      data.08's datasheet() on stdout (with the tokens directory when it
      exists).

The example corpus plants what each
stage removes: 2 too short, 2 French, 1 repetitive, 4 verbatim copies, 5
one-word edits, 3 documents holding a 15-word span of the protected
stories, and 5 PII matches; 169 documents remain (140 stories, 24 notes,
5 stories whose PII is scrubbed). The token counts are those of the byte
tokenizer under the document-hash split with val_permille 100, which the
contracts fix, so every correct pipeline prints the same numbers.

Why it is needed: every model from Pass 5 on trains on this pipeline's
output, so its result must be reproducible (the same `output_sha256` for 1
and 4 workers), auditable (the ledger records every document's licence and
fate), and free of evaluation text (decontamination), or no later
evaluation number can be trusted.

## The n-gram and neural language models, trained and scored through the CLI

The statistical language models, run through the system's own entry point: a
Kneser-Ney 4-gram (L2.1) trained and scored to within 0.5% of the exact
reference perplexity, then Bengio's NPLM (L2.2) trained with the autograd
to the reference's calibrated perplexity (below the count bigram's), and
generation from the NPLM that repeats itself exactly for a fixed seed. Both
scores are reported in bits per byte (M00.1, through M11.2's accumulator).
word2vec (L2.3) is not run here: the L6.7 zoo is where it is used.

The solution defines the Pass 3 language-model verbs of the `tinyllm` role
([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes"). Every verb keeps the rules of
that page: exit 2 on a usage error, the last stdout line is one JSON object.
Token files are formats/tokens-bin.md shards of byte ids (the byte
tokenizer, D32), so one token is one byte.

  {tinyllm} lm train ngram --n N --train <tokens.bin> --out <dir> [--discount modified|D]
      Fits the NGramLM (L2.1) on the whole file as ONE sequence (one <s>
      before its first token), V = 256, and writes <dir>/model.safetensors
      (NGramLM.save) and <dir>/config.json
      {"tl_arch": "ngram", "tl_tokenizer": "bytes", "vocab_size": 256, ...}.
      Final line: {"out": "<dir>", "arch": "ngram", "n": N, "tokens": <int>}

  {tinyllm} lm train nplm --train <tokens.bin> --out <dir> --context C --d-emb M
                --hidden H --steps S --batch B --lr X [--momentum MU]
                [--weight-decay WD] [--clip G] [--no-direct] [--seed S]
      Builds NPLM(256, C, M, H, direct=not --no-direct, rng=PCG32(S).substream("init")),
      trains it with train_nplm (L2.2) for exactly S steps on the windows of
      the whole file, shuffled by PCG32(S).substream("shuffle"), and writes
      the model directory with save_nplm (tokenizer "bytes").
      Final line: {"out": "<dir>", "arch": "nplm", "steps": S, "loss": <float>, "params": <int>}
      with loss the mean training loss of the last 50 steps.

  {tinyllm} eval --model <dir> --data <tokens.bin>
      Scores the file as ONE sequence with the model its config.json names
      (tl_arch ngram: every token, the first from <s>; nplm: every token with
      a full window) and sums the per-token NLLs with M11.2's NLLAccumulator,
      one byte per token. Final line:
      {"model": "<tl_arch>", "ppl": <float>, "nll_mean": <float>, "bpb": <float>,
       "tokens": <int>, "bytes": <int>}

  {tinyllm} generate --model <nplm dir> --prompt <text> --seed S --max-tokens N [--greedy] [--out <file>]
      The Pass 1 verb (cli-roles.md) on an nplm directory: NPLM.generate
      (L2.2) from the prompt's bytes at --temperature (default 1.0). With
      --out, the final line is also written to <file>.

The corpus is a synthetic stand-in for TinyStories: a train file and a
validation file of 141 532 and 14 675 bytes of short stories. Computed with
exact fractions, the KN-4 model's perplexity on it is
1.8699475121183176176, and a correct implementation is within 0.5% of that.
The add-one count bigram (L0.0) fitted on the train file gives 2.142401587
nats per byte on the validation file; the NPLM beats it. Why it is needed:
the n-gram model is the strongest count-based baseline, and the NPLM is the
first model whose embeddings and hidden layer are learned, which is the
step every later architecture builds on.
