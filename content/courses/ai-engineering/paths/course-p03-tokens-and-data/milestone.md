# What the System Does After Pass 3

After Pass 3 the system has three new capabilities. Its tokenizers match GPT-2 and SmolLM2 exactly, in Python and in Rust. Its corpus pipeline cleans, shards, licenses, and tokenizes a corpus deterministically, with a ledger and decontamination. And its n-gram and neural language models train and score through the CLI. The engine keeps serving the bigram, and everything from earlier passes keeps working. This page gives the commands the finished solution runs, what each produces, and why each piece is needed.

The tokenizers come from the [Tokenizers](../../07-tokenizers/README.md) chapter, from [the tokenizer protocol](../../07-tokenizers/01-tokenizer-protocol-and-char.md) through [byte-level BPE](../../07-tokenizers/02-byte-level-bpe.md), [WordPiece](../../07-tokenizers/03-wordpiece.md), [Unigram](../../07-tokenizers/04-unigram-lm.md), [fast BPE in Rust](../../07-tokenizers/05-rust-fast-bpe.md), and [tokenizer metrics](../../07-tokenizers/06-tokenizer-metrics.md), supported by [a Robin Hood hash map](../../../programming-and-algorithms/15-systems-data-structures/05-robin-hood-hash-map.md), [a binary heap with lazy deletion](../../../programming-and-algorithms/15-systems-data-structures/06-binary-heap-lazy-deletion.md), and [property-based tests](../../../software-engineering/03-testing/02-property-based-tests.md). The pipeline is the corpus-pipeline chapter of *Data Engineering*, from [async fetch](../../../data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md) to [the ledger and the datasheet](../../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md), with [a Bloom filter](../../../programming-and-algorithms/15-systems-data-structures/08-bloom-filter.md) and the [token stream](../../06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md) from Pass 2. The language models are [the Kneser-Ney n-gram model](../../08-statistical-language-models/01-ngram-kneser-ney.md), [Bengio's NPLM](../../08-statistical-language-models/02-bengio-nplm.md), and [word2vec](../../08-statistical-language-models/03-word2vec-and-ppmi-svd.md). The foundations under all three (Unicode, hashing, information theory, probability, linear algebra, data ethics, and the asyncio primer) are listed in the [pass opening](README.md).

## The tokenizers match GPT-2 and SmolLM2 exactly

The Python BPE trains a vocabulary and encodes exactly like Hugging Face on GPT-2's and SmolLM2's `tokenizer.json`. The Rust `tl-tok` independently reads the same files and produces the same ids.

The solution defines the Pass 3 tokenizer verbs of the `tinyllm` and `tl-tok` roles under "Verbs of later passes" in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md). Every verb keeps the rules of that page: exit 2 on a usage error, and the last stdout line is one JSON object. A texts file ending in `.jsonl` is JSON lines with a string `"text"` (other keys are ignored); any other file is one text per line.

```text
{tinyllm} tok train --algo bpe --vocab N --in <text file> --out <dir> [--min-freq F] [--special S]...
    Trains the BPETokenizer on the lines of the file and saves
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
```

The training corpus is a synthetic stand-in for the TinyStories 5 MB slice, and the vocabulary is 512. The encoded texts are 2,000 hard cases (emoji sequences, CJK, whitespace runs, contractions in every case, Unicode digits), and the expected ids come from Hugging Face tokenizers.

Why this matters: every later model's ids come from these tokenizers, and a single id that differs from Hugging Face makes a published checkpoint read the wrong embedding row. The Rust tokenizer is independent so that the engine can tokenize without Python, and `bench` shows why batching matters.

## The corpus pipeline is deterministic and auditable

The `corpus` CLI composes the pipeline's stages: fetch, filter, exact and near dedup, decontamination, PII scrub, shards, ledger, and tokens. It cleans the small story corpus into shards, a manifest, a reconciled ledger, and `.bin` token streams that pass the format contracts. The output hash is the same for 1 and 4 workers in two separate runs, the decontamination count is recorded, and an unlicensed source stops the build with exit 65.

The solution defines the Pass 3 verbs of the `corpus` role in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), with the same rules. Every output goes under `$TL_ARTIFACTS`, which the caller sets: `corpus/` (with `raw/`, `LEDGER.jsonl`, and `<dataset>/<version>/` holding the shards and `_MANIFEST.json`) and `tokens/<tokenizer_id>/<dataset>/`, in the layouts of `formats/corpus-shard.md` and `formats/tokens-bin.md`. In a config, a source URL without a scheme and a relative `decontam.protected` path resolve against the config file's directory. The fetch stage speaks HTTP only, so such sources are served on a loopback port while it runs (the reference uses `http.server` in a thread), and the documents' URLs are written back as the config gave them (`"sources/stories.jsonl#12"`): the port must not reach the shards.

```text
{corpus} run --config <toml> [--until <stage>] [--workers N]
    Stages fetch, filter (length, lang, gopher, repetition), dedup_exact,
    dedup_near (and decontamination), pii, shard, tokenize; --until stops
    after one (default tokenize). After shard: reconcile the ledger and
    verify it; a failed verification exits 65. Final line, also written to
    $TL_ARTIFACTS/corpus/RUN-<dataset>-<version>.json:
    {"dataset", "version", "stage", "n_docs", "n_shards", "filters": {...},
     "exact_dropped", "near_dropped", "decontaminated", "pii_redactions",
     "train_tokens", "val_tokens", "output_sha256"}
    output_sha256 = sha256 of the lines "<path>\t<sha256 of the file>\n",
    sorted by path, for every file under corpus/<dataset>/<version>/ and
    tokens/<tokenizer_id>/<dataset>/, paths relative to $TL_ARTIFACTS.
    No time, absolute path, port, or worker count may appear in the line.

{corpus} ledger verify --config <toml>
    The ledger's check() on corpus/LEDGER.jsonl and the config's corpus;
    exit 0, or 65 with the problems on stderr.

{corpus} datasheet --config <toml>
    The ledger's datasheet() on stdout (with the tokens directory when it
    exists).
```

The example corpus plants exactly what each stage removes: 2 documents too short, 2 in French, 1 repetitive, 4 verbatim copies, 5 one-word edits, 3 documents holding a 15-word span of the protected stories, and 5 PII matches. 169 documents remain: 140 stories, 24 notes, and 5 stories whose PII is scrubbed. The token counts are those of the byte tokenizer under the document-hash split with `val_permille` 100, which the contracts fix, so every correct pipeline prints the same numbers.

Why this matters: every model from Pass 5 on trains on this pipeline's output, so its result must be reproducible (the same `output_sha256` for 1 and 4 workers), auditable (the ledger records every document's licence and fate), and free of evaluation text (decontamination). Without all three, no later evaluation number can be trusted.

## The n-gram and neural language models run through the CLI

The statistical language models run through the system's own entry point. A Kneser-Ney 4-gram is trained and scored to within 0.5% of the exact reference perplexity. Bengio's NPLM is then trained with the autograd to the reference's calibrated perplexity, below the count bigram's, and generation from the NPLM repeats itself exactly for a fixed seed. Both scores are reported in bits per byte, using the conversions of [Exponents, logs, and units of information](../../../mathematics/06-precalculus/01-exponents-logs-and-units-of-information.md) through the accumulator of [Perplexity, bits per byte, and the NLL accumulator](../../../mathematics/17-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md). word2vec is not run here; the [model zoo](../../12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) of Pass 5 is where it is used.

The solution defines the Pass 3 language-model verbs of the `tinyllm` role in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), with the same rules. Token files are `formats/tokens-bin.md` shards of byte ids (the byte tokenizer, decision D32), so one token is one byte.

```text
{tinyllm} lm train ngram --n N --train <tokens.bin> --out <dir> [--discount modified|D]
    Fits the NGramLM on the whole file as ONE sequence (one <s> before its
    first token), V = 256, and writes <dir>/model.safetensors
    (NGramLM.save) and <dir>/config.json
    {"tl_arch": "ngram", "tl_tokenizer": "bytes", "vocab_size": 256, ...}.
    Final line: {"out": "<dir>", "arch": "ngram", "n": N, "tokens": <int>}

{tinyllm} lm train nplm --train <tokens.bin> --out <dir> --context C --d-emb M
              --hidden H --steps S --batch B --lr X [--momentum MU]
              [--weight-decay WD] [--clip G] [--no-direct] [--seed S]
    Builds NPLM(256, C, M, H, direct=not --no-direct, rng=PCG32(S).substream("init")),
    trains it with train_nplm for exactly S steps on the windows of the
    whole file, shuffled by PCG32(S).substream("shuffle"), and writes the
    model directory with save_nplm (tokenizer "bytes").
    Final line: {"out": "<dir>", "arch": "nplm", "steps": S, "loss": <float>, "params": <int>}
    with loss the mean training loss of the last 50 steps.

{tinyllm} eval --model <dir> --data <tokens.bin>
    Scores the file as ONE sequence with the model its config.json names
    (tl_arch ngram: every token, the first from <s>; nplm: every token with
    a full window) and sums the per-token NLLs with the NLLAccumulator,
    one byte per token. Final line:
    {"model": "<tl_arch>", "ppl": <float>, "nll_mean": <float>, "bpb": <float>,
     "tokens": <int>, "bytes": <int>}

{tinyllm} generate --model <nplm dir> --prompt <text> --seed S --max-tokens N [--greedy] [--out <file>]
    The Pass 1 verb (cli-roles.md) on an nplm directory: NPLM.generate
    from the prompt's bytes at --temperature (default 1.0). With --out,
    the final line is also written to <file>.
```

The corpus is a synthetic stand-in for TinyStories: a train file and a validation file of 141 532 and 14 675 bytes of short stories. Computed with exact fractions, the KN-4 model's perplexity on it is 1.8699475121183176176, and a correct implementation is within 0.5% of that. The add-one count bigram from the [byte bigram](../../06-numerics-and-autograd/01-byte-bigram.md) section, fitted on the train file, gives 2.142401587 nats per byte on the validation file, and the NPLM beats it.

Why this matters: the n-gram model is the strongest count-based baseline, and the NPLM is the first model whose embeddings and hidden layer are learned, which is the step every later architecture builds on. [Pass 4](../course-p04-sequence-models/README.md) adds models that carry state across the whole sequence.
