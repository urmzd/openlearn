# What the system does after Pass 4

**Pass result**: RNN/LSTM/GRU LMs, seq2seq with attention, beam search in the CLI.

Each capability below is built from the listed modules. The sections that follow give the commands the finished solution runs, what each one produces, and why each piece is needed. Everything from earlier passes keeps working.

| What the system does | Built from |
|---|---|
| The recurrent language models, trained, ordered, and scored through the CLI. | `L3.1`, `L3.2`, `L3.3`, `L3.4`, `L3.6` |
| The encoder-decoders translate dates, and attention closes the gap. | `L3.4`, `L4.1`, `L4.2`, `L4.3`, `L4.4`, `L4.5`, `craft.07` |
| The statistics both rest on: the bootstrap and confidence intervals for the reported metrics. | `M07.4`, `S-M07c` |


## Details

## The recurrent language models, trained, ordered, and scored through the CLI

The recurrent language models, run through the system's own entry point: an Elman
RNN (L3.1, fused as one op by L3.6), an LSTM (L3.2), and a GRU (L3.3), each
trained with stateful truncated BPTT (L3.6) on Tiny Shakespeare, the LSTM
sampling text, and the LSTM trained on Pass 3's story corpus and scored on
its validation file, where it beats Pass 3's NPLM in bits per byte.
L3.4 (the bidirectional encoder) is used by the encoder-decoders below.

The solution defines the Pass 4 language-model verbs of the `tinyllm` role
([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), "Verbs of later passes"); exit 2 on a usage error, the
last stdout line is one JSON object. Text files are read as UTF-8 bytes,
.bin files as formats/tokens-bin.md shards of byte ids (D32).

  {tinyllm} train rnnlm --cell rnn|lstm|gru --data <file> --out <dir> --steps S --seed S
                --d-emb E --hidden H --bptt K --batch B --lr X --clip G
      The last 10% of the file's tokens (int(n * 0.1)) are held out. Builds
      RNNLM(256, E, H, cell, rng=PCG32(seed).substream("init")), trains it
      with train_tbptt(model, head, K, B, AdamW(lr=X, weight_decay=0), G, S)
      and writes the model directory with save_rnnlm (tokenizer "bytes").
      Final line: {"out", "arch": "rnnlm", "cell", "steps": S, "loss": <mean
      of the last 50 losses>, "val_bpc": <mean nll of the tail, in bits>,
      "val_tokens", "params"}

  {tinyllm} eval --model <rnnlm dir> --data <file> [--tail-frac F]
      RNNLM.nll over the file (or over its last int(n * F) tokens), summed
      with M11.2's NLLAccumulator, one byte per token. Final line:
      {"model": "rnnlm", "ppl", "nll_mean", "bpb", "tokens", "bytes"}

  {tinyllm} generate --model <rnnlm dir> --prompt <text> --seed S --max-tokens N [--out <file>]
      RNNLM.generate from the prompt's bytes at --temperature (default 1).
      Final line {"ids", "text"}, also written to <file> with --out.

Expected results. Over five seeds the RNN's bits per character stay above
the LSTM's, so the ordering RNN > LSTM holds for every seed, and the LSTM's
bits per byte on Pass 3's validation file are below 0.6261, the best of five
NPLMs trained with Pass 3's configuration (0.6261 to 0.6559 bpb). Short
300-step runs show the same ordering.

Corpora: Tiny Shakespeare (Karpathy's char-rnn file, public domain) and
Pass 3's synthetic stand-in for TinyStories.

Why it is needed: the recurrent models are the first that carry state
across a whole sequence. Stateful truncated BPTT is how they train on long
text with bounded memory, and the gap between the RNN and the gated cells
is the vanishing-gradient argument made measurable.

## The encoder-decoders## The encoder-decoders translate dates, and attention closes the gap

The encoder-decoders, run through the system's own entry point: three seq2seq
models (L4.1, its bidirectional encoder from L3.4) trained on the dates
task, without attention, with Bahdanau's (L4.2), and with Luong's (L4.3),
then decoded with the beam search (L4.4) and graded with the sequence
metrics (L4.5), whose deeper test suite is craft.07.

The solution defines the two Pass 4 verbs of the `tinyllm` role ([`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md),
"Verbs of later passes"); exit 2 on a usage error, the last stdout line is
one JSON object. Task files hold one `source<TAB>target` pair per line.

  {tinyllm} train seq2seq --task <pairs.tsv> --attn none|bahdanau|luong --out <dir> --steps S --seed S
                --d-emb E --hidden H --d-attn A --batch B --lr X --clip G
      One L1.1 CharTokenizer over every source and target, specials
      <pad>, <bos>, <eos>, shared by both sides. One init = PCG32(seed)
      .substream("init") builds the attention module (bahdanau:
      AdditiveAttention(H, H, A); luong: LuongAttention(H, "general"))
      and then Seq2Seq(V, V, E, H, "gru", attention). Each of S steps takes
      the next B pairs of an order shuffled per epoch by
      PCG32(seed).substream("shuffle"), and runs L0.5's train_step on the
      teacher-forced cross-entropy (padding targets ignored) with AdamW(lr=X,
      weight_decay=0) and clip G. Writes save_seq2seq and tinyllm_char.json.
      Final line: {"out", "arch": "seq2seq", "attn", "steps", "loss", "params"}

  {tinyllm} translate --model <seq2seq dir> --in <pairs.txt> --beam K
      Decodes every source with the beam_search (beam K, max 16 tokens)
      over decode_step, and again greedily (beam 1); scores both against the
      targets with the exact_match. Final line:
      {"model", "beam", "n", "em", "em_lo", "em_hi" (metric_ci, 1000
       resamples, PCG32(0)), "em_greedy", "em_gain" (em - em_greedy),
       "em_long", "em_short" (sources of at least / under 20 characters),
       "n_long", "chrf"}

Expected results: each attention model reaches EM 0.97 overall and on the
long bucket; the model without attention stays at or below 0.87 on the long
bucket, so the gap is at least 0.1; and on that model beam 5 beats greedy
(em_gain > 0; with attention both decode every date, so there is nothing to
gain). The model without attention gets 600 steps and still trails: the
bottleneck, not the budget. The attention models get 400, because on some
platforms seed 0 of Luong's model reached only 0.968 after 300 (different
float rounding in training). Over seeds 0 to 4: attention EM 0.995 to 1.0
after 300 steps; no attention after 600 steps 0.26 to 0.82 on the long
bucket, em_gain 0.003 to 0.027 (after 300 steps beam lost to greedy on one
seed of five, which is why the weaker model trains longer).

Each training takes 10 to 40 s. Data: 8000 synthetic `source<TAB>target`
date pairs and 600 held-out pairs, 184 with a source of 20 characters or
more.

Why it is needed: a fixed-size encoder state is a bottleneck that long
inputs overflow, and attention removes it by letting every decoder step
look at every encoder position. That is the idea the transformer of Pass 5
is built from.
