# What the System Does After Pass 4

After Pass 4 the CLI trains, orders, and scores recurrent language models (RNN, LSTM, and GRU), and its encoder-decoders translate dates, with attention closing the gap that a fixed-size encoder state leaves on long inputs. Beam search is part of the CLI. Both capabilities report their metrics with bootstrap confidence intervals. Everything from earlier passes keeps working. This page gives the commands the finished solution runs, what each produces, and why each piece is needed.

The recurrent models come from the [Recurrent networks](../../09-recurrent-networks/README.md) chapter: [the vanilla RNN](../../09-recurrent-networks/01-vanilla-rnn-manual-bptt.md), [the LSTM](../../09-recurrent-networks/02-lstm.md), [the GRU](../../09-recurrent-networks/03-gru.md), [the bidirectional RNN](../../09-recurrent-networks/04-bidirectional-rnn.md), and [the RNN language model with stateful TBPTT](../../09-recurrent-networks/06-rnn-language-model-stateful-tbptt.md). The encoder-decoders come from [Attention origins](../../10-attention/README.md): [the encoder-decoder with teacher forcing](../../10-attention/01-encoder-decoder-teacher-forcing.md), [Bahdanau attention](../../10-attention/02-bahdanau-additive-attention.md), [Luong attention](../../10-attention/03-luong-attention-input-feeding.md), [beam search](../../10-attention/04-beam-search.md), and [the sequence metrics](../../10-attention/05-sequence-metrics.md), whose deeper test suite is [Mutation testing in depth](../../../software-engineering/03-testing/05-mutation-testing-in-depth.md). The statistics under both are [LLN, CLT, confidence intervals, and the bootstrap](../../../mathematics/13-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md) and [its problem set](../../../mathematics/13-probability-statistics/11-problem-set-c.md).

## Recurrent language models, trained, ordered, and scored

Three recurrent language models run through the system's own entry point: an Elman RNN (its time loop fused into one op by the stateful-TBPTT section), an LSTM, and a GRU, each trained with stateful truncated BPTT on Tiny Shakespeare. The LSTM samples text, and an LSTM trained on Pass 3's story corpus and scored on its validation file beats Pass 3's NPLM in bits per byte. The bidirectional encoder is used by the encoder-decoders in the next section.

The solution defines the Pass 4 language-model verbs of the `tinyllm` role under "Verbs of later passes" in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md): exit 2 on a usage error, and the last stdout line is one JSON object. Text files are read as UTF-8 bytes, and `.bin` files as `formats/tokens-bin.md` shards of byte ids (decision D32).

```text
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
    with the NLLAccumulator, one byte per token. Final line:
    {"model": "rnnlm", "ppl", "nll_mean", "bpb", "tokens", "bytes"}

{tinyllm} generate --model <rnnlm dir> --prompt <text> --seed S --max-tokens N [--out <file>]
    RNNLM.generate from the prompt's bytes at --temperature (default 1).
    Final line {"ids", "text"}, also written to <file> with --out.
```

The expected results are orderings, not just numbers. Over five seeds the RNN's bits per character stay above the LSTM's, so the ordering RNN > LSTM holds for every seed. The LSTM's bits per byte on Pass 3's validation file are below 0.6261, the best of five NPLMs trained with Pass 3's configuration (which range from 0.6261 to 0.6559 bpb). Short 300-step runs show the same ordering. The corpora are Tiny Shakespeare (Karpathy's char-rnn file, public domain) and Pass 3's synthetic stand-in for TinyStories.

Why this matters: the recurrent models are the first that carry state across a whole sequence. Stateful truncated BPTT is how they train on long text with bounded memory, and the gap between the RNN and the gated cells is the vanishing-gradient argument made measurable.

## The encoder-decoders translate dates, and attention closes the gap

Three seq2seq models, each with the bidirectional encoder, are trained on a dates task: one without attention, one with Bahdanau's, and one with Luong's. They are then decoded with beam search and graded with the sequence metrics.

The solution defines two Pass 4 verbs of the `tinyllm` role in [`cli-roles.md`](../../../../../course/contracts/spec/cli-roles.md), with the same rules. Task files hold one `source<TAB>target` pair per line.

```text
{tinyllm} train seq2seq --task <pairs.tsv> --attn none|bahdanau|luong --out <dir> --steps S --seed S
              --d-emb E --hidden H --d-attn A --batch B --lr X --clip G
    One CharTokenizer over every source and target, specials <pad>, <bos>,
    <eos>, shared by both sides. One init = PCG32(seed).substream("init")
    builds the attention module (bahdanau: AdditiveAttention(H, H, A);
    luong: LuongAttention(H, "general")) and then
    Seq2Seq(V, V, E, H, "gru", attention). Each of S steps takes the next B
    pairs of an order shuffled per epoch by PCG32(seed).substream("shuffle"),
    and runs the training loop's train_step on the teacher-forced
    cross-entropy (padding targets ignored) with AdamW(lr=X,
    weight_decay=0) and clip G. Writes save_seq2seq and tinyllm_char.json.
    Final line: {"out", "arch": "seq2seq", "attn", "steps", "loss", "params"}

{tinyllm} translate --model <seq2seq dir> --in <pairs.txt> --beam K
    Decodes every source with the beam_search (beam K, max 16 tokens) over
    decode_step, and again greedily (beam 1); scores both against the
    targets with the exact_match. Final line:
    {"model", "beam", "n", "em", "em_lo", "em_hi" (metric_ci, 1000
     resamples, PCG32(0)), "em_greedy", "em_gain" (em - em_greedy),
     "em_long", "em_short" (sources of at least / under 20 characters),
     "n_long", "chrf"}
```

Each attention model reaches an exact match of 0.97 overall and on the long bucket. The model without attention stays at or below 0.87 on the long bucket, so the gap is at least 0.1, and on that model beam 5 beats greedy (`em_gain > 0`). With attention both decoders get every date right, so there is nothing for the beam to gain.

The step budgets are deliberate. The model without attention gets 600 steps and still trails, which shows that the bottleneck, not the budget, is the problem. The attention models get 400, because on some platforms seed 0 of Luong's model reached only 0.968 after 300 steps (different float rounding in training). Over seeds 0 to 4, the attention models reach an exact match of 0.995 to 1.0 after 300 steps; the model without attention, after 600 steps, reaches 0.26 to 0.82 on the long bucket, with `em_gain` from 0.003 to 0.027. After only 300 steps, beam search lost to greedy on one seed of five, which is why the weaker model trains longer.

Each training takes 10 to 40 seconds. The data are 8000 synthetic `source<TAB>target` date pairs and 600 held-out pairs, 184 of which have a source of 20 characters or more.

Why this matters: a fixed-size encoder state is a bottleneck that long inputs overflow, and attention removes it by letting every decoder step look at every encoder position. That is the idea [Pass 5](../course-p05-transformer/README.md) builds the transformer from.
