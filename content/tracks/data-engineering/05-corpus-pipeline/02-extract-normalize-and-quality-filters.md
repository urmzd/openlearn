<!-- ol:module data.02 -->
# Extract, normalize, quality filters (generator stages)

## Overview

| | |
|---|---|
| **Module** | `data.02` · build · Python · Pass 3 · 4 to 6 h |
| **You build** | `python/corpus/stage.py`: `Doc`, `Stage`, `compose`, `extract`; `python/corpus/filter.py`: `normalize_text`, `normalize_unicode`, `length_filter`, `identify_lang`, `lang_filter`, `gopher_reasons`, `gopher_rules`, `dup_ngram_char_frac`, `repetition_filter`, `ppl_filter`, `build_filters` |
| **Contract** | [`course/contracts/py/corpus/stage.pyi`](../../../../course/contracts/py/corpus/stage.pyi) · [`course/contracts/py/corpus/filter.pyi`](../../../../course/contracts/py/corpus/filter.pyi) · config: the `filters` table of [`corpus-config.schema.json`](../../../../course/contracts/formats/corpus-config.schema.json) |
| **Tests** | `course/tests/data.02/` (what they check: section 4); fixtures `course/fixtures/data.02/` (200 golden documents) |
| **Needs** | `data.01` `read_raw` and the fetch `Manifest` (or `--ref-deps`) · reading: `lang.08` |
| **Used by** | `data.03` exact dedup consumes your stages' output and builds on `Doc` and `compose` · `data.04` near dedup, `data.05` PII scrub, and `data.06` sharding are stages of the same type, and C1 plugs the `L2.1` n-gram model into `ppl_filter` |
| **Milestone** | `MS-corpus` |
| **Optional depth** | Rae et al., *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (2021), appendix A.1 (the quality rules, free); Penedo et al., *The FineWeb Datasets* (2024), section 3; Unicode Standard Annex #15, *Unicode Normalization Forms* (free); David Beazley, "Generator Tricks for Systems Programmers" (free) |

## Key Takeaways

- A stage is a function from an iterator of documents to an iterator of documents. Written as a generator, a whole pipeline holds one document at a time, whatever the corpus size (`test_stages_are_lazy`, `test_pipeline_memory_stays_flat`).
- Normalization is NFC (not NFKC), one line-ending convention, no control characters, and it is **idempotent**: normalizing twice changes nothing (`test_normalize_rules_one_by_one`, `test_normalize_is_idempotent`).
- Language identification by stop-word share is a vote: each stop word votes for every language that lists it, and the winner's share of the votes is the confidence (`test_hand_example_normalize_and_identify`, `test_identify_lang_hand_labelled`).
- The Gopher rules are seven cheap statistics with published thresholds; each one catches a different kind of junk, and every boundary is inclusive (`test_gopher_rules_hand_labelled`, `test_gopher_boundaries_are_the_papers`).
- The configured pipeline is checked end to end on 200 documents whose fate was decided when they were written, byte for byte (`test_golden_200_documents`).

## How to work this chapter

```bash
ol start data.02              # stubs python/corpus/stage.py and filter.py
ol tests data.02              # read the test catalog first
ol check data.02              # needs data.01 passing (or --ref-deps)
ol diff  data.02              # after passing: your code against the reference
```

---

## 1. Why now

`data.01` left you raw documents: every line of every source, exactly as published. Train a tokenizer or a language model on that and it learns whatever the sources contain: the same story with Windows line endings and with Unix ones, accented letters encoded two different ways, pages in German when you wanted English, lists of hashtags, keyword spam that repeats one sentence twenty times. At TinyStories scale the damage is a worse loss curve; at web scale, FineWeb measured each of these filters moving downstream benchmark scores. And the corpus will not fit in memory for long: C1's TinyStories is about 2 GB of text, the pipeline runs inside a durable activity with a memory limit, and a stage that builds a list of all documents is the one that gets the pod killed. This module defines the shape every later stage shares, `Stage = Callable[[Iterator[Doc]], Iterator[Doc]]`, and the first filters built on it.

## 2. Principles

### 2.1 Documents and stages

A `Doc` is immutable: `id` (`<source_id>:<k>`, `k` the document's position in its source), `source_id`, `text`, and `meta` (a mapping of facts about the document: its URL, license, and, as stages add them, its language and scores). A stage that changes a document yields a **new** `Doc` with `dataclasses.replace(doc, text=..., meta={**doc.meta, "lang": "en"})`; it never assigns into `doc.meta`, because the caller may still hold the old document.

A **generator** is a function with `yield` in it. Calling it runs nothing; each `next()` runs it until the next `yield`. So

```python
def length_filter(min_chars, max_chars):
    def stage(docs):
        for doc in docs:
            if min_chars <= len(doc.text) <= max_chars:
                yield doc
    return stage
```

pulls one document from upstream, decides, yields it or not, and pulls the next only when downstream asks. `compose(a, b, c)` returns `lambda docs: c(b(a(docs)))`: three generators stacked, still lazy. Taking 50 documents from the composed pipeline pulls exactly 50 from the source when no stage drops any. The only ways to break this are to collect (`list(docs)`, `sorted(docs)`, `len(list(...))`) or to read ahead in batches; either one makes memory grow with the corpus, and on an endless stream it never returns.

`extract(manifest)` is the first stage's input: for each fetched or cached source of a `data.01` manifest, in manifest order, it reads the raw parts with `read_raw` and yields `Doc(id=f"{source_id}:{k}", ...)` with `meta = {url, fetched_at, license_spdx}`; `k` restarts at 0 for each source. Quarantined sources are skipped by status.

### 2.2 Normalization

The same visible text has many byte encodings. `normalize_text` maps them to one, in this order:

| Step | Rule | Why |
|---|---|---|
| 1 | `\r\n` and lone `\r` become `\n` | Windows and old Mac line endings |
| 2 | remove every character of Unicode category **Cc** (control) except `\n` and `\t`, and remove U+FEFF (the byte order mark) | `\x00`, `\x07`, escape sequences: invisible, never meant as text |
| 3 | Unicode **NFC** | `é` can be one code point (U+00E9) or `e` plus a combining acute accent (U+0301); NFC composes them, so equal text has equal bytes |
| 4 | strip spaces and tabs at the end of each line | invisible differences between copies |
| 5 | runs of 3 or more `\n` become 2 | one blank line separates paragraphs (data.03 splits on it) |
| 6 | strip white space at both ends | |

NFKC would also rewrite "compatibility" characters: the ligature `ﬁ` becomes `fi`, `²` becomes `2`, full-width `Ａ` becomes `A`. That changes what the text says, so this pipeline uses NFC and leaves the tokenizer to see what was written. Removing only category Cc keeps format characters (Cf) such as the zero-width joiner U+200D, which holds multi-person emoji together. The order matters for the law the tests check, **idempotence** (`normalize_text(normalize_text(s)) == normalize_text(s)`): if NFC ran before step 2, deleting a BOM between `a` and a combining accent would leave an uncomposed `a` plus accent that a second pass would compose. A document that normalizes to the empty string is dropped by `normalize_unicode`.

### 2.3 Language identification by stop-word share

Stop words are the most frequent function words of a language ("the", "of", "and"); almost every sentence of ordinary prose holds several. `filter.py` carries a table `STOPWORDS` of 25 per language:

| Code | Stop words |
|---|---|
| `de` | der die das und ist nicht ein eine zu den mit sich des auf für im dem von auch es er sie wir ich war |
| `en` | the of and to a in is it that was he she for on with as his her they at be this have from but |
| `es` | el la los las de y que en un una es por con para del se no lo al su como pero muy yo está |
| `fr` | le la les de des et est un une du en que qui dans pour pas sur au ce il elle nous vous je avec |
| `it` | il la di che e un una per non sono del della con si gli le ma come anche io questo nel è ha lo |
| `nl` | de het een en van is dat op te in met niet zijn voor er maar ook als bij nog ik je wat hij ze |
| `pt` | o a os as de e que do da em um uma para com não no na por mais se ao dos das ele ela |

| Symbol | Meaning | Type |
|---|---|---|
| $w_1, \dots, w_N$ | the **lang words** of the text: maximal runs of letters in `text.lower()` (regex `[^\W\d_]+`) | strings |
| $S_\ell$ | `STOPWORDS[ℓ]` for language code $\ell$ | set of strings |
| $h_\ell = \lvert\{ i : w_i \in S_\ell \}\rvert$ | hits for $\ell$: how many words are stop words of $\ell$ | integer |
| $H = \sum_\ell h_\ell$ | all hits; a word listed by two languages counts for both | integer |
| $\ell^* = \arg\max_\ell h_\ell$ | the identified language; ties go to the alphabetically first code | string |
| $c = h_{\ell^*} / H$ | the confidence | float in $(0, 1]$ |

When $H < 2$ there is too little evidence and the answer is `("und", 0.0)` (undetermined). `lang_filter(min_conf, langs)` keeps a document when $\ell^* \in$ `langs` and $c \ge$ `min_conf`, and records `meta["lang"]` and `meta["lang_conf"]` (data.06 writes them into the shard's `lang` column). The corpus config's default `lang_min_conf = 0.65` is FineWeb's threshold for fastText's language classifier; here it is a share of votes, which is lower than fastText's probability for the same text, so short documents that share many words with Dutch or Portuguese ("in", "a", "is") can fall below it.

### 2.4 The Gopher quality rules

Rae et al. (2021, appendix A.1) filter a web corpus with cheap document statistics. With `words = text.split()` and $n$ = `len(words)`:

| Rule (name in `GOPHER_RULES`) | A document **fails** when | Default thresholds | Catches |
|---|---|---|---|
| `words` | $n <$ `min_words` or $n >$ `max_words` | 50, 100,000 | fragments, dumps |
| `mean_word_len` | $n = 0$, or the mean of `len(w)` is outside [`min_mean_word_len`, `max_mean_word_len`] | 3, 10 | character soup, run-together URLs |
| `symbol_ratio` | `text.count("#") / n` or `(text.count("...") + text.count("…")) / n` exceeds `max_symbol_ratio` | 0.1 | hashtag spam, teaser text |
| `bullet_lines` | among non-blank lines, the fraction whose `lstrip()` starts with `•`, `‣`, `◦`, `-`, or `*` exceeds `max_bullet_lines` | 0.9 | navigation menus, link lists |
| `ellipsis_lines` | among non-blank lines, the fraction whose `rstrip()` ends with `...` or `…` exceeds `max_ellipsis_lines` | 0.3 | truncated previews |
| `alpha_words` | $n = 0$, or the fraction of words containing **at least one** letter is below `min_alpha_words` | 0.8 | tables of numbers, code dumps |
| `stopwords` | fewer than `min_stopwords` words, lowercased and stripped of the characters ``"'.,;:!?()[]{}`` at both ends, are in {the, be, to, of, and, that, have, with} (occurrences, not distinct words) | 2 | non-prose: keyword lists, boilerplate |

Every boundary is inclusive: exactly 50 words, a ratio of exactly 0.1, exactly 30% ellipsis lines all pass. `gopher_reasons(text, **thresholds)` returns the names of the failed rules in the order above; `gopher_rules(**thresholds)` is the stage that keeps documents with no reasons, and it rejects an unknown threshold name when it is **built**, so a typo in the config fails before any document flows.

### 2.5 Repetition

Spam and broken scrapes repeat themselves. Count how much of the text sits inside repeated word n-grams:

| Symbol | Meaning | Type |
|---|---|---|
| $w_0, \dots, w_{L-1}$ | `text.split()` | strings |
| $g_i = (w_i, \dots, w_{i+n-1})$ | the n-gram starting at word $i$, $0 \le i \le L - n$ | tuple |
| duplicated | an n-gram value that occurs at least twice among the $g_i$ | |
| covered | word $w_j$ lies inside some occurrence $g_i$ of a duplicated n-gram ($i \le j < i + n$) | |
| $f_n = \dfrac{\sum_{j \text{ covered}} \lvert w_j \rvert}{\sum_j \lvert w_j \rvert}$ | `dup_ngram_char_frac(text, n)`: the share of characters (spaces not counted) inside repeated n-grams; 0 when $L < n$ | float in $[0, 1]$ |

Every occurrence of a duplicated n-gram counts, the first one included. `repetition_filter(n, max_frac)` keeps documents with $f_n \le$ `max_frac` (config defaults $n = 3$, 0.3).

### 2.6 Perplexity and the configured pipeline

`ppl_filter(lm, max_ppl)` keeps documents a language model does not find too surprising: any object with `perplexity(text) -> float` works, and in C1 the `L2.1` n-gram model trained on clean text plugs in. The score is recorded as `meta["ppl"]`.

`build_filters(filters, lm=None)` turns the corpus config's `filters` table into one stage, defaults from the schema, in this order: `normalize_unicode`, `length_filter(min_chars, max_chars)`, `lang_filter(lang_min_conf, langs)`, `gopher_rules(**gopher)`, `repetition_filter(repetition.n, repetition.max_frac)`, and `ppl_filter(lm, ppl.max_ppl)` when the table has `ppl.max_ppl` (a `ValueError` when it does but no `lm` is passed). Normalization comes first so every later rule counts the characters the model will see.

## 3. Worked example by hand

**Normalize.** Take the 41 code points

```text
U+FEFF  Z o e U+0308 " and the cat"  \r\n \r\n \r\n \r\n  "sat on a mat."  \t " " U+0007
```

Step 1 turns the four `\r\n` into four `\n`. Step 2 removes U+FEFF and the BEL control U+0007. Step 3 composes `e` + U+0308 into `ë` (U+00EB). Step 4 strips the tab and space at the end of the last line. Step 5 turns the four `\n` into two. The result is `Zoë and the cat\n\nsat on a mat.`, and normalizing it again changes nothing.

**Identify.** In "The cat and the dog sat in a box." the lang words are the, cat, and, the, dog, sat, in, a, box:

| Word | `de` | `en` | `es` | `fr` | `it` | `nl` | `pt` |
|---|---|---|---|---|---|---|---|
| the (twice) | | 2 | | | | | |
| and | | 1 | | | | | |
| in | | 1 | | | | 1 | |
| a | | 1 | | | | | 1 |
| total $h_\ell$ | 0 | 5 | 0 | 0 | 0 | 1 | 1 |

$H = 7$, $\ell^* =$ `en`, $c = 5/7 \approx 0.714 \ge 0.65$: kept, with `meta["lang"] = "en"`.

**Gopher.** "Buy now. #deal #sale #wow ..." has 6 words: `words` fails (6 < 50); the mean word length is $(3 + 4 + 5 + 5 + 4 + 3)/6 = 4$, fine; `symbol_ratio` fails ($3/6 = 0.5 > 0.1$, and the ellipsis alone gives $1/6$); its one line ends with `...`, so `ellipsis_lines` fails ($1/1 > 0.3$); 5 of 6 words hold a letter ($0.83 \ge 0.8$); no Gopher stop word appears, so `stopwords` fails. `gopher_reasons` returns `["words", "symbol_ratio", "ellipsis_lines", "stopwords"]`.

**Repetition.** "the cat sat the cat sat down", $n = 3$: the trigrams are (the cat sat), (cat sat the), (sat the cat), (the cat sat), (cat sat down). Only (the cat sat) occurs twice, at $i = 0$ and $i = 3$, so words 0 to 5 are covered: $6 \times 3 = 18$ characters of $18 + 4 = 22$, $f_3 = 18/22 \approx 0.818$.

## 4. The interface

```python
@dataclass(frozen=True, slots=True)
class Doc: id: str; source_id: str; text: str; meta: Mapping[str, Any] = {}
Stage = Callable[[Iterator[Doc]], Iterator[Doc]]
def compose(*stages: Stage) -> Stage: ...
def extract(manifest: Manifest) -> Iterator[Doc]: ...

def normalize_text(text: str) -> str: ...
def normalize_unicode(docs: Iterator[Doc]) -> Iterator[Doc]: ...          # a Stage itself
def length_filter(min_chars: int = 50, max_chars: int = 100_000) -> Stage: ...
def identify_lang(text: str) -> tuple[str, float]: ...
def lang_filter(min_conf: float = 0.65, langs: Sequence[str] = ("en",)) -> Stage: ...
def gopher_reasons(text: str, *, min_words=50, max_words=100_000, min_mean_word_len=3.0,
                   max_mean_word_len=10.0, max_symbol_ratio=0.1, max_bullet_lines=0.9,
                   max_ellipsis_lines=0.3, min_alpha_words=0.8, min_stopwords=2) -> list[str]: ...
def gopher_rules(**thresholds: float) -> Stage: ...
def dup_ngram_char_frac(text: str, n: int) -> float: ...
def repetition_filter(n: int = 3, max_frac: float = 0.3) -> Stage: ...
def ppl_filter(lm: Any, max_ppl: float) -> Stage: ...
def build_filters(filters: Mapping[str, Any], lm: Any = None) -> Stage: ...
```

`STOPWORDS`, `GOPHER_STOPWORDS`, `GOPHER_RULES`, `BULLETS`, and `ELLIPSES` are module constants with the values of section 2. Standard library only (`re`, `unicodedata`, `collections`, `dataclasses`); `extract` reads through `corpus.fetch.read_raw`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_normalize_and_identify` | unit | the section 3 normalization and `("en", 5/7)` | you and the test agree on both definitions |
| `test_normalize_rules_one_by_one` | unit | one hand-labelled case per rule of 2.2, including NFC not NFKC and a kept ZWJ | each rule is what the corpus will look like |
| `test_normalize_is_idempotent` | property | 500 seeded random strings over the rules' special characters | reruns and re-normalized text never drift |
| `test_normalize_unicode_drops_empty_and_keeps_identity` | unit | an empty-after-normalization document is dropped; id and meta kept | no empty rows in shards |
| `test_length_filter_bounds_are_inclusive` | boundary | `min_chars` and `max_chars` both pass | the config's numbers mean what they say |
| `test_identify_lang_hand_labelled` | unit | English, German, French, Dutch, digits, one stop word | the `lang` column of the shards |
| `test_identify_lang_ties_go_to_the_first_code` | boundary | "de de" is `("es", 0.25)` | the answer never depends on dict order |
| `test_identify_lang_uses_the_stopword_table` | property | confidence is $k/(k + j)$ for your own table | the formula, separate from the word lists |
| `test_lang_filter_threshold_is_inclusive_and_records_meta` | boundary | $c =$ `min_conf` is kept; meta set on a new document, input untouched | stages share documents safely |
| `test_gopher_rules_hand_labelled` | unit | one text failing exactly each rule, a clean story, punctuated words, the empty text | each rule catches its own junk |
| `test_gopher_boundaries_are_the_papers` | boundary | 50 words, ratio 0.1, 30% ellipsis lines, 90% bullets all pass | real text at every boundary survives |
| `test_gopher_stopwords_count_occurrences_after_stripping_punctuation` | boundary | "The," and "(the)" count; one occurrence is not enough | prose with punctuation passes |
| `test_gopher_rules_reject_unknown_thresholds_when_built` | boundary | `min_word=10` raises when the stage is built | config typos fail early |
| `test_dup_ngram_hand_example` | unit | the section 3 value $18/22$ and edge cases | the repetition statistic itself |
| `test_repetition_filter_keeps_the_boundary` | boundary | $f_n =$ `max_frac` is kept; `n = 0` rejected | the threshold is inclusive |
| `test_ppl_filter_takes_any_perplexity_object` | unit | a fake model; equal to `max_ppl` kept; `meta["ppl"]` set | C1 plugs in the L2.1 model |
| `test_compose_runs_left_to_right` | unit | `compose(a, b)` is b after a; `compose()` is the identity | normalization runs before counting |
| `test_stages_are_lazy` | property | 50 documents taken from an endless source pull exactly 50 | no stage collects its input |
| `test_pipeline_memory_stays_flat` | property | 2,000 documents through the whole pipeline peak under 0.5 MB of allocations | the pipeline fits an activity's memory limit |
| `test_golden_200_documents` | conformance | 200 generated documents: kept ids, order, `lang`, and the sha256 of every output text | the configured pipeline, byte for byte |
| `test_build_filters_validates_the_table` | boundary | unknown key, unknown Gopher name, `ppl` without a model; `ppl` with a model | reviewed config, early errors |
| `test_gopher_rule_names_are_the_contract` | unit | `GOPHER_RULES` names and order | the config and the datasheet use these names |
| `test_extract_numbers_documents_per_source_across_parts` | unit | ids `a:0` to `a:2` across two parts, then `b:0`; meta from the raw line | stable ids for every later stage |
| `test_extract_skips_quarantined_sources` | boundary | a quarantined entry contributes nothing, even with parts on disk | bytes that failed their checksum never enter |

The golden corpus (`course/oracle/data.02/golden_corpus.py`) has 125 documents to keep, 15 of them noisy copies of good stories (CRLF, NFD accents, a BOM, controls, trailing blanks) whose expected output is the clean story, and 75 to drop, each built to fail one named rule by a wide margin: 5 empty, 5 too short in characters, 15 in German, French, Spanish, or Dutch, 12 under 50 words, 8 hashtag-heavy, 8 bullet lists, 7 trailing off in ellipses, 7 a third numbers, and 8 repeating one sentence six times.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. NFKC instead of NFC | ligatures, superscripts, and full-width text silently rewritten | `test_normalize_rules_one_by_one` (mutant `s01`) |
| 2. Removing every invisible character (category Cf too) | emoji sequences joined by U+200D fall apart | `test_normalize_rules_one_by_one` (mutant `s03`) |
| 3. Composing (NFC) before removing controls | normalizing twice gives a different text | `test_normalize_is_idempotent` (mutant `s05`) |
| 4. `conf > min_conf` | documents exactly at the threshold dropped | `test_lang_filter_threshold_is_inclusive_and_records_meta` (mutant `s09`) |
| 5. `doc.meta["lang"] = ...` on the incoming document | an upstream holder of the document sees meta it never set | `test_lang_filter_threshold_is_inclusive_and_records_meta` (mutant `s10`) |
| 6. `w.isalpha()` for "contains a letter" | every punctuated word ("cat,") counts as non-alphabetic; ordinary prose fails | `test_gopher_rules_hand_labelled` (mutant `s14`) |
| 7. Comparing stop words without lowercasing and stripping punctuation | "The," does not count; short clean documents fail `stopwords` | `test_gopher_stopwords_count_occurrences_after_stripping_punctuation` (mutant `s19`) |
| 8. Ignoring unknown threshold names | a typo in the config silently uses the default | `test_gopher_rules_reject_unknown_thresholds_when_built` (mutant `s20`) |
| 9. A stage that reads its input in batches, or calls `list()` | memory grows with the corpus; on an endless stream it never returns | `test_stages_are_lazy` (mutant `s26`) |
| 10. A stage that keeps every document it has seen | the activity runs out of memory halfway through the corpus | `test_pipeline_memory_stays_flat` (mutant `s32`) |
| 11. Counting only the repeats after the first occurrence | repetition is underestimated by half; spam passes | `test_dup_ngram_hand_example` (mutant `s21`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.01` | `extract` reads the raw parts through `read_raw` and skips quarantined sources from the fetch `Manifest` |
| Forward | `data.03` | `exact_dedup` is a stage over your `Doc`s, run after the filters |
| Forward | `data.04` | near dedup (MinHash, LSH) is a stage over the same `Doc`s, composed after exact dedup |
| Forward | `data.05` | the PII scrub is a stage that returns new `Doc`s with redaction counts in `meta` |
| Forward | `data.06` | the shard writer consumes the final `Doc` stream and writes `meta["lang"]` into the `lang` column |
| Forward | `C1` | `ppl_filter` with the `L2.1` n-gram model trained on clean text |

If you skip this module, `ol check data.03` stops with `data.03 needs data.02`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Stage`, `compose` | datatrove `PipelineStep` and executors | the same iterator-in, iterator-out steps, sharded across processes and Slurm tasks, with per-step stats | `datatrove/pipeline/base.py` |
| `gopher_rules` | datatrove `GopherQualityFilter`; FineWeb's custom filters | the full Gopher set plus FineWeb's line-level rules (lines ending in punctuation, short-line ratio) | `datatrove/pipeline/filters/gopher_quality_filter.py` |
| `repetition_filter` | datatrove `GopherRepetitionFilter` | duplicate lines, paragraphs, top n-gram fractions for n = 2 to 4, duplicate n-gram fractions for n = 5 to 10 | `datatrove/pipeline/filters/gopher_repetition_filter.py` |
| `identify_lang` | fastText `lid.176`, GlotLID | 176 (or 1,600+) languages from character n-grams, calibrated probabilities | fasttext.cc, "Language identification" |
| `ppl_filter` | CCNet's KenLM perplexity buckets | a 5-gram model per language, documents bucketed into head, middle, tail | Wenzek et al. 2019, CCNet |
