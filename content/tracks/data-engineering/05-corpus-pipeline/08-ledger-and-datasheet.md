<!-- ol:module data.08 -->
# Licensing ledger verification and the datasheet

## Overview

| | |
|---|---|
| **Module** | `data.08` · build · Python · Pass 3 · 2 to 3 h |
| **You build** | `python/corpus/ledger.py`: `read_ledger`, `latest`, `row_errors`, `permitted_uses`, `verify`, `check`, `reconcile`, `datasheet`, the `LedgerError` exception, and the tables `ALLOWLIST`, `EX_DATAERR` |
| **Contract** | [`course/contracts/py/corpus/ledger.pyi`](../../../../course/contracts/py/corpus/ledger.pyi) · the rows: [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json) · the template: [`templates/DATASHEET.md`](../../../../course/contracts/templates/DATASHEET.md) · exit codes: [`spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md) |
| **Tests** | `course/tests/data.08/` (what they check: section 4) |
| **Needs** | `data.05` the PII kinds · `data.06` `read_shards` and the manifest · `data.07` the tokens manifest (or `--ref-deps`) · reading: `data.01` (it appends the ledger rows), `ethics.01` (the license policy) |
| **Used by** | `ethics.08`; `ethics.03`'s check judges the datasheet's licenses with your `permitted_uses` · later: `dur.12` refuses to release a model whose sources are not licensed for training · `ops.08` purges a revoked source |
| **Milestone** | `MS-corpus` |
| **Optional depth** | Gebru et al., [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010) (free); the [SPDX license expression syntax](https://spdx.github.io/spdx-spec/v2.3/SPDX-license-expressions/) (free); Longpre et al., [*The Data Provenance Initiative*](https://arxiv.org/abs/2310.16787) (free) |

## Key Takeaways

- Every row of training text traces to a ledger row whose license permits training, under the same license string; anything that does not trace fails the build (`test_every_shard_row_traces_to_its_source`).
- License expressions are evaluated, not pattern-matched: `OR` is the union of what each license permits, `AND` the intersection, and anything outside the allowlist is unknown (`test_expressions`).
- A licensing failure exits 65, a data error, which the durable engine never retries: no retry will change a license (`test_unknown_license_is_a_data_error`).
- The ledger is append-only and its last row per source wins, so a revocation is one appended line (`test_latest_row_wins`); the datasheet is generated from the same manifest and ledger, so its numbers cannot drift from the data (`test_datasheet`).

## How to work this chapter

```bash
ol start data.08              # stubs ledger.py into python/corpus/
ol tests data.08              # the course test catalog
ol tdd red data.08            # rung R3: your tests first, failing against the stubs
ol check data.08              # exit code is the verdict
ol check data.08 --ref-deps   # only if data.05, data.06, or data.07 is not passing yet
ol mutate data.08             # how many planted bugs your tests catch
ol diff  data.08              # after passing: your code against the reference
```

Your CLI gains `{corpus} ledger verify` and `{corpus} datasheet` (MS-corpus fixes their flags); copy the generated datasheet to `docs/DATASHEET.md` and write its Motivation section yourself.

---

## 1. Why now

After `data.07` you have a corpus, shards, and token streams, and a ledger that `data.01` filled as it fetched: one row per source with its URL, license, and hash. Nothing checks that the two agree. A shard row could come from a source the ledger never heard of, a non-commercial source could claim "train", a source revoked last week could still be in the shards, and the `kept` counts still say what fetch saw, not what survived the filters. When `C1` releases a model, `dur.12` must answer "may this model be trained on this data?", and when someone asks what is in your corpus, you need a datasheet whose numbers come from the data, not from memory. This module verifies the ledger against the shards, reconciles its counts, and generates the datasheet.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $U(\ell)$ | the set of uses license $\ell$ permits, a subset of {train, eval} | `frozenset[str]` |
| $\ell_1 \text{ OR } \ell_2$ | the licensee may pick either license | expression |
| $\ell_1 \text{ AND } \ell_2$ | both licenses apply at once | expression |
| row | one ledger line for one source | `dict` |
| $c(s)$ | rows in the shards whose `source_id` is $s$ | `int` |

**The allowlist is a policy, written as data.** `ALLOWLIST` maps SPDX license ids to the uses they permit: permissive and attribution licenses (`CC0-1.0`, `CC-BY-4.0`, `CDLA-Permissive-2.0`, `CDLA-Sharing-1.0`, `MIT`, `Apache-2.0`, ...) permit train and eval; non-commercial and no-derivatives licenses (`CC-BY-NC-4.0`, `CC-BY-ND-4.0`, ...) permit eval only, because training a model is the kind of reuse they exclude. `ethics.01` is where that policy is argued; this module enforces it. An id not in the table is **unknown**, never guessed.

**Expressions.** SPDX writes combined licenses as expressions. `MIT OR Apache-2.0` (dual licensing: you choose) permits $U(\text{MIT}) \cup U(\text{Apache-2.0})$. `MIT AND CC-BY-NC-4.0` (both apply, as when a dataset bundles two works) permits $U(\text{MIT}) \cap U(\text{CC-BY-NC-4.0})$. `AND` binds tighter than `OR`, so `A OR B AND C` is $U(A) \cup (U(B) \cap U(C))$. Operators are upper case; parentheses and `WITH` exceptions are outside this subset, so they make the expression unknown.

**What verification checks.** `verify(ledger, corpus, use)` returns every problem, an empty list when the corpus may be used for `use`:

1. every ledger row follows `formats/ledger.schema.json` (`row_errors`; a bad row is a problem, and its source then has no valid row);
2. for every source's **latest** valid row: its license is known; its `allowed_uses` claim nothing its license does not permit;
3. for a source with shard rows: it is not `revoked`, `use` is in its `allowed_uses`, and every one of its rows carries its license string;
4. every shard row's `source_id` has a ledger row;
5. every source's `kept` equals $c(s)$.

Sources with no rows in the shards (an eval-only benchmark) need not permit training. `check` raises `LedgerError(problems)`; its `exit_code` is `EX_DATAERR = 65`.

**Exit 65 means do not retry.** `spec/subprocess-activity.md` gives the durable engine (`data.09`) one rule: 75 (and most other codes) means try again, 65 means the input is wrong and retrying is pointless. A license problem is the textbook data error; exiting 75 would make `CorpusBuild` retry until its budget runs out and then dead-letter a build that could never pass.

**Append-only, last row wins.** The ledger is history. `data.01` appends a row per fetch; `reconcile(ledger, corpus)` appends, for every source, a copy of its latest row with `kept` $= c(s)$, `dropped` $= \max(0, n_\text{docs} - c(s))$, and the filters applied; a revocation (drill `ops.08`) appends a row with `revoked: true`. `latest` keeps the last row per `source_id`. Nothing is rewritten, so the ledger shows what was decided and when.

**The datasheet.** Gebru et al. ask fixed questions of every dataset: motivation, composition, collection, preprocessing, uses, distribution. `datasheet()` fills `templates/DATASHEET.md` with what the files know: documents, splits, languages, PII counts (Composition); every source with URL, license, retrieval time, and hash (Collection); every drop count and dedup parameter, the decontamination count, and the token counts (Preprocessing); each source's allowed uses (Uses); the config hash and the ledger path (Distribution and maintenance). It leaves Motivation to you: why the dataset exists is not in any file.

## 3. Worked example by hand

**What each expression permits.**

| License expression | Evaluation | Permits |
|---|---|---|
| `CC-BY-4.0` | listed | train, eval |
| `CC-BY-NC-4.0` | listed | eval |
| `MIT OR CC-BY-NC-4.0` | $\{t, e\} \cup \{e\}$ | train, eval |
| `MIT AND CC-BY-NC-4.0` | $\{t, e\} \cap \{e\}$ | eval |
| `CC-BY-NC-4.0 OR CC-BY-ND-4.0 AND MIT` | $\{e\} \cup (\{e\} \cap \{t, e\})$ | eval |
| `Apache-2.0 WITH LLVM-exception` | `WITH` is outside the subset | unknown: exit 65 |

**One corpus, two sources.** The shards hold `stories:0`, `stories:1` (CC-BY-4.0) and `papers:0` (CC-BY-NC-4.0). The ledger:

| source_id | license_spdx | allowed_uses | n_docs | kept |
|---|---|---|---|---|
| stories | CC-BY-4.0 | train, eval | 2 | 2 |
| papers | CC-BY-NC-4.0 | train | 1 | 1 |

Walk the checks for `use = "train"`. stories: license known, claims $\{t, e\} \subseteq \{t, e\}$, not revoked, train allowed, both rows say CC-BY-4.0, kept $2 = c$. No problem. papers: license known, but claims train, which CC-BY-NC-4.0 does not permit: **one problem**. (Train is in its claimed `allowed_uses`, so check 3 passes; the lie is caught at check 2.) Every shard row traces to a ledger row. `verify` returns one problem naming papers, and `check` raises `LedgerError` with exit code 65.

This is `test_hand_example`.

## 4. The interface

```python
# python/corpus/ledger.py (the full contract is contracts/py/corpus/ledger.pyi)
EX_DATAERR: int                                   # 65
ALLOWLIST: dict[str, frozenset[str]]

class LedgerError(Exception):                     # .exit_code == 65, .problems
    def __init__(self, problems: list[str]) -> None

def read_ledger(path: Path) -> list[dict]
def latest(rows) -> dict[str, dict]
def row_errors(row) -> list[str]
def permitted_uses(license_spdx: str, allowlist=ALLOWLIST) -> frozenset[str] | None
def verify(ledger: Path, corpus: Path, *, use="train", allowlist=ALLOWLIST) -> list[str]
def check(ledger: Path, corpus: Path, *, use="train", allowlist=ALLOWLIST) -> None
def reconcile(ledger: Path, corpus: Path, *, filters_applied=()) -> list[dict]
def datasheet(ledger: Path, corpus: Path, tokens: Path | None = None) -> str
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | the section 3 expressions and the one problem in the two-source corpus | you and the tests agree on the rules |
| `test_expressions` | unit | union, intersection, AND before OR; parentheses, WITH, lower case, unknown ids are unknown | a license is never guessed |
| `test_unknown_license_is_a_data_error` | boundary | `check` raises `LedgerError`, exit code 65, problems name the source | `data.09` and `dur.12` stop instead of retrying |
| `test_rows_follow_the_schema` | conformance | `row_errors` agrees with `ledger.schema.json` on a valid row and eleven violations | the ledger is a contract too |
| `test_bad_rows_are_reported` | boundary | a malformed row is a problem, and its source's rows no longer trace | no crash, no silent pass |
| `test_every_shard_row_traces_to_its_source` | conformance | rows from an unlisted source, or under another license string, fail | the design's traceability rule |
| `test_latest_row_wins` | unit | an appended revocation applies; reversed order passes | revocation is one appended line |
| `test_use_is_checked_only_where_rows_exist` | unit | an eval-only source passes until its rows enter a training corpus; a train-only source fails for eval | benchmarks may live in the ledger |
| `test_kept_counts_and_reconcile` | unit | fetch's `kept` fails; `reconcile` appends corrected rows (kept, dropped, filters) and leaves history | the ledger says what survived |
| `test_read_ledger` | boundary | blank lines skipped; a non-object or broken line raises naming it | no silently skipped row |
| `test_datasheet` | unit | the template's sections in order; split, PII, source, drop, and token numbers from the files; Motivation left to you | the datasheet cannot drift from the data |

**Your tests (rung R3, red then green).** Under `python/tests/data-08-ledger/`, failing first against the stubs: the hand-example expressions; an unknown license raising `LedgerError` with exit code 65; a clean corpus passing `check`; untraceable rows and a license mismatch failing; a non-commercial source claiming train; the latest row and revocation; schema violations; `reconcile`; `read_ledger` rejecting non-objects; the datasheet's seven headings. `ol mutate data.08` grades them: 0.70 of the mutants, including the one behind Pitfall 1.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. exiting 75 (or 1) on a license failure | `CorpusBuild` retries a build that can never pass, then dead-letters it | `test_unknown_license_is_a_data_error` (mutant `s05`) |
| 2. checking only the ledger, not the shard rows | rows from an unlisted source, or relicensed on the way, reach training | `test_every_shard_row_traces_to_its_source` (mutants `s08`, `s09`) |
| 3. reading the first row of a source | a revocation appended later is ignored | `test_latest_row_wins` (mutant `s10`) |
| 4. trusting fetch's `kept` | the datasheet and the release gate count documents the filters removed | `test_kept_counts_and_reconcile` (mutant `s12`) |
| 5. OR as intersection, AND as union | dual-licensed sources rejected, bundled non-commercial sources accepted | `test_expressions`, `test_hand_example` (mutants `s01`, `s02`) |
| 6. skipping unknown parts of an expression, or stripping parentheses | `MIT OR GPL-3.0-only` passes as MIT; `(MIT)` passes untested | `test_expressions` (mutants `s03`, `s04`) |
| 7. dropping invalid rows silently | a malformed row disappears instead of failing the build | `test_bad_rows_are_reported` (mutant `s07`) |
| 8. rewriting the ledger in `reconcile` | the history of what was fetched and decided is lost | `test_kept_counts_and_reconcile` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.05` | the PII kinds and their manifest names, reported in the datasheet |
| Back | `data.06` | `read_shards` gives every row's source and license; the manifest gives the counts |
| Back | `data.07` | the tokens manifest gives the token count of each split |
| Back | `data.01` | `fetch` appends the rows this module verifies |
| Back | `ethics.01` | the license policy behind `ALLOWLIST` |
| Forward | `dur.12` | `ModelRelease` runs `check(use="train")` before export and fails the release on exit 65 (Pass 9) |
| Forward | `ops.08` | the data-incident drill appends a revocation and purges the derived shards until `verify` passes again (Pass 11) |
| Forward | `ethics.03` | its check runs your `permitted_uses` over the ethics.01 allowlist on every license the datasheet lists; the model card links the datasheet (Pass 9) |

`MS-corpus` already runs `ledger verify` through your CLI; `ethics.03` (Pass 9) is the registered call site.

| Forward | `ethics.08` | The consent and model-card evidence practice calls this implementation. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ALLOWLIST`, `permitted_uses` | the SPDX license list, ScanCode | parsing full SPDX expressions with exceptions and parentheses; detecting licenses from file text | `nexB/scancode-toolkit`, `spdx/license-list-data` |
| `verify` | the Data Provenance Initiative's explorer | per-dataset license, source, and creator audits across thousands of fine-tuning datasets | Longpre et al. 2023 |
| the ledger | dataset cards and lineage stores (OpenLineage, Hugging Face dataset cards) | lineage events per job, queryable across pipelines | the OpenLineage spec |
| `datasheet` | Hugging Face dataset cards, Data Statements (Bender and Friedman) | community templates, YAML metadata for search, language-variety statements | `huggingface_hub` `DatasetCard` |
