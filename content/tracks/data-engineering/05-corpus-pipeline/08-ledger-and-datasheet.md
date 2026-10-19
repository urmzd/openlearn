# Licensing ledger verification and the datasheet

## Overview

| | |
|---|---|
| **Module** | `data.08` · build · Python · Pass 3 · 2 to 3 h |
| **Solution** | [`python/corpus/ledger.py`](../../../../course/solutions/python/corpus/ledger.py): `read_ledger`, `latest`, `row_errors`, `permitted_uses`, `verify`, `check`, `reconcile`, `datasheet`, the `LedgerError` exception, and the tables `ALLOWLIST`, `EX_DATAERR` |
| **Contract** | [`course/contracts/py/corpus/ledger.pyi`](../../../../course/contracts/py/corpus/ledger.pyi) · the rows: [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json) · the template: [`templates/DATASHEET.md`](../../../../course/contracts/templates/DATASHEET.md) · exit codes: [`spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md) |
| **Needs** | `data.05` the PII kinds · `data.06` `read_shards` and the manifest · `data.07` the tokens manifest · reading: `data.01` (it appends the ledger rows), `ethics.01` (the license policy) |
| **Used by** | `ethics.08`; `ethics.03`'s check judges the datasheet's licenses with `permitted_uses` · later: `dur.12` refuses to release a model whose sources are not licensed for training · `ops.08` purges a revoked source |
| **Optional depth** | Gebru et al., [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010) (free); the [SPDX license expression syntax](https://spdx.github.io/spdx-spec/v2.3/SPDX-license-expressions/) (free); Longpre et al., [*The Data Provenance Initiative*](https://arxiv.org/abs/2310.16787) (free) |

## Key Takeaways

- Every row of training text traces to a ledger row whose license permits training, under the same license string; anything that does not trace fails the build.
- License expressions are evaluated, not pattern-matched: `OR` is the union of what each license permits, `AND` the intersection, and anything outside the allowlist is unknown.
- A licensing failure exits 65, a data error, which the durable engine never retries: no retry will change a license.
- The ledger is append-only and its last row per source wins, so a revocation is one appended line; the datasheet is generated from the same manifest and ledger, so its numbers cannot drift from the data.

## How to read this chapter

Sections 1 to 3 build license evaluation, the five verification checks, the append-only ledger, and the datasheet from first principles and verify a two-source corpus by hand. Section 4 fixes the interface and lists the cases a correct solution must handle, including the model test suite. Section 5 walks through `ledger.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The solution's corpus CLI ([`entry/python/corpus/__main__.py`](../../../../course/solutions/entry/python/corpus/__main__.py)) exposes this module as two verbs: `corpus ledger verify --config C` (runs `check`, exits 65 on a licensing problem) and `corpus datasheet --config C [--out FILE]` (prints the generated datasheet). The generated datasheet becomes `docs/DATASHEET.md` once a person writes its Motivation section; see the finished [`DATASHEET.md`](../../../../course/solutions/docs/DATASHEET.md) for an example.

---

## 1. Why now

After `data.07` there is a corpus, shards, and token streams, and a ledger that `data.01` filled as it fetched: one row per source with its URL, license, and hash. Nothing checks that the two agree. A shard row could come from a source the ledger never heard of, a non-commercial source could claim "train", a source revoked last week could still be in the shards, and the `kept` counts still say what fetch saw, not what survived the filters. When `C1` releases a model, `dur.12` must answer "may this model be trained on this data?", and when someone asks what is in the corpus, the answer must be a datasheet whose numbers come from the data, not from memory. This module verifies the ledger against the shards, reconciles its counts, and generates the datasheet.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $U(\ell)$ | the set of uses license $\ell$ permits, a subset of {train, eval} | `frozenset[str]` |
| $\ell_1 \text{ OR } \ell_2$ | the licensee may pick either license | expression |
| $\ell_1 \text{ AND } \ell_2$ | both licenses apply at once | expression |
| row | one ledger line for one source | `dict` |
| $c(s)$ | rows in the shards whose `source_id` is $s$ | `int` |

**The allowlist is a policy, written as data.** `ALLOWLIST` maps SPDX license ids to the uses they permit: permissive and attribution licenses (`CC0-1.0`, `CC-BY-4.0`, `CDLA-Permissive-2.0`, `CDLA-Sharing-1.0`, `MIT`, `Apache-2.0`, ...) permit train and eval; non-commercial and no-derivatives licenses (`CC-BY-NC-4.0`, `CC-BY-ND-4.0`, ...) permit eval only, because training a model is the kind of reuse they exclude. `ethics.01` is where that policy is argued; this module enforces it. An id not in the table is **unknown**, never guessed.

**Expressions.** SPDX writes combined licenses as expressions. `MIT OR Apache-2.0` (dual licensing: the licensee chooses) permits $U(\text{MIT}) \cup U(\text{Apache-2.0})$. `MIT AND CC-BY-NC-4.0` (both apply, as when a dataset bundles two works) permits $U(\text{MIT}) \cap U(\text{CC-BY-NC-4.0})$. `AND` binds tighter than `OR`, so `A OR B AND C` is $U(A) \cup (U(B) \cap U(C))$. Operators are upper case; parentheses and `WITH` exceptions are outside this subset, so they make the expression unknown.

**What verification checks.** `verify(ledger, corpus, use)` returns every problem, an empty list when the corpus may be used for `use`:

1. every ledger row follows `formats/ledger.schema.json` (`row_errors`; a bad row is a problem, and its source then has no valid row);
2. for every source's **latest** valid row: its license is known; its `allowed_uses` claim nothing its license does not permit;
3. for a source with shard rows: it is not `revoked`, `use` is in its `allowed_uses`, and every one of its rows carries its license string;
4. every shard row's `source_id` has a ledger row;
5. every source's `kept` equals $c(s)$.

Sources with no rows in the shards (an eval-only benchmark) need not permit training. `check` raises `LedgerError(problems)`; its `exit_code` is `EX_DATAERR = 65`.

**Exit 65 means do not retry.** `spec/subprocess-activity.md` gives the durable engine (`data.09`) one rule: 75 (and most other codes) means try again, 65 means the input is wrong and retrying is pointless. A license problem is the textbook data error; exiting 75 would make `CorpusBuild` retry until its budget runs out and then dead-letter a build that could never pass.

**Append-only, last row wins.** The ledger is history. `data.01` appends a row per fetch; `reconcile(ledger, corpus)` appends, for every source, a copy of its latest row with `kept` $= c(s)$, `dropped` $= \max(0, n_\text{docs} - c(s))$, and the filters applied; a revocation (drill `ops.08`) appends a row with `revoked: true`. `latest` keeps the last row per `source_id`. Nothing is rewritten, so the ledger shows what was decided and when.

**The datasheet.** Gebru et al. ask fixed questions of every dataset: motivation, composition, collection, preprocessing, uses, distribution. `datasheet()` fills `templates/DATASHEET.md` with what the files know: documents, splits, languages, PII counts (Composition); every source with URL, license, retrieval time, and hash (Collection); every drop count and dedup parameter, the decontamination count, and the token counts (Preprocessing); each source's allowed uses (Uses); the config hash and the ledger path (Distribution and maintenance). It leaves Motivation to a person: why the dataset exists is not in any file.

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

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| The hand example | unit | the section 3 expressions and the one problem in the two-source corpus | the rules are exactly the ones above |
| Expressions | unit | union, intersection, AND before OR; parentheses, WITH, lower case, unknown ids are unknown | a license is never guessed |
| Unknown license | boundary | `check` raises `LedgerError`, exit code 65, problems name the source | `data.09` and `dur.12` stop instead of retrying |
| Row schema | conformance | `row_errors` agrees with `ledger.schema.json` on a valid row and eleven violations | the ledger is a contract too |
| Bad rows | boundary | a malformed row is a problem, and its source's rows no longer trace | no crash, no silent pass |
| Traceability | conformance | rows from an unlisted source, or under another license string, fail | every row of text traces to a license |
| Latest row | unit | an appended revocation applies; reversed order passes | revocation is one appended line |
| Use only where rows exist | unit | an eval-only source passes until its rows enter a training corpus; a train-only source fails for eval | benchmarks may live in the ledger |
| Kept counts and reconcile | unit | fetch's `kept` fails; `reconcile` appends corrected rows (kept, dropped, filters) and leaves history | the ledger says what survived |
| Reading the ledger | boundary | blank lines skipped; a non-object or broken line raises naming it | no silently skipped row |
| The datasheet | unit | the template's sections in order; split, PII, source, drop, and token numbers from the files; Motivation left as a placeholder | the datasheet cannot drift from the data |

**The model test suite.** [`test_ledger_cases.py`](../../../../course/solutions/learner-tests/data.08/python/tests/data-08-ledger/test_ledger_cases.py) covers the hand-example expressions; an unknown license raising `LedgerError` with exit code 65; a clean corpus passing `check`; untraceable rows and a license mismatch failing; a non-commercial source claiming train; the latest row and revocation; schema violations; `reconcile`; `read_ledger` rejecting non-objects; the datasheet's headings. The exit-code case is the one that catches pitfall 1.

## 5. The solution, step by step

### 5.1 From first principles

1. **The question is "may this text be used for this purpose?", asked of every row of text**, so verification must join the shards with the ledger, not inspect the ledger alone. Any shard row whose source has no valid ledger row, or whose license string differs from it, is untraceable.
2. **A license permits a set of uses, and SPDX operators are set operators.** `OR` gives the licensee a choice, so the permitted uses are the union; `AND` imposes both, so they are the intersection. With `AND` binding tighter, splitting on ` OR ` first and then on ` AND ` evaluates the precedence correctly without a parser.
3. **Unknown means unknown.** Any token not in the allowlist (including `WITH`, parentheses, or lower-case operators, which stay glued to ids) makes the whole expression unknown. Guessing would turn a policy into a pattern match.
4. **A claim can be stricter than the license, never looser.** `allowed_uses` must be a subset of what the license permits; a non-commercial source claiming train is a problem even before its rows are checked.
5. **The use check applies only where text exists.** An eval-only benchmark in the ledger is fine until its rows appear in a training corpus.
6. **History must survive, and the latest decision must win.** So the ledger is append-only, `latest` keeps the last row per source, and both `reconcile` and revocation append.
7. **Report everything, then fail once.** `verify` collects every problem in a deterministic order (sorted sources), so a reviewer fixes them all in one pass; `check` turns a non-empty list into exit 65, the code the durable engine never retries.
8. **The datasheet is generated from the same files** (corpus manifest, ledger, tokens manifest), so its numbers are the data's numbers. Only Motivation, which no file knows, is left for a person.

### 5.2 The code, annotated

License evaluation ([`ledger.py`](../../../../course/solutions/python/corpus/ledger.py)):

```python
def permitted_uses(license_spdx, allowlist=ALLOWLIST) -> Optional[frozenset[str]]:
    tokens = license_spdx.split()
    if not tokens:
        return None
    result: frozenset[str] = frozenset()
    for alternative in " ".join(tokens).split(" OR "):   # normalized spaces; OR binds loosest
        ids = alternative.split(" AND ")
        term: Optional[frozenset[str]] = None
        for lid in ids:
            if lid not in allowlist:
                return None        # "(MIT)", "Apache-2.0 WITH ...", "MIT or X": all unknown
            uses = frozenset(allowlist[lid])
            term = uses if term is None else term & uses     # AND: intersection
        result |= term or frozenset()                        # OR: union
    return result
```

Reading the ledger refuses anything it cannot interpret, naming the line:

```python
def read_ledger(path: Path) -> list[dict[str, Any]]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: not JSON: {e}") from None
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{i}: a ledger line must be a JSON object")
        rows.append(row)
    return rows


def latest(rows):
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r["source_id"]] = dict(r)   # later rows overwrite earlier: last row wins
    return out
```

`row_errors` checks every required key, rejects unknown keys, and validates each field's type and shape (lower-case source ids, RFC 3339 UTC times, 64-hex hashes, non-negative integers that are not `bool`, the closed vocabularies of `allowed_uses` and `pii_policy`), mirroring `ledger.schema.json`.

The verification itself, the five checks of section 2 in order:

```python
def verify(ledger, corpus, *, use="train", allowlist=ALLOWLIST) -> list[str]:
    problems: list[str] = []
    good = []
    for i, row in enumerate(read_ledger(ledger), 1):
        errs = row_errors(row)
        if errs:
            problems.append(f"ledger row {i}: " + "; ".join(errs))   # 1: reported, not dropped silently
        else:
            good.append(row)
    cur = latest(good)                       # the latest VALID row per source
    counts, licenses = _shard_counts(corpus) # rows and license strings per source, from read_shards
    for sid in sorted(cur):
        row = cur[sid]
        lic = permitted_uses(row["license_spdx"], allowlist)
        if lic is None:
            problems.append(f"source {sid}: license {row['license_spdx']!r} is unknown (not in the allowlist)")
            continue
        extra = sorted(set(row["allowed_uses"]) - lic)               # 2: claims within the license
        if extra:
            problems.append(f"source {sid}: allowed_uses claims {extra}, which {row['license_spdx']} does not permit")
        if counts[sid]:                                              # 3: only where text exists
            if row.get("revoked", False):
                problems.append(f"source {sid}: revoked, but {counts[sid]} shard rows come from it")
            if use not in row["allowed_uses"]:
                problems.append(f"source {sid}: not allowed for {use}, but {counts[sid]} shard rows come from it")
            bad = sorted(licenses[sid] - {row["license_spdx"]})      # every row under the ledger's license
            if bad:
                problems.append(f"source {sid}: shard rows say license {bad}, the ledger says {row['license_spdx']!r}")
        if row["kept"] != counts[sid]:                               # 5: kept is what survived
            problems.append(f"source {sid}: ledger kept {row['kept']}, the shards hold {counts[sid]} rows")
    for sid in sorted(set(counts) - set(cur)):                       # 4: every shard row traces
        problems.append(f"source {sid}: {counts[sid]} shard rows, but no valid ledger row")
    return problems
```

Because `read_shards` verifies each shard's hash against the corpus manifest, the rows counted here are the rows the tokenizer saw.

Failing and reconciling:

```python
class LedgerError(Exception):
    exit_code = EX_DATAERR               # 65: a data error, never retried

def check(ledger, corpus, *, use="train", allowlist=ALLOWLIST) -> None:
    problems = verify(ledger, corpus, use=use, allowlist=allowlist)
    if problems:
        raise LedgerError(problems)

def reconcile(ledger, corpus, *, filters_applied=()):
    cur = latest(read_ledger(ledger))
    counts, _ = _shard_counts(corpus)
    rows = []
    for sid in sorted(cur):
        row = dict(cur[sid])                                  # a copy of the latest row
        row["kept"] = counts[sid]
        row["dropped"] = max(0, int(row["n_docs"]) - counts[sid])
        row["filters_applied"] = list(filters_applied)
        rows.append(row)
    with open(ledger, "a", encoding="utf-8") as f:            # append: history is kept
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows
```

`datasheet` reads the corpus manifest, the latest ledger rows, and every shard row (for split and language counts), then fills the template's sections in order: Motivation (a placeholder bullet), Composition, Collection (one table row per source), Preprocessing (filter, dedup, decontamination, and token counts when a tokens directory is given), Uses, and Distribution and maintenance.

Full solution: [`ledger.py`](../../../../course/solutions/python/corpus/ledger.py).

### 5.3 Why this solution

- **Cost.** One pass over the ledger and one over the shards (`_shard_counts`), so linear in both; memory is a counter per source.
- **A full SPDX parser** (license-expression, ScanCode) handles parentheses and `WITH` exceptions. For a policy gate, rejecting what the subset cannot evaluate is the safe failure: an unknown expression costs a reviewer one look, a wrongly accepted one costs a license violation.
- **Checking only the ledger** would be simpler and miss the failures that matter: rows from an unlisted source, rows relicensed on the way, and revoked sources still in the shards. The join is the point.
- **Fail-fast on the first problem** would be simpler, but makes a reviewer rerun the build once per problem. Collecting all, sorted by source, gives one actionable list.
- **Rewriting the ledger** in `reconcile` would keep it short, but loses what was fetched, decided, and when. Appending costs a few lines per build.
- **Accepted tradeoff.** The allowlist is a policy table that must be kept current by people (`ethics.01`); a license not yet in it fails the build until someone adds it.

### 5.4 Explain it back

1. Why does `MIT OR CC-BY-NC-4.0` permit training but `MIT AND CC-BY-NC-4.0` does not?
   <details><summary>Answer</summary>`OR` lets the licensee choose MIT, so the permitted uses are the union $\{t, e\} \cup \{e\} = \{t, e\}$. `AND` means both licenses bind at once, so only uses both permit are allowed: $\{t, e\} \cap \{e\} = \{e\}$.</details>
2. Why is a license failure exit 65 rather than 75?
   <details><summary>Answer</summary>The subprocess activity contract tells the durable engine that 75 means "try again" and 65 means "the input is wrong". No retry changes a license, so 75 would make `CorpusBuild` retry until its budget runs out and then dead-letter a build that could never pass.</details>
3. In the hand example, why is the problem with `papers` reported by check 2 and not check 3?
   <details><summary>Answer</summary>Check 3 asks whether `use` is in the claimed `allowed_uses`, and the row claims train, so it passes. Check 2 asks whether the claim is within what the license permits; CC-BY-NC-4.0 permits only eval, so claiming train is the problem.</details>
4. How is a source revoked, and why does that need no rewrite of the ledger?
   <details><summary>Answer</summary>A copy of its latest row with `revoked: true` is appended. `latest` keeps the last row per source, so the revocation wins, and `verify` fails while any shard row still comes from that source.</details>
5. Why is the source with shard rows but only an invalid ledger row reported as untraceable?
   <details><summary>Answer</summary>Invalid rows are reported and excluded from `good` before `latest`, so the source has no valid row in `cur`; the final loop then reports its shard rows as having no valid ledger row. A malformed row cannot license anything.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. exiting 75 (or 1) on a license failure | `CorpusBuild` retries a build that can never pass, then dead-letters it | `LedgerError.exit_code = EX_DATAERR` (65) |
| 2. checking only the ledger, not the shard rows | rows from an unlisted source, or relicensed on the way, reach training | `_shard_counts` reads every shard row; checks 3 and 4 compare them with the ledger |
| 3. reading the first row of a source | a revocation appended later is ignored | `latest` keeps the last row per source |
| 4. trusting fetch's `kept` | the datasheet and the release gate count documents the filters removed | check 5 compares `kept` with the shard count; `reconcile` corrects it |
| 5. OR as intersection, AND as union | dual-licensed sources rejected, bundled non-commercial sources accepted | split on ` OR ` for union, ` AND ` for intersection |
| 6. skipping unknown parts of an expression, or stripping parentheses | `MIT OR GPL-3.0-only` passes as MIT; `(MIT)` passes untested | any id not in the allowlist returns `None` for the whole expression |
| 7. dropping invalid rows silently | a malformed row disappears instead of failing the build | every `row_errors` result becomes a problem |
| 8. rewriting the ledger in `reconcile` | the history of what was fetched and decided is lost | `reconcile` opens the ledger in append mode |

## 7. Where it's used next


| Direction | Module | How it uses this |
|---|---|---|
| Back | `data.05` | the PII kinds and their manifest names, reported in the datasheet |
| Back | `data.06` | `read_shards` gives every row's source and license; the manifest gives the counts |
| Back | `data.07` | the tokens manifest gives the token count of each split |
| Back | `data.01` | `fetch` appends the rows this module verifies |
| Back | `ethics.01` | the license policy behind `ALLOWLIST` |
| Forward | `dur.12` | `ModelRelease` runs `check(use="train")` before export and fails the release on exit 65 (Pass 9) |
| Forward | `ops.08` | the data-incident drill appends a revocation and purges the derived shards until `verify` passes again (Pass 11) |
| Forward | `ethics.03` | its check runs `permitted_uses` over the ethics.01 allowlist on every license the datasheet lists; the model card links the datasheet (Pass 9) |
| Forward | `ethics.08` | the consent and model-card evidence practice calls this implementation |

The corpus CLI already runs `check` at the end of every `corpus run` (after `reconcile`); `ethics.03` (Pass 9) is the registered call site.

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ALLOWLIST`, `permitted_uses` | the SPDX license list, ScanCode | parsing full SPDX expressions with exceptions and parentheses; detecting licenses from file text | `nexB/scancode-toolkit`, `spdx/license-list-data` |
| `verify` | the Data Provenance Initiative's explorer | per-dataset license, source, and creator audits across thousands of fine-tuning datasets | Longpre et al. 2023 |
| the ledger | dataset cards and lineage stores (OpenLineage, Hugging Face dataset cards) | lineage events per job, queryable across pipelines | the OpenLineage spec |
| `datasheet` | Hugging Face dataset cards, Data Statements (Bender and Friedman) | community templates, YAML metadata for search, language-variety statements | `huggingface_hub` `DatasetCard` |
