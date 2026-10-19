# Licensing ledger verification and the datasheet

After [tokenize and pack](07-tokenize-and-pack.md) there is a corpus, shards, and token streams, and a ledger that [the fetch stage](01-async-fetch-resume-checksums.md) filled as it downloaded: one row per source with its URL, license, and hash. Nothing yet checks that the two agree. A shard row could come from a source the ledger never heard of, a non-commercial source could claim "train", a source revoked last week could still be in the shards, and the `kept` counts still say what fetch saw, not what survived the filters.

When the TinyStories capstone releases a model, the release workflow ([ModelRelease](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) in *AI Platform Engineering*) must answer "may this model be trained on this data?". And when someone asks what is in the corpus, the answer must be a datasheet whose numbers come from the data, not from memory. This section verifies the ledger against the shards, reconciles its counts, and generates the datasheet. The license policy it enforces is argued in [Data licensing and the ledger](../../responsible-ai/01-data-licensing/01-data-licensing-and-the-ledger.md) in *Responsible AI*.

You will use the PII kinds from [the PII scrub](05-pii-scrub.md), `read_shards` and the manifest from [the shard writer](06-parquet-shards-and-manifest.md), and the tokens manifest from the previous section. The solution is [`python/corpus/ledger.py`](../../../../course/solutions/python/corpus/ledger.py) (`read_ledger`, `latest`, `row_errors`, `permitted_uses`, `verify`, `check`, `reconcile`, `datasheet`, the `LedgerError` exception, and the tables `ALLOWLIST`, `EX_DATAERR`). Its contract is [`ledger.pyi`](../../../../course/contracts/py/corpus/ledger.pyi); the rows follow [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json), the datasheet follows [`templates/DATASHEET.md`](../../../../course/contracts/templates/DATASHEET.md), and the exit codes come from [`spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md).

The solution's corpus CLI ([`entry/python/corpus/__main__.py`](../../../../course/solutions/entry/python/corpus/__main__.py)) exposes this code as two verbs: `corpus ledger verify --config C` (runs `check`, exits 65 on a licensing problem) and `corpus datasheet --config C [--out FILE]` (prints the generated datasheet). The generated datasheet becomes `docs/DATASHEET.md` once a person writes its Motivation section; the finished [`DATASHEET.md`](../../../../course/solutions/docs/DATASHEET.md) is an example.

> **Learning objectives**
> - Evaluate SPDX license expressions with `OR` as union and `AND` as intersection, and reject anything outside the allowlist.
> - Verify that every row of training text traces to a ledger row whose license permits the use.
> - Explain why a licensing failure must exit 65 and never be retried.
> - Keep the ledger append-only, with the last row per source winning, and reconcile its counts.
> - Generate a datasheet whose numbers come from the files.

## The allowlist and license expressions

The notation for this section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $U(\ell)$ | the set of uses license $\ell$ permits, a subset of {train, eval} | `frozenset[str]` |
| $\ell_1 \text{ OR } \ell_2$ | the licensee may pick either license | expression |
| $\ell_1 \text{ AND } \ell_2$ | both licenses apply at once | expression |
| row | one ledger line for one source | `dict` |
| $c(s)$ | rows in the shards whose `source_id` is $s$ | `int` |

The allowlist is a policy, written as data. `ALLOWLIST` maps SPDX license ids to the uses they permit. Permissive and attribution licenses (`CC0-1.0`, `CC-BY-4.0`, `CDLA-Permissive-2.0`, `CDLA-Sharing-1.0`, `MIT`, `Apache-2.0`, and others) permit train and eval. Non-commercial and no-derivatives licenses (`CC-BY-NC-4.0`, `CC-BY-ND-4.0`, and others) permit eval only, because training a model is the kind of reuse they exclude. An id not in the table is **unknown**, never guessed.

SPDX writes combined licenses as expressions, and they are evaluated, not pattern-matched. `MIT OR Apache-2.0` is dual licensing, where the licensee chooses, so it permits $U(\text{MIT}) \cup U(\text{Apache-2.0})$. `MIT AND CC-BY-NC-4.0` means both apply, as when a dataset bundles two works, so it permits $U(\text{MIT}) \cap U(\text{CC-BY-NC-4.0})$. `AND` binds tighter than `OR`, so `A OR B AND C` is $U(A) \cup (U(B) \cap U(C))$. Operators are upper case; parentheses and `WITH` exceptions are outside this subset, so they make the expression unknown.

## What verification checks

With licenses evaluable, verification joins the ledger with the shards. `verify(ledger, corpus, use)` returns every problem, and an empty list when the corpus may be used for `use`. It runs five checks:

1. every ledger row follows `formats/ledger.schema.json` (`row_errors`; a bad row is a problem, and its source then has no valid row);
2. for every source's **latest** valid row, its license is known and its `allowed_uses` claim nothing its license does not permit;
3. for a source with shard rows, it is not `revoked`, `use` is in its `allowed_uses`, and every one of its rows carries its license string;
4. every shard row's `source_id` has a ledger row;
5. every source's `kept` equals $c(s)$.

Sources with no rows in the shards (an eval-only benchmark, say) need not permit training. `check` raises `LedgerError(problems)`, whose `exit_code` is `EX_DATAERR = 65`. The upshot is that every row of training text traces to a ledger row whose license permits training, under the same license string, and anything that does not trace fails the build.

### Exit 65 means do not retry

The subprocess activity contract gives the durable engine one rule: 75 (and most other codes) means try again, 65 means the input is wrong and retrying is pointless. A license problem is the textbook data error, since no retry will change a license. Exiting 75 would make `CorpusBuild` retry until its budget runs out and then dead-letter a build that could never pass.

## Append-only history and the datasheet

The ledger is history. Fetch appends a row per fetch. `reconcile(ledger, corpus)` appends, for every source, a copy of its latest row with `kept` $= c(s)$, `dropped` $= \max(0, n_\text{docs} - c(s))$, and the filters applied. A revocation (as in the drill of [Drill: data incident](../../systems/05-incident-response-and-chaos/08-data-incident.md) in *Systems and Architecture*) appends a row with `revoked: true`. `latest` keeps the last row per `source_id`. Nothing is rewritten, so the ledger shows what was decided and when, and a revocation is one appended line.

The datasheet answers the fixed questions Gebru et al. ask of every dataset: motivation, composition, collection, preprocessing, uses, distribution. `datasheet()` fills `templates/DATASHEET.md` with what the files know: documents, splits, languages, and PII counts (Composition); every source with URL, license, retrieval time, and hash (Collection); every drop count and dedup parameter, the decontamination count, and the token counts (Preprocessing); each source's allowed uses (Uses); and the config hash and the ledger path (Distribution and maintenance). It leaves Motivation to a person, because why the dataset exists is not in any file. Since the datasheet is generated from the same manifest and ledger, its numbers cannot drift from the data.

## Working it by hand

Start with what each expression permits:

| License expression | Evaluation | Permits |
|---|---|---|
| `CC-BY-4.0` | listed | train, eval |
| `CC-BY-NC-4.0` | listed | eval |
| `MIT OR CC-BY-NC-4.0` | $\{t, e\} \cup \{e\}$ | train, eval |
| `MIT AND CC-BY-NC-4.0` | $\{t, e\} \cap \{e\}$ | eval |
| `CC-BY-NC-4.0 OR CC-BY-ND-4.0 AND MIT` | $\{e\} \cup (\{e\} \cap \{t, e\})$ | eval |
| `Apache-2.0 WITH LLVM-exception` | `WITH` is outside the subset | unknown: exit 65 |

Now one corpus with two sources. The shards hold `stories:0`, `stories:1` (CC-BY-4.0) and `papers:0` (CC-BY-NC-4.0), and the ledger reads:

| source_id | license_spdx | allowed_uses | n_docs | kept |
|---|---|---|---|---|
| stories | CC-BY-4.0 | train, eval | 2 | 2 |
| papers | CC-BY-NC-4.0 | train | 1 | 1 |

Walk the checks for `use = "train"`. For stories, the license is known, it claims $\{t, e\} \subseteq \{t, e\}$, it is not revoked, train is allowed, both rows say CC-BY-4.0, and kept $2 = c$: no problem. For papers, the license is known, but it claims train, which CC-BY-NC-4.0 does not permit: **one problem**. Train is in its claimed `allowed_uses`, so check 3 passes; the lie is caught at check 2. Every shard row traces to a ledger row. `verify` returns one problem naming papers, and `check` raises `LedgerError` with exit code 65.

## Designing the interface

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

## Building it step by step

Eight observations produce the design.

1. **The question is "may this text be used for this purpose?", asked of every row of text**, so verification must join the shards with the ledger, not inspect the ledger alone. Any shard row whose source has no valid ledger row, or whose license string differs from it, is untraceable.
2. **A license permits a set of uses, and SPDX operators are set operators.** `OR` gives the licensee a choice, so the permitted uses are the union; `AND` imposes both, so they are the intersection. With `AND` binding tighter, splitting on ` OR ` first and then on ` AND ` evaluates the precedence correctly without a parser.
3. **Unknown means unknown.** Any token not in the allowlist (including `WITH`, parentheses, or lower-case operators, which stay glued to ids) makes the whole expression unknown. Guessing would turn a policy into a pattern match.
4. **A claim can be stricter than the license, never looser.** `allowed_uses` must be a subset of what the license permits; a non-commercial source claiming train is a problem even before its rows are checked.
5. **The use check applies only where text exists.** An eval-only benchmark in the ledger is fine until its rows appear in a training corpus.
6. **History must survive, and the latest decision must win.** So the ledger is append-only, `latest` keeps the last row per source, and both `reconcile` and revocation append.
7. **Report everything, then fail once.** `verify` collects every problem in a deterministic order (sorted sources), so a reviewer fixes them all in one pass; `check` turns a non-empty list into exit 65, the code the durable engine never retries.
8. **The datasheet is generated from the same files** (corpus manifest, ledger, tokens manifest), so its numbers are the data's numbers. Only Motivation, which no file knows, is left for a person.

### License evaluation

In [`ledger.py`](../../../../course/solutions/python/corpus/ledger.py):

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

### Reading the ledger

Reading refuses anything it cannot interpret, naming the line:

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

### Verification

The five checks, in order:

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

### Failing and reconciling

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

`datasheet` reads the corpus manifest, the latest ledger rows, and every shard row (for split and language counts), then fills the template's sections in order: Motivation (a placeholder bullet), Composition, Collection (one table row per source), Preprocessing (filter, dedup, decontamination, and token counts when a tokens directory is given), Uses, and Distribution and maintenance. The full solution is [`ledger.py`](../../../../course/solutions/python/corpus/ledger.py).

### Why this design

The cost is one pass over the ledger and one over the shards (`_shard_counts`), so linear in both; memory is a counter per source.

A full SPDX parser (license-expression, ScanCode) handles parentheses and `WITH` exceptions. For a policy gate, rejecting what the subset cannot evaluate is the safe failure: an unknown expression costs a reviewer one look, while a wrongly accepted one costs a license violation. Checking only the ledger would be simpler and would miss the failures that matter: rows from an unlisted source, rows relicensed on the way, and revoked sources still in the shards. The join is the point. Failing fast on the first problem would also be simpler, but it makes a reviewer rerun the build once per problem; collecting all, sorted by source, gives one actionable list. Rewriting the ledger in `reconcile` would keep it short but lose what was fetched, decided, and when; appending costs a few lines per build.

One tradeoff is accepted: the allowlist is a policy table that people must keep current, as the licensing chapter of *Responsible AI* describes, and a license not yet in it fails the build until someone adds it.

## What a correct verifier must handle

The tests pin the rules, the exit code, and the history:

| Case | What must hold | Why it matters |
|---|---|---|
| The hand example | the expressions above and the one problem in the two-source corpus | the rules are exactly the ones above |
| Expressions | union, intersection, AND before OR; parentheses, WITH, lower case, unknown ids are unknown | a license is never guessed |
| Unknown license | `check` raises `LedgerError`, exit code 65, problems name the source | `CorpusBuild` and `ModelRelease` stop instead of retrying |
| Row schema | `row_errors` agrees with `ledger.schema.json` on a valid row and eleven violations | the ledger is a contract too |
| Bad rows | a malformed row is a problem, and its source's rows no longer trace | no crash, no silent pass |
| Traceability | rows from an unlisted source, or under another license string, fail | every row of text traces to a license |
| Latest row | an appended revocation applies; reversed order passes | revocation is one appended line |
| Use only where rows exist | an eval-only source passes until its rows enter a training corpus; a train-only source fails for eval | benchmarks may live in the ledger |
| Kept counts and reconcile | fetch's `kept` fails; `reconcile` appends corrected rows (kept, dropped, filters) and leaves history | the ledger says what survived |
| Reading the ledger | blank lines skipped; a non-object or broken line raises naming it | no silently skipped row |
| The datasheet | the template's sections in order; split, PII, source, drop, and token numbers from the files; Motivation left as a placeholder | the datasheet cannot drift from the data |

The reference suite [`test_ledger_cases.py`](../../../../course/solutions/learner-tests/data.08/python/tests/data-08-ledger/test_ledger_cases.py) covers the hand-example expressions; an unknown license raising `LedgerError` with exit code 65; a clean corpus passing `check`; untraceable rows and a license mismatch failing; a non-commercial source claiming train; the latest row and revocation; schema violations; `reconcile`; `read_ledger` rejecting non-objects; and the datasheet's headings. The exit-code case is the one that catches a retryable exit on a license failure.

## What can go wrong

> **Warning:** Exiting 75 (or 1) on a license failure makes `CorpusBuild` retry a build that can never pass, then dead-letter it. `LedgerError.exit_code` is `EX_DATAERR` (65).

Checking only the ledger, not the shard rows, lets rows from an unlisted source, or rows relicensed on the way, reach training; `_shard_counts` reads every shard row, and checks 3 and 4 compare them with the ledger. Reading the first row of a source ignores a revocation appended later; `latest` keeps the last row per source. Trusting fetch's `kept` makes the datasheet and the release gate count documents the filters removed; check 5 compares `kept` with the shard count, and `reconcile` corrects it.

Expression evaluation has two traps. Treating OR as intersection and AND as union rejects dual-licensed sources and accepts bundled non-commercial ones; the solution splits on ` OR ` for union and ` AND ` for intersection. Skipping unknown parts of an expression, or stripping parentheses, lets `MIT OR GPL-3.0-only` pass as MIT and `(MIT)` pass untested; any id not in the allowlist returns `None` for the whole expression.

The last two are about history. Dropping invalid rows silently makes a malformed row disappear instead of failing the build, so every `row_errors` result becomes a problem. Rewriting the ledger in `reconcile` loses the history of what was fetched and decided, so `reconcile` opens the ledger in append mode.

## Where this leads

This stage reads the PII kinds and their manifest names from the scrub, every row's source and license plus the counts from the shard writer, the token counts from the tokens manifest, and the rows [fetch](01-async-fetch-resume-checksums.md) appended; the policy behind `ALLOWLIST` comes from [Data licensing and the ledger](../../responsible-ai/01-data-licensing/01-data-licensing-and-the-ledger.md). The corpus CLI already runs `check` at the end of every `corpus run`, after `reconcile`.

Several later chapters call it. [ModelRelease](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) runs `check(use="train")` before export and fails the release on exit 65. [Drill: data incident](../../systems/05-incident-response-and-chaos/08-data-incident.md) appends a revocation and purges the derived shards until `verify` passes again. In *Responsible AI*, [Datasheet and model card](../../responsible-ai/03-model-and-data-cards/01-datasheet-and-model-card.md) runs `permitted_uses` over the licensing allowlist on every license the datasheet lists, and its model card links the datasheet; it is the registered call site. [Consent and biometric evidence in model cards](../../responsible-ai/02-privacy-and-pii/02-consent-and-biometric-evidence.md) also calls this implementation.

### Further reading

- Gebru et al., [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010).
- The [SPDX license expression syntax](https://spdx.github.io/spdx-spec/v2.3/SPDX-license-expressions/); the SPDX license list (`spdx/license-list-data`) and ScanCode (`nexB/scancode-toolkit`): parsing full SPDX expressions with exceptions and parentheses, and detecting licenses from file text.
- Longpre et al., [*The Data Provenance Initiative*](https://arxiv.org/abs/2310.16787), and its explorer: per-dataset license, source, and creator audits across thousands of fine-tuning datasets.
- The OpenLineage spec and Hugging Face dataset cards: lineage events per job, queryable across pipelines.
- Hugging Face `DatasetCard` (`huggingface_hub`) and Data Statements (Bender and Friedman): community templates, YAML metadata for search, and language-variety statements.

## Key terms

- **Allowlist**: the policy table mapping SPDX license ids to the uses they permit.
- **SPDX expression**: a combined license where `OR` means choice (union of uses) and `AND` means both apply (intersection).
- **Traceability**: every shard row's source has a valid ledger row under the same license string.
- **Latest row**: the last ledger row for a source, which wins over earlier ones.
- **Reconcile**: appending corrected `kept`, `dropped`, and `filters_applied` values computed from the shards.
- **Datasheet**: the generated document answering Gebru et al.'s questions from the corpus files.

## Summary

- Every row of training text traces to a ledger row whose license permits training, under the same license string; anything that does not trace fails the build.
- License expressions are evaluated, not pattern-matched: `OR` is the union of what each license permits, `AND` the intersection, and anything outside the allowlist is unknown.
- A licensing failure exits 65, a data error, which the durable engine never retries: no retry will change a license.
- The ledger is append-only and its last row per source wins, so a revocation is one appended line.
- The datasheet is generated from the same manifest and ledger, so its numbers cannot drift from the data.

## Review questions

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
