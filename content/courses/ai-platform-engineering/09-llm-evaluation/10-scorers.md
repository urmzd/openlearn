# Agent evaluation scorers

## Overview

| | |
|---|---|
| **Module** | `ag.10` · build · Go · Pass 10 · 3 to 4 h |
| **Solution** | `go/agent/eval/scorers/`: [`text.go`](../../../../course/solutions/go/agent/eval/scorers/text.go) (`Normalize`, `Exact`, `Regex`, `Schema`), [`timing.go`](../../../../course/solutions/go/agent/eval/scorers/timing.go) (`TTFT`, `TTLT`, `ITL`), [`tools.go`](../../../../course/solutions/go/agent/eval/scorers/tools.go) (`ToolSuccess`, `ToolCalled`), [`retrieval.go`](../../../../course/solutions/go/agent/eval/scorers/retrieval.go) (`HitAtK`, `MRR`, `NDCG`, `CitationPrecision`, `CitationRecall`), [`state.go`](../../../../course/solutions/go/agent/eval/scorers/state.go) (`Diff`, `GradeState`, `StateScorer`), [`registry.go`](../../../../course/solutions/go/agent/eval/scorers/registry.go) (`ByName`) |
| **Needs** | `ag.09`, `ag.02`, `ag.08`; reading: `L4.5` and [case study 05](../../case-studies/05-agent-eval-harness/) |
| **Used by** | `ag.12` experiment runner, `craft.23`; the `ctl eval` verb scores every case with `tool_success` and the case's own `scorer_args` |

## How to read this chapter

Sections 1 to 3 say what each family of scorers measures and work the retrieval metrics by hand. Section 4 lists the interface and the cases a correct scorer must handle. Section 5 walks through the solution scorer by scorer and explains why each one is written the way it is; try to predict each step before reading it.

## 1. Why now

The runner from ag.09 can execute cases and report measurements, but it does not define what a good answer means. This module turns common agent outcomes into repeatable measurements. Each scorer is small and deterministic, so the same observation always produces the same score.

## 2. Principles

Keep the measurement narrow. Exact match checks normalized answer text. Ranking metrics inspect ordered chunk ids. A state grader checks the resulting state, not the agent's explanation. A missing input for a measurement is an error, because a missing ground truth is not evidence of failure.

For retrieval, let relevance grade at rank `i` be `g_i`. Then `DCG@k = sum((2^g_i - 1) / log2(i + 1))`, for ranks starting at one. `nDCG@k` divides by the ideal ranking's DCG. `MRR` is the reciprocal rank of the first relevant result; `hit@k` is one if any relevant result occurs in the first k.

For state changes, compute `delta = after - before`. The task passes only when required changes are a subset of delta and delta is a subset of allowed changes. This catches both missing work and unrequested work.

## 3. Worked example

Suppose relevant chunks are `{a: 2, c: 1}` and retrieval returns `[b, a, a, c]`. Repeated ids count at their first rank only, so the ranking is `[b, a, c]`. `hit@1 = 0` and `hit@2 = 1`; `MRR = 1/2`. DCG uses grades `[0, 2, 1]` at ranks one through three:

$$\mathrm{DCG@3} = \frac{2^0 - 1}{\log_2 2} + \frac{2^2 - 1}{\log_2 3} + \frac{2^1 - 1}{\log_2 4} = 0 + 1.8928 + 0.5 = 2.3928.$$

The ideal grades are `[2, 1]`: $\mathrm{IDCG} = 3/1 + 1/\log_2 3 = 3.6309$, so $\mathrm{nDCG@3} = 0.6590$. At $k = 2$ the grades are `[0, 2]` and $\mathrm{nDCG@2} = 1.8928 / 3.6309 = 0.5213$. Had the duplicate `a` been kept, `c` would sit at rank 4 and the second `a` would count again at rank 3; keeping this example hand-computable makes off-by-one rank bugs visible.

## 4. The interface

The constructors live in `go/agent/eval/scorers`: `Exact`, `Regex`, `Schema`, `TTFT`, `TTLT`, `ITL`, `ToolSuccess`, `ToolCalled`, `HitAtK`, `MRR`, `NDCG`, `CitationPrecision`, `CitationRecall`, and `StateScorer` (with `Diff` and `GradeState` exposed for direct use). Each returns an `eval.Scorer` whose name is the one eval specs use (`exact`, `ttft_ms`, `hit@5`, ...); `ByName` builds every scorer that needs no configuration from that name. Scorers read what the subject recorded: the output, `Timing.TokenTimes`, and the annotations `tool_calls`, `retrieved`, `citations`, `state_before`, `state_after`, and `scorer_args`.

### What a correct solution must handle

| Case | What it checks |
|---|---|
| the ranking hand example | hand-computed duplicate-free ranking metrics |
| text and schema scorers | normalized exact match and schema rejection |
| timing metrics | first token and inter-token timing |
| state-change criteria | required and forbidden state changes |

## 5. The solution, step by step

### 5.1 From first principles

1. **A score is a function of the observation alone.** No model, no clock, no randomness: then the same observation always gets the same score and two runs differ only because the subject did. Every scorer is a plain function wrapped in a tiny adapter (`fn`) that gives it a name.
2. **Missing input is an error, not a 0.** A case without ground truth, a run that streamed no tokens, a run that dispatched no tools: none of these is evidence about quality. Returning an error lets `ag.09` exclude and count it. So each scorer checks its inputs first and returns a named sentinel (`ErrNoGroundTruth`, `ErrNoTokens`, `ErrNoToolCalls`).
3. **Measure the narrowest thing that answers the question.** Text equality after normalization, not semantic similarity; the first relevant rank, not a guess at usefulness; the state delta, not the agent's description of what it did.
4. **Ranks start at one, and each item counts once.** Every ranking metric depends on positions, so duplicates are removed first, keeping the first occurrence, and the discount uses $\log_2(i + 1)$ for 1-based rank $i$.
5. **A state change is graded as sets.** With $R$ the required changes, $A$ the allowed ones, and $\Delta$ the observed delta, pass iff $R \subseteq \Delta \subseteq A$. The left inclusion catches missing work, the right one unrequested work. To compute $\Delta$ faithfully, absent and `null` must stay different values.

### 5.2 The code, annotated

**Normalization and exact match.** `Normalize` is deliberately small: case, outer whitespace and punctuation, and internal runs of whitespace. Internal punctuation is meaning (`fnv-1a` is not `fnv1a`), so it stays.

```go
func Normalize(s string) string {
	s = strings.ToLower(s)
	// Trailing "." or leading quotes are formatting, not content.
	s = strings.TrimFunc(s, func(r rune) bool { return unicode.IsSpace(r) || unicode.IsPunct(r) })
	return strings.Join(strings.Fields(s), " ") // collapse every internal whitespace run
}

func Exact() eval.Scorer {
	return fn{"exact", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		want, ok := answer(o.GroundTruth) // a string, or an object's "answer" string
		if !ok {
			return eval.Score{}, ErrNoGroundTruth // no answer key: excluded, not failed
		}
		if Normalize(OutputText(o)) == Normalize(want) {
			return eval.Score{Value: 1}, nil
		}
		return eval.Score{Value: 0, Reason: fmt.Sprintf("want %q", want)}, nil
	}}
}
```

`Regex` compiles a fixed pattern once, or takes `scorer_args.regex.pattern` per case (an empty pattern means "per case"; a case that has none is `ErrNoGroundTruth`). `Schema` compiles the schema once with `ag.02`'s validator and scores 0 with every violation in the reason. A malformed output is a 0 here, not an error: producing valid JSON is exactly what is being measured.

**Timing.** All three read `TokenTimes`, the offsets `ag.09`'s subjects record per streamed piece.

```go
// TTFT: the first arrival. TTLT: the last. No tokens: ErrNoTokens, never 0 ms.
func ITL() eval.Scorer {
	return fn{"itl_ms", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		t := o.Timing.TokenTimes
		if len(t) < 2 { // one token has no gap to measure
			return eval.Score{}, ErrNoTokens
		}
		// The mean of the n-1 gaps telescopes to (last - first) / (n - 1).
		return eval.Score{Value: millis(t[len(t)-1]-t[0]) / float64(len(t)-1)}, nil
	}}
}
```

For token times 10, 20, 40 ms: TTFT 10, TTLT 40, ITL $(40 - 10)/2 = 15$ ms.

**Tool use.** Only calls the gate allowed actually ran, so only they count. A run with no dispatched calls has no success rate.

```go
func ToolSuccess() eval.Scorer {
	return fn{"tool_success", func(_ context.Context, o eval.Observation) (eval.Score, error) {
		calls, err := ToolCalls(o) // the tool_calls annotation; absent = no calls
		if err != nil {
			return eval.Score{}, err
		}
		ran, ok := 0, 0
		for _, c := range calls {
			if c.Verdict != "allow" { // denied or held for approval: never executed
				continue
			}
			ran++
			if !c.IsError {
				ok++
			}
		}
		if ran == 0 {
			return eval.Score{}, ErrNoToolCalls // not a perfect score
		}
		return eval.Score{Value: float64(ok) / float64(ran)}, nil
	}}
}
```

`ToolCalled` scores 1 only for an allowed, successful call to the expected tool (from `scorer_args.tool_called.name` or the ground truth's `tool`).

**Retrieval.** `Retrieved` deduplicates while preserving order; the metrics then index the deduplicated list.

```go
seen := map[string]bool{}
out := ids[:0] // filter in place: keeps the first occurrence, in rank order
for _, id := range ids {
	if !seen[id] {
		seen[id] = true
		out = append(out, id)
	}
}

func DCG(grades []float64) float64 {
	var s float64
	for i, g := range grades { // i is 0-based; rank is i+1, discount log2(rank+1) = log2(i+2)
		s += (math.Pow(2, g) - 1) / math.Log2(float64(i+2))
	}
	return s
}
```

`NDCG(k)` takes the grades of the first $k$ retrieved ids (0 for irrelevant ones) and divides by the DCG of the $k$ best grades in the ground truth, sorted descending. `Relevance` accepts `chunks` as a list (each grade 1) or an object of grades and drops grades $\le 0$, so the ideal DCG is never 0 and the ratio is always defined. `MRR` returns $1/(i+1)$ at the first relevant index, 0 if none.

**Citations.** `CitationPrecision` resolves the answer's `[n]` markers with `ag.08`'s `assemble.Resolve` against the run's citation ledger; a number no block carried counts as a wrong citation. An answer with no markers has no precision (0/0), so it is an error; `CitationRecall` for the same answer is a well-defined 0 (none of the relevant chunks cited).

**State changes.** `Diff` computes $\Delta$: created and deleted rows, and changed fields of kept rows. It iterates the union of fields so a removed field is seen, and it distinguishes absent from `null`.

```go
for f := range fields { // union of before and after fields
	if skip[f] { // bookkeeping such as updated_at is never a task-visible change
		continue
	}
	old, oldOK := b[f]
	nw, newOK := a[f]
	switch {
	case oldOK && !newOK:
		delta[row+"."+f] = json.RawMessage(Absent) // gone, which is not the same as set to null
	case !sameJSON(old, nw): // compares by meaning: 5 and 5.0 are equal; absent never equals null
		delta[row+"."+f] = nw
	}
}
```

`GradeState` then checks the two inclusions and, when asked, read-before-write:

```go
for k, want := range t.Required { // R ⊆ Δ, with the required value
	if got, ok := delta[k]; !ok || !sameJSON(got, want) {
		g.Criteria[RequiredChanges] = append(g.Criteria[RequiredChanges], "missing "+k)
	}
}
for k := range delta { // Δ ⊆ A ("ticket/7.*" allows every field of that row)
	if !allowed(k, t.Allowed) {
		g.Criteria[OnlyRequestedChanges] = append(g.Criteria[OnlyRequestedChanges], "unauthorized "+k)
	}
}
if t.ReadBeforeWrite {
	g.Criteria[ReadBeforeWrite] = append([]string{}, unreadWrites(calls, cfg)...)
}
// ... sort each criterion's evidence; Passed iff every list is empty
```

`unreadWrites` walks the allowed, successful calls in order, collecting every `id` returned by a read tool's result, and flags a write whose `id` was not returned earlier: "a read is what came back, not what was asked for". `StateScorer` wraps this as a 0/1 score with the failed criteria as the reason.

Full solution: [`go/agent/eval/scorers/`](../../../../course/solutions/go/agent/eval/scorers/).

### 5.3 Why this solution

- **Cost.** Every scorer is linear in its input: the output length, the number of tokens, calls, retrieved ids, or state fields. They add nothing measurable to an eval dominated by model calls.
- **Why not an embedding or LLM similarity for answers?** It would not be deterministic and it would need its own validation. Where the target is explicit, code is cheaper and exact; the judge of `ag.11` is for what code cannot decide.
- **Why grade state instead of the transcript?** The agent's claim "I closed ticket 7" can be wrong. The diff of the system is the ground truth of what happened, and the set relation catches over-reach (closing ticket 8 too) that a check of "did ticket 7 close" would miss.
- **Tradeoff accepted:** normalization is minimal, so `"42"` and `"forty-two"` do not match. Widening it would make the scorer guess; suites that need flexible matching use `Regex` or the judge.

### 5.4 Explain it back

1. Why does `CitationPrecision` return an error for an answer with no citations, while `CitationRecall` returns 0?

<details><summary>Answer</summary>

Precision is good citations over all citations: with no citations it is 0/0, undefined, and reporting 0 or 1 would invent a measurement. Recall is cited relevant chunks over relevant chunks: the denominator is the ground truth, which is non-empty, so the answer is a true 0.

</details>

2. What does nDCG@3 become for `[b, a, a, c]` if duplicates are not removed?

<details><summary>Answer</summary>

The first three items are `[b, a, a]` with grades `[0, 2, 2]`: $\mathrm{DCG} = 3/\log_2 3 + 3/\log_2 4 = 1.8928 + 1.5 = 3.3928$, and $\mathrm{nDCG} = 3.3928 / 3.6309 = 0.934$, far above the correct 0.659. Repeating a relevant chunk would be rewarded, which is why `Retrieved` deduplicates first.

</details>

3. Why does `ToolSuccess` skip calls whose verdict is not `allow`, and why is a run with none an error?

<details><summary>Answer</summary>

A denied or held call never executed, so it has no success or failure; counting it would blame tools for the gate's decision. A run that executed nothing has no ratio to report (0/0); scoring it 1 would reward an agent for never trying.

</details>

4. A task requires `ticket/7.status = "closed"` and allows `ticket/7.*`. The agent closes ticket 7 and also sets `ticket/8.priority`. Which criterion fails, and what would a check of only the required changes have reported?

<details><summary>Answer</summary>

`only_requested_changes` fails with `unauthorized ticket/8.priority`, because that key is in $\Delta$ but not in $A$. A required-only check would pass the run, rewarding an agent that changes things it was not asked to.

</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| An answer with no citation has undefined precision. | `CitationPrecision` returns an error when the answer has no markers; citation recall scores it 0. |
| A missing token stream is not zero latency. | `TTFT` and `TTLT` return `ErrNoTokens` for no tokens, `ITL` for fewer than two. |
| Ranking over duplicate chunk ids. | `Retrieved` removes repeats, keeping each id at its first rank, before any metric runs. |
| Treating an absent field and a JSON `null` as the same. | `Diff` emits `{"$absent":true}` for a removed field, and `sameJSON` never equals absent with `null`. |
| Counting bookkeeping fields such as `updated_at` as user-visible changes. | `Diff` skips the `Ignore` fields, `["updated_at"]` by default. |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.09` | provides the observation, scorer, and runner interfaces |
| Back | `ag.02` | provides the JSON Schema validator behind `Schema` |
| Back | `ag.08` | provides citation parsing and resolution |
| Forward | `ag.12` | computes paired deltas for every scorer |
| Forward | `craft.23` | uses agent evaluations as test evidence |

The helpdesk-world fixture exercises both text and state-change outcomes, and the agent pass's eval run reports the scorer results with confidence intervals.

## Going further

| Solution piece | Production equivalent | What it adds |
|---|---|---|
| deterministic scorers | saige eval package | composable and versioned measurements |
| retrieval scores | Ragas | faithfulness and context precision/recall |
| state grader | application-specific oracle | explicit before/after invariants |
