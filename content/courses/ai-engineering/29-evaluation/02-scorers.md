# Agent evaluation scorers

The [eval runner](01-eval-runner.md) executes cases and reports measurements, but it does not say what a good answer is. That is the scorer's job. This section turns the common outcomes of an agent run (the answer text, the timing of the stream, the tools it called, the chunks it retrieved and cited, and the changes it made to the world) into repeatable measurements. Each scorer is small and deterministic, so the same observation always produces the same score, and two eval runs differ only because the subject did.

You will need the runner's `Scorer` and `Observation` types, the JSON Schema validator from [Tools, a registry, and JSON Schema argument validation](../28-agents/02-tools-and-schemas.md), and the citation parsing from [Context Assembly, Citations, and the search_docs Tool](../27-retrieval-and-rag/03-context-assembly.md). The text metrics continue the ideas of [Sequence metrics: exact match, BLEU, chrF](../10-attention/05-sequence-metrics.md) in *AI Engineering*, and the [agent eval harness case study](../32-case-study-agent-evals/) shows the state-change grader in a larger setting. The solution lives in `go/agent/eval/scorers/`: [`text.go`](../../../../course/solutions/go/agent/eval/scorers/text.go) (`Normalize`, `Exact`, `Regex`, `Schema`), [`timing.go`](../../../../course/solutions/go/agent/eval/scorers/timing.go) (`TTFT`, `TTLT`, `ITL`), [`tools.go`](../../../../course/solutions/go/agent/eval/scorers/tools.go) (`ToolSuccess`, `ToolCalled`), [`retrieval.go`](../../../../course/solutions/go/agent/eval/scorers/retrieval.go) (`HitAtK`, `MRR`, `NDCG`, `CitationPrecision`, `CitationRecall`), [`state.go`](../../../../course/solutions/go/agent/eval/scorers/state.go) (`Diff`, `GradeState`, `StateScorer`), and [`registry.go`](../../../../course/solutions/go/agent/eval/scorers/registry.go) (`ByName`). The `ctl eval` verb scores every case with `tool_success` and the case's own `scorer_args`.

> **Learning objectives**
> - Write scorers that are pure functions of the observation and return an error, not a 0, when their input is missing.
> - Compute hit@k, MRR, and nDCG@k by hand over a deduplicated, 1-based ranking.
> - Measure TTFT, TTLT, and inter-token latency from recorded token times.
> - Grade a state change as the set relation $R \subseteq \Delta \subseteq A$, keeping absent and `null` distinct.

## Keeping each measurement narrow

The guiding rule is to measure the narrowest thing that answers the question. Exact match checks normalized answer text. Ranking metrics inspect ordered chunk ids. A state grader checks the resulting state, not the agent's explanation of it. And a missing input for a measurement is an error, because a missing ground truth is not evidence of failure; returning an error lets the runner exclude and count the case.

For retrieval, let the relevance grade at rank $i$ be $g_i$, with ranks starting at one. Then

$$\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{2^{g_i} - 1}{\log_2(i + 1)},$$

and $\mathrm{nDCG@}k$ divides by the DCG of the ideal ranking. $\mathrm{MRR}$ is the reciprocal rank of the first relevant result, and $\mathrm{hit@}k$ is one if any relevant result occurs in the first $k$.

For state changes, compute $\Delta = \text{after} - \text{before}$. The task passes only when the required changes are a subset of $\Delta$ and $\Delta$ is a subset of the allowed changes. That one relation catches both missing work and unrequested work.

## Working the ranking metrics by hand

Ranking metrics are where off-by-one bugs hide, so work one example carefully. Suppose the relevant chunks are `{a: 2, c: 1}` and retrieval returns `[b, a, a, c]`. Repeated ids count at their first rank only, so the ranking is `[b, a, c]`. Then $\mathrm{hit@1} = 0$, $\mathrm{hit@2} = 1$, and $\mathrm{MRR} = 1/2$. DCG uses grades `[0, 2, 1]` at ranks one through three:

$$\mathrm{DCG@3} = \frac{2^0 - 1}{\log_2 2} + \frac{2^2 - 1}{\log_2 3} + \frac{2^1 - 1}{\log_2 4} = 0 + 1.8928 + 0.5 = 2.3928.$$

The ideal grades are `[2, 1]`, so $\mathrm{IDCG} = 3/1 + 1/\log_2 3 = 3.6309$ and $\mathrm{nDCG@3} = 0.6590$. At $k = 2$ the grades are `[0, 2]` and $\mathrm{nDCG@2} = 1.8928 / 3.6309 = 0.5213$. Had the duplicate `a` been kept, `c` would sit at rank 4 and the second `a` would count again at rank 3. Keeping this example hand-computable is what makes rank bugs visible.

## The scorer interface

The constructors live in `go/agent/eval/scorers`: `Exact`, `Regex`, `Schema`, `TTFT`, `TTLT`, `ITL`, `ToolSuccess`, `ToolCalled`, `HitAtK`, `MRR`, `NDCG`, `CitationPrecision`, `CitationRecall`, and `StateScorer`, with `Diff` and `GradeState` exposed for direct use. Each returns an `eval.Scorer` whose name is the one eval specs use (`exact`, `ttft_ms`, `hit@5`, and so on), and `ByName` builds every scorer that needs no configuration from that name. Scorers read only what the subject recorded: the output, `Timing.TokenTimes`, and the annotations `tool_calls`, `retrieved`, `citations`, `state_before`, `state_after`, and `scorer_args`.

## Building the scorers step by step

Five principles shape every scorer in the package.

1. **A score is a function of the observation alone.** No model, no clock, no randomness: then the same observation always gets the same score and two runs differ only because the subject did. Every scorer is a plain function wrapped in a tiny adapter (`fn`) that gives it a name.
2. **Missing input is an error, not a 0.** A case without ground truth, a run that streamed no tokens, a run that dispatched no tools: none of these is evidence about quality. Returning an error lets the runner exclude and count it. So each scorer checks its inputs first and returns a named sentinel (`ErrNoGroundTruth`, `ErrNoTokens`, `ErrNoToolCalls`).
3. **Measure the narrowest thing that answers the question.** Text equality after normalization, not semantic similarity; the first relevant rank, not a guess at usefulness; the state delta, not the agent's description of what it did.
4. **Ranks start at one, and each item counts once.** Every ranking metric depends on positions, so duplicates are removed first, keeping the first occurrence, and the discount uses $\log_2(i + 1)$ for 1-based rank $i$.
5. **A state change is graded as sets.** With $R$ the required changes, $A$ the allowed ones, and $\Delta$ the observed delta, the task passes if and only if $R \subseteq \Delta \subseteq A$. The left inclusion catches missing work, the right one unrequested work. To compute $\Delta$ faithfully, absent and `null` must stay different values.

### Normalization and exact match

`Normalize` is deliberately small: it handles case, outer whitespace and punctuation, and internal runs of whitespace. Internal punctuation is meaning (`fnv-1a` is not `fnv1a`), so it stays.

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

`Regex` compiles a fixed pattern once, or takes `scorer_args.regex.pattern` per case; an empty pattern means "per case", and a case that has none is `ErrNoGroundTruth`. `Schema` compiles the schema once with the tool registry's validator and scores 0 with every violation in the reason. A malformed output is a 0 here, not an error, because producing valid JSON is exactly what is being measured.

### Timing

All three timing scorers read `TokenTimes`, the offsets the runner's subjects record for each streamed piece.

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

For token times of 10, 20, and 40 ms, TTFT is 10, TTLT is 40, and ITL is $(40 - 10)/2 = 15$ ms.

### Tool use

Only calls the gate allowed actually ran, so only they count, and a run with no dispatched calls has no success rate at all.

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

`ToolCalled` scores 1 only for an allowed, successful call to the expected tool, named by `scorer_args.tool_called.name` or the ground truth's `tool`.

### Retrieval

`Retrieved` deduplicates while preserving order, and the metrics then index the deduplicated list.

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

`NDCG(k)` takes the grades of the first $k$ retrieved ids (0 for irrelevant ones) and divides by the DCG of the $k$ best grades in the ground truth, sorted descending. `Relevance` accepts `chunks` as a list (each grade 1) or an object of grades, and drops grades $\le 0$, so the ideal DCG is never 0 and the ratio is always defined. `MRR` returns $1/(i+1)$ at the first relevant 0-based index $i$, and 0 if there is none.

### Citations

`CitationPrecision` resolves the answer's `[n]` markers with `assemble.Resolve` from context assembly, against the run's citation ledger; a number that no block carried counts as a wrong citation. An answer with no markers has no precision (0/0), so it is an error. `CitationRecall` for the same answer is a well-defined 0, since none of the relevant chunks were cited.

### State changes

`Diff` computes $\Delta$: created and deleted rows, and changed fields of kept rows. It iterates over the union of fields so that a removed field is seen, and it distinguishes absent from `null`.

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

`unreadWrites` walks the allowed, successful calls in order, collecting every `id` returned by a read tool's result, and flags a write whose `id` was not returned earlier. The rule is that a read is what came back, not what was asked for. `StateScorer` wraps all this as a 0/1 score with the failed criteria as the reason.

The full solution is in [`go/agent/eval/scorers/`](../../../../course/solutions/go/agent/eval/scorers/).

### Why this design

Every scorer is linear in its input: the output length, or the number of tokens, calls, retrieved ids, or state fields. They add nothing measurable to an eval dominated by model calls.

You might reach for an embedding or LLM similarity to compare answers. It would not be deterministic, and it would need its own validation. Where the target is explicit, code is cheaper and exact; [the LLM judge](03-llm-judge.md) is for what code cannot decide.

Grading state instead of the transcript matters because the agent's claim "I closed ticket 7" can be wrong. The diff of the system is the ground truth of what happened, and the set relation catches over-reach (closing ticket 8 too) that a check of "did ticket 7 close" would miss.

One tradeoff is accepted: normalization is minimal, so `"42"` and `"forty-two"` do not match. Widening it would make the scorer guess; suites that need flexible matching use `Regex` or the judge.

## What can go wrong

A correct scorer package must pass four groups of checks: the hand-computed, duplicate-free ranking example above; normalized exact match and schema rejection for the text scorers; first-token and inter-token timing; and required and forbidden state changes. The pitfalls behind those checks are worth naming one by one.

An answer with no citation has undefined precision, so `CitationPrecision` returns an error when the answer has no markers, while citation recall scores it 0. A missing token stream is not zero latency: `TTFT` and `TTLT` return `ErrNoTokens` when there are no tokens, and `ITL` when there are fewer than two. Ranking over duplicate chunk ids inflates the score, so `Retrieved` removes repeats, keeping each id at its first rank, before any metric runs.

> **Warning:** Treating an absent field and a JSON `null` as the same value hides real changes. `Diff` emits `{"$absent":true}` for a removed field, and `sameJSON` never treats absent as equal to `null`.

Finally, bookkeeping fields such as `updated_at` change on every write and are not user-visible changes. `Diff` skips the `Ignore` fields, `["updated_at"]` by default.

## Where this leads

[Paired A/B experiments](04-ab-experiments.md) compute paired deltas for every scorer defined here, and [Agent evals as tests](../../software-engineering/03-testing/09-agent-evals-as-tests.md) in *Software Engineering* uses agent evaluations as test evidence. The helpdesk-world fixture exercises both text and state-change outcomes, and the eval run over the finished agent reports these scorer results with confidence intervals.

In production, these pieces have larger cousins. Deterministic scorers grow into packages like saige's eval package, with composable and versioned measurements. Retrieval scores grow into tools like Ragas, which adds faithfulness and context precision and recall. The state grader becomes an application-specific oracle with explicit before-and-after invariants.

### Further reading

- The saige eval package, for composable, versioned scorers.
- Ragas, for faithfulness and context precision/recall metrics.

## Key terms

- **scorer**: a deterministic function from one observation to a score, or to an error when its input is missing.
- **normalization**: lowercasing, trimming outer whitespace and punctuation, and collapsing internal whitespace before exact match.
- **nDCG@k**: discounted cumulative gain over the first $k$ ranks, divided by the DCG of the ideal ranking.
- **MRR**: the reciprocal of the rank of the first relevant result.
- **TTFT / TTLT / ITL**: time to first token, time to last token, and mean inter-token latency.
- **state delta ($\Delta$)**: the created rows, deleted rows, and changed fields between the before and after state.

## Summary

- Every scorer is a pure function of the observation; missing input is a named error that the runner excludes and counts.
- Ranking metrics deduplicate first and use 1-based ranks, which the hand example ($\mathrm{nDCG@3} = 0.6590$) pins down.
- Timing scorers read recorded token times and refuse to report latency for a stream that did not happen.
- Tool success counts only calls the gate allowed, and a run with none has no score.
- State changes pass only when $R \subseteq \Delta \subseteq A$, with absent and `null` kept distinct and bookkeeping fields ignored.

## Review questions

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
