# LLM judge with position control

The code scorers in [Agent evaluation scorers](10-scorers.md) are reliable when the target is explicit, such as an exact answer or a state delta. They cannot judge whether an explanation is useful or whether a response follows a nuanced rubric. A language model can help with that, but only if you treat its output as what it is: a noisy measurement from an instrument that can malfunction, drift, and favour whichever answer it sees first.

This section builds a judge that respects that. You will write a rubric scorer that parses the judge's reply strictly and refuses to invent a score, a pairwise comparison that controls for position bias by asking in both orders, Cohen's kappa for validating the judge against human labels, and a deterministic sampler that cuts cost without changing which cases are judged between runs. The judge plugs into the [eval runner](09-eval-runner.md) as an ordinary `Scorer`, and in production its generator wraps a provider from [Types, OpenAI-compatible provider, retry wrapper](../13-agent-sdk/01-types-and-provider.md). Kappa builds on the ideas in [LLN, CLT, confidence intervals, bootstrap](../../mathematics-for-computing/07-probability-statistics/04-lln-clt-confidence-intervals-bootstrap.md) in *Mathematics for Computing*.

The solution is a single file, [`go/agent/eval/judge/judge.go`](../../../../course/solutions/go/agent/eval/judge/judge.go), which holds `Generator`, `NewJudgeScorer` (rubric scoring with strict parsing), `Pairwise` (position-swapped comparison), `Kappa` (agreement with human labels), `Sampled` (score a deterministic subset), and `ErrJudgeOutput`.

> **Learning objectives**
> - Explain why a malformed judge reply must be an excluded, counted error rather than a score of 0.
> - Parse a judge's JSON reply strictly: one object, known keys, a score in $[0, 1]$, non-empty evidence.
> - Detect position bias by asking a pairwise question in both orders and mapping the second answer back.
> - Compute Cohen's kappa by hand and explain why it can be near 0 when raw agreement is high.
> - Sample a deterministic subset of cases by hashing their ids.

## Treating the judge as an instrument

Ask the judge for a constrained JSON response with a numeric score and short evidence. Reject malformed output, out-of-range values, and missing evidence, and never turn a judge failure into a score of zero. For pairwise comparisons, ask both A/B and B/A, map the second answer back to the original labels, and resolve disagreement with a fixed rule.

When human labels for the same cases are available, measure how well the judge agrees with them using Cohen's kappa,

$$\kappa = \frac{p_o - p_e}{1 - p_e},$$

where $p_o$ is the observed agreement and $p_e$ is the agreement expected by chance from the label marginals. Sampling reduces judge cost; when you sample, report both the number of cases sampled and the tolerance used.

## Working it by hand

Start with a position swap. For a pair of responses, the first prompt shows A then B, and the judge returns `A`. The swapped prompt shows B then A, and the judge again returns `A`, which maps back to the original B. The judge's verdict moved with the position, so the comparison is marked position-sensitive and cannot silently count as a clean win. A judge that prefers the first option regardless of quality produces exactly this pattern on every pair.

Now kappa. Human labels are `[good, good, bad, bad]` and judge labels are `[good, bad, bad, bad]`. Observed agreement is $p_o = 3/4$. The humans say good half the time and the judge a quarter of the time, so chance agreement is $p_e = (2 \cdot 1 + 2 \cdot 3)/16 = 0.5$ (both good, plus both bad). Then $\kappa = (0.75 - 0.5)/(1 - 0.5) = 0.5$: the judge agrees with humans halfway between chance and perfectly, even though raw agreement looks like 75%.

## Designing the API

The package is small:

```go
package judge // import "tinyllm/agent/eval/judge"

var ErrJudgeOutput error
type Generator interface { Generate(ctx context.Context, prompt string) (string, error) }

func NewJudgeScorer(g Generator, rubric string, opts ...JudgeOption) eval.Scorer // name "llm_judge"
type PairwiseResult struct { Winner string; PositionSensitive bool; First, Swapped string }
func Pairwise(ctx context.Context, g Generator, prompt, a, b string) (PairwiseResult, error)
func Kappa(a, b []string) (float64, error)
func Sampled(inner eval.Scorer, n int, tolerance float64) eval.Scorer
```

The generator takes a prompt and returns raw text. That narrow interface means a deterministic fake can stand in for the model in tests without network access, while in production it wraps a provider.

## Building it step by step

Six principles determine the design.

1. **The judge is an instrument, so its failures are missing readings.** A reply that is not the exact JSON asked for carries no score. Turning it into 0 would blame the answer for the judge's mistake. So every failure is `ErrJudgeOutput` (or the generator's error), which the runner excludes and counts.
2. **Parse strictly, because the model will drift.** Models add prose, extra keys, or a second object. Accepting "the first number in the text" would grade `"I'd give it 0.9, but really 0.2"` as 0.9. The decoder therefore accepts exactly one JSON object with known keys, a score in $[0, 1]$, and non-empty evidence.
3. **Position is a nuisance variable; control it by design.** Language models favour one slot. Asking once in each order and mapping the second answer back to the original labels turns position bias into a detectable disagreement instead of a silent win.
4. **Disagreement needs a fixed rule.** If the two orders disagree, neither answer can be trusted; the rule is "tie, and flag it". A fixed rule makes the result independent of which order happened to be asked first.
5. **Validate the judge against people, beyond chance.** Raw agreement overstates quality when one label dominates; kappa subtracts the agreement the marginals alone would produce.
6. **Sampling must be deterministic.** To cut cost, judge a subset; choosing it by a hash of the case id gives the same subset every run, so two runs and an A/B pairing judge the same cases.

### Rubric scoring

The prompt states the exact reply format, and the decoder enforces it.

```go
func (s judgeScorer) Score(ctx context.Context, o eval.Observation) (eval.Score, error) {
	// ... decode the output and the reference answer (JSON strings) into in
	prompt := fmt.Sprintf("Rubric:\n%s\nAnswer:\n%s\nReference:\n%s\nReturn JSON {\"score\": number from 0 to 1, \"evidence\": nonempty string}.", s.rubric, in.Output, in.Reference)
	raw, err := s.g.Generate(ctx, prompt)
	if err != nil {
		return eval.Score{}, err // the endpoint failed: excluded and counted by the runner
	}
	var out struct {
		Score    *float64 `json:"score"` // a pointer, so a missing score is not 0
		Evidence string   `json:"evidence"`
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields() // {"score": 1, "confidence": ...} is not the format asked for
	if dec.Decode(&out) != nil ||
		dec.Decode(new(any)) != io.EOF || // exactly one value: no trailing prose or second object
		out.Score == nil || *out.Score < 0 || *out.Score > 1 ||
		out.Evidence == "" { // a verdict without evidence cannot be audited
		return eval.Score{}, ErrJudgeOutput
	}
	return eval.Score{Value: *out.Score, Reason: out.Evidence}, nil
}
```

The second `Decode` is the cheap way to require "nothing after the object": it must hit end of input. Leading and trailing whitespace is fine; a trailing sentence is not.

### Pairwise with a swap

Pairwise comparison applies the same strict parse to `{"winner": "A" | "B" | "tie"}`, then relabels the second answer.

```go
x, e := ask(a, b) // a in slot A
if e != nil {
	return PairwiseResult{}, e
}
y, e := ask(b, a) // b in slot A: the judge's "A" now means b
if e != nil {
	return PairwiseResult{}, e
}
if y == "A" { // map the swapped answer back to the original labels
	y = "B"
} else if y == "B" {
	y = "A"
}
winner := x
if x != y { // the verdict moved with the position: not a clean win
	winner = "tie"
}
return PairwiseResult{Winner: winner, PositionSensitive: x != y, First: x, Swapped: y}, nil
```

In the hand example, `x = "A"` and the raw swapped answer `"A"` maps to `y = "B"`. Since `x != y`, the result is a tie with `PositionSensitive: true`. Keeping `First` and `Swapped` in the result lets a report count how often the judge is position-sensitive, which measures its bias.

### Kappa

`Kappa` counts marginals and agreements in one pass, then applies the formula. It must handle the degenerate case where chance agreement is already 1.

```go
po := float64(agree) / float64(len(a))
var pe float64
for k, n := range ag { // sum over labels of P(a says k) * P(b says k)
	pe += float64(n*bg[k]) / float64(len(a)*len(a))
}
if pe == 1 { // both raters used one and the same label throughout: 0/0
	if po == 1 {
		return 1, nil
	}
	return 0, nil
}
return (po - pe) / (1 - pe), nil
```

### Sampling

A case is judged when the first byte of the SHA-256 of its id falls in bucket 0 of $n$. The rest return an error naming the tolerance, so the runner excludes them and its `errored` count reports how many were not judged.

```go
h := sha256.Sum256([]byte(o.ID))
bucket := int(h[0]) % s.n // depends only on the case id: same subset every run, in both A/B arms
if bucket != 0 {
	return eval.Score{}, fmt.Errorf("judge: not sampled (tolerance %.4g; id %s)", s.tolerance, hex.EncodeToString(h[:4]))
}
return s.inner.Score(ctx, o)
```

The sampled scorer keeps the inner scorer's name, so its metric lines up with an unsampled run of the same judge.

The full solution is [`judge.go`](../../../../course/solutions/go/agent/eval/judge/judge.go).

### Why this design

The cost is one generator call per scored case and two per pairwise comparison; sampling divides the scored cases by about $n$. Kappa is linear in the number of labels.

You might ask for a 1 to 10 score and divide by 10. Coarse integer scales invite the model to cluster on a few values, and the conversion adds nothing; asking for $[0, 1]$ directly and rejecting anything outside keeps one scale across code and judge scorers.

You might also average the two pairwise orders. But averaging "A wins" and "B wins" into half a win hides that the judge contradicted itself. Flagging the pair keeps the evidence and makes bias countable.

A seeded random sample would also be reproducible, but it would pick different cases when the suite changes. The hash depends only on the case id, so adding cases does not reshuffle the subset.

Three tradeoffs are accepted. First, `h[0] % n` is slightly uneven when 256 is not a multiple of $n$: for $n = 3$, bucket 0 gets 86 of 256 values, about 33.6%. Second, the `tolerance` argument is validated and reported in the message but does not change which cases are chosen. Third, `WithPairwise` is accepted for API compatibility but currently configures nothing; pairwise comparison is the separate `Pairwise` function.

## What can go wrong

A correct judge passes five kinds of checks: the hand-worked position swap and winner mapping, parsing of a valid rubric response, rejection of malformed and out-of-range responses, deterministic sample selection, and kappa against a set of frozen labels. Each one guards against a specific pitfall.

> **Warning:** Extracting the first digit from prose instead of parsing and validating the full response will grade the judge's rambling, not the answer. The solution accepts one JSON object, rejects unknown fields, requires end of input, and checks that the score is in $[0, 1]$ and the evidence is non-empty.

Combining the two pairwise answers without swapping positions and inverting the second result makes position bias invisible; `Pairwise` asks in both orders and maps the second answer's `A`/`B` back before comparing. Mixing judge errors with low scores blames answers for the judge's failures; every parse failure is `ErrJudgeOutput` and every generator failure is returned as an error, so the runner excludes and counts them. Finally, it is easy to read a sampled scorer's uncertainty as if it were the inner scorer's value spread over the full suite. Because unsampled cases are errors, `n` and `errored` in the report show how many cases the interval actually rests on.

## Where this leads

[Paired A/B experiments](12-ab-experiments.md) compare versions using both code and judge scorers, and [Agent evals as tests](../../software-craftsmanship/03-testing-mentality/23-agent-evals-as-tests.md) in *Software Craftsmanship* uses judge scores as evaluation evidence. The eval report for the finished agent includes the judge's sample count, tolerance, and agreement with the frozen labels fixture.

Each piece has a production analogue. The pairwise position swap scales up into LMSYS Chatbot Arena, which aggregates many human comparisons into model rankings. The sampled judge becomes an eval budget policy, with cost controls and a measured quality tolerance. The rubric becomes a human annotation guide that produces calibrated labels for judge validation.

### Further reading

- LMSYS Chatbot Arena, for pairwise comparison at scale.
- Your organization's human annotation guide, as the source of calibrated labels to validate a judge against.

## Key terms

- **LLM judge**: a language model used as a scorer, treated as a noisy instrument whose malformed replies are missing readings.
- **strict parsing**: accepting exactly one JSON object with known keys and valid values, and nothing after it.
- **position bias**: a judge's tendency to prefer whichever answer occupies a particular slot.
- **position-sensitive pair**: a pairwise comparison whose verdict changed when the answers swapped positions; resolved as a flagged tie.
- **Cohen's kappa**: agreement between two raters corrected for the agreement their label marginals would produce by chance.

## Summary

- A judge failure (malformed reply or endpoint error) is an excluded, counted error, never a 0.
- Strict parsing catches trailing prose, extra keys, out-of-range scores, and missing evidence.
- Asking in both orders turns position bias into a countable disagreement; disagreement resolves to a flagged tie.
- Kappa corrects raw agreement for chance: 75% agreement in the hand example is only $\kappa = 0.5$.
- Hash-based sampling judges the same subset every run and in both A/B arms.

## Review questions

1. The judge replies `{"score": 0.8, "evidence": "cites the right section"} I hope this helps!`. What does the scorer return, and why is that right?

   <details><summary>Answer</summary>

   `ErrJudgeOutput`: the second `Decode` finds more input instead of end of file. The reply did not follow the format, so the instrument malfunctioned. Accepting it would teach nothing about the model and would let format drift go unnoticed; excluding and counting it shows up in `errored`.

   </details>

2. A judge always answers `"A"`. What does `Pairwise` report for every pair, and how would you detect the bias across a suite?

   <details><summary>Answer</summary>

   `First = "A"`, the swapped `"A"` maps to `Swapped = "B"`, so every pair is a tie with `PositionSensitive: true`. Across a suite the fraction of position-sensitive pairs is 100%, which is the measurement of the bias. Without the swap, `a` would win every comparison.

   </details>

3. Raw agreement between the judge and humans is 90%, but kappa is near 0. How?

   <details><summary>Answer</summary>

   When one label dominates, say 90% of human labels are `good` and the judge says `good` every time, agreement is 90% by chance alone: $p_e \approx 0.9$, so $\kappa = (0.9 - 0.9)/(1 - 0.9) = 0$. The judge has learned the base rate, not the rubric.

   </details>

4. Why must a skipped case in `Sampled` be an error rather than a score of 0 or a missing row?

   <details><summary>Answer</summary>

   A 0 would read as a bad answer. A missing row would break the runner's case-order slots and the A/B pairing. An error is the runner's existing channel for "no measurement": excluded from the mean, counted in `errored`, so the report shows how many cases were judged.

   </details>
