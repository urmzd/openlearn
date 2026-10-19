# LLM judge with position control

## Overview

| | |
|---|---|
| **Module** | `ag.11` · build · Go · Pass 10 · 3 h |
| **Solution** | [`go/agent/eval/judge/judge.go`](../../../../course/solutions/go/agent/eval/judge/judge.go): `Generator`, `NewJudgeScorer` (rubric scoring with strict parsing), `Pairwise` (position-swapped comparison), `Kappa` (agreement with human labels), `Sampled` (score a deterministic subset), `ErrJudgeOutput` |
| **Needs** | `ag.01` provider and `ag.09` runner; reading: `M07.4` |
| **Used by** | `ag.12` experiment runner, `craft.23` |

## How to read this chapter

Sections 1 to 3 explain why a model's verdict is a noisy measurement and work a position swap by hand. Section 4 fixes the interface. Section 5 walks through the solution function by function and explains why each choice is the right one; try to predict each step before reading it.

## 1. Why now

Code scorers in ag.10 are reliable when the target is explicit, such as an exact answer or a state delta. They cannot judge whether an explanation is useful or whether a response follows a nuanced rubric. A language model can help, but its output must be treated as a noisy measurement.

## 2. Principles

Ask for a constrained JSON response with a numeric score and short evidence. Reject malformed output, out-of-range values, and missing evidence. Never turn a judge failure into score zero. For pairwise comparisons, ask both A/B and B/A, map the second answer back to the original labels, and resolve disagreement using a fixed rule.

When repeated human labels are available, Cohen's kappa is `κ = (p_o - p_e)/(1 - p_e)`, where `p_o` is observed agreement and `p_e` is agreement expected from the label marginals. Sampling reduces judge cost; report both the number sampled and tolerance used.

## 3. Worked example

For a pair of responses, the first prompt says A then B and returns `A`. The swapped prompt says B then A and returns `A`, which maps to original B. The judge disagrees across positions, so the comparison is marked position-sensitive and cannot silently count as a clean win. A judge that prefers the first option regardless of quality produces exactly this pattern on every pair.

**Kappa by hand.** Human labels `[good, good, bad, bad]`, judge labels `[good, bad, bad, bad]`. Observed agreement $p_o = 3/4$. The humans say good half the time, the judge a quarter of the time, so chance agreement is $p_e = (2 \cdot 1 + 2 \cdot 3)/16 = 0.5$ (both good, plus both bad). Then $\kappa = (0.75 - 0.5)/(1 - 0.5) = 0.5$: the judge agrees with humans halfway between chance and perfectly, even though raw agreement looks like 75%.

## 4. The interface

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

The generator takes a prompt and returns raw text, so a deterministic fake can stand in for the model without network access; in production it wraps an `ag.01` provider.

### What a correct solution must handle

| Case | What it checks |
|---|---|
| position swap | hand-worked position swap and winner mapping |
| parsing evidence | valid rubric response parsing |
| malformed output | malformed and out-of-range responses |
| sampled tolerance | deterministic sample selection |
| kappa against labels | agreement against frozen labels |

## 5. The solution, step by step

### 5.1 From first principles

1. **The judge is an instrument, so its failures are missing readings.** A reply that is not the exact JSON asked for carries no score. Turning it into 0 would blame the answer for the judge's mistake. So every failure is `ErrJudgeOutput` (or the generator's error), which `ag.09` excludes and counts.
2. **Parse strictly, because the model will drift.** Models add prose, extra keys, or a second object. Accepting "the first number in the text" would grade `"I'd give it 0.9, but really 0.2"` as 0.9. The decoder therefore accepts exactly one JSON object with known keys, a score in $[0, 1]$, and non-empty evidence.
3. **Position is a nuisance variable; control it by design.** Language models favour one slot. Asking once in each order and mapping the second answer back to the original labels turns position bias into a detectable disagreement instead of a silent win.
4. **Disagreement needs a fixed rule.** If the two orders disagree, neither answer can be trusted; the rule is "tie, and flag it". A fixed rule makes the result independent of which order happened to be asked first.
5. **Validate the judge against people, beyond chance.** Raw agreement overstates quality when one label dominates; kappa subtracts the agreement the marginals alone would produce.
6. **Sampling must be deterministic.** To cut cost, judge a subset; choosing it by a hash of the case id gives the same subset every run, so two runs and an A/B pairing judge the same cases.

### 5.2 The code, annotated

**Rubric scoring.** The prompt states the exact reply format; the decoder enforces it.

```go
func (s judgeScorer) Score(ctx context.Context, o eval.Observation) (eval.Score, error) {
	// ... decode the output and the reference answer (JSON strings) into in
	prompt := fmt.Sprintf("Rubric:\n%s\nAnswer:\n%s\nReference:\n%s\nReturn JSON {\"score\": number from 0 to 1, \"evidence\": nonempty string}.", s.rubric, in.Output, in.Reference)
	raw, err := s.g.Generate(ctx, prompt)
	if err != nil {
		return eval.Score{}, err // the endpoint failed: excluded and counted by ag.09
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

**Pairwise with a swap.** The same strict parse, applied to `{"winner": "A" | "B" | "tie"}`; then the second answer is relabelled.

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

In section 3, `x = "A"` and the raw swapped answer `"A"` maps to `y = "B"`: `x != y`, so the result is a tie with `PositionSensitive: true`. Keeping `First` and `Swapped` in the result lets a report count how often the judge is position-sensitive, which measures its bias.

**Kappa.** Count marginals and agreements in one pass, then apply the formula; handle the degenerate case where chance agreement is already 1.

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

**Sampling.** A case is judged when the first byte of SHA-256 of its id falls in bucket 0 of $n$; the rest return an error naming the tolerance, so `ag.09` excludes them and its `errored` count reports how many were not judged.

```go
h := sha256.Sum256([]byte(o.ID))
bucket := int(h[0]) % s.n // depends only on the case id: same subset every run, in both A/B arms
if bucket != 0 {
	return eval.Score{}, fmt.Errorf("judge: not sampled (tolerance %.4g; id %s)", s.tolerance, hex.EncodeToString(h[:4]))
}
return s.inner.Score(ctx, o)
```

The sampled scorer keeps the inner scorer's name, so its metric lines up with an unsampled run of the same judge.

Full solution: [`judge.go`](../../../../course/solutions/go/agent/eval/judge/judge.go).

### 5.3 Why this solution

- **Cost.** One generator call per scored case, two per pairwise comparison; sampling divides the scored cases by about $n$. Kappa is O(labels).
- **Why not ask for a 1 to 10 score and divide?** Coarse integer scales invite the model to cluster on a few values and the conversion adds nothing; asking for $[0, 1]$ directly and rejecting anything outside keeps one scale across code and judge scorers.
- **Why not average the two pairwise orders?** Averaging "A wins" and "B wins" into half a win hides that the judge contradicted itself. Flagging the pair keeps the evidence and makes bias countable.
- **Why a hash and not a random sample?** A random sample with a seed would also be reproducible, but would pick different cases when the suite changes; the hash depends only on the case id, so adding cases does not reshuffle the subset.
- **Tradeoffs accepted:** `h[0] % n` is slightly uneven when 256 is not a multiple of $n$ (for $n = 3$, bucket 0 gets 86 of 256 values, about 33.6%). The `tolerance` argument is validated and reported in the message but does not change which cases are chosen. `WithPairwise` is accepted for API compatibility but currently configures nothing; pairwise comparison is the separate `Pairwise` function.

### 5.4 Explain it back

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

A 0 would read as a bad answer. A missing row would break `ag.09`'s case-order slots and the A/B pairing. An error is the runner's existing channel for "no measurement": excluded from the mean, counted in `errored`, so the report shows how many cases were judged.

</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Extracting the first digit from prose instead of parsing and validating the full response. | One JSON object, unknown fields rejected, end of input required, score in $[0, 1]$, evidence non-empty. |
| Combining the two pairwise answers without swapping positions and inverting the second result. | `Pairwise` asks in both orders and maps the second answer's `A`/`B` back before comparing. |
| Mixing judge errors with low scores. | Every parse failure is `ErrJudgeOutput` and every generator failure is returned as an error, so `ag.09` excludes and counts them. |
| Reading a sampled scorer's uncertainty as its inner scorer's value spread. | Unsampled cases are errors, so `n` and `errored` in the report show how many cases the interval rests on. |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | provides the provider used by the judge generator |
| Back | `ag.09` | stores judge results and scorer failures |
| Forward | `ag.12` | compares versions with code and judge scorers |
| Forward | `craft.23` | uses judge scores as evaluation evidence |

The agent pass's eval report includes the judge's sample count, tolerance, and agreement with the frozen labels fixture.

## Going further

| Solution piece | Production equivalent | What it adds |
|---|---|---|
| pairwise position swap | LMSYS Chatbot Arena | many human comparisons aggregated into model rankings |
| sampled judge | eval budget policy | cost controls with a measured quality tolerance |
| rubric | human annotation guide | calibrated labels for judge validation |
