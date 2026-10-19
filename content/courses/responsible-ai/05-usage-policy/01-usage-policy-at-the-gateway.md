# Usage policy at the gateway

## Overview

| | |
|---|---|
| **Module** | `ethics.05` · practice · policy and documentation · Pass 10 · 1 to 2 h |
| **Solution** | the policy document [`course/solutions/docs/USAGE_POLICY.md`](../../../../course/solutions/docs/USAGE_POLICY.md) and the ordered rules [`course/solutions/entry/deploy/policy.v1.yaml`](../../../../course/solutions/entry/deploy/policy.v1.yaml), enforced by `gw.08`'s evaluator in [`go/gateway/policy/policy.go`](../../../../course/solutions/go/gateway/policy/policy.go) (`Check`, `Probabilities`, `Middleware`) |
| **Contract** | [`policy.v1.schema.json`](../../../../course/contracts/formats/policy.v1.schema.json), [`linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json) |
| **Needs** | `gw.08` usage policy enforcement, `L6.5` linear head, `data.05` privacy detectors |
| **Used by** | the agent pass's end-to-end run, which exercises the policy matrix, the red-team prompts, the audit log, and gateway behaviour |
| **Optional depth** | NIST AI Risk Management Framework, Govern and Measure functions |

## Key takeaways

- Policy rules are ordered and the first matching rule wins.
- Model and tenant restrictions are explicit; classifier rules use the engine embedding endpoint and a versioned linear head.
- A classifier or embeddings failure fails closed with 503.
- Every deny has a stable rule id, a useful reason, and a privacy-safe audit record.
- The policy describes permitted use; it does not claim the classifier can determine intent or replace human review.

## How to read this chapter

Sections 1 to 3 explain why the gateway needs a policy layer, how the linear head turns an embedding into a probability, and work the two kinds of decision by hand. Section 4 fixes what the artifact must contain. Section 5 walks through the reference rules, the evaluator that enforces them, and the policy document, and explains why each choice is right. Try to predict each decision before reading it.

## 1. Why now

The agent now reaches users through the gateway, and its tool gates cannot prevent every harmful inference request. A policy layer gives operators a place to set model, tenant, token, and classifier restrictions, and gives users a consistent explanation when a request is blocked.

## 2. Principles

The policy document is configuration, not a promise that a model is safe. Keep rules narrow, explain their scope, and review false positives and false negatives. The linear head scores normalized embeddings using `z = W e + b`; the softmax probability for class `c` is `exp(z_c) / sum_j exp(z_j)`. Its training labels, threshold, embedding model, and held-out metrics belong in the model card. The Go gateway evaluates the exported head so Python is not in the request path.

| Symbol | Meaning |
|---|---|
| `e` | normalized embedding returned by engine `/v1/embeddings` |
| `W` | classifier weights, one row per class |
| `b` | one bias per class |
| `z` | class logits |
| `τ` | probability threshold that activates a classifier rule |

## 3. Worked example

Suppose a request has `max_tokens = 2048`, tenant `free`, and matches a rule capped at 1024 tokens. That rule denies before any classifier call. For a separate request, logits `[0, 2]` give unsafe probability `e²/(1+e²) ≈ 0.881`; with threshold `0.9`, that classifier rule allows it. The red-team fixtures exercise both cases, including the embeddings service failing.

## 4. The artifact

`USAGE_POLICY.md` states scope, responsible owner, appeal path, review cadence, model and data limitations, and the actions users can expect. The ordered rules follow the schema and include model restrictions, a token cap, and the fitted classifier head at `models/smol-135m/heads/usage.json`. The document explains why each red-team fixture is denied without copying sensitive user content into audit logs.

### What a correct solution must handle

| Case | Why it exists | Expected result |
|---|---|---|
| schema and rule order | Rejects ambiguous or misspelled rules | First matching deny wins |
| red-team prompts denied | Checks configured refusals at the policy boundary | Every blocked fixture returns 451 with its rule id |
| no PII in audit examples | Avoids retaining prompt PII in logs | Rule id and request id only; redact detector matches |
| classifier dependency down | A timeout is not evidence of safety | 503, and a `classifier-unavailable` audit event |

## 5. The solution, step by step

### From first principles

1. **Order rules from cheap and certain to expensive and statistical.** A model restriction and a token cap are exact checks on request metadata; a classifier needs an embedding call and gives a probability. Putting the exact checks first means a request they deny never pays for, or depends on, the classifier (section 3's first case).
2. **First match wins, so targeted denies come before anything broad.** With ordered evaluation the meaning of a rule depends on its position; a broad allow above a targeted deny would silently disable the deny.
3. **Threshold the probability, not the logit.** The head's threshold is a probability in $[0, 1]$; a raw logit has no fixed scale. The evaluator normalizes the embedding, computes logits, applies a stable softmax, and compares the class probability with `τ`.
4. **Fail closed.** If the embedding service or the head is unavailable, the classifier rule cannot be evaluated, so the request is refused with 503 (a temporary error) rather than allowed: a timeout is not evidence that a prompt is safe.
5. **A deny is explainable and private.** Each deny returns 451 with a stable rule id and a user-safe reason; the audit event records identifiers (request, tenant, model, rule, outcome, time) and never the prompt, the completion, or the embedding.
6. **The document owns the process.** Rules change; the policy names an owner, a revision, a review cadence, an appeal path, and what the classifier cannot do (infer intent, replace human review).

### The rules, annotated

```yaml
version: 1
default_action: allow            # the policy names what is denied; everything else passes
rules:
  - id: restricted-model         # exact metadata check: cheapest, first
    match:
      model: ["experimental-unsafe"]
    action: deny
    reason: "This model is not available through the public gateway"
  - id: free-token-cap           # exact metadata check, before any classifier call
    match:
      tenant: ["free"]
      max_tokens: 1024           # matches when the request asks for MORE than 1024
    action: deny
    reason: "Requested output limit exceeds the free tier"
  - id: unsafe-prompt            # statistical check: last, and fails closed
    match: {}                    # every request that reached this rule
    classifier:
      head: "models/smol-135m/heads/usage.json"   # versioned head, fitted in L6.5
      class: "unsafe"
      threshold: 0.9             # a probability, not a logit
    action: deny
    reason: "The request was blocked by the usage policy"
```

`default_action: allow` is a deliberate choice for a public inference gateway: the rules enumerate what is not permitted. The reasons are short and safe to show a user; they say which policy blocked the request without revealing classifier details. Full file: [`policy.v1.yaml`](../../../../course/solutions/entry/deploy/policy.v1.yaml). The local-services variant, [`policy.local.yaml`](../../../../course/solutions/entry/config/policy.local.yaml), keeps the same rule ids and order but uses a head fitted on `hash16` vectors, because no embedding model runs in PR CI.

### The evaluator, annotated

The YAML only means something through the code that enforces it. The core of `gw.08`'s `Check`:

```go
for _, rule := range p.Document.Rules {             // in file order: first match wins
	if !matches(rule.Match, principal, req, maxTokens) {
		continue
	}
	if rule.Classifier != nil {
		head, ok := p.Heads[rule.Classifier.Head]
		if !ok || p.Embedder == nil {
			return Decision{}, ErrEmbeddingUnavailable  // no head or embedder: an error, never an allow
		}
		emb, err := p.Embedder.Embed(ctx, head.EmbeddingModel, text)
		if err != nil {
			return Decision{}, ErrEmbeddingUnavailable  // step 4: fail closed
		}
		probs, err := Probabilities(head, emb)        // normalize, W e + b, stable softmax
		// ... error handling; resolve the class index (the rule's class, or the head's last class)
		threshold := head.Threshold
		if rule.Classifier.Threshold != nil {
			threshold = *rule.Classifier.Threshold     // the rule may tighten or loosen the head's default
		}
		if probs[idx] < threshold {
			continue                                   // below τ: this rule does not match
		}
	}
	decision := Decision{Allow: rule.Action == "allow", RuleID: rule.ID, Reason: rule.Reason}
	if !decision.Allow {
		p.audit(ctx, AuditEvent{RequestID: requestID, Tenant: principal.Tenant, Model: req.Model,
			RuleID: rule.ID, Reason: rule.Reason, Outcome: "deny", At: p.now()})   // no prompt field exists
	}
	return decision, nil
}
// ... no rule matched: default_action decides
```

```go
func matches(m Match, principal auth.Principal, req *route.InferenceRequest, maxTokens int) bool {
	if len(m.Model) > 0 && !contains(m.Model, req.Model) {
		return false
	}
	if len(m.Tenant) > 0 && !contains(m.Tenant, principal.Tenant) {
		return false
	}
	if m.MaxTokens != nil && maxTokens <= *m.MaxTokens {
		return false                                   // within the cap: the cap rule does not match
	}
	return true                                        // empty match {}: matches everything
}
```

Every condition present must hold, and an absent condition is no restriction, so `match: {}` matches every request. Section 3's free-tier request with `max_tokens = 2048` matches `free-token-cap` (tenant `free`, $2048 > 1024$) and is denied before the classifier is consulted.

```go
// Probabilities applies the exported linear head to a normalized embedding.
func Probabilities(h Head, embedding []float64) ([]float64, error) {
	// ... shape checks: len(embedding) == h.Dim, one W row and one bias per class
	norm := 0.0
	for _, v := range embedding {
		norm = math.Hypot(norm, v)                     // overflow-safe Euclidean norm
	}
	// ... x = embedding / norm (left at zero for a zero vector)
	// ... logits[c] = h.B[c] + sum_j W[c][j] * x[j], tracking the maximum
	for i, z := range logits {
		logits[i] = math.Exp(z - max)                  // subtract the max: no overflow
		sum += logits[i]
	}
	for i := range logits {
		logits[i] /= sum
	}
	return logits, nil
}
```

The head was fitted on normalized embeddings (`L6.5`), so the gateway normalizes again: a head applied to unnormalized vectors would see logits on a different scale. For logits `[0, 2]` the unsafe probability is $e^2/(1 + e^2) \approx 0.881 < 0.9$, so `unsafe-prompt` does not match and the request falls through to `default_action: allow`.

`Middleware` turns decisions into HTTP: an error from `Check` writes 503 `policy_unavailable` and audits a `classifier-unavailable` deny; a deny writes 451 with `rule_id: reason`; an allow passes the request on. Its comment states the privacy rule as a design constraint: the prompt text is deliberately not part of `AuditEvent`; only non-content identifiers are kept. Full evaluator: [`policy.go`](../../../../course/solutions/go/gateway/policy/policy.go).

### The policy document, annotated

The reference [`USAGE_POLICY.md`](../../../../course/solutions/docs/USAGE_POLICY.md) opens with ownership and process, not rules:

> **Owner:** Gateway Safety Review (gateway maintainers and the privacy lead) · **Revision:** `usage-policy-2026-10-09.1` · **Review cadence:** before each model or classifier change, and at least every 90 days · **Appeal:** users can request review through the support channel with the request ID; include no prompt text in the ticket unless the user chooses to provide it.

Then it scopes itself honestly: it "does not certify that a model is safe, infer a user's intent, or replace human review." Its sections cover:

- **Decision order:** each rule in order with what it does, the fail-closed rule for classifier errors, and the change rule: a change to rule order, model, head, or threshold requires a revision bump and a red-team review.
- **User outcome and review:** the response carries the rule id, a reason, and a request id; appeals are reviewed against the policy revision and rule id without copying prompt content into logs or tickets; emergency changes need a named approver and a follow-up within one business day.
- **Privacy-safe audit fields:** request id, policy revision, rule id, decision, timestamp, model id, classifier version; never prompt text, completions, credentials, or raw embeddings; decision metadata kept 30 days, de-identified aggregate counts up to 13 months to detect drift.
- **Limitations and monitoring:** the threshold trades false positives for false negatives and is not a measure of certainty or intent; deny rates are monitored by rule and model; the classifier rule is suspended if its dependency or calibration is unhealthy.

### Why this solution

- **Ordered first-match over a scoring system.** A weighted score across rules is harder to explain to a user and to audit; first-match gives every decision exactly one rule id, which is what the response, the audit, and the appeal all refer to.
- **Fail closed over fail open.** Failing open would make the classifier rule disappear whenever the embedding service is slow, which is precisely when an attacker would want it gone. The cost is availability: a 503 during an embedding outage. The document makes that tradeoff explicit and gives operators a way to suspend the rule deliberately.
- **A linear head in Go over a Python classifier in the path.** The head is a small matrix exported to JSON; evaluating it in the gateway adds microseconds and no new service. A full classifier would be more accurate but adds latency, a dependency, and another failure mode on every request.
- **Tradeoffs accepted.** A linear head on embeddings will misclassify some prompts in both directions; the policy says so, keeps the threshold reviewable, and provides an appeal path rather than claiming accuracy it does not have.

### Explain it back

**1. Why are `restricted-model` and `free-token-cap` placed before `unsafe-prompt`?**

<details><summary>Answer</summary>

They are exact, cheap checks on request metadata. Placed first, a request they deny never triggers an embedding call, so it is decided quickly, does not depend on the embedding service being up, and gets the more specific rule id and reason. With first-match semantics, order also decides which rule is reported, and the most specific explanation should win.

</details>

**2. What happens when the embeddings endpoint times out, and why is that the right behaviour?**

<details><summary>Answer</summary>

`Check` returns `ErrEmbeddingUnavailable`, and `Middleware` writes a 503 `policy_unavailable` and audits a `classifier-unavailable` deny. The request is not allowed: without the embedding the classifier rule cannot be evaluated, and treating "could not check" as "safe" would let anyone bypass the rule by making the dependency slow.

</details>

**3. Logits `[0, 2]` and threshold 0.9: is the request denied? What would change if the threshold were applied to the raw logit?**

<details><summary>Answer</summary>

The unsafe probability is $e^2/(1 + e^2) \approx 0.881 < 0.9$, so the classifier rule does not match and the request is allowed by the default. Comparing the raw logit 2 with 0.9 would deny it, and in general a logit has no fixed scale, so the threshold would mean something different for every head. The threshold is defined on the probability, which is what the head's held-out metrics describe.

</details>

**4. Why does the audit event contain the request id but not the prompt?**

<details><summary>Answer</summary>

The request id lets a reviewer connect an appeal, a response, and the decision without the audit log becoming a store of user prompts, which can contain personal data (`ethics.02`). The policy revision and rule id say why the request was denied; the prompt itself is only seen in an appeal if the user chooses to provide it.

</details>

**5. Why does a change of threshold require a policy revision bump and a red-team review?**

<details><summary>Answer</summary>

The threshold decides which prompts are blocked, so changing it changes the policy as much as adding a rule. The revision is recorded in every audit event, so a bump lets reviewers tell which policy made a past decision; the red-team review checks that every fixture that should be denied still is.

</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Putting a broad allow rule before a targeted deny | targeted denies come first, there is no broad allow rule, and `default_action: allow` applies only when nothing matched; step 2 |
| Treating an embeddings timeout as safe | `Check` returns `ErrEmbeddingUnavailable` and `Middleware` answers 503; step 4 |
| Logging the prompt in a deny event | `AuditEvent` has no prompt field; only identifiers are recorded; step 5 |
| Applying the threshold to a raw logit | `Probabilities` applies the softmax and the rule compares `probs[idx]` with `τ`; step 3 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.08` | Read the enforcement behavior this policy artifact configures. |
| Back | `L6.5` | Inspect how the exported linear head is fitted and represented. |
| Back | `data.05` | Reuse its privacy detector categories when writing safe audit examples. |
| Forward | the agent pass | Exercises the configured policy matrix, red-team fixture set, audit log, and gateway behavior end to end. |
| Forward | `field.06` | Uses this policy in a customer security review. |

## Going further

Production policy systems add policy version rollout, appeals, jurisdiction-specific rules, monitoring for drift, and human review for high-impact decisions. Keep each addition traceable to an owner and an observed failure mode.
