# Usage policy enforcement

## Overview

| | |
|---|---|
| **Module** | `gw.08` · build · Go · Pass 10 · 2 to 3 h |
| **Solution** | [`policy.go`](../../../../course/solutions/go/gateway/policy/policy.go): `Load`, `Evaluator.Check`, `matches`, `Probabilities`, `Redact`, `HTTPEmbedder`, and `Middleware` (the chain's policy stage); its place in the chain is set by `Handler` in [`chain.go`](../../../../course/solutions/go/gateway/server/chain.go); the gateway entry point loads it in [`policy.go`](../../../../course/solutions/entry/go/cmd/gateway/policy.go) |
| **Contract** | [`policy.v1.schema.json`](../../../../course/contracts/formats/policy.v1.schema.json), [`linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json) |
| **Needs** | [`gw.01`](01-server-skeleton.md) middleware chain, [`gw.02`](../08-authorization-and-access-control/01-api-keys-and-scopes.md) principal, [`gw.05`](05-routing.md) `route.InferenceRequest`, [`ethics.05`](../../responsible-ai/05-usage-policy/01-usage-policy-at-the-gateway.md) policy, [`data.05`](../../data-engineering/05-corpus-pipeline/05-pii-scrub.md) detector list |
| **Used by** | `gw.07`'s `gateway.Stages` puts it in the chain's `Policy` slot and mounts its admin handler; the agent track's end-to-end run sends its requests through it |
| **Optional depth** | OPA and Cedar policy evaluation |

## Key takeaways

- Evaluate policy rules in file order and use the first match.
- Model, tenant, and token limits are cheap checks; classifier rules use engine embeddings.
- Stable rule ids travel through audit events and the 451 response.
- Classifier dependencies fail closed with 503.
- Redact request text with the Go PII detector port before logging.

## How to read this chapter

Sections 1 to 3 state the rules and work one classifier decision by hand. Section 4 fixes the interface. Section 5 walks through the solution in `policy.go` and explains why each choice is the right one; try to predict each step before reading it.

## 1. Why now

The agent now sends user requests through the gateway, so a usage restriction must be enforced on every inference path. Middleware placement after authentication gives the policy a tenant and model identity while keeping it ahead of rate limiting, cache, routing, and proxying.

## 2. Principles

`Policy.Check(ctx, principal, request)` returns an allow or deny decision. A rule matches only when all stated `match` fields match. `max_tokens: N` matches a request whose requested output cap exceeds `N`. A classifier rule calls the configured engine's `/v1/embeddings`, normalizes the returned vector, evaluates the exported head, and compares the selected class probability with its threshold. A matching deny is audit-logged and returned as 451 `usage_policy`; dependency failure is 503.

## 3. Worked example

For an `unsafe` class logit of 2 and a `safe` logit of 0, `p(unsafe) = exp(2)/(exp(0)+exp(2)) ≈ 0.881`. The decision is allow at threshold 0.9 and deny at threshold 0.85. A zero embedding produces the softmax of the bias vector, not NaN. The large-vector fixture ensures normalization and dot products remain finite.

## 4. Interface

```go
package policy // import "tinyllm/gateway/policy"

type Rule struct { ID string; Match Match; Classifier *ClassifierRule; Action, Reason string }
type Match struct { Model, Tenant []string; MaxTokens *int }          // YAML: a scalar or a list
type ClassifierRule struct { Head, Class string; Threshold *float64 }
type Document struct { Version int; DefaultAction string; Rules []Rule }
type Head struct { EmbeddingModel string; Dim int; Classes []string; W [][]float64; B []float64; Threshold float64 }
type Decision struct { Allow bool; RuleID, Reason string }
type AuditEvent struct { RequestID, Tenant, Model, RuleID, Outcome, Reason string; At time.Time }
type Audit func(context.Context, AuditEvent) error
type Embedder interface { Embed(ctx context.Context, model, text string) ([]float64, error) }
type HTTPEmbedder struct { BaseURL string; Client *http.Client } // POST {BaseURL}/v1/embeddings
var ErrEmbeddingUnavailable error

func Load(policyPath, artifactRoot string, embedder Embedder, audit Audit) (*Evaluator, error)
func (p *Evaluator) Check(ctx context.Context, principal auth.Principal, req *route.InferenceRequest,
	maxTokens int, text, requestID string) (Decision, error)
func Probabilities(h Head, embedding []float64) ([]float64, error)
func Redact(text string) string
func Middleware(e *Evaluator, embeddingsURL string) server.Middleware // for server.Deps.Policy
```

The middleware is consumed by `server.Deps.Policy`. It uses the exchange's parsed request and the authenticated principal, and it does not log the request body.

### What a correct solution must handle

| Case | Why it exists | Expected result |
|---|---|---|
| ordered rules over model, tenant, and `max_tokens` | protects rule semantics | the first matching rule decides, the same way every time |
| loading the YAML policy | checks the versioned operator policy artifact | a valid policy loads; malformed YAML, unknown fields, and head paths outside the artifacts root are refused |
| the linear head against the Python export | protects the Python/Go boundary | probabilities differ by at most `1e-6` |
| zero and very large embeddings | keeps normalization finite at boundary values | finite scores for both |
| the embeddings service failing | avoids fail-open behavior | 503 and an audit event |
| every deny | makes both rule and default denials durable | one audit event per deny with the matching rule id |
| PII in request text | protects audit privacy | emails, phone numbers, Luhn-valid card numbers, IPs, and keys are redacted |

## 5. The solution, step by step

### 5.1 From first principles

1. **An operator must be able to predict a decision by reading the file.** That rules out "most specific wins" or scoring: the only order a reader can follow is file order, and the first rule whose conditions all hold decides. A rule with no conditions is a catch-all, so it belongs last.
2. **Cheap checks go before expensive ones, within a rule.** Model, tenant, and `max_tokens` are string and integer comparisons; a classifier needs an embeddings call. So `matches` runs first and the classifier only for rules that already matched.
3. **`max_tokens: N` is a limit, so it matches requests that exceed it.** "Deny requests asking for more than 4096 tokens" is `maxTokens > N`; the code expresses it as "does not match when `maxTokens <= N`".
4. **A classifier is a linear head on a unit-length embedding.** The Python side trained it on normalized embeddings, so Go must normalize the same way, compute $z = Wx + b$, and apply softmax: logits are not probabilities (with logits 2 and 0, a threshold of 0.9 on the raw logit would fire; $p = 0.881$ does not).
5. **Numerical edges must stay finite.** A zero vector has no direction (divide by zero), a huge one overflows a naive sum of squares, and large logits overflow `exp`. `math.Hypot` accumulates the norm without overflow, a zero norm leaves $x = 0$ (so $p$ is the softmax of the bias), and softmax subtracts the max logit.
6. **If the policy cannot be evaluated, the answer is no.** An embeddings outage that allowed traffic would turn every classifier rule off exactly when the system is degraded. So any evaluation error is a 503 and is audited.
7. **The audit trail must name why without storing what.** Each deny records the stable rule id, tenant, model, and request id, but no prompt text; `Redact` exists for any log line that must mention request text.

### 5.2 The code, annotated

`Load` reads the policy strictly and resolves classifier heads inside the artifacts root only:

```go
dec := yaml.NewDecoder(bytes.NewReader(b))
dec.KnownFields(true) // a misspelled field is an error, not a silently ignored condition
if err = dec.Decode(&doc); err != nil {
	return nil, err
}
if doc.Version != 1 || len(doc.Rules) == 0 {
	return nil, errors.New("invalid or empty policy document")
}
for _, rule := range doc.Rules {
	// ...
	rel := filepath.Clean(rule.Classifier.Head)
	if filepath.IsAbs(rel) || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return nil, fmt.Errorf("head path %q escapes artifacts", rel) // policy input cannot read arbitrary files
	}
	// ... read and decode the head; refuse it unless dim > 0, >= 2 classes, and W, b match the classes
}
```

`Match.UnmarshalYAML` accepts `model: smol` and `model: [smol, tiny]` alike and refuses unknown match keys, so a typo cannot widen a rule.

`matches` is step 2 and 3. Every stated condition must hold; an absent one is ignored:

```go
func matches(m Match, principal auth.Principal, req *route.InferenceRequest, maxTokens int) bool {
	if len(m.Model) > 0 && !contains(m.Model, req.Model) {
		return false
	}
	if len(m.Tenant) > 0 && !contains(m.Tenant, principal.Tenant) {
		return false
	}
	if m.MaxTokens != nil && maxTokens <= *m.MaxTokens { // matches only requests that exceed the cap
		return false
	}
	return true
}
```

`Check` walks the rules in order (step 1). A classifier rule that matched on its cheap fields then has to pass its threshold, otherwise evaluation continues with the next rule:

```go
for _, rule := range p.Document.Rules {
	if !matches(rule.Match, principal, req, maxTokens) {
		continue
	}
	if rule.Classifier != nil {
		head, ok := p.Heads[rule.Classifier.Head]
		if !ok || p.Embedder == nil {
			return Decision{}, ErrEmbeddingUnavailable // cannot evaluate: fail closed (step 6)
		}
		emb, err := p.Embedder.Embed(ctx, head.EmbeddingModel, text)
		if err != nil {
			return Decision{}, ErrEmbeddingUnavailable
		}
		probs, err := Probabilities(head, emb)
		// ... error check; class defaults to the head's last class; unknown class is an error
		threshold := head.Threshold
		if rule.Classifier.Threshold != nil {
			threshold = *rule.Classifier.Threshold // the rule may tighten or loosen the head's default
		}
		if probs[idx] < threshold {
			continue // the classifier did not fire: this rule does not apply
		}
	}
	decision := Decision{Allow: rule.Action == "allow", RuleID: rule.ID, Reason: rule.Reason}
	if !decision.Allow {
		p.audit(ctx, AuditEvent{RequestID: requestID, Tenant: principal.Tenant, Model: req.Model,
			RuleID: rule.ID, Reason: rule.Reason, Outcome: "deny", At: p.now()}) // no request text
	}
	return decision, nil
}
if p.Document.DefaultAction == "deny" {
	// ... Decision{RuleID: "default-deny"}, audited the same way
}
return Decision{Allow: true}, nil
```

`Probabilities` is steps 4 and 5:

```go
norm := 0.0
for _, v := range embedding {
	norm = math.Hypot(norm, v) // sqrt(norm^2 + v^2) without overflow or underflow
}
x := make([]float64, len(embedding))
if norm > 0 { // a zero vector stays zero: p becomes softmax(b), never NaN
	for i, v := range embedding {
		x[i] = v / norm
	}
}
// logits[c] = b[c] + W[c] . x, tracking the max
sum := 0.0
for i, z := range logits {
	logits[i] = math.Exp(z - max) // stable softmax: the largest term is exp(0) = 1
	sum += logits[i]
}
for i := range logits {
	logits[i] /= sum
}
```

With the worked example's logits (`safe` 0, `unsafe` 2), the max is 2, the terms are $e^{-2} \approx 0.135$ and $1$, and $p(\text{unsafe}) = 1/1.135 \approx 0.881$.

`Middleware` connects it to the chain. It needs a principal (it sits after authn), concatenates the completions `prompt` and every chat message's string `content` into the text to classify, and maps the outcome to HTTP:

```go
decision, err := e.Check(r.Context(), principal, &route.InferenceRequest{Model: parsed.Model}, parsed.MaxTokens, text, x.RequestID)
if err != nil {
	e.audit(r.Context(), AuditEvent{ /* ... */ RuleID: "classifier-unavailable", Outcome: "deny" /* ... */ })
	server.WriteError(w, http.StatusServiceUnavailable, "policy_unavailable", "policy_unavailable", "", "usage policy dependency unavailable")
	return
}
if !decision.Allow {
	server.WriteError(w, http.StatusUnavailableForLegalReasons, "policy_error", "usage_policy", "", decision.RuleID+": "+decision.Reason)
	return // 451 names the stable rule id, so the client and the audit log agree
}
next.ServeHTTP(w, r)
```

In `chain.go`, `Handler` wraps the policy around the rate limiter and inside authn, so a denied request never spends rate-limit budget, never reaches the cache, and always has a tenant. `Redact` applies the email, phone, card (only when the digits pass the Luhn check), IP (only when `net.ParseIP` accepts it), and key patterns, in that order. Full solution: [`policy.go`](../../../../course/solutions/go/gateway/policy/policy.go).

### 5.3 Why this solution

- **Cost.** Rules without classifiers cost a few comparisons. A classifier rule costs one embeddings call (milliseconds, bounded by the embedder's 5 s timeout) and $O(CD)$ arithmetic for $C$ classes and dimension $D$. Putting cheap matches first means most requests never embed.
- **First match rather than a policy language** (OPA/Rego, Cedar): those express richer conditions and conflicts, but an ordered list is something an operator can review line by line, and its audit records name exactly one rule.
- **A linear head exported from Python** rather than calling a model server for classification: the gateway already has an embeddings endpoint, the head is a small JSON file, and parity with the training code can be checked to `1e-6`.
- **Fail closed** trades availability for safety: an embeddings outage makes classifier-guarded traffic 503. That is the intended choice for a usage restriction; a team that prefers availability would have to say so in the policy, not get it by accident.
- **Tradeoff accepted:** audit failures are ignored (`_ = p.Audit(...)`): a broken audit sink does not block traffic. The deny itself still happens.

### 5.4 Explain it back

1. Why does `matches` return false when `maxTokens <= N`, not when `maxTokens >= N`?
   <details><summary>Answer</summary>`max_tokens: N` describes requests that exceed a cap. The rule should apply (match) when the request asks for more than `N`, so it must not match when the request is at or below `N`. The opposite direction would deny every small request and allow every large one.</details>
2. With logits 0 and 2 and a rule threshold of 0.85, which rule decides, and what changes at threshold 0.9?
   <details><summary>Answer</summary>$p(\text{unsafe}) \approx 0.881 \ge 0.85$, so the classifier rule fires and its action decides (a deny returns 451 with its id). At 0.9, $0.881 < 0.9$, so `Check` continues to the next rule and, if none matches, to the default action.</details>
3. What does a zero embedding produce, and why is that acceptable?
   <details><summary>Answer</summary>`x` stays all zeros, so every logit is its bias and the probabilities are `softmax(b)`. It is finite and deterministic, and the bias encodes the head's prior; dividing by a zero norm would give NaN, and every comparison with NaN is false, which would silently let the rule pass.</details>
4. Why is an embeddings error a 503 instead of skipping the classifier rule?
   <details><summary>Answer</summary>Skipping it would allow exactly the requests the rule exists to stop, whenever the embeddings service is down or slow. A 503 tells the client the policy could not be evaluated, and the audit event records it.</details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Matching `max_tokens` in the opposite direction | `matches` rejects `maxTokens <= N`, so only requests above the cap match (step 3) |
| Reordering rules | `Check` iterates `Document.Rules` in file order and returns at the first decision |
| Using raw logits as probabilities | `Probabilities` normalizes, applies the head, and takes a stable softmax before the threshold |
| Allowing an embeddings error | `Check` returns `ErrEmbeddingUnavailable`; `Middleware` answers 503 and audits it |
| Returning a deny without recording its rule id | every deny, including `default-deny`, goes through `p.audit` with `RuleID` |
| Auditing raw prompt text | `AuditEvent` has no text field; `Redact` covers any log line that must mention request text |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.01` | Places the policy middleware in the authenticated request chain. |
| Back | `gw.02` | Supplies the principal used for tenant and model matching. |
| Back | `gw.05` | `route.InferenceRequest` is the parsed request the policy matches on. |
| Forward | `gw.07` | `gateway.Deps` places the stage in the chain; `gateway.Admin` mounts `/admin/v1/policy`. |
| Forward | agent track | The agent's requests pass through the policy on the public gateway interface. |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| ordered rules | signed policy revisions and rollout controls | Auditable change history and staged activation | OPA and Cedar policy evaluation |
| 451 decision | measured false-positive review | Operator feedback and correction loops | Envoy AI Gateway policy examples |
