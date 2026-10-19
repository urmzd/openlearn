# Secrets and authorization review

## Overview

| | |
|---|---|
| **Module** | `craft.19` · practice · docs · Pass 11 · 1 h reading |
| **Solution** | Model review: [`security-review.md`](../../../../course/solutions/docs/security-review.md), an authorization matrix, a secret inventory, required release evidence, and a finding policy; checked against the gateway's key handling in [`auth.go`](../../../../course/solutions/go/gateway/auth/auth.go) |
| **Needs** | [`craft.17`](01-threat-model-stride.md) (the threat model's spoofing and control-plane rows) · [`gw.02`](../../ai-platform-engineering/08-authorization-and-access-control/01-api-keys-and-scopes.md) (API keys and scopes) |
| **Used by** | no code call site; this is an independent artifact |

## Key Takeaways

- An authorization review is a matrix of principals, actions, and resources; every allow is derived from the authenticated principal and the resource owner, never from a caller-supplied id.
- Secrets are inventoried with their source, runtime use, logging rule, and rotation path; the presented secret is never logged.
- Release is blocked on named failures (cross-tenant access, exposed credentials, unauthenticated route mutation), and each finding has an owner and retest evidence.

## How to read this chapter

Sections 1 to 3 explain what a secrets and authorization review covers and the checks it starts from. Section 4 lists what a complete review must contain. Section 5 walks through the model review, derives its matrix from first principles, and checks its claims against the gateway's code. Try to fill in the matrix yourself before reading the model's.

---

## 1. Why now

API keys and tenant scopes protect model access and usage data. Review the authorization design as a matrix of principals, actions, and resources, then inspect where secrets can leak.

## 2. Principles

Store only keyed hashes of API secrets, compare credentials in constant time, scope every operation, and audit denies without logging presented secrets or prompt content. Rotate and revoke through a documented path.

## 3. Worked example

Check that an infer-only key cannot call admin routes, tenant A cannot retrieve tenant B usage, and revocation takes effect within the documented cache window. Search source, logs, image layers, and CI artifacts for credential patterns.

By hand, the first check: an infer-only key calls `GET /admin/v1/...`. The gateway's `RequiredScope` maps any `/admin/` path to the `admin` scope; `Authorize` finds the principal lacks it and returns `ErrInsufficient`; the middleware answers 403 `insufficient_scope` and logs the key id and route, not the key. The third check: with `key_cache_ttl_s = 30`, a key revoked at time $t$ may still be accepted by a gateway that cached it at $t - 1$ until $t + 29$. The cache window is that TTL, and the review must state it and measure it.

## 4. The artifact

The artifact is `docs/security-review.md`.

### What a correct review must contain

| Element | What it states | Why it matters downstream |
|---|---|---|
| Scope, date, reviewers | What was reviewed, when, by whom. | Findings are tied to a version of the system. |
| Authorization matrix | Every principal against every action, with the condition on each allow. | Gaps and over-grants are visible at a glance. |
| Secret inventory | Each secret's source, runtime use, logging rule, rotation and recovery. | Leaks and stale credentials have a known response. |
| Required evidence | Tests and scans to attach before release, including the measured revocation window. | Claims are checked, not asserted. |
| Finding policy | Owner, severity, due date, retest; what blocks release. | Two reviewers make the same release decision. |

## 5. The solution, step by step

### 5.1 From first principles

1. Authorization is a function of (principal, action, resource). A matrix with principals as rows and actions as columns makes every cell an explicit decision, so a missing rule shows up as a blank cell instead of a default nobody chose.
2. The principal must come from authentication, and the resource's owner from the system's records. Anything the caller sends (a tenant id in a URL or body) only selects which resource is meant; if it also decided access, any caller could claim any tenant.
3. A denial must not leak information. If "not yours" and "does not exist" answered differently, a caller could enumerate other tenants' identifiers. So both get the same external error.
4. A stored secret is a liability: whoever reads the store can use it. So the gateway stores a keyed hash (HMAC under a pepper kept elsewhere), which cannot be reversed and cannot be guessed offline without the pepper.
5. Comparison time must not depend on how much of a guess matches, or the response time leaks the secret byte by byte. So comparisons are constant time.
6. Logs, images, and diagnostic bundles are copied widely. A secret that reaches any of them is effectively public. So logs carry identifiers (key id, request id, tenant id) and never the presented secret, headers, prompts, or tool arguments.
7. Caching verified keys trades revocation latency for speed. The trade is acceptable only if the window is stated and measured.

### 5.2 The model review, annotated

**The matrix.**

| Principal | Inference | Read own usage | Read another tenant | Update routes | Rotate/revoke keys |
|---|---:|---:|---:|---:|---:|
| `tenant-key:infer` | allow, quota checked | deny | deny | deny | deny |
| `tenant-key:admin` | allow, quota checked | allow, same tenant only | deny | deny | allow, same tenant only |
| `operator:release` | deny | deny | deny | allow, audited and conditional on current ETag | deny |
| anonymous | deny before dispatch | deny | deny | deny | deny |

Reading it as step 1 intends:

- The "Read another tenant" column is all deny. That is the cross-tenant rule as a column, and it is what the release blocks on.
- Every allow has a condition: quota for inference, "same tenant only" for usage and keys, ETag and audit for routes. An unconditional allow would be a smell.
- `operator:release` can change routes but cannot infer or read usage. Separating duties means a compromised release credential cannot read tenant data.
- Anonymous is denied "before dispatch": no handler runs, so no unauthenticated code path exists to have a bug in.

> Every allow decision is derived from the authenticated principal and resource owner. A caller supplied tenant id is a selector, never proof of authorization. A mismatch returns the same external error shape as a missing resource to avoid confirming another tenant's identifiers.

Steps 2 and 3, as the review's rules.

**Checking the matrix against the code.** The gateway's [`auth.go`](../../../../course/solutions/go/gateway/auth/auth.go) is where the first column is enforced:

```go
// RequiredScope is the scope a path needs: /v1/embeddings needs embed,
// /admin/v1/... admin, every other path (chat, completions, models) infer.
func RequiredScope(route string) string {
	switch {
	case strings.HasPrefix(route, "/admin/"):
		return ScopeAdmin
	case route == "/v1/embeddings":
		return ScopeEmbed
	default:
		return ScopeInfer          // the default is the least privileged scope
	}
}

func Authorize(p Principal, route, model string) error {
	if !p.HasScope(RequiredScope(route)) {
		return ErrInsufficient     // 403 insufficient_scope
	}
	if model != "" && len(p.Models) > 0 {   // an optional per-key model allowlist
		for _, m := range p.Models {
			if m == model {
				return nil
			}
		}
		return ErrInsufficient
	}
	return nil
}
```

The scope comes from the path, and `p` comes from `Lookup`, so neither is caller-asserted. Later stages read the principal from the request context with `PrincipalFrom` (the response cache scopes entries by its tenant, the ledger records its tenant and key id), and the admin handler in [`ledger/admin.go`](../../../../course/solutions/go/gateway/ledger/admin.go) fails closed: no principal is a 401 and no admin scope a 403, even if it were mounted outside the authentication stage by mistake. Whether each tenant-scoped cell ("same tenant only") holds for a given route is what the review's cross-tenant test establishes: the matrix states the target, and the test is the evidence.

**Secret inventory.**

| Secret | Source | Runtime use | Logging rule | Rotation / recovery |
|---|---|---|---|---|
| Tenant API key | one-time admin issuance | keyed hash lookup and constant-time comparison | never log input, hash, or header | issue replacement, verify it, revoke old key, confirm cache expiry |
| Service identity | workload identity | gateway to engine | log principal id only | rotate through platform identity controller |
| Route signing credential | secret manager | operator control plane | never place in command output or audit payload | revoke credential, restore last known route using current ETag |

The API key row maps onto the code:

```go
// Digest is the stored form of a secret: hex HMAC-SHA256 keyed by the
// pepper. Without the pepper, a leaked keys file does not let anyone test
// guesses offline.
func Digest(pepper []byte, secret string) string {
	m := hmac.New(sha256.New, pepper)
	m.Write([]byte(secret))
	return hex.EncodeToString(m.Sum(nil))
}

func (s *Store) Lookup(ctx context.Context, presented string) (Principal, error) {
	id, secret, ok := ParseKey(presented)   // "tl_<id>_<secret>": id for O(1) lookup
	if !ok {
		return Principal{}, ErrInvalidKey
	}
	digest := Digest(s.pepper, secret)
	now := s.clock.Now()
	// ...
	if c, ok := s.cache[id]; ok && now.Before(c.until) {
		if hmac.Equal([]byte(c.digest), []byte(digest)) {   // constant-time comparison
			return c.p, nil                                 // served from cache until the TTL
		}
		return Principal{}, ErrInvalidKey
	}
	// ... reload the keys file
	r, ok := s.recs[id]
	if !ok || !hmac.Equal([]byte(r.HMAC), []byte(digest)) {
		return Principal{}, ErrInvalidKey
	}
	if r.Revoked || (r.ExpiresAt != nil && !now.Before(*r.ExpiresAt)) {
		return Principal{}, ErrInvalidKey   // revoked and expired look the same as wrong
	}
	// ... cache the principal for s.ttl
}
```

Step 4 is `Digest`; step 5 is `hmac.Equal`; step 7 is the cache: a cached entry is returned without consulting the revoked flag until `c.until`, which is why "confirm cache expiry" is part of rotation and why the review measures the window. `NewStore` refuses an empty pepper, so the gateway never falls back to bare hashes. On failure the middleware logs `"key_id", id` only (step 6).

> No secret is baked into an image, committed to config, or copied into a diagnostic bundle. Logs retain request id, tenant id, key id, decision, and policy reason; they omit authorization headers, prompts, and tool arguments. Incident responders can revoke by key id without recovering the presented secret.

The last sentence is why keys carry a public id: revocation needs an identifier that is safe to log and to type into a runbook.

**Required evidence before release.**

> - Tests show an infer-only key cannot call administrative routes.
> - Cross-tenant usage lookup is denied even when the caller supplies another tenant id.
> - Revocation takes effect within the declared cache window; the measured window is attached to the release record.
> - Secret scanning covers source, container layers, CI artifacts, and logs. Findings include the scan revision and disposition.
> - Route updates require the current ETag, are audited, and have a tested rollback.

One check per high-risk cell of the matrix and per leak path of the inventory. Container layers are listed separately from source because a secret deleted in a later layer is still present in the earlier one.

**Finding policy.**

> Release is blocked for cross-tenant access, exposed credentials, or unauthenticated route mutation. A false positive is closed only with evidence and reviewer approval.

The three blockers are the three failures that harm others and cannot be undone after the fact. Everything else is triaged by severity and due date.

Full model answer: [`security-review.md`](../../../../course/solutions/docs/security-review.md).

### 5.3 Why this approach

- **A matrix** rather than a prose list of rules: blanks and over-grants are visible, and each cell maps to a test.
- **Keyed hash (HMAC with a pepper)** rather than a plain hash: API secrets here are 32 random characters, so brute force is already hard, but the pepper means a leaked keys file alone is useless; a slow password hash (bcrypt) would add latency on every request for no gain on high-entropy secrets.
- **A short verified-key cache** rather than a store read per request: the keys file is reread only on a cache miss. The cost is a bounded revocation delay, stated and measured.
- **Tradeoff accepted:** returning the same error for "not found" and "not yours" makes some client debugging harder, in exchange for not confirming other tenants' identifiers.

### 5.4 Explain it back

**Why is a tenant id in the request URL not enough to authorize reading that tenant's usage?**

<details><summary>Answer</summary>

The caller controls the URL. If the id decided access, any authenticated key could read any tenant by changing it. The id selects the resource; access is decided by comparing the authenticated principal's tenant with the resource's owner.

</details>

**A key is revoked, but requests with it succeed for another 20 seconds. Is that a bug?**

<details><summary>Answer</summary>

Not if the declared cache window (the key cache TTL) is at least 20 seconds: `Lookup` serves a cached principal until the entry expires without rereading the revoked flag. It is a bug if it exceeds the declared window, which is why the release record attaches the measured window.

</details>

**Why does `Lookup` compare digests with `hmac.Equal` instead of `==`?**

<details><summary>Answer</summary>

String equality returns as soon as a byte differs, so its running time reveals how long the matching prefix is. `hmac.Equal` takes the same time whatever the contents, so timing tells an attacker nothing.

</details>

**Why does an incident responder never need the presented secret?**

<details><summary>Answer</summary>

Every key has a public id, logged on each decision, and revocation works by id. The responder can find which key misbehaved and revoke it from logs alone, so no process or person ever needs to see or store the secret.

</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The review names a control without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | "Required evidence before release" lists the test or scan for each high-risk cell, and the code locations are named. |
| 2 | The failure or recovery case is omitted | The happy path passes while an operator has no safe response under failure. | The inventory's last column gives rotation and recovery for every secret. |
| 3 | The owner, threshold, or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | Scope, date, and reviewers head the document; the finding policy names the three release blockers. |
| 4 | Authorizing from a caller-supplied id | Cross-tenant reads. | The matrix rule: a tenant id is a selector, never proof; the cross-tenant test is required evidence. |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | [`craft.17`](01-threat-model-stride.md) | The spoofing, tampering, and control-plane threats this review checks in detail. |
| Back | [`gw.02`](../../ai-platform-engineering/08-authorization-and-access-control/01-api-keys-and-scopes.md) | The key format, scopes, and store whose behaviour the review verifies. |

## Going further

| Resource | What to inspect |
|---|---|
| [OWASP ASVS](https://owasp.org/www-project-application-security-verification-standard/) | Verification requirements for authentication, session, and access control, usable as a checklist against the matrix. |
| [`auth.go`](../../../../course/solutions/go/gateway/auth/auth.go) | `NewStore`, `Lookup`, `Authorize`, and `Middleware`: every claim in the review's API-key row, in code. |
