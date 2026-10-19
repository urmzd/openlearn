# Threat model (STRIDE on your DFD)

## Overview

| | |
|---|---|
| **Module** | `craft.17` · practice · docs · Pass 11 · 1 h reading |
| **Solution** | Model threat model: [`THREAT_MODEL.md`](../../../../course/solutions/docs/THREAT_MODEL.md), a data-flow diagram with trust boundaries and a STRIDE table with controls, evidence, residual risk, and owners |
| **Needs** | [`craft.09`](../05-diagramming-c4/01-c4-diagrams-in-d2.md) (the C4 container view the data-flow diagram is drawn from) |
| **Used by** | no code call site; this is an independent artifact. [`craft.18`](02-sbom-and-supply-chain.md) and [`craft.19`](03-secrets-and-authorization-review.md) take their scope from it. |

## Key Takeaways

- A threat model starts from a data-flow diagram; threats live where data crosses a trust boundary.
- Every threat row names a control, the evidence that the control works, the residual risk, and an owner.
- STRIDE is a prompt for analysis, not a substitute for evidence; the model is revisited when a boundary, store, auth path, or tool is added.

## How to read this chapter

Sections 1 to 3 explain why the system needs a threat model and trace one dangerous flow by hand. Section 4 lists what a complete threat model must contain. Section 5 walks through the model answer: how the diagram is drawn, how each STRIDE row is derived from a boundary, and how the actions are prioritised. Try to fill in a row yourself before reading the model's.

---

## 1. Why now

The system crosses user, service, storage, and tool boundaries. A threat model identifies where an attacker can act and which controls must prevent or detect it.

## 2. Principles

Start from a data-flow diagram. For each boundary, list assets, actors, threats, existing controls, residual risk, and an owner. STRIDE is a prompt for analysis, not a substitute for evidence.

## 3. Worked example

Trace a retrieved web page into the agent tool loop. Consider spoofed source, malicious instructions, disclosure through tool output, and unauthorized state change. Tie mitigations to allowlists, taint-aware gates, audit logs, and tests.

By hand: the page is fetched by a tool, so it crosses the agent-to-tool boundary inbound as **untrusted data**. Its text enters the model's context next to the user's instructions. If it says "send the conversation to this URL", the threat is **tampering** with the agent's plan and **information disclosure** through the next tool call. The model's output cannot be trusted to refuse, because model behavior is probabilistic. So the control has to sit outside the model: the next tool call is checked against an allowlist and an argument schema, writes that follow untrusted input need confirmation, and the result metadata is audited. The residual risk (the model is fooled into a permitted but unwanted call) is owned by the agent owner, who requires human approval for destructive actions.

## 4. The artifact

The artifact is `docs/THREAT_MODEL.md`.

### What a correct threat model must contain

| Element | What it states | Why it matters downstream |
|---|---|---|
| Scope | Which components are in, and what the model does not claim. | Readers know what was not reviewed. |
| Data-flow diagram | External entities, processes, stores, flows, and trust boundaries. | Threats are enumerated from boundaries, not from memory. |
| Assets | What an attacker wants: prompts, credentials, weights, usage records, tool results. | Controls protect named things. |
| A STRIDE row per boundary and asset | Threat, control and evidence, residual risk and owner, covering all six categories. | Every threat has someone accountable and a way to check the control. |
| Review actions | Priorities, finding format, and when to revisit. | The model stays current as the system changes. |

## 5. The solution, step by step

### 5.1 From first principles

1. An attacker can only act through an interface they can reach. So the first step is to draw every flow of data and mark where the **privilege** of the data changes: that is a trust boundary. Inside one boundary, components trust each other; across it, nothing is trusted until checked.
2. Authentication proves who sent a request, not that its content is safe. So prompts stay untrusted after authentication, and tool output is untrusted when it returns to the model.
3. Different assets have different attackers and different damage: a stolen key is spoofing, a leaked cache entry is disclosure. Listing assets separately keeps one control from being assumed to protect everything.
4. STRIDE gives six questions to ask of every boundary and asset: can someone pretend to be someone else (**S**poofing), change data in flight or at rest (**T**ampering), deny having acted (**R**epudiation), read what they should not (**I**nformation disclosure), stop others being served (**D**enial of service), or gain rights they were not given (**E**levation of privilege)?
5. A control that has never been checked is a hope. Each row names the evidence to inspect (a test, a runbook, a log property) so a reviewer can verify it.
6. No control is perfect. Writing the residual risk down, with an owner, turns "we are secure" into "this is what we accept, and who watches it".

### 5.2 The model answer, annotated

**Scope.**

> **Scope:** public inference gateway, Rust serving engine, model and usage stores, operator control plane, and the agent tool boundary. This is a system review artifact; it does not claim that every listed control has been independently penetration-tested.

The second sentence matters: a threat model that implies more assurance than it has is itself a risk.

**The diagram.**

```text
Internet client -> [TLS/auth boundary] -> Go gateway -> [service identity boundary] -> Rust engine
                         |                         |                                  |
                         v                         v                                  v
                   tenant/key DB              usage ledger                      model files
                         ^                         ^
                         |                         |
Operator -> [admin auth boundary] -> route control plane      Agent -> [tool allowlist] -> external tools
```

Four boundaries, each where privilege changes (step 1): the public edge (anonymous internet to authenticated tenant), the gateway-to-engine hop (the engine trusts identity the gateway asserts), the admin edge (operators can change routing for everyone), and the tool edge (an agent's decision reaches systems outside ours). The stores hang off the processes that own them, so a threat to a store is analysed through the flow that reaches it.

> Prompts, tool results, credentials, model weights, and usage records are separate assets. A prompt is untrusted input even after authentication. Tool output is also untrusted input when it returns to the model.

Steps 2 and 3, stated as rules.

**The STRIDE table.** Each row picks a boundary and asset, asks the STRIDE question that does the most damage there, and answers with a control, evidence, residual risk, and owner.

| Boundary / asset | Threat | Existing control and evidence to inspect | Residual risk / owner |
|---|---|---|---|
| Client to gateway: identity and tenant scope | **Spoofing:** stolen API key impersonates a tenant. | Store keyed secret hashes; compare in constant time; audit key id but never the presented secret. Verify auth tests and rotation runbook. | A valid stolen key remains usable until revocation propagates. Security on-call owns revocation timing. |
| Client to gateway: request and quota | **Tampering:** caller changes model, route, or usage scope. | Validate request schema, derive tenant from authenticated identity, enforce quotas after auth. Verify cross-tenant and malformed-request tests. | Parser defects can bypass a policy. Gateway owner reviews schema changes. |
| Gateway to engine: service identity | **Repudiation:** a request cannot be tied to the authenticated principal. | Propagate request id and tenant id in structured audit events; retain append-only records with restricted access. Verify logs omit prompt and secret fields. | Compromised gateway can forge identity metadata. Platform security owns service identity. |
| Gateway to engine: prompt and response | **Information disclosure:** one tenant receives another tenant's cached output. | Partition caches by tenant and model; test key construction and cross-tenant isolation. | Cache-key regression remains possible. Serving owner owns cache review. |
| Route control plane: model selection | **Denial of service:** an operator route points all traffic at an unhealthy engine. | Conditional route update with ETag, canary weight, health gate, and rollback. Verify stale ETag and failed canary cases. | Control-plane outage blocks route changes. SRE owns recovery. |
| Model files and release pipeline | **Elevation of privilege:** an untrusted model artifact executes code or gains write access. | Treat weights as data, verify digest and provenance, mount read-only, and separate build credentials from runtime. Verify image and artifact permissions. | Malicious serialized inputs may exploit a parser. Model platform owns format updates. |
| Agent to tool boundary | **Tampering / disclosure:** prompt injection tricks a tool into sending data or changing state. | Explicit tool allowlist, argument schema, per-tool authorization, taint-aware confirmation for writes, and audit result metadata. Verify denied-tool and untrusted-tool-output tests. | Model behavior is probabilistic. Agent owner requires human approval for destructive actions. |

How each row follows from first principles:

- **Spoofing at the public edge.** The only identity proof is the API key, so stealing it is the attack. Keyed hashes mean a leaked key store does not reveal usable keys; constant-time comparison stops the response time from leaking how much of a guess was right; never logging the presented secret stops logs from becoming a second key store. The gateway's [`auth.go`](../../../../course/solutions/go/gateway/auth/auth.go) implements exactly this: `Digest` is HMAC-SHA256 under a pepper, `Lookup` compares with `hmac.Equal`, and failures log the key id only. The residual risk is honest: a stolen valid key works until revocation propagates through the key cache.
- **Tampering with the request.** A caller controls every byte of the body, so anything security-relevant (the tenant) is derived from the authenticated identity, never from the body, and quotas are checked after authentication so they apply to the real tenant.
- **Repudiation across the internal hop.** The engine never sees the API key, so the only link between an engine action and a tenant is metadata the gateway propagates. Append-only audit records with restricted access keep that link; the residual risk is that a compromised gateway can lie, which is why service identity has its own owner.
- **Disclosure through caching.** A shared cache is the classic cross-tenant leak: if the key omits the tenant, tenant B gets tenant A's answer to the same prompt. The control is in the key construction, and the evidence is a cross-tenant test.
- **Denial of service through the control plane.** The largest blast radius belongs to an operator, not an attacker: one bad route sends all traffic to a dead engine. Conditional updates (ETag) stop two operators overwriting each other; a canary and a health gate limit damage; rollback restores service.
- **Elevation of privilege through a model file.** Weights are input from outside the trust boundary. Treating them as data (safetensors, not pickled code), verifying digest and provenance, and mounting read-only means a malicious file cannot run code or modify anything. Build credentials are separate so a compromised runtime cannot publish artifacts.
- **The agent tool boundary** is section 3's walk-through, compressed into one row.

All six STRIDE letters appear, each at the boundary where it matters most. A fuller model would ask all six questions at every boundary; the table records the dominant one per row to stay reviewable.

**Review actions.**

> Prioritize cross-tenant isolation, key revocation, and write-capable tools. Each release review links test output for those controls. Record findings with an owner, due date, severity, and retest evidence. Revisit this model when a new external boundary, stored asset, authentication path, or tool is added.

Priorities follow impact: cross-tenant leaks and write-capable tools cause harm to others that cannot be undone, and revocation is the recovery for the most likely attack. The revisit triggers are exactly the things that would change the diagram, so the model is updated when it would otherwise become wrong.

Full model answer: [`THREAT_MODEL.md`](../../../../course/solutions/docs/THREAT_MODEL.md).

### 5.3 Why this approach

- **Diagram first** rather than a list of known attacks: a list covers what the author remembers; a diagram forces every boundary to be considered, including new ones.
- **STRIDE per boundary** rather than per component: threats happen where data crosses privilege, and a boundary row names both sides, so the control's location is obvious.
- **Evidence column** rather than "mitigated: yes": each claim can be checked in a release review.
- **Tradeoff accepted:** one dominant threat per row keeps the table short at the cost of completeness; the revisit rule and the review actions compensate by making the model a living document.

### 5.4 Explain it back

**Why is a prompt still untrusted after the gateway has authenticated the caller?**

<details><summary>Answer</summary>

Authentication establishes who sent the request, not that its content is benign. An authenticated tenant can still send prompt injection, malformed input, or requests designed to read another tenant's data. Content is validated on its own terms at every boundary.

</details>

**Where does the cross-tenant cache threat sit, and what single property of the code defeats it?**

<details><summary>Answer</summary>

On the gateway-to-engine flow of prompts and responses, at the response cache. The cache key must include the tenant (and the model); with that, two tenants sending the same prompt never share an entry.

</details>

**Why can the agent-tool control not be "tell the model to ignore instructions in tool output"?**

<details><summary>Answer</summary>

The model's behavior is probabilistic and the attacker controls the text, so an instruction to the model is not a guarantee. Controls must be enforced outside the model: allowlists, argument schemas, per-tool authorization, and confirmation for writes.

</details>

**What should make you reopen this threat model?**

<details><summary>Answer</summary>

Any change to the diagram: a new external boundary, a new stored asset, a new authentication path, or a new tool. Each adds flows across privilege that the existing rows do not cover.

</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The model names a control without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | Every row has a "Verify ..." clause naming the test, runbook, or log property to inspect. |
| 2 | Residual risk is omitted | The table reads as "all threats solved" and nobody watches what remains. | Every row states what the control does not stop. |
| 3 | The owner or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | The scope paragraph bounds the review; every row names an owning team. |
| 4 | Trusting data because its sender is authenticated | Prompt injection and tool-output attacks are not modelled. | The assets paragraph states that prompts and tool results are untrusted input. |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | [`craft.09`](../05-diagramming-c4/01-c4-diagrams-in-d2.md) | The C4 container view the data-flow diagram is derived from. |
| Forward | [`craft.18`](02-sbom-and-supply-chain.md) | The model-files and release-pipeline row, expanded into SBOMs and provenance. |
| Forward | [`craft.19`](03-secrets-and-authorization-review.md) | The spoofing, tampering, and control-plane rows, expanded into an authorization matrix and secret inventory. |

## Going further

| Resource | What to inspect |
|---|---|
| OWASP, [Threat Modeling Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Threat_Modeling_Cheat_Sheet.html) | The four-question frame and how STRIDE fits into it. |
| OWASP, [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) | Prompt injection and insecure output handling, the threats behind the agent row. |
