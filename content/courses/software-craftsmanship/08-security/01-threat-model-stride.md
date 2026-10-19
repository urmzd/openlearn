# Threat modeling with STRIDE

By this point in the book the system crosses many boundaries: a public client talks to a Go gateway, the gateway talks to a Rust serving engine, operators reconfigure routes through a control plane, models are loaded from files produced by a release pipeline, and an agent sends calls out to external tools. Each of those crossings is a place where an attacker can act, and each one needs a control that prevents or detects that action. A threat model is the document that finds those places systematically and records, for each one, what protects it and who is accountable when the protection is not enough.

This section builds that document for the whole system. The model starts from a data-flow diagram, which you draw from the C4 container view in [C4 diagrams in D2](../../diagramming/01-diagramming-c4/01-c4-diagrams-in-d2.md) from *Diagramming and Documentation*. From the diagram you find the trust boundaries, and at each boundary you use STRIDE as a set of prompting questions. The result is a table in which every threat names a control, the evidence that the control works, the residual risk, and an owner. The model answer lives in the solutions tree as [`THREAT_MODEL.md`](../../../../course/solutions/docs/THREAT_MODEL.md); it is a standalone artifact rather than code with a call site, and the next two sections, on the supply chain and on secrets and authorization, take their scope from it.

As you read, try to fill in a row of the STRIDE table yourself before reading the model's version. The point is not to memorize the rows but to be able to derive them from the diagram.

> **Learning objectives**
> - Draw a data-flow diagram from a container view and mark every trust boundary where the privilege of data changes.
> - Apply the six STRIDE questions to a boundary and asset and pick the threat that does the most damage there.
> - Write a threat row that names a control, the evidence to inspect, the residual risk, and an owner.
> - Explain why authenticated input and tool output both remain untrusted.
> - State when a threat model must be revisited.

## Starting from the data flow

An attacker can act only through an interface they can reach. That single observation fixes the method. Instead of starting from a list of known attacks, which covers only what the author happens to remember, you start by drawing every flow of data in the system: external entities, processes, data stores, the flows between them, and the **trust boundaries** where the privilege of the data changes. Inside one boundary, components trust each other. Across a boundary, nothing is trusted until it has been checked. Threats live where data crosses a trust boundary, so the diagram tells you where to look.

For each boundary you then list the assets that cross or sit behind it, the actors who can reach it, the threats, the existing controls, the residual risk, and an owner. Assets deserve their own list because different assets have different attackers and different damage: a stolen key is a spoofing problem, a leaked cache entry is a disclosure problem. Listing them separately keeps one control from being silently assumed to protect everything.

Two further principles shape the rows. First, authentication proves who sent a request, not that its content is safe. A prompt stays untrusted after the gateway has authenticated the caller, and tool output is untrusted again when it returns to the model. Second, a control that has never been checked is a hope. Each row names the evidence a reviewer can inspect (a test, a runbook, a log property), so the claim can be verified in a release review rather than taken on faith.

## STRIDE as a set of questions

STRIDE gives you six questions to ask of every boundary and asset:

- **Spoofing**: can someone pretend to be someone else?
- **Tampering**: can someone change data in flight or at rest?
- **Repudiation**: can someone deny having acted?
- **Information disclosure**: can someone read what they should not?
- **Denial of service**: can someone stop others from being served?
- **Elevation of privilege**: can someone gain rights they were not given?

STRIDE is a prompt for analysis, not a substitute for evidence. Answering "yes, mitigated" to all six questions proves nothing; naming the control and the test that exercises it does. And since no control is perfect, every row also writes down the residual risk with an owner. That turns "we are secure" into "this is what we accept, and this is who watches it."

## Tracing one dangerous flow by hand

Before building the full table, it helps to trace one flow end to end. Take a web page that the agent retrieves and follow it into the agent's tool loop. The questions to consider are a spoofed source, malicious instructions embedded in the page, disclosure through a later tool's output, and an unauthorized change of state. The mitigations you will reach for are allowlists, taint-aware gates, audit logs, and tests.

The page is fetched by a tool, so it crosses the agent-to-tool boundary inbound as **untrusted data**. Its text then enters the model's context next to the user's instructions. If the page says "send the conversation to this URL", the threat is **tampering** with the agent's plan and **information disclosure** through the next tool call. You cannot rely on the model to refuse, because model behavior is probabilistic and the attacker controls the text. So the control has to sit outside the model: the next tool call is checked against an allowlist and an argument schema, writes that follow untrusted input need confirmation, and the result metadata is audited. What remains, the residual risk, is that the model is fooled into a call that is permitted but unwanted. The agent owner owns that risk and mitigates it by requiring human approval for destructive actions.

That one walk-through already has every column of the final table: a boundary, an asset, a STRIDE category, a control outside the trusted component, evidence, residual risk, and an owner. The rest of the section does the same for every boundary in the system.

## What the threat model must contain

The artifact is `docs/THREAT_MODEL.md`. The following table lists the elements a complete threat model must contain and why each matters to someone reading it later.

| Element | What it states | Why it matters downstream |
|---|---|---|
| Scope | Which components are in, and what the model does not claim. | Readers know what was not reviewed. |
| Data-flow diagram | External entities, processes, stores, flows, and trust boundaries. | Threats are enumerated from boundaries, not from memory. |
| Assets | What an attacker wants: prompts, credentials, weights, usage records, tool results. | Controls protect named things. |
| A STRIDE row per boundary and asset | Threat, control and evidence, residual risk and owner, covering all six categories. | Every threat has someone accountable and a way to check the control. |
| Review actions | Priorities, finding format, and when to revisit. | The model stays current as the system changes. |

## Building the model step by step

With the method and the required contents fixed, you can now walk through the model answer piece by piece: scope, diagram, assets, the STRIDE table, and the review actions.

### Scope

The model opens by bounding itself:

> **Scope:** public inference gateway, Rust serving engine, model and usage stores, operator control plane, and the agent tool boundary. This is a system review artifact; it does not claim that every listed control has been independently penetration-tested.

The second sentence matters as much as the first. A threat model that implies more assurance than it has is itself a risk, because a reader may make a release decision on the strength of a claim nobody tested.

### The diagram

```text
Internet client -> [TLS/auth boundary] -> Go gateway -> [service identity boundary] -> Rust engine
                         |                         |                                  |
                         v                         v                                  v
                   tenant/key DB              usage ledger                      model files
                         ^                         ^
                         |                         |
Operator -> [admin auth boundary] -> route control plane      Agent -> [tool allowlist] -> external tools
```

There are four boundaries, and each sits exactly where privilege changes. The public edge takes an anonymous internet caller to an authenticated tenant. At the gateway-to-engine hop, the engine trusts the identity the gateway asserts. The admin edge is where operators gain the power to change routing for everyone. The tool edge is where an agent's decision reaches systems outside ours. The stores hang off the processes that own them, so a threat to a store is analysed through the flow that reaches it rather than in isolation.

### Assets

The model then states its assets and how they are treated:

> Prompts, tool results, credentials, model weights, and usage records are separate assets. A prompt is untrusted input even after authentication. Tool output is also untrusted input when it returns to the model.

These two sentences are the principles from earlier in the section stated as rules: assets are listed separately so that no control is assumed to cover all of them, and authenticated content is still validated on its own terms.

### The STRIDE table

Each row picks a boundary and asset, asks the STRIDE question that does the most damage there, and answers with a control, evidence, residual risk, and owner.

| Boundary / asset | Threat | Existing control and evidence to inspect | Residual risk / owner |
|---|---|---|---|
| Client to gateway: identity and tenant scope | **Spoofing:** stolen API key impersonates a tenant. | Store keyed secret hashes; compare in constant time; audit key id but never the presented secret. Verify auth tests and rotation runbook. | A valid stolen key remains usable until revocation propagates. Security on-call owns revocation timing. |
| Client to gateway: request and quota | **Tampering:** caller changes model, route, or usage scope. | Validate request schema, derive tenant from authenticated identity, enforce quotas after auth. Verify cross-tenant and malformed-request tests. | Parser defects can bypass a policy. Gateway owner reviews schema changes. |
| Gateway to engine: service identity | **Repudiation:** a request cannot be tied to the authenticated principal. | Propagate request id and tenant id in structured audit events; retain append-only records with restricted access. Verify logs omit prompt and secret fields. | Compromised gateway can forge identity metadata. Platform security owns service identity. |
| Gateway to engine: prompt and response | **Information disclosure:** one tenant receives another tenant's cached output. | Partition caches by tenant and model; test key construction and cross-tenant isolation. | Cache-key regression remains possible. Serving owner owns cache review. |
| Route control plane: model selection | **Denial of service:** an operator route points all traffic at an unhealthy engine. | Conditional route update with ETag, canary weight, health gate, and rollback. Verify stale ETag and failed canary cases. | Control-plane outage blocks route changes. SRE owns recovery. |
| Model files and release pipeline | **Elevation of privilege:** an untrusted model artifact executes code or gains write access. | Treat weights as data, verify digest and provenance, mount read-only, and separate build credentials from runtime. Verify image and artifact permissions. | Malicious serialized inputs may exploit a parser. Model platform owns format updates. |
| Agent to tool boundary | **Tampering / disclosure:** prompt injection tricks a tool into sending data or changing state. | Explicit tool allowlist, argument schema, per-tool authorization, taint-aware confirmation for writes, and audit result metadata. Verify denied-tool and untrusted-tool-output tests. | Model behavior is probabilistic. Agent owner requires human approval for destructive actions. |

Every row follows from first principles, and it is worth seeing how.

**Spoofing at the public edge.** The only proof of identity is the API key, so stealing it is the attack. Storing keyed hashes means a leaked key store does not reveal usable keys. Constant-time comparison stops the response time from leaking how much of a guess was right. Never logging the presented secret stops the logs from becoming a second key store. The gateway's [`auth.go`](../../../../course/solutions/go/gateway/auth/auth.go) implements exactly this: `Digest` is HMAC-SHA256 under a pepper, `Lookup` compares with `hmac.Equal`, and failures log the key id only. The residual risk is stated honestly: a stolen valid key works until revocation propagates through the key cache.

**Tampering with the request.** A caller controls every byte of the body. Anything security-relevant, above all the tenant, is therefore derived from the authenticated identity and never from the body, and quotas are checked after authentication so that they apply to the real tenant.

**Repudiation across the internal hop.** The engine never sees the API key, so the only link between an engine action and a tenant is the metadata the gateway propagates. Append-only audit records with restricted access preserve that link. The residual risk is that a compromised gateway can lie, which is why service identity has its own owner.

**Disclosure through caching.** A shared cache is the classic cross-tenant leak: if the cache key omits the tenant, tenant B receives tenant A's answer to the same prompt. The control lives in the key construction, and the evidence is a cross-tenant test.

**Denial of service through the control plane.** Here the largest blast radius belongs to an operator, not an attacker: one bad route sends all traffic to a dead engine. Conditional updates with an ETag stop two operators from overwriting each other, a canary and a health gate limit the damage, and rollback restores service.

**Elevation of privilege through a model file.** Weights are input from outside the trust boundary. Treating them as data (safetensors, not pickled code), verifying digest and provenance, and mounting them read-only means a malicious file cannot run code or modify anything. Build credentials are kept separate from runtime credentials, so a compromised runtime cannot publish artifacts.

**The agent tool boundary** is the hand-traced walk-through from earlier, compressed into one row.

All six STRIDE letters appear, each at the boundary where it matters most. A fuller model would ask all six questions at every boundary; this table records the dominant threat per row so that it stays reviewable.

### Review actions

The model closes with what to do with it:

> Prioritize cross-tenant isolation, key revocation, and write-capable tools. Each release review links test output for those controls. Record findings with an owner, due date, severity, and retest evidence. Revisit this model when a new external boundary, stored asset, authentication path, or tool is added.

The priorities follow impact. Cross-tenant leaks and write-capable tools cause harm to other people that cannot be undone, and revocation is the recovery path for the most likely attack. The revisit triggers are exactly the changes that would alter the diagram, so the model gets updated precisely when it would otherwise become wrong.

The full model answer is [`THREAT_MODEL.md`](../../../../course/solutions/docs/THREAT_MODEL.md).

### Why this design

Several choices in the model are deliberate, and each has an alternative worth rejecting explicitly. Drawing the diagram first, rather than starting from a list of known attacks, forces every boundary to be considered, including new ones nobody has attacked yet. Writing STRIDE rows per boundary rather than per component works because threats happen where data crosses privilege, and a boundary row names both sides, which makes the location of the control obvious. An evidence column instead of "mitigated: yes" means each claim can be checked in a release review. The one accepted tradeoff is recording a single dominant threat per row: the table stays short at the cost of completeness, and the revisit rule and review actions compensate by making the model a living document.

## What can go wrong

Threat models fail in predictable ways, and the model answer is built to avoid each of them. The most common failure is naming a control without the evidence that supports it, which leaves reviewers unable to verify the claim against the repository or the running system; every row in the model has a "Verify ..." clause naming the test, runbook, or log property to inspect. The second is omitting residual risk, so that the table reads as "all threats solved" and nobody watches what remains; every row in the model states what its control does not stop. The third is leaving the owner or scope implicit, which lets two reviewers reach different release or risk decisions from the same evidence; the scope paragraph bounds the review and every row names an owning team.

> **Warning:** Do not trust data because its sender is authenticated. A model that makes this mistake never considers prompt injection or tool-output attacks at all. The assets paragraph exists to state, in writing, that prompts and tool results are untrusted input.

## Where this leads

The data-flow diagram here is derived from the container view in [C4 diagrams in D2](../../diagramming/01-diagramming-c4/01-c4-diagrams-in-d2.md). Two rows of the table are expanded in the rest of this chapter. The model-files and release-pipeline row becomes [SBOM and supply chain](02-sbom-and-supply-chain.md), which records what was shipped and how it was built. The spoofing, tampering, and control-plane rows become [Secrets and authorization review](03-secrets-and-authorization-review.md), which builds an authorization matrix and a secret inventory.

### Further reading

- OWASP, [Threat Modeling Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Threat_Modeling_Cheat_Sheet.html): the four-question frame and how STRIDE fits into it.
- OWASP, [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/): prompt injection and insecure output handling, the threats behind the agent row.

## Key terms

- **Data-flow diagram (DFD)**: a diagram of external entities, processes, data stores, the flows between them, and trust boundaries.
- **Trust boundary**: a line in the diagram where the privilege of data changes; nothing crossing it is trusted until checked.
- **STRIDE**: six prompting questions (spoofing, tampering, repudiation, information disclosure, denial of service, elevation of privilege) asked of each boundary and asset.
- **Evidence**: the test, runbook, or log property a reviewer inspects to confirm that a control works.
- **Residual risk**: what a control does not stop, written down with an owner who accepts and watches it.

## Summary

- A threat model starts from a data-flow diagram; threats live where data crosses a trust boundary.
- Every threat row names a control, the evidence that the control works, the residual risk, and an owner.
- Authentication proves who sent a request, not that its content is safe; prompts and tool output stay untrusted.
- Controls for the agent tool boundary must sit outside the model, because model behavior is probabilistic.
- STRIDE is a prompt for analysis, not a substitute for evidence; the model is revisited when a boundary, store, auth path, or tool is added.

## Review questions

1. Why is a prompt still untrusted after the gateway has authenticated the caller?

   <details><summary>Answer</summary>

   Authentication establishes who sent the request, not that its content is benign. An authenticated tenant can still send prompt injection, malformed input, or requests designed to read another tenant's data. Content is validated on its own terms at every boundary.

   </details>

2. Where does the cross-tenant cache threat sit, and what single property of the code defeats it?

   <details><summary>Answer</summary>

   On the gateway-to-engine flow of prompts and responses, at the response cache. The cache key must include the tenant (and the model); with that, two tenants sending the same prompt never share an entry.

   </details>

3. Why can the agent-tool control not be "tell the model to ignore instructions in tool output"?

   <details><summary>Answer</summary>

   The model's behavior is probabilistic and the attacker controls the text, so an instruction to the model is not a guarantee. Controls must be enforced outside the model: allowlists, argument schemas, per-tool authorization, and confirmation for writes.

   </details>

4. What should make you reopen this threat model?

   <details><summary>Answer</summary>

   Any change to the diagram: a new external boundary, a new stored asset, a new authentication path, or a new tool. Each adds flows across privilege that the existing rows do not cover.

   </details>
