# TinyLLM threat model

**Scope:** public inference gateway, Rust serving engine, model and usage stores, operator control plane, and the agent tool boundary. This is a system review artifact; it does not claim that every listed control has been independently penetration-tested.

## Data flow and trust boundaries

```text
Internet client -> [TLS/auth boundary] -> Go gateway -> [service identity boundary] -> Rust engine
                         |                         |                                  |
                         v                         v                                  v
                   tenant/key DB              usage ledger                      model files
                         ^                         ^
                         |                         |
Operator -> [admin auth boundary] -> route control plane      Agent -> [tool allowlist] -> external tools
```

Prompts, tool results, credentials, model weights, and usage records are separate assets. A prompt is untrusted input even after authentication. Tool output is also untrusted input when it returns to the model.

## STRIDE review

| Boundary / asset | Threat | Existing control and evidence to inspect | Residual risk / owner |
|---|---|---|---|
| Client to gateway: identity and tenant scope | **Spoofing:** stolen API key impersonates a tenant. | Store keyed secret hashes; compare in constant time; audit key id but never the presented secret. Verify auth tests and rotation runbook. | A valid stolen key remains usable until revocation propagates. Security on-call owns revocation timing. |
| Client to gateway: request and quota | **Tampering:** caller changes model, route, or usage scope. | Validate request schema, derive tenant from authenticated identity, enforce quotas after auth. Verify cross-tenant and malformed-request tests. | Parser defects can bypass a policy. Gateway owner reviews schema changes. |
| Gateway to engine: service identity | **Repudiation:** a request cannot be tied to the authenticated principal. | Propagate request id and tenant id in structured audit events; retain append-only records with restricted access. Verify logs omit prompt and secret fields. | Compromised gateway can forge identity metadata. Platform security owns service identity. |
| Gateway to engine: prompt and response | **Information disclosure:** one tenant receives another tenant's cached output. | Partition caches by tenant and model; test key construction and cross-tenant isolation. | Cache-key regression remains possible. Serving owner owns cache review. |
| Route control plane: model selection | **Denial of service:** an operator route points all traffic at an unhealthy engine. | Conditional route update with ETag, canary weight, health gate, and rollback. Verify stale ETag and failed canary cases. | Control-plane outage blocks route changes. SRE owns recovery. |
| Model files and release pipeline | **Elevation of privilege:** an untrusted model artifact executes code or gains write access. | Treat weights as data, verify digest and provenance, mount read-only, and separate build credentials from runtime. Verify image and artifact permissions. | Malicious serialized inputs may exploit a parser. Model platform owns format updates. |
| Agent to tool boundary | **Tampering / disclosure:** prompt injection tricks a tool into sending data or changing state. | Explicit tool allowlist, argument schema, per-tool authorization, taint-aware confirmation for writes, and audit result metadata. Verify denied-tool and untrusted-tool-output tests. | Model behavior is probabilistic. Agent owner requires human approval for destructive actions. |

## Review actions

Prioritize cross-tenant isolation, key revocation, and write-capable tools. Each release review links test output for those controls. Record findings with an owner, due date, severity, and retest evidence. Revisit this model when a new external boundary, stored asset, authentication path, or tool is added.
