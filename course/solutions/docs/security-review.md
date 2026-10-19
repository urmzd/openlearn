# Secrets and authorization review

**Review date:** 2026-10-09. **Scope:** gateway key authentication, tenant-scoped inference and usage, administrative route updates, and runtime secret handling. **Reviewers:** gateway owner and security reviewer. This reference demonstrates the evidence format; replace the example status with CI and deployment evidence for the release being reviewed.

## Authorization matrix

| Principal | Inference | Read own usage | Read another tenant | Update routes | Rotate/revoke keys |
|---|---:|---:|---:|---:|---:|
| `tenant-key:infer` | allow, quota checked | deny | deny | deny | deny |
| `tenant-key:admin` | allow, quota checked | allow, same tenant only | deny | deny | allow, same tenant only |
| `operator:release` | deny | deny | deny | allow, audited and conditional on current ETag | deny |
| anonymous | deny before dispatch | deny | deny | deny | deny |

Every allow decision is derived from the authenticated principal and resource owner. A caller supplied tenant id is a selector, never proof of authorization. A mismatch returns the same external error shape as a missing resource to avoid confirming another tenant's identifiers.

## Secret inventory and handling

| Secret | Source | Runtime use | Logging rule | Rotation / recovery |
|---|---|---|---|---|
| Tenant API key | one-time admin issuance | keyed hash lookup and constant-time comparison | never log input, hash, or header | issue replacement, verify it, revoke old key, confirm cache expiry |
| Service identity | workload identity | gateway to engine | log principal id only | rotate through platform identity controller |
| Route signing credential | secret manager | operator control plane | never place in command output or audit payload | revoke credential, restore last known route using current ETag |

No secret is baked into an image, committed to config, or copied into a diagnostic bundle. Logs retain request id, tenant id, key id, decision, and policy reason; they omit authorization headers, prompts, and tool arguments. Incident responders can revoke by key id without recovering the presented secret.

## Required evidence before release

- Tests show an infer-only key cannot call administrative routes.
- Cross-tenant usage lookup is denied even when the caller supplies another tenant id.
- Revocation takes effect within the declared cache window; the measured window is attached to the release record.
- Secret scanning covers source, container layers, CI artifacts, and logs. Findings include the scan revision and disposition.
- Route updates require the current ETag, are audited, and have a tested rollback.

**Finding policy:** each failed control has an owner, severity, due date, and retest link. Release is blocked for cross-tenant access, exposed credentials, or unauthenticated route mutation. A false positive is closed only with evidence and reviewer approval.
