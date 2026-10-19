# Commercial and security questionnaire: Northstar Commerce

**Prepared by:** Alex Rivera, solutions engineer  
**Customer reviewer:** Priya Shah, security  
**Commercial sponsor:** Maya Chen, platform lead  
**Status:** draft; answers require customer review before external use

| Question | Evidence-based answer | Status / owner |
|---|---|---|
| What data is processed? | The proposed workflow may process support prompts and generated responses. The POC uses a redacted, approved sample; no production data is authorized yet. | Confirm exact fields with Priya |
| Where is data stored and for how long? | The POC retention period and storage location are not yet approved. No claim about production residency can be made from the current course configuration. | Priya to approve before data transfer |
| Who can access the system? | The deployment uses scoped API keys and role-based operational access. Customer-specific identities, rotation, and audit retention must be configured and verified before production. | Northstar IAM owner to supply mapping |
| How are dependencies tracked? | The course requires dependency allowlists and image SBOM evidence. The production image inventory and vulnerability review must be attached to the release record. | Alex attaches generated SBOM; security signs exceptions |
| How are secrets handled? | Credentials belong in the approved secret manager and must not appear in source, logs, fixtures, or support tickets. Rotation and revocation owners must be named. | Northstar operations to document process |
| What happens during an incident? | The runbook defines detection, containment, rollback, evidence preservation, and escalation. Notification timing and customer contacts must be agreed in the contract. | Maya and Priya to finalize |
| Which subprocessors are involved? | None are assumed by this local POC. The production hosting, telemetry, and model asset sources require a current inventory. | Procurement/security to validate before contracting |

## Commercial assumptions

No price or savings estimate is approved. Any proposal must state request volume, retained data, support hours, deployment footprint, and service level assumptions. The customer owns review of procurement terms and the service level; the engineering team owns only the technical estimates and their evidence.

## Open items before external delivery

1. Confirm data classes, geography, and retention with Priya.
2. Attach production threat model, per-image SBOM, dependency exceptions, and access review.
3. Confirm escalation contacts and incident notification terms.
4. Have procurement validate hosting and subprocessor statements.

**Next action:** Alex assembles the evidence packet by Oct 21. Priya approves technical security statements; Maya approves the commercial assumptions. Until those approvals, this document is an internal draft, not a security certification.
