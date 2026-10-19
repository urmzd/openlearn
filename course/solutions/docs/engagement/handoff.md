# Operational handoff: Northstar Commerce

**Customer service owner:** Northstar platform operations  
**Escalation owner:** Maya Chen, platform lead  
**Engineering contact:** Alex Rivera, solutions engineer  
**Handoff state:** rehearsal required before production cutover

## Service boundary and normal operation

The serving path consists of the gateway, engine, tokenizer/model assets, and durable workflow services used by the support-assistant integration. Northstar operations owns cluster health, access requests, paging, and the customer-facing fallback. The engineering team owns defect triage for the course implementation during the agreed support window. These boundaries must be reflected in the final service inventory and customer contract.

During normal operation, responders check gateway request/error and latency panels, engine queue and capacity panels, durable workflow backlog, and the current deployment revision. Use the linked runbooks for engine crash loops, TTFT budget burn, durable redeliveries, and dead letters. Do not replay customer prompts into an unapproved environment.

## Escalation and recovery

1. The on-call responder records the alert, affected cohort, start time, and current revision.
2. If customer-visible responses are incorrect, unsafe, or unavailable, route traffic to the agreed fallback and declare an incident.
3. The incident commander owns customer updates. Maya coordinates product and security decisions; Alex supports diagnosis. Priya leads any suspected data exposure response.
4. Preserve approved logs and trace identifiers, not raw prompt contents unless the retention policy explicitly permits them.
5. Resume only after the rollback or repair is verified, the customer owner accepts the recovery evidence, and the incident commander records the decision.

## Readiness rehearsal

Northstar operations must demonstrate that it can find the dashboards, identify the serving revision, execute the fallback, page the right owner, and verify recovery. Record the rehearsal date, participants, elapsed time, gaps, and owners. The handoff is incomplete while any critical step depends on undocumented knowledge or a private credential held by the project team.

**Next action:** Maya schedules a 45-minute rehearsal with operations and Priya before cutover. Alex prepares a clean runbook bundle and removes any non-approved contact details. **Acceptance owner:** Maya signs operational readiness after gaps are closed; Northstar operations accepts ongoing ownership.
