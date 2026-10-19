# Production migration plan: Northstar Commerce

**Status:** draft pending architecture and security approval  
**Customer owner:** Maya Chen  
**Migration lead:** Alex Rivera  
**Operations owner:** Northstar platform operations

## Preconditions

- POC evaluation has passed its approved gates and the production design review has an owner for each open risk.
- The customer approves the production data flow, retention, access scopes, and incident contact path.
- The service has a tested rollback to the existing manual workflow and a known-good serving revision.
- API and KV compatibility checks pass for every component in the rollout. A KV format change uses the versioned migration and hash verification procedure; an API v2 rollout preserves the documented v1 compatibility window.
- The on-call team has rehearsed the rollback and can read the relevant runbooks without the project team.

## Staged rollout

| Stage | Cohort and duration | Expansion gate |
|---|---|---|
| 0: shadow | 1 day, no customer-visible responses | No data boundary violations; logs and dashboards validated |
| 1: internal canary | 5 support users, 2 days | Error rate below approved limit; p95 latency within target; support lead approves samples |
| 2: limited use | 10% of eligible requests, 3 days | No unresolved critical quality or safety event; capacity headroom confirmed |
| 3: broad use | 50%, 5 days | Operations accepts alert load and runbook rehearsal; sponsor signs expansion |
| 4: default route | gradual, at least 1 day per step | Customer approves each step and manual fallback remains available |

At each stage compare candidate and baseline on the same metrics. A gate is not passed by missing data. The change owner records start/end times, revision, cohort, decision, and approver.

## Rollback and recovery

Immediately route new requests to the known-good manual or previous API path if the error budget is breached, a critical safety issue is confirmed, data reaches an unapproved destination, or the customer incident commander calls stop. Freeze expansion, preserve relevant logs under the approved retention policy, notify Maya and Priya, and open an incident record. Resume only after the cause is understood, a fix is reviewed, and the prior stage's gate passes again.

**Next action:** Alex drafts the API/KV compatibility matrix and rollback rehearsal; Northstar operations names the incident commander. **Approval:** Maya owns stage expansion; Priya owns data/security approval. This plan remains a draft until both approve it.
