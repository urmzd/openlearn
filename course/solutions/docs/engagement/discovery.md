# Mock engagement discovery: Northstar Commerce

**Status:** discovery complete, qualification pending  
**Date:** 2026-10-09  
**Customer owner:** Maya Chen, platform lead  
**Supersource owner:** Alex Rivera, solutions engineer

## Customer problem

Northstar reports that its internal support assistant becomes slow during weekday order-reconciliation peaks. The stated symptom is “responses take too long”; no customer trace or production percentile report has been supplied yet. The current impact claim is that support staff retry requests and switch to manual lookup. We have not measured the number of affected staff or minutes lost.

The discovery call established a decision: Northstar will decide by 2026-10-23 whether to fund a bounded proof of concept. Maya owns the problem statement and will arrange access to a redacted request sample. The security reviewer, Priya Shah, must approve the sample before transfer. The budget owner is not yet identified.

## Questions answered and evidence

| Topic | Current answer | Evidence / confidence |
|---|---|---|
| Request mix | Mostly order lookup and policy questions; prompt and output lengths are unknown | Maya's interview notes, low confidence |
| Impact | Staff retry and use manual lookup during peaks | Customer report, not instrumented |
| Baseline | No agreed latency or error baseline | Open; requires gateway/load report |
| Constraints | No raw customer identifiers in a shared test set | Priya's verbal requirement; written policy pending |
| Decision | Sponsor wants a go/no-go recommendation by Oct 23 | Meeting notes, confirmed by Maya |

We did not promise a latency improvement or production capacity. The course platform's load reports and runbooks show how to gather relevant evidence, but they are not measurements of Northstar's workload.

## Success conditions for the next meeting

1. Maya supplies a redacted sample with a data dictionary and prompt/output length distribution.
2. Priya approves the transfer method and retention period in writing.
3. The customer and engineering team agree on baseline traffic, latency percentiles, error rate, and the peak window.
4. A named budget owner confirms the decision process and date.

**Next action:** Maya sends the data dictionary and proposed sample by Oct 14. Alex reviews it for missing workload fields and schedules qualification. **Unresolved:** budget owner, measured impact, retention period, and production data boundary.
