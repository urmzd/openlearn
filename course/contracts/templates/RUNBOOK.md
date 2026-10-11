# Runbook: <alert or failure name>

<One sentence: what is broken from a user's point of view when this page is
open, and which alert or symptom brought you here.>

## Symptoms

- <What the pager, a user, or a dashboard shows: the alert name, the error
  code users get (for example 503 no_capacity), the panel that turns red.>
- <What `kubectl get pods -n <system>` or `/readyz` shows.>

## Diagnosis

1. <The first command to run, exactly, and what a healthy and a broken
   output look like.>
2. <The next check, chosen by the output of the previous one.>
3. <How to tell this failure from the ones that look like it.>

## Mitigation

1. <The fastest safe action that restores service, exactly (a rollback, a
   drain, a scale-up), and how to confirm it worked.>
2. <What not to do, and why (for example: do not delete the WAL PVC).>

## Follow-up

- <The fix that removes the cause, and where it is tracked.>
- <Links: the alert rule, the dashboard, the ADR or chapter that explains
  the component.>

<!-- contracts/templates/RUNBOOK.md (ops.00, craft.10): copy to
     docs/runbooks/<name>.md, one per required alert (otel/slo.schema.json)
     and per drill. `ol drill end` checks the Symptoms, Diagnosis, and
     Mitigation sections. -->
