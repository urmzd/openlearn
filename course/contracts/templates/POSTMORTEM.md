# Postmortem: <short title> (<YYYY-MM-DD>)

Blameless: describe what the system and the process allowed, not who
erred.

## Summary

<Two or three sentences: what failed, for how long, how it was fixed.>

## Impact

- <Who was affected and how: failed requests (count and ratio), latency,
  error-budget minutes burned, data lost or delayed (none is an answer).>

## Timeline

All times UTC.

| Time | Event |
|---|---|
| <HH:MM:SS> | <injection or trigger> |
| <HH:MM:SS> | <first signal: alert, page, user report> |
| <HH:MM:SS> | <each diagnosis step and decision> |
| <HH:MM:SS> | <mitigation applied> |
| <HH:MM:SS> | <recovery confirmed, and how> |

## Root cause

<The chain of causes, down to the one that, once removed, prevents this
class of failure. Name the component, the file, and the condition.>

## Detection

<How it was detected and how long that took. Would an alert have caught it
sooner? Which one, with what threshold?>

## Resolution

<What restored service, and what fixed the cause (they may differ).>

## Action items

| Action | Kind (prevent, detect, mitigate) | Owner | Done |
|---|---|---|---|
| <action> | <kind> | <you> | <link or "open"> |

<!-- contracts/templates/POSTMORTEM.md (ops.*, every drill): copy to
     docs/postmortems/<date>-<drill>.md. `ol drill end` checks that every
     section heading above is present and non-empty; content is graded with
     course/rubrics/postmortem.md. -->
