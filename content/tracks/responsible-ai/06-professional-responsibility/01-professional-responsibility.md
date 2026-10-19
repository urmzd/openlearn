# Professional responsibility and disclosure

## Overview

| | |
|---|---|
| **Module** | `ethics.06` · proof · docs · Pass 11 · 1 to 2 h |
| **Solution** | a model answer for `docs/responsibility/reflection.md` in section 5: a written review of one consequential claim, with its evidence, and the design-review checklist answered line by line |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |

## Key Takeaways

- Tie each claim to the exact platform evidence that produces it.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 state what professional responsibility asks of an engineer shipping this system and set up a worked review of one claim. Section 4 fixes what the reflection must contain. Section 5 is a model answer, with the reasoning behind each paragraph. Try writing the review of section 3 yourself before reading the model answer.

---

## 1. Why now

A production system affects people through data choices, model behavior, access controls, and incident decisions. Professional responsibility includes making those effects legible to users and decision makers.

## 2. Principles

Disclose capabilities and limits in language suited to the audience. Correct material errors promptly. Protect confidential information, distinguish measured results from assumptions, and escalate harm through a named channel.

## 3. Worked example

Choose one consequential claim from the model card or customer evaluation. Explain what evidence supports it, what is unknown, who could be harmed if it is overstated, and what wording or product control reduces that risk.

The model answer below takes the safety claim of the reference model card ([`MODEL_CARD.md`](../../../../course/solutions/docs/MODEL_CARD.md), `ethics.03`): the row "toxicity-rate over 18 prompts: 0.000 (0.000, 0.176)". It is a good choice because it is easy to overstate: "the model produced no toxic output" is true and sounds like "the model is safe", which the evidence does not support.

## 4. Claim and checklist

`docs/responsibility/reflection.md` is the worked reflection or design review. It answers each line of the design-review checklist below with evidence from the completed system; each line that cannot be answered "yes" becomes a review note with an owner and next step.

- The problem, its users, and the constraints are stated before any solution.
- At least two options are compared on the same criteria, with numbers where numbers exist.
- The chosen option names what it gives up, and why that cost is acceptable.
- Every interface between components names its contract (an `.h`, `.proto`, OpenAPI path, or schema).
- Failure modes are listed with how each is detected and how the system degrades.
- Capacity and latency budgets are estimated, with the arithmetic shown.
- Rollout and rollback are described, including data and contract migrations.
- Open questions are listed with an owner.

## 5. The solution, step by step

### From first principles

1. **Start from the claim as a reader will hear it.** A disclosure is judged by what its audience concludes, not by whether each word is true. "0 toxic outputs" will be heard as "safe".
2. **Separate what was measured from what was assumed.** The measurement is a rate on a fixed prompt set with a specific scorer; the assumptions are that the prompts represent real use and the scorer detects real toxicity. Each assumption needs its own evidence or its own caveat.
3. **Size the evidence.** With $n = 18$ and $k = 0$, the Wilson upper bound is $z^2/(n + z^2) = 3.8415/21.8415 \approx 0.176$: the data are consistent with a true rate up to about 18%. The scorer's own ROC-AUC on labelled text is 0.752, so a miss is plausible.
4. **Name who is harmed by overstatement.** Users routed to the model, operators who rely on the card to decide routing, and anyone who repeats the claim downstream.
5. **Reduce the risk with wording and with a control.** Wording alone can be ignored; a product control (the usage policy routes no user traffic to the model) makes the claim's limits enforceable. Both are needed.
6. **Close the loop.** Every unresolved point gets an owner, a next step, and a condition that would change the decision.

### The model answer

The text below is a complete `docs/responsibility/reflection.md`. Comments in brackets explain why each paragraph is there; they are not part of the document.

```markdown
# Reflection: the safety row of shakespeare-kn4 v0

## Claim under review

The model card's evaluation table reports "toxicity-rate over 18 prompts:
0.000 (0.000, 0.176), release threshold: upper bound at most 0.2". The
release gate (dur.12) passed it on that row.

## Audience and constraints

Readers: operators deciding which models the gateway may route to, and
anyone who reads the published card. Constraint: the model is a byte-level
4-gram that follows no instructions; it is released as a teaching baseline.

## Evidence

- Measured: 0 of 18 sampled outputs scored at or above 0.5 by the ethics.04
  lexicon scorer, one seeded PCG32 stream, 80 bytes per prompt. Wilson 95%
  interval (0.000, 0.176).
- Instrument quality: the scorer's ROC-AUC on the 30 labelled ethics.04 texts
  is 0.752; it misses insults without a listed word.
- Assumed, not measured: that the 18 prompts represent how anyone would use
  the model, and that a lexicon hit is what a person would call toxic.

## Options compared

| Option | What a reader concludes | Cost |
|---|---|---|
| A. Report the row as is | "the model is safe" | none now; a false reassurance later |
| B. Report the row with the interval and the scorer's AUC | "no evidence of toxicity at this sample size, weak scorer" | one paragraph |
| C. B, plus a control that routes no user traffic to the model | as B, and the claim cannot be relied on for user-facing use | one policy rule, already present |

Chosen: C. It gives up nothing the model is for (it is not for users) and
makes the limit enforceable rather than advisory.

## Who could be harmed if it is overstated

Users routed to the model, if an operator read the row as a safety
certification; downstream readers who quote "0 toxic outputs".

## Wording and controls

- The card states the interval and that "a rate of zero here is weak evidence
  on its own" (done).
- The card's out-of-scope list says the gateway must route no user traffic
  to the model; the usage policy (ethics.05) enforces model restrictions by
  rule id (done).

## Failure modes

| Failure | Detection | Degradation |
|---|---|---|
| The model is added to a user route | the usage policy denies it; deny rate by rule is monitored | requests get 451 with a rule id |
| The scorer misses toxic text | periodic human review of a sample, through the privacy-approved process | the row is reported as a lower bound on concern, not certification |
| The card is copied without the interval | none automatic | correction issued by the card owner (below) |

## Checklist

- Problem, users, constraints stated first: yes (sections 1 and 2).
- Two or more options compared with numbers: yes (A, B, C; n = 18, upper
  bound 0.176, AUC 0.752).
- What the choice gives up: yes (C costs one rule; nothing the model is for).
- Contracts named: yes (formats/eval-results.schema.json for the row,
  formats/policy.v1.schema.json for the rule).
- Failure modes with detection and degradation: yes (table above).
- Capacity and latency budgets: not applicable to a disclosure; the policy
  rule is a metadata check with no measurable latency. Noted, no action.
- Rollout and rollback: partly. Note R1 below.
- Open questions with owners: yes (below).

## Review notes and open questions

- R1 (owner: release owner): the card has no documented procedure for
  issuing a correction if a number is found wrong. Next step: add a
  "Corrections" line to the card template that records the date, the old and
  new value, and the reason. Done when the template has it and this card
  uses it.
- Q1 (owner: data owner): would a larger prompt set (at least 150 prompts,
  so that 0 events gives an upper bound near 0.025) change any routing
  decision? Only if the model is ever proposed for user traffic; revisit then.

## Decision

Keep the row, with its interval and scorer caveat, and keep the routing
control. Escalation channel for any observed harm: the gateway safety review
named in docs/USAGE_POLICY.md.
```

### Why this answer works

- **The claim is reviewed as heard, not as written.** Option A is true and still misleading; the review exists to catch exactly that.
- **Every number has a source.** The interval comes from the eval row, the AUC from the scorer row, the routing control from the usage policy; a reader can check each one.
- **Measured and assumed are separate sections.** That is the distinction the principles ask for, and it is what tells the next reviewer which assumptions to test.
- **A "no" on the checklist becomes work, not a shrug.** Rollback of a published claim (a correction) was missing; the review turns it into note R1 with an owner and a done condition.
- **Tradeoffs accepted.** The review does not demand more evaluation now, because the routing control already prevents the harm and a larger suite would only matter for a use the model is not for. It records the condition (a proposal for user traffic) under which that changes.

### Explain it back

**1. Why is reporting "0 toxic outputs" without the interval a disclosure failure, even though it is true?**

<details><summary>Answer</summary>

Readers will conclude the model is safe. With 18 prompts, zero events is consistent with a true rate up to about 18% (the Wilson upper bound), and the scorer itself is a weak classifier (ROC-AUC 0.752 on labelled text), so some toxic outputs may have gone undetected. A disclosure is responsible when the conclusion a reader draws matches the evidence, not merely when each sentence is literally true.

</details>

**2. Why does the model answer insist on a product control in addition to better wording?**

<details><summary>Answer</summary>

Wording can be ignored, misread, or stripped when a number is copied. A control (the usage policy denying user traffic to the model, by rule id) makes the limit hold regardless of who reads the card, and produces evidence (deny counts by rule) that the limit is being exercised.

</details>

**3. How many prompts would make "0 of n" bound the rate below 2.5%, and why does the review not require them now?**

<details><summary>Answer</summary>

At $k = 0$ the Wilson upper bound is $z^2/(n + z^2)$; setting it to 0.025 gives $n \approx 3.84/0.025 - 3.84 \approx 150$. The review does not require it now because the model is not for user traffic and the routing control already prevents the harm; it records the condition (a proposal for user traffic) under which the question must be revisited, with an owner.

</details>

**4. What turns a checklist "no" into something useful?**

<details><summary>Answer</summary>

An owner, a concrete next step, and a done condition. "Rollback partly" on its own is a complaint; R1 says who adds a corrections procedure, where, and how anyone can tell it is finished.

</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Treating version selection or a document claim as implicit | A reader or client cannot determine which contract or evidence applies. | the review names the model and version, the exact row, and the schemas it follows |
| 2 | Skipping the failure path | The successful example works, but malformed input or rollback breaks the system. | a failure-mode table with detection and degradation, and a correction procedure as R1 |
| 3 | Omitting a named owner and recovery condition | The artifact cannot guide an operator when conditions change. | every note and question has an owner and a done or revisit condition |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `ethics.05` | The usage policy is the control the review relies on; its rule ids appear in the failure table. |
| Back | `craft.19` | Design-review practice: the same checklist, applied to a technical change instead of a disclosure. |

## Going further

| Resource | What to inspect |
|---|---|
| [Course contracts](../../../../course/contracts/) | Use the current contracts and system boundaries as evidence when defending this review. |
| [Reference model card](../../../../course/solutions/docs/MODEL_CARD.md) | The claims a review like this one examines. |
