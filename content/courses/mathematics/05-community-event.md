# Project: plan a community event

This project brings together everything in the book. You will define quantities and write a cost function as in [Quantities and relationships](01-quantities-and-relationships.md), reason about how cost changes as in [Change and growth](02-change-and-growth.md), find a feasible plan under constraints as in [Vectors and constraints](03-vectors-and-constraints.md), and treat attendance as uncertain as in [Uncertainty and evidence](04-uncertainty-and-evidence.md).

Use these ideas to propose an event plan, and write your reasoning so another person can check it. A notebook or spreadsheet is enough; code is optional.

> **Learning objectives**
> - Build a cost model with stated quantities, units, and assumptions.
> - Find a plan that satisfies every constraint and identify the limit a budget imposes.
> - Use expected attendance correctly, without treating it as a guarantee.
> - Stress-test a plan against changed conditions and say what you would change.

## The situation

There are 30 registrants. The room costs 60 currency units. A standard refreshment pack costs 4 and a larger pack costs 6. Your total budget is 200. Each attendee needs one pack. Previous comparable events suggest an attendance probability of 0.75 per registrant.

## Build your plan

Work through these steps in order, writing down your reasoning for each.

1. Define your quantities and units. Write a cost function for standard and larger pack counts.
2. Propose a plan that supplies all 30 registrants and stays within budget. Find the largest number of larger packs you could include.
3. Calculate expected attendance. Explain why this number does not guarantee enough refreshments.
4. Stress-test your plan: what if everyone attends, or the room price rises to 90? Explain what you would change.
5. Name two assumptions you would check with the organizer before spending money.

## Check your work

When you have a plan, compare it with the one below.

<details>
<summary>Compare with a feasible plan</summary>

With $s+l=30$, the cost is $60+4s+6l=180+2l$. The budget allows at most ten larger packs. Twenty standard packs and ten larger packs cost 200 and supply everyone.

Expected attendance is $30(0.75)=22.5$; this is an average, not a possible headcount or an upper bound. Supplying all 30 avoids a shortage under the stated registration limit.

With a 90 room price, even 30 standard packs cost 210. Serving everyone exceeds the budget by at least 10. You need a budget increase, lower costs, or an agreed change in the event plan.

</details>

You are ready to continue when you can explain your model, check its units and constraints, and identify where its assumptions could fail. These are self-checks, not automated grading.

## Where this leads

Each idea in this project opens onto a full subject in *Mathematics*:

- For functions, growth, and algebra: [Precalculus](06-precalculus/README.md).
- For change and accumulation: [Calculus 1](07-calculus-1/README.md).
- For many quantities and constraints: [Linear Algebra](09-linear-algebra/README.md).
- For evidence and uncertainty: [Probability and Statistics](13-probability-statistics/README.md).

[Choose your next subject](study-plan.md) helps you pick one. You do not need to finish all of *Mathematics* before applying what you have learned.

## Summary

- A good plan starts with named quantities, units, and a cost function.
- Constraints decide feasibility: here, 30 packs and a budget of 200 allow at most ten larger packs.
- Expected attendance (22.5) is an average, not a headcount; plan for the registration limit.
- Stress-testing reveals when a plan breaks, as it does when the room price rises to 90.

## Review questions

1. Why does the feasible plan supply 30 packs when expected attendance is only 22.5?
   <details><summary>Answer</summary>
   Expected attendance is an average over many comparable events, not an upper bound. Any number of registrants up to 30 might come, so buying 22 or 23 packs risks a shortage. Supplying all 30 registrants guarantees everyone a pack under the stated registration limit, and the budget allows it.
   </details>
2. With the room at 90, why can no choice of packs serve everyone within budget?
   <details><summary>Answer</summary>
   Standard packs are the cheapest option, and 30 of them cost 120. With a 90 room the cheapest plan that serves everyone costs 210, which is 10 over the budget of 200. Any larger packs only raise the cost, so the budget, the costs, or the plan itself must change.
   </details>
