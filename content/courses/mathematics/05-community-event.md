# 5. Project: plan a community event

Use the ideas from the previous chapters to propose an event plan. Write your reasoning so another person can check it. A notebook or spreadsheet is enough; code is optional.

## The situation

There are 30 registrants. The room costs 60 currency units. A standard refreshment pack costs 4 and a larger pack costs 6. Your total budget is 200. Each attendee needs one pack. Previous comparable events suggest an attendance probability of 0.75 per registrant.

## Build your plan

1. Define your quantities and units. Write a cost function for standard and larger pack counts.
2. Propose a plan that supplies all 30 registrants and stays within budget. Find the largest number of larger packs you could include.
3. Calculate expected attendance. Explain why this number does not guarantee enough refreshments.
4. Stress-test your plan: what if everyone attends, or the room price rises to 90? Explain what you would change.
5. Name two assumptions you would check with the organizer before spending money.

## Check your work

<details>
<summary>Compare with a feasible plan</summary>

With $s+l=30$, the cost is $60+4s+6l=180+2l$. The budget allows at most ten larger packs. Twenty standard packs and ten larger packs cost 200 and supply everyone.

Expected attendance is $30(0.75)=22.5$; this is an average, not a possible headcount or an upper bound. Supplying all 30 avoids a shortage under the stated registration limit.

With a 90 room price, even 30 standard packs cost 210. Serving everyone exceeds the budget by at least 10. You need a budget increase, lower costs, or an agreed change in the event plan.

</details>

You are ready to continue when you can explain your model, check its units and constraints, and identify where its assumptions could fail. These are self-checks, not automated grading.

## Choose what to study next

- For functions, growth, and algebra: [precalculus](../mathematics-for-computing/00-precalculus/).
- For change and accumulation: [calculus](../mathematics-for-computing/01-calculus-1/).
- For many quantities and constraints: [linear algebra](../mathematics-for-computing/03-linear-algebra/).
- For evidence and uncertainty: [probability and statistics](../mathematics-for-computing/07-probability-statistics/).

Use the [study guide](study-plan.md) to choose one subject. You do not need to finish the entire mathematics library before applying what you have learned.
