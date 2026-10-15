# 4. Uncertainty and evidence

Twenty people register. Will all twenty attend? A cost calculation can be exact for a given attendance while the attendance itself remains uncertain.

## Use an average carefully

Suppose previous, comparable events suggest each registrant has a 0.75 probability of attending. This is an assumption based on past evidence, not a promise about this event.

The **expected attendance** is:

$$
20 \times 0.75 = 15.
$$

An expected value describes an average across repeated situations under the model. It does not say that exactly 15 people will arrive.

## A small example

Consider just two registrants. If their attendance is independent and each has probability 0.75 of attending:

| Attendance | Probability |
|---|---|
| Neither | $0.25 \times 0.25 = 0.0625$ |
| Exactly one | $2 \times 0.75 \times 0.25 = 0.375$ |
| Both | $0.75 \times 0.75 = 0.5625$ |

The probabilities sum to 1. The expected attendance is $0(0.0625)+1(0.375)+2(0.5625)=1.5$, even though an actual event cannot have one and a half attendees.

Independence means one person's attendance does not change the other's probability. Shared transport or weather can make this assumption unreasonable. Expected counts still add without independence, but this probability table requires it.

## Try it

Why might buying only 15 packs for 20 registrants leave someone without refreshments? What evidence would make the 0.75 estimate less trustworthy?

<details>
<summary>Check your reasoning</summary>

Attendance can exceed its expected value. A different audience, season, location, or registration process could change attendance behavior. A shared disruption could also make attendance correlated. An average alone does not determine how many packs to buy.

</details>

Before moving on, explain the difference between “expected attendance is 15” and “attendance will be 15.”

Next: [Plan a community event](05-community-event.md). Further study: [probability and statistics](../../tracks/math/07-probability-statistics/).
