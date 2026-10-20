# Uncertainty and evidence

Twenty people register. Will all twenty attend? A cost calculation can be exact for a given attendance while the attendance itself remains uncertain.

Until now every chapter has treated attendance as a known number. This chapter shows how to reason when it is not: how to describe a chance with a probability, how to compute an expected value, and, just as important, what an expected value does not tell you. You will also see where these calculations depend on assumptions such as independence, and what evidence should make you doubt them.

> **Learning objectives**
> - Compute an expected count from a number of trials and a probability.
> - Build a small probability table for independent events and check that it sums to 1.
> - Explain the difference between an expected value and a guaranteed outcome.
> - Identify when independence is an unreasonable assumption.

## Use an average carefully

Suppose previous, comparable events suggest each registrant has a 0.75 probability of attending. This is an assumption based on past evidence, not a promise about this event.

The **expected attendance** is:

$$
20 \times 0.75 = 15.
$$

An expected value describes an average across repeated situations under the model. It does not say that exactly 15 people will arrive.

## A small example

To see what the average hides, consider just two registrants. If their attendance is independent and each has probability 0.75 of attending, the possible outcomes have these probabilities:

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

## Where this leads

You now have every idea the project needs: a cost function, a way to compare kinds of change, constraints on a plan, and careful reasoning about uncertain attendance. The next chapter, [Project: plan a community event](05-community-event.md), brings them together. For the full theory of probability and of drawing conclusions from data, see the [Probability and Statistics](13-probability-statistics/README.md) chapter in *Mathematics*.

## Key terms

- **Probability**: a number between 0 and 1 describing how likely an outcome is under a model.
- **Expected value**: the average outcome across many repetitions of a situation under the model.
- **Independence**: two events are independent when knowing one happened does not change the probability of the other.

## Summary

- A probability is an assumption drawn from evidence, not a promise.
- Expected count = number of trials times probability: $20 \times 0.75 = 15$.
- An expected value can be a number no single outcome can take, such as 1.5 attendees.
- Probability tables for several people multiply individual probabilities only when the events are independent; expected counts add either way.

## Review questions

1. Explain the difference between "expected attendance is 15" and "attendance will be 15."
   <details><summary>Answer</summary>
   "Expected attendance is 15" describes the average over many comparable events under the model; any one event may have more or fewer attendees. "Attendance will be 15" is a prediction of the exact outcome of this event, which the model cannot make. Even with two registrants, the expected 1.5 attendees is not a possible outcome at all.
   </details>
2. Why does the probability table require independence when the expected count does not?
   <details><summary>Answer</summary>
   The table multiplies the two people's probabilities, for example $0.75 \times 0.75$ for "both", which is correct only if one person's attendance does not affect the other's. The expected count is a sum of each person's expected attendance, $0.75 + 0.75 = 1.5$, and expected values add whether or not the events are independent.
   </details>
