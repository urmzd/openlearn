# Change and growth

An extra attendee adds 4 to the event cost. But an invitation shared by each recipient can grow by multiplication. These are different kinds of change, and confusing them leads to very different predictions.

This chapter takes the cost rule from [Quantities and relationships](01-quantities-and-relationships.md) and asks how its output changes when the input changes. You will see that a fixed addition gives the same step every time, while repeated multiplication gives steps that keep getting larger, and you will practise telling the two apart and writing a rule for each.

> **Learning objectives**
> - Compute how much a linear rule changes when its input changes, and recognize a constant rate.
> - Write a rule for repeated multiplication, such as $R(r) = 10 \times 2^r$.
> - Convert a percentage growth rate into a multiplier.
> - Compare additive and multiplicative growth over the first few steps.

## Fixed additions

Our cost rule is $C(n) = 60 + 4n$. Adding five attendees changes cost by:

$$
C(n+5)-C(n)=4(5)=20.
$$

The starting attendance does not matter: each extra person adds the same cost. The rate is 4 currency units per person. The fixed room cost disappears when we compare two attendance levels.

## Repeated multiplication

Invitations behave differently. Suppose 10 people receive an invitation, and the number of people reached doubles each round. Let $r$ count rounds, with the initial group at $r=0$:

$$
R(r)=10 \times 2^r.
$$

At rounds 0, 1, 2, and 3, the model gives 10, 20, 40, and 80 people. The additions get larger even though the multiplier stays the same.

This assumes each round reaches distinct new people. Real sharing overlaps and eventually slows, so this rule should not be extended indefinitely.

## Try it

A mailing list starts with 100 subscribers and grows by 10 percent each month. Write a rule for its size after $m$ months, assuming the same growth rate continues. Compare its first two months with adding 10 subscribers each month.

<details>
<summary>Check your reasoning</summary>

A 10 percent increase multiplies the current size by 1.1, so $S(m)=100(1.1)^m$. It gives 110 then 121. Adding 10 each month gives 110 then 120. These are model values; actual subscriber counts are whole numbers.

</details>

## Where this leads

So far each quantity has been a single number. The next chapter, [Vectors and constraints](03-vectors-and-constraints.md), tracks several quantities at once and asks which plans satisfy several requirements together. For more on growth, see [Exponents, logarithms, and units of information](06-precalculus/01-exponents-logs-and-units-of-information.md) and the [Calculus 1](07-calculus-1/README.md) chapter in *Mathematics*.

## Key terms

- **Rate of change**: how much the output changes per unit change in the input; for $C(n) = 60 + 4n$ it is 4 per person.
- **Linear growth**: change by adding the same amount at each step.
- **Exponential growth**: change by multiplying by the same factor at each step.
- **Multiplier**: the factor applied at each step; a 10 percent increase is a multiplier of 1.1.

## Summary

- A linear rule changes by the same amount for every unit step, whatever the starting point.
- A multiplicative rule changes by the same factor, so its steps keep growing.
- A percentage increase of $p$ percent is a multiplier of $1 + p/100$.
- Growth models have limits: real sharing overlaps and slows, so do not extend a rule indefinitely.

## Review questions

1. Explain why doubling and adding two are different operations.
   <details><summary>Answer</summary>
   Adding two always increases a quantity by 2, whatever its size. Doubling increases it by its current size, so the increase depends on where you start: doubling 10 adds 10, doubling 40 adds 40. Starting from 2, the two happen to agree once, but after that doubling pulls ahead and the gap keeps widening.
   </details>
2. Why does the room cost disappear when you compute $C(n+5) - C(n)$?
   <details><summary>Answer</summary>
   The room cost of 60 appears in both $C(n+5)$ and $C(n)$, so it cancels in the difference. A change in attendance changes only the refreshment part, $4 \times 5 = 20$.
   </details>
