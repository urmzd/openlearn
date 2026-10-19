# 2. Change and growth

An extra attendee adds 4 to the event cost. But an invitation shared by each recipient can grow by multiplication. These are different kinds of change.

## Fixed additions

Our cost rule is $C(n) = 60 + 4n$. Adding five attendees changes cost by:

$$
C(n+5)-C(n)=4(5)=20.
$$

The starting attendance does not matter: each extra person adds the same cost. The rate is 4 currency units per person. The fixed room cost disappears when we compare two attendance levels.

## Repeated multiplication

Suppose 10 people receive an invitation, and the number of people reached doubles each round. Let $r$ count rounds, with the initial group at $r=0$:

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

Before moving on, explain why doubling and adding two are different operations.

Next: [Vectors and constraints](03-vectors-and-constraints.md). Further study: [exponents and logarithms](../mathematics-for-computing/00-precalculus/01-exponents-logs-and-units-of-information.md) and [calculus](../mathematics-for-computing/01-calculus-1/).
