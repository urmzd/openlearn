# 3. Vectors and constraints

The event now offers two refreshment packs. A standard pack costs 4 currency units and a larger pack costs 6. We need to track both counts without confusing them.

## Keep related quantities together

A **vector** is an ordered collection of numbers. Write the pack counts as $(s,l)$: standard first, larger second. The plan $(12,8)$ means 12 standard packs and 8 larger packs. The order matters.

The corresponding prices are $(4,6)$. Multiply matching entries and add to find the refreshment cost:

$$
4s+6l.
$$

For $(12,8)$, the cost is $4(12)+6(8)=96$. Including the room, the total is $60+96=156$.

## Combine requirements

Suppose we need exactly 20 packs and have a total budget of 160. The plan must satisfy both:

$$
s+l=20, \qquad 60+4s+6l \le 160.
$$

Since $s=20-l$, substitute into the budget:

$$
60+4(20-l)+6l \le 160,
$$

which simplifies to $140+2l \le 160$, so $l \le 10$. Counts must also be whole numbers at least zero. Up to ten larger packs fit this budget.

## Try it

For the same 20 packs, increase the budget to 170. What is the maximum number of larger packs? Is $(4,16)$ feasible?

<details>
<summary>Check your reasoning</summary>

$140+2l \le 170$ gives $l \le 15$. The maximum is 15 larger packs and 5 standard packs. The plan $(4,16)$ costs $60+16+96=172$, so it exceeds the budget.

</details>

Before moving on, explain why satisfying the pack count alone does not guarantee a feasible plan.

Next: [Uncertainty and evidence](04-uncertainty-and-evidence.md). Further study: [linear algebra](../mathematics-for-computing/03-linear-algebra/) and [optimization](../mathematics-for-computing/10-optimization/).
