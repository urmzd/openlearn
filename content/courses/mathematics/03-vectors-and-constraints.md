# Vectors and constraints

The event now offers two refreshment packs. A standard pack costs 4 currency units and a larger pack costs 6. We need to track both counts without confusing them.

Real plans involve several quantities at once, and several requirements that must all hold. This chapter introduces the vector, a way to keep related quantities together in a fixed order, and shows how to combine requirements into a set of conditions a plan must satisfy. By the end you will be able to decide whether a plan is feasible and find the most of something a budget allows.

> **Learning objectives**
> - Write related quantities as a vector and explain why the order matters.
> - Compute a total cost by multiplying matching entries of two vectors and adding.
> - Combine an equation and an inequality and solve them by substitution.
> - Decide whether a given plan is feasible.

## Keep related quantities together

A **vector** is an ordered collection of numbers. Write the pack counts as $(s,l)$: standard first, larger second. The plan $(12,8)$ means 12 standard packs and 8 larger packs. The order matters.

The corresponding prices are $(4,6)$. Multiply matching entries and add to find the refreshment cost:

$$
4s+6l.
$$

For $(12,8)$, the cost is $4(12)+6(8)=96$. Including the room, the total is $60+96=156$.

## Combine requirements

A cost is only half of a plan; the plan also has to meet its requirements. Suppose we need exactly 20 packs and have a total budget of 160. The plan must satisfy both:

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

## Where this leads

Every calculation so far has assumed you know exactly how many people will come. The next chapter, [Uncertainty and evidence](04-uncertainty-and-evidence.md), drops that assumption. For the full theory of vectors and of choosing the best feasible plan, see the [Linear Algebra](09-linear-algebra/README.md) and [Optimization](16-optimization/README.md) chapters in *Mathematics*.

## Key terms

- **Vector**: an ordered collection of numbers, such as $(s, l)$.
- **Constraint**: a condition a plan must satisfy, written as an equation or an inequality.
- **Feasible plan**: a plan that satisfies every constraint at once.
- **Substitution**: using one equation to replace a quantity in another, such as $s = 20 - l$.

## Summary

- A vector keeps related quantities together; the position of each entry says what it means.
- Multiplying matching entries of a count vector and a price vector and adding gives a total cost.
- Requirements become constraints; a feasible plan satisfies all of them.
- Substitution reduces several constraints to one condition on a single quantity, such as $l \le 10$.

## Review questions

1. Explain why satisfying the pack count alone does not guarantee a feasible plan.
   <details><summary>Answer</summary>
   The pack count $s + l = 20$ is only one of the constraints. A plan with the right number of packs can still cost too much: $(4, 16)$ has 20 packs but costs 172, more than a budget of 170. A feasible plan must satisfy the budget constraint and the count constraint together, and have whole-number counts at least zero.
   </details>
2. What is the difference between the plans $(12, 8)$ and $(8, 12)$?
   <details><summary>Answer</summary>
   Order matters in a vector. $(12, 8)$ is 12 standard and 8 larger packs, costing $48 + 48 = 96$; $(8, 12)$ is 8 standard and 12 larger packs, costing $32 + 72 = 104$.
   </details>
