# Quantities and relationships

A community group wants to host an event. The room costs 60 currency units and refreshments cost 4 per attendee. How much should the group budget?

Questions like this one are where mathematics starts: a situation in words, a few numbers, and a decision to make. This chapter shows you how to turn the situation into a simple model, a rule that connects the quantities you know to the one you want, and how to check that model with units and common sense. Every later chapter in this book builds on the cost rule you write here. All you need is arithmetic, a notebook, and a calculator.

> **Learning objectives**
> - Name the quantities in a situation and state the units of each.
> - Write a relationship between quantities as a function, such as $C(n) = 60 + 4n$.
> - Evaluate a function for a given input and use units to check the result.
> - Say what a model leaves out and which inputs it allows.

## Describe the relationship

Start with words: total cost is the room cost plus the refreshments for everyone. Let $n$ mean the number of attendees and $C(n)$ mean the total cost, measured in currency units.

$$
C(n) = 60 + 4n.
$$

The notation $C(n)$ means "the cost when attendance is $n$." It does not mean $C$ multiplied by $n$. This rule is a **function**: each allowed input gives one output. Here, allowed inputs are whole numbers at least zero.

## Work an example

With the rule written down, you can put numbers into it. For 20 attendees, substitute 20 for $n$:

$$
C(20) = 60 + 4(20) = 140.
$$

The units help check the model: 4 currency units per person times 20 people gives 80 currency units. Adding the room cost gives 140.

A model leaves things out. This one assumes the room price is fixed and refreshments cost the same for every attendee. It excludes transport and equipment.

## Try it

The budget is 180. What is the largest attendance this model allows? What changes if the room costs 80 instead?

<details>
<summary>Check your reasoning</summary>

Solve $60 + 4n \le 180$: subtract 60, then divide by 4, giving $n \le 30$. The maximum is 30 people. With an 80 room cost, $80 + 4n \le 180$ gives a maximum of 25.

</details>

## Where this leads

The cost rule $C(n) = 60 + 4n$ grows by the same amount for every extra attendee. The next chapter, [Change and growth](02-change-and-growth.md), compares that kind of change with growth by repeated multiplication. For a full treatment of functions and algebra, see the [Precalculus](../mathematics-for-computing/01-precalculus/README.md) chapter of *Mathematics for Computing*.

## Key terms

- **Quantity**: something you can measure or count, together with its unit.
- **Function**: a rule that gives exactly one output for each allowed input.
- **Model**: a simplified description of a situation that keeps the parts needed for a decision and leaves the rest out.

## Summary

- Start from words, then name each quantity and its unit.
- A function such as $C(n) = 60 + 4n$ connects an input (attendance) to an output (cost).
- Units are a quick check: currency per person times people gives currency.
- Every model has assumptions and allowed inputs; state them.

## Review questions

1. Explain why attendance cannot be negative or fractional in this model.
   <details><summary>Answer</summary>
   The input $n$ counts people, and you cannot have fewer than zero people or part of a person. So the allowed inputs are whole numbers at least zero. The formula $60 + 4n$ would happily accept $-3$ or $2.5$, which is why the allowed inputs must be stated alongside the rule.
   </details>
2. Does $C(n)$ mean $C$ times $n$?
   <details><summary>Answer</summary>
   No. $C(n)$ is read "$C$ of $n$": the output of the function $C$ when the input is $n$. For example, $C(20) = 140$ is the cost for 20 attendees.
   </details>
