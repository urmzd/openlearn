# Tools and a worked proof

This part of [Discrete math 1 problem set: sets, counting, proofs, induction, relations, bijections](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Connect set, logic, counting, and induction tools to a worked argument.

## The ideas you will use

Before the problems, here is a review of the definitions and methods they rely on, beginning with the notation:


| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in A$ | $x$ is an element of the set $A$ | |
| $A \subseteq B$ | every element of $A$ is in $B$ | |
| $A \cup B,\ A \cap B,\ A \setminus B$ | union (in either), intersection (in both), difference (in $A$, not in $B$) | sets |
| $A^c$ | complement $U \setminus A$ inside a universe $U$ | set |
| $\lvert A \rvert$ | the number of elements of a finite set | integer |
| $\lnot, \land, \lor, \Rightarrow, \iff$ | not, and, or, implies, if and only if | propositions |
| $\forall, \exists$ | "for every", "there exists" | quantifiers |
| $n!$ | $1 \cdot 2 \cdots n$, with $0! = 1$ | integer |
| $\binom{n}{k}$ | $\frac{n!}{k!(n-k)!}$, the number of $k$-element subsets of an $n$-set | integer |
| $a \mid b$ | $a$ divides $b$: $b = ka$ for an integer $k$ | relation |
| $f: X \to Y$ | a function: each $x \in X$ has exactly one image $f(x) \in Y$ | |
| $g \circ f$ | composition, $x \mapsto g(f(x))$ | function |


### Sets

A **set** is an unordered collection of distinct elements, written by listing, $\{1, 2, 3\}$, or by a rule, $\{x \in \mathbb{Z} : x > 0\}$. Order and repetition do not matter: $\{2, 1, 2\} = \{1, 2\}$. Two sets are **equal** when each is a subset of the other, which is how set identities are proved: take an arbitrary element of one side and show it is in the other, then the reverse ("double inclusion"). The **power set** $\mathcal{P}(A)$ is the set of all subsets of $A$; it has $2^{\lvert A \rvert}$ elements, because each element is independently in or out.

A boolean mask is a set in disguise. An attention mask over $T$ positions is a subset of $\{0, \dots, T-1\}^2$, the pairs (query $i$, key $j$) that may interact, stored as a 0/1 matrix. The causal mask is $\{(i, j) : j \le i\}$, a padding mask is $\{(i, j) : j < L\}$, and applying both is their intersection, elementwise AND.

### Logic and quantifiers

A **proposition** is a statement that is true or false. Connectives build new ones, defined by truth tables: $P \land Q$ is true when both are, $P \lor Q$ when at least one is, $\lnot P$ flips $P$, and $P \Rightarrow Q$ is false only when $P$ is true and $Q$ false. Two propositions are **logically equivalent** when they agree on every row of the truth table. The **contrapositive** $\lnot Q \Rightarrow \lnot P$ is equivalent to $P \Rightarrow Q$; the **converse** $Q \Rightarrow P$ is not. **De Morgan's laws** move a negation inward and swap the connective: $\lnot (P \lor Q) \iff \lnot P \land \lnot Q$ and $\lnot (P \land Q) \iff \lnot P \lor \lnot Q$. For sets they read $(A \cup B)^c = A^c \cap B^c$.

A **predicate** $P(x)$ becomes a proposition once $x$ is fixed. **Quantifiers** close it over a domain: $\forall x\, P(x)$ ("for every $x$") and $\exists x\, P(x)$ ("there exists $x$"). Negation swaps them: $\lnot \forall x\, P(x) \iff \exists x\, \lnot P(x)$, and $\lnot \exists x\, P(x) \iff \forall x\, \lnot P(x)$. With nested quantifiers, swap each one and negate the innermost predicate.

### Counting

Four rules count almost everything in this book.

- **Product rule.** A choice made in $k$ independent steps with $n_1, \dots, n_k$ options has $n_1 n_2 \cdots n_k$ outcomes. Strings of length $k$ over an alphabet of size $n$: $n^k$.
- **Ordered without repetition.** Arranging $k$ of $n$ distinct items in order: $n(n-1)\cdots(n-k+1) = \frac{n!}{(n-k)!}$. All $n$ of them: $n!$.
- **Unordered without repetition.** Choosing a $k$-subset: $\binom{n}{k}$, the ordered count divided by the $k!$ orders of each subset.
- **Unordered with repetition ("stars and bars").** Placing $n$ identical items into $k$ distinct boxes is arranging $n$ stars and $k - 1$ bars in a row: $\binom{n + k - 1}{k - 1}$.

The **sum rule** adds the sizes of disjoint cases. When the cases overlap, **inclusion-exclusion** corrects the double count: $\lvert A \cup B \rvert = \lvert A \rvert + \lvert B \rvert - \lvert A \cap B \rvert$.

Model sizes are counts of this kind. A matrix of shape $m \times n$ holds $mn$ numbers; a layer's parameter count is the sum over its matrices and vectors. A matrix product $C = AB$ with $A$ of shape $m \times k$ and $B$ of shape $k \times n$ computes $mn$ outputs, each a sum of $k$ products, so it takes $mkn$ multiplications and as many additions: $2mkn$ floating-point operations (FLOPs). A forward pass over $N$ parameters costs about $2N$ FLOPs per token, and training about $6N$ (the backward pass costs twice the forward), so training on $D$ tokens costs about $6ND$. The KV cache stores one key vector and one value vector of length $d_h$ per KV head per layer per token, at $b$ bytes per number: $2 L H_{kv} d_h b$ bytes per token.

### Proof methods

A **proof** is a finite chain of statements, each a definition, an assumption, an earlier result, or a logical consequence of earlier lines, ending in the claim. The methods differ in what they assume first.

| Method | To prove $P \Rightarrow Q$ | Use it when |
|---|---|---|
| Direct | assume $P$, derive $Q$ | definitions unfold forward (q17) |
| Contrapositive | assume $\lnot Q$, derive $\lnot P$ | $\lnot Q$ gives you something to compute with (q18) |
| Contradiction | assume $P$ and $\lnot Q$, derive a false statement | the claim says something does not exist (q19, q20) |
| Cases | split $P$ into exhaustive cases, prove each | parity, signs, ranges (q21) |
| Counterexample | exhibit one $x$ with $\lnot Q(x)$ | disproving a $\forall$ claim (q23) |
| Pigeonhole | more than $k$ objects in $k$ boxes put two in one box | collisions (q24) |

Definitions are the raw material: an integer $n$ is **even** if $n = 2k$ and **odd** if $n = 2k + 1$ for an integer $k$; a number is **rational** if it is $p/q$ for integers $p, q$ with $q \ne 0$; an integer $p \ge 2$ is **prime** if its only positive divisors are 1 and $p$.

### Induction and loop invariants

**Induction** proves $P(n)$ for every integer $n \ge n_0$ in two steps: the **base case** $P(n_0)$, and the **inductive step** "if $P(k)$ then $P(k + 1)$" for every $k \ge n_0$. The assumption $P(k)$ is the **induction hypothesis**, and the step must use it rather than the conclusion. **Strong induction** assumes $P(m)$ for every $m$ with $n_0 \le m < n$ and proves $P(n)$; it suits claims where $n$ breaks into smaller pieces of arbitrary size, such as factorizations.

A **loop invariant** is induction over iterations. To prove a loop correct, show three things: the invariant holds before the first iteration (initialization), one iteration preserves it (maintenance), and the loop stops (termination: some nonnegative integer strictly decreases). At exit, the invariant plus the exit condition imply the result. The online softmax kernel and the topological sort both rest on arguments of this shape.

### Relations

A **relation** $R$ on a set $X$ is a set of ordered pairs from $X \times X$; write $x R y$ for $(x, y) \in R$. It is **reflexive** if $x R x$ for every $x$, **symmetric** if $x R y \Rightarrow y R x$, **antisymmetric** if $x R y \land y R x \Rightarrow x = y$, and **transitive** if $x R y \land y R z \Rightarrow x R z$.

An **equivalence relation** is reflexive, symmetric, and transitive. Its **equivalence classes** $[x] = \{y : x R y\}$ partition $X$ into disjoint nonempty blocks, and every partition comes from exactly one equivalence relation, so counting equivalence relations on an $n$-set is counting its partitions (the Bell number $B_n$: $B_1 = 1, B_2 = 2, B_3 = 5$). A **partial order** is reflexive, antisymmetric, and transitive, like $\le$ on numbers, $\subseteq$ on sets, or "must run before" on the nodes of a computation graph ([Graphs, DAGs, and an iterative topological sort](../../12-discrete-math-2/01-graphs-dags-and-topological-sort.md)).

### Functions and bijections

A function $f: X \to Y$ is **injective** (one-to-one) if $f(x) = f(x') \Rightarrow x = x'$, **surjective** (onto) if every $y \in Y$ equals some $f(x)$, and **bijective** if both; a bijection has an inverse $f^{-1}$. Between finite sets of equal size, injective and surjective imply each other. The number of functions from a $k$-set to an $n$-set is $n^k$, the injective ones number $\frac{n!}{(n-k)!}$, and the bijections of an $n$-set to itself number $n!$. To prove $f$ injective, take $x \ne x'$ and show $f(x) \ne f(x')$, often by cases on where $x$ and $x'$ lie.

## A count and a proof, worked by hand

These two examples are siblings of q11 and q26, not problems of the set, and they show the level of detail the answers use.

**Count.** SmolLM2-135M's attention uses grouped-query attention: width $d = 576$, 9 query heads and 3 key/value heads of size $d_h = 64$, no biases. How many parameters do its four projections hold?

The query projection maps width 576 to $9 \cdot 64 = 576$ outputs: a $576 \times 576$ matrix, $331{,}776$ parameters. The key and value projections each map 576 to $3 \cdot 64 = 192$ outputs: $576 \cdot 192 = 110{,}592$ each. The output projection maps the 576 concatenated head outputs back to 576: $331{,}776$. Total: $2 \cdot 331{,}776 + 2 \cdot 110{,}592 = 884{,}736$. Over 30 layers that is $26{,}542{,}080$, about a fifth of the model.

**Proof.** Claim: $1 + 3 + 5 + \dots + (2n - 1) = n^2$ for every integer $n \ge 1$.

*Method: induction on $n$.* Base case $n = 1$: the left side is $1 = 1^2$. Induction hypothesis: for some $k \ge 1$, $\sum_{i=1}^{k} (2i - 1) = k^2$. Inductive step: $\sum_{i=1}^{k+1} (2i - 1) = k^2 + (2(k+1) - 1)$ by the hypothesis, $= k^2 + 2k + 1 = (k+1)^2$, the claim for $n = k + 1$. By induction the claim holds for every $n \ge 1$.

What makes it a complete proof: the claim and every symbol are stated, the method is named, the base case is explicit, the hypothesis is stated for $n = k$ and used in the step (not the conclusion), and the last line restates the claim. The model proofs for q26 to q30 below have the same shape.

## The problems

The problems follow the same order as the review: sets and logic, counting, proof techniques, induction and loop invariants, relations, and functions and bijections. Answers are exact (for example $\binom{10}{3}$ or $3/4$, not $0.75$). Lettered parts share one answer block. The tag after each problem says what kind of answer is expected.

## Review questions

1. Why start a counting proof by defining the counted objects?

   <details><summary>Answer</summary>

   Without a clear sample space, formulas can count duplicates or solve a different problem.

   </details>

[Contents](README.md) · [Next: Sets, logic, and counting](02-sets-logic-and-counting.md)
