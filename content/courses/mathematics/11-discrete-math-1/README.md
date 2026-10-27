# Discrete Math 1

Discrete mathematics is the mathematics of distinct, separated structures, as opposed to the continuous structures of calculus. Its central activity is proof: constructing airtight logical arguments. Every concept in it appears directly in computer science. Sets become data structures, logic becomes boolean algebra and type systems, induction becomes recursive correctness arguments, relations become database schemas, and functions become the foundation of programming. Counting, meanwhile, is how you size anything: a model's parameters, the FLOPs of a forward pass, the bytes of a cache.

This chapter starts where the rest of the book needs it most. Its problem set builds the vocabulary and the proof habits that later chapters assume: sets and logic, counting, the standard proof methods, induction and loop invariants, relations, and functions. Its two build sections then put two of those ideas to work. The first counts a transformer's parameters, FLOPs, KV-cache bytes, and training memory exactly, from a handful of config integers. The second uses injective, surjective, and bijective to derive the byte-to-character map that every GPT-2 style tokenizer depends on. The survey after the section list covers the wider territory of a first discrete math course, following Hammack's *Book of Proof*, with a worked example for each topic.

> **Learning objectives**
> - Read and write rigorous proofs: direct, contrapositive, contradiction, cases, and induction, including loop invariants.
> - Use the language of sets, logic, and quantifiers, including negating nested quantifiers.
> - Count with the product rule, permutations, combinations, and stars and bars, and apply them to parameter, FLOP, and KV-cache counts.
> - Classify relations, and recognize equivalence relations as partitions.
> - Prove functions injective, surjective, or bijective, and explain why only bijections have inverses.

## Before you begin

The chapter needs no earlier mathematics beyond high-school algebra. For the counting section, powers of ten and logarithms from [Exponents, logarithms, and units of information](../06-precalculus/01-exponents-logs-and-units-of-information.md) help with reading counts like $6.7 \times 10^9$. The problem set comes last in the list below by file order, but it is designed to be worked first.

## In this chapter

1. [Counting: parameters, FLOPs, KV bytes, and memory plans](01-counting-params-flops-kv-bytes.md): counts a decoder's parameters per component so they match Hugging Face exactly for dense, GQA, MoE, and MLA models, and derives KV-cache bytes, FLOPs per token, and a training memory plan, all as exact integers.
2. [Injective, surjective, bijective: the GPT-2 byte map](02-injective-surjective-bijective-gpt2-byte-map.md): builds the three definitions as checks on a dict, and derives GPT-2's bijection from the 256 bytes to 256 visible characters that byte-level BPE vocabularies are written in.
3. [Discrete math 1 problem set](03-problem-set/README.md): problems with worked answers and model proofs on sets and logic, counting, proof techniques, induction and loop invariants, relations, and functions and bijections. Work it before the two sections above.

## A survey of the subject

The sections above focus on the parts of discrete math that code depends on. The rest of this opening surveys a first course in the subject, following Hammack's *Book of Proof*. Each topic gives the key definitions and theorems, worked examples, and the textbook exercises to practice on.

### Sets

Book of Proof chapter 1, sections 1.1 to 1.8. A **set** is an unordered collection of distinct elements, described by listing, $\{1,2,3\}$, or in set-builder notation, $\{x \in \mathbb{Z} : x > 0\}$. $A$ is a **subset** of $B$ if every element of $A$ is in $B$. The **power set** $\mathcal{P}(A)$ is the set of all subsets of $A$. The **union** is $A \cup B = \{x : x \in A \text{ or } x \in B\}$, the **intersection** is $A \cap B = \{x : x \in A \text{ and } x \in B\}$, and the **complement** is $A^c = \{x \in U : x \notin A\}$. The **Cartesian product** is $A \times B = \{(a,b) : a \in A, b \in B\}$, with $\lvert A \times B \rvert = \lvert A \rvert \cdot \lvert B \rvert$.

Three results recur. **De Morgan's laws**, $(A \cup B)^c = A^c \cap B^c$ and $(A \cap B)^c = A^c \cup B^c$, say that "not (this or that)" is "not this and not that": negation swaps union and intersection. **Inclusion-exclusion** for two sets, $\lvert A \cup B \rvert = \lvert A \rvert + \lvert B \rvert - \lvert A \cap B \rvert$, corrects the double count of the overlap. And $\lvert \mathcal{P}(A) \rvert = 2^{\lvert A \rvert}$, because each element is independently in or out.

For example, prove $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$. Let $x \in A \cap (B \cup C)$. Then $x \in A$ and $x \in B \cup C$. If $x \in B$, then $x \in A \cap B$, so $x$ is in the right side. If $x \in C$, then $x \in A \cap C$, so again $x$ is in the right side. The reverse inclusion is similar.

Practice: Hammack chapter 1 exercises 1 to 8, 13 to 18, and 25 to 30; for a challenge, 31 to 36.

### Logic

Book of Proof chapter 2, sections 2.1 to 2.10. A **proposition** is a statement that is either true or false. $P \land Q$ is true only when both are; $P \lor Q$ when at least one is; $\lnot P$ reverses the truth value. The **conditional** $P \Rightarrow Q$ is false only when $P$ is true and $Q$ is false, and the **biconditional** $P \iff Q$ is true when $P$ and $Q$ have the same truth value. A **predicate** is a statement with a variable, such as $P(x)$ = "$x$ is even". The **universal quantifier** $\forall x\, P(x)$ asserts $P$ for every $x$ in the domain; the **existential quantifier** $\exists x\, P(x)$ asserts it for at least one.

$P \Rightarrow Q$ is equivalent to $\lnot P \lor Q$: "if $P$ then $Q$" fails only when $P$ is true and $Q$ is false. It is also equivalent to its **contrapositive** $\lnot Q \Rightarrow \lnot P$: "if it rained then the ground is wet" is the same as "if the ground is dry then it didn't rain." And negation swaps quantifiers, $\lnot \forall x\, P(x) \iff \exists x\, \lnot P(x)$ and $\lnot \exists x\, P(x) \iff \forall x\, \lnot P(x)$, just as De Morgan's laws swap AND and OR.

For example, negate "for every $\epsilon > 0$ there exists $\delta > 0$ such that $|x - a| < \delta$ implies $|f(x) - L| < \epsilon$." The negation is "there exists $\epsilon > 0$ such that for all $\delta > 0$ there exists $x$ with $|x - a| < \delta$ and $|f(x) - L| \ge \epsilon$." Each quantifier flips, and the implication becomes a conjunction with the consequent negated.

Practice: Hammack chapter 2 exercises 1 to 10 and 15 to 22; for a challenge, 23 to 28.

### Counting

Book of Proof chapter 3, sections 3.1 to 3.5. The **multiplication principle**: if task 1 has $m$ outcomes and task 2 has $n$, the sequence has $mn$. The **addition principle**: if the tasks are mutually exclusive, the total is $m + n$. A **permutation** is an ordered arrangement, $P(n,k) = n!/(n-k)!$; a **combination** is an unordered selection, $\binom{n}{k} = n!/(k!(n-k)!)$, read "$n$ choose $k$".

The **binomial theorem** is $(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k$: expanding $(x+y)^n$, each term chooses $x$ from some factors and $y$ from the rest, and there are $\binom{n}{k}$ ways to choose the $k$ factors for $y$. **Pascal's identity**, $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$, says that to choose $k$ items from $n$, you either include item $n$ (then choose $k-1$ from the remaining $n-1$) or exclude it (choose $k$ from $n-1$). **Stars and bars**: the number of ways to distribute $n$ identical objects into $k$ distinct bins is $\binom{n+k-1}{k-1}$, the arrangements of $n$ stars and $k-1$ dividers in a row.

Two examples. How many 5-card poker hands contain exactly 2 aces? Choose 2 aces from 4, $\binom{4}{2} = 6$, and the remaining 3 cards from the 48 non-aces, $\binom{48}{3} = 17296$, for $6 \cdot 17296 = 103776$. How many bit strings of length 8 have exactly 3 ones? Choose the positions of the ones: $\binom{8}{3} = 56$.

Practice: Hammack chapter 3 exercises 1 to 12 and 17 to 24; for a challenge, 25 to 30.

### Direct proof

Book of Proof chapter 4, sections 4.1 to 4.5. A **direct proof** of $P \Rightarrow Q$ assumes $P$ and uses definitions, theorems, and logical reasoning to conclude $Q$. The technique is always the same: assume the hypothesis, unpack the definitions, manipulate using known results, and arrive at the conclusion.

Prove that if $n$ is odd, then $n^2$ is odd. Assume $n$ is odd, so $n = 2k + 1$ for some integer $k$. Then $n^2 = (2k+1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1$. Since $2k^2 + 2k$ is an integer, $n^2$ has the form $2m + 1$, so it is odd.

Prove that if $a \mid b$ and $b \mid c$, then $a \mid c$. Since $a \mid b$, $b = ak$ for some integer $k$; since $b \mid c$, $c = bj$ for some integer $j$. Then $c = (ak)j = a(kj)$, and since $kj$ is an integer, $a \mid c$.

Practice: Hammack chapter 4 exercises 1 to 8, 13 to 18, and 23 to 28; for a challenge, 29 to 34.

### Contrapositive proof

Book of Proof chapter 5, sections 5.1 to 5.3. A **contrapositive proof** of $P \Rightarrow Q$ proves the logically equivalent $\lnot Q \Rightarrow \lnot P$ instead. Use it when the negation of $Q$ gives you more to work with than assuming $P$ directly.

Prove that if $n^2$ is even, then $n$ is even. The contrapositive is: if $n$ is odd, then $n^2$ is odd. Assume $n = 2k + 1$; then $n^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1$, which is odd. This avoids the difficulty of working directly from "$n^2$ is even."

Prove that if $5x + 3$ is odd, then $x$ is even. The contrapositive: if $x$ is odd, $x = 2k + 1$, then $5x + 3 = 10k + 5 + 3 = 10k + 8 = 2(5k + 4)$, which is even.

Practice: Hammack chapter 5 exercises 1 to 8 and 11 to 16; for a challenge, 17 to 20.

### Proof by contradiction

Book of Proof chapter 6, sections 6.1 to 6.4. A **proof by contradiction** assumes the statement is false and derives a logical contradiction: reason until you reach an impossibility (a number that is both even and odd, or a rational that is irrational), and conclude that the original statement must be true.

Prove that $\sqrt2$ is irrational. Assume $\sqrt2 = a/b$ in lowest terms, $\gcd(a, b) = 1$. Then $2 = a^2/b^2$, so $a^2 = 2b^2$. Thus $a^2$ is even, so $a$ is even (by the contrapositive proof above). Write $a = 2c$. Then $4c^2 = 2b^2$, so $b^2 = 2c^2$, and $b$ is even. But then both $a$ and $b$ are even, contradicting $\gcd(a, b) = 1$.

Prove that there are infinitely many primes. Assume there are finitely many, $p_1, p_2, \dots, p_n$, and consider $N = p_1 p_2 \cdots p_n + 1$. $N$ is not divisible by any $p_i$ (the remainder is 1), so $N$ is either prime itself or has a prime factor not in the list. Either way, the list was not complete: a contradiction.

Practice: Hammack chapter 6 exercises 1 to 8 and 13 to 18; for a challenge, 19 to 24.

### Non-conditional proofs

Book of Proof chapter 7, sections 7.1 to 7.4. Some statements are not a single implication. An **if-and-only-if proof** proves both directions, $P \Rightarrow Q$ and $Q \Rightarrow P$. An **existence proof** shows that an object with the desired property exists, constructively or not. A **uniqueness proof** shows at most one object has the property: assume two, and show they are equal. A **proof by cases** breaks the claim into exhaustive cases and proves each separately.

Prove that $n$ is odd if and only if $n^2$ is odd. ($\Rightarrow$) If $n = 2k + 1$, then $n^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1$ is odd. ($\Leftarrow$) By contrapositive: if $n = 2k$, then $n^2 = 4k^2 = 2(2k^2)$ is even. Both directions are established.

Practice: Hammack chapter 7 exercises 1 to 8, 13 to 18, and 25 to 30; for a challenge, 31 to 36.

### Mathematical induction

Book of Proof chapter 10, sections 10.1 to 10.4. **Weak induction** proves $P(\text{base})$, then assumes $P(k)$ and proves $P(k+1)$, and concludes $P(n)$ for all $n \ge \text{base}$. **Strong induction** assumes $P(j)$ for all $\text{base} \le j \le k$ and proves $P(k+1)$. **Structural induction** inducts on recursively defined structures such as trees, lists, and formulas. Underneath all three is the **well-ordering principle**: every nonempty subset of the natural numbers has a least element. It is logically equivalent to induction: if induction failed, there would be a smallest counterexample, contradicting well-ordering.

Prove $1 + 2 + \dots + n = n(n+1)/2$. Base case $n = 1$: the left side is 1 and the right side is $1 \cdot 2/2 = 1$. Inductive step: assume $1 + 2 + \dots + k = k(k+1)/2$. Then $1 + 2 + \dots + k + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2$, the formula with $n = k + 1$.

Prove $2^n > n$ for all $n \ge 1$, in strong induction style. Base: $2^1 = 2 > 1$. Assume $2^j > j$ for all $1 \le j \le k$. Then $2^{k+1} = 2 \cdot 2^k > 2k \ge k + 1$, since $k \ge 1$.

By structural induction, every binary tree with $n$ internal nodes has $n + 1$ leaves. Base: a tree with 0 internal nodes has 1 leaf. Inductive step: a tree with a root has a left subtree ($n_L$ internal nodes, $n_L + 1$ leaves) and a right subtree ($n_R$ internal, $n_R + 1$ leaves). Its internal nodes total $n_L + n_R + 1$, and its leaves $(n_L + 1) + (n_R + 1) = n_L + n_R + 2$, which is the internal count plus 1.

Practice: Hammack chapter 10 exercises 1 to 10, 15 to 22, and 27 to 32; for a challenge, 33 to 38.

### Relations

Book of Proof chapter 11, sections 11.1 to 11.6. A **relation** on $A$ is a subset $R$ of $A \times A$; write $a\,R\,b$ for $(a,b) \in R$. It is **reflexive** if $a\,R\,a$ for all $a$, **symmetric** if $a\,R\,b$ implies $b\,R\,a$, and **transitive** if $a\,R\,b$ and $b\,R\,c$ imply $a\,R\,c$. An **equivalence relation** has all three properties, and the **equivalence class** of $a$ is $[a] = \{x \in A : x\,R\,a\}$. A **partition** is a collection of nonempty, pairwise disjoint subsets whose union is $A$. A **partial order** is reflexive, **antisymmetric** ($a\,R\,b$ and $b\,R\,a$ imply $a = b$), and transitive.

The **fundamental theorem of equivalence relations** says the equivalence classes of an equivalence relation on $A$ form a partition of $A$, and conversely every partition defines an equivalence relation. They are the same concept viewed from two angles: "being equivalent" means "being in the same part."

For example, on $\mathbb{Z}$ define $a \sim b$ if and only if $3 \mid (a - b)$, congruence mod 3. It is reflexive, since $3 \mid (a - a) = 0$; symmetric, since $3 \mid (a - b)$ implies $3 \mid -(a - b) = (b - a)$; and transitive, since $3 \mid (a - b)$ and $3 \mid (b - c)$ imply $3 \mid (a - b) + (b - c) = (a - c)$. Its classes are $[0] = \{\dots, -6, -3, 0, 3, 6, \dots\}$, $[1] = \{\dots, -5, -2, 1, 4, 7, \dots\}$, and $[2] = \{\dots, -4, -1, 2, 5, 8, \dots\}$, which partition $\mathbb{Z}$.

Practice: Hammack chapter 11 exercises 1 to 10 and 15 to 22; for a challenge, 23 to 28.

### Functions

Book of Proof chapter 12, sections 12.1 to 12.6. A **function** from $A$ to $B$ is a relation in which every element of $A$ is related to exactly one element of $B$. It is an **injection** (one-to-one) if $f(a) = f(b)$ implies $a = b$, a **surjection** (onto) if every $b \in B$ has some $a$ with $f(a) = b$, and a **bijection** if both, a one-to-one correspondence. **Composition** is $(g \circ f)(x) = g(f(x))$, and the **inverse** $f^{-1}$ exists if and only if $f$ is a bijection, with $f^{-1}(f(a)) = a$.

Four theorems organize the topic. Composition preserves the properties: if $f$ and $g$ are injective, so is $g \circ f$, and likewise for surjective; composing two "no-collision" maps gives a "no-collision" map, and composing two "everything is hit" maps gives an "everything is hit" map. $f$ has an inverse if and only if it is a bijection: to undo a function, every output must come from exactly one input. **Cantor's theorem**, $\lvert A \rvert < \lvert \mathcal{P}(A) \rvert$ for any set $A$ (there is no surjection from $A$ onto its power set), says there are always more subsets than elements, which is the source of different sizes of infinity. **Schröder-Bernstein**: if there are injections $A \to B$ and $B \to A$, there is a bijection between them; if each set fits inside the other, they are the same size.

Two examples. $f: \mathbb{Z} \to \mathbb{Z}$, $f(n) = 2n + 1$, is injective but not surjective. If $2a + 1 = 2b + 1$ then $a = b$; but $2n + 1 = 4$ requires $n = 3/2 \notin \mathbb{Z}$, so 4 has no preimage. And $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^3$, is a bijection: if $a^3 = b^3$ then $a = b$ (the cube root is well defined on $\mathbb{R}$), and for any $y$, $x = y^{1/3}$ satisfies $f(x) = y$. Its inverse is $f^{-1}(y) = y^{1/3}$.

Practice: Hammack chapter 12 exercises 1 to 10, 15 to 22, and 27 to 34; for a challenge, 35 to 40.

### Choosing a technique

Much of the skill in the subject is recognizing which method fits a claim. This table pairs each with when to reach for it:

| Technique | When to use | Key formula or idea |
|---|---|---|
| Direct proof | prove $P \Rightarrow Q$ when definitions lead forward | assume $P$, derive $Q$ |
| Contrapositive | $P \Rightarrow Q$ is hard; $\lnot Q \Rightarrow \lnot P$ is easier | negate both, reverse the direction |
| Contradiction | existence, uniqueness, or irrationality proofs | assume the negation, derive an impossibility |
| Weak induction | prove $P(n)$ for all $n \ge$ base | base case, plus $P(k) \Rightarrow P(k+1)$ |
| Strong induction | $P(k+1)$ depends on several prior cases | base case, plus (all $P(j)$ for $j \le k$) $\Rightarrow P(k+1)$ |
| Structural induction | properties of recursive structures (trees, lists) | base structure, plus induction on construction |
| Proof by cases | the domain splits naturally | prove each case; make sure the cases are exhaustive |
| Counting (multiplication) | sequential independent choices | multiply the options at each stage |
| Counting (combinations) | unordered selections | $\binom{n}{k} = n!/(k!(n-k)!)$ |
| Bijective proof | show two counts are equal | find a one-to-one correspondence |

## Key terms

- **Power set**: the set $\mathcal{P}(A)$ of all subsets of $A$; it has $2^{\lvert A \rvert}$ elements because each element is independently in or out of a subset.
- **Cartesian product**: $A \times B = \{(a, b) : a \in A, b \in B\}$, the set of all ordered pairs, with $\lvert A \times B \rvert = \lvert A \rvert \cdot \lvert B \rvert$; relations and functions are subsets of it.
- **De Morgan's laws**: $(A \cup B)^c = A^c \cap B^c$ and $(A \cap B)^c = A^c \cup B^c$, and likewise $\lnot(P \lor Q) \iff \lnot P \land \lnot Q$; negation swaps union with intersection and OR with AND.
- **Universal quantifier / existential quantifier**: $\forall x\, P(x)$ asserts $P$ for every $x$ in the domain, and $\exists x\, P(x)$ for at least one; negating a statement flips each quantifier, $\lnot \forall x\, P(x) \iff \exists x\, \lnot P(x)$.
- **Multiplication principle**: if one task has $m$ outcomes and a second has $n$ for each of them, the sequence has $mn$ outcomes; with the addition principle for mutually exclusive cases, it is the basis of all counting.
- **Binomial theorem**: $(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k$, because each term picks $y$ from $k$ of the $n$ factors, which can be done in $\binom{n}{k}$ ways.
- **Stars and bars**: the number of ways to distribute $n$ identical objects into $k$ distinct bins is $\binom{n+k-1}{k-1}$, counted as arrangements of $n$ stars and $k - 1$ dividers in a row.
- **Direct proof**: a proof of $P \Rightarrow Q$ that assumes $P$, unpacks the definitions, and reasons forward with known results until it reaches $Q$.
- **Proof by contradiction**: a proof that assumes the statement is false and derives an impossibility, so the statement must be true; it is the standard route to results such as the irrationality of $\sqrt 2$.
- **Strong induction**: induction in which the step assumes $P(j)$ for every $j$ from the base case up to $k$, not just $P(k)$, before proving $P(k+1)$; it fits claims where a case depends on several earlier ones.
- **Structural induction**: induction over a recursively defined structure such as a list, tree, or formula: prove the property for the base structures, then show that each construction rule preserves it.
- **Well-ordering principle**: every nonempty set of natural numbers has a least element; it is logically equivalent to induction, since a failed induction would have a smallest counterexample.
- **Equivalence class**: for an equivalence relation $R$ on $A$, the set $[a] = \{x \in A : x\,R\,a\}$ of everything equivalent to $a$; the classes partition $A$, and every partition defines an equivalence relation.
- **Partial order**: a relation that is reflexive, antisymmetric ($a\,R\,b$ and $b\,R\,a$ imply $a = b$), and transitive, such as $\le$ or "is a subset of"; some pairs may be incomparable, which is why dependency graphs need a topological sort.
- **Cantor's theorem**: $\lvert A \rvert < \lvert \mathcal{P}(A) \rvert$ for every set $A$, because no function from $A$ onto its power set exists; it shows there are different sizes of infinity.
- **Schröder-Bernstein theorem**: if there are injections $A \to B$ and $B \to A$, there is a bijection between $A$ and $B$; if each set fits inside the other, they have the same size.

## Where this leads

These ideas surface throughout computing. Propositional logic is boolean algebra, circuit design, and SAT solving; quantifiers and predicates are the WHERE clauses of database queries. Induction proves recursive algorithms correct, and loop invariants prove iterative ones, as in [Graphs, DAGs, and an iterative topological sort](../12-discrete-math-2/01-graphs-dags-and-topological-sort.md). Sets underlie data structures such as hash sets and tree sets, and type systems. Equivalence relations underlie union-find and database normalization, and partial orders underlie topological sort and dependency resolution. Bijections are the basis of encoding and decoding, hashing, and encryption, and [Modular arithmetic, hashing, and PCG32](../12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) continues in that direction. Counting and combinatorics drive algorithm analysis, from average cases and expected values in [Probability and statistics](../13-probability-statistics/) to the binomial sums in divide-and-conquer recurrences.

### Further reading

- Richard Hammack, [*Book of Proof*](https://people.vcu.edu/~rhammack/BookOfProof/) (CC BY-ND), the textbook this chapter follows. Write your proofs by hand in complete sentences and compare them with Hammack's solutions for style and rigor.
