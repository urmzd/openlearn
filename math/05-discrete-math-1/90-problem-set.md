<!-- ol:module S-M05 -->
# Discrete math 1 problem set: sets, counting, proofs, induction, relations, bijections

## Overview

| | |
|---|---|
| **Module** | `S-M05` · solve · none · Pass 2 · 8 to 10 h |
| **You build** | answers in `solve/S-M05.toml` (34 checked by SymPy) and 18 proofs in `solve/S-M05/qN.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M05/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M05/problems.md` and in section 4 |
| **Needs** | high-school algebra. Reading: the [Discrete Math 1 topic](README.md), sections 1 to 6 |
| **Used by** | no call site (a solve set). Do it before `M06.1` (toposort correctness is an induction), `M06.3` (PCG32 is counting and modular arithmetic), `M05.1` (parameter, FLOP, and KV-byte counts), `M05.2` (the GPT-2 byte bijection), `L5.2` (mask algebra), and `L9.2` (the online softmax invariant) |
| **Milestone** | `MS-P2` (the Pass 2 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Hammack, *Book of Proof* (free), ch. 1 to 10 and 12; Velleman, *How to Prove It*, ch. 3 and 6 |

## Key Takeaways

- An attention mask is a set of allowed (query, key) pairs, so combining masks is set intersection and a fully masked row is the negation of "every row has a key" (q4, q5).
- Every size in your system is a count: parameters are sums of matrix shapes, FLOPs are $2mkn$ per matmul, and KV-cache bytes are $2 \cdot L \cdot H_{kv} \cdot d_h \cdot b$ per token (q9 to q15).
- A loop is correct when an invariant holds before it, survives one iteration, and implies the result at exit; the online softmax of `L9.2` is proved this way (q29, q30).
- An equivalence relation is the same thing as "has the same image under some function", which is how Unicode normalization groups strings (q36).
- A bijection is injective plus surjective; GPT-2's byte map is checked by exactly those two halves (q38 to q41).

## How to work this chapter

```bash
ol start S-M05              # writes solve/S-M05.toml and one file per proof
ol check S-M05              # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M05 --regrade    # ask the rubrics again after you change a proof
```

---

## 1. Why now

Pass 1 gave you a running tracer: a byte bigram trained in Python, served from Rust, behind a Go gateway. Pass 2 turns it into a stack you can reason about, and the next build modules are arguments as much as code. `M06.1` must order every node of the autograd graph before backward runs, and the only convincing evidence that it does is an induction over the traversal. `M06.3` builds PCG32, whose state wraps modulo $2^{64}$ and whose output is a counted rotation of bits. `M05.1` turns a `ModelConfig` into parameter, FLOP, and KV-byte counts that decide whether a model fits on your laptop. `L5.2` builds attention masks from boolean algebra, and `L9.2` streams a softmax in one pass under a loop invariant. This set gives you the vocabulary and the proof habits those modules assume: sets and logic, counting, the standard proof methods, induction, relations, and functions.

## 2. Principles

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

### 2.1 Sets

A **set** is an unordered collection of distinct elements, written by listing, $\{1, 2, 3\}$, or by a rule, $\{x \in \mathbb{Z} : x > 0\}$. Order and repetition do not matter: $\{2, 1, 2\} = \{1, 2\}$. Two sets are **equal** when each is a subset of the other, which is how set identities are proved: take an arbitrary element of one side and show it is in the other, then the reverse ("double inclusion"). The **power set** $\mathcal{P}(A)$ is the set of all subsets of $A$; it has $2^{\lvert A \rvert}$ elements, because each element is independently in or out.

A boolean mask is a set in disguise. An attention mask over $T$ positions is a subset of $\{0, \dots, T-1\}^2$, the pairs (query $i$, key $j$) that may interact, stored as a 0/1 matrix. The causal mask is $\{(i, j) : j \le i\}$, a padding mask is $\{(i, j) : j < L\}$, and applying both is their intersection, elementwise AND.

### 2.2 Logic and quantifiers

A **proposition** is a statement that is true or false. Connectives build new ones, defined by truth tables: $P \land Q$ is true when both are, $P \lor Q$ when at least one is, $\lnot P$ flips $P$, and $P \Rightarrow Q$ is false only when $P$ is true and $Q$ false. Two propositions are **logically equivalent** when they agree on every row of the truth table. The **contrapositive** $\lnot Q \Rightarrow \lnot P$ is equivalent to $P \Rightarrow Q$; the **converse** $Q \Rightarrow P$ is not. **De Morgan's laws** move a negation inward and swap the connective: $\lnot (P \lor Q) \iff \lnot P \land \lnot Q$ and $\lnot (P \land Q) \iff \lnot P \lor \lnot Q$. For sets they read $(A \cup B)^c = A^c \cap B^c$.

A **predicate** $P(x)$ becomes a proposition once $x$ is fixed. **Quantifiers** close it over a domain: $\forall x\, P(x)$ ("for every $x$") and $\exists x\, P(x)$ ("there exists $x$"). Negation swaps them: $\lnot \forall x\, P(x) \iff \exists x\, \lnot P(x)$, and $\lnot \exists x\, P(x) \iff \forall x\, \lnot P(x)$. With nested quantifiers, swap each one and negate the innermost predicate.

### 2.3 Counting

Four rules count almost everything in this course.

- **Product rule.** A choice made in $k$ independent steps with $n_1, \dots, n_k$ options has $n_1 n_2 \cdots n_k$ outcomes. Strings of length $k$ over an alphabet of size $n$: $n^k$.
- **Ordered without repetition.** Arranging $k$ of $n$ distinct items in order: $n(n-1)\cdots(n-k+1) = \frac{n!}{(n-k)!}$. All $n$ of them: $n!$.
- **Unordered without repetition.** Choosing a $k$-subset: $\binom{n}{k}$, the ordered count divided by the $k!$ orders of each subset.
- **Unordered with repetition ("stars and bars").** Placing $n$ identical items into $k$ distinct boxes is arranging $n$ stars and $k - 1$ bars in a row: $\binom{n + k - 1}{k - 1}$.

The **sum rule** adds the sizes of disjoint cases. When the cases overlap, **inclusion-exclusion** corrects the double count: $\lvert A \cup B \rvert = \lvert A \rvert + \lvert B \rvert - \lvert A \cap B \rvert$.

Model sizes are counts of this kind. A matrix of shape $m \times n$ holds $mn$ numbers; a layer's parameter count is the sum over its matrices and vectors. A matrix product $C = AB$ with $A$ of shape $m \times k$ and $B$ of shape $k \times n$ computes $mn$ outputs, each a sum of $k$ products, so it takes $mkn$ multiplications and as many additions: $2mkn$ floating-point operations (FLOPs). A forward pass over $N$ parameters costs about $2N$ FLOPs per token, and training about $6N$ (the backward pass costs twice the forward), so training on $D$ tokens costs about $6ND$. The KV cache stores one key vector and one value vector of length $d_h$ per KV head per layer per token, at $b$ bytes per number: $2 L H_{kv} d_h b$ bytes per token.

### 2.4 Proof methods

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

### 2.5 Induction and loop invariants

**Induction** proves $P(n)$ for every integer $n \ge n_0$ in two steps: the **base case** $P(n_0)$, and the **inductive step** "if $P(k)$ then $P(k + 1)$" for every $k \ge n_0$. The assumption $P(k)$ is the **induction hypothesis**, and the step must use it rather than the conclusion. **Strong induction** assumes $P(m)$ for every $m$ with $n_0 \le m < n$ and proves $P(n)$; it suits claims where $n$ breaks into smaller pieces of arbitrary size, such as factorizations.

A **loop invariant** is induction over iterations. To prove a loop correct, show three things: the invariant holds before the first iteration (initialization), one iteration preserves it (maintenance), and the loop stops (termination: some nonnegative integer strictly decreases). At exit, the invariant plus the exit condition imply the result. `L9.2` and `M06.1` both rest on arguments of this shape.

### 2.6 Relations

A **relation** $R$ on a set $X$ is a set of ordered pairs from $X \times X$; write $x R y$ for $(x, y) \in R$. It is **reflexive** if $x R x$ for every $x$, **symmetric** if $x R y \Rightarrow y R x$, **antisymmetric** if $x R y \land y R x \Rightarrow x = y$, and **transitive** if $x R y \land y R z \Rightarrow x R z$.

An **equivalence relation** is reflexive, symmetric, and transitive. Its **equivalence classes** $[x] = \{y : x R y\}$ partition $X$ into disjoint nonempty blocks, and every partition comes from exactly one equivalence relation, so counting equivalence relations on an $n$-set is counting its partitions (the Bell number $B_n$: $B_1 = 1, B_2 = 2, B_3 = 5$). A **partial order** is reflexive, antisymmetric, and transitive, like $\le$ on numbers, $\subseteq$ on sets, or "must run before" on the nodes of a computation graph (`M06.1`).

### 2.7 Functions and bijections

A function $f: X \to Y$ is **injective** (one-to-one) if $f(x) = f(x') \Rightarrow x = x'$, **surjective** (onto) if every $y \in Y$ equals some $f(x)$, and **bijective** if both; a bijection has an inverse $f^{-1}$. Between finite sets of equal size, injective and surjective imply each other. The number of functions from a $k$-set to an $n$-set is $n^k$, the injective ones number $\frac{n!}{(n-k)!}$, and the bijections of an $n$-set to itself number $n!$. To prove $f$ injective, take $x \ne x'$ and show $f(x) \ne f(x')$, often by cases on where $x$ and $x'$ lie.

### 2.8 How your answers are checked

`ol check` parses each answer as ASCII math and compares it with the key in SymPy: an `[expr]` must equal the key as a formula (symbolically, or at 32 random points of its domain), a `[number]` marked exact must be an exact value (`3/4`, not `0.75`), a `[set]` matches element for element in any order, and a `[matrix]` entry by entry. Feedback names what differs, never the expected answer. A proof is graded by you: `ol check` prints each line of its rubric and you answer y or n, and every line must be a yes.

## 3. Worked example by hand

This is a sibling of q11 and q26, not one of the graded problems.

**Count.** SmolLM2-135M's attention uses grouped-query attention: width $d = 576$, 9 query heads and 3 key/value heads of size $d_h = 64$, no biases. How many parameters do its four projections hold?

The query projection maps width 576 to $9 \cdot 64 = 576$ outputs: a $576 \times 576$ matrix, $331{,}776$ parameters. The key and value projections each map 576 to $3 \cdot 64 = 192$ outputs: $576 \cdot 192 = 110{,}592$ each. The output projection maps the 576 concatenated head outputs back to 576: $331{,}776$. Total: $2 \cdot 331{,}776 + 2 \cdot 110{,}592 = 884{,}736$. In `solve/` this would be `answer = "2*576*576 + 2*576*192"`: any expression with the right exact value passes. Over 30 layers that is $26{,}542{,}080$, about a fifth of the model.

**Proof.** Claim: $1 + 3 + 5 + \dots + (2n - 1) = n^2$ for every integer $n \ge 1$.

*Method: induction on $n$.* Base case $n = 1$: the left side is $1 = 1^2$. Induction hypothesis: for some $k \ge 1$, $\sum_{i=1}^{k} (2i - 1) = k^2$. Inductive step: $\sum_{i=1}^{k+1} (2i - 1) = k^2 + (2(k+1) - 1)$ by the hypothesis, $= k^2 + 2k + 1 = (k+1)^2$, the claim for $n = k + 1$. By induction the claim holds for every $n \ge 1$.

Read it against the proof rubric (`course/rubrics/proof.md`): the claim and every symbol are stated, the method is named, the base case is explicit, the hypothesis is stated for $n = k$ and used in the step (not the conclusion), and the last line restates the claim. Your proofs for q26 to q30 should look like this.

## 4. The problem set

Write each answer in `solve/S-M05.toml`; lettered parts are their own tables:

```toml
[q4.a]
answer = "[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]]"
[q9]
answer = "m*n + m"
[q6]
proof = "S-M05/q6.md"
```

ASCII math: `x^2`, `2^32`, `binomial(10, 3)`, `factorial(5)`, `exp(-1)`, sets `{1, 2}`, matrices as lists of rows. The tag after each problem is its answer type.

<!-- ol:problems S-M05 -->

### Sets and logic

**q1.** Let $U = \{1, 2, \dots, 10\}$, $A = \{1, 2, 3, 4, 5, 6\}$ and $B = \{2, 4, 6, 8\}$.
(a) $A \cap B$. (b) $A \setminus B$. (c) The complement of $A \cup B$ in $U$. `[set]`

**q2.** How many subsets of $\{1, 2, 3, 4, 5, 6\}$ contain $1$ but not $2$? `[number]`

**q3.** For propositions $P$ and $Q$: (a) is $P \Rightarrow Q$ logically equivalent to $\lnot Q \Rightarrow \lnot P$? (b) Is $P \Rightarrow Q$ logically equivalent to $Q \Rightarrow P$? `[bool]`

**q4.** An attention mask is a 0/1 matrix $M$ whose row $i$ is a query position and column $j$ a key position; $M_{ij} = 1$ means "query $i$ may attend to key $j$". With $T = 4$ positions, the causal mask allows $j \le i$ and the padding mask allows $j < 3$ (position 3 is padding).
(a) Give the combined mask (causal AND padding) as a $4 \times 4$ matrix. `[matrix]`
(b) How many $(i, j)$ pairs does it allow? `[number]`

**q5.** Which statement is the negation of "for every query row $i$ there is a key $j$ with $M_{ij} = 1$"? `[choice]`
(a) There is a row $i$ such that $M_{ij} = 0$ for every key $j$.
(b) $M_{ij} = 0$ for every row $i$ and every key $j$.
(c) There is a row $i$ and a key $j$ with $M_{ij} = 0$.
(d) For every row $i$ there is a key $j$ with $M_{ij} = 0$.

**q6.** Prove De Morgan's law $(A \cup B)^c = A^c \cap B^c$ for subsets $A, B$ of a universe $U$. `[proof]`

### Counting

**q7.** (a) In how many ways can you choose 3 of 10 attention heads to prune? (b) How many byte strings of length 4 are there? (c) How many entries does a byte bigram table have (ordered pairs of bytes, repeats allowed)? `[number]`

**q8.** In how many ways can 5 identical tokens be placed into 3 distinct buckets, empty buckets allowed? `[number]`

**q9.** A linear layer $y = Wx + b$ maps $\mathbb{R}^n$ to $\mathbb{R}^m$. How many parameters does it have? `[expr in m, n]`

**q10.** A language model has an embedding table with $V$ rows of width $d$ and an untied output head that maps width $d$ to $V$ logits without a bias. How many parameters do the two hold together? `[expr in V, d]`

**q11.** A decoder block of width $d$ has four $d \times d$ attention projections ($Q, K, V, O$) without biases, an MLP $d \to 4d \to d$ (two matrices, no biases), and two RMSNorm gain vectors of length $d$. How many parameters does the block have? `[expr in d]`

**q12.** Counting one multiplication and one addition as two floating-point operations, how many FLOPs does the product of an $m \times k$ matrix and a $k \times n$ matrix take? `[expr in m, k, n]`

**q13.** The training-compute rule of thumb is $C \approx 6ND$ FLOPs for $N$ parameters and $D$ tokens. Give $C$ for $N = 10^7$ and $D = 2 \times 10^8$. `[number]`

**q14.** SmolLM2-135M has 30 layers, 3 key/value heads, and head dimension 64, and caches keys and values in bf16 (2 bytes per number). How many bytes of KV cache does one token take? `[number]`

**q15.** How many bytes does that cache take for a context of 2048 tokens? `[number]`

**q16.** How many integers in $\{1, 2, \dots, 100\}$ are divisible by 3 or by 5? `[number]`

### Proof techniques

**q17.** Prove directly: if $n$ is an odd integer, then $n^2$ is odd. `[proof]`

**q18.** Prove by contrapositive: if $n$ is an integer and $n^2$ is even, then $n$ is even. `[proof]`

**q19.** Prove by contradiction: $\sqrt{2}$ is irrational. `[proof]`

**q20.** Prove that there are infinitely many primes. `[proof]`

**q21.** Prove by cases: $n^2 + n$ is even for every integer $n$. `[proof]`

**q22.** Prove: for real $a, b \ge 0$, $\frac{a + b}{2} \ge \sqrt{ab}$, with equality exactly when $a = b$. `[proof]`

**q23.** Claim: $n^2 - n + 41$ is prime for every integer $n \ge 1$. (a) Is the claim true? `[bool]` (b) Give the smallest $n \ge 1$ for which $n^2 - n + 41$ is not prime. `[number]`

**q24.** Prove (pigeonhole): any function from a set of 257 byte strings to 8-bit hash values maps two different strings to the same value. `[proof]`

**q25.** Prove: the sum of a rational number and an irrational number is irrational. `[proof]`

### Induction and loop invariants

**q26.** Prove by induction: $1 + 2 + \dots + n = \frac{n(n+1)}{2}$ for every integer $n \ge 1$. `[proof]`

**q27.** Prove by induction: $2^n > n^2$ for every integer $n \ge 5$. `[proof]`

**q28.** Prove by strong induction: every integer $n \ge 2$ is a product of one or more primes. `[proof]`

**q29.** Prove that this loop returns $x^n$ for every integer $n \ge 0$, using the invariant $r \cdot b^e = x^n$ (the same squaring trick computes PCG32's jump-ahead):

```
r = 1; b = x; e = n
while e > 0:
    if e is odd: r = r * b
    b = b * b
    e = floor(e / 2)
return r
```

`[proof]`

**q30.** The online softmax (the C kernel of `L9.2`) reads $x_0, x_1, \dots$ once, keeping a running maximum $m$ and a running sum $s$. It starts at $m = -\infty$, $s = 0$ (with $e^{-\infty} = 0$), and for each new $x_i$ sets $m' = \max(m, x_i)$ and $s' = s \cdot e^{m - m'} + e^{x_i - m'}$. Prove that after reading $x_0, \dots, x_{i-1}$ it holds that $m = \max_{j < i} x_j$ and $s = \sum_{j < i} e^{x_j - m}$. `[proof]`

**q31.** Give a closed form for $1^2 + 2^2 + \dots + n^2$. `[expr in n]`

**q32.** Give a closed form for $2^0 + 2^1 + \dots + 2^{n-1}$. `[expr in n]`

**q33.** Run the loop of q30 on $x = (1, 3, 2)$. Give the final $s$ exactly. `[number]`

### Relations

**q34.** Let $R = \{(1,1), (2,2), (3,3), (1,2), (2,1)\}$ on $\{1, 2, 3\}$. Is $R$ (a) reflexive, (b) symmetric, (c) transitive? `[bool]`

**q35.** How many equivalence relations are there on a 4-element set? `[number]`

**q36.** Prove: for any function $f: X \to Y$, the relation $x \sim y \iff f(x) = f(y)$ is an equivalence relation on $X$. (Unicode normalization in `L1.1` is this relation with $f = \mathrm{NFC}$.) `[proof]`

**q37.** Prove: divisibility ($a \mid b$ when $b = ka$ for some integer $k$) is a partial order on the positive integers. `[proof]`

### Functions and bijections

**q38.** Let $f: \mathbb{Z} \to \mathbb{Z}$, $f(n) = 2n + 1$. Is $f$ (a) injective, (b) surjective? `[bool]`

**q39.** (a) How many bijections are there from a 5-element set to itself? (b) How many injective functions are there from a 3-element set to a 5-element set? `[number]`

**q40.** Prove: if $f: X \to Y$ and $g: Y \to Z$ are bijections, then $g \circ f$ is a bijection. `[proof]`

**q41.** GPT-2's byte-to-character map (`M05.2`) keeps the 188 printable bytes $P = \{33, \dots, 126\} \cup \{161, \dots, 172\} \cup \{174, \dots, 255\}$ as themselves ($f(b) = b$) and sends the other 68 bytes, in increasing order, to $256, 257, \dots, 323$. Prove that $f: \{0, \dots, 255\} \to \mathbb{N}$ is injective, so it is a bijection onto its image. `[proof]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Combining masks with OR, or forgetting the padding key | a pad position gets attention weight | q4 (canary: the causal-only mask) |
| Negating only the inner predicate of a nested quantifier | you test "some entry is 0" instead of "some row is all 0" | q5 (canary: choice c) |
| Counting multiply-adds instead of FLOPs | FLOP and roofline numbers off by 2 | q12 (canary m*k*n) |
| Using $2ND$ (forward only) for training compute | a training budget 3 times too small | q13 (canary 2*10^15) |
| Caching keys only, or sizing the cache by query heads | KV memory off by 2 or by the GQA group size | q14 (canaries 11520 and 69120) |
| Adding overlapping cases twice | inclusion-exclusion overcount | q16 (canary 53) |
| Checking many cases instead of proving | a "for every $n$" claim believed from $n = 1$ to $40$ | q23 (canary 40) |
| Using the conclusion inside the inductive step | a circular proof that the rubric rejects | q26 to q30 rubric line 2 or 3 |
| Forgetting the running sum must be rescaled when the max changes | online softmax overflows or is wrong after a new max | q30 rubric, q33 (canary without the shift) |
| Confusing all functions with injective ones | $n^k$ where $\frac{n!}{(n-k)!}$ was asked | q39 (canaries 3125 and 125) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `M06.1` | toposort over the autograd graph; its correctness proof is an induction over the traversal (S-M06a q20) |
| Forward | `M06.3` | PCG32 and SplitMix64: counting bits, rotations, and arithmetic modulo $2^{64}$ |
| Forward | `M05.1` | `param_count`, `flops_per_token`, `kv_bytes_per_token`, `memory_plan` are q9 to q15 as code |
| Forward | `M05.2` | `bytes_to_unicode` is the bijection of q41; its tests check injective and surjective |
| Forward | `L5.2` | causal and padding masks combined as sets (q4, q5) |
| Forward | `L9.2` | the online softmax kernel in C; q30 is its correctness argument |
| Forward | `L1.1` | Unicode normalization as an equivalence relation (q36) |
| Forward | `S-M06a` | graphs, modular arithmetic, and hashing build on these counting and proof tools |
