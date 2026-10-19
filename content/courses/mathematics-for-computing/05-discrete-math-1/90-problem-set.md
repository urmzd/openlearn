# Discrete math 1 problem set: sets, counting, proofs, induction, relations, bijections

The chapters ahead are arguments as much as code. [Graphs, DAGs, and an iterative topological sort](../06-discrete-math-2/01-graphs-dags-and-topological-sort.md) must order every node of the autograd graph before backward runs, and the only convincing evidence that it does is an induction over the traversal. [Modular arithmetic, hashing, and PCG32](../06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) builds a generator whose state wraps modulo $2^{64}$ and whose output is a counted rotation of bits. [Counting: parameters, FLOPs, KV bytes, and memory plans](01-counting-params-flops-kv-bytes.md) turns a `ModelConfig` into counts that decide whether a model fits on a laptop, and [Injective, surjective, bijective: the GPT-2 byte map](02-injective-surjective-bijective-gpt2-byte-map.md) proves a byte table invertible. In *Build Your Own LLM System*, [Masks: causal, padding, sliding window, additive](../../build-an-llm/p05-transformer-2017/02-masks.md) builds attention masks from boolean algebra, and [Softmax in C: three-pass and online two-pass](../../build-an-llm/p09-kernels/05-softmax-three-pass-and-online.md) streams a softmax in one pass under a loop invariant.

This set gives the vocabulary and the proof habits those sections assume: sets and logic, counting, the standard proof methods, induction, relations, and functions. It needs only high-school algebra and the overview in the [chapter opening](README.md), and it is best worked before the two sections of this chapter.

The set has 23 problems with exact answers (34 parts) and 18 proofs, whose model proofs are also in [`course/solutions/solve/S-M05/`](../../../../course/solutions/solve/S-M05/). Every problem is followed by a collapsed worked answer with the final result or a complete model proof, and the common wrong answers and why they are wrong. Try each problem before opening its answer.

> **Learning objectives**
> - Manipulate sets and quantified statements, including negating nested quantifiers, and read an attention mask as a set.
> - Count with the product, permutation, combination, and stars-and-bars rules, and apply them to parameters, FLOPs, and KV-cache bytes.
> - Write complete direct, contrapositive, contradiction, case, and pigeonhole proofs.
> - Prove statements by induction and prove loops correct with invariants.
> - Classify relations, and prove functions injective, surjective, or bijective.

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

An **equivalence relation** is reflexive, symmetric, and transitive. Its **equivalence classes** $[x] = \{y : x R y\}$ partition $X$ into disjoint nonempty blocks, and every partition comes from exactly one equivalence relation, so counting equivalence relations on an $n$-set is counting its partitions (the Bell number $B_n$: $B_1 = 1, B_2 = 2, B_3 = 5$). A **partial order** is reflexive, antisymmetric, and transitive, like $\le$ on numbers, $\subseteq$ on sets, or "must run before" on the nodes of a computation graph ([Graphs, DAGs, and an iterative topological sort](../06-discrete-math-2/01-graphs-dags-and-topological-sort.md)).

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

### Sets and logic

**q1.** Let $U = \{1, 2, \dots, 10\}$, $A = \{1, 2, 3, 4, 5, 6\}$ and $B = \{2, 4, 6, 8\}$.
(a) $A \cap B$. (b) $A \setminus B$. (c) The complement of $A \cup B$ in $U$. `[set]`

<details><summary>Answer</summary>

**(a) $\{2, 4, 6\}$, (b) $\{1, 3, 5\}$, (c) $\{7, 9, 10\}$.** (a) The elements in both: $2, 4, 6$. (b) The elements of $A$ that are not in $B$: $1, 3, 5$. (c) $A \cup B = \{1, 2, 3, 4, 5, 6, 8\}$, and the rest of $U$ is $\{7, 9, 10\}$. Common mistakes: the union $\{1, 2, 3, 4, 5, 6, 8\}$ for (a); $B \setminus A = \{8\}$ for (b), since difference is not symmetric; and $\{7, 8, 9, 10\}$ for (c), the complement of $A$ alone. These are the operations of the mask algebra in [Masks: causal, padding, sliding window, additive](../../build-an-llm/p05-transformer-2017/02-masks.md) in *Build Your Own LLM System*.

</details>

**q2.** How many subsets of $\{1, 2, 3, 4, 5, 6\}$ contain $1$ but not $2$? `[number]`

<details><summary>Answer</summary>

**$16$.** The product rule: 1 must be in, 2 must be out, and each of the other 4 elements is independently in or out, so $1 \cdot 1 \cdot 2^4 = 16$. A common mistake is 32, which fixes only one of the two elements.

</details>

**q3.** For propositions $P$ and $Q$: (a) is $P \Rightarrow Q$ logically equivalent to $\lnot Q \Rightarrow \lnot P$? (b) Is $P \Rightarrow Q$ logically equivalent to $Q \Rightarrow P$? `[bool]`

<details><summary>Answer</summary>

**(a) True, (b) false.** (a) The contrapositive has the same truth table as $P \Rightarrow Q$: both are false only when $P$ is true and $Q$ is false. (b) The converse is not equivalent: with $P$ false and $Q$ true, $P \Rightarrow Q$ is true but $Q \Rightarrow P$ is false. The contrapositive is the move behind q18; the converse is the classic slip.

</details>

**q4.** An attention mask is a 0/1 matrix $M$ whose row $i$ is a query position and column $j$ a key position; $M_{ij} = 1$ means "query $i$ may attend to key $j$". With $T = 4$ positions, the causal mask allows $j \le i$ and the padding mask allows $j < 3$ (position 3 is padding).
(a) Give the combined mask (causal AND padding) as a $4 \times 4$ matrix. `[matrix]`
(b) How many $(i, j)$ pairs does it allow? `[number]`

<details><summary>Answer</summary>

**(a) $\begin{pmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 0 \end{pmatrix}$, (b) $9$.** (a) Entry $(i, j)$ is 1 when $j \le i$ (causal) and $j < 3$ (padding): the lower triangle with column 3 cleared. (b) Rows allow $1 + 2 + 3 + 3 = 9$ pairs. Common mistakes: the causal mask alone, which keeps $M_{33} = 1$ (10 pairs) and lets the pad key receive attention; and the upper triangle $j \ge i$, which lets a query look at the future.

</details>

**q5.** Which statement is the negation of "for every query row $i$ there is a key $j$ with $M_{ij} = 1$"? `[choice]`
(a) There is a row $i$ such that $M_{ij} = 0$ for every key $j$.
(b) $M_{ij} = 0$ for every row $i$ and every key $j$.
(c) There is a row $i$ and a key $j$ with $M_{ij} = 0$.
(d) For every row $i$ there is a key $j$ with $M_{ij} = 0$.

<details><summary>Answer</summary>

**(a).** The statement is $\forall i\ \exists j\ M_{ij} = 1$. Negating swaps each quantifier and negates the predicate: $\exists i\ \forall j\ M_{ij} = 0$, a row with no allowed key. That fully masked row is the NaN case of [Masks: causal, padding, sliding window, additive](../../build-an-llm/p05-transformer-2017/02-masks.md) and of [Stable numerics: logsumexp, softmax, and compensated sums](../09-numerical-methods-and-floating-point/02-stable-numerics.md). A common mistake is (c), which negates only the inner predicate and describes any matrix with a single 0; (b) is much stronger than the negation.

</details>

**q6.** Prove De Morgan's law $(A \cup B)^c = A^c \cap B^c$ for subsets $A, B$ of a universe $U$. `[proof]`

<details><summary>Answer</summary>

Claim: for subsets $A, B$ of a universe $U$, $(A \cup B)^c = A^c \cap B^c$, where $X^c = U \setminus X$.

Method: show each side is a subset of the other (double inclusion).

($\subseteq$) Let $x \in (A \cup B)^c$. Then $x \in U$ and $x \notin A \cup B$. By the definition of union, $x \in A \cup B$ means "$x \in A$ or $x \in B$", so $x \notin A \cup B$ means "not ($x \in A$ or $x \in B$)", which by De Morgan's law for propositions is "$x \notin A$ and $x \notin B$". Hence $x \in A^c$ and $x \in B^c$, so $x \in A^c \cap B^c$.

($\supseteq$) Let $x \in A^c \cap B^c$. Then $x \in U$, $x \notin A$, and $x \notin B$. So "$x \in A$ or $x \in B$" is false, that is $x \notin A \cup B$, and $x \in (A \cup B)^c$.

Each set contains the other, so $(A \cup B)^c = A^c \cap B^c$.

Model proof also in [`q6.md`](../../../../course/solutions/solve/S-M05/q6.md).

</details>

### Counting

**q7.** (a) In how many ways can you choose 3 of 10 attention heads to prune? (b) How many byte strings of length 4 are there? (c) How many entries does a byte bigram table have (ordered pairs of bytes, repeats allowed)? `[number]`

<details><summary>Answer</summary>

**(a) $120$, (b) $4{,}294{,}967{,}296$, (c) $65{,}536$.** (a) An unordered choice: $\binom{10}{3} = \frac{10 \cdot 9 \cdot 8}{3!} = 120$. (b) Four positions with 256 options each: $256^4 = 2^{32}$. (c) Ordered pairs with repetition: $256^2 = 65{,}536$. Common mistakes: 720 for (a) counts ordered choices; 1024 for (b) mixes up base and exponent; 32,640 $= \binom{256}{2}$ for (c) counts unordered pairs of distinct bytes, but a bigram table needs $(a, b)$ and $(b, a)$ and $(a, a)$.

</details>

**q8.** In how many ways can 5 identical tokens be placed into 3 distinct buckets, empty buckets allowed? `[number]`

<details><summary>Answer</summary>

**$21$.** Stars and bars: 5 stars and 2 bars in a row, $\binom{5 + 3 - 1}{3 - 1} = \binom{7}{2} = 21$. Common mistakes: $3^5 = 243$ treats the tokens as distinct, and $\binom{5}{2} = 10$ forgets that the bars are placed among the stars.

</details>

**q9.** A linear layer $y = Wx + b$ maps $\mathbb{R}^n$ to $\mathbb{R}^m$. How many parameters does it have? `[expr in m, n]`

<details><summary>Answer</summary>

**$mn + m$.** $W$ is $m \times n$, so $mn$ weights, plus one bias per output: $m$. This is how `param_count` in [Counting: parameters, FLOPs, KV bytes, and memory plans](01-counting-params-flops-kv-bytes.md) adds biases. Common mistakes: $mn$ forgets the bias, and $mn + n$ sizes the bias by the input.

</details>

**q10.** A language model has an embedding table with $V$ rows of width $d$ and an untied output head that maps width $d$ to $V$ logits without a bias. How many parameters do the two hold together? `[expr in V, d]`

<details><summary>Answer</summary>

**$2Vd$.** The embedding is a $V \times d$ table, and the untied head is a $d \to V$ matrix: $Vd$ each. Tying the two (`tie_embeddings` in `ModelConfig`) halves this to $Vd$, which is the common wrong answer here.

</details>

**q11.** A decoder block of width $d$ has four $d \times d$ attention projections ($Q, K, V, O$) without biases, an MLP $d \to 4d \to d$ (two matrices, no biases), and two RMSNorm gain vectors of length $d$. How many parameters does the block have? `[expr in d]`

<details><summary>Answer</summary>

**$12d^2 + 2d$.** Attention: $4 \cdot d^2$. MLP: $d \cdot 4d + 4d \cdot d = 8d^2$. Norms: $2d$. This is where the rule "$N \approx 12 L d^2$" comes from. Common mistakes: $12d^2$ forgets the norm gains, and $8d^2 + 2d$ counts only one MLP matrix.

</details>

**q12.** Counting one multiplication and one addition as two floating-point operations, how many FLOPs does the product of an $m \times k$ matrix and a $k \times n$ matrix take? `[expr in m, k, n]`

<details><summary>Answer</summary>

**$2mkn$.** $C$ has $mn$ entries, and each is a dot product of length $k$: $k$ multiplications and $k$ additions into the accumulator, $2k$ FLOPs. That is the $2MKN$ of [Vectors, matrices, row-major layout, and matmul](../03-linear-algebra/01-vectors-matrices-and-matmul.md) and of the roofline in [the matmul kernel](../../build-an-llm/p09-kernels/04-tiled-batch-invariant-matmul.md). A common mistake is $mkn$, which counts multiply-adds, not FLOPs.

</details>

**q13.** The training-compute rule of thumb is $C \approx 6ND$ FLOPs for $N$ parameters and $D$ tokens. Give $C$ for $N = 10^7$ and $D = 2 \times 10^8$. `[number]`

<details><summary>Answer</summary>

**$1.2 \times 10^{16}$.** $6 \cdot 10^7 \cdot 2 \cdot 10^8 = 12 \cdot 10^{15}$. The 6 is 2 for the forward pass and 4 for the backward pass. A common mistake is $2 \cdot 10^{15}$, which uses $2ND$, the forward pass only.

</details>

**q14.** SmolLM2-135M has 30 layers, 3 key/value heads, and head dimension 64, and caches keys and values in bf16 (2 bytes per number). How many bytes of KV cache does one token take? `[number]`

<details><summary>Answer</summary>

**$23{,}040$.** $2 \cdot L \cdot H_{kv} \cdot d_h \cdot b = 2 \cdot 30 \cdot 3 \cdot 64 \cdot 2 = 23{,}040$ bytes. Common mistakes: 11,520 caches keys only (no factor 2), and 69,120 sizes the cache by the 9 query heads instead of the 3 key/value heads, which is exactly what GQA saves.

</details>

**q15.** How many bytes does that cache take for a context of 2048 tokens? `[number]`

<details><summary>Answer</summary>

**$47{,}185{,}920$.** $23{,}040 \cdot 2048 = 47{,}185{,}920$ bytes, exactly 45 MiB. Keys only would give half, 23,592,960.

</details>

**q16.** How many integers in $\{1, 2, \dots, 100\}$ are divisible by 3 or by 5? `[number]`

<details><summary>Answer</summary>

**$47$.** Inclusion-exclusion: $\lfloor 100/3 \rfloor + \lfloor 100/5 \rfloor - \lfloor 100/15 \rfloor = 33 + 20 - 6 = 47$. A common mistake is 53, which counts the multiples of 15 twice.

</details>

### Proof techniques

**q17.** Prove directly: if $n$ is an odd integer, then $n^2$ is odd. `[proof]`

<details><summary>Answer</summary>

Claim: if $n$ is an odd integer, then $n^2$ is odd.

Method: direct proof from the definition (an integer $n$ is odd when $n = 2k + 1$ for some integer $k$).

Let $n$ be odd, so $n = 2k + 1$ for an integer $k$. Then
$n^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1$.
Since $k$ is an integer, $j = 2k^2 + 2k$ is an integer, and $n^2 = 2j + 1$.

By the definition of odd, $n^2$ is odd.

Model proof also in [`q17.md`](../../../../course/solutions/solve/S-M05/q17.md).

</details>

**q18.** Prove by contrapositive: if $n$ is an integer and $n^2$ is even, then $n$ is even. `[proof]`

<details><summary>Answer</summary>

Claim: for an integer $n$, if $n^2$ is even then $n$ is even.

Method: contrapositive. The statement $P \Rightarrow Q$ is logically equivalent to $\lnot Q \Rightarrow \lnot P$ (they have the same truth table), so it suffices to prove: if $n$ is not even, then $n^2$ is not even. Every integer is even or odd and not both, so this reads: if $n$ is odd, then $n^2$ is odd.

That is exactly q17: write $n = 2k + 1$, then $n^2 = 2(2k^2 + 2k) + 1$ is odd.

The contrapositive holds, so the original claim holds: if $n^2$ is even, then $n$ is even.

Model proof also in [`q18.md`](../../../../course/solutions/solve/S-M05/q18.md).

</details>

**q19.** Prove by contradiction: $\sqrt{2}$ is irrational. `[proof]`

<details><summary>Answer</summary>

Claim: $\sqrt{2}$ is irrational (it is not $p/q$ for integers $p, q$ with $q \ne 0$).

Method: contradiction.

Suppose $\sqrt{2} = p/q$ with integers $p, q$, $q \ne 0$. Dividing by common factors, we may assume the fraction is in lowest terms: $p$ and $q$ share no factor greater than 1.

Squaring gives $p^2 = 2q^2$, so $p^2$ is even, and by q18 $p$ is even: $p = 2r$ for an integer $r$. Substituting, $4r^2 = 2q^2$, so $q^2 = 2r^2$ is even, and by q18 again $q$ is even.

Then 2 divides both $p$ and $q$, contradicting lowest terms. So the assumption is false, and $\sqrt{2}$ is irrational.

Model proof also in [`q19.md`](../../../../course/solutions/solve/S-M05/q19.md).

</details>

**q20.** Prove that there are infinitely many primes. `[proof]`

<details><summary>Answer</summary>

Claim: there are infinitely many primes (integers $p \ge 2$ whose only positive divisors are 1 and $p$).

Method: contradiction (Euclid).

Suppose there are only finitely many primes, $p_1, p_2, \dots, p_k$. Let $N = p_1 p_2 \cdots p_k + 1$. Then $N \ge 3 > 1$.

For each $i$, dividing $N$ by $p_i$ leaves remainder 1, because $p_i$ divides the product $p_1 \cdots p_k$. So no $p_i$ divides $N$.

But every integer greater than 1 has at least one prime factor (q28; or take its smallest divisor greater than 1, which must be prime). That prime factor is a prime not in the list $p_1, \dots, p_k$, contradicting that the list holds every prime.

So there are infinitely many primes.

Model proof also in [`q20.md`](../../../../course/solutions/solve/S-M05/q20.md).

</details>

**q21.** Prove by cases: $n^2 + n$ is even for every integer $n$. `[proof]`

<details><summary>Answer</summary>

Claim: for every integer $n$, $n^2 + n$ is even.

Method: proof by cases. Every integer is even or odd, so the two cases cover all $n$.

Case 1, $n$ even: $n = 2k$ for an integer $k$. Then $n^2 + n = 4k^2 + 2k = 2(2k^2 + k)$, and $2k^2 + k$ is an integer.

Case 2, $n$ odd: $n = 2k + 1$. Then $n^2 + n = 4k^2 + 4k + 1 + 2k + 1 = 4k^2 + 6k + 2 = 2(2k^2 + 3k + 1)$, and $2k^2 + 3k + 1$ is an integer.

In both cases $n^2 + n$ is 2 times an integer, so it is even for every integer $n$.

Model proof also in [`q21.md`](../../../../course/solutions/solve/S-M05/q21.md).

</details>

**q22.** Prove: for real $a, b \ge 0$, $\frac{a + b}{2} \ge \sqrt{ab}$, with equality exactly when $a = b$. `[proof]`

<details><summary>Answer</summary>

Claim: for real numbers $a, b \ge 0$, $\frac{a+b}{2} \ge \sqrt{ab}$, with equality exactly when $a = b$.

Method: direct, by rewriting the difference as a square.

Because $a, b \ge 0$, the real square roots $\sqrt{a}, \sqrt{b}$ exist, $a = (\sqrt{a})^2$, $b = (\sqrt{b})^2$, and $\sqrt{ab} = \sqrt{a}\sqrt{b}$. Then
$$\frac{a + b}{2} - \sqrt{ab} = \frac{(\sqrt a)^2 - 2\sqrt a \sqrt b + (\sqrt b)^2}{2} = \frac{(\sqrt a - \sqrt b)^2}{2}.$$
A square of a real number is $\ge 0$, so the difference is $\ge 0$, which is the inequality.

Equality holds exactly when $(\sqrt a - \sqrt b)^2 = 0$, that is $\sqrt a = \sqrt b$. Squaring both sides (and taking roots back, both nonnegative) shows this is the same as $a = b$.

So $\frac{a+b}{2} \ge \sqrt{ab}$, with equality if and only if $a = b$.

Model proof also in [`q22.md`](../../../../course/solutions/solve/S-M05/q22.md).

</details>

**q23.** Claim: $n^2 - n + 41$ is prime for every integer $n \ge 1$. (a) Is the claim true? `[bool]` (b) Give the smallest $n \ge 1$ for which $n^2 - n + 41$ is not prime. `[number]`

<details><summary>Answer</summary>

**(a) False, (b) $41$.** At $n = 41$: $41^2 - 41 + 41 = 41^2$, which is not prime. For $n = 1, \dots, 40$ the values are prime, which is why checking many cases is not a proof; one counterexample disproves a "for every" claim. A common mistake is 40, which belongs to the sibling polynomial $n^2 + n + 41$ ($40^2 + 40 + 41 = 41^2$); here $40^2 - 40 + 41 = 1601$ is prime.

</details>

**q24.** Prove (pigeonhole): any function from a set of 257 byte strings to 8-bit hash values maps two different strings to the same value. `[proof]`

<details><summary>Answer</summary>

Claim: any function $h$ from a set $S$ of 257 byte strings to 8-bit values maps two different strings to the same value.

Method: pigeonhole principle (if more than $k$ objects go into $k$ boxes, some box gets two).

An 8-bit value is one of $2^8 = 256$ possibilities, so the codomain has 256 elements (the boxes). Suppose, for contradiction, that $h$ maps different strings to different values, that is, $h$ is injective. Then the 257 strings would have 257 distinct images, a subset of size 257 of a set of size 256, which is impossible.

So $h$ is not injective: there are strings $s \ne t$ in $S$ with $h(s) = h(t)$. Every hash table into a fixed number of buckets has collisions once there are more keys than buckets.

Model proof also in [`q24.md`](../../../../course/solutions/solve/S-M05/q24.md).

</details>

**q25.** Prove: the sum of a rational number and an irrational number is irrational. `[proof]`

<details><summary>Answer</summary>

Claim: if $r$ is rational and $x$ is irrational, then $r + x$ is irrational.

Method: contradiction.

Suppose $r + x = q$ for a rational $q$. Then $x = q - r$. Write $q = a/b$ and $r = c/d$ with integers $a, b, c, d$ and $b, d \ne 0$. Then $x = (ad - bc)/(bd)$ with $bd \ne 0$, a ratio of integers, so $x$ is rational.

This contradicts the assumption that $x$ is irrational. So $r + x$ is not rational: it is irrational.

Model proof also in [`q25.md`](../../../../course/solutions/solve/S-M05/q25.md).

</details>

### Induction and loop invariants

**q26.** Prove by induction: $1 + 2 + \dots + n = \frac{n(n+1)}{2}$ for every integer $n \ge 1$. `[proof]`

<details><summary>Answer</summary>

Claim: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$ for every integer $n \ge 1$.

Method: induction on $n$.

Base case $n = 1$: the left side is $1$ and the right side is $1 \cdot 2 / 2 = 1$.

Induction hypothesis: for some integer $k \ge 1$, $\sum_{i=1}^{k} i = \frac{k(k+1)}{2}$.

Inductive step: $\sum_{i=1}^{k+1} i = \left(\sum_{i=1}^{k} i\right) + (k + 1) = \frac{k(k+1)}{2} + (k+1)$ by the hypothesis, $= \frac{k(k+1) + 2(k+1)}{2} = \frac{(k+1)(k+2)}{2}$, which is the formula for $n = k + 1$.

By induction, the formula holds for every $n \ge 1$.

Model proof also in [`q26.md`](../../../../course/solutions/solve/S-M05/q26.md).

</details>

**q27.** Prove by induction: $2^n > n^2$ for every integer $n \ge 5$. `[proof]`

<details><summary>Answer</summary>

Claim: $2^n > n^2$ for every integer $n \ge 5$.

Method: induction on $n$ starting at 5.

Base case $n = 5$: $2^5 = 32 > 25 = 5^2$.

Induction hypothesis: $2^k > k^2$ for some integer $k \ge 5$.

Inductive step: $2^{k+1} = 2 \cdot 2^k > 2k^2$ by the hypothesis. Now $2k^2 \ge (k+1)^2$ is the same as $k^2 \ge 2k + 1$, that is $k^2 - 2k - 1 \ge 0$, that is $(k - 1)^2 \ge 2$, which holds for every $k \ge 3$ and so for $k \ge 5$. Therefore $2^{k+1} > 2k^2 \ge (k+1)^2$.

By induction, $2^n > n^2$ for every $n \ge 5$. (The claim fails at $n = 4$, where $16 = 16$, which is why the base case is 5.)

Model proof also in [`q27.md`](../../../../course/solutions/solve/S-M05/q27.md).

</details>

**q28.** Prove by strong induction: every integer $n \ge 2$ is a product of one or more primes. `[proof]`

<details><summary>Answer</summary>

Claim: every integer $n \ge 2$ is a product of one or more primes.

Method: strong induction on $n$.

Strong hypothesis: for a fixed $n \ge 2$, every integer $m$ with $2 \le m < n$ is a product of primes. We show $n$ is too.

Case 1, $n$ is prime: then $n$ itself is a product of one prime. This covers the base case $n = 2$, which is prime.

Case 2, $n$ is not prime: then $n$ has a divisor $a$ with $1 < a < n$, so $n = ab$ with $b = n / a$, and $1 < b < n$ as well. Both $a$ and $b$ lie in $[2, n)$, so by the hypothesis $a = p_1 \cdots p_r$ and $b = q_1 \cdots q_s$ are products of primes, and $n = p_1 \cdots p_r q_1 \cdots q_s$ is one too.

By strong induction, every integer $n \ge 2$ is a product of primes.

Model proof also in [`q28.md`](../../../../course/solutions/solve/S-M05/q28.md).

</details>

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

<details><summary>Answer</summary>

Claim: for every integer $n \ge 0$ the loop returns $x^n$.

Method: loop invariant $I$: $r \cdot b^e = x^n$, with $e$ a nonnegative integer.

Initialization: before the first iteration $r = 1$, $b = x$, $e = n$, so $r \cdot b^e = x^n$.

Maintenance: suppose $I$ holds and $e > 0$ at the top of an iteration; write $r', b', e'$ for the values after it.
- If $e$ is odd, $e = 2t + 1$ with $t = \lfloor e/2 \rfloor$. Then $r' = rb$, $b' = b^2$, $e' = t$, and $r' b'^{e'} = r b \cdot b^{2t} = r b^{2t+1} = r b^e = x^n$.
- If $e$ is even, $e = 2t$. Then $r' = r$, $b' = b^2$, $e' = t$, and $r' b'^{e'} = r b^{2t} = r b^e = x^n$.
In both cases $I$ holds again and $e'$ is a nonnegative integer.

Termination: while $e > 0$, $e' = \lfloor e/2 \rfloor < e$, so $e$ is a strictly decreasing sequence of nonnegative integers and reaches 0 after finitely many iterations.

At exit $e = 0$, so $I$ gives $r \cdot b^0 = r = x^n$, and the loop returns $x^n$.

Model proof also in [`q29.md`](../../../../course/solutions/solve/S-M05/q29.md).

</details>

**q30.** The online softmax (the C kernel of [Softmax in C: three-pass and online two-pass](../../build-an-llm/p09-kernels/05-softmax-three-pass-and-online.md)) reads $x_0, x_1, \dots$ once, keeping a running maximum $m$ and a running sum $s$. It starts at $m = -\infty$, $s = 0$ (with $e^{-\infty} = 0$), and for each new $x_i$ sets $m' = \max(m, x_i)$ and $s' = s \cdot e^{m - m'} + e^{x_i - m'}$. Prove that after reading $x_0, \dots, x_{i-1}$ it holds that $m = \max_{j < i} x_j$ and $s = \sum_{j < i} e^{x_j - m}$. `[proof]`

<details><summary>Answer</summary>

Claim: after the loop has read $x_0, \dots, x_{i-1}$, $m = \max_{j < i} x_j$ and $s = \sum_{j<i} e^{x_j - m}$ (for $i = 0$: $m = -\infty$, the max of nothing, and $s = 0$, the empty sum).

Method: induction on $i$, the number of elements read (a loop invariant).

Base $i = 0$: the loop starts with $m = -\infty$ and $s = 0$, which are the empty max and the empty sum.

Step: assume the claim after $i$ elements, with values $m, s$, and read $x_i$. Then $m' = \max(m, x_i) = \max(\max_{j<i} x_j, x_i) = \max_{j \le i} x_j$. For the sum, using the hypothesis and $e^{a}e^{b} = e^{a+b}$:
$$s \cdot e^{m - m'} = \sum_{j<i} e^{x_j - m} e^{m - m'} = \sum_{j<i} e^{x_j - m'},$$
so $s' = s e^{m - m'} + e^{x_i - m'} = \sum_{j \le i} e^{x_j - m'}$. When $i = 0$, $s = 0$ and the first term is 0 (with $e^{-\infty} = 0$), so $s' = e^{x_0 - x_0} = 1$, which also matches. The claim holds after $i + 1$ elements.

By induction it holds after every element. Every exponent $x_j - m$ is $\le 0$, so no term exceeds 1 and $s$ never overflows.

Model proof also in [`q30.md`](../../../../course/solutions/solve/S-M05/q30.md).

</details>

**q31.** Give a closed form for $1^2 + 2^2 + \dots + n^2$. `[expr in n]`

<details><summary>Answer</summary>

**$\frac{n(n + 1)(2n + 1)}{6}$.** Provable by induction like q26: the step adds $(k + 1)^2$ and $\frac{k(k+1)(2k+1)}{6} + (k+1)^2 = \frac{(k+1)(k+2)(2k+3)}{6}$. Check: $n = 2$ gives $5$. Common mistakes: $n^3/3$ is only the leading term, and $\frac{n(n+1)}{2}$ is the sum of $n$, not $n^2$.

</details>

**q32.** Give a closed form for $2^0 + 2^1 + \dots + 2^{n-1}$. `[expr in n]`

<details><summary>Answer</summary>

**$2^n - 1$.** A geometric series: $\sum_{i=0}^{n-1} 2^i = \frac{2^n - 1}{2 - 1}$. Check: $n = 3$ gives $1 + 2 + 4 = 7$. A common mistake is $2^{n+1} - 1$, an off-by-one in the last term (that sums up to $2^n$).

</details>

**q33.** Run the loop of q30 on $x = (1, 3, 2)$. Give the final $s$ exactly. `[number]`

<details><summary>Answer</summary>

**$1 + e^{-1} + e^{-2}$.** Start $m = -\infty$, $s = 0$. Read 1: $m = 1$, $s = 0 + e^0 = 1$. Read 3: $m = 3$, $s = 1 \cdot e^{1 - 3} + e^0 = e^{-2} + 1$. Read 2: $m = 3$, $s = (1 + e^{-2}) \cdot e^0 + e^{2 - 3} = 1 + e^{-1} + e^{-2}$. This equals $\sum_j e^{x_j - 3}$, as q30 promises. Common mistakes: $e^1 + e^3 + e^2$ forgets the shift by the running max, which is exactly what keeps $s$ from overflowing; $e^{-1} + e^{-2}$ drops the max element's own term $e^0 = 1$.

</details>

### Relations

**q34.** Let $R = \{(1,1), (2,2), (3,3), (1,2), (2,1)\}$ on $\{1, 2, 3\}$. Is $R$ (a) reflexive, (b) symmetric, (c) transitive? `[bool]`

<details><summary>Answer</summary>

**(a) True, (b) true, (c) true.** (a) All of $(1,1), (2,2), (3,3)$ are in $R$. (b) The only non-diagonal pairs $(1,2)$ and $(2,1)$ come together. (c) Every chain $x R y R z$ stays inside $\{1, 2\}$ or inside $\{3\}$, and both blocks are fully related. $R$ is the equivalence relation with classes $\{1, 2\}$ and $\{3\}$.

</details>

**q35.** How many equivalence relations are there on a 4-element set? `[number]`

<details><summary>Answer</summary>

**$15$.** Equivalence relations on a set are its partitions, so the answer is the Bell number $B_4$. Count partitions of $\{1, 2, 3, 4\}$ by block sizes: $4$ (1 way), $3 + 1$ (4), $2 + 2$ (3), $2 + 1 + 1$ (6), $1 + 1 + 1 + 1$ (1), total 15. Common mistakes: 16 $= 2^4$ counts subsets, and 24 $= 4!$ counts orderings.

</details>

**q36.** Prove: for any function $f: X \to Y$, the relation $x \sim y \iff f(x) = f(y)$ is an equivalence relation on $X$. (Unicode normalization in [The tokenizer protocol and the char tokenizer](../../build-an-llm/p01-tokenizers/01-tokenizer-protocol-and-char.md) is this relation with $f = \mathrm{NFC}$.) `[proof]`

<details><summary>Answer</summary>

Claim: for any function $f: X \to Y$, the relation $x \sim y \iff f(x) = f(y)$ is an equivalence relation on $X$.

Method: check the three defining properties directly; each reduces to the same property of equality in $Y$.

Reflexive: for every $x \in X$, $f(x) = f(x)$, so $x \sim x$.

Symmetric: if $x \sim y$ then $f(x) = f(y)$, so $f(y) = f(x)$, so $y \sim x$.

Transitive: if $x \sim y$ and $y \sim z$ then $f(x) = f(y)$ and $f(y) = f(z)$, so $f(x) = f(z)$, so $x \sim z$.

So $\sim$ is an equivalence relation; its classes are the nonempty preimages $f^{-1}(\{y\})$.

Model proof also in [`q36.md`](../../../../course/solutions/solve/S-M05/q36.md).

</details>

**q37.** Prove: divisibility ($a \mid b$ when $b = ka$ for some integer $k$) is a partial order on the positive integers. `[proof]`

<details><summary>Answer</summary>

Claim: on the positive integers, $a \mid b$ (there is an integer $k$ with $b = ka$) is a partial order: reflexive, antisymmetric, and transitive.

Method: check each property from the definition.

Reflexive: $a = 1 \cdot a$, so $a \mid a$.

Antisymmetric: suppose $a \mid b$ and $b \mid a$, so $b = ka$ and $a = lb$ for integers $k, l$. Then $a = lka$, and since $a > 0$, $lk = 1$. Both $k = b/a$ and $l = a/b$ are positive because $a, b > 0$, and the only positive integers with product 1 are $k = l = 1$. So $b = a$.

Transitive: if $b = ka$ and $c = lb$, then $c = (lk)a$ with $lk$ an integer, so $a \mid c$.

So divisibility is a partial order on the positive integers. Antisymmetry needs positivity: on all integers $2 \mid -2$ and $-2 \mid 2$ but $2 \ne -2$.

Model proof also in [`q37.md`](../../../../course/solutions/solve/S-M05/q37.md).

</details>

### Functions and bijections

**q38.** Let $f: \mathbb{Z} \to \mathbb{Z}$, $f(n) = 2n + 1$. Is $f$ (a) injective, (b) surjective? `[bool]`

<details><summary>Answer</summary>

**(a) True, (b) false.** (a) $2n + 1 = 2n' + 1$ gives $n = n'$. (b) $2n + 1$ is always odd, so no even integer (for example 0) is an image. Injective and surjective are the two halves of the bijection test in [Injective, surjective, bijective: the GPT-2 byte map](02-injective-surjective-bijective-gpt2-byte-map.md), and they are independent.

</details>

**q39.** (a) How many bijections are there from a 5-element set to itself? (b) How many injective functions are there from a 3-element set to a 5-element set? `[number]`

<details><summary>Answer</summary>

**(a) $120$, (b) $60$.** (a) A bijection of a 5-set is an ordering: $5! = 120$. (b) The first element has 5 images, the next 4, the last 3: $5 \cdot 4 \cdot 3 = 60$. Common mistakes: $5^5 = 3125$ and $5^3 = 125$ count every function, and $\binom{5}{3} = 10$ forgets that which element goes where matters.

</details>

**q40.** Prove: if $f: X \to Y$ and $g: Y \to Z$ are bijections, then $g \circ f$ is a bijection. `[proof]`

<details><summary>Answer</summary>

Claim: if $f: X \to Y$ and $g: Y \to Z$ are bijections, then $g \circ f: X \to Z$, $x \mapsto g(f(x))$, is a bijection.

Method: prove injective and surjective separately.

Injective: suppose $g(f(x)) = g(f(x'))$. Since $g$ is injective, $f(x) = f(x')$; since $f$ is injective, $x = x'$.

Surjective: let $z \in Z$. Since $g$ is surjective there is $y \in Y$ with $g(y) = z$; since $f$ is surjective there is $x \in X$ with $f(x) = y$. Then $(g \circ f)(x) = z$.

Being injective and surjective, $g \circ f$ is a bijection (its inverse is $f^{-1} \circ g^{-1}$).

Model proof also in [`q40.md`](../../../../course/solutions/solve/S-M05/q40.md).

</details>

**q41.** GPT-2's byte-to-character map ([Injective, surjective, bijective: the GPT-2 byte map](02-injective-surjective-bijective-gpt2-byte-map.md)) keeps the 188 printable bytes $P = \{33, \dots, 126\} \cup \{161, \dots, 172\} \cup \{174, \dots, 255\}$ as themselves ($f(b) = b$) and sends the other 68 bytes, in increasing order, to $256, 257, \dots, 323$. Prove that $f: \{0, \dots, 255\} \to \mathbb{N}$ is injective, so it is a bijection onto its image. `[proof]`

<details><summary>Answer</summary>

Claim: the GPT-2 byte map $f$ is injective on $\{0, \dots, 255\}$.

Setup: $P = \{33..126\} \cup \{161..172\} \cup \{174..255\}$ has $94 + 12 + 82 = 188$ elements, so its complement $Q = \{0..255\} \setminus P$ has $256 - 188 = 68$ elements. List $Q$ in increasing order as $q_0 < q_1 < \dots < q_{67}$; then $f(b) = b$ for $b \in P$ and $f(q_t) = 256 + t$.

Method: take bytes $a \ne b$ and show $f(a) \ne f(b)$, in three cases that cover every pair.

Both in $P$: $f(a) = a \ne b = f(b)$.

Both in $Q$: $a = q_s$, $b = q_t$ with $s \ne t$ (the listing has no repeats), so $f(a) = 256 + s \ne 256 + t = f(b)$.

One in each, say $a \in P$, $b \in Q$: $f(a) = a \le 255 < 256 \le f(b)$.

So $f$ is injective. A function is always surjective onto its image, so $f$ is a bijection from the 256 bytes onto its 256 image characters, and `unicode_to_bytes` is its inverse.

Model proof also in [`q41.md`](../../../../course/solutions/solve/S-M05/q41.md).

</details>

## What can go wrong

The common wrong answers discussed under the problems come from a small number of recurring mistakes. Each has a symptom you would see in real code, and a habit that prevents it:

| Pitfall | Symptom | How to avoid it |
|---|---|---|
| Combining masks with OR, or forgetting the padding key | a pad position gets attention weight | intersect the two masks (AND) and clear the pad column (q4) |
| Negating only the inner predicate of a nested quantifier | one tests "some entry is 0" instead of "some row is all 0" | swap every quantifier, outermost first, then negate the predicate (q5) |
| Counting multiply-adds instead of FLOPs | FLOP and roofline numbers off by 2 | count a multiplication and an addition separately: $2k$ per output entry (q12) |
| Using $2ND$ (forward only) for training compute | a training budget 3 times too small | use $6ND$: 2 for the forward pass, 4 for the backward (q13) |
| Caching keys only, or sizing the cache by query heads | KV memory off by 2 or by the GQA group size | write the factor 2 for keys and values and use $H_{kv}$, not $H$ (q14) |
| Adding overlapping cases twice | inclusion-exclusion overcount | subtract the intersection once (q16) |
| Checking many cases instead of proving | a "for every $n$" claim believed from $n = 1$ to $40$ | look for one counterexample, $n = 41$, instead of checking cases (q23) |
| Using the conclusion inside the inductive step | a circular proof | state the hypothesis for $k$ and use it only to derive the case $k + 1$ (q26 to q30) |
| Forgetting the running sum must be rescaled when the max changes | online softmax overflows or is wrong after a new max | multiply $s$ by $e^{m - m'}$ on every step, so every term is shifted by the current max (q30, q33) |
| Confusing all functions with injective ones | $n^k$ where $\frac{n!}{(n-k)!}$ was asked | count images one element at a time without reuse: $n(n-1)\cdots$ (q39) |

## Where this leads

The tools of this set are used directly in code throughout the book and in *Build Your Own LLM System*:

- [Graphs, DAGs, and an iterative topological sort](../06-discrete-math-2/01-graphs-dags-and-topological-sort.md) orders the autograd graph; its correctness proof is an induction over the traversal (q20 of [the first discrete math 2 problem set](../06-discrete-math-2/90-problem-set-a.md)).
- [Modular arithmetic, hashing, and PCG32](../06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) builds PCG32 and SplitMix64 from counting bits, rotations, and arithmetic modulo $2^{64}$.
- [Counting: parameters, FLOPs, KV bytes, and memory plans](01-counting-params-flops-kv-bytes.md) turns q9 to q15 into code: `param_count`, `flops_per_token`, `kv_bytes_per_token`, and `memory_plan`.
- [Injective, surjective, bijective: the GPT-2 byte map](02-injective-surjective-bijective-gpt2-byte-map.md) implements the bijection of q41; a correct solution must be injective and surjective.
- [Masks: causal, padding, sliding window, additive](../../build-an-llm/p05-transformer-2017/02-masks.md) combines causal and padding masks as sets (q4, q5).
- [Softmax in C: three-pass and online two-pass](../../build-an-llm/p09-kernels/05-softmax-three-pass-and-online.md) is the online softmax kernel; q30 is its correctness argument.
- [The tokenizer protocol and the char tokenizer](../../build-an-llm/p01-tokenizers/01-tokenizer-protocol-and-char.md) treats Unicode normalization as an equivalence relation (q36).
- [The first discrete math 2 problem set](../06-discrete-math-2/90-problem-set-a.md) builds graphs, modular arithmetic, and hashing on these counting and proof tools.

### Further reading

- Hammack, *Book of Proof* (free), chapters 1 to 10 and 12.
- Velleman, *How to Prove It*, chapters 3 and 6.

## Key terms

- **Set**: an unordered collection of distinct elements.
- **Contrapositive**: $\lnot Q \Rightarrow \lnot P$, logically equivalent to $P \Rightarrow Q$.
- **Binomial coefficient**: $\binom{n}{k}$, the number of $k$-element subsets of an $n$-element set.
- **Induction hypothesis**: the assumption $P(k)$ used to prove $P(k + 1)$.
- **Loop invariant**: a statement that holds before a loop, survives each iteration, and with the exit condition implies the result.
- **Equivalence relation**: a reflexive, symmetric, transitive relation; its classes partition the set.
- **Bijection**: a function that is both injective and surjective, and so has an inverse.

## Summary

- An attention mask is a set of allowed (query, key) pairs, so combining masks is set intersection and a fully masked row is the negation of "every row has a key" (q4, q5).
- Every size in the system is a count: parameters are sums of matrix shapes, FLOPs are $2mkn$ per matmul, and KV-cache bytes are $2 \cdot L \cdot H_{kv} \cdot d_h \cdot b$ per token (q9 to q15).
- A loop is correct when an invariant holds before it, survives one iteration, and implies the result at exit; the online softmax kernel is proved this way (q29, q30).
- An equivalence relation is the same thing as "has the same image under some function", which is how Unicode normalization groups strings (q36).
- A bijection is injective plus surjective; GPT-2's byte map is checked by exactly those two halves (q38 to q41).
