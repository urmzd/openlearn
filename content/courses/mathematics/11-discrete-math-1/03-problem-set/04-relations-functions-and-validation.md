# Relations, functions, and validation

This part of [Discrete math 1 problem set: sets, counting, proofs, induction, relations, bijections](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Check relation properties and prove function/bijection claims with explicit domains.

<a id="relations-1"></a>
### Relations

**q34.** Let $R = \{(1,1), (2,2), (3,3), (1,2), (2,1)\}$ on $\{1, 2, 3\}$. Is $R$ (a) reflexive, (b) symmetric, (c) transitive? `[bool]`

<details><summary>Answer</summary>

**(a) True, (b) true, (c) true.** (a) All of $(1,1), (2,2), (3,3)$ are in $R$. (b) The only non-diagonal pairs $(1,2)$ and $(2,1)$ come together. (c) Every chain $x R y R z$ stays inside $\{1, 2\}$ or inside $\{3\}$, and both blocks are fully related. $R$ is the equivalence relation with classes $\{1, 2\}$ and $\{3\}$.

</details>

**q35.** How many equivalence relations are there on a 4-element set? `[number]`

<details><summary>Answer</summary>

**$15$.** Equivalence relations on a set are its partitions, so the answer is the Bell number $B_4$. Count partitions of $\{1, 2, 3, 4\}$ by block sizes: $4$ (1 way), $3 + 1$ (4), $2 + 2$ (3), $2 + 1 + 1$ (6), $1 + 1 + 1 + 1$ (1), total 15. Common mistakes: 16 $= 2^4$ counts subsets, and 24 $= 4!$ counts orderings.

</details>

**q36.** Prove: for any function $f: X \to Y$, the relation $x \sim y \iff f(x) = f(y)$ is an equivalence relation on $X$. (Unicode normalization in [The tokenizer protocol and the char tokenizer](../../../ai-engineering/07-tokenizers/01-tokenizer-protocol-and-char.md) is this relation with $f = \mathrm{NFC}$.) `[proof]`

<details><summary>Answer</summary>

Claim: for any function $f: X \to Y$, the relation $x \sim y \iff f(x) = f(y)$ is an equivalence relation on $X$.

Method: check the three defining properties directly; each reduces to the same property of equality in $Y$.

Reflexive: for every $x \in X$, $f(x) = f(x)$, so $x \sim x$.

Symmetric: if $x \sim y$ then $f(x) = f(y)$, so $f(y) = f(x)$, so $y \sim x$.

Transitive: if $x \sim y$ and $y \sim z$ then $f(x) = f(y)$ and $f(y) = f(z)$, so $f(x) = f(z)$, so $x \sim z$.

So $\sim$ is an equivalence relation; its classes are the nonempty preimages $f^{-1}(\{y\})$.

Model proof also in [`q36.md`](../../../../../course/solutions/solve/S-M05/q36.md).

</details>

**q37.** Prove: divisibility ($a \mid b$ when $b = ka$ for some integer $k$) is a partial order on the positive integers. `[proof]`

<details><summary>Answer</summary>

Claim: on the positive integers, $a \mid b$ (there is an integer $k$ with $b = ka$) is a partial order: reflexive, antisymmetric, and transitive.

Method: check each property from the definition.

Reflexive: $a = 1 \cdot a$, so $a \mid a$.

Antisymmetric: suppose $a \mid b$ and $b \mid a$, so $b = ka$ and $a = lb$ for integers $k, l$. Then $a = lka$, and since $a > 0$, $lk = 1$. Both $k = b/a$ and $l = a/b$ are positive because $a, b > 0$, and the only positive integers with product 1 are $k = l = 1$. So $b = a$.

Transitive: if $b = ka$ and $c = lb$, then $c = (lk)a$ with $lk$ an integer, so $a \mid c$.

So divisibility is a partial order on the positive integers. Antisymmetry needs positivity: on all integers $2 \mid -2$ and $-2 \mid 2$ but $2 \ne -2$.

Model proof also in [`q37.md`](../../../../../course/solutions/solve/S-M05/q37.md).

</details>

<a id="functions-and-bijections-1"></a>
### Functions and bijections

**q38.** Let $f: \mathbb{Z} \to \mathbb{Z}$, $f(n) = 2n + 1$. Is $f$ (a) injective, (b) surjective? `[bool]`

<details><summary>Answer</summary>

**(a) True, (b) false.** (a) $2n + 1 = 2n' + 1$ gives $n = n'$. (b) $2n + 1$ is always odd, so no even integer (for example 0) is an image. Injective and surjective are the two halves of the bijection test in [Injective, surjective, bijective: the GPT-2 byte map](../02-injective-surjective-bijective-gpt2-byte-map.md), and they are independent.

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

Model proof also in [`q40.md`](../../../../../course/solutions/solve/S-M05/q40.md).

</details>

**q41.** GPT-2's byte-to-character map ([Injective, surjective, bijective: the GPT-2 byte map](../02-injective-surjective-bijective-gpt2-byte-map.md)) keeps the 188 printable bytes $P = \{33, \dots, 126\} \cup \{161, \dots, 172\} \cup \{174, \dots, 255\}$ as themselves ($f(b) = b$) and sends the other 68 bytes, in increasing order, to $256, 257, \dots, 323$. Prove that $f: \{0, \dots, 255\} \to \mathbb{N}$ is injective, so it is a bijection onto its image. `[proof]`

<details><summary>Answer</summary>

Claim: the GPT-2 byte map $f$ is injective on $\{0, \dots, 255\}$.

Setup: $P = \{33..126\} \cup \{161..172\} \cup \{174..255\}$ has $94 + 12 + 82 = 188$ elements, so its complement $Q = \{0..255\} \setminus P$ has $256 - 188 = 68$ elements. List $Q$ in increasing order as $q_0 < q_1 < \dots < q_{67}$; then $f(b) = b$ for $b \in P$ and $f(q_t) = 256 + t$.

Method: take bytes $a \ne b$ and show $f(a) \ne f(b)$, in three cases that cover every pair.

Both in $P$: $f(a) = a \ne b = f(b)$.

Both in $Q$: $a = q_s$, $b = q_t$ with $s \ne t$ (the listing has no repeats), so $f(a) = 256 + s \ne 256 + t = f(b)$.

One in each, say $a \in P$, $b \in Q$: $f(a) = a \le 255 < 256 \le f(b)$.

So $f$ is injective. A function is always surjective onto its image, so $f$ is a bijection from the 256 bytes onto its 256 image characters, and `unicode_to_bytes` is its inverse.

Model proof also in [`q41.md`](../../../../../course/solutions/solve/S-M05/q41.md).

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

The tools of this set are used directly in code throughout the book and in *AI Engineering*:

- [Graphs, DAGs, and an iterative topological sort](../../12-discrete-math-2/01-graphs-dags-and-topological-sort.md) orders the autograd graph; its correctness proof is an induction over the traversal (q20 of [the first discrete math 2 problem set](../../12-discrete-math-2/04-problem-set-a.md)).
- [Modular arithmetic, hashing, and PCG32](../../12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) builds PCG32 and SplitMix64 from counting bits, rotations, and arithmetic modulo $2^{64}$.
- [Counting: parameters, FLOPs, KV bytes, and memory plans](../01-counting-params-flops-kv-bytes.md) turns q9 to q15 into code: `param_count`, `flops_per_token`, `kv_bytes_per_token`, and `memory_plan`.
- [Injective, surjective, bijective: the GPT-2 byte map](../02-injective-surjective-bijective-gpt2-byte-map.md) implements the bijection of q41; a correct solution must be injective and surjective.
- [Masks: causal, padding, sliding window, additive](../../../ai-engineering/11-transformer/02-masks.md) combines causal and padding masks as sets (q4, q5).
- [Softmax in C: three-pass and online two-pass](../../../ai-engineering/22-kernels/04-softmax-three-pass-and-online.md) is the online softmax kernel; q30 is its correctness argument.
- [The tokenizer protocol and the char tokenizer](../../../ai-engineering/07-tokenizers/01-tokenizer-protocol-and-char.md) treats Unicode normalization as an equivalence relation (q36).
- [The first discrete math 2 problem set](../../12-discrete-math-2/04-problem-set-a.md) builds graphs, modular arithmetic, and hashing on these counting and proof tools.

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

## Review questions

1. Why are equal finite cardinalities alone not enough to prove a proposed function is a bijection?

   <details><summary>Answer</summary>

   A specific function still needs the relevant injectivity or surjectivity proof; cardinality alone does not establish its mapping behavior.

   </details>

[Contents](README.md)
