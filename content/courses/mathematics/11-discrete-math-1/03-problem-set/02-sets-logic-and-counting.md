# Sets, logic, and counting

This part of [Discrete math 1 problem set: sets, counting, proofs, induction, relations, bijections](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Work the set, quantified-logic, and counting problems before revealing solutions.

### Sets and logic

**q1.** Let $U = \{1, 2, \dots, 10\}$, $A = \{1, 2, 3, 4, 5, 6\}$ and $B = \{2, 4, 6, 8\}$.
(a) $A \cap B$. (b) $A \setminus B$. (c) The complement of $A \cup B$ in $U$. `[set]`

<details><summary>Answer</summary>

**(a) $\{2, 4, 6\}$, (b) $\{1, 3, 5\}$, (c) $\{7, 9, 10\}$.** (a) The elements in both: $2, 4, 6$. (b) The elements of $A$ that are not in $B$: $1, 3, 5$. (c) $A \cup B = \{1, 2, 3, 4, 5, 6, 8\}$, and the rest of $U$ is $\{7, 9, 10\}$. Common mistakes: the union $\{1, 2, 3, 4, 5, 6, 8\}$ for (a); $B \setminus A = \{8\}$ for (b), since difference is not symmetric; and $\{7, 8, 9, 10\}$ for (c), the complement of $A$ alone. These are the operations of the mask algebra in [Masks: causal, padding, sliding window, additive](../../../ai-engineering/11-transformer/02-masks.md) in *AI Engineering*.

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

**(a).** The statement is $\forall i\ \exists j\ M_{ij} = 1$. Negating swaps each quantifier and negates the predicate: $\exists i\ \forall j\ M_{ij} = 0$, a row with no allowed key. That fully masked row is the NaN case of [Masks: causal, padding, sliding window, additive](../../../ai-engineering/11-transformer/02-masks.md) and of [Stable numerics: logsumexp, softmax, and compensated sums](../../15-numerical-methods-and-floating-point/02-stable-numerics.md). A common mistake is (c), which negates only the inner predicate and describes any matrix with a single 0; (b) is much stronger than the negation.

</details>

**q6.** Prove De Morgan's law $(A \cup B)^c = A^c \cap B^c$ for subsets $A, B$ of a universe $U$. `[proof]`

<details><summary>Answer</summary>

Claim: for subsets $A, B$ of a universe $U$, $(A \cup B)^c = A^c \cap B^c$, where $X^c = U \setminus X$.

Method: show each side is a subset of the other (double inclusion).

($\subseteq$) Let $x \in (A \cup B)^c$. Then $x \in U$ and $x \notin A \cup B$. By the definition of union, $x \in A \cup B$ means "$x \in A$ or $x \in B$", so $x \notin A \cup B$ means "not ($x \in A$ or $x \in B$)", which by De Morgan's law for propositions is "$x \notin A$ and $x \notin B$". Hence $x \in A^c$ and $x \in B^c$, so $x \in A^c \cap B^c$.

($\supseteq$) Let $x \in A^c \cap B^c$. Then $x \in U$, $x \notin A$, and $x \notin B$. So "$x \in A$ or $x \in B$" is false, that is $x \notin A \cup B$, and $x \in (A \cup B)^c$.

Each set contains the other, so $(A \cup B)^c = A^c \cap B^c$.

Model proof also in [`q6.md`](../../../../../course/solutions/solve/S-M05/q6.md).

</details>

<a id="counting-1"></a>
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

**$mn + m$.** $W$ is $m \times n$, so $mn$ weights, plus one bias per output: $m$. This is how `param_count` in [Counting: parameters, FLOPs, KV bytes, and memory plans](../01-counting-params-flops-kv-bytes.md) adds biases. Common mistakes: $mn$ forgets the bias, and $mn + n$ sizes the bias by the input.

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

**$2mkn$.** $C$ has $mn$ entries, and each is a dot product of length $k$: $k$ multiplications and $k$ additions into the accumulator, $2k$ FLOPs. That is the $2MKN$ of [Vectors, matrices, row-major layout, and matmul](../../09-linear-algebra/01-vectors-matrices-and-matmul.md) and of the roofline in [the matmul kernel](../../../ai-engineering/22-kernels/03-tiled-batch-invariant-matmul.md). A common mistake is $mkn$, which counts multiply-adds, not FLOPs.

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

## Review questions

1. How can a quantified claim be disproved?

   <details><summary>Answer</summary>

   Give a counterexample with the required domain membership and show it violates the predicate.

   </details>

[Contents](README.md) · [Next: Proofs, induction, and invariants](03-proofs-induction-and-invariants.md)
