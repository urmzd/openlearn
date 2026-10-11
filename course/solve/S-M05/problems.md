# S-M05 problems: sets, logic, counting, proofs, induction, relations, bijections

Answer every question in `solve/S-M05.toml` (written by `ol start S-M05`).
The tag after each question is its answer type: `[set]` is `{a, b}`,
`[matrix]` is a list of rows `[[1, 0], [0, 1]]`, `[number]` is an exact value
(`120`, `3/4`, `2^32`), `[expr]` is a formula in the named variables
(`m*n + m`), `[bool]` is `true` or `false`, `[choice]` is one letter, and
`[proof]` is a file `solve/S-M05/qN.md` that you grade against its rubric.

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
