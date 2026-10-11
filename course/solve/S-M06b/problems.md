# S-M06b problems: trees, birthday bounds, MinHash and LSH, Bloom filters, recurrences

Answer every question in `solve/S-M06b.toml` (written by `ol start S-M06b`).
The tag after each question is its answer type: `[number]` is a value (exact
where the question says so; otherwise an expression such as `10*log(2)` or a
decimal to the stated precision is fine), `[expr]` a formula in the named
variables, and `[proof]` a file `solve/S-M06b/qN.md` graded against its rubric.
`log` is the natural logarithm; write $\log_2 x$ as `log(x)/log(2)`.

### Trees

**q1.** A tree has 12 nodes. How many edges does it have? `[number]`

**q2.** Build the trie (`M06.2`) of the five keys `car`, `cart`, `care`, `cat`, `dog`. How many nodes does it have, counting the root? `[number]`

**q3.** A full binary tree is a rooted tree in which every node has either 0 or 2 children. One has 10 leaves. How many internal (non-leaf) nodes does it have? `[number]`

**q4.** How many different trees are there on the 5 labelled nodes $\{1, 2, 3, 4, 5\}$? (Two trees differ when some pair of nodes is joined in one and not in the other.) `[number]`

**q5.** A complete binary tree fills every level except possibly the last, and fills the last level from the left. One has $n = 100$ nodes. What is its height, the number of edges on its longest root-to-leaf path? `[number]`

**q6.** Prove that every tree with $n \ge 1$ nodes has exactly $n - 1$ edges. `[proof]`

### Inclusion-exclusion and birthday bounds

**q7.** Three sets have $\lvert A \rvert = \lvert B \rvert = \lvert C \rvert = 10$, every pairwise intersection has 3 elements, and $\lvert A \cap B \cap C \rvert = 1$. Give $\lvert A \cup B \cup C \rvert$. `[number]`

**q8.** How many integers in $\{1, 2, \dots, 100\}$ are divisible by 2, 3, or 5? `[number]`

**q9.** A derangement of $\{1, 2, 3, 4\}$ is a permutation that moves every element. How many are there? `[number]`

**q10.** Three keys are hashed independently and uniformly into $m = 10$ buckets. Give the exact probability that all three land in different buckets. `[number]` (exact)

**q11.** $k$ keys are hashed independently and uniformly into $m$ buckets. Give the expected number of unordered pairs of keys that share a bucket, as a formula in $k$ and $m$. `[expr]` (variables `k`, `m`)

**q12.** Using $P(\text{no collision among } k \text{ keys}) \approx e^{-k^2/(2m)}$, find the $k$ at which a collision becomes as likely as not, for a 32-bit hash ($m = 2^{32}$). `[number]` (to 3 significant digits)

### Jaccard, MinHash, and LSH

**q13.** Give the Jaccard similarity $J(A, B) = \lvert A \cap B \rvert / \lvert A \cup B \rvert$ of the shingle sets $A = \{ab, bc, cd, de\}$ and $B = \{bc, cd, de, ef, fg\}$. `[number]` (exact)

**q14.** Let $A$ and $B$ be non-empty subsets of a finite universe $U$ and $\pi$ a uniformly random permutation of $U$. Prove that $P(\min \pi(A) = \min \pi(B)) = J(A, B)$. (This is why one MinHash value per permutation is an unbiased estimate of Jaccard similarity.) `[proof]`

**q15.** A MinHash signature with 128 independent permutations estimates $J$ by the fraction of the 128 positions that agree. Each position agrees with probability $J$, independently. Give the variance of the estimate when $J = 1/2$. `[number]` (exact)

**q16.** LSH splits a signature into $b$ bands of $r$ rows; two documents become a candidate pair when all $r$ rows of at least one band agree. Each row agrees with probability $s$ (their Jaccard similarity), independently. Give the probability that they become candidates for $b = 16$, $r = 8$ (the `data.04` defaults), as a formula in $s$. `[expr]` (variable `s`)

**q17.** The S-curve of q16 rises most steeply near the threshold $t \approx (1/b)^{1/r}$. Give $t$ for $b = 16$, $r = 8$. `[number]` (to 3 significant digits)

**q18.** With $b = 16$, $r = 8$, give the probability that two documents with $s = 1/2$ become candidates. `[number]` (to 6 significant digits)

### Bloom filters

A Bloom filter has $m$ bits, holds $n$ keys, and sets $k$ bits per key with independent uniform hashes. A lookup of a key that was never inserted is a false positive when all $k$ of its bits are set.

**q19.** Give the standard approximation of the false-positive rate, using $(1 - 1/m)^{kn} \approx e^{-kn/m}$, as a formula in $k$, $n$, and $m$. `[expr]` (variables `k`, `n`, `m`)

**q20.** The rate of q19 is smallest at $k = (m/n) \ln 2$. Give this optimal $k$ for 10 bits per key ($m/n = 10$), before rounding to an integer. `[number]` (to 6 significant digits)

**q21.** At the optimal $k$ of q20 every bit is set with probability $1/2$. Give the false-positive rate at that $k$. `[number]` (to 6 significant digits)

**q22.** With the optimal $k$, the bits per key needed for a target false-positive rate $p$ are $m/n = -\ln p / (\ln 2)^2$. Give $m/n$ for $p = 0.01$. `[number]` (to 6 significant digits)

### Recurrences and generating functions

**q23.** Solve $T(n) = 2\,T(n/2) + n$ with $T(1) = 0$ for $n$ a power of 2. `[expr]` (variable `n`)

**q24.** Solve $T(n) = T(n - 1) + n$ with $T(0) = 0$. `[expr]` (variable `n`)

**q25.** Solve $a_n = 3a_{n-1} - 2a_{n-2}$ with $a_0 = 0$ and $a_1 = 1$. `[expr]` (variable `n`)

**q26.** Give the ordinary generating function $A(x) = \sum_{n \ge 0} a_n x^n$ of the sequence $a_n = n + 1$ (that is, $1, 2, 3, \dots$) in closed form. `[expr]` (variable `x`)

**q27.** How many binary strings of length 10 contain no two consecutive 1s? `[number]`

**q28.** Recursive matrix multiplication splits each $n \times n$ matrix into four $n/2 \times n/2$ blocks. (a) The plain method does 8 half-size products and $\Theta(n^2)$ additions: $T(n) = 8\,T(n/2) + n^2$. Give the exponent $c$ with $T(n) = \Theta(n^c)$. (b) Strassen's method does 7: $T(n) = 7\,T(n/2) + n^2$. Give $c$. `[number]`

**q29.** Prove by induction that $T(n) = 2\,T(n/2) + n$ with $T(1) = 0$ gives $T(n) = n \log_2 n$ for every $n$ that is a power of 2. `[proof]`
