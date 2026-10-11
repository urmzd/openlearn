<!-- ol:module S-M06b -->
# Solve set: trees, birthday bounds, Jaccard/MinHash/LSH S-curve, Bloom FP rate, recurrences

## Overview

| | |
|---|---|
| **Module** | `S-M06b` · solve · none · Pass 3 · 4 to 5 h |
| **You build** | answers in `solve/S-M06b.toml` (27 checked by SymPy) and 3 proofs in `solve/S-M06b/qN.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M06b/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M06b/problems.md` and in section 4 |
| **Needs** | reading: `S-M06a` (graphs, hashing, expected collisions), `M06.2` (tries), and the [Discrete Math 2 topic](README.md) |
| **Used by** | no call site (a solve set). Read before `ds.08` (sizing a Bloom filter: q19 to q22), `data.04` (MinHash and LSH bands: q13 to q18), and `L9.1` (the cost of tiled matmul: q28) |
| **Milestone** | `MS-P3` (the Pass 3 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Leskovec, Rajaraman, and Ullman, *Mining of Massive Datasets*, ch. 3 (MinHash, LSH); Broder and Mitzenmacher, "Network Applications of Bloom Filters: A Survey" (2004); Graham, Knuth, and Patashnik, *Concrete Mathematics*, ch. 7 (generating functions) |

## Key Takeaways

- A tree on $n$ nodes has $n - 1$ edges, and a trie stores each shared prefix once, so its size is one more than the number of distinct prefixes (q1, q2, q6).
- Collisions arrive at about $\sqrt{m}$ keys, not $m$: a 32-bit hash is likely to collide after about 77 000 keys (q11, q12).
- One random permutation's minimum agrees on two sets with probability exactly their Jaccard similarity, and banding $b \times r$ MinHash rows turns that into an S-shaped candidate probability with its threshold near $(1/b)^{1/r}$ (q14, q16 to q18).
- A Bloom filter with $m/n$ bits per key is best with $k = (m/n) \ln 2$ hashes, which sets half its bits and gives a false-positive rate of $2^{-k}$ (q20 to q22).
- Divide-and-conquer costs follow from their recurrences: $2T(n/2) + n$ is $n \log_2 n$, and 8 half-size products cost $n^3$ (q23, q28, q29).

## How to work this chapter

```bash
ol start S-M06b             # writes solve/S-M06b.toml and one file per proof
ol check S-M06b             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M06b --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

Pass 3 builds the data structures your corpus pipeline and tokenizers stand on, and every one of them has a number you must choose before it runs. The trie (`M06.2`) is a tree whose size you should be able to predict. The Bloom filter (`ds.08`) that `data.03` uses to drop exact duplicates needs a bit budget and a hash count; pick them by feel and the filter is ten times leakier than it needs to be, silently. The near-duplicate stage (`data.04`) splits 128 MinHash values into 16 bands of 8 rows, which decides at what similarity two documents get compared at all; the wrong split either compares everything or misses the near-copies. And the hash tables keyed by 32- or 64-bit hashes need to know when collisions start. None of these mistakes raises an error. This set gives you the formulas, derived, so the parameters in your code are calculations.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $n$ | the number of nodes of a tree, of keys in a filter, or the size of a recurrence's input | `int` |
| $\lvert A \rvert$ | the number of elements of a finite set $A$ | `int` |
| $m$ | the number of buckets of a hash, or of bits of a Bloom filter | `int` |
| $k$ | the number of keys hashed (birthday), or of hash functions per key (Bloom) | `int` |
| $J(A, B) = \lvert A \cap B \rvert / \lvert A \cup B \rvert$ | the Jaccard similarity of two non-empty sets | `float` in $[0, 1]$ |
| $\pi$ | a uniformly random permutation of the universe $U$ | permutation |
| $b$, $r$ | the LSH bands and rows per band, $b r$ = signature length | `int` |
| $s$ | the probability that one MinHash row agrees (the pair's Jaccard similarity) | `float` |
| $T(n)$ | the cost of a recursive algorithm on input size $n$ | function |
| $A(x) = \sum_{n \ge 0} a_n x^n$ | the ordinary generating function of a sequence $a_0, a_1, \dots$ | formal power series |

**Trees.** A tree is a connected graph with no cycle. Between two of its nodes there is exactly one path; it has exactly $n - 1$ edges (q6 asks for the proof); a tree with at least two nodes has at least two leaves. Rooting it makes every non-root node the child of exactly one parent. In a full binary tree every internal node has two children, so counting edges two ways, $2 \cdot \text{internal} = n - 1$ with $n = \text{internal} + \text{leaves}$, gives $\text{internal} = \text{leaves} - 1$. A complete binary tree of $n$ nodes is the shape of a binary heap (node $i$ has children $2i$ and $2i + 1$), and its height is $\lfloor \log_2 n \rfloor$. Cayley's formula counts the trees on $n$ labelled nodes: $n^{n-2}$.

**Inclusion-exclusion.** To count a union, add the sizes, subtract every pairwise overlap (counted twice), add back every triple overlap (subtracted once too often), and so on:

$$\lvert A \cup B \cup C \rvert = \lvert A \rvert + \lvert B \rvert + \lvert C \rvert - \lvert A \cap B \rvert - \lvert A \cap C \rvert - \lvert B \cap C \rvert + \lvert A \cap B \cap C \rvert.$$

The same alternating sum over the events "element $i$ is fixed" counts derangements.

**Birthday bounds.** Hash $k$ keys uniformly into $m$ buckets. The keys are all in different buckets with probability $\prod_{i=0}^{k-1}(1 - i/m)$: the $(i+1)$-th key must avoid the $i$ occupied buckets. Using $1 - x \le e^{-x}$, this is at most $e^{-k(k-1)/(2m)} \approx e^{-k^2/(2m)}$. Each of the $\binom{k}{2}$ pairs collides with probability $1/m$, so by linearity of expectation the expected number of colliding pairs is $\binom{k}{2}/m$, and by the union bound that is also an upper bound on the probability of any collision. All three say the same thing: collisions start at $k \approx \sqrt{m}$.

**Jaccard and MinHash.** Two documents are sets of shingles (overlapping substrings); their similarity is $J$. Apply a random permutation $\pi$ to the universe and keep the minimum of each set. Among $A \cup B$, the element with the smallest $\pi$ value is equally likely to be any of the $\lvert A \cup B \rvert$ elements; the two minima agree exactly when it lies in $A \cap B$. So $P(\min \pi(A) = \min \pi(B)) = J(A, B)$ (q14). With $N$ independent permutations, the fraction that agree is an unbiased estimate of $J$ with variance $J(1 - J)/N$ (a mean of Bernoulli variables). `data.04` stands in for permutations with seeded hash functions.

**LSH banding.** Comparing every pair of $10^6$ documents is $5 \times 10^{11}$ comparisons. Instead, split each signature into $b$ bands of $r$ rows and hash each band; two documents become candidates when they agree on every row of some band. With per-row agreement $s$: one band agrees with probability $s^r$, it fails with $1 - s^r$, all $b$ fail with $(1 - s^r)^b$, so

$$P(\text{candidate}) = 1 - (1 - s^r)^b.$$

As a function of $s$ this is an S-curve: near 0 for dissimilar pairs, near 1 for similar ones, rising most steeply around $t \approx (1/b)^{1/r}$. More rows per band push the threshold up and sharpen the curve; more bands pull it down.

**Bloom filters.** A Bloom filter is $m$ bits, all 0. Inserting a key sets the $k$ bits its $k$ hashes point to. A lookup reports "present" when all $k$ of its bits are set, so it never misses an inserted key, and it reports a false positive when an absent key finds all its bits set by others. After $n$ insertions one bit is still 0 with probability $(1 - 1/m)^{kn} \approx e^{-kn/m}$. Treating the $k$ probed bits as independent (an approximation, accurate for large $m$), the false-positive rate is

$$f(k) = \big(1 - e^{-kn/m}\big)^k.$$

Write $p = e^{-kn/m}$ for the fraction of zero bits, so $k = -(m/n) \ln p$ and $\ln f = -(m/n) \ln p \, \ln(1 - p)$. That is symmetric under $p \leftrightarrow 1 - p$ and is smallest at $p = 1/2$: the best filter has half its bits set, $k = (m/n) \ln 2$, and $f = 2^{-k}$. Solving $2^{-k} = p_{\text{target}}$ for the bits per key gives $m/n = -\ln p_{\text{target}} / (\ln 2)^2 \approx 1.44 \log_2(1/p_{\text{target}})$.

**Recurrences.** A recurrence defines a sequence by earlier terms. Three ways to solve one:

1. *Unroll and sum.* $T(n) = T(n-1) + n$ unrolls to $1 + 2 + \dots + n$. For $T(n) = 2T(n/2) + n$ each of the $\log_2 n$ levels of the recursion costs $n$ in total.
2. *Characteristic equation.* For $a_n = c_1 a_{n-1} + c_2 a_{n-2}$ try $a_n = \lambda^n$: then $\lambda^2 = c_1 \lambda + c_2$. With distinct roots $\lambda_1, \lambda_2$ the general solution is $\alpha \lambda_1^n + \beta \lambda_2^n$, and the two initial values fix $\alpha$ and $\beta$.
3. *Generating functions.* Multiply the recurrence by $x^n$ and sum, and it becomes an equation for $A(x)$. The basic pair is $\sum_n x^n = 1/(1 - x)$, and differentiating it gives $\sum_n (n + 1) x^n = 1/(1-x)^2$.

For divide-and-conquer costs $T(n) = a\,T(n/b) + n^d$, compare $\log_b a$ with $d$: the leaves dominate when $\log_b a > d$, giving $T(n) = \Theta(n^{\log_b a})$, which is why 8 half-size matrix products cost $n^3$ and Strassen's 7 cost $n^{\log_2 7}$.

## 3. Worked example by hand

These are siblings of the graded problems, not answers to them.

**Size a Bloom filter at 8 bits per key.** The optimal number of hashes is $k^* = (m/n) \ln 2 = 8 \times 0.6931 = 5.545$. A filter needs an integer, so compare the neighbours with $f(k) = (1 - e^{-k/8})^k$:

| $k$ | $e^{-k/8}$ | $1 - e^{-k/8}$ | $f(k)$ |
|---|---|---|---|
| 5 | 0.5353 | 0.4647 | 0.02168 |
| 6 | 0.4724 | 0.5276 | 0.02158 |

$k = 6$ is (barely) better, giving about 2.16% false positives. At the real-valued optimum the rate would be $2^{-5.545} = 0.0214$. For 1% you would need $m/n = \ln 100 / (\ln 2)^2 = 9.59$ bits per key.

**Read an LSH S-curve with 20 bands of 5 rows.** The threshold is $(1/20)^{1/5} = 0.549$. At $s = 0.8$: $s^5 = 0.32768$, a band fails with $0.67232$, all 20 fail with $0.67232^{20} = 0.000356$, so the pair becomes a candidate with probability $0.99964$. At $s = 0.3$: $s^5 = 0.00243$, $(1 - 0.00243)^{20} = 0.9525$, so only $0.0475$. Similar pairs are almost always compared and dissimilar ones rarely.

**A model proof.** *Claim:* a full binary tree with $L \ge 1$ leaves has $L - 1$ internal nodes. *Method: strong induction on $L$.* *Base:* $L = 1$ is a single node, which is a leaf, with 0 internal nodes. *Step:* let $L \ge 2$ and assume the claim for every full binary tree with fewer than $L$ leaves. The root is internal (a leaf root would mean $L = 1$), so it has two subtrees, each a full binary tree, with $L_1 \ge 1$ and $L_2 \ge 1$ leaves, $L_1 + L_2 = L$, so each has fewer than $L$. By the hypothesis they have $L_1 - 1$ and $L_2 - 1$ internal nodes. Adding the root: $(L_1 - 1) + (L_2 - 1) + 1 = L - 1$. *So* every full binary tree with $L$ leaves has $L - 1$ internal nodes. Every symbol is defined, the hypothesis is used for the subtrees and not for the tree itself, and the last line restates the claim: that is what the rubrics of q6 and q29 ask for.

## 4. The problem set

Write each answer in `solve/S-M06b.toml`; lettered parts are their own tables:

```toml
[q11]
answer = "k*(k-1)/(2*m)"
[q28.a]
answer = "3"
```

<!-- ol:problems S-M06b -->

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

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Counting a trie as one node per character | sizes a node arena for the worst case, ignoring shared prefixes | q2 (canary 18) |
| Forgetting to add back the triple overlap | a union undercounted by every element in all three sets | q7 (canary 21), q8 (canary 71) |
| Reading the union bound as the probability | "1 - 3/10" instead of the exact product | q10 (canary 7/10), q18 (canary 16/256) |
| Expecting collisions only near a full table | a 32-bit-keyed cache that collides after 77 000 entries, not 4 billion | q12 (canaries 2^16 and 2^31) |
| Dividing the overlap by one set's size | a similarity that is not symmetric and overstates small documents | q13 (canaries 3/5 and 3/4) |
| Swapping bands and rows, or AND and OR | the S-curve's threshold lands far from the intended similarity | q16 (canaries) |
| One hash per Bloom key, or a single probe | a false-positive rate formula off by orders of magnitude | q19 (canaries) |
| Confusing $k$ with bits per key | a filter sized at 6.6 bits per key that misses its 1% target | q22 (canary log(100)/log(2)) |
| Natural log where $\log_2$ is meant | costs off by a factor of $\ln 2$ | q23 (canary n*log(n)) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M06a` | expected colliding pairs (its q17) and modular hashing, extended here to birthday bounds |
| Back | `M06.2` | the trie whose node count q2 predicts |
| Forward | `ds.08` | `Bloom::with_rate(n, p)` computes $m$ and $k$ with q20 to q22 |
| Forward | `data.04` | MinHash signatures of 128 values in 16 bands of 8 rows: q15 to q18 |
| Forward | `L9.1` | tiled matmul's cost and the roofline use the recurrence of q28 |
