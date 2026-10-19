# Solve set: trees, birthday bounds, Jaccard/MinHash/LSH S-curve, Bloom FP rate, recurrences

## Overview

| | |
|---|---|
| **Module** | `S-M06b` · solve · none · Pass 3 · 4 to 5 h |
| **Solution** | a worked answer after every problem in section 4: 26 problems with answers (27 parts) and 3 proofs (model proofs also in [`course/solutions/solve/S-M06b/`](../../../../course/solutions/solve/S-M06b/)) |
| **Contract** | none: a pen and paper set |
| **Needs** | reading: `S-M06a` (graphs, hashing, expected collisions), `M06.2` (tries), and the [Discrete Math 2 topic](README.md) |
| **Used by** | no call site (a solve set). Read before `ds.08` (sizing a Bloom filter: q19 to q22), `data.04` (MinHash and LSH bands: q13 to q18), and `L9.1` (the cost of tiled matmul: q28) |
| **Optional depth** | Leskovec, Rajaraman, and Ullman, *Mining of Massive Datasets*, ch. 3 (MinHash, LSH); Broder and Mitzenmacher, "Network Applications of Bloom Filters: A Survey" (2004); Graham, Knuth, and Patashnik, *Concrete Mathematics*, ch. 7 (generating functions) |

## Key Takeaways

- A tree on $n$ nodes has $n - 1$ edges, and a trie stores each shared prefix once, so its size is one more than the number of distinct prefixes (q1, q2, q6).
- Collisions arrive at about $\sqrt{m}$ keys, not $m$: a 32-bit hash is likely to collide after about 77 000 keys (q11, q12).
- One random permutation's minimum agrees on two sets with probability exactly their Jaccard similarity, and banding $b \times r$ MinHash rows turns that into an S-shaped candidate probability with its threshold near $(1/b)^{1/r}$ (q14, q16 to q18).
- A Bloom filter with $m/n$ bits per key is best with $k = (m/n) \ln 2$ hashes, which sets half its bits and gives a false-positive rate of $2^{-k}$ (q20 to q22).
- Divide-and-conquer costs follow from their recurrences: $2T(n/2) + n$ is $n \log_2 n$, and 8 half-size products cost $n^3$ (q23, q28, q29).

## How to read this chapter

Sections 1 to 3 derive the formulas and work a Bloom filter, an LSH curve, and a proof by hand. Section 4 is the problem set: every problem is followed by a collapsed worked answer with the final result or a complete model proof, and the common wrong answers and why they are wrong. Try each problem before opening its answer.

---

## 1. Why now

Pass 3 builds the data structures the corpus pipeline and tokenizers stand on, and every one of them has a number that must be chosen before it runs. The trie (`M06.2`) is a tree whose size can be predicted. The Bloom filter (`ds.08`) that `data.03` uses to drop exact duplicates needs a bit budget and a hash count; pick them by feel and the filter is ten times leakier than it needs to be, silently. The near-duplicate stage (`data.04`) splits 128 MinHash values into 16 bands of 8 rows, which decides at what similarity two documents get compared at all; the wrong split either compares everything or misses the near-copies. And the hash tables keyed by 32- or 64-bit hashes need to know when collisions start. None of these mistakes raises an error. This set derives the formulas, so the parameters in the solution's code are calculations.

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

These are siblings of the problems in section 4, not answers to them.

**Size a Bloom filter at 8 bits per key.** The optimal number of hashes is $k^* = (m/n) \ln 2 = 8 \times 0.6931 = 5.545$. A filter needs an integer, so compare the neighbours with $f(k) = (1 - e^{-k/8})^k$:

| $k$ | $e^{-k/8}$ | $1 - e^{-k/8}$ | $f(k)$ |
|---|---|---|---|
| 5 | 0.5353 | 0.4647 | 0.02168 |
| 6 | 0.4724 | 0.5276 | 0.02158 |

$k = 6$ is (barely) better, giving about 2.16% false positives. At the real-valued optimum the rate would be $2^{-5.545} = 0.0214$. For 1% one needs $m/n = \ln 100 / (\ln 2)^2 = 9.59$ bits per key.

**Read an LSH S-curve with 20 bands of 5 rows.** The threshold is $(1/20)^{1/5} = 0.549$. At $s = 0.8$: $s^5 = 0.32768$, a band fails with $0.67232$, all 20 fail with $0.67232^{20} = 0.000356$, so the pair becomes a candidate with probability $0.99964$. At $s = 0.3$: $s^5 = 0.00243$, $(1 - 0.00243)^{20} = 0.9525$, so only $0.0475$. Similar pairs are almost always compared and dissimilar ones rarely.

**A model proof.** *Claim:* a full binary tree with $L \ge 1$ leaves has $L - 1$ internal nodes. *Method: strong induction on $L$.* *Base:* $L = 1$ is a single node, which is a leaf, with 0 internal nodes. *Step:* let $L \ge 2$ and assume the claim for every full binary tree with fewer than $L$ leaves. The root is internal (a leaf root would mean $L = 1$), so it has two subtrees, each a full binary tree, with $L_1 \ge 1$ and $L_2 \ge 1$ leaves, $L_1 + L_2 = L$, so each has fewer than $L$. By the hypothesis they have $L_1 - 1$ and $L_2 - 1$ internal nodes. Adding the root: $(L_1 - 1) + (L_2 - 1) + 1 = L - 1$. *So* every full binary tree with $L$ leaves has $L - 1$ internal nodes. Every symbol is defined, the hypothesis is used for the subtrees and not for the tree itself, and the last line restates the claim: the model proofs of q6 and q29 in section 4 have the same shape.

## 4. The problem set, with worked answers

Answers are exact where the problem says so; otherwise a closed form is given together with a decimal to the stated precision. `log` means the natural logarithm, and $\log_2 x = \ln x / \ln 2$. Lettered parts share one answer block. The tag after each problem says what kind of answer is expected.

### Trees

**q1.** A tree has 12 nodes. How many edges does it have? `[number]`

<details><summary>Answer</summary>

**$11$.** A tree on $n$ nodes has $n - 1$ edges (q6): every non-root node has exactly one edge to its parent. This is how a parent array or a trie arena sizes itself. Common mistakes: 12 gives every node an edge, including the root; 66 $= \binom{12}{2}$ is the complete graph.

</details>

**q2.** Build the trie (`M06.2`) of the five keys `car`, `cart`, `care`, `cat`, `dog`. How many nodes does it have, counting the root? `[number]`

<details><summary>Answer</summary>

**$10$.** Count distinct prefixes plus the root. Under `c`: `c`, `ca`, `car`, then `cart`, `care`, and `cat` (6 nodes). Under `d`: `d`, `do`, `dog` (3 nodes). With the root: $1 + 6 + 3 = 10$. Shared prefixes are stored once, which is what makes a trie small. Common mistakes: 18 is 1 plus the 17 characters of the keys, with no sharing; 9 forgets the root.

</details>

**q3.** A full binary tree is a rooted tree in which every node has either 0 or 2 children. One has 10 leaves. How many internal (non-leaf) nodes does it have? `[number]`

<details><summary>Answer</summary>

**$9$.** In a full binary tree, internal $=$ leaves $- 1$ (the model proof in section 3). Counting edges two ways: $2I = n - 1 = I + L - 1$, so $I = L - 1 = 9$. A BPE merge tree and a Huffman tree have this shape. Common mistakes: 10, and 19, which is the total number of nodes.

</details>

**q4.** How many different trees are there on the 5 labelled nodes $\{1, 2, 3, 4, 5\}$? (Two trees differ when some pair of nodes is joined in one and not in the other.) `[number]`

<details><summary>Answer</summary>

**$125$.** Cayley's formula, $n^{n-2} = 5^3 = 125$. Common mistakes: $5! = 120$ counts orderings, and $5^4 = 625$ is $n^{n-1}$, the count of rooted trees.

</details>

**q5.** A complete binary tree fills every level except possibly the last, and fills the last level from the left. One has $n = 100$ nodes. What is its height, the number of edges on its longest root-to-leaf path? `[number]`

<details><summary>Answer</summary>

**$6$.** The height of a complete binary tree with $n$ nodes is $\lfloor \log_2 n \rfloor$: levels 0 to 5 hold $1 + 2 + \dots + 32 = 63$ nodes, and the remaining 37 sit on level 6. A common mistake is 7, which counts levels instead of edges.

</details>

**q6.** Prove that every tree with $n \ge 1$ nodes has exactly $n - 1$ edges. `[proof]`

<details><summary>Answer</summary>

**Claim.** A tree is a connected graph with no cycle. For every $n \ge 1$, every tree with $n$ nodes has exactly $n - 1$ edges.

**Lemma (a leaf exists).** A tree $T$ with $n \ge 2$ nodes has a node of degree 1. Since $T$ is connected and has at least two nodes, it has an edge, so paths of length at least 1 exist. Take a longest path $v_0 v_1 \dots v_\ell$ (paths have no repeated nodes, so there are finitely many). The end $v_0$ is adjacent to $v_1$. If $v_0$ had another neighbour $w$, then either $w = v_i$ for some $i \ge 2$, and $v_0 v_1 \dots v_i v_0$ is a cycle, which a tree does not have; or $w$ is off the path, and $w v_0 v_1 \dots v_\ell$ is a longer path, contradicting the choice. So $v_0$ has degree 1.

**Lemma (removing a leaf leaves a tree).** Let $v$ be a leaf of $T$ with its one edge $vu$, and let $T' = T - v$. $T'$ has no cycle, because every cycle of $T'$ is a cycle of $T$. $T'$ is connected: for nodes $x, y \ne v$, the path from $x$ to $y$ in $T$ cannot pass through $v$, since an interior node of a path needs two distinct path neighbours and $v$ has only one. So that path lies in $T'$.

**Proof (induction on $n$).** Base case $n = 1$: a single node has no edge, since an edge would need a second endpoint (and a loop is a cycle). So it has $0 = n - 1$ edges.

Inductive step: let $n \ge 2$ and assume every tree with $n - 1$ nodes has $n - 2$ edges. Let $T$ be a tree with $n$ nodes. By the first lemma it has a leaf $v$; by the second, $T' = T - v$ is a tree with $n - 1$ nodes, and it has exactly one edge fewer than $T$ (the edge $vu$). The induction hypothesis applied to $T'$ gives $n - 2$ edges for $T'$, so $T$ has $(n - 2) + 1 = n - 1$ edges. By induction the claim holds for every $n \ge 1$.

Model proof also in [`q6.md`](../../../../course/solutions/solve/S-M06b/q6.md).

</details>

### Inclusion-exclusion and birthday bounds

**q7.** Three sets have $\lvert A \rvert = \lvert B \rvert = \lvert C \rvert = 10$, every pairwise intersection has 3 elements, and $\lvert A \cap B \cap C \rvert = 1$. Give $\lvert A \cup B \cup C \rvert$. `[number]`

<details><summary>Answer</summary>

**$22$.** $10 + 10 + 10 - 3 - 3 - 3 + 1 = 22$. A common mistake is 21, which forgets to add the triple intersection back: an element in all three sets is added 3 times and subtracted 3 times, so it must be added once more. 30 makes no correction at all.

</details>

**q8.** How many integers in $\{1, 2, \dots, 100\}$ are divisible by 2, 3, or 5? `[number]`

<details><summary>Answer</summary>

**$74$.** Let $D_k$ be the multiples of $k$. $|D_2| + |D_3| + |D_5| = 50 + 33 + 20 = 103$; pairwise $|D_6| + |D_{10}| + |D_{15}| = 16 + 10 + 6 = 32$; triple $|D_{30}| = 3$. Total $103 - 32 + 3 = 74$. Common mistakes: 103 is the plain sum, and 71 forgets to add $|D_{30}|$ back.

</details>

**q9.** A derangement of $\{1, 2, 3, 4\}$ is a permutation that moves every element. How many are there? `[number]`

<details><summary>Answer</summary>

**$9$.** Inclusion-exclusion over the events "element $i$ is fixed": $4! - \binom41 3! + \binom42 2! - \binom43 1! + \binom44 0! = 24 - 24 + 12 - 4 + 1 = 9$. Common mistakes: 24 is every permutation; 8 drops the last term.

</details>

**q10.** Three keys are hashed independently and uniformly into $m = 10$ buckets. Give the exact probability that all three land in different buckets. `[number]` (exact)

<details><summary>Answer</summary>

**$18/25$.** The second key must avoid one bucket and the third two: $1 \cdot \frac{9}{10} \cdot \frac{8}{10} = \frac{72}{100} = \frac{18}{25}$. Common mistakes: $7/10 = 1 - 3/10$ subtracts the union bound on a collision, which is only an upper bound; $7/25$ is the complement, the probability of at least one collision.

</details>

**q11.** $k$ keys are hashed independently and uniformly into $m$ buckets. Give the expected number of unordered pairs of keys that share a bucket, as a formula in $k$ and $m$. `[expr]` (variables `k`, `m`)

<details><summary>Answer</summary>

**$\frac{k(k - 1)}{2m}$.** There are $\binom{k}{2}$ unordered pairs, each colliding with probability $1/m$, and expectation is linear. Common mistakes: $k^2/m$ and $k(k-1)/m$ count ordered pairs (and the first also self-pairs).

</details>

**q12.** Using $P(\text{no collision among } k \text{ keys}) \approx e^{-k^2/(2m)}$, find the $k$ at which a collision becomes as likely as not, for a 32-bit hash ($m = 2^{32}$). `[number]` (to 3 significant digits)

<details><summary>Answer</summary>

**$\approx 77{,}200$** ($2^{16}\sqrt{2\ln 2} \approx 77{,}163$). Set $e^{-k^2/(2m)} = \frac12$: $k^2 = 2m\ln 2$, $k = \sqrt{2 \ln 2 \cdot 2^{32}} = 2^{16}\sqrt{2\ln 2}$. So a 32-bit hash collides after about 77,000 keys, not 4 billion. Common mistakes: $2^{16} = \sqrt m$ drops the constant $\sqrt{2\ln 2} \approx 1.18$, and $2^{31}$ (half the table) is the intuition the birthday bound refutes.

</details>

### Jaccard, MinHash, and LSH

**q13.** Give the Jaccard similarity $J(A, B) = \lvert A \cap B \rvert / \lvert A \cup B \rvert$ of the shingle sets $A = \{ab, bc, cd, de\}$ and $B = \{bc, cd, de, ef, fg\}$. `[number]` (exact)

<details><summary>Answer</summary>

**$1/2$.** $A \cap B = \{bc, cd, de\}$ (3 elements), $A \cup B = \{ab, bc, cd, de, ef, fg\}$ (6), so $J = 3/6$. Common mistakes: $3/5$ divides by $|B|$ and $3/4$ by $|A|$; neither is symmetric in $A$ and $B$.

</details>

**q14.** Let $A$ and $B$ be non-empty subsets of a finite universe $U$ and $\pi$ a uniformly random permutation of $U$. Prove that $P(\min \pi(A) = \min \pi(B)) = J(A, B)$. (This is why one MinHash value per permutation is an unbiased estimate of Jaccard similarity.) `[proof]`

<details><summary>Answer</summary>

**Claim.** Let $A, B$ be non-empty subsets of a finite universe $U$ and $\pi$ a uniformly random permutation of $U$ (a bijection $U \to \{1, \dots, |U|\}$). Then $P(\min \pi(A) = \min \pi(B)) = |A \cap B| / |A \cup B|$, which is the Jaccard similarity $J(A, B)$.

**Proof.** Let $m$ be the element of $A \cup B$ with the smallest value $\pi(m)$; it is well defined because $A \cup B$ is non-empty and $\pi$ is injective. For a uniform $\pi$ every ordering of $A \cup B$ is equally likely, and by symmetry each of the $|A \cup B|$ elements is equally likely to be first. So $m$ is uniformly distributed over $A \cup B$.

Now compare the two minima. $\min \pi(A) = \pi(m)$ when $m \in A$, because no element of $A \subseteq A \cup B$ has a smaller value; likewise $\min \pi(B) = \pi(m)$ when $m \in B$.

- If $m \in A \cap B$, both minima equal $\pi(m)$, so they are equal.
- If $m \in A \setminus B$, then $\min \pi(A) = \pi(m)$, while every element of $B$ differs from $m$ and has a larger value, so $\min \pi(B) > \pi(m)$ and the minima differ. The case $m \in B \setminus A$ is symmetric.

So the minima are equal exactly when $m \in A \cap B$, and

$$P(\min \pi(A) = \min \pi(B)) = P(m \in A \cap B) = \frac{|A \cap B|}{|A \cup B|} = J(A, B).$$

Hence the indicator that one MinHash position agrees has mean $J(A, B)$: it is an unbiased estimate of the Jaccard similarity.

Model proof also in [`q14.md`](../../../../course/solutions/solve/S-M06b/q14.md).

</details>

**q15.** A MinHash signature with 128 independent permutations estimates $J$ by the fraction of the 128 positions that agree. Each position agrees with probability $J$, independently. Give the variance of the estimate when $J = 1/2$. `[number]` (exact)

<details><summary>Answer</summary>

**$1/512$.** The estimate is a mean of 128 independent Bernoulli($J$) variables, so its variance is $J(1 - J)/128 = \frac{1/4}{128} = \frac{1}{512}$, a standard deviation of about 0.044. Common mistakes: $1/256$, and $1/128$, which forgets the factor $J(1 - J)$.

</details>

**q16.** LSH splits a signature into $b$ bands of $r$ rows; two documents become a candidate pair when all $r$ rows of at least one band agree. Each row agrees with probability $s$ (their Jaccard similarity), independently. Give the probability that they become candidates for $b = 16$, $r = 8$ (the `data.04` defaults), as a formula in $s$. `[expr]` (variable `s`)

<details><summary>Answer</summary>

**$1 - (1 - s^8)^{16}$.** One band (8 rows) agrees with probability $s^8$, fails with $1 - s^8$; all 16 bands fail with $(1 - s^8)^{16}$; the pair is a candidate unless they all fail. Common mistakes: $1 - (1 - s^{16})^8$ swaps bands and rows, and $(1 - (1 - s)^8)^{16}$ swaps AND and OR (it would require every band to have at least one agreeing row).

</details>

**q17.** The S-curve of q16 rises most steeply near the threshold $t \approx (1/b)^{1/r}$. Give $t$ for $b = 16$, $r = 8$. `[number]` (to 3 significant digits)

<details><summary>Answer</summary>

**$\approx 0.707$** ($2^{-1/2}$). $(1/16)^{1/8} = 2^{-4/8} = 2^{-1/2} \approx 0.7071$. So with the `data.04` defaults, pairs above about 0.7 similarity are likely to be compared. Common mistakes: $1/16$ forgets the $r$-th root, and $1/2$ guesses.

</details>

**q18.** With $b = 16$, $r = 8$, give the probability that two documents with $s = 1/2$ become candidates. `[number]` (to 6 significant digits)

<details><summary>Answer</summary>

**$\approx 0.0607019$** ($1 - (255/256)^{16}$). $s^8 = 1/256$, a band fails with $255/256$, so the pair is a candidate with probability $1 - (255/256)^{16} \approx 0.0607019$. Common mistakes: $16/256 = 0.0625$ is the union bound over bands, close but not the probability; $1/256$ is one band.

</details>

### Bloom filters

A Bloom filter has $m$ bits, holds $n$ keys, and sets $k$ bits per key with independent uniform hashes. A lookup of a key that was never inserted is a false positive when all $k$ of its bits are set.

**q19.** Give the standard approximation of the false-positive rate, using $(1 - 1/m)^{kn} \approx e^{-kn/m}$, as a formula in $k$, $n$, and $m$. `[expr]` (variables `k`, `n`, `m`)

<details><summary>Answer</summary>

**$\left(1 - e^{-kn/m}\right)^k$.** After $kn$ bit-settings, one bit is still 0 with probability $(1 - 1/m)^{kn} \approx e^{-kn/m}$, so set with $1 - e^{-kn/m}$. A false positive needs all $k$ probed bits set: raise to the $k$-th power. Common mistakes: $(1 - e^{-n/m})^k$ sets only one bit per key, and $1 - e^{-kn/m}$ probes one bit.

</details>

**q20.** The rate of q19 is smallest at $k = (m/n) \ln 2$. Give this optimal $k$ for 10 bits per key ($m/n = 10$), before rounding to an integer. `[number]` (to 6 significant digits)

<details><summary>Answer</summary>

**$\approx 6.93147$** ($10 \ln 2$). Directly from $k^* = (m/n) \ln 2$. Common mistakes: $10/\ln 2 \approx 14.4$ inverts the factor, and 7 rounds before it was asked (a real filter does round, and compares the neighbours as in section 3).

</details>

**q21.** At the optimal $k$ of q20 every bit is set with probability $1/2$. Give the false-positive rate at that $k$. `[number]` (to 6 significant digits)

<details><summary>Answer</summary>

**$\approx 0.00819255$** ($2^{-10 \ln 2}$). With half the bits set, each probe hits a set bit with probability $1/2$, so $f = 2^{-k^*} = 2^{-6.93147} \approx 0.0082$, about 0.82%. Common mistakes: $1/1024 = 2^{-10}$ uses bits per key as the number of hashes, and $(1/2)^7$ rounds $k$ first.

</details>

**q22.** With the optimal $k$, the bits per key needed for a target false-positive rate $p$ are $m/n = -\ln p / (\ln 2)^2$. Give $m/n$ for $p = 0.01$. `[number]` (to 6 significant digits)

<details><summary>Answer</summary>

**$\approx 9.58506$** ($\ln 100 / (\ln 2)^2$). $-\ln 0.01 = \ln 100 \approx 4.6052$, and $(\ln 2)^2 \approx 0.48045$. So 1% needs about 9.6 bits per key, with $k = 9.585 \cdot \ln 2 \approx 6.6$ hashes. Common mistakes: $\ln 100 / \ln 2 \approx 6.64$ is $k$, not bits per key, and a filter sized at 6.6 bits per key misses its target; 100 confuses the rate with its reciprocal.

</details>

### Recurrences and generating functions

**q23.** Solve $T(n) = 2\,T(n/2) + n$ with $T(1) = 0$ for $n$ a power of 2. `[expr]` (variable `n`)

<details><summary>Answer</summary>

**$n \log_2 n$.** Unroll: the recursion has $\log_2 n$ levels, and level $i$ has $2^i$ subproblems of size $n/2^i$, each costing $n/2^i$, so every level costs $n$. The leaves cost $T(1) = 0$. q29 proves it by induction. Common mistakes: $n \ln n$ uses the natural log, off by a factor $\ln 2$; $n^2$ ignores the halving.

</details>

**q24.** Solve $T(n) = T(n - 1) + n$ with $T(0) = 0$. `[expr]` (variable `n`)

<details><summary>Answer</summary>

**$\frac{n(n + 1)}{2}$.** Unroll: $T(n) = n + (n - 1) + \dots + 1 + T(0)$. Common mistakes: $n^2/2$ keeps only the leading term, and $n(n - 1)/2$ is off by one (it stops at $n - 1$).

</details>

**q25.** Solve $a_n = 3a_{n-1} - 2a_{n-2}$ with $a_0 = 0$ and $a_1 = 1$. `[expr]` (variable `n`)

<details><summary>Answer</summary>

**$2^n - 1$.** Characteristic equation $\lambda^2 = 3\lambda - 2$, roots 1 and 2, so $a_n = \alpha + \beta 2^n$. $a_0 = 0$ gives $\alpha = -\beta$; $a_1 = 1$ gives $-\beta + 2\beta = 1$, so $\beta = 1$, $\alpha = -1$. Check: $a_2 = 3 - 0 = 3 = 2^2 - 1$. Common mistakes: $2^n$ and $2^{n-1}$ ignore the initial values.

</details>

**q26.** Give the ordinary generating function $A(x) = \sum_{n \ge 0} a_n x^n$ of the sequence $a_n = n + 1$ (that is, $1, 2, 3, \dots$) in closed form. `[expr]` (variable `x`)

<details><summary>Answer</summary>

**$\frac{1}{(1 - x)^2}$.** Differentiate $\sum_{n \ge 0} x^n = \frac{1}{1 - x}$: $\sum_{n \ge 1} n x^{n-1} = \sum_{n \ge 0} (n + 1) x^n = \frac{1}{(1-x)^2}$. Common mistakes: $\frac{1}{1-x}$ is the sequence of all 1s, and $\frac{x}{(1-x)^2}$ generates $a_n = n$, not $n + 1$.

</details>

**q27.** How many binary strings of length 10 contain no two consecutive 1s? `[number]`

<details><summary>Answer</summary>

**$144$.** Let $c_n$ count the valid strings of length $n$. A valid string ends in 0 (anything valid of length $n - 1$ before it) or in 01 (anything valid of length $n - 2$): $c_n = c_{n-1} + c_{n-2}$, with $c_1 = 2$, $c_2 = 3$. So $c_n = F_{n+2}$: $2, 3, 5, 8, 13, 21, 34, 55, 89, 144$. Common mistakes: 89 is $c_9$, off by one; 1024 counts every string.

</details>

**q28.** Recursive matrix multiplication splits each $n \times n$ matrix into four $n/2 \times n/2$ blocks. (a) The plain method does 8 half-size products and $\Theta(n^2)$ additions: $T(n) = 8\,T(n/2) + n^2$. Give the exponent $c$ with $T(n) = \Theta(n^c)$. (b) Strassen's method does 7: $T(n) = 7\,T(n/2) + n^2$. Give $c$. `[number]`

<details><summary>Answer</summary>

**(a) $3$, (b) $\log_2 7 \approx 2.807$.** Compare $\log_b a$ with $d = 2$. (a) $\log_2 8 = 3 > 2$: the leaves dominate, $T(n) = \Theta(n^3)$. (b) $\log_2 7 \approx 2.807 > 2$: $\Theta(n^{\log_2 7})$. Common mistakes: 2 for (a) reads the cost of the additions; 3 for (b) ignores that one fewer product changes the exponent; $\ln 7$ uses the wrong base.

</details>

**q29.** Prove by induction that $T(n) = 2\,T(n/2) + n$ with $T(1) = 0$ gives $T(n) = n \log_2 n$ for every $n$ that is a power of 2. `[proof]`

<details><summary>Answer</summary>

**Claim.** Let $T(1) = 0$ and $T(n) = 2T(n/2) + n$ for $n \ge 2$ a power of 2. Writing $n = 2^j$, we prove $T(2^j) = 2^j \cdot j$ for every integer $j \ge 0$, which is $T(n) = n \log_2 n$ for every power of 2.

**Base case $j = 0$.** $T(2^0) = T(1) = 0 = 1 \cdot 0 = 2^0 \cdot 0$.

**Inductive step.** Assume $T(2^j) = 2^j \cdot j$ for some $j \ge 0$. Since $2^{j+1} \ge 2$, the recurrence applies with $n = 2^{j+1}$ and $n/2 = 2^j$:

$$T(2^{j+1}) = 2\,T(2^j) + 2^{j+1}.$$

By the hypothesis $T(2^j) = 2^j j$, so

$$T(2^{j+1}) = 2 \cdot 2^j \cdot j + 2^{j+1} = 2^{j+1} j + 2^{j+1} = 2^{j+1}(j + 1).$$

This is the claim for $j + 1$. By induction $T(2^j) = 2^j j$ for all $j \ge 0$, and with $j = \log_2 n$ this reads $T(n) = n \log_2 n$.

Model proof also in [`q29.md`](../../../../course/solutions/solve/S-M06b/q29.md).

</details>

## 5. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Counting a trie as one node per character | sizes a node arena for the worst case, ignoring shared prefixes | counts distinct prefixes plus the root (q2) |
| Forgetting to add back the triple overlap | a union undercounted by every element in all three sets | alternates signs through every intersection size, including the triple (q7, q8) |
| Reading the union bound as the probability | "1 - 3/10" instead of the exact product | multiplies the exact per-step probabilities; uses the union bound only as a bound (q10, q18) |
| Expecting collisions only near a full table | a 32-bit-keyed cache that collides after 77 000 entries, not 4 billion | solves $e^{-k^2/(2m)} = 1/2$: $k \approx 1.18\sqrt m$ (q12) |
| Dividing the overlap by one set's size | a similarity that is not symmetric and overstates small documents | divides by the union, which is symmetric (q13) |
| Swapping bands and rows, or AND and OR | the S-curve's threshold lands far from the intended similarity | AND within a band ($s^r$), OR across bands ($1 - (1 - s^r)^b$) (q16) |
| One hash per Bloom key, or a single probe | a false-positive rate formula off by orders of magnitude | counts $kn$ bit-settings and $k$ probes (q19) |
| Confusing $k$ with bits per key | a filter sized at 6.6 bits per key that misses its 1% target | keeps $m/n = -\ln p/(\ln 2)^2$ apart from $k = (m/n)\ln 2$ (q22) |
| Natural log where $\log_2$ is meant | costs off by a factor of $\ln 2$ | writes $\log_2$ explicitly, since the recursion halves (q23) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M06a` | expected colliding pairs (its q17) and modular hashing, extended here to birthday bounds |
| Back | `M06.2` | the trie whose node count q2 predicts |
| Forward | `ds.08` | `Bloom::with_rate(n, p)` computes $m$ and $k$ with q20 to q22 |
| Forward | `data.04` | MinHash signatures of 128 values in 16 bands of 8 rows: q15 to q18 |
| Forward | `L9.1` | tiled matmul's cost and the roofline use the recurrence of q28 |
