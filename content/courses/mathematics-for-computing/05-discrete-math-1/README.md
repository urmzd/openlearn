# Discrete Math 1

## Overview
- **Textbook**: *Book of Proof* by Richard Hammack -- https://people.vcu.edu/~rhammack/BookOfProof/ (CC BY-ND)
- **Prerequisites**: None
- **Estimated time**: 4 weeks at 10-12 hrs/week

## Key Takeaways
- Learn to read and write rigorous mathematical proofs -- the single most transferable mathematical skill
- Master the language of sets, logic, and quantifiers used in every CS theory course
- Develop fluency with direct proof, contrapositive, contradiction, and induction
- Understand relations and functions as formal structures underlying databases and programming
- Build counting skills (permutations, combinations) essential for algorithm analysis and probability

## How to Study
- Read each chapter and attempt the exercises before looking at solutions
- Write proofs by hand in complete sentences -- proof is a form of technical writing
- For each proof technique, practice on at least 10 problems until the pattern becomes automatic
- Compare your proofs with Hammack's solutions for style and rigor
- Revisit counting problems when you reach probability and algorithms courses

---

# Concepts & Techniques

## Core Insight

Discrete mathematics is the mathematics of distinct, separated structures (as opposed to the
continuous structures of calculus). Proof is the central activity: you must learn to construct
airtight logical arguments. Every concept in this course -- sets, logic, counting, relations,
functions -- appears directly in computer science. Sets become data structures, logic becomes
boolean algebra and type systems, induction becomes recursive correctness arguments, relations
become database schemas, and functions become the foundation of programming.

## 1. Sets

**Textbook sections**: Ch 1, Sections 1.1-1.8

**Key definitions**:
- **Set**: An unordered collection of distinct elements; described by listing {1,2,3} or set-builder {x in Z : x > 0}
- **Subset**: A is a subset of B if every element of A is in B
- **Power set**: P(A) = the set of all subsets of A; |P(A)| = 2^|A|
- **Union**: A union B = {x : x in A or x in B}
- **Intersection**: A intersect B = {x : x in A and x in B}
- **Complement**: A^c = {x in U : x not in A}
- **Cartesian product**: A x B = {(a,b) : a in A, b in B}; |A x B| = |A| * |B|

**Key theorems**:
- **De Morgan's Laws**: (A union B)^c = A^c intersect B^c, and (A intersect B)^c = A^c union B^c. *Intuition*: "not (this or that)" is the same as "not this and not that" -- negation swaps union and intersection.
- **Inclusion-Exclusion (2 sets)**: |A union B| = |A| + |B| - |A intersect B|. *Intuition*: adding the sizes double-counts the overlap, so subtract it once.
- **Power Set Cardinality**: |P(A)| = 2^|A|. *Intuition*: for each element, independently choose "in" or "out" -- that is 2 choices per element.

**Worked example**:
> Prove that A intersect (B union C) = (A intersect B) union (A intersect C). Let x be in A intersect (B union C). Then x is in A and x is in B union C. Case 1: x in B. Then x in A intersect B, so x in (A intersect B) union (A intersect C). Case 2: x in C. Then x in A intersect C, so x in (A intersect B) union (A intersect C). The reverse direction is similar.

**Essential problems**: Hammack Ch 1, Exercises: #1-8, #13-18, #25-30
**Challenge problems**: Hammack Ch 1, Exercises: #31-36

## 2. Logic

**Textbook sections**: Ch 2, Sections 2.1-2.10

**Key definitions**:
- **Proposition**: A statement that is either true or false
- **Conjunction (AND)**: P AND Q is true only when both P and Q are true
- **Disjunction (OR)**: P OR Q is true when at least one of P, Q is true
- **Negation (NOT)**: NOT P reverses the truth value
- **Conditional**: P => Q (if P then Q); false only when P is true and Q is false
- **Biconditional**: P <=> Q (P if and only if Q); true when P and Q have the same truth value
- **Predicate**: A statement with a variable, e.g., P(x) = "x is even"
- **Universal quantifier**: For all x, P(x) -- asserts P holds for every x in the domain
- **Existential quantifier**: There exists x such that P(x) -- asserts P holds for at least one x

**Key theorems**:
- **Logical Equivalences**: P => Q is equivalent to NOT P OR Q. *Intuition*: "if P then Q" only fails when P is true and Q is false; in all other cases, the implication holds.
- **Contrapositive**: P => Q is equivalent to NOT Q => NOT P. *Intuition*: "if it rained then the ground is wet" is the same as "if the ground is dry then it didn't rain."
- **Negation of Quantifiers**: NOT(for all x, P(x)) is equivalent to there exists x such that NOT P(x). NOT(there exists x, P(x)) is equivalent to for all x, NOT P(x). *Intuition*: negation swaps quantifiers, just as De Morgan's Laws swap AND/OR.

**Worked example**:
> Negate "For every epsilon > 0, there exists delta > 0 such that |x-a| < delta implies |f(x)-L| < epsilon." Negation: "There exists epsilon > 0 such that for all delta > 0, there exists x with |x-a| < delta and |f(x)-L| >= epsilon." Each quantifier flips, and the implication becomes a conjunction with negated consequent.

**Essential problems**: Hammack Ch 2, Exercises: #1-10, #15-22
**Challenge problems**: Hammack Ch 2, Exercises: #23-28

## 3. Counting

**Textbook sections**: Ch 3, Sections 3.1-3.5

**Key definitions**:
- **Multiplication principle**: If task 1 has m outcomes and task 2 has n outcomes, the sequence has m*n outcomes
- **Addition principle**: If task 1 has m outcomes and task 2 has n outcomes (mutually exclusive), total is m+n
- **Permutation**: An ordered arrangement; P(n,k) = n! / (n-k)!
- **Combination**: An unordered selection; C(n,k) = n! / (k!(n-k)!)
- **Binomial coefficient**: C(n,k), also written "n choose k"

**Key theorems**:
- **Binomial Theorem**: (x + y)^n = Sum from k=0 to n of C(n,k) * x^(n-k) * y^k. *Intuition*: expanding (x+y)^n, each term chooses x from some factors and y from the rest. The number of ways to choose k factors for y is C(n,k).
- **Pascal's Identity**: C(n,k) = C(n-1,k-1) + C(n-1,k). *Intuition*: to choose k items from n, either include item n (then choose k-1 from the remaining n-1) or exclude it (choose k from n-1).
- **Stars and Bars**: The number of ways to distribute n identical objects into k distinct bins is C(n+k-1, k-1). *Intuition*: arrange n stars and k-1 dividers in a row.

**Worked example**:
> How many 5-card poker hands contain exactly 2 aces? Choose 2 aces from 4: C(4,2) = 6. Choose remaining 3 cards from the 48 non-aces: C(48,3) = 17296. Total: 6 * 17296 = 103776.

> How many bit strings of length 8 have exactly 3 ones? Choose positions for the ones: C(8,3) = 56.

**Essential problems**: Hammack Ch 3, Exercises: #1-12, #17-24
**Challenge problems**: Hammack Ch 3, Exercises: #25-30

## 4. Direct Proof

**Textbook sections**: Ch 4, Sections 4.1-4.5

**Key definitions**:
- **Direct proof of P => Q**: Assume P is true, then use definitions, theorems, and logical reasoning to conclude Q

**Key technique**:
- Assume the hypothesis. Unpack definitions. Manipulate using known results. Arrive at the conclusion.

**Worked example**:
> Prove: If n is odd, then n^2 is odd. Assume n is odd, so n = 2k+1 for some integer k. Then n^2 = (2k+1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1. Since 2k^2 + 2k is an integer, n^2 has the form 2m+1, so n^2 is odd.

> Prove: If a | b and b | c, then a | c. Since a | b, we have b = ak for some integer k. Since b | c, we have c = bj for some integer j. Then c = (ak)j = a(kj). Since kj is an integer, a | c.

**Essential problems**: Hammack Ch 4, Exercises: #1-8, #13-18, #23-28
**Challenge problems**: Hammack Ch 4, Exercises: #29-34

## 5. Contrapositive Proof

**Textbook sections**: Ch 5, Sections 5.1-5.3

**Key definitions**:
- **Contrapositive proof of P => Q**: Prove NOT Q => NOT P instead (logically equivalent)

**Key technique**:
- Use when the negation of Q gives you more to work with than assuming P directly.

**Worked example**:
> Prove: If n^2 is even, then n is even. Contrapositive: If n is odd, then n^2 is odd. Assume n = 2k+1. Then n^2 = 4k^2+4k+1 = 2(2k^2+2k)+1, which is odd. (This avoids the difficulty of working directly from "n^2 is even.")

> Prove: If 5x + 3 is odd, then x is even. Contrapositive: If x is odd (x = 2k+1), then 5x+3 = 10k+5+3 = 10k+8 = 2(5k+4), which is even.

**Essential problems**: Hammack Ch 5, Exercises: #1-8, #11-16
**Challenge problems**: Hammack Ch 5, Exercises: #17-20

## 6. Proof by Contradiction

**Textbook sections**: Ch 6, Sections 6.1-6.4

**Key definitions**:
- **Proof by contradiction**: Assume the statement is false, then derive a logical contradiction

**Key technique**:
- Assume NOT(statement). Reason logically until you reach an impossibility (e.g., a number that is both even and odd, or a rational that is irrational). Conclude the original statement must be true.

**Worked example**:
> Prove: sqrt(2) is irrational. Assume sqrt(2) = a/b where a/b is in lowest terms (gcd(a,b)=1). Then 2 = a^2/b^2, so a^2 = 2b^2. Thus a^2 is even, so a is even (by contrapositive from concept 5). Write a = 2c. Then 4c^2 = 2b^2, so b^2 = 2c^2, so b is even. But then both a and b are even, contradicting gcd(a,b)=1.

> Prove: There are infinitely many primes. Assume there are finitely many: p_1, p_2, ..., p_n. Consider N = p_1*p_2*...*p_n + 1. N is not divisible by any p_i (remainder 1). So N is either prime itself or has a prime factor not in the list. Contradiction.

**Essential problems**: Hammack Ch 6, Exercises: #1-8, #13-18
**Challenge problems**: Hammack Ch 6, Exercises: #19-24

## 7. Non-Conditional Proofs

**Textbook sections**: Ch 7, Sections 7.1-7.4

**Key definitions**:
- **If-and-only-if proof**: Prove both directions: P => Q and Q => P
- **Existence proof**: Show an object with the desired property exists (constructive or non-constructive)
- **Uniqueness proof**: Show at most one object has the property (assume two, show they are equal)
- **Proof by cases**: Break into exhaustive cases and prove each separately

**Worked example**:
> Prove: n is odd if and only if n^2 is odd. (=>) If n is odd, n=2k+1, n^2=4k^2+4k+1=2(2k^2+2k)+1 is odd. (<=) Contrapositive: if n is even, n=2k, n^2=4k^2=2(2k^2) is even. Both directions established.

**Essential problems**: Hammack Ch 7, Exercises: #1-8, #13-18, #25-30
**Challenge problems**: Hammack Ch 7, Exercises: #31-36

## 8. Mathematical Induction

**Textbook sections**: Ch 10, Sections 10.1-10.4

**Key definitions**:
- **Weak induction**: Prove P(base). Assume P(k), prove P(k+1). Conclude P(n) for all n >= base.
- **Strong induction**: Prove P(base). Assume P(j) for all base <= j <= k, prove P(k+1).
- **Structural induction**: Induction on recursively defined structures (trees, lists, formulas)

**Key theorems**:
- **Well-Ordering Principle**: Every nonempty subset of the natural numbers has a least element. *Intuition*: this is logically equivalent to the principle of induction -- if induction fails, there is a smallest counterexample, contradicting well-ordering.

**Worked example**:
> Prove: 1 + 2 + ... + n = n(n+1)/2. Base case: n=1, LHS = 1, RHS = 1(2)/2 = 1. Inductive step: Assume 1+2+...+k = k(k+1)/2. Then 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2. This is the formula with n = k+1.

> Prove: 2^n > n for all n >= 1 (strong induction style). Base: 2^1 = 2 > 1. Assume 2^j > j for all 1 <= j <= k. Then 2^(k+1) = 2 * 2^k > 2k >= k+1 (since k >= 1). Done.

> Structural induction: Prove every binary tree with n internal nodes has n+1 leaves. Base: 0 internal nodes, 1 leaf. Inductive: A tree with root has left subtree (n_L internal, n_L+1 leaves) and right subtree (n_R internal, n_R+1 leaves). Total internal = n_L + n_R + 1. Total leaves = (n_L+1) + (n_R+1) = n_L + n_R + 2 = (internal nodes) + 1.

**Essential problems**: Hammack Ch 10, Exercises: #1-10, #15-22, #27-32
**Challenge problems**: Hammack Ch 10, Exercises: #33-38

## 9. Relations

**Textbook sections**: Ch 11, Sections 11.1-11.6

**Key definitions**:
- **Relation on A**: A subset R of A x A; we write a R b or (a,b) in R
- **Reflexive**: a R a for all a in A
- **Symmetric**: a R b implies b R a
- **Transitive**: a R b and b R c implies a R c
- **Equivalence relation**: A relation that is reflexive, symmetric, and transitive
- **Equivalence class**: [a] = {x in A : x R a}; the set of all elements equivalent to a
- **Partition**: A collection of nonempty, pairwise disjoint subsets whose union is A
- **Partial order**: A relation that is reflexive, antisymmetric (a R b and b R a implies a = b), and transitive

**Key theorems**:
- **Fundamental Theorem of Equivalence Relations**: The equivalence classes of an equivalence relation on A form a partition of A, and conversely every partition of A defines an equivalence relation. *Intuition*: equivalence relations and partitions are the same concept viewed from two angles. "Being equivalent" means "being in the same part."

**Worked example**:
> On Z, define a ~ b iff 3 | (a - b) (congruence mod 3). Reflexive: 3 | (a-a) = 0. Symmetric: if 3 | (a-b), then 3 | (-(a-b)) = (b-a). Transitive: if 3 | (a-b) and 3 | (b-c), then 3 | ((a-b)+(b-c)) = (a-c). Equivalence classes: [0] = {...,-6,-3,0,3,6,...}, [1] = {...,-5,-2,1,4,7,...}, [2] = {...,-4,-1,2,5,8,...}. These partition Z.

**Essential problems**: Hammack Ch 11, Exercises: #1-10, #15-22
**Challenge problems**: Hammack Ch 11, Exercises: #23-28

## 10. Functions

**Textbook sections**: Ch 12, Sections 12.1-12.6

**Key definitions**:
- **Function**: A relation f from A to B such that every element of A is related to exactly one element of B
- **Injection (one-to-one)**: f(a) = f(b) implies a = b
- **Surjection (onto)**: For every b in B, there exists a in A with f(a) = b
- **Bijection**: A function that is both injective and surjective; establishes a one-to-one correspondence
- **Composition**: (g o f)(x) = g(f(x))
- **Inverse**: f^(-1) exists iff f is a bijection; f^(-1)(f(a)) = a

**Key theorems**:
- **Composition Preserves Properties**: If f and g are injective, then g o f is injective. If f and g are surjective, then g o f is surjective. *Intuition*: composing two "no-collision" maps gives a "no-collision" map; composing two "everything-is-hit" maps gives an "everything-is-hit" map.
- **Bijection and Inverse**: f has an inverse iff f is a bijection. *Intuition*: to "undo" a function, every output must come from exactly one input.
- **Cantor's Theorem**: |A| < |P(A)| for any set A (there is no surjection from A onto its power set). *Intuition*: there are always more subsets of a set than elements -- this is the source of different sizes of infinity.
- **Schroder-Bernstein**: If there exist injections A -> B and B -> A, then there exists a bijection A <-> B. *Intuition*: if each set "fits inside" the other, they must be the same size.

**Worked example**:
> Prove f: Z -> Z defined by f(n) = 2n + 1 is injective but not surjective. Injection: if 2a+1 = 2b+1, then a = b. Not surjective: is there n with 2n+1 = 4? That requires n = 3/2, which is not in Z. So 4 has no preimage.

> Prove f: R -> R defined by f(x) = x^3 is a bijection. Injection: if a^3 = b^3, then a = b (cube root is well-defined on R). Surjection: for any y in R, x = y^(1/3) satisfies f(x) = y. Inverse: f^(-1)(y) = y^(1/3).

**Essential problems**: Hammack Ch 12, Exercises: #1-10, #15-22, #27-34
**Challenge problems**: Hammack Ch 12, Exercises: #35-40

---

## Technique Catalog

| Technique | When to Use | Key Formula/Idea |
|-----------|-------------|------------------|
| Direct Proof | Prove P => Q when definitions lead forward | Assume P, derive Q |
| Contrapositive | P => Q is hard; NOT Q => NOT P is easier | Negate both, reverse direction |
| Contradiction | Existence/uniqueness or irrationality proofs | Assume NOT(statement), derive impossibility |
| Weak Induction | Prove P(n) for all n >= base | Base + (P(k) => P(k+1)) |
| Strong Induction | P(k+1) depends on multiple prior cases | Base + (all P(j) for j <= k => P(k+1)) |
| Structural Induction | Properties of recursive structures (trees, lists) | Base structure + induction on construction |
| Proof by Cases | When the domain naturally splits | Prove each case separately, ensure exhaustive |
| Counting (Multiplication) | Sequential independent choices | Multiply the number of options at each stage |
| Counting (Combinations) | Unordered selections | C(n,k) = n! / (k!(n-k)!) |
| Bijective Proof | Show two counts are equal | Find a one-to-one correspondence |

## Connections to CS & Algorithms

| Math Concept | CS Application | Repo Link |
|-------------|----------------|-----------|
| Propositional logic | Boolean algebra, circuit design, SAT solvers | algorithms track |
| Quantifiers & predicates | Database query languages (SQL WHERE clauses) | systems track |
| Induction | Proving correctness of recursive algorithms | algorithms track |
| Sets & operations | Data structures (HashSet, TreeSet), type systems | algorithms track |
| Relations (equivalence) | Union-Find data structure, database normalization | algorithms track |
| Relations (partial order) | Topological sort, dependency resolution | algorithms track |
| Functions (bijections) | Hashing, encoding/decoding, encryption | systems track |
| Counting / combinatorics | Algorithm analysis (average case, expected values) | [07-probability-statistics](../07-probability-statistics/) |
| Binomial theorem | Analyzing divide-and-conquer recurrences | algorithms track |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google/Meta | Combinatorics in interview problems (counting valid configurations) | Medium |
| Amazon | Logic puzzles and constraint reasoning in interviews | Medium |
| Any SWE role | Proving loop invariants, arguing correctness of algorithms | Medium |
| Security (Cloudflare, CrowdStrike) | Logic underpins formal verification and access control | Hard |
| Database companies (Snowflake, Databricks) | Relations, normalization, query equivalence | Medium |
| Functional programming shops | Functions as first-class objects, composition, bijections | Medium |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M05.1` | [Counting: params, FLOPs, KV bytes, memory plans](01-counting-params-flops-kv-bytes.md) | build | 5 |
| 2 | `M05.2` | [Injective, surjective, bijective; the GPT-2 byte map](02-injective-surjective-bijective-gpt2-byte-map.md) | build | 3 |
| 3 | `S-M05` | [Discrete math 1 problem set: sets, counting, proofs, induction, relations, bijections](90-problem-set.md) | solve | 2 |
