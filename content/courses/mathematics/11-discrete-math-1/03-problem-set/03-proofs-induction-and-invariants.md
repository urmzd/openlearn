# Proofs, induction, and invariants

This part of [Discrete math 1 problem set: sets, counting, proofs, induction, relations, bijections](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Choose a proof method and connect induction to loop invariants.

### Proof techniques

**q17.** Prove directly: if $n$ is an odd integer, then $n^2$ is odd. `[proof]`

<details><summary>Answer</summary>

Claim: if $n$ is an odd integer, then $n^2$ is odd.

Method: direct proof from the definition (an integer $n$ is odd when $n = 2k + 1$ for some integer $k$).

Let $n$ be odd, so $n = 2k + 1$ for an integer $k$. Then
$n^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1$.
Since $k$ is an integer, $j = 2k^2 + 2k$ is an integer, and $n^2 = 2j + 1$.

By the definition of odd, $n^2$ is odd.

Model proof also in [`q17.md`](../../../../../course/solutions/solve/S-M05/q17.md).

</details>

**q18.** Prove by contrapositive: if $n$ is an integer and $n^2$ is even, then $n$ is even. `[proof]`

<details><summary>Answer</summary>

Claim: for an integer $n$, if $n^2$ is even then $n$ is even.

Method: contrapositive. The statement $P \Rightarrow Q$ is logically equivalent to $\lnot Q \Rightarrow \lnot P$ (they have the same truth table), so it suffices to prove: if $n$ is not even, then $n^2$ is not even. Every integer is even or odd and not both, so this reads: if $n$ is odd, then $n^2$ is odd.

That is exactly q17: write $n = 2k + 1$, then $n^2 = 2(2k^2 + 2k) + 1$ is odd.

The contrapositive holds, so the original claim holds: if $n^2$ is even, then $n$ is even.

Model proof also in [`q18.md`](../../../../../course/solutions/solve/S-M05/q18.md).

</details>

**q19.** Prove by contradiction: $\sqrt{2}$ is irrational. `[proof]`

<details><summary>Answer</summary>

Claim: $\sqrt{2}$ is irrational (it is not $p/q$ for integers $p, q$ with $q \ne 0$).

Method: contradiction.

Suppose $\sqrt{2} = p/q$ with integers $p, q$, $q \ne 0$. Dividing by common factors, we may assume the fraction is in lowest terms: $p$ and $q$ share no factor greater than 1.

Squaring gives $p^2 = 2q^2$, so $p^2$ is even, and by q18 $p$ is even: $p = 2r$ for an integer $r$. Substituting, $4r^2 = 2q^2$, so $q^2 = 2r^2$ is even, and by q18 again $q$ is even.

Then 2 divides both $p$ and $q$, contradicting lowest terms. So the assumption is false, and $\sqrt{2}$ is irrational.

Model proof also in [`q19.md`](../../../../../course/solutions/solve/S-M05/q19.md).

</details>

**q20.** Prove that there are infinitely many primes. `[proof]`

<details><summary>Answer</summary>

Claim: there are infinitely many primes (integers $p \ge 2$ whose only positive divisors are 1 and $p$).

Method: contradiction (Euclid).

Suppose there are only finitely many primes, $p_1, p_2, \dots, p_k$. Let $N = p_1 p_2 \cdots p_k + 1$. Then $N \ge 3 > 1$.

For each $i$, dividing $N$ by $p_i$ leaves remainder 1, because $p_i$ divides the product $p_1 \cdots p_k$. So no $p_i$ divides $N$.

But every integer greater than 1 has at least one prime factor (q28; or take its smallest divisor greater than 1, which must be prime). That prime factor is a prime not in the list $p_1, \dots, p_k$, contradicting that the list holds every prime.

So there are infinitely many primes.

Model proof also in [`q20.md`](../../../../../course/solutions/solve/S-M05/q20.md).

</details>

**q21.** Prove by cases: $n^2 + n$ is even for every integer $n$. `[proof]`

<details><summary>Answer</summary>

Claim: for every integer $n$, $n^2 + n$ is even.

Method: proof by cases. Every integer is even or odd, so the two cases cover all $n$.

Case 1, $n$ even: $n = 2k$ for an integer $k$. Then $n^2 + n = 4k^2 + 2k = 2(2k^2 + k)$, and $2k^2 + k$ is an integer.

Case 2, $n$ odd: $n = 2k + 1$. Then $n^2 + n = 4k^2 + 4k + 1 + 2k + 1 = 4k^2 + 6k + 2 = 2(2k^2 + 3k + 1)$, and $2k^2 + 3k + 1$ is an integer.

In both cases $n^2 + n$ is 2 times an integer, so it is even for every integer $n$.

Model proof also in [`q21.md`](../../../../../course/solutions/solve/S-M05/q21.md).

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

Model proof also in [`q22.md`](../../../../../course/solutions/solve/S-M05/q22.md).

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

Model proof also in [`q24.md`](../../../../../course/solutions/solve/S-M05/q24.md).

</details>

**q25.** Prove: the sum of a rational number and an irrational number is irrational. `[proof]`

<details><summary>Answer</summary>

Claim: if $r$ is rational and $x$ is irrational, then $r + x$ is irrational.

Method: contradiction.

Suppose $r + x = q$ for a rational $q$. Then $x = q - r$. Write $q = a/b$ and $r = c/d$ with integers $a, b, c, d$ and $b, d \ne 0$. Then $x = (ad - bc)/(bd)$ with $bd \ne 0$, a ratio of integers, so $x$ is rational.

This contradicts the assumption that $x$ is irrational. So $r + x$ is not rational: it is irrational.

Model proof also in [`q25.md`](../../../../../course/solutions/solve/S-M05/q25.md).

</details>

<a id="induction-and-loop-invariants-1"></a>
### Induction and loop invariants

**q26.** Prove by induction: $1 + 2 + \dots + n = \frac{n(n+1)}{2}$ for every integer $n \ge 1$. `[proof]`

<details><summary>Answer</summary>

Claim: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$ for every integer $n \ge 1$.

Method: induction on $n$.

Base case $n = 1$: the left side is $1$ and the right side is $1 \cdot 2 / 2 = 1$.

Induction hypothesis: for some integer $k \ge 1$, $\sum_{i=1}^{k} i = \frac{k(k+1)}{2}$.

Inductive step: $\sum_{i=1}^{k+1} i = \left(\sum_{i=1}^{k} i\right) + (k + 1) = \frac{k(k+1)}{2} + (k+1)$ by the hypothesis, $= \frac{k(k+1) + 2(k+1)}{2} = \frac{(k+1)(k+2)}{2}$, which is the formula for $n = k + 1$.

By induction, the formula holds for every $n \ge 1$.

Model proof also in [`q26.md`](../../../../../course/solutions/solve/S-M05/q26.md).

</details>

**q27.** Prove by induction: $2^n > n^2$ for every integer $n \ge 5$. `[proof]`

<details><summary>Answer</summary>

Claim: $2^n > n^2$ for every integer $n \ge 5$.

Method: induction on $n$ starting at 5.

Base case $n = 5$: $2^5 = 32 > 25 = 5^2$.

Induction hypothesis: $2^k > k^2$ for some integer $k \ge 5$.

Inductive step: $2^{k+1} = 2 \cdot 2^k > 2k^2$ by the hypothesis. Now $2k^2 \ge (k+1)^2$ is the same as $k^2 \ge 2k + 1$, that is $k^2 - 2k - 1 \ge 0$, that is $(k - 1)^2 \ge 2$, which holds for every $k \ge 3$ and so for $k \ge 5$. Therefore $2^{k+1} > 2k^2 \ge (k+1)^2$.

By induction, $2^n > n^2$ for every $n \ge 5$. (The claim fails at $n = 4$, where $16 = 16$, which is why the base case is 5.)

Model proof also in [`q27.md`](../../../../../course/solutions/solve/S-M05/q27.md).

</details>

**q28.** Prove by strong induction: every integer $n \ge 2$ is a product of one or more primes. `[proof]`

<details><summary>Answer</summary>

Claim: every integer $n \ge 2$ is a product of one or more primes.

Method: strong induction on $n$.

Strong hypothesis: for a fixed $n \ge 2$, every integer $m$ with $2 \le m < n$ is a product of primes. We show $n$ is too.

Case 1, $n$ is prime: then $n$ itself is a product of one prime. This covers the base case $n = 2$, which is prime.

Case 2, $n$ is not prime: then $n$ has a divisor $a$ with $1 < a < n$, so $n = ab$ with $b = n / a$, and $1 < b < n$ as well. Both $a$ and $b$ lie in $[2, n)$, so by the hypothesis $a = p_1 \cdots p_r$ and $b = q_1 \cdots q_s$ are products of primes, and $n = p_1 \cdots p_r q_1 \cdots q_s$ is one too.

By strong induction, every integer $n \ge 2$ is a product of primes.

Model proof also in [`q28.md`](../../../../../course/solutions/solve/S-M05/q28.md).

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

Model proof also in [`q29.md`](../../../../../course/solutions/solve/S-M05/q29.md).

</details>

**q30.** The online softmax (the C kernel of [Softmax in C: three-pass and online two-pass](../../../ai-engineering/22-kernels/04-softmax-three-pass-and-online.md)) reads $x_0, x_1, \dots$ once, keeping a running maximum $m$ and a running sum $s$. It starts at $m = -\infty$, $s = 0$ (with $e^{-\infty} = 0$), and for each new $x_i$ sets $m' = \max(m, x_i)$ and $s' = s \cdot e^{m - m'} + e^{x_i - m'}$. Prove that after reading $x_0, \dots, x_{i-1}$ it holds that $m = \max_{j < i} x_j$ and $s = \sum_{j < i} e^{x_j - m}$. `[proof]`

<details><summary>Answer</summary>

Claim: after the loop has read $x_0, \dots, x_{i-1}$, $m = \max_{j < i} x_j$ and $s = \sum_{j<i} e^{x_j - m}$ (for $i = 0$: $m = -\infty$, the max of nothing, and $s = 0$, the empty sum).

Method: induction on $i$, the number of elements read (a loop invariant).

Base $i = 0$: the loop starts with $m = -\infty$ and $s = 0$, which are the empty max and the empty sum.

Step: assume the claim after $i$ elements, with values $m, s$, and read $x_i$. Then $m' = \max(m, x_i) = \max(\max_{j<i} x_j, x_i) = \max_{j \le i} x_j$. For the sum, using the hypothesis and $e^{a}e^{b} = e^{a+b}$:
$$s \cdot e^{m - m'} = \sum_{j<i} e^{x_j - m} e^{m - m'} = \sum_{j<i} e^{x_j - m'},$$
so $s' = s e^{m - m'} + e^{x_i - m'} = \sum_{j \le i} e^{x_j - m'}$. When $i = 0$, $s = 0$ and the first term is 0 (with $e^{-\infty} = 0$), so $s' = e^{x_0 - x_0} = 1$, which also matches. The claim holds after $i + 1$ elements.

By induction it holds after every element. Every exponent $x_j - m$ is $\le 0$, so no term exceeds 1 and $s$ never overflows.

Model proof also in [`q30.md`](../../../../../course/solutions/solve/S-M05/q30.md).

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

## Review questions

1. What must be established for an invariant to prove a loop result?

   <details><summary>Answer</summary>

   Initialization, preservation, and the postcondition on termination; termination itself requires a separate argument when claiming total correctness.

   </details>

[Contents](README.md) · [Next: Relations, functions, and validation](04-relations-functions-and-validation.md)
