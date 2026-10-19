# Solve set: functions, exp/log, trig and Euler, series, polynomials, inequalities

## Overview

| | |
|---|---|
| **Module** | `S-M00` · solve · pen and paper · Pass 2 · 4 to 6 h |
| **Solution** | worked answers to all 60 problems, each under its problem in section 4 (the chapter is self-contained) |
| **Needs** | nothing. Each part points to the chapter section that teaches it |
| **Used by** | the four build modules of this topic (`M00.1` to `M00.4`) as their pen-and-paper side; parts 1 and 6 are `M00.5` (functions, inverses, monotonicity, inequalities), which `M01.1` and `M05.2` assume. Together with those modules it completes the Pass 2 math foundation |
| **Optional depth** | OpenStax, *Precalculus 2e* (free), ch. 1 (functions), 2 to 3 (linear and polynomial functions, inequalities), 6 (exponentials and logs), 5 to 8 (trigonometry, complex numbers), 11 (sequences and series) |

## Key Takeaways

- A **function** assigns exactly one output to each input in its **domain**; its **inverse** exists exactly when no two inputs share an output, and undoes it: $f^{-1}(f(x)) = x$.
- A strictly **monotonic** function (always increasing, or always decreasing) is one-to-one, so it has an inverse; that is why $\exp$, $\ln$, the sigmoid, and softplus can all be inverted.
- Solving an **inequality** is solving the matching equation and testing the sign between its solutions; multiplying or dividing by a negative number, or applying a decreasing function, flips the direction.
- The other four parts are the pen-and-paper side of `M00.1` to `M00.4`: if a problem in part 2 to 5 stops you, read the beat 2 section of that module's chapter it names.

## How to read this chapter

Section 2 teaches the functions, inverses, and inequalities of parts 1 and 6 and maps parts 2 to 5 to the chapters that teach them. Section 3 works two sibling problems by hand. Section 4 states all 60 problems; each has a worked answer behind an "Answer" fold, with the derivation and the most common wrong answer and why it is wrong. Attempt each problem on paper before opening its answer.

---

## 1. Why now

Pass 2 turns the math under the tracer system into code: logarithms that measure information (`M00.1`), rotations that encode position (`M00.2`), ladders of frequencies (`M00.3`), and polynomial evaluation that kernels run (`M00.4`), and after them calculus, linear algebra, probability, and optimization. Each of those chapters assumes you can manipulate the expressions fluently by hand: invert a function, combine logarithms, use an angle-addition formula, sum a geometric series, factor a polynomial, and solve an inequality. A bug in a derivation becomes a bug in code that tests catch late and explain poorly. This set practises the hand skills first, and its worked answers show the derivations and the usual wrong turns. Parts 1 and 6 also teach `M00.5`, functions, inverses, monotonicity, and inequalities, which has no build module of its own because none of it becomes code; `M01.1` (limits and derivatives) and `M05.2` build on it directly.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f: A \to B$ | a function from the set $A$ (its domain) to the set $B$ | |
| $f(x)$ | the output of $f$ at the input $x$ | |
| $\operatorname{dom} f$ | the domain: the inputs where $f$ is defined | a set or interval |
| $\operatorname{ran} f$ | the range: the outputs $f$ actually takes | a set or interval |
| $f \circ g$ | composition, $(f \circ g)(x) = f(g(x))$ | |
| $f^{-1}$ | the inverse function: $f^{-1}(y) = x$ exactly when $f(x) = y$ | |
| $(a, b)$, $[a, b]$ | the open interval $a < x < b$ and the closed interval $a \le x \le b$ | |
| $\cup$ | union of sets, written `U` in answers | |
| $\sigma(z)$ | the sigmoid $1/(1 + e^{-z})$ | |
| $\lvert x\rvert$ | absolute value: $x$ if $x \ge 0$, $-x$ otherwise | |

### 2.1 Functions, domains, ranges, composition

A **function** $f$ assigns to each input $x$ in its **domain** exactly one output $f(x)$. A formula alone leaves the domain implicit: it is every real $x$ where the formula makes sense. $\ln(4 - x^2)$ needs $4 - x^2 > 0$, so its domain is $(-2, 2)$; $1/(x - 3)$ needs $x \ne 3$. The **range** is the set of outputs that actually occur: the sigmoid $\sigma(z) = 1/(1 + e^{-z})$ takes every value strictly between 0 and 1 and never reaches either, so its range is $(0, 1)$. **Composition** chains functions: $(f \circ g)(x) = f(g(x))$ applies $g$ first. Order matters: with $f(x) = x^2$ and $g(x) = x + 1$, $f(g(x)) = (x + 1)^2$ but $g(f(x)) = x^2 + 1$.

### 2.2 Inverses and monotonicity

$f$ is **one-to-one** when different inputs give different outputs. Then every output $y$ in the range comes from exactly one $x$, and the **inverse** $f^{-1}(y) = x$ undoes $f$: $f^{-1}(f(x)) = x$ and $f(f^{-1}(y)) = y$. To find it, write $y = f(x)$ and solve for $x$; the graph of $f^{-1}$ is the graph of $f$ reflected across the line $y = x$.

A function is **strictly increasing** when $x_1 < x_2$ implies $f(x_1) < f(x_2)$ (strictly decreasing: $f(x_1) > f(x_2)$). Either kind, called strictly **monotonic**, is one-to-one, because different inputs are ordered and so are their outputs. $e^x$, $\ln x$, $x^3 + x$, $\sigma$, and softplus $\ln(1 + e^x)$ are strictly increasing; $x^2$ is not monotonic on all of $\mathbb{R}$ ($(-2)^2 = 2^2$), so it has an inverse only on a half-line, where $\sqrt{\ }$ is it. `M01.1` gives the practical test: a function whose derivative is positive everywhere is strictly increasing.

### 2.3 Inequalities

An inequality is solved the way an equation is, with one extra rule: **multiplying or dividing both sides by a negative number reverses the direction** ($-2x < 6$ means $x > -3$), and so does applying a strictly decreasing function. Applying a strictly increasing function ($e^x$, $\ln$, $\sqrt{\ }$ on non-negative numbers) keeps it, which is how $e^x \le 2$ becomes $x \le \ln 2$.

For a polynomial or a quotient of polynomials, find where it is zero or undefined, cut the line there, and test the sign on each piece. $x^2 - 5x + 6 = (x - 2)(x - 3)$ is zero at 2 and 3, positive outside and negative between, so $x^2 - 5x + 6 > 0$ on $(-\infty, 2) \cup (3, \infty)$. An endpoint where the expression is zero belongs to the answer for $\ge$ and $\le$, never one where it is undefined. An absolute value is a distance: $|x - 3| < 2$ means "within 2 of 3", the interval $(1, 5)$. Two inequalities appear again in optimization and information theory: $x + 1/x \ge 2$ for $x > 0$ (with equality at $x = 1$; multiply by $x$ and get $(x - 1)^2 \ge 0$), and $\ln x \le x - 1$ (equality only at $x = 1$), the inequality behind Gibbs' inequality in `M11.1`.

### 2.4 Where the other parts are taught

| Part | Questions | Taught in |
|---|---|---|
| 2. exponents, logarithms, units | q11 to q22 | [`M00.1`](01-exponents-logs-and-units-of-information.md) sections 2.1 to 2.4 |
| 3. trigonometry, complex numbers, Euler | q23 to q34 | [`M00.2`](02-trig-rotations-and-eulers-formula.md) sections 2.1 to 2.6 |
| 4. sequences and geometric series | q35 to q44 | [`M00.3`](03-sequences-and-frequency-ladders.md) sections 2.1 to 2.5 |
| 5. polynomials | q45 to q52 | [`M00.4`](04-polynomials-horner-and-stable-roots.md) sections 2.1 to 2.5 |

## 3. Worked example by hand

**A worked solution of a sibling of q5 and q6: the inverse and range of $f(x) = 2/(1 + e^{-x})$.**

1. *Range.* $e^{-x}$ takes every value in $(0, \infty)$, so $1 + e^{-x}$ takes every value in $(1, \infty)$ and $2/(1 + e^{-x})$ every value in $(0, 2)$, approaching 2 as $x \to \infty$ and 0 as $x \to -\infty$ without reaching either. Range: `(0, 2)`.
2. *Monotonic, so invertible.* As $x$ grows, $e^{-x}$ shrinks, the denominator shrinks, and $f(x)$ grows: $f$ is strictly increasing, hence one-to-one on $\mathbb{R}$.
3. *Solve $y = f(x)$ for $x$*, for $0 < y < 2$:
   $$y(1 + e^{-x}) = 2 \;\Rightarrow\; e^{-x} = \frac{2}{y} - 1 = \frac{2 - y}{y} \;\Rightarrow\; -x = \ln\frac{2 - y}{y} \;\Rightarrow\; x = \ln\frac{y}{2 - y}.$$
4. *Check.* $f(0) = 2/(1 + 1) = 1$, and $f^{-1}(1) = \ln(1/1) = 0$. Correct.
5. *Answer*, in the variable of the question: $f^{-1}(y) = \ln\dfrac{y}{2 - y}$. Equivalent forms such as $\ln y - \ln(2 - y)$ or $-\ln(2/y - 1)$ are the same function; what matters is equality, not spelling, and a quick test at a couple of points (step 4) is how to confirm two forms agree.

**And one inequality, a sibling of q56: $\dfrac{x + 1}{x - 4} \le 0$.** The numerator is zero at $-1$ and the denominator at $4$. Signs: for $x < -1$ both factors are negative and the quotient positive; for $-1 < x < 4$ the numerator is positive and the denominator negative, so the quotient is negative; for $x > 4$ it is positive. Zero at $x = -1$ is allowed by $\le$; $x = 4$ divides by zero. Answer: `[-1, 4)`.

## 4. Problem set with worked answers

Try each problem before opening its answer. The tag after each question is the kind of answer expected: an expression in the named variables, a number, an interval, a set, a vector, or true/false. Exact values are expected where they exist; a decimal is given alongside as a check.

### Functions and inverses

**q1.** $f(x) = 3x - 7$. Give $f^{-1}(x)$. `[expr in x]`

<details><summary>Answer</summary>

$f^{-1}(x) = \dfrac{x + 7}{3}$. $f$ multiplies by 3, then subtracts 7, so the inverse undoes those steps in reverse order: add 7, then divide by 3. Formally, $y = 3x - 7 \Rightarrow x = (y + 7)/3$, then rename $y$ to $x$. Check: $f^{-1}(f(2)) = f^{-1}(-1) = 6/3 = 2$. A common mistake is $(x - 7)/3$, which undoes the steps in the original order.

</details>

**q2.** $f(x) = e^{2x + 1}$. Give $f^{-1}(x)$ for $x > 0$. `[expr in x]`

<details><summary>Answer</summary>

$f^{-1}(x) = \dfrac{\ln x - 1}{2}$. From $y = e^{2x + 1}$, take $\ln$ (defined because $y > 0$): $\ln y = 2x + 1$, so $x = (\ln y - 1)/2$. A common mistake is $\ln(x)/2 - 1$, which halves before subtracting: the last step of $f$ (exponentiate) is undone first, then "plus 1", then "times 2".

</details>

**q3.** $g(x) = \dfrac{2x + 1}{x - 3}$ for $x \ne 3$. Give $g^{-1}(x)$. `[expr in x]`

<details><summary>Answer</summary>

$g^{-1}(x) = \dfrac{3x + 1}{x - 2}$. Solve $y = (2x + 1)/(x - 3)$: $y(x - 3) = 2x + 1 \Rightarrow yx - 2x = 3y + 1 \Rightarrow x(y - 2) = 3y + 1 \Rightarrow x = (3y + 1)/(y - 2)$. Check: $g(4) = 9$, and $(27 + 1)/(9 - 2) = 4$. A common mistake is the reciprocal $(x - 3)/(2x + 1)$: $1/g$ is not $g^{-1}$.

</details>

**q4.** $f(x) = x^2$ and $g(x) = x + 1$. Give $(f \circ g)(x) = f(g(x))$. `[expr in x]`

<details><summary>Answer</summary>

$(x + 1)^2$. $g$ runs first and gives $x + 1$; $f$ squares that. A common mistake is $x^2 + 1$, which is $g(f(x))$: composition is applied right to left.

</details>

**q5.** The sigmoid is $\sigma(z) = 1 / (1 + e^{-z})$. Give its inverse, the logit $\sigma^{-1}(p)$, for $0 < p < 1$. `[expr in p]`

<details><summary>Answer</summary>

$\sigma^{-1}(p) = \ln\dfrac{p}{1 - p}$. As in the worked example with 2 replaced by 1: $p(1 + e^{-z}) = 1 \Rightarrow e^{-z} = (1 - p)/p \Rightarrow z = \ln(p/(1 - p))$. This is the logit a classifier head outputs (`L6.5`). Check: $\sigma(0) = 1/2$ and $\ln(1) = 0$. A common mistake is $\ln((1 - p)/p)$, which forgets the sign from $-z$.

</details>

**q6.** Give the range of $\sigma$ (the set of values it takes over all real $z$). `[interval]`

<details><summary>Answer</summary>

$(0, 1)$. $1 + e^{-z}$ takes every value in $(1, \infty)$, so its reciprocal takes every value in $(0, 1)$, approaching but never reaching either end. A common mistake is $[0, 1]$: the sigmoid never outputs exactly 0 or 1, which is why $\ln \sigma$ is always finite.

</details>

**q7.** Give the domain of $h(x) = \ln(4 - x^2)$ (the real $x$ where it is defined). `[interval]`

<details><summary>Answer</summary>

$(-2, 2)$. $\ln$ needs a positive argument: $4 - x^2 > 0 \iff x^2 < 4 \iff -2 < x < 2$. A common mistake is $[-2, 2]$: at $x = \pm 2$ the argument is 0 and $\ln 0$ is undefined.

</details>

**q8.** Is $f(x) = x^3 + x$ one-to-one on all of $\mathbb{R}$? (Hint: is it monotonic?) `[bool]`

<details><summary>Answer</summary>

True. Both $x^3$ and $x$ are strictly increasing, and a sum of strictly increasing functions is strictly increasing, so $x_1 < x_2$ gives $f(x_1) < f(x_2)$ and no two inputs share an output. (With `M01.1`: $f'(x) = 3x^2 + 1 > 0$ everywhere.)

</details>

**q9.** $f(x) = 2^x$ and $g(x) = \log_2 x$. Compute $f(g(16)) + g(f(3))$. `[number]`

<details><summary>Answer</summary>

$19$. $f$ and $g$ are inverses, so $f(g(16)) = 16$ and $g(f(3)) = 3$. Step by step: $g(16) = 4$, $f(4) = 16$; $f(3) = 8$, $g(8) = 3$. A common mistake is 11, from computing $f(3) + g(8) = 8 + 3$, which composes the wrong way round in the first term.

</details>

**q10.** Softplus is $s(x) = \ln(1 + e^x)$. Give $s^{-1}(y)$ for $y > 0$. `[expr in y]`

<details><summary>Answer</summary>

$s^{-1}(y) = \ln(e^y - 1)$. $y = \ln(1 + e^x) \Rightarrow e^y = 1 + e^x \Rightarrow e^x = e^y - 1 \Rightarrow x = \ln(e^y - 1)$, defined exactly when $e^y > 1$, that is $y > 0$, which is the range of softplus. A common mistake is $\ln(e^y + 1)$, which applies softplus again.

</details>

### Exponents, logarithms, and units of information

**q11.** $\log_2 8 + \log_2 4$. `[number]`

<details><summary>Answer</summary>

$5$. $\log_2 8 = 3$ and $\log_2 4 = 2$; equivalently $\log_2(8 \cdot 4) = \log_2 32 = 5$. A common mistake is 6, multiplying the logs ($3 \times 2$): the rule $\log(uv) = \log u + \log v$ turns a product inside the log into a sum outside, and $\log_2(8 + 4)$ is a different number again (about 3.58).

</details>

**q12.** $\log_8 32$. `[number]`

<details><summary>Answer</summary>

$5/3$. Change of base: $\log_8 32 = \log_2 32 / \log_2 8 = 5/3$. Check: $8^{5/3} = (2^3)^{5/3} = 2^5 = 32$. A common mistake is $3/5$, the change of base inverted.

</details>

**q13.** Simplify $e^{3 \ln x}$ for $x > 0$. `[expr in x]`

<details><summary>Answer</summary>

$x^3$. $3 \ln x = \ln x^3$, and $e^{\ln u} = u$. A common mistake is $3x$: the 3 is an exponent of $x$ once it leaves the logarithm.

</details>

**q14.** Solve $2^x = 1000$ for $x$. `[number]`, decimal ok

<details><summary>Answer</summary>

$x = \log_2 1000 = \dfrac{\ln 1000}{\ln 2} \approx 9.966$. Take $\ln$ of both sides: $x \ln 2 = \ln 1000$. A common mistake is $\ln 2 / \ln 1000$, the ratio inverted; a sanity check is $2^{10} = 1024$, so $x$ is just under 10.

</details>

**q15.** How many bits is one nat? `[number]`, decimal ok

<details><summary>Answer</summary>

$1/\ln 2 \approx 1.4427$ bits. An amount of information $I$ nats is $I / \ln 2$ bits, because $\log_2 u = \ln u / \ln 2$. A nat is larger than a bit ($e > 2$), so the answer must exceed 1. A common mistake is $\ln 2 \approx 0.693$, which is one bit expressed in nats.

</details>

**q16.** A model's mean negative log-likelihood is $2.0$ nats per token on a text of $1000$ tokens and $4000$ bytes. Give its bits per byte. `[number]`, decimal ok

<details><summary>Answer</summary>

$\dfrac{1}{2 \ln 2} \approx 0.7213$ bits per byte. Total information: $2.0 \times 1000 = 2000$ nats $= 2000 / \ln 2$ bits. Per byte: $(2000/\ln 2)/4000 = 1/(2 \ln 2)$. Two common mistakes: $2/\ln 2 \approx 2.885$ is bits per token (divided by tokens, not bytes), and $1/2$ is nats per byte (never converted). Bits per byte divides by bytes so that models with different tokenizers can be compared (`M00.1`, `M11.2`).

</details>

**q17.** A model's perplexity is $e^{\text{mean NLL in nats}} = 8$ per token. Give its mean NLL in bits per token. `[number]`

<details><summary>Answer</summary>

$3$. Mean NLL $= \ln 8$ nats $= \ln 8 / \ln 2 = \log_2 8 = 3$ bits. Perplexity 8 means "as uncertain as a uniform choice among 8", which takes 3 bits. A common mistake is $\ln 8 \approx 2.079$, which is the answer in nats.

</details>

**q18.** Write $a = \ln 2$ and $b = \ln 3$. Express $\ln 72$ in $a$ and $b$. `[expr in a, b]`

<details><summary>Answer</summary>

$3a + 2b$. $72 = 8 \cdot 9 = 2^3 \cdot 3^2$, and logs turn the product into a sum and the powers into factors: $\ln 72 = 3 \ln 2 + 2 \ln 3$. A common mistake is $2a + 3b$, with the exponents swapped.

</details>

**q19.** Solve $\log_3 x + \log_3 (x - 8) = 2$. `[number]`

<details><summary>Answer</summary>

$x = 9$. Both logs need positive arguments, so $x > 8$. Combine: $\log_3(x(x - 8)) = 2 \Rightarrow x^2 - 8x = 9 \Rightarrow (x - 9)(x + 1) = 0$. Of the roots 9 and $-1$, only 9 is in the domain. Check: $\log_3 9 + \log_3 1 = 2 + 0$. A common mistake is to keep $-1$, the root outside the domain $x > 8$.

</details>

**q20.** Solve $4^x = 8$. `[number]`

<details><summary>Answer</summary>

$x = 3/2$. Write both as powers of 2: $2^{2x} = 2^3$, so $2x = 3$. A common mistake is 2, from $8/4$: exponents are solved with logarithms or a common base, not by dividing.

</details>

**q21.** A quantity decays as $N(t) = N_0 e^{-kt}$ with $k > 0$. After what time is it half of $N_0$? `[expr in k]`

<details><summary>Answer</summary>

$t = \dfrac{\ln 2}{k}$. $e^{-kt} = 1/2 \Rightarrow -kt = -\ln 2$. A common mistake is $1/(2k)$, which treats the exponential as linear.

</details>

**q22.** How many decimal digits does $2^{100}$ have? (Use $\log_{10} 2 \approx 0.30103$.) `[number]`

<details><summary>Answer</summary>

$31$. A positive integer $N$ has $\lfloor \log_{10} N \rfloor + 1$ digits (a number with $d$ digits lies in $[10^{d-1}, 10^d)$). $\log_{10} 2^{100} = 100 \times 0.30103 = 30.103$, so $\lfloor 30.103 \rfloor + 1 = 31$. A common mistake is 30, forgetting the $+ 1$.

</details>

### Trigonometry, complex numbers, and Euler's formula

**q23.** $\sin(\pi/6)$. `[number]`

<details><summary>Answer</summary>

$1/2$. In the 30-60-90 triangle with hypotenuse 2, the side opposite the $30°$ angle is 1. A common mistake is $\sqrt 3/2$, which is $\cos(\pi/6)$.

</details>

**q24.** $\cos(2\pi/3)$. `[number]`

<details><summary>Answer</summary>

$-1/2$. $2\pi/3 = 120°$ is in the second quadrant, where the $x$ coordinate of the unit circle is negative; its reference angle is $\pi/3$, with cosine $1/2$. A common mistake is $+1/2$, dropping the quadrant's sign.

</details>

**q25.** Write $\cos a \cos b - \sin a \sin b$ as one trigonometric function of $a$ and $b$. `[expr in a, b]`

<details><summary>Answer</summary>

$\cos(a + b)$, the angle-addition formula. One way to see it: the real part of $e^{ia} e^{ib} = (\cos a + i \sin a)(\cos b + i \sin b)$ is $\cos a \cos b - \sin a \sin b$, and $e^{ia} e^{ib} = e^{i(a + b)}$. A common mistake is $\cos(a - b)$, which has a plus sign between the products.

</details>

**q26.** Rotate the point $(1, 0)$ counterclockwise by $\pi/3$. Give the image. `[vector]`

<details><summary>Answer</summary>

$[1/2, \sqrt 3/2]$. Rotating $(1, 0)$ by $\theta$ lands at $(\cos\theta, \sin\theta)$: the first column of $R(\theta)$ is where the first basis vector goes. A common mistake is $[\sqrt 3/2, 1/2]$, a rotation by $\pi/6$.

</details>

**q27.** $R(\pi/2)\,R(\pi/3) = R(\theta)$ with $\theta \in [0, 2\pi)$, where $R(\theta)$ is the rotation by $\theta$. Give $\theta$. `[number]`

<details><summary>Answer</summary>

$5\pi/6$. Composing rotations adds their angles: $\pi/2 + \pi/3 = 3\pi/6 + 2\pi/6$. A common mistake is $\pi/6$, from subtracting.

</details>

**q28.** $(1 + i)^8$. `[number]`

<details><summary>Answer</summary>

$16$. $(1 + i)^2 = 1 + 2i + i^2 = 2i$, so $(1 + i)^8 = (2i)^4 = 16 i^4 = 16$. In polar form: $1 + i = \sqrt 2\, e^{i\pi/4}$, and the eighth power is $(\sqrt 2)^8 e^{i 2\pi} = 16$. A common mistake is $-16$, from $i^4 = -1$; in fact $i^4 = (i^2)^2 = 1$.

</details>

**q29.** $|3 + 4i|$. `[number]`

<details><summary>Answer</summary>

$5$. $|a + bi| = \sqrt{a^2 + b^2} = \sqrt{9 + 16}$. A common mistake is 7, adding the parts: the modulus is a length, by Pythagoras.

</details>

**q30.** Rotate $3 + 4i$ by the angle $t$ with $\cos t = 3/5$ and $\sin t = 4/5$: compute $(3 + 4i)(\cos t + i \sin t)$. `[number]`

<details><summary>Answer</summary>

$-\dfrac75 + \dfrac{24}{5} i$. Expand: $(3 + 4i)(3/5 + 4i/5) = 9/5 + 12i/5 + 12i/5 + 16i^2/5 = (9 - 16)/5 + (24/5) i$. The modulus stays 5, as a rotation must: $\sqrt{49 + 576}/5 = 25/5$. Two common mistakes: 5, which is $3 \cdot \tfrac35 + 4 \cdot \tfrac45$, a dot product rather than a complex multiplication; and $-7/5 - 24i/5$, a sign slip in the imaginary part.

</details>

**q31.** $e^{i\pi/2}$. `[number]`

<details><summary>Answer</summary>

$i$. Euler's formula: $\cos(\pi/2) + i \sin(\pi/2) = 0 + i$. A quarter turn takes 1 to $i$. A common mistake is 1, which is $e^{i \cdot 0}$ (or $e^{2\pi i}$).

</details>

**q32.** Express $\cos t$ using only exponentials $e^{it}$ and $e^{-it}$. `[expr in t]`

<details><summary>Answer</summary>

$\cos t = \dfrac{e^{it} + e^{-it}}{2}$. Add $e^{it} = \cos t + i \sin t$ and $e^{-it} = \cos t - i \sin t$: the sines cancel and $2 \cos t$ remains. A common mistake is the difference over 2, which is $i \sin t$.

</details>

**q33.** For the unit complex numbers $u = e^{ia}$ and $v = e^{ib}$, simplify $\operatorname{Re}(u\,\bar{v})$, where $\bar{v}$ is the complex conjugate. `[expr in a, b]`

<details><summary>Answer</summary>

$\cos(a - b)$. $\bar v = e^{-ib}$, so $u \bar v = e^{i(a - b)}$ and its real part is $\cos(a - b)$. This is why a RoPE attention score depends only on the difference of positions (`M00.2`, section 2.5). A common mistake is $\cos(a + b)$, from forgetting the conjugate.

</details>

**q34.** A RoPE dimension pair rotates by $\omega p$ radians at position $p$, with $\omega = 0.01$. After how many positions does the pattern repeat (the period)? `[number]`

<details><summary>Answer</summary>

$200\pi \approx 628.3$ positions. The pattern repeats when the angle grows by $2\pi$: $\omega \lambda = 2\pi \Rightarrow \lambda = 2\pi/0.01$. A common mistake is $100\pi$, using $\pi$ instead of $2\pi$ as the full turn.

</details>

### Sequences and geometric series

**q35.** $1 + \tfrac12 + \tfrac14 + \tfrac18 + \cdots$ (forever). `[number]`

<details><summary>Answer</summary>

$2$. A geometric series with $a = 1$ and $|r| = 1/2 < 1$ sums to $a/(1 - r) = 1/(1/2)$. A common mistake is 1, the sum of the terms after the first.

</details>

**q36.** Give $\sum_{j=0}^{n-1} r^j$ in closed form for $r \ne 1$. `[expr in r, n]`

<details><summary>Answer</summary>

$\dfrac{1 - r^n}{1 - r}$. With $S = 1 + r + \cdots + r^{n-1}$, $S - rS = 1 - r^n$ because every middle term cancels. The sum has $n$ terms, $j = 0$ to $n - 1$. A common mistake is $(1 - r^{n+1})/(1 - r)$, which counts $n + 1$ terms.

</details>

**q37.** The sum of the first 10 terms of $3, 6, 12, 24, \ldots$ `[number]`

<details><summary>Answer</summary>

$3069$. $a = 3$, $r = 2$, $n = 10$: $3 (2^{10} - 1)/(2 - 1) = 3 \times 1023$. Common mistakes are 6141 (11 terms) and 1533 (9 terms): the exponent in $r^n$ is the number of terms.

</details>

**q38.** The RoPE ladder of a 128-dimensional head with base $10000$ is $\omega_i = 10000^{-2i/128}$ for $i = 0, \ldots, 63$. Give the ratio $\omega_{i+1}/\omega_i$. `[number]`

<details><summary>Answer</summary>

$10000^{-1/64} \approx 0.8660$. $\omega_{i+1}/\omega_i = 10000^{-2(i+1)/128 + 2i/128} = 10000^{-2/128}$. A common mistake is $10000^{-1/128}$, which forgets the factor 2 in $2i/d$ (pairs of dimensions share one frequency).

</details>

**q39.** $1 + 2 + \cdots + 100$. `[number]`

<details><summary>Answer</summary>

$5050$. Pair the first and last terms: $1 + 100 = 2 + 99 = \cdots = 101$, and there are 50 pairs; in general $n(n + 1)/2$. A common mistake is 5000, from $100 \times 100/2$.

</details>

**q40.** An exponential moving average $m_t = \beta m_{t-1} + (1 - \beta) g_t$ starts at $m_0 = 0$ with $\beta = 0.9$ and sees $g_t = 1$ at every step. Give $m_3$. `[number]`

<details><summary>Answer</summary>

$271/1000 = 0.271$. Step by step: $m_1 = 0.1$, $m_2 = 0.9 \times 0.1 + 0.1 = 0.19$, $m_3 = 0.9 \times 0.19 + 0.1 = 0.271$. In closed form, $m_t = (1 - \beta)(1 + \beta + \cdots + \beta^{t-1}) = 1 - \beta^t$, the geometric sum of q36. This is the bias toward 0 that Adam's correction divides out (`M02.2`). A common mistake is $0.729 = \beta^3$, which is $1 - m_3$.

</details>

**q41.** In q40, $m_3 = (1 - \beta)(g_3 + \beta g_2 + \beta^2 g_1)$. Give the weight on the oldest gradient $g_1$. `[number]`

<details><summary>Answer</summary>

$81/1000 = 0.081$. The weight is $(1 - \beta)\beta^2 = 0.1 \times 0.81$. A common mistake is $1/10$, the newest gradient's weight $(1 - \beta)$.

</details>

**q42.** $\lim_{n \to \infty} \dfrac{2n + 1}{n + 3}$. `[number]`

<details><summary>Answer</summary>

$2$. Divide numerator and denominator by $n$: $(2 + 1/n)/(1 + 3/n) \to 2/1$. A common mistake is $1/3$, which is the value at $n = 0$, not the limit.

</details>

**q43.** Write $0.272727\ldots$ (27 repeating) as a fraction. `[number]`

<details><summary>Answer</summary>

$3/11$. With $x = 0.2727\ldots$, $100x = 27.2727\ldots$, so $99x = 27$ and $x = 27/99 = 3/11$. Equivalently, the geometric series $27/100 + 27/100^2 + \cdots = (27/100)/(1 - 1/100)$. A common mistake is $27/100$, only the first term.

</details>

**q44.** ALiBi with 4 heads uses the slopes $2^{-2}, 2^{-4}, 2^{-6}, 2^{-8}$. Give their sum. `[number]`

<details><summary>Answer</summary>

$85/256$. Over the common denominator 256: $64 + 16 + 4 + 1 = 85$. As a geometric sum with $a = 1/4$, $r = 1/4$, $n = 4$: $(1/4)(1 - 4^{-4})/(3/4) = (1 - 1/256)/3 = 85/256$. A common mistake is $1/3$, the infinite sum.

</details>

### Polynomials

**q45.** Evaluate $p(x) = 2 - 3x + x^2 + 4x^3$ at $x = 2$ (Horner's rule makes it three multiply-adds). `[number]`

<details><summary>Answer</summary>

$32$. Horner from the highest coefficient down, $4, 1, -3, 2$: $4 \to 4 \cdot 2 + 1 = 9 \to 9 \cdot 2 - 3 = 15 \to 15 \cdot 2 + 2 = 32$. Directly: $2 - 6 + 4 + 32$. A common mistake is 10, from running Horner with the coefficients in the order written ($2, -3, 1, 4$), which evaluates the reversed polynomial.

</details>

**q46.** Divide $x^3 - 6x^2 + 11x - 6$ by $x - 1$. Give the quotient. `[expr in x]`

<details><summary>Answer</summary>

$x^2 - 5x + 6$. Synthetic division by the root 1 is Horner at $x = 1$: $1 \to 1 - 6 = -5 \to -5 + 11 = 6 \to 6 - 6 = 0$. The intermediate values are the quotient's coefficients and the last is the remainder, 0, so $x = 1$ is a root. A common mistake is $x^2 - 7x + 18$, which divides by $x + 1$ (Horner at $-1$).

</details>

**q47.** Give all roots of $x^3 - 6x^2 + 11x - 6$. `[set]`

<details><summary>Answer</summary>

$\{1, 2, 3\}$. From q46, $x^3 - 6x^2 + 11x - 6 = (x - 1)(x^2 - 5x + 6) = (x - 1)(x - 2)(x - 3)$. Vieta checks it: the roots sum to 6 and multiply to 6. A common mistake is $\{-1, -2, -3\}$, reading the roots off the factors' constants without the sign.

</details>

**q48.** Give the roots of $2x^2 + 3x - 2$. `[set]`

<details><summary>Answer</summary>

$\{1/2, -2\}$. Factor: $2x^2 + 3x - 2 = (2x - 1)(x + 2)$, or use $x = (-3 \pm \sqrt{9 + 16})/4 = (-3 \pm 5)/4$. A common mistake is $\{-1/2, 2\}$, with both signs flipped.

</details>

**q49.** Give the smaller root of $x^2 - 10^6 x + 1$ exactly (radicals allowed) or to 15 significant digits. `[number]`

<details><summary>Answer</summary>

$500000 - \sqrt{249999999999} \approx 1.00000000000100 \times 10^{-6}$. The quadratic formula gives $x = (10^6 \pm \sqrt{10^{12} - 4})/2 = 500000 \pm \sqrt{249999999999}$. Evaluated naively in floating point, the minus sign subtracts two numbers that agree in about 12 digits and loses them. The stable route (`M00.4`, pitfall 2) computes the large root $x_1 = 500000 + \sqrt{249999999999}$ first and uses Vieta, $x_1 x_2 = c/a = 1$, so $x_2 = 1/x_1 = 10^{-6} + 10^{-18} + \cdots$. A common mistake is $10^{-6}$ exactly (the approximation $c/b$), which drops the $10^{-18}$ correction that the 15th significant digit shows.

</details>

**q50.** Without solving, give the product of the roots of $3x^2 - 7x + 2$. `[number]`

<details><summary>Answer</summary>

$2/3$. Vieta: for $ax^2 + bx + c$, the product of the roots is $c/a$. (The roots are $2$ and $1/3$.) A common mistake is 2, which is $c$ without dividing by $a$.

</details>

**q51.** How many real roots does $x^2 + x + 1$ have? `[number]`

<details><summary>Answer</summary>

$0$. The discriminant is $b^2 - 4ac = 1 - 4 = -3 < 0$, so both roots are complex. Equivalently, $x^2 + x + 1 = (x + 1/2)^2 + 3/4 > 0$. A common mistake is 2, from counting the roots of a quadratic without checking the discriminant.

</details>

**q52.** The Taylor polynomial of $e^x$ of degree 3 around 0 is $c_0 + c_1 x + c_2 x^2 + c_3 x^3$. Give $[c_0, c_1, c_2, c_3]$. `[vector]`

<details><summary>Answer</summary>

$[1, 1, 1/2, 1/6]$. $c_k = f^{(k)}(0)/k!$, and every derivative of $e^x$ is $e^x$, equal to 1 at 0, so $c_k = 1/k!$: $1/0!, 1/1!, 1/2!, 1/3!$. A common mistake is $[1, 1, 1/2, 1/3]$, using $1/k$ instead of $1/k!$.

</details>

### Inequalities

**q53.** Solve $|x - 3| < 2$. `[interval]`

<details><summary>Answer</summary>

$(1, 5)$. "Within distance 2 of 3": $-2 < x - 3 < 2$, so $1 < x < 5$. The inequality is strict, so the endpoints are excluded. A common mistake is $[1, 5]$.

</details>

**q54.** Solve $x^2 - 5x + 6 > 0$. `[interval]`

<details><summary>Answer</summary>

$(-\infty, 2) \cup (3, \infty)$. $x^2 - 5x + 6 = (x - 2)(x - 3)$ is zero at 2 and 3; the parabola opens upward, so it is positive outside the roots. A common mistake is $(2, 3)$, which is where it is negative.

</details>

**q55.** The smallest integer $n$ with $2^n > 1000$ (the bits needed to number 1000 items). `[number]`

<details><summary>Answer</summary>

$10$. $2^9 = 512 < 1000 < 1024 = 2^{10}$; equivalently $n > \log_2 1000 \approx 9.97$ (q14), and the smallest such integer is 10. A common mistake is 9, rounding $\log_2 1000$ down.

</details>

**q56.** Solve $\dfrac{x - 1}{x + 2} \ge 0$. `[interval]`

<details><summary>Answer</summary>

$(-\infty, -2) \cup [1, \infty)$. The numerator is zero at 1 and the denominator at $-2$. Signs: for $x < -2$ both are negative, quotient positive; for $-2 < x < 1$ the quotient is negative; for $x > 1$ positive. $x = 1$ gives 0, allowed by $\ge$; $x = -2$ divides by zero. A common mistake is $(-\infty, -2] \cup [1, \infty)$, which includes the point where the expression is undefined.

</details>

**q57.** For which probabilities $p \in (0, 1)$ is the surprisal $-\log_2 p$ greater than 3 bits? `[interval]`

<details><summary>Answer</summary>

$(0, 1/8)$. $-\log_2 p > 3 \iff \log_2 p < -3 \iff p < 2^{-3}$: multiplying by $-1$ flips the inequality, and $2^x$ is increasing, so applying it keeps the direction. Rarer events are more surprising. A common mistake is $(1/8, 1)$, which forgets the flip.

</details>

**q58.** Solve $e^x \le 2$. `[interval]`

<details><summary>Answer</summary>

$(-\infty, \ln 2]$. $\ln$ is strictly increasing, so applying it keeps the direction: $x \le \ln 2$. Equality is allowed, so the endpoint is included. A common mistake is $(-\infty, \ln 2)$, dropping the endpoint.

</details>

**q59.** The smallest value of $x + 1/x$ over $x > 0$. `[number]`

<details><summary>Answer</summary>

$2$, at $x = 1$. Multiply $x + 1/x \ge 2$ by $x > 0$ (direction kept): $x^2 + 1 \ge 2x \iff (x - 1)^2 \ge 0$, which always holds, with equality only at $x = 1$, where $1 + 1 = 2$. A common mistake is 1, the minimizer instead of the minimum.

</details>

**q60.** Solve $\ln x < x - 1$ over $x > 0$. `[interval]`

<details><summary>Answer</summary>

$(0, 1) \cup (1, \infty)$. $\ln x \le x - 1$ for every $x > 0$, with equality only at $x = 1$ (the line $x - 1$ is tangent to $\ln x$ there, and $\ln$ curves below its tangents). The strict inequality therefore holds everywhere except $x = 1$. A common mistake is $(0, \infty)$, which includes the point of equality.

</details>

## 5. Pitfalls

| Pitfall | Where it bites later | Where the worked answers address it |
|---|---|---|
| inverting the steps in the original order ($f^{-1}(x) = (x - 7)/3$ for $f(x) = 3x - 7$) | every inverse in `M07.1` (inverse-CDF sampling) | q1, q3 |
| composing in the wrong order: $g(f(x))$ instead of $f(g(x))$ | the chain rule in `M01.2` and backpropagation in `L0.1` | q4, q9 |
| a closed endpoint where the expression is undefined | the domain of $\ln$ in every loss | q7, q56 |
| $\log(a + b) = \log a + \log b$ | rewriting losses in `M11.1` | q11, q18 |
| one term too many, or the wrong weight, in a geometric sum | Adam's bias correction (`M02.2`) | q37, q40, q41 |
| decimals where an exact value exists | a rounded value hides whether the derivation was right, and q49 shows a case where rounding is the whole bug | every answer gives the exact form first and a decimal only as a check |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `M00.1` | part 2 is the pen-and-paper side of its logarithms and units |
| Forward | `M00.2` | part 3: rotations, Euler's formula, and the relative-angle identity of q33 |
| Forward | `M00.3` | part 4: geometric sums, the RoPE ratio (q38), ALiBi slopes (q44), EMA weights (q40, q41) |
| Forward | `M00.4` | part 5: Horner (q45), factoring, Vieta, and the cancellation case (q49) |
| Forward | `M01.1` | limits (q42) and monotonicity (q8) start calculus |
| Forward | `M05.2` | functions, inverses, and bijections in discrete math |
| Forward | `M07.1` | inverse functions become inverse-CDF sampling |
