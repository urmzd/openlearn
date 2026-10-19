# Problem set: functions, logarithms, trigonometry, series, polynomials, and inequalities

The sections that follow this one turn the mathematics under a language-model system into code: logarithms that measure information, rotations that encode position, ladders of frequencies, and polynomial evaluation that kernels run, and after them calculus, linear algebra, probability, and optimization. Each of them assumes you can manipulate the expressions fluently by hand: invert a function, combine logarithms, use an angle-addition formula, sum a geometric series, factor a polynomial, and solve an inequality. A bug in a derivation becomes a bug in code that tests catch late and explain poorly, so this set practises the hand skills first.

The set has sixty problems in six parts. Four of the parts are the pen-and-paper side of the four sections of this chapter, and if a problem in one of them stops you, the section that teaches it is named at the start of the part. The other two parts, on functions and inverses and on inequalities, have no section of their own, because none of that material becomes code; this section teaches it below, before the problems. [Derivatives and finite differences](../02-calculus-1/01-derivatives-and-finite-differences.md) and [Injective, surjective, bijective](../06-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md) build on it directly.

Every problem has a worked answer folded under **Answer**, with the derivation and the most common wrong answer and why it is wrong. Attempt each problem on paper before opening its answer. The set is self-contained; nothing here needs code.

> **Learning objectives**
> - Find the domain, range, composition, and inverse of a function, and use monotonicity to decide whether an inverse exists.
> - Solve linear, polynomial, rational, absolute-value, exponential, and logarithmic inequalities, keeping track of when the direction flips and which endpoints belong.
> - Manipulate logarithms, convert between nats and bits, and compute bits per byte by hand.
> - Use angle addition, rotations, and Euler's formula, and sum geometric series including EMA weights and ALiBi slopes.
> - Evaluate polynomials with Horner's rule, factor them, apply Vieta's formulas, and recognise when the quadratic formula cancels.

## Functions, domains, ranges, and composition

The symbols used in the teaching part are collected here first:

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

A **function** $f$ assigns to each input $x$ in its **domain** exactly one output $f(x)$. A formula alone leaves the domain implicit: it is every real $x$ where the formula makes sense. $\ln(4 - x^2)$ needs $4 - x^2 > 0$, so its domain is $(-2, 2)$; $1/(x - 3)$ needs $x \ne 3$. The **range** is the set of outputs that actually occur. The sigmoid $\sigma(z) = 1/(1 + e^{-z})$ takes every value strictly between 0 and 1 and never reaches either, so its range is $(0, 1)$.

**Composition** chains functions: $(f \circ g)(x) = f(g(x))$ applies $g$ first. Order matters. With $f(x) = x^2$ and $g(x) = x + 1$, $f(g(x)) = (x + 1)^2$ but $g(f(x)) = x^2 + 1$.

## Inverses and monotonicity

A function $f$ is **one-to-one** when different inputs give different outputs. Then every output $y$ in the range comes from exactly one $x$, and the **inverse** $f^{-1}(y) = x$ undoes $f$: $f^{-1}(f(x)) = x$ and $f(f^{-1}(y)) = y$. To find it, write $y = f(x)$ and solve for $x$. The graph of $f^{-1}$ is the graph of $f$ reflected across the line $y = x$.

How do you know a function is one-to-one? A function is **strictly increasing** when $x_1 < x_2$ implies $f(x_1) < f(x_2)$, and strictly decreasing when it implies $f(x_1) > f(x_2)$. Either kind, called strictly **monotonic**, is one-to-one, because different inputs are ordered and so are their outputs. $e^x$, $\ln x$, $x^3 + x$, $\sigma$, and softplus $\ln(1 + e^x)$ are strictly increasing, which is why all of them can be inverted. $x^2$ is not monotonic on all of $\mathbb{R}$, since $(-2)^2 = 2^2$, so it has an inverse only on a half-line, where $\sqrt{\ }$ is it. [Derivatives and finite differences](../02-calculus-1/01-derivatives-and-finite-differences.md) gives the practical test: a function whose derivative is positive everywhere is strictly increasing.

## Inequalities

An inequality is solved the way an equation is, with one extra rule: **multiplying or dividing both sides by a negative number reverses the direction** ($-2x < 6$ means $x > -3$), and so does applying a strictly decreasing function. Applying a strictly increasing function ($e^x$, $\ln$, $\sqrt{\ }$ on non-negative numbers) keeps the direction, which is how $e^x \le 2$ becomes $x \le \ln 2$.

For a polynomial or a quotient of polynomials, find where it is zero or undefined, cut the line there, and test the sign on each piece. $x^2 - 5x + 6 = (x - 2)(x - 3)$ is zero at 2 and 3, positive outside and negative between, so $x^2 - 5x + 6 > 0$ on $(-\infty, 2) \cup (3, \infty)$. An endpoint where the expression is zero belongs to the answer for $\ge$ and $\le$; an endpoint where it is undefined never does. An absolute value is a distance: $|x - 3| < 2$ means "within 2 of 3", the interval $(1, 5)$.

Two inequalities appear again in optimization and information theory. The first is $x + 1/x \ge 2$ for $x > 0$, with equality at $x = 1$; multiply by $x$ and get $(x - 1)^2 \ge 0$. The second is $\ln x \le x - 1$, with equality only at $x = 1$, the inequality behind Gibbs' inequality in [Entropy, cross-entropy, and KL](../12-information-theory/01-entropy-cross-entropy-and-kl.md).

## Where the other parts are taught

The four middle parts each correspond to one section of this chapter:

| Part | Problems | Taught in |
|---|---|---|
| Exponents, logarithms, units | 11 to 22 | [Exponents, logarithms, and units of information](01-exponents-logs-and-units-of-information.md) |
| Trigonometry, complex numbers, Euler | 23 to 34 | [Trigonometry, rotations, complex numbers, and Euler's formula](02-trig-rotations-and-eulers-formula.md) |
| Sequences and geometric series | 35 to 44 | [Sequences, geometric series, and frequency ladders](03-sequences-and-frequency-ladders.md) |
| Polynomials | 45 to 52 | [Polynomials, Horner's rule, and stable quadratic roots](04-polynomials-horner-and-stable-roots.md) |

## Working two problems by hand

Before the set itself, here are two worked problems that are siblings of problems in it, showing the level of care the answers expect.

The first is a sibling of problems 5 and 6: find the inverse and range of $f(x) = 2/(1 + e^{-x})$. Start with the range. $e^{-x}$ takes every value in $(0, \infty)$, so $1 + e^{-x}$ takes every value in $(1, \infty)$ and $2/(1 + e^{-x})$ every value in $(0, 2)$, approaching 2 as $x \to \infty$ and 0 as $x \to -\infty$ without reaching either. The range is `(0, 2)`. Next, check that $f$ is invertible: as $x$ grows, $e^{-x}$ shrinks, the denominator shrinks, and $f(x)$ grows, so $f$ is strictly increasing and hence one-to-one on $\mathbb{R}$. Now solve $y = f(x)$ for $x$, for $0 < y < 2$:

$$y(1 + e^{-x}) = 2 \;\Rightarrow\; e^{-x} = \frac{2}{y} - 1 = \frac{2 - y}{y} \;\Rightarrow\; -x = \ln\frac{2 - y}{y} \;\Rightarrow\; x = \ln\frac{y}{2 - y}.$$

Check it: $f(0) = 2/(1 + 1) = 1$, and $f^{-1}(1) = \ln(1/1) = 0$. So the answer, in the variable of the question, is $f^{-1}(y) = \ln\dfrac{y}{2 - y}$. Equivalent forms such as $\ln y - \ln(2 - y)$ or $-\ln(2/y - 1)$ are the same function; what matters is equality, not spelling, and a quick test at a couple of points, as in the check, is how to confirm two forms agree.

The second is a sibling of problem 56: solve $\dfrac{x + 1}{x - 4} \le 0$. The numerator is zero at $-1$ and the denominator at $4$. For $x < -1$ both factors are negative and the quotient is positive; for $-1 < x < 4$ the numerator is positive and the denominator negative, so the quotient is negative; for $x > 4$ it is positive. Zero at $x = -1$ is allowed by $\le$, while $x = 4$ divides by zero. The answer is `[-1, 4)`.

## How the problems are stated

The tag after each question is the kind of answer expected: an expression in the named variables, a number, an interval, a set, a vector, or true/false. Exact values are expected where they exist; a decimal is given alongside only as a check.

## Problems on functions and inverses

The first ten problems practise domains, ranges, composition, and inverses, including the inverses of the sigmoid and softplus that recur throughout the books.

**1.** $f(x) = 3x - 7$. Give $f^{-1}(x)$. `[expr in x]`

<details><summary>Answer</summary>

$f^{-1}(x) = \dfrac{x + 7}{3}$. $f$ multiplies by 3, then subtracts 7, so the inverse undoes those steps in reverse order: add 7, then divide by 3. Formally, $y = 3x - 7 \Rightarrow x = (y + 7)/3$, then rename $y$ to $x$. Check: $f^{-1}(f(2)) = f^{-1}(-1) = 6/3 = 2$. A common mistake is $(x - 7)/3$, which undoes the steps in the original order.

</details>

**2.** $f(x) = e^{2x + 1}$. Give $f^{-1}(x)$ for $x > 0$. `[expr in x]`

<details><summary>Answer</summary>

$f^{-1}(x) = \dfrac{\ln x - 1}{2}$. From $y = e^{2x + 1}$, take $\ln$ (defined because $y > 0$): $\ln y = 2x + 1$, so $x = (\ln y - 1)/2$. A common mistake is $\ln(x)/2 - 1$, which halves before subtracting: the last step of $f$ (exponentiate) is undone first, then "plus 1", then "times 2".

</details>

**3.** $g(x) = \dfrac{2x + 1}{x - 3}$ for $x \ne 3$. Give $g^{-1}(x)$. `[expr in x]`

<details><summary>Answer</summary>

$g^{-1}(x) = \dfrac{3x + 1}{x - 2}$. Solve $y = (2x + 1)/(x - 3)$: $y(x - 3) = 2x + 1 \Rightarrow yx - 2x = 3y + 1 \Rightarrow x(y - 2) = 3y + 1 \Rightarrow x = (3y + 1)/(y - 2)$. Check: $g(4) = 9$, and $(27 + 1)/(9 - 2) = 4$. A common mistake is the reciprocal $(x - 3)/(2x + 1)$: $1/g$ is not $g^{-1}$.

</details>

**4.** $f(x) = x^2$ and $g(x) = x + 1$. Give $(f \circ g)(x) = f(g(x))$. `[expr in x]`

<details><summary>Answer</summary>

$(x + 1)^2$. $g$ runs first and gives $x + 1$; $f$ squares that. A common mistake is $x^2 + 1$, which is $g(f(x))$: composition is applied right to left.

</details>

**5.** The sigmoid is $\sigma(z) = 1 / (1 + e^{-z})$. Give its inverse, the logit $\sigma^{-1}(p)$, for $0 < p < 1$. `[expr in p]`

<details><summary>Answer</summary>

$\sigma^{-1}(p) = \ln\dfrac{p}{1 - p}$. As in the worked example with 2 replaced by 1: $p(1 + e^{-z}) = 1 \Rightarrow e^{-z} = (1 - p)/p \Rightarrow z = \ln(p/(1 - p))$. This is the logit a classifier head outputs (see [Fine-tuning heads](../../build-an-llm/07-objectives/05-fine-tuning-heads-and-the-linear-head-export.md) in *Build Your Own LLM System*). Check: $\sigma(0) = 1/2$ and $\ln(1) = 0$. A common mistake is $\ln((1 - p)/p)$, which forgets the sign from $-z$.

</details>

**6.** Give the range of $\sigma$ (the set of values it takes over all real $z$). `[interval]`

<details><summary>Answer</summary>

$(0, 1)$. $1 + e^{-z}$ takes every value in $(1, \infty)$, so its reciprocal takes every value in $(0, 1)$, approaching but never reaching either end. A common mistake is $[0, 1]$: the sigmoid never outputs exactly 0 or 1, which is why $\ln \sigma$ is always finite.

</details>

**7.** Give the domain of $h(x) = \ln(4 - x^2)$ (the real $x$ where it is defined). `[interval]`

<details><summary>Answer</summary>

$(-2, 2)$. $\ln$ needs a positive argument: $4 - x^2 > 0 \iff x^2 < 4 \iff -2 < x < 2$. A common mistake is $[-2, 2]$: at $x = \pm 2$ the argument is 0 and $\ln 0$ is undefined.

</details>

**8.** Is $f(x) = x^3 + x$ one-to-one on all of $\mathbb{R}$? (Hint: is it monotonic?) `[bool]`

<details><summary>Answer</summary>

True. Both $x^3$ and $x$ are strictly increasing, and a sum of strictly increasing functions is strictly increasing, so $x_1 < x_2$ gives $f(x_1) < f(x_2)$ and no two inputs share an output. (With the derivative from [Derivatives and finite differences](../02-calculus-1/01-derivatives-and-finite-differences.md): $f'(x) = 3x^2 + 1 > 0$ everywhere.)

</details>

**9.** $f(x) = 2^x$ and $g(x) = \log_2 x$. Compute $f(g(16)) + g(f(3))$. `[number]`

<details><summary>Answer</summary>

$19$. $f$ and $g$ are inverses, so $f(g(16)) = 16$ and $g(f(3)) = 3$. Step by step: $g(16) = 4$, $f(4) = 16$; $f(3) = 8$, $g(8) = 3$. A common mistake is 11, from computing $f(3) + g(8) = 8 + 3$, which composes the wrong way round in the first term.

</details>

**10.** Softplus is $s(x) = \ln(1 + e^x)$. Give $s^{-1}(y)$ for $y > 0$. `[expr in y]`

<details><summary>Answer</summary>

$s^{-1}(y) = \ln(e^y - 1)$. $y = \ln(1 + e^x) \Rightarrow e^y = 1 + e^x \Rightarrow e^x = e^y - 1 \Rightarrow x = \ln(e^y - 1)$, defined exactly when $e^y > 1$, that is $y > 0$, which is the range of softplus. A common mistake is $\ln(e^y + 1)$, which applies softplus again.

</details>

## Problems on exponents, logarithms, and units

Problems 11 to 22 are the pen-and-paper side of [Exponents, logarithms, and units of information](01-exponents-logs-and-units-of-information.md): the laws of logarithms, change of base, nats and bits, and bits per byte.

**11.** $\log_2 8 + \log_2 4$. `[number]`

<details><summary>Answer</summary>

$5$. $\log_2 8 = 3$ and $\log_2 4 = 2$; equivalently $\log_2(8 \cdot 4) = \log_2 32 = 5$. A common mistake is 6, multiplying the logs ($3 \times 2$): the rule $\log(uv) = \log u + \log v$ turns a product inside the log into a sum outside, and $\log_2(8 + 4)$ is a different number again (about 3.58).

</details>

**12.** $\log_8 32$. `[number]`

<details><summary>Answer</summary>

$5/3$. Change of base: $\log_8 32 = \log_2 32 / \log_2 8 = 5/3$. Check: $8^{5/3} = (2^3)^{5/3} = 2^5 = 32$. A common mistake is $3/5$, the change of base inverted.

</details>

**13.** Simplify $e^{3 \ln x}$ for $x > 0$. `[expr in x]`

<details><summary>Answer</summary>

$x^3$. $3 \ln x = \ln x^3$, and $e^{\ln u} = u$. A common mistake is $3x$: the 3 is an exponent of $x$ once it leaves the logarithm.

</details>

**14.** Solve $2^x = 1000$ for $x$. `[number]`, decimal ok

<details><summary>Answer</summary>

$x = \log_2 1000 = \dfrac{\ln 1000}{\ln 2} \approx 9.966$. Take $\ln$ of both sides: $x \ln 2 = \ln 1000$. A common mistake is $\ln 2 / \ln 1000$, the ratio inverted; a sanity check is $2^{10} = 1024$, so $x$ is just under 10.

</details>

**15.** How many bits is one nat? `[number]`, decimal ok

<details><summary>Answer</summary>

$1/\ln 2 \approx 1.4427$ bits. An amount of information $I$ nats is $I / \ln 2$ bits, because $\log_2 u = \ln u / \ln 2$. A nat is larger than a bit ($e > 2$), so the answer must exceed 1. A common mistake is $\ln 2 \approx 0.693$, which is one bit expressed in nats.

</details>

**16.** A model's mean negative log-likelihood is $2.0$ nats per token on a text of $1000$ tokens and $4000$ bytes. Give its bits per byte. `[number]`, decimal ok

<details><summary>Answer</summary>

$\dfrac{1}{2 \ln 2} \approx 0.7213$ bits per byte. Total information: $2.0 \times 1000 = 2000$ nats $= 2000 / \ln 2$ bits. Per byte: $(2000/\ln 2)/4000 = 1/(2 \ln 2)$. Two common mistakes: $2/\ln 2 \approx 2.885$ is bits per token (divided by tokens, not bytes), and $1/2$ is nats per byte (never converted). Bits per byte divides by bytes so that models with different tokenizers can be compared (see [Exponents, logarithms, and units of information](01-exponents-logs-and-units-of-information.md) and [Perplexity and bits per byte](../12-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md)).

</details>

**17.** A model's perplexity is $e^{\text{mean NLL in nats}} = 8$ per token. Give its mean NLL in bits per token. `[number]`

<details><summary>Answer</summary>

$3$. Mean NLL $= \ln 8$ nats $= \ln 8 / \ln 2 = \log_2 8 = 3$ bits. Perplexity 8 means "as uncertain as a uniform choice among 8", which takes 3 bits. A common mistake is $\ln 8 \approx 2.079$, which is the answer in nats.

</details>

**18.** Write $a = \ln 2$ and $b = \ln 3$. Express $\ln 72$ in $a$ and $b$. `[expr in a, b]`

<details><summary>Answer</summary>

$3a + 2b$. $72 = 8 \cdot 9 = 2^3 \cdot 3^2$, and logs turn the product into a sum and the powers into factors: $\ln 72 = 3 \ln 2 + 2 \ln 3$. A common mistake is $2a + 3b$, with the exponents swapped.

</details>

**19.** Solve $\log_3 x + \log_3 (x - 8) = 2$. `[number]`

<details><summary>Answer</summary>

$x = 9$. Both logs need positive arguments, so $x > 8$. Combine: $\log_3(x(x - 8)) = 2 \Rightarrow x^2 - 8x = 9 \Rightarrow (x - 9)(x + 1) = 0$. Of the roots 9 and $-1$, only 9 is in the domain. Check: $\log_3 9 + \log_3 1 = 2 + 0$. A common mistake is to keep $-1$, the root outside the domain $x > 8$.

</details>

**20.** Solve $4^x = 8$. `[number]`

<details><summary>Answer</summary>

$x = 3/2$. Write both as powers of 2: $2^{2x} = 2^3$, so $2x = 3$. A common mistake is 2, from $8/4$: exponents are solved with logarithms or a common base, not by dividing.

</details>

**21.** A quantity decays as $N(t) = N_0 e^{-kt}$ with $k > 0$. After what time is it half of $N_0$? `[expr in k]`

<details><summary>Answer</summary>

$t = \dfrac{\ln 2}{k}$. $e^{-kt} = 1/2 \Rightarrow -kt = -\ln 2$. A common mistake is $1/(2k)$, which treats the exponential as linear.

</details>

**22.** How many decimal digits does $2^{100}$ have? (Use $\log_{10} 2 \approx 0.30103$.) `[number]`

<details><summary>Answer</summary>

$31$. A positive integer $N$ has $\lfloor \log_{10} N \rfloor + 1$ digits (a number with $d$ digits lies in $[10^{d-1}, 10^d)$). $\log_{10} 2^{100} = 100 \times 0.30103 = 30.103$, so $\lfloor 30.103 \rfloor + 1 = 31$. A common mistake is 30, forgetting the $+ 1$.

</details>

## Problems on trigonometry, complex numbers, and Euler's formula

Problems 23 to 34 go with [Trigonometry, rotations, complex numbers, and Euler's formula](02-trig-rotations-and-eulers-formula.md): unit-circle values, angle addition, rotations as complex multiplication, and the relative-angle identity behind RoPE.

**23.** $\sin(\pi/6)$. `[number]`

<details><summary>Answer</summary>

$1/2$. In the 30-60-90 triangle with hypotenuse 2, the side opposite the $30°$ angle is 1. A common mistake is $\sqrt 3/2$, which is $\cos(\pi/6)$.

</details>

**24.** $\cos(2\pi/3)$. `[number]`

<details><summary>Answer</summary>

$-1/2$. $2\pi/3 = 120°$ is in the second quadrant, where the $x$ coordinate of the unit circle is negative; its reference angle is $\pi/3$, with cosine $1/2$. A common mistake is $+1/2$, dropping the quadrant's sign.

</details>

**25.** Write $\cos a \cos b - \sin a \sin b$ as one trigonometric function of $a$ and $b$. `[expr in a, b]`

<details><summary>Answer</summary>

$\cos(a + b)$, the angle-addition formula. One way to see it: the real part of $e^{ia} e^{ib} = (\cos a + i \sin a)(\cos b + i \sin b)$ is $\cos a \cos b - \sin a \sin b$, and $e^{ia} e^{ib} = e^{i(a + b)}$. A common mistake is $\cos(a - b)$, which has a plus sign between the products.

</details>

**26.** Rotate the point $(1, 0)$ counterclockwise by $\pi/3$. Give the image. `[vector]`

<details><summary>Answer</summary>

$[1/2, \sqrt 3/2]$. Rotating $(1, 0)$ by $\theta$ lands at $(\cos\theta, \sin\theta)$: the first column of $R(\theta)$ is where the first basis vector goes. A common mistake is $[\sqrt 3/2, 1/2]$, a rotation by $\pi/6$.

</details>

**27.** $R(\pi/2)\,R(\pi/3) = R(\theta)$ with $\theta \in [0, 2\pi)$, where $R(\theta)$ is the rotation by $\theta$. Give $\theta$. `[number]`

<details><summary>Answer</summary>

$5\pi/6$. Composing rotations adds their angles: $\pi/2 + \pi/3 = 3\pi/6 + 2\pi/6$. A common mistake is $\pi/6$, from subtracting.

</details>

**28.** $(1 + i)^8$. `[number]`

<details><summary>Answer</summary>

$16$. $(1 + i)^2 = 1 + 2i + i^2 = 2i$, so $(1 + i)^8 = (2i)^4 = 16 i^4 = 16$. In polar form: $1 + i = \sqrt 2\, e^{i\pi/4}$, and the eighth power is $(\sqrt 2)^8 e^{i 2\pi} = 16$. A common mistake is $-16$, from $i^4 = -1$; in fact $i^4 = (i^2)^2 = 1$.

</details>

**29.** $|3 + 4i|$. `[number]`

<details><summary>Answer</summary>

$5$. $|a + bi| = \sqrt{a^2 + b^2} = \sqrt{9 + 16}$. A common mistake is 7, adding the parts: the modulus is a length, by Pythagoras.

</details>

**30.** Rotate $3 + 4i$ by the angle $t$ with $\cos t = 3/5$ and $\sin t = 4/5$: compute $(3 + 4i)(\cos t + i \sin t)$. `[number]`

<details><summary>Answer</summary>

$-\dfrac75 + \dfrac{24}{5} i$. Expand: $(3 + 4i)(3/5 + 4i/5) = 9/5 + 12i/5 + 12i/5 + 16i^2/5 = (9 - 16)/5 + (24/5) i$. The modulus stays 5, as a rotation must: $\sqrt{49 + 576}/5 = 25/5$. Two common mistakes: 5, which is $3 \cdot \tfrac35 + 4 \cdot \tfrac45$, a dot product rather than a complex multiplication; and $-7/5 - 24i/5$, a sign slip in the imaginary part.

</details>

**31.** $e^{i\pi/2}$. `[number]`

<details><summary>Answer</summary>

$i$. Euler's formula: $\cos(\pi/2) + i \sin(\pi/2) = 0 + i$. A quarter turn takes 1 to $i$. A common mistake is 1, which is $e^{i \cdot 0}$ (or $e^{2\pi i}$).

</details>

**32.** Express $\cos t$ using only exponentials $e^{it}$ and $e^{-it}$. `[expr in t]`

<details><summary>Answer</summary>

$\cos t = \dfrac{e^{it} + e^{-it}}{2}$. Add $e^{it} = \cos t + i \sin t$ and $e^{-it} = \cos t - i \sin t$: the sines cancel and $2 \cos t$ remains. A common mistake is the difference over 2, which is $i \sin t$.

</details>

**33.** For the unit complex numbers $u = e^{ia}$ and $v = e^{ib}$, simplify $\operatorname{Re}(u\,\bar{v})$, where $\bar{v}$ is the complex conjugate. `[expr in a, b]`

<details><summary>Answer</summary>

$\cos(a - b)$. $\bar v = e^{-ib}$, so $u \bar v = e^{i(a - b)}$ and its real part is $\cos(a - b)$. This is why a RoPE attention score depends only on the difference of positions (see [Why RoPE rotates pairs](02-trig-rotations-and-eulers-formula.md#why-rope-rotates-pairs)). A common mistake is $\cos(a + b)$, from forgetting the conjugate.

</details>

**34.** A RoPE dimension pair rotates by $\omega p$ radians at position $p$, with $\omega = 0.01$. After how many positions does the pattern repeat (the period)? `[number]`

<details><summary>Answer</summary>

$200\pi \approx 628.3$ positions. The pattern repeats when the angle grows by $2\pi$: $\omega \lambda = 2\pi \Rightarrow \lambda = 2\pi/0.01$. A common mistake is $100\pi$, using $\pi$ instead of $2\pi$ as the full turn.

</details>

## Problems on sequences and geometric series

Problems 35 to 44 go with [Sequences, geometric series, and frequency ladders](03-sequences-and-frequency-ladders.md): geometric sums, the RoPE ratio, EMA weights, and ALiBi slopes.

**35.** $1 + \tfrac12 + \tfrac14 + \tfrac18 + \cdots$ (forever). `[number]`

<details><summary>Answer</summary>

$2$. A geometric series with $a = 1$ and $|r| = 1/2 < 1$ sums to $a/(1 - r) = 1/(1/2)$. A common mistake is 1, the sum of the terms after the first.

</details>

**36.** Give $\sum_{j=0}^{n-1} r^j$ in closed form for $r \ne 1$. `[expr in r, n]`

<details><summary>Answer</summary>

$\dfrac{1 - r^n}{1 - r}$. With $S = 1 + r + \cdots + r^{n-1}$, $S - rS = 1 - r^n$ because every middle term cancels. The sum has $n$ terms, $j = 0$ to $n - 1$. A common mistake is $(1 - r^{n+1})/(1 - r)$, which counts $n + 1$ terms.

</details>

**37.** The sum of the first 10 terms of $3, 6, 12, 24, \ldots$ `[number]`

<details><summary>Answer</summary>

$3069$. $a = 3$, $r = 2$, $n = 10$: $3 (2^{10} - 1)/(2 - 1) = 3 \times 1023$. Common mistakes are 6141 (11 terms) and 1533 (9 terms): the exponent in $r^n$ is the number of terms.

</details>

**38.** The RoPE ladder of a 128-dimensional head with base $10000$ is $\omega_i = 10000^{-2i/128}$ for $i = 0, \ldots, 63$. Give the ratio $\omega_{i+1}/\omega_i$. `[number]`

<details><summary>Answer</summary>

$10000^{-1/64} \approx 0.8660$. $\omega_{i+1}/\omega_i = 10000^{-2(i+1)/128 + 2i/128} = 10000^{-2/128}$. A common mistake is $10000^{-1/128}$, which forgets the factor 2 in $2i/d$ (pairs of dimensions share one frequency).

</details>

**39.** $1 + 2 + \cdots + 100$. `[number]`

<details><summary>Answer</summary>

$5050$. Pair the first and last terms: $1 + 100 = 2 + 99 = \cdots = 101$, and there are 50 pairs; in general $n(n + 1)/2$. A common mistake is 5000, from $100 \times 100/2$.

</details>

**40.** An exponential moving average $m_t = \beta m_{t-1} + (1 - \beta) g_t$ starts at $m_0 = 0$ with $\beta = 0.9$ and sees $g_t = 1$ at every step. Give $m_3$. `[number]`

<details><summary>Answer</summary>

$271/1000 = 0.271$. Step by step: $m_1 = 0.1$, $m_2 = 0.9 \times 0.1 + 0.1 = 0.19$, $m_3 = 0.9 \times 0.19 + 0.1 = 0.271$. In closed form, $m_t = (1 - \beta)(1 + \beta + \cdots + \beta^{t-1}) = 1 - \beta^t$, the geometric sum of problem 36. This is the bias toward 0 that Adam's correction divides out (see [Series and the EMA](../03-calculus-2/02-series-and-the-ema.md)). A common mistake is $0.729 = \beta^3$, which is $1 - m_3$.

</details>

**41.** In problem 40, $m_3 = (1 - \beta)(g_3 + \beta g_2 + \beta^2 g_1)$. Give the weight on the oldest gradient $g_1$. `[number]`

<details><summary>Answer</summary>

$81/1000 = 0.081$. The weight is $(1 - \beta)\beta^2 = 0.1 \times 0.81$. A common mistake is $1/10$, the newest gradient's weight $(1 - \beta)$.

</details>

**42.** $\lim_{n \to \infty} \dfrac{2n + 1}{n + 3}$. `[number]`

<details><summary>Answer</summary>

$2$. Divide numerator and denominator by $n$: $(2 + 1/n)/(1 + 3/n) \to 2/1$. A common mistake is $1/3$, which is the value at $n = 0$, not the limit.

</details>

**43.** Write $0.272727\ldots$ (27 repeating) as a fraction. `[number]`

<details><summary>Answer</summary>

$3/11$. With $x = 0.2727\ldots$, $100x = 27.2727\ldots$, so $99x = 27$ and $x = 27/99 = 3/11$. Equivalently, the geometric series $27/100 + 27/100^2 + \cdots = (27/100)/(1 - 1/100)$. A common mistake is $27/100$, only the first term.

</details>

**44.** ALiBi with 4 heads uses the slopes $2^{-2}, 2^{-4}, 2^{-6}, 2^{-8}$. Give their sum. `[number]`

<details><summary>Answer</summary>

$85/256$. Over the common denominator 256: $64 + 16 + 4 + 1 = 85$. As a geometric sum with $a = 1/4$, $r = 1/4$, $n = 4$: $(1/4)(1 - 4^{-4})/(3/4) = (1 - 1/256)/3 = 85/256$. A common mistake is $1/3$, the infinite sum.

</details>

## Problems on polynomials

Problems 45 to 52 go with [Polynomials, Horner's rule, and stable quadratic roots](04-polynomials-horner-and-stable-roots.md): Horner's rule, synthetic division, Vieta's formulas, and the cancellation case.

**45.** Evaluate $p(x) = 2 - 3x + x^2 + 4x^3$ at $x = 2$ (Horner's rule makes it three multiply-adds). `[number]`

<details><summary>Answer</summary>

$32$. Horner from the highest coefficient down, $4, 1, -3, 2$: $4 \to 4 \cdot 2 + 1 = 9 \to 9 \cdot 2 - 3 = 15 \to 15 \cdot 2 + 2 = 32$. Directly: $2 - 6 + 4 + 32$. A common mistake is 10, from running Horner with the coefficients in the order written ($2, -3, 1, 4$), which evaluates the reversed polynomial.

</details>

**46.** Divide $x^3 - 6x^2 + 11x - 6$ by $x - 1$. Give the quotient. `[expr in x]`

<details><summary>Answer</summary>

$x^2 - 5x + 6$. Synthetic division by the root 1 is Horner at $x = 1$: $1 \to 1 - 6 = -5 \to -5 + 11 = 6 \to 6 - 6 = 0$. The intermediate values are the quotient's coefficients and the last is the remainder, 0, so $x = 1$ is a root. A common mistake is $x^2 - 7x + 18$, which divides by $x + 1$ (Horner at $-1$).

</details>

**47.** Give all roots of $x^3 - 6x^2 + 11x - 6$. `[set]`

<details><summary>Answer</summary>

$\{1, 2, 3\}$. From problem 46, $x^3 - 6x^2 + 11x - 6 = (x - 1)(x^2 - 5x + 6) = (x - 1)(x - 2)(x - 3)$. Vieta checks it: the roots sum to 6 and multiply to 6. A common mistake is $\{-1, -2, -3\}$, reading the roots off the factors' constants without the sign.

</details>

**48.** Give the roots of $2x^2 + 3x - 2$. `[set]`

<details><summary>Answer</summary>

$\{1/2, -2\}$. Factor: $2x^2 + 3x - 2 = (2x - 1)(x + 2)$, or use $x = (-3 \pm \sqrt{9 + 16})/4 = (-3 \pm 5)/4$. A common mistake is $\{-1/2, 2\}$, with both signs flipped.

</details>

**49.** Give the smaller root of $x^2 - 10^6 x + 1$ exactly (radicals allowed) or to 15 significant digits. `[number]`

<details><summary>Answer</summary>

$500000 - \sqrt{249999999999} \approx 1.00000000000100 \times 10^{-6}$. The quadratic formula gives $x = (10^6 \pm \sqrt{10^{12} - 4})/2 = 500000 \pm \sqrt{249999999999}$. Evaluated naively in floating point, the minus sign subtracts two numbers that agree in about 12 digits and loses them. The stable route of [Polynomials, Horner's rule, and stable quadratic roots](04-polynomials-horner-and-stable-roots.md#cancellation-and-the-stable-formula) computes the large root $x_1 = 500000 + \sqrt{249999999999}$ first and uses Vieta, $x_1 x_2 = c/a = 1$, so $x_2 = 1/x_1 = 10^{-6} + 10^{-18} + \cdots$. A common mistake is $10^{-6}$ exactly (the approximation $c/b$), which drops the $10^{-18}$ correction that the 15th significant digit shows.

</details>

**50.** Without solving, give the product of the roots of $3x^2 - 7x + 2$. `[number]`

<details><summary>Answer</summary>

$2/3$. Vieta: for $ax^2 + bx + c$, the product of the roots is $c/a$. (The roots are $2$ and $1/3$.) A common mistake is 2, which is $c$ without dividing by $a$.

</details>

**51.** How many real roots does $x^2 + x + 1$ have? `[number]`

<details><summary>Answer</summary>

$0$. The discriminant is $b^2 - 4ac = 1 - 4 = -3 < 0$, so both roots are complex. Equivalently, $x^2 + x + 1 = (x + 1/2)^2 + 3/4 > 0$. A common mistake is 2, from counting the roots of a quadratic without checking the discriminant.

</details>

**52.** The Taylor polynomial of $e^x$ of degree 3 around 0 is $c_0 + c_1 x + c_2 x^2 + c_3 x^3$. Give $[c_0, c_1, c_2, c_3]$. `[vector]`

<details><summary>Answer</summary>

$[1, 1, 1/2, 1/6]$. $c_k = f^{(k)}(0)/k!$, and every derivative of $e^x$ is $e^x$, equal to 1 at 0, so $c_k = 1/k!$: $1/0!, 1/1!, 1/2!, 1/3!$. A common mistake is $[1, 1, 1/2, 1/3]$, using $1/k$ instead of $1/k!$.

</details>

## Problems on inequalities

The last eight problems return to the inequalities taught at the start of this section, including two that reappear in optimization and information theory.

**53.** Solve $|x - 3| < 2$. `[interval]`

<details><summary>Answer</summary>

$(1, 5)$. "Within distance 2 of 3": $-2 < x - 3 < 2$, so $1 < x < 5$. The inequality is strict, so the endpoints are excluded. A common mistake is $[1, 5]$.

</details>

**54.** Solve $x^2 - 5x + 6 > 0$. `[interval]`

<details><summary>Answer</summary>

$(-\infty, 2) \cup (3, \infty)$. $x^2 - 5x + 6 = (x - 2)(x - 3)$ is zero at 2 and 3; the parabola opens upward, so it is positive outside the roots. A common mistake is $(2, 3)$, which is where it is negative.

</details>

**55.** The smallest integer $n$ with $2^n > 1000$ (the bits needed to number 1000 items). `[number]`

<details><summary>Answer</summary>

$10$. $2^9 = 512 < 1000 < 1024 = 2^{10}$; equivalently $n > \log_2 1000 \approx 9.97$ (problem 14), and the smallest such integer is 10. A common mistake is 9, rounding $\log_2 1000$ down.

</details>

**56.** Solve $\dfrac{x - 1}{x + 2} \ge 0$. `[interval]`

<details><summary>Answer</summary>

$(-\infty, -2) \cup [1, \infty)$. The numerator is zero at 1 and the denominator at $-2$. Signs: for $x < -2$ both are negative, quotient positive; for $-2 < x < 1$ the quotient is negative; for $x > 1$ positive. $x = 1$ gives 0, allowed by $\ge$; $x = -2$ divides by zero. A common mistake is $(-\infty, -2] \cup [1, \infty)$, which includes the point where the expression is undefined.

</details>

**57.** For which probabilities $p \in (0, 1)$ is the surprisal $-\log_2 p$ greater than 3 bits? `[interval]`

<details><summary>Answer</summary>

$(0, 1/8)$. $-\log_2 p > 3 \iff \log_2 p < -3 \iff p < 2^{-3}$: multiplying by $-1$ flips the inequality, and $2^x$ is increasing, so applying it keeps the direction. Rarer events are more surprising. A common mistake is $(1/8, 1)$, which forgets the flip.

</details>

**58.** Solve $e^x \le 2$. `[interval]`

<details><summary>Answer</summary>

$(-\infty, \ln 2]$. $\ln$ is strictly increasing, so applying it keeps the direction: $x \le \ln 2$. Equality is allowed, so the endpoint is included. A common mistake is $(-\infty, \ln 2)$, dropping the endpoint.

</details>

**59.** The smallest value of $x + 1/x$ over $x > 0$. `[number]`

<details><summary>Answer</summary>

$2$, at $x = 1$. Multiply $x + 1/x \ge 2$ by $x > 0$ (direction kept): $x^2 + 1 \ge 2x \iff (x - 1)^2 \ge 0$, which always holds, with equality only at $x = 1$, where $1 + 1 = 2$. A common mistake is 1, the minimizer instead of the minimum.

</details>

**60.** Solve $\ln x < x - 1$ over $x > 0$. `[interval]`

<details><summary>Answer</summary>

$(0, 1) \cup (1, \infty)$. $\ln x \le x - 1$ for every $x > 0$, with equality only at $x = 1$ (the line $x - 1$ is tangent to $\ln x$ there, and $\ln$ curves below its tangents). The strict inequality therefore holds everywhere except $x = 1$. A common mistake is $(0, \infty)$, which includes the point of equality.

</details>

## What can go wrong

The worked answers return again and again to a handful of mistakes, and each of them bites later in the books. Inverting the steps in the original order, as in $f^{-1}(x) = (x - 7)/3$ for $f(x) = 3x - 7$, breaks every inverse in [Categorical sampling](../08-probability-statistics/02-categorical-sampling.md), where inverse-CDF sampling depends on it; problems 1 and 3 address it. Composing in the wrong order, $g(f(x))$ instead of $f(g(x))$, breaks the chain rule in [Jacobians and the chain rule](../05-calculus-3/02-jacobians-and-the-chain-rule.md) and backpropagation in [Tensor, broadcasting backward, and no_grad](../../build-an-llm/01-foundations/02-tensor-and-broadcasting-backward.md); see problems 4 and 9. A closed endpoint where the expression is undefined matters for the domain of $\ln$ in every loss (problems 7 and 56). The false rule $\log(a + b) = \log a + \log b$ wrecks the loss rewrites of [Entropy, cross-entropy, and KL](../12-information-theory/01-entropy-cross-entropy-and-kl.md) (problems 11 and 18). One term too many, or the wrong weight, in a geometric sum throws off Adam's bias correction in [Series and the EMA](../03-calculus-2/02-series-and-the-ema.md) (problems 37, 40, and 41).

> **Warning:** Do not give decimals where an exact value exists. A rounded value hides whether the derivation was right, and problem 49 shows a case where rounding is the whole bug. Every answer gives the exact form first and a decimal only as a check.

## Where this leads

Each part feeds a section of this chapter: the logarithms and units of [Exponents, logarithms, and units of information](01-exponents-logs-and-units-of-information.md); the rotations, Euler's formula, and the relative-angle identity of problem 33 in [Trigonometry, rotations, complex numbers, and Euler's formula](02-trig-rotations-and-eulers-formula.md); the geometric sums, the RoPE ratio of problem 38, the ALiBi slopes of problem 44, and the EMA weights of problems 40 and 41 in [Sequences, geometric series, and frequency ladders](03-sequences-and-frequency-ladders.md); and Horner's rule (problem 45), factoring, Vieta, and the cancellation case of problem 49 in [Polynomials, Horner's rule, and stable quadratic roots](04-polynomials-horner-and-stable-roots.md). Beyond this chapter, limits (problem 42) and monotonicity (problem 8) start calculus in [Derivatives and finite differences](../02-calculus-1/01-derivatives-and-finite-differences.md); functions, inverses, and bijections return in discrete mathematics in [Injective, surjective, bijective](../06-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md); and inverse functions become inverse-CDF sampling in [Categorical sampling](../08-probability-statistics/02-categorical-sampling.md).

### Further reading

- OpenStax, *Precalculus 2e* (free): chapter 1 on functions, chapters 2 and 3 on linear and polynomial functions and inequalities, chapter 6 on exponentials and logarithms, chapters 5 to 8 on trigonometry and complex numbers, and chapter 11 on sequences and series.

## Key terms

- **Function**: a rule assigning exactly one output to each input in its domain.
- **Domain and range**: the inputs where a function is defined, and the outputs it actually takes.
- **Composition**: $(f \circ g)(x) = f(g(x))$, applying $g$ first.
- **One-to-one**: different inputs always give different outputs; exactly the functions with an inverse.
- **Inverse function**: $f^{-1}$ with $f^{-1}(f(x)) = x$ and $f(f^{-1}(y)) = y$.
- **Strictly monotonic**: always increasing or always decreasing, and therefore one-to-one.

## Summary

- A function assigns exactly one output to each input in its domain; its inverse exists exactly when no two inputs share an output, and undoes it: $f^{-1}(f(x)) = x$.
- A strictly monotonic function is one-to-one, so it has an inverse; that is why $\exp$, $\ln$, the sigmoid, and softplus can all be inverted.
- Solving an inequality is solving the matching equation and testing the sign between its solutions; multiplying or dividing by a negative number, or applying a decreasing function, flips the direction.
- The middle four parts are the pen-and-paper side of this chapter's four sections; if a problem stops you, read the section it names.

## Review questions

1. Why does a strictly monotonic function always have an inverse on its range?
   <details><summary>Answer</summary>If $x_1 \ne x_2$, one is smaller, say $x_1 < x_2$, and strict monotonicity gives $f(x_1) < f(x_2)$ or $f(x_1) > f(x_2)$; either way $f(x_1) \ne f(x_2)$. So $f$ is one-to-one, every output in the range comes from exactly one input, and $f^{-1}$ is well defined.</details>
2. When solving an inequality by cutting the line at the zeros and poles of an expression, which cut points can belong to the answer?
   <details><summary>Answer</summary>A zero of the expression belongs to the answer for $\ge$ or $\le$ and not for strict inequalities. A point where the expression is undefined, such as a zero of a denominator, never belongs.</details>
3. Why does applying $\ln$ to both sides of $e^x \le 2$ keep the direction, while multiplying $\log_2 p < -3$ by $-1$ flips it?
   <details><summary>Answer</summary>$\ln$ is strictly increasing, so it preserves order; multiplying by a negative number is a strictly decreasing map, so it reverses order.</details>
