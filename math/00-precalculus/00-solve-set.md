<!-- ol:module S-M00 -->
# Solve set: functions, exp/log, trig and Euler, series, polynomials, inequalities

## Overview

| | |
|---|---|
| **Module** | `S-M00` · solve · pen and paper, checked by SymPy · Pass 2 · 4 to 6 h |
| **You write** | `solve/S-M00.toml` in your repo: 60 answers in ASCII math (`ol start S-M00` writes the template) |
| **Problems** | `course/solve/S-M00/problems.md`, reproduced in section 4 |
| **Checked by** | `ol check S-M00`: each answer is compared with a hidden key by SymPy (symbolically, then numerically at seeded points); a part passes at 80% |
| **Needs** | nothing. Each part points to the chapter section that teaches it |
| **Used by** | the four build modules of this topic (`M00.1` to `M00.4`) as their pen-and-paper check; parts 1 and 6 are `M00.5` (functions, inverses, monotonicity, inequalities), which `M01.1` and `M05.2` assume |
| **Milestone** | `MS-P2` (the Pass 2 gate: every math module and solve part of the pass) |
| **Optional depth** | OpenStax, *Precalculus 2e* (free), ch. 1 (functions), 2 to 3 (linear and polynomial functions, inequalities), 6 (exponentials and logs), 5 to 8 (trigonometry, complex numbers), 11 (sequences and series) |

## Key Takeaways

- A **function** assigns exactly one output to each input in its **domain**; its **inverse** exists exactly when no two inputs share an output, and undoes it: $f^{-1}(f(x)) = x$.
- A strictly **monotonic** function (always increasing, or always decreasing) is one-to-one, so it has an inverse; that is why $\exp$, $\ln$, the sigmoid, and softplus can all be inverted.
- Solving an **inequality** is solving the matching equation and testing the sign between its solutions; multiplying or dividing by a negative number, or applying a decreasing function, flips the direction.
- The other four parts are the pen-and-paper side of `M00.1` to `M00.4`: if a problem in part 2 to 5 stops you, read the beat 2 section of that module's chapter it names.

## How to work this chapter

```bash
ol start S-M00              # writes solve/S-M00.toml with an empty answer per question
$EDITOR solve/S-M00.toml    # answer = "(x + 7)/3", one line per question
ol check S-M00              # per-question pass/fail; feedback never shows the expected answer
```

Answers are ASCII math: `x^2`, `sqrt(x)`, `log(x)` (natural log), `E`, `pi`, `I`, `oo`; intervals like `(-oo, 2) U [3, 5)`; sets `{1, 2}`; vectors `[1, 2]`. A wrong answer reports where it differs (`q4 FAIL: differs at x=1.30`), not what it should be.

---

## 1. Why now

Pass 2 turns the math under your tracer system into code: logarithms that measure information (`M00.1`), rotations that encode position (`M00.2`), ladders of frequencies (`M00.3`), and polynomial evaluation that kernels run (`M00.4`), and after them calculus, linear algebra, probability, and optimization. Each of those chapters assumes you can manipulate the expressions fluently by hand: invert a function, combine logarithms, use an angle-addition formula, sum a geometric series, factor a polynomial, and solve an inequality. A bug in a derivation becomes a bug in code that tests catch late and explain poorly. This set checks the hand skills first, with a checker that accepts any correct form of an answer. Parts 1 and 6 also teach `M00.5`, functions, inverses, monotonicity, and inequalities, which has no build module of its own because none of it becomes code; `M01.1` (limits and derivatives) and `M05.2` build on it directly.

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
5. *Answer*, as you would write it in `solve/S-M00.toml` (in the variable of the question): `answer = "log(y/(2 - y))"`. The checker also accepts `log(y) - log(2 - y)` or `-log(2/y - 1)`, because it tests equality, not spelling: symbolically first, then at 32 seeded points of the domain the key declares.

**And one inequality, a sibling of q56: $\dfrac{x + 1}{x - 4} \le 0$.** The numerator is zero at $-1$ and the denominator at $4$. Signs: for $x < -1$ both factors are negative and the quotient positive; for $-1 < x < 4$ the numerator is positive and the denominator negative, so the quotient is negative; for $x > 4$ it is positive. Zero at $x = -1$ is allowed by $\le$; $x = 4$ divides by zero. Answer: `[-1, 4)`.

## 4. Problem set

Write each answer in `solve/S-M00.toml` as `answer = "..."` under its `[qN]` header. The tag after each question is the answer type.

<!-- ol:problems S-M00 -->

### Functions and inverses

**q1.** $f(x) = 3x - 7$. Give $f^{-1}(x)$. `[expr in x]`

**q2.** $f(x) = e^{2x + 1}$. Give $f^{-1}(x)$ for $x > 0$. `[expr in x]`

**q3.** $g(x) = \dfrac{2x + 1}{x - 3}$ for $x \ne 3$. Give $g^{-1}(x)$. `[expr in x]`

**q4.** $f(x) = x^2$ and $g(x) = x + 1$. Give $(f \circ g)(x) = f(g(x))$. `[expr in x]`

**q5.** The sigmoid is $\sigma(z) = 1 / (1 + e^{-z})$. Give its inverse, the logit $\sigma^{-1}(p)$, for $0 < p < 1$. `[expr in p]`

**q6.** Give the range of $\sigma$ (the set of values it takes over all real $z$). `[interval]`

**q7.** Give the domain of $h(x) = \ln(4 - x^2)$ (the real $x$ where it is defined). `[interval]`

**q8.** Is $f(x) = x^3 + x$ one-to-one on all of $\mathbb{R}$? (Hint: is it monotonic?) `[bool]`

**q9.** $f(x) = 2^x$ and $g(x) = \log_2 x$. Compute $f(g(16)) + g(f(3))$. `[number]`

**q10.** Softplus is $s(x) = \ln(1 + e^x)$. Give $s^{-1}(y)$ for $y > 0$. `[expr in y]`

### Exponents, logarithms, and units of information

**q11.** $\log_2 8 + \log_2 4$. `[number]`

**q12.** $\log_8 32$. `[number]`

**q13.** Simplify $e^{3 \ln x}$ for $x > 0$. `[expr in x]`

**q14.** Solve $2^x = 1000$ for $x$. `[number]`, decimal ok

**q15.** How many bits is one nat? `[number]`, decimal ok

**q16.** A model's mean negative log-likelihood is $2.0$ nats per token on a text of $1000$ tokens and $4000$ bytes. Give its bits per byte. `[number]`, decimal ok

**q17.** A model's perplexity is $e^{\text{mean NLL in nats}} = 8$ per token. Give its mean NLL in bits per token. `[number]`

**q18.** Write $a = \ln 2$ and $b = \ln 3$. Express $\ln 72$ in $a$ and $b$. `[expr in a, b]`

**q19.** Solve $\log_3 x + \log_3 (x - 8) = 2$. `[number]`

**q20.** Solve $4^x = 8$. `[number]`

**q21.** A quantity decays as $N(t) = N_0 e^{-kt}$ with $k > 0$. After what time is it half of $N_0$? `[expr in k]`

**q22.** How many decimal digits does $2^{100}$ have? (Use $\log_{10} 2 \approx 0.30103$.) `[number]`

### Trigonometry, complex numbers, and Euler's formula

**q23.** $\sin(\pi/6)$. `[number]`

**q24.** $\cos(2\pi/3)$. `[number]`

**q25.** Write $\cos a \cos b - \sin a \sin b$ as one trigonometric function of $a$ and $b$. `[expr in a, b]`

**q26.** Rotate the point $(1, 0)$ counterclockwise by $\pi/3$. Give the image. `[vector]`

**q27.** $R(\pi/2)\,R(\pi/3) = R(\theta)$ with $\theta \in [0, 2\pi)$, where $R(\theta)$ is the rotation by $\theta$. Give $\theta$. `[number]`

**q28.** $(1 + i)^8$. `[number]`

**q29.** $|3 + 4i|$. `[number]`

**q30.** Rotate $3 + 4i$ by the angle $t$ with $\cos t = 3/5$ and $\sin t = 4/5$: compute $(3 + 4i)(\cos t + i \sin t)$. `[number]`

**q31.** $e^{i\pi/2}$. `[number]`

**q32.** Express $\cos t$ using only exponentials $e^{it}$ and $e^{-it}$. `[expr in t]`

**q33.** For the unit complex numbers $u = e^{ia}$ and $v = e^{ib}$, simplify $\operatorname{Re}(u\,\bar{v})$, where $\bar{v}$ is the complex conjugate. `[expr in a, b]`

**q34.** A RoPE dimension pair rotates by $\omega p$ radians at position $p$, with $\omega = 0.01$. After how many positions does the pattern repeat (the period)? `[number]`

### Sequences and geometric series

**q35.** $1 + \tfrac12 + \tfrac14 + \tfrac18 + \cdots$ (forever). `[number]`

**q36.** Give $\sum_{j=0}^{n-1} r^j$ in closed form for $r \ne 1$. `[expr in r, n]`

**q37.** The sum of the first 10 terms of $3, 6, 12, 24, \ldots$ `[number]`

**q38.** The RoPE ladder of a 128-dimensional head with base $10000$ is $\omega_i = 10000^{-2i/128}$ for $i = 0, \ldots, 63$. Give the ratio $\omega_{i+1}/\omega_i$. `[number]`

**q39.** $1 + 2 + \cdots + 100$. `[number]`

**q40.** An exponential moving average $m_t = \beta m_{t-1} + (1 - \beta) g_t$ starts at $m_0 = 0$ with $\beta = 0.9$ and sees $g_t = 1$ at every step. Give $m_3$. `[number]`

**q41.** In q40, $m_3 = (1 - \beta)(g_3 + \beta g_2 + \beta^2 g_1)$. Give the weight on the oldest gradient $g_1$. `[number]`

**q42.** $\lim_{n \to \infty} \dfrac{2n + 1}{n + 3}$. `[number]`

**q43.** Write $0.272727\ldots$ (27 repeating) as a fraction. `[number]`

**q44.** ALiBi with 4 heads uses the slopes $2^{-2}, 2^{-4}, 2^{-6}, 2^{-8}$. Give their sum. `[number]`

### Polynomials

**q45.** Evaluate $p(x) = 2 - 3x + x^2 + 4x^3$ at $x = 2$ (Horner's rule makes it three multiply-adds). `[number]`

**q46.** Divide $x^3 - 6x^2 + 11x - 6$ by $x - 1$. Give the quotient. `[expr in x]`

**q47.** Give all roots of $x^3 - 6x^2 + 11x - 6$. `[set]`

**q48.** Give the roots of $2x^2 + 3x - 2$. `[set]`

**q49.** Give the smaller root of $x^2 - 10^6 x + 1$ exactly (radicals allowed) or to 15 significant digits. `[number]`

**q50.** Without solving, give the product of the roots of $3x^2 - 7x + 2$. `[number]`

**q51.** How many real roots does $x^2 + x + 1$ have? `[number]`

**q52.** The Taylor polynomial of $e^x$ of degree 3 around 0 is $c_0 + c_1 x + c_2 x^2 + c_3 x^3$. Give $[c_0, c_1, c_2, c_3]$. `[vector]`

### Inequalities

**q53.** Solve $|x - 3| < 2$. `[interval]`

**q54.** Solve $x^2 - 5x + 6 > 0$. `[interval]`

**q55.** The smallest integer $n$ with $2^n > 1000$ (the bits needed to number 1000 items). `[number]`

**q56.** Solve $\dfrac{x - 1}{x + 2} \ge 0$. `[interval]`

**q57.** For which probabilities $p \in (0, 1)$ is the surprisal $-\log_2 p$ greater than 3 bits? `[interval]`

**q58.** Solve $e^x \le 2$. `[interval]`

**q59.** The smallest value of $x + 1/x$ over $x > 0$. `[number]`

**q60.** Solve $\ln x < x - 1$ over $x > 0$. `[interval]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Where it bites later | Caught by |
|---|---|---|
| inverting the steps in the original order ($f^{-1}(x) = (x - 7)/3$ for $f(x) = 3x - 7$) | every inverse in `M07.1` (inverse-CDF sampling) | q1, q3 |
| composing in the wrong order: $g(f(x))$ instead of $f(g(x))$ | the chain rule in `M01.2` and backpropagation in `L0.1` | q4, q9 |
| a closed endpoint where the expression is undefined | the domain of $\ln$ in every loss | q7, q56 |
| $\log(a + b) = \log a + \log b$ | rewriting losses in `M11.1` | q11, q18 |
| one term too many, or the wrong weight, in a geometric sum | Adam's bias correction (`M02.2`) | q37, q40, q41 |
| decimals where an exact value is asked | none: an exact value is what the checker can prove equal | every exact number answer, with the message "give an exact value (no decimals)" |

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
