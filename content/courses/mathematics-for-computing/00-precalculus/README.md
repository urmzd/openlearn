# Precalculus

Precalculus is the toolbox the rest of the mathematics in this book is written in: exponents and logarithms, trigonometry and complex numbers, sequences and series, and polynomials. None of it is new if you took algebra at school, but this chapter treats each topic the way a language-model system uses it, and each one has a concrete call site. Logarithms measure information, so a model's loss in nats becomes bits by dividing by $\ln 2$, and bits per byte is the number every tokenizer and model in *Build Your Own LLM System* is judged by. Rotations encode position: rotating a 2D pair by $\theta$ is multiplying $x + iy$ by $e^{i\theta}$, and RoPE applies that to every pair of a query and a key. Geometric ladders set frequencies: RoPE's inverse frequencies $b^{-2i/d}$ and ALiBi's head slopes are geometric sequences, whose sums and ratios decide how far a position signal reaches. And polynomials approximate the functions a C kernel cannot call.

A second theme runs underneath: how you evaluate a formula matters as much as the formula. Horner's rule is the cheapest and most stable way to evaluate a polynomial, and the textbook quadratic formula can lose most of its digits for the small root when $b^2 \gg 4ac$. Several sections show a formula that is correct on paper and wrong in floating point, and the rewrite that fixes it.

The chapter opens with a problem set that practises all of these skills by hand, and then builds each topic into a small, tested Python module. Each build section derives its solution from first principles and walks through the code line by line.

> **Learning objectives**
> - Manipulate exponents and logarithms, convert between nats and bits, and compute bits per byte for a model.
> - Rotate vectors with matrices and with complex numbers, and explain why rotations make attention scores depend only on relative position.
> - Sum geometric series accurately, including near a ratio of 1, and build RoPE's frequency ladder and ALiBi's slopes.
> - Evaluate polynomials with Horner's rule and compute both roots of a quadratic without catastrophic cancellation.
> - Find inverses of functions, use monotonicity, and solve inequalities by hand.

## Before you begin

You need high-school algebra: solving linear equations and manipulating fractions and powers. The build sections use Python and numpy; if you are new to numpy arrays, dtypes, and broadcasting, read [Python and numpy](../../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md) in *Programming Languages* first. Read each section with a notebook open and check every identity numerically before you trust it, with `math.log`, `cmath.exp`, or `numpy`.

## In this chapter

[Problem set: functions, logarithms, trigonometry, series, polynomials, and inequalities](00-solve-set.md) is the pen-and-paper side of the chapter: sixty problems with a worked answer for every one, including the most common wrong answer and why it is wrong. It also teaches functions, inverses, monotonicity, and inequalities, which have no build section because none of that becomes code. Work it first, or alongside the sections it points to.

[Exponents, logarithms, and units of information](01-exponents-logs-and-units-of-information.md) derives the laws of exponents and logarithms and the change-of-base formula, shows that the base of a logarithm is a unit of information, and builds the bits-per-byte conversion that compares models across tokenizers.

[Trigonometry, rotations, complex numbers, and Euler's formula](02-trig-rotations-and-eulers-formula.md) builds rotations from the unit circle, proves that the dot product of two rotated vectors depends only on the difference of their angles, and implements the pairwise rotation that RoPE calls, in matrix and complex form.

[Sequences, geometric series, and frequency ladders](03-sequences-and-frequency-ladders.md) sums geometric series, shows why the closed form loses digits near a ratio of 1 and how `expm1` and `log1p` fix it, and builds RoPE's inverse frequencies and ALiBi's head slopes.

[Polynomials, Horner's rule, and stable quadratic roots](04-polynomials-horner-and-stable-roots.md) evaluates polynomials with Horner's rule, bounds its error, and replaces the textbook quadratic formula with one that never cancels.

## Where this chapter leads

[Calculus 1](../01-calculus-1/README.md) builds limits and derivatives on functions, exponentials, and logarithms. In [Numerical Methods and Floating Point](../09-numerical-methods-and-floating-point/README.md), range reduction and polynomial approximation of `exp` start from Horner's rule. In [Information Theory](../11-information-theory/README.md), entropy and perplexity are logarithms with a unit. And in *[Build Your Own LLM System](../../build-an-llm/README.md)*, RoPE, ALiBi, and bits per byte are the first places this mathematics runs.

## Further reading

- OpenStax, [*Precalculus 2e*](https://openstax.org/details/books/precalculus-2e) (free, CC BY), the primary reference for the chapter.
- OpenStax, [*College Algebra 2e*](https://openstax.org/details/books/college-algebra-2e) (free), for the algebra review.
- Paul's Online Notes, [Algebra and Trig Review](https://tutorial.math.lamar.edu/Extras/AlgebraTrigReview/AlgebraTrigReview.aspx) (free).
- 3Blue1Brown, [Euler's formula with introductory group theory](https://www.youtube.com/watch?v=mvmuCPvRoWQ) (free).
