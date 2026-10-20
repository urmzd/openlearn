# Math and bit manipulation

Integers are bit strings, and many problems that look like they need a loop over digits or a chain of arithmetic collapse into a handful of constant-time bitwise operations once you see them that way. Bit manipulation replaces arithmetic with $O(1)$ bitwise operations; math problems test number-theory fundamentals such as modular arithmetic, greatest common divisors, and binomial coefficients. Both reward recognizing the pattern rather than brute-forcing.

This chapter first collects the bit-level patterns (single-bit operations, XOR tricks, bit counting, shifts as arithmetic), then the number-theory ones, and closes with a quick reference of tricks and four JavaScript solutions. JavaScript adds a wrinkle you must keep in mind throughout: its bitwise operators work on **32-bit signed** integers, so the sign bit and the difference between `>>` and `>>>` matter.

> **Learning objectives**
> - Set, clear, toggle, and test individual bits, and isolate or clear the lowest set bit.
> - Use XOR's self-cancelling property to find a unique element.
> - Count set bits with Kernighan's algorithm and with a DP over powers of two.
> - Add integers without `+` using XOR and carries, and reason about 32-bit signed semantics.
> - Apply modular arithmetic, Euclid's algorithm, and binomial coefficients.

## Before you begin

You should be comfortable with binary representation and two's complement. [Modular arithmetic, hashing, and PCG32](../../mathematics/12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics* covers the modular arithmetic used here, and [Counting params, FLOPs, and KV bytes](../../mathematics/11-discrete-math-1/01-counting-params-flops-kv-bytes.md) covers counting.

## Single-bit operations

Everything else is built from six one-liners. Shifting `1` left by `i` makes a mask for bit `i`; OR sets it, AND with the complement clears it, XOR toggles it, and AND tests it. Two more isolate and clear the lowest set bit, using the fact that `-x` in two's complement flips every bit above the lowest set bit:

```
Set bit i:      x |  (1 << i)
Clear bit i:    x & ~(1 << i)
Toggle bit i:   x ^  (1 << i)
Check bit i:    x &  (1 << i) != 0
Extract lowest set bit: x & (-x)
Clear lowest set bit:   x & (x - 1)
```

## XOR tricks

XOR's key property is that `a ^ a = 0` and `a ^ 0 = a`, and it is commutative and associative. XOR every element together, and every value that appears twice cancels, leaving the one that appears once:

```python
result = 0
for num in nums:
    result ^= num
# result is the number that appears once
```

Single Number uses this directly; Missing Number can XOR the values against the indices $0, \dots, n$. Single Number II, where the others appear three times, needs per-bit counting modulo 3 instead.

## Counting bits

Kernighan's algorithm counts set bits in $O(k)$, where $k$ is the number of set bits, by repeatedly clearing the lowest one:

```python
count = 0
while n:
    n &= n - 1  # clear lowest set bit
    count += 1
```

Number of 1 Bits, Counting Bits (for every number in a range), and Hamming Distance (the popcount of `a ^ b`) are the standard problems.

## Shifts as arithmetic

Shifts multiply and divide by powers of two, a power of two has exactly one set bit, and addition itself decomposes into XOR (the sum without carries) and AND shifted left (the carries):

```
Multiply by 2^k:  x << k
Divide by 2^k:    x >> k
Check if power of 2: x > 0 and (x & (x-1)) == 0
Sum without +: carry = (a & b) << 1; a = a ^ b; repeat until carry == 0
```

Sum of Two Integers, Power of Two, Reverse Bits, and Divide Two Integers use these identities.

## Modular arithmetic

Turning to math, modular arithmetic lets you keep intermediate results small, because sums and products can be reduced at every step:

```
(a + b) mod m = ((a mod m) + (b mod m)) mod m
(a * b) mod m = ((a mod m) * (b mod m)) mod m
Modular exponentiation: pow(base, exp, mod) — built into Python
Modular inverse (when m is prime): pow(a, m-2, m)
```

The inverse follows from Fermat's little theorem: for prime $m$, $a^{m-1} \equiv 1 \pmod m$, so $a^{m-2}$ is $a^{-1}$. Pow(x, n), Count Primes, and Super Pow are the standard problems.

## GCD and number theory

Euclid's algorithm computes the greatest common divisor by repeatedly replacing the pair with the smaller number and the remainder, and the least common multiple follows from it:

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

lcm = a * b // gcd(a, b)
```

Bézout's identity adds that $\gcd(a, b) = ax + by$ for some integers $x$ and $y$, which the extended Euclidean algorithm finds; that is how modular inverses are computed when the modulus is not prime.

## Combinatorics

Binomial coefficients appear constantly. Python has them built in, and Pascal's rule gives a DP:

```python
# n choose k
from math import comb  # Python 3.8+
# or: dp[n][k] = dp[n-1][k-1] + dp[n-1][k]   (Pascal's triangle)
```

Unique Paths, for instance, has the closed form $\binom{m+n-2}{m-1}$: choose which of the $m + n - 2$ moves go down. Pascal's Triangle and the Catalan numbers are the other classic problems.

## Quick reference

These tricks replace common brute-force idioms:

| Operation | Brute force | Bit or math trick |
|-----------|-------------|---------------|
| is even? | `n % 2 == 0` | `n & 1 == 0` |
| multiply by 2 | `n * 2` | `n << 1` |
| swap `a`, `b` | a temporary | `a ^= b; b ^= a; a ^= b` |
| absolute value | if/else | `(n ^ (n >> 31)) - (n >> 31)` |
| average without overflow | `(a + b) / 2` | `(a & b) + ((a ^ b) >> 1)` |

Interviewers favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | bit manipulation with math | Medium-Hard |
| NVIDIA | bitwise operations (GPU relevance) | Medium |
| Jane Street | number theory, probability | Hard |
| Citadel | fast arithmetic, modular math | Hard |
| HRT | bit tricks for low latency | Hard |

## Key terms

- **Bit mask**: an integer whose set bits select positions, such as `1 << i` for bit `i`; OR with it sets the bit, AND with its complement clears it, XOR toggles it, and AND tests it.
- **Lowest set bit**: the least significant 1 bit of an integer; `x & -x` isolates it and `x & (x - 1)` clears it, both because negation in two's complement flips every bit above it.
- **XOR trick**: using `a ^ a = 0`, `a ^ 0 = a`, and the commutativity and associativity of XOR to cancel values that appear twice, so XOR-ing a whole collection leaves the one value that appears once.
- **Kernighan's algorithm**: counting the set bits (the popcount) of an integer by repeatedly clearing the lowest one with `n &= n - 1`, in time proportional to the number of set bits.
- **Unsigned right shift**: JavaScript's `>>>`, which shifts in zeros and reads the result as an unsigned 32-bit integer, unlike `>>`, which copies the sign bit into the vacated positions.
- **Pascal's rule**: the identity $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$, which builds Pascal's triangle and gives a DP for binomial coefficients without large factorials.

## In this chapter

Each solution is a JavaScript function; run it with Node by adding a small driver, or paste it into an online judge.

- [Counting Bits](count-number-of-bits.js): returns the popcount of every number from $0$ to $n$ in $O(n)$ with a DP over powers of two. Each power of two has one set bit, and any other $i$ has one more set bit than $i$ minus the largest power of two not exceeding it.
- [Number of 1 Bits](number-of-1-bits.js): tests the low bit 32 times, shifting with the unsigned `>>>` so that a set sign bit does not fill the number with ones.
- [Reverse Bits](reverse-bits.js): moves the low bit of the input onto the low end of the output 32 times, then applies `>>> 0` so the result is read as an unsigned 32-bit value rather than a negative number.
- [Sum of Two Integers](sum-of-two-integers.js): adds without `+` by repeating "sum is `a ^ b`, carry is `(a & b) << 1`" until the carry is zero. Because JavaScript's bitwise operators work on 32-bit signed integers, negative numbers come out right too: the carry eventually shifts out of the top bit.
