# 10-math-bit

## Summary
- Contains: `count-number-of-bits.js`, `number-of-1-bits.js`, `reverse-bits.js`, `sum-of-two-integers.js`.
- Bit-manipulation practice in JavaScript.

## Key takeaways
- Bitwise operations can replace loops for constant-time transformations.
- Pay attention to JS 32-bit bitwise semantics and sign bits.

## How to run
- These are solution functions. Run with Node by adding a small driver, or paste into an online judge.

---

# Math & Bit Manipulation Patterns

## Core Insight

Bit manipulation replaces arithmetic with O(1) bitwise operations. Math problems test number theory fundamentals. Both are about recognizing the pattern, not brute-forcing.

## Bit Manipulation Patterns

### Pattern 1: Single-Bit Operations

```
Set bit i:      x |  (1 << i)
Clear bit i:    x & ~(1 << i)
Toggle bit i:   x ^  (1 << i)
Check bit i:    x &  (1 << i) != 0
Extract lowest set bit: x & (-x)
Clear lowest set bit:   x & (x - 1)
```

### Pattern 2: XOR Tricks

**Key property**: `a ^ a = 0` and `a ^ 0 = a`

**Find the single number**:
```python
result = 0
for num in nums:
    result ^= num
# result is the number that appears once
```

**Interview problems**: Single Number, Single Number II (use bit counting), Missing Number

### Pattern 3: Counting Bits

**Kernighan's algorithm** — count set bits in O(k) where k = number of set bits:
```python
count = 0
while n:
    n &= n - 1  # clear lowest set bit
    count += 1
```

**Interview problems**: Number of 1 Bits, Counting Bits (for range), Hamming Distance

### Pattern 4: Bit Shifting for Arithmetic

```
Multiply by 2^k:  x << k
Divide by 2^k:    x >> k
Check if power of 2: x > 0 and (x & (x-1)) == 0
Sum without +: carry = (a & b) << 1; a = a ^ b; repeat until carry == 0
```

**Interview problems**: Sum of Two Integers, Power of Two, Reverse Bits, Divide Two Integers

## Math Patterns

### Pattern 5: Modular Arithmetic

```
(a + b) mod m = ((a mod m) + (b mod m)) mod m
(a * b) mod m = ((a mod m) * (b mod m)) mod m
Modular exponentiation: pow(base, exp, mod) — built into Python
Modular inverse (when m is prime): pow(a, m-2, m)
```

**Interview problems**: Pow(x, n), Count Primes, Super Pow

### Pattern 6: GCD and Number Theory

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

lcm = a * b // gcd(a, b)
```

**Bezout's identity**: `gcd(a, b) = a*x + b*y` for some integers x, y (Extended Euclidean).

### Pattern 7: Combinatorics

```python
# n choose k
from math import comb  # Python 3.8+
# or: dp[n][k] = dp[n-1][k-1] + dp[n-1][k]   (Pascal's triangle)
```

**Interview problems**: Unique Paths (= C(m+n-2, m-1)), Pascal's Triangle, Catalan Numbers

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Bit manipulation + math | Medium-Hard |
| NVIDIA | Bitwise ops (GPU relevance) | Medium |
| Jane Street | Number theory, probability | Hard |
| Citadel | Fast arithmetic, modular math | Hard |
| HRT | Bit tricks for low-latency | Hard |

## Quick Reference

| Operation | Brute Force | Bit/Math Trick |
|-----------|-------------|---------------|
| Is even? | n % 2 == 0 | n & 1 == 0 |
| Multiply by 2 | n * 2 | n << 1 |
| Swap a, b | temp | a ^= b; b ^= a; a ^= b |
| Abs value | if/else | (n ^ (n >> 31)) - (n >> 31) |
| Average without overflow | (a + b) / 2 | (a & b) + ((a ^ b) >> 1) |
