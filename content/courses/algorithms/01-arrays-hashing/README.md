# Arrays and hashing

Most array problems reduce to a single question: **how do I avoid the brute-force $O(n^2)$ scan?** The obvious solution to "find two numbers that sum to a target" or "does this array contain a duplicate?" compares every element with every other. The answer is almost always a hash map or hash set, which trades memory for time by giving $O(1)$ expected lookups. Instead of asking "which earlier element pairs with this one?" by scanning, you ask "have I already seen what I need?" in constant time.

This chapter opens the book because nearly every efficient solution that follows uses the same move somewhere. It presents five patterns built on it (complement lookup, frequency counting, using indices as hash keys, prefix sums, and set membership), gives a template for each, and then walks through four classic problems whose JavaScript solutions sit next to this text.

> **Learning objectives**
> - Recognize when a pair, group, or frequency problem can be solved in $O(n)$ with a hash map or set.
> - Apply the complement-lookup, frequency-counting, and prefix-sum templates.
> - Use array indices as an implicit hash when values are bounded, to reach $O(1)$ extra space.
> - Choose between brute force, sorting, hashing, and index marking from their time and space costs.

## Before you begin

You need to be comfortable with loops and arrays in JavaScript or Python. It helps to know what a hash table does and why its lookups are constant time on average; [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing* covers the math, and [the Swiss table](../16-systems-data-structures/02-swiss-table.md) in this book's systems chapter builds a production hash map.

## Complement lookup with a hash map

Use this pattern when you need to find pairs or groups that satisfy a relationship such as a sum, difference, or product. Walk the array once; for each element, compute the partner it would need, check whether that partner is already in the map, and only then record the current element:

```
seen = {}
for each element:
    complement = target - element
    if complement in seen:
        return answer
    seen[element] = index
```

Instead of checking every pair in $O(n^2)$, you check "have I already seen what I need?" in $O(1)$. Recording the element after the check matters: it stops an element from pairing with itself. The pattern solves Two Sum, Two Sum II, 4Sum II, and Subarray Sum Equals K.

## Frequency counting

Problems about duplicates, anagrams, majority elements, or "most or least frequent" all start by counting. Build a map from value to count in one pass, then answer the question from the map:

```
freq = {}
for each element:
    freq[element] = freq.get(element, 0) + 1
# Then query the frequency map
```

Contains Duplicate, Valid Anagram, Group Anagrams, Top K Frequent Elements, and Majority Element are all instances.

## Indices as hash keys

When the values are bounded, for example all in the range $[1, n]$, the array itself can serve as the hash table, and you get $O(1)$ extra space. Mark value $v$ as seen by negating the entry at index $v - 1$; afterwards, a positive entry at index $i$ means $i + 1$ never appeared:

```
for each element:
    index = abs(element) - 1
    nums[index] = -abs(nums[index])  # mark as seen
# Positive values at index i mean i+1 is missing
```

Missing Number, Find All Duplicates, and First Missing Positive use this idea, as do related tricks that let the indices $0, \dots, n-1$ stand in for the values you expect to see.

## Prefix sums

Range sums, subarray sums, and "number of subarrays with sum $k$" yield to prefix sums. A subarray from $i$ to $j$ sums to $k$ exactly when the running sum at $j$ minus the running sum before $i$ is $k$, so you keep a map from each running sum to how often it has occurred and look up `running_sum - k`:

```
prefix = {0: 1}  # sum -> count
running_sum = 0
for each element:
    running_sum += element
    if running_sum - k in prefix:
        count += prefix[running_sum - k]
    prefix[running_sum] = prefix.get(running_sum, 0) + 1
```

The entry `{0: 1}` accounts for subarrays that start at the beginning. The same running-aggregate idea, with products instead of sums, solves Product of Array Except Self; Subarray Sum Equals K and Contiguous Array are the other classic cases.

## Sets for constant-time membership

Finally, when the question is "does $x$ exist?", a set is all you need. Longest Consecutive Sequence shows how far that goes: put every number in a set, and start counting a streak only from an element whose predecessor is absent, so each streak is walked once and the whole algorithm is $O(n)$:

```
s = set(nums)
for element in s:
    if element - 1 not in s:  # start of a sequence
        count streak from element
```

Intersection of Two Arrays and Contains Duplicate are simpler uses of the same structure.

## Choosing a technique

The patterns differ mainly in what they cost. This table summarizes the options:

| Technique | Time | Space | When to pick |
|-----------|------|-------|-------------|
| Brute force (nested loops) | $O(n^2)$ | $O(1)$ | never in interviews |
| Sort + scan | $O(n \log n)$ | $O(1)$ | when order matters |
| Hash map | $O(n)$ | $O(n)$ | the default choice |
| Index marking | $O(n)$ | $O(1)$ | values bounded in $[1, n]$ |
| Prefix sum + hash | $O(n)$ | $O(n)$ | subarray sum problems |

Two kinds of mistakes account for most wrong answers in this family: off-by-one errors in array scans, and missing-element edge cases (an empty array, the missing value being $0$ or $n$).

Interviewers lean on different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | subarray sum with prefix + hash | Medium-Hard |
| Meta | frequency counting + sorting | Medium |
| Amazon | Two Sum variants (sorted, BST) | Easy-Medium |
| Stripe | practical data transformation | Medium |
| Jane Street | mathematical properties of sets | Hard |

## In this chapter

Each solution is a JavaScript function. To run one, add a small driver that calls it and prints the result and run it with Node, or paste the function into an online judge.

- [Two Sum](two-sum.js): complement lookup in its purest form. A map from value to index is checked for `target - nums[i]` before `nums[i]` is stored, giving the pair's indices in one pass, or an empty array if there is none.
- [Contains Duplicate](contains-duplicate.js): a one-line answer that builds a `Set` from the array and compares its size with the array's length.
- [Missing Number](missing-number.js): finds the one value missing from $0, \dots, n$ in $O(1)$ space by letting the indices stand for the expected values. Starting from $n$ and adding `i - nums[i]` at each index leaves exactly the missing number.
- [Product of Array Except Self](product-of-array-expect-self.js): a prefix-product pass fills each slot with the product of everything to its left, and a backward pass multiplies in a running product `R` of everything to its right, all without division. For `[1, 2, 3, 4]` the result is `[24, 12, 8, 6]`.
