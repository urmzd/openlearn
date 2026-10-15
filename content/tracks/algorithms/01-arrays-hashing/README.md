# 01-arrays-hashing

## Summary
- Contains: `contains-duplicate.js`, `missing-number.js`, `product-of-array-expect-self.js`, `two-sum.js`.
- JavaScript solutions for classic arrays and hashing interview problems.

## Key takeaways
- Hash maps/sets enable O(n) lookups for pair and frequency problems.
- Watch for off-by-one and missing-element edge cases in array scans.

## How to run
- These are solution functions. Run with Node by adding a small driver, or paste into an online judge.

---

# Arrays & Hashing Patterns

## Core Insight

Most array problems reduce to **"how do I avoid the brute-force O(n²) scan?"** The answer is almost always a hash map or hash set that trades space for time by giving O(1) lookups.

## Pattern 1: Hash Map for Complement Lookup

**When to use**: You need to find pairs/groups that satisfy some relationship (sum, difference, product).

**Template**:
```
seen = {}
for each element:
    complement = target - element
    if complement in seen:
        return answer
    seen[element] = index
```

**Interview problems**: Two Sum, Two Sum II, 4Sum II, Subarray Sum Equals K

**Why it works**: Instead of checking every pair O(n²), you check "have I already seen what I need?" in O(1).

## Pattern 2: Frequency Counting

**When to use**: Problems about duplicates, anagrams, majority elements, or "most/least frequent."

**Template**:
```
freq = {}
for each element:
    freq[element] = freq.get(element, 0) + 1
# Then query the frequency map
```

**Interview problems**: Contains Duplicate, Valid Anagram, Group Anagrams, Top K Frequent Elements, Majority Element

## Pattern 3: Index as Hash Key

**When to use**: Array values are bounded (e.g., values in range [1, n]) and you need O(1) space.

**Template**:
```
for each element:
    index = abs(element) - 1
    nums[index] = -abs(nums[index])  # mark as seen
# Positive values at index i mean i+1 is missing
```

**Interview problems**: Missing Number, Find All Duplicates, First Missing Positive

## Pattern 4: Prefix Sum

**When to use**: Range sum queries, subarray sums, or "number of subarrays with sum = k."

**Template**:
```
prefix = {0: 1}  # sum -> count
running_sum = 0
for each element:
    running_sum += element
    if running_sum - k in prefix:
        count += prefix[running_sum - k]
    prefix[running_sum] = prefix.get(running_sum, 0) + 1
```

**Interview problems**: Subarray Sum Equals K, Contiguous Array, Product of Array Except Self

## Pattern 5: Set for O(1) Membership

**When to use**: "Does X exist?" or "find the longest consecutive sequence."

**Template**:
```
s = set(nums)
for element in s:
    if element - 1 not in s:  # start of a sequence
        count streak from element
```

**Interview problems**: Longest Consecutive Sequence, Intersection of Two Arrays, Contains Duplicate

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Subarray sum with prefix + hash | Medium-Hard |
| Meta | Frequency counting + sorting | Medium |
| Amazon | Two Sum variants (sorted, BST) | Easy-Medium |
| Stripe | Practical data transformation | Medium |
| Jane Street | Mathematical properties of sets | Hard |

## Complexity Cheat Sheet

| Technique | Time | Space | When to Pick |
|-----------|------|-------|-------------|
| Brute force (nested loops) | O(n²) | O(1) | Never in interviews |
| Sort + scan | O(n log n) | O(1) | When order matters |
| Hash map | O(n) | O(n) | Default choice |
| Index marking | O(n) | O(1) | Values bounded [1,n] |
| Prefix sum + hash | O(n) | O(n) | Subarray sum problems |
