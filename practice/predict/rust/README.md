# Rust: predict the output

Six Rust snippets, each with its exact output and an explanation of every
numbered line. Read the source, write down what you think it prints, then open
the fold. See the [predict introduction](../README.md) for how to use these
pages.

Every snippet is its own binary in one Cargo package, [`predict/`](predict/)
(edition 2024). To reproduce an output, run `cargo run --quiet --bin NN-name`
inside `predict/`. The outputs are the same in debug and release builds: the
snippets that touch overflow say explicitly which overflow behaviour they want.

## 01-shadowing-and-blocks

Source: [`predict/src/bin/01-shadowing-and-blocks.rs`](predict/src/bin/01-shadowing-and-blocks.rs)

What does a second `let` do to the first, and what value does a block have?

<details><summary>Output and explanation</summary>

```text
1. 11
2. 10
3. 9
4. big 40
5. 3
6. ()
7. 600
```

Two rules: `let` always introduces a **new binding**. Reusing a name
*shadows* the earlier binding from that point to the end of the current
scope; it does not mutate anything, and when the scope ends the earlier
binding is visible again. And Rust is expression-oriented: a block `{ ... }`,
an `if`, and a `loop` all have values. A block's value is its final expression
if that expression has **no trailing semicolon**; otherwise it is `()`.

**Line 1: `11`.** `x` is 5, then a new `x` is `5 * 2 = 10`. Inside the inner
block, a third `x` is `10 + 1 = 11`, and that is the one printed.

**Line 2: `10`.** The inner `x` died with the inner block, so the `x` from
`let x = x * 2` is visible again.
*Common wrong prediction:* `11`, from reading `let x = x + 1` as an assignment
to the outer variable. Shadowing in a nested block never leaks outward, which
is what separates it from `x += 1` on a `let mut x`.

**Line 3: `9`.** The block's final expression `a * a` (no semicolon) is its
value.

**Line 4: `big 40`.** `if` is an expression, so `size` is `"big"` because
`9 > 5`. `loop` can return a value through `break value`: `n` reaches 4 and the
loop evaluates to `4 * 10`.

**Line 5: `3`.** The second `spaces` is a new binding of a different type
(`usize`, the byte length of `"   "`). Shadowing can change the type;
assignment to a `let mut` could not.
*Common wrong prediction:* a compile error, from thinking reusing a name with a
new type is a type mismatch.

**Line 6: `()`.** `42;` is a statement: the semicolon discards the value. The
block has no final expression, so it evaluates to the unit value `()`, which
`{:?}` prints as `()`.
*Common wrong prediction:* `42`. One semicolon changes the type of the block,
which is also why a function body ending in `x;` instead of `x` fails to
compile with "expected `i32`, found `()`".

**Line 7: `600`.** Each iteration's `let i = i * 100` shadows the loop variable
for that iteration only, so the sum is `100 + 200 + 300`. The loop counter
itself is unaffected, and the next iteration starts from the next range value.

</details>

## 02-drop-order

Source: [`predict/src/bin/02-drop-order.rs`](predict/src/bin/02-drop-order.rs)

When exactly does each destructor run, and which bindings are not bindings at
all?

<details><summary>Output and explanation</summary>

```text
1. inner scope starts
2. inner scope ends
  drop c
3. vec built
  drop v0
  drop v1
4. before wildcard
  drop wildcard
5. after wildcard
6. after named
7. temporary
  drop temporary
8. main ends
  drop named
  drop b
  drop a
```

The rules: a value is dropped when its **owner** goes out of scope. Local
variables are dropped at the end of their scope in **reverse declaration
order**. A value with no owner (a temporary) is dropped at the end of the
statement that created it. And `_` is a pattern, not a variable: `let _ = expr`
binds nothing, so the value has no owner.

**Lines 1 and 2, then `drop c`.** `_c` lives in the inner block and is dropped
at its closing brace, right after line 2 prints.

**Line 3, then `drop v0`, `drop v1`.** `drop(v)` moves the vector into the
function `std::mem::drop`, whose body is empty: the vector is dropped when that
function returns. A `Vec` drops its elements **front to back**, in index order.
*Common wrong prediction:* `v1` before `v0`, by applying the reverse rule for
local variables to elements. Reverse order applies to variables in a scope
(and struct fields drop in declaration order, also front to back).

**Line 4, then `drop wildcard`, then line 5.** `let _ = Noisy("wildcard")`
does not bind the value, so it is a temporary dropped at the end of that
statement, before line 5.
*Common wrong prediction:* `drop wildcard` at the end of `main`. This is the
expensive one: `let _ = mutex.lock()` releases the lock immediately, so the
"guarded" code below it runs unguarded.

**Line 6, and no drop yet.** `_named` *is* a binding (a name that happens to
start with an underscore, which only silences the unused-variable lint). It
lives until the end of `main`.
*Common wrong prediction:* treating `_named` like `_` and expecting an
immediate drop.

**Line 7: `7. temporary`, then `drop temporary`.** `Noisy("temporary")` is a
temporary inside the `println!` statement. It lives until the end of that full
statement, so the print uses `.0` first and the drop comes after.

**Line 8, then `drop named`, `drop b`, `drop a`.** At the end of `main`, the
remaining locals drop in reverse declaration order: `_named` (declared last),
then `_b`, then `_a`. `v` is not here: it was moved into `drop` and the moved-
from variable has nothing left to drop.
*Common wrong prediction:* `a`, `b`, `named`, declaration order. Reverse order
is what lets a later value safely borrow from an earlier one: the borrower is
always destroyed first.

</details>

## 03-iterator-laziness

Source: [`predict/src/bin/03-iterator-laziness.rs`](predict/src/bin/03-iterator-laziness.rs)

When does a closure passed to `map` actually run, and how many times?

<details><summary>Output and explanation</summary>

```text
1. adapter built, nothing mapped yet
  mapping 1
  mapping 2
2. [2, 4]
  checking 1
  checking 2
  checking 3
3. Some(3)
4. 3
5. (5, Some(5))
6. [2, 4] 2
7. [(1, 'a'), (2, 'b')]
8. 15
```

The rule: iterator adapters (`map`, `filter`, `inspect`, `take`, `zip`) are
**lazy**. Calling one only builds a struct that wraps the previous iterator and
the closure. Work happens when a consumer (`collect`, `find`, `count`, `sum`,
a `for` loop) calls `next`, and each call pulls exactly one element through the
whole chain. A consumer that stops early stops the work early.

**Line 1.** `v.iter().map(...)` built a `Map` value and ran nothing, so no
`mapping` line precedes it. (Iterator adapters are marked `#[must_use]`, so
the compiler warns about a chain that is never consumed, for exactly this
reason.)
*Common wrong prediction:* five `mapping` lines before line 1, from the eager
model of `Array.prototype.map` in JavaScript or `list(map(...))` in Python.

**Line 2 block.** `take(2)` makes `collect` stop after two items, so the
closure runs only for 1 and 2.
*Common wrong prediction:* all five `mapping` lines followed by `[2, 4]`,
believing `map` processes the whole input and `take` trims afterwards.

**Line 3 block.** `find` pulls until its predicate is true. `inspect` sees each
pulled element, so it prints for 1, 2, 3 and stops. `find` returns
`Option<&i32>`, printed as `Some(3)`.

**Line 4: `3`.** The odd numbers 1, 3, 5 pass `filter`; `map` multiplies them
(the products are discarded); `count` counts 3. This is one pass over `v`, with
each element flowing through both adapters before the next is pulled, not one
pass per adapter.

**Line 5: `(5, Some(5))`.** `size_hint` returns a lower and optional upper bound
on the remaining length. `map` cannot change the length, so it forwards the
slice iterator's exact hint, and nothing is pulled: the `never printed` closure
never runs.
*Common wrong prediction:* five `never printed` lines, from thinking that asking
about length requires walking the elements. (A `filter` would answer
`(0, Some(5))`, because it cannot know without running the predicate.)

**Line 6: `[2, 4] 2`.** An iterator is consumed by its consumer and cannot be
reused (it is moved), so to use the results twice you `collect` once. The
items are `&i32` references into `v`, printed as their values.

**Line 7: `[(1, 'a'), (2, 'b')]`.** `zip` ends when either side ends, here after
the two characters of `"ab"`.

**Line 8: `15`.** `sum` consumes the iterator and adds `1 + 2 + 3 + 4 + 5`.

</details>

## 04-integer-overflow

Source: [`predict/src/bin/04-integer-overflow.rs`](predict/src/bin/04-integer-overflow.rs)

What happens when arithmetic or a cast does not fit the type?

<details><summary>Output and explanation</summary>

```text
1. 4
2. None
3. 255
4. (4, true)
5. -3 -1
6. -4 1
7. 44
8. 4294967295
9. 2
10. -1
11. 2147483647 0
12. None 2
13. 2147483647 2147483648
```

The rules: plain `+` on integers **panics on overflow in debug builds and wraps
in release builds** (by default), so overflow-sensitive code should name the
behaviour it wants with an explicit method. Integer `as` casts never fail:
widening sign- or zero-extends, narrowing **keeps the low bits**, and
same-width signed/unsigned casts **reinterpret the bits**. Float-to-int `as`
truncates toward zero and **saturates** at the target's bounds, with NaN
becoming 0.

**Line 1: `4`.** `wrapping_add` computes modulo 2^8: `260 - 256 = 4`.

**Line 2: `None`.** `checked_add` returns `None` instead of an out-of-range
result.

**Line 3: `255`.** `saturating_add` clamps to `u8::MAX`.

**Line 4: `(4, true)`.** `overflowing_add` returns the wrapped result and a flag
saying overflow happened.
*Common wrong prediction for lines 1 to 4:* a panic. That is what `a + 10`
would do in a debug build; these methods exist so the result does not depend
on the build profile.

**Line 5: `-3 -1`.** Integer `/` truncates toward zero: `-7 / 2 = -3.5`, so
`-3`. The remainder satisfies `(a / b) * b + a % b == a`, so
`-7 - (-3 * 2) = -1`: `%` takes the sign of the dividend.
*Common wrong prediction:* `-4 1`, from Python, where `//` floors and `%` takes
the sign of the divisor.

**Line 6: `-4 1`.** Euclidean division picks the quotient that makes the
remainder non-negative: `-7 = 2 * (-4) + 1`. `rem_euclid` is the one to use for
modular arithmetic (array wrap-around, hashing into buckets), because it is
never negative.

**Line 7: `44`.** `300` is `0x12C`; narrowing to `u8` keeps the low 8 bits,
`0x2C = 44`, which is `300 mod 256`.
*Common wrong prediction:* `255` (expecting saturation, which only float casts
do) or a panic (expecting a checked cast; that is `u8::try_from(big)`).

**Line 8: `4294967295`.** `-1i32` is all 32 bits set in two's complement.
Reinterpreting those bits as `u32` gives `2^32 - 1`.

**Line 9: `2`.** Float-to-int truncates toward zero; it does not round.
*Common wrong prediction:* `3`.

**Line 10: `-1`.** `u8::MAX` is `0xFF`; as `i8` those bits are `-1`.

**Line 11: `2147483647 0`.** `1e10` is out of `i32` range, so the cast saturates
to `i32::MAX`; NaN casts to 0. (Before Rust 1.45 these casts were undefined
behaviour; they were defined as saturating to close that hole.)
*Common wrong prediction:* a wrapped value, by analogy with integer narrowing,
or a garbage value, from C, where an out-of-range float-to-int conversion is
undefined.

**Line 12: `None 2`.** Shifting a `u8` by 9 is shifting by at least its bit
width, which `checked_shl` reports as `None` (a plain `1u8 << 9` would panic in
debug). `wrapping_shl` masks the shift amount to the bit width, `9 & 7 = 1`, so
it computes `1 << 1 = 2`. It does not wrap the *result*.
*Common wrong prediction:* `None 0`, believing the 1 is shifted out of the
byte.

**Line 13: `2147483647 2147483648`.** `i64::from(i32::MAX)` widens losslessly
(`From` exists only for conversions that cannot fail), so adding 1 is ordinary
`i64` arithmetic. `i32::MAX + 1` would not compile here, because the compiler
evaluates constant overflow at compile time and rejects it.

</details>

## 05-string-vs-str

Source: [`predict/src/bin/05-string-vs-str.rs`](predict/src/bin/05-string-vs-str.rs)

What do `len`, slicing, and iteration count in a Rust string, and when does
slicing refuse?

<details><summary>Output and explanation</summary>

```text
1. 6 5
2. None
3. Some("h")
4. [(0, 'h'), (1, 'é'), (3, 'l'), (4, 'l'), (5, 'o')]
5. HÉLLO 6
6. ["a", "b", "", "c"]
7. ["a", "b"]
8. true true
9. 2 4
10. olléh
11. [edge] 8
```

The rule: `String` (owned, growable) and `&str` (borrowed view) are both
**guaranteed valid UTF-8**, and both are measured and sliced in **bytes**. A
range slice must start and end on a character boundary; `s[a..b]` panics if it
does not, and `s.get(a..b)` returns `None`. `chars()` iterates Unicode scalar
values. `é` (U+00E9) is two bytes in UTF-8, so `"héllo"` is 6 bytes and 5
chars.

**Line 1: `6 5`.** `len` is bytes; `chars().count()` is code points.
*Common wrong prediction:* `5 5`.

**Line 2: `None`.** Byte 2 is the middle of `é` (bytes 1 and 2), so `0..2`
would produce invalid UTF-8. Rust never creates an invalid `&str`, so `get`
refuses.
*Common wrong prediction:* `Some("hé")` (slicing by characters) or
`Some("h\xc3")`, which is what Go does in its snippet 05. Go strings are just
bytes; Rust's `str` type carries a validity guarantee, and slicing must keep
it.

**Line 3: `Some("h")`.** `0..1` ends on a boundary.

**Line 4.** `char_indices` yields each char with its starting **byte** offset,
so after `é` at 1 the next char is at 3.
*Common wrong prediction:* offsets `0, 1, 2, 3, 4`.

**Line 5: `HÉLLO 6`.** `to_uppercase` maps chars; `É` (U+00C9) is also two
bytes, so the length happens to stay 6. In general case conversion can change
byte length (for example `'ß'` uppercases to `"SS"`), which is why it returns a
new `String`.

**Line 6: `["a", "b", "", "c"]`.** `split` yields the text between every pair of
separators, so two adjacent commas produce an empty string.
*Common wrong prediction:* `["a", "b", "c"]`, from assuming empty pieces are
skipped.

**Line 7: `["a", "b"]`.** `split_whitespace` treats any run of whitespace as one
separator and drops leading and trailing empties.

**Line 8: `true true`.** The standard library implements `PartialEq` between
`&str` and `String` in both directions, so comparing them compares their bytes
with no conversion.

**Line 9: `2 4`.** Length is bytes in use; capacity is bytes allocated.
`with_capacity(4)` reserves room for at least 4 bytes, and pushing 2 bytes does
not reallocate. (The API promises *at least* 4; the current implementation
allocates exactly 4.)
*Common wrong prediction:* `2 2`, conflating capacity with length.

**Line 10: `olléh`.** `chars().rev()` reverses code points. That works here
because `é` is a single precomposed code point. If it were written as `e`
followed by the combining accent U+0301, reversing code points would move the
accent onto the wrong letter. Reversing user-visible characters needs grapheme
segmentation, which is outside `std`.

**Line 11: `[edge] 8`.** `trim` returns a `&str` borrowing the middle of
`padded`: no allocation, no change to the original, whose length is still 8
bytes.

</details>

## 06-move-and-rc

Source: [`predict/src/bin/06-move-and-rc.rs`](predict/src/bin/06-move-and-rc.rs)

When does assignment move, when does it copy, and how does shared ownership
count its owners?

<details><summary>Output and explanation</summary>

```text
1. [1, 2, 3] [1, 2, 3]
2. [1, 2, 3]
3. 5 5
4. 1
5. 2
6. 3
7. 2
8. 1
9. true
10. 2 30
```

The rules: assigning or passing a value **moves** it, transferring ownership
and making the source unusable, unless the type is `Copy` (plain bits such as
integers), in which case it is copied. `clone()` is always an explicit deep
copy. `Rc<T>` provides shared ownership through a reference count stored next
to the value: `Rc::clone` increments it (copying only the pointer), dropping an
`Rc` decrements it, and the value is freed when it reaches zero.

**Line 1.** `a.clone()` allocated a second vector with the same contents. Two
owners, two buffers.

**Line 2.** `let c = b` moved the vector's header (pointer, length, capacity)
into `c`; no elements were copied, and `b` can no longer be used (using it
would be a compile error).
*Common wrong prediction:* that this copies the vector, from languages where
assignment copies values, or that `b` and `c` now share it, from languages
where assignment copies references. In Rust it is neither: one owner,
transferred.

**Line 3: `5 5`.** `i32` is `Copy`, so `m = n` copies and both stay usable.

**Line 4: `1`.** A new `Rc` has one strong owner.

**Line 5: `2`.** `Rc::clone(&r)` adds an owner. The `String` is not cloned.
*Common wrong prediction:* that `Rc::clone` deep-copies like `Vec::clone`.
Writing `Rc::clone(&r)` rather than `r.clone()` is the convention that makes
the cheap pointer copy obvious.

**Line 6: `3`.** `_r3` is a third owner while the inner block runs.

**Line 7: `2`.** `_r3` was dropped at the end of the block, decrementing the
count. Each printed number corresponds to a scope boundary you can see.

**Line 8: `1`.** `drop(r2)` moves `r2` into `drop`, releasing that owner before
the end of `main`.

**Line 9: `true`.** `Rc::ptr_eq` compares the allocations. Every clone points at
the same single `String`; that is why `Rc` gives only shared (immutable)
access: mutating through one handle would be visible through all of them,
which needs `RefCell` or similar to be checked.

**Line 10: `2 30`.** `takes_ref(&owned)` borrows, so `owned` is still usable on
the next line. `takes_ownership(owned)` moves it in; the function consumes it
with `into_iter()` and returns the sum `10 + 20`. After that call, `owned` is
gone.
*Common wrong prediction:* a compile error from calling `takes_ref` first. The
order is what makes it legal: borrowing and then moving is fine, moving and
then borrowing is not.

</details>
