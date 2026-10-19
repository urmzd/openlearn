# Predicting TypeScript output

TypeScript's types describe a program; they do not run it. What runs is
JavaScript, with JavaScript's number type, its coercing `==`, its
call-site-dependent `this`, and its event loop. Most TypeScript surprises come
from forgetting that: trusting a type assertion the runtime never checks,
assuming `sort` compares numbers, or expecting a `setTimeout(fn, 0)` to run
before a promise callback. The six snippets in this section each isolate one
of those runtime rules.

Use them as the [predict introduction](../README.md) describes: read the
source, commit to the exact output line by line, then open the fold. Each
explanation gives the rule in terms of what the JavaScript engine actually
does, applies it to the snippet, and names the common wrong prediction and the
model that produces it. By the end you should be able to look at TypeScript
and see the JavaScript underneath, which is where the behaviour lives.

> **Learning objectives**
> - Explain the limits of IEEE 754 doubles, including the 53-bit integer range.
> - Predict default `sort` order and how iteration methods treat array holes.
> - Apply the Abstract Equality algorithm and the relational comparison rules to mixed-type operands.
> - Determine `this` from the call site, `bind`, or an arrow function's defining scope.
> - Explain type erasure, and order synchronous code, microtasks, and timers on the event loop.

## Before you begin

The outputs were produced with Node 22, which runs `.ts` files by **stripping
the types** without checking them (Node 22.18 and later do this by default;
earlier 22.x releases need `--experimental-strip-types`). That is the point of
several snippets: what runs is JavaScript, and the types have no say in it.
[`package.json`](package.json) sets `"type": "module"`, so every file is an ES
module and runs in strict mode. To reproduce an output, run
`node predict/NN-name.ts` from this directory.

Two printing details help when you write your predictions down:
`console.log` prints its arguments separated by single spaces, and prints
strings without quotes.

## Number precision

Source: [`predict/01-number-precision.ts`](predict/01-number-precision.ts)

What can a JavaScript `number` represent exactly, and what do its operators do
at the edges?

<details><summary>Output and explanation</summary>

```text
1. 0.30000000000000004
2. false
3. 0.30
4. 9007199254740991
5. true
6. 9007199254740992
7. true false -Infinity
8. Infinity NaN true
9. -2147483648 0 5 -5
10. 8 8 0 0
11. -0 1 3
12. 9007199254740993n bigint
```

The rule: every `number` is an **IEEE 754 double**: 52 stored fraction bits
plus an implicit leading 1, so 53 bits of integer precision. Integers up to
2^53 are exact; above that, the gap between neighbouring doubles is 2 or more.
TypeScript's `number` is the same type with a compile-time label.

**Line 1: `0.30000000000000004`.** 0.1 and 0.2 have no finite binary expansion,
so each is stored as the nearest double. Their sum rounds to a double just
above the one nearest 0.3, and JavaScript prints the shortest decimal that
round-trips to that exact double.
*Common wrong prediction:* `0.3`, from assuming decimal literals are stored as
decimals.

**Line 2: `false`.** The two sides are different doubles.

**Line 3: `0.30`.** `toFixed(2)` rounds for *display* and returns a string
(printed without quotes). It is formatting, not a fix for the comparison.

**Line 4: `9007199254740991`.** `MAX_SAFE_INTEGER` is 2^53 - 1, the largest `n`
for which `n` and `n + 1` are both exactly representable.

**Line 5: `true`.** `MAX + 1` is 2^53, exact. `MAX + 2` would be 2^53 + 1, which
needs 54 bits; it rounds to the nearest even neighbour, 2^53. Both sides are
the same double.
*Common wrong prediction:* `false`, believing integer arithmetic is exact for
any integer.

**Line 6: `9007199254740992`.** The literal 2^53 + 1 is rounded at parse time
to 2^53. The program never had the value it typed.
*Common wrong prediction:* `9007199254740993`. This is how 64-bit IDs from a
JSON API get silently corrupted in a browser.

**Line 7: `true false -Infinity`.** IEEE 754 has a signed zero. `===` treats
`-0` and `0` as equal, but `Object.is` distinguishes them, and so does division:
`1 / -0` is `-Infinity`.

**Line 8: `Infinity NaN true`.** Division by zero does not throw in floating
point; it produces infinities, and `0 / 0` produces NaN.
*Common wrong prediction:* a thrown error, from languages where integer
division by zero traps.

**Line 9: `-2147483648 0 5 -5`.** Bitwise operators first apply `ToInt32`:
truncate toward zero, take the value modulo 2^32, and reinterpret as a signed
32-bit integer. 2^31 wraps to -2^31; 2^32 wraps to 0; 5.9 and -5.9 truncate to
5 and -5.
*Common wrong prediction:* `2147483648 4294967296 5 -6`. `x | 0` is a
truncation, not a floor, and it silently wraps anything above 2^31 - 1.

**Line 10: `8 8 0 0`.** `parseInt` reads the longest valid integer prefix and
stops at the first invalid character: `"08"` is 8 (modern engines do not treat
a leading 0 as octal), and `"0.9"` stops at the `.`, giving 0. `Number` converts
the whole string: `"08"` is 8, and the empty string converts to **0**, not NaN.
*Common wrong prediction:* `Number("")` is NaN. It is 0, which is why
`Number(input)` accepts empty form fields as zero.

**Line 11: `-0 1 3`.** `Math.round` rounds halves **toward +Infinity**. -0.5
rounds up to -0 (the sign of zero is kept), 0.5 to 1, 2.5 to 3.
`console.log` shows `-0` distinctly.
*Common wrong prediction:* `0 1 2`, either expecting `-0` to print as `0` or
expecting banker's rounding (round half to even), which is what Python's
`round` does.

**Line 12: `9007199254740993n bigint`.** A `BigInt` literal is an
arbitrary-precision integer, so it keeps the value the double could not.
Mixing `bigint` and `number` in arithmetic throws a `TypeError`.

</details>

## Array sort and holes

Source: [`predict/02-array-sort-and-holes.ts`](predict/02-array-sort-and-holes.ts)

How does `sort` compare by default, and what is a hole in an array?

<details><summary>Output and explanation</summary>

```text
1. 1,10,100,9
2. 1,9,10,100
3. 1,2,3 true
4. 3 false undefined
5. 2,,6
6. 2
7. 3 || --
8. 1,7,11
9. 1,NaN,3
10. false true -1
11. 10,9,,true
12. 2,3 1,4
```

Two rules. First, `Array.prototype.sort` with no comparator **converts every
element to a string and compares UTF-16 code units**, and it sorts in place.
Second, arrays are objects with integer-like keys, and a **hole** is an index
with no property at all, which is different from a property holding
`undefined`. Iteration methods from ES5 (`map`, `filter`, `forEach`) skip
holes; spread and `for...of` (which use the iterator) treat them as
`undefined`; `join` prints both holes and `undefined` (and `null`) as empty
strings.

**Line 1: `1,10,100,9`.** Compared as strings, `"10"` and `"100"` sort before
`"9"` because `"1" < "9"`.
*Common wrong prediction:* `1,9,10,100`, from assuming numbers sort
numerically.

**Line 2: `1,9,10,100`.** A comparator returning a negative, zero, or positive
number gives numeric order; `a - b` is the standard one.

**Line 3: `1,2,3 true`.** `sort` mutates the array and returns the same array
object.
*Common wrong prediction:* that `nums` is unchanged and `sorted` is a copy.
(`toSorted` is the non-mutating version.)

**Line 4: `3 false undefined`.** `[1, , 3]` has length 3 but no property `"1"`,
so `1 in sparse` is false. Reading a missing property gives `undefined`,
which is why reading alone cannot tell a hole from an `undefined`.

**Line 5: `2,,6`.** `map` skips the hole (the callback is not called for it)
but the result keeps the hole at the same index; `join` prints it as empty.
*Common wrong prediction:* `2,NaN,6`, from expecting the callback to receive
`undefined` and compute `undefined * 2`.

**Line 6: `2`.** `filter` skips holes entirely, so only two elements are
offered and kept.

**Line 7: `3 || --`.** `Array(3)` with one numeric argument makes an empty
array of length 3: three holes. Spreading uses the iterator, which yields
`undefined` for each index, so `[...Array(3)]` is three real `undefined`
values; `join("|")` prints each as empty, giving `||`. `Array(3).join("-")`
joins three empties with two separators.
*Common wrong prediction:* `undefined|undefined|undefined`, not knowing `join`
renders `undefined` as empty.

**Line 8: `1,7,11`.** `map` calls its callback with `(value, index, array)`.
`Number` ignores the extra arguments.

**Line 9: `1,NaN,3`.** `parseInt` takes a second argument, the radix, and `map`
passes the index there: `parseInt("1", 0)` treats radix 0 as "default" and
gives 1; `parseInt("7", 1)` has the invalid radix 1, giving NaN;
`parseInt("11", 2)` reads binary 11, which is 3. The type assertion in the
source hides this from the compiler.
*Common wrong prediction:* `1,7,11`. Pass `(s) => parseInt(s, 10)` instead.

**Line 10: `false true -1`.** Objects compare by reference, so a fresh `{ a: 1 }`
is never found. `includes` uses SameValueZero, under which NaN equals NaN;
`indexOf` uses strict equality (`===`), under which it does not.
*Common wrong prediction:* `false true 0`, assuming the two methods agree.

**Line 11: `10,9,,true`.** Default sort stringifies `10`, `"9"`, `true`, `null`
to `"10"`, `"9"`, `"true"`, `"null"` and orders them `"10" < "9" < "null" <
"true"`. The sorted array is `[10, "9", null, true]`, and `join` prints `null`
as empty. (Only `undefined` gets special treatment in `sort`: it always goes
last.)
*Common wrong prediction:* a type error, or `null` sorting first as if it were
0.

**Line 12: `2,3 1,4`.** `splice(start, count)` removes elements in place and
returns them; `slice` would return a copy and leave `xs` alone.

</details>

## Equality and coercion

Source: [`predict/03-equality-coercion.ts`](predict/03-equality-coercion.ts)

What does `==` actually do, and how do truthiness and the fallback operators
treat "empty" values?

<details><summary>Output and explanation</summary>

```text
1. true false
2. true false
3. false true false
4. true true true false
5. [object Object] string
6. false true
7. object object number undefined
8. true false
9. true true false true false
10. fallback 0 x 
11. undefined absent
12. 1,2,3 [object Object] null
```

Line 10 ends with a trailing space: the last value is the empty string.

The rule: `===` compares type and value with no conversion. `==` runs the
**Abstract Equality algorithm**: `null` and `undefined` equal each other and
nothing else; otherwise, if the types differ, booleans become numbers, objects
become primitives (via `valueOf`/`toString`), and a string compared with a
number becomes a number. Relational operators (`<`, `>=`) use a *different*
algorithm that converts both sides to numbers, including `null` (to 0).

**Line 1: `true false`.** `1 == "1"` converts the string to the number 1.
`===` sees different types.

**Line 2: `true false`.** `null == undefined` is a special case in the `==`
algorithm; they are different types, so `===` is false.

**Line 3: `false true false`.** `==` never converts `null`: it is equal only to
`null` and `undefined`, so `null == 0` is false. `>=` converts `null` to 0, so
`0 >= 0` is true and `0 > 0` is false.
*Common wrong prediction:* `true true false` or `false false false`, assuming
`==` and `>=` use one consistent notion of equality. They do not; `null >= 0`
being true while `null == 0` and `null > 0` are false is the result.

**Line 4: `true true true false`.** `[] == false`: the boolean becomes 0, the
array becomes the primitive `""` (its `toString`), which becomes 0, and
`0 == 0`. `"" == 0` and `"0" == 0` convert the string to a number (0 both
times). `"" == "0"` compares two strings directly, with no conversion, and
they differ.
*Common wrong prediction:* `false` for `[] == false` because "an empty array
is truthy" (it is: see line 9). `==` does not use truthiness. And `==` is not
transitive: `""` and `"0"` both equal `0` but not each other.

**Line 5: `[object Object] string`.** `+` with an object operand converts both
to primitives: `[]` becomes `""`, `{}` becomes `"[object Object]"`, and since a
string is involved, `+` concatenates. `[] + []` is `""`, a string.
*Common wrong prediction:* `0` or `NaN`, from expecting numeric addition.

**Line 6: `false true`.** NaN is not equal to itself under `===` (IEEE 754),
but `Object.is` uses SameValue, which treats NaN as itself.

**Line 7: `object object number undefined`.** `typeof null` is `"object"`, a
bug from the first implementation kept for compatibility. Arrays are objects.
NaN is a number value.
*Common wrong prediction:* `null array NaN undefined`.

**Line 8: `true false`.** `Array.isArray` is the reliable array test. `null` is
a primitive, not an object, so `instanceof Object` is false despite line 7.

**Line 9: `true true false true false`.** The only falsy values are `false`,
`0`, `-0`, `0n`, `""`, `null`, `undefined`, and `NaN`. Every object, including
an empty array or object, is truthy, and `"0"` is a non-empty string.
*Common wrong prediction:* `false` for `[]` or `{}`, from Python, where empty
containers are falsy.

**Line 10: `fallback 0 x ` (trailing space).** `||` returns the right side when
the left is **falsy**, so `0` and `""` both fall back. `??` returns the right
side only when the left is **`null` or `undefined`**, so `0` and `""` are kept.
*Common wrong prediction:* `fallback fallback x x`, treating `??` as a new
spelling of `||`. `??` exists precisely so that 0 and the empty string can be
real values.

**Line 11: `undefined absent`.** `maybe.a` is `undefined`, so `?.` stops the
whole remaining chain and evaluates to `undefined` without reading `.b`. `??`
then supplies `"absent"`.

**Line 12: `1,2,3 [object Object] null`.** `String(array)` calls
`Array.prototype.toString`, which is `join(",")`, and `join` stringifies nested
arrays the same way, so nesting flattens. A plain object has only the generic
`Object.prototype.toString`. `String(null)` is `"null"` (unlike `join`, which
renders `null` as empty).

</details>

## How this is bound

Source: [`predict/04-this-binding.ts`](predict/04-this-binding.ts)

What is `this` inside a method, and when does it stop being the object?

<details><summary>Output and explanation</summary>

```text
1. 1 2
2. threw TypeError
3. 3
4. 4 4
5. 101 4
6. 7 undefined
7. 5,6
8. 1001
```

The rule: for an ordinary function, `this` is decided **at the call site**.
`obj.f()` passes `obj` as `this`; a bare call `f()` passes `undefined` in
strict mode (class bodies and ES modules are always strict). `call`, `apply`,
and `bind` set it explicitly, and a function produced by `bind` has its `this`
fixed permanently. **Arrow functions have no `this` of their own**: they read
`this` from the scope where they were defined, and nothing at the call site can
change it.

**Line 1: `1 2`.** `c.inc()` is a method call, so `this` is `c`. Arguments are
evaluated left to right, so the counter goes to 1 and then 2.

**Line 2: `threw TypeError`.** `const loose = c.inc` copies the function, not
the pairing with `c`. `loose()` is a bare call, so `this` is `undefined` and
`this.n` throws `TypeError` ("Cannot read properties of undefined").
*Common wrong prediction:* `3`, believing a method remembers its object. In
sloppy-mode scripts it would instead silently read `globalThis.n`; strict mode
turns that silent bug into an exception.

**Line 3: `3`.** `bind(c)` returns a new function whose `this` is always `c`.
The failed call in line 2 did not change `c.n`.

**Line 4: `4 4`.** `incArrow` is a class field, so it is created per instance
in the constructor, and the arrow captured that constructor's `this` (the
instance). Calling it detached still increments `c.n`, to 4; then `c.n` is
read, also 4.
*Common wrong prediction:* a `TypeError` as in line 2. Arrow-function fields
are the usual fix for callbacks, at the cost of one function object per
instance instead of one shared method on the prototype.

**Line 5: `101 4`.** `call` invokes `inc` with `this` set to `{ n: 100 }`, which
it increments to 101. `c` is untouched.

**Line 6: `7 undefined`.** `obj.get()` is a method call, so `this` is `obj`.
`getArrow` is an arrow defined in an object literal: an object literal does
not create a `this` scope, so the arrow's `this` is the module's (which is
`undefined` in an ES module). The arrow reads `globalThis.n`, which does not
exist.
*Common wrong prediction:* `7 7`, believing an arrow defined "inside" an object
binds to that object. Only functions create a `this`; braces of an object
literal do not.

**Line 7: `5,6`.** `map` calls its callback as a plain function with
`(value, index, array)`. `incArrow` ignores the arguments and its `this` is
fixed to `c`, so the two calls increment `c.n` to 5 and 6.
*Common wrong prediction:* `2,3` (treating the elements as inputs) or a
`TypeError` (expecting the callback to lose `this`, which `c.inc` would).

**Line 8: `1001`.** A bound function ignores any `this` it is called with, and
binding it again only wraps it: the outer wrapper passes `{ n: 0 }` to the
inner bound function, which discards it and uses `{ n: 1000 }`.
*Common wrong prediction:* `1`, assuming the last `bind` wins.

</details>

## Types are erased

Source: [`predict/05-types-are-erased.ts`](predict/05-types-are-erased.ts)

What is left of TypeScript's types when the program actually runs?

<details><summary>Output and explanation</summary>

```text
1. string undefined
2. not-a-number1
3. 4
4. undefined
5. undefined a
6. false true
7. {"b":null,"d":1}
8. [null,null,3]
9. 1,2,b,a
10. true
```

The rule: TypeScript's types are **erased**. Interfaces, type aliases, generic
parameters, and `as` assertions compile to nothing; the program that runs is
the JavaScript left after deleting them. An assertion (`as User`) is a claim
the compiler accepts without checking, and the runtime never checks it either.
The only runtime checks are the ones written in JavaScript.

**Line 1: `string undefined`.** `raw as User` changed what the compiler
believes, not the object. At run time `user.id` is still the string
`"not-a-number"` and there is no `name` property.
*Common wrong prediction:* `number` and an error for the missing name,
treating `as` like a cast that converts or validates.

**Line 2: `not-a-number1`.** The compiler typed `user.id + 1` as numeric
addition, but the runtime sees a string plus a number, and `+` concatenates.

**Line 3: `4`.** `JSON.parse` returns `any`, so assigning it to `Shape` is
unchecked; this time the data happens to match, and `r * 2` is 4.

**Line 4: `undefined`.** The parsed object has `kind: "square"` but no `s`.
Narrowing on `kind` convinces the compiler that `s` is a number; at run time
the property is missing.
*Common wrong prediction:* an error at the `JSON.parse` line, from expecting
the annotation to validate the data. Validation at a trust boundary must be
done by code (a schema library or a hand-written guard).

**Line 5: `undefined a`.** `first<number>([])` returns `xs[0]` of an empty array,
`undefined`. The `<number>` exists only for the compiler.

**Line 6: `false true`.** `looksLikeUser` is a type guard written in
JavaScript, so it really runs: `user.id` is a string, so it is false; the
second object has a numeric `id`, so it is true. (It only checks `id`, so it
is a weak guard: the compiler trusts a `v is User` predicate as much as an
assertion.)

**Line 7: `{"b":null,"d":1}`.** `JSON.stringify` omits object properties whose
values are `undefined` or functions, because JSON has no representation for
them. `null` is valid JSON and is kept.

**Line 8: `[null,null,3]`.** In an array a value cannot be omitted without
shifting indices, so `undefined` and functions become `null`.
*Common wrong prediction:* `[3]` or `[undefined,null,3]`, assuming arrays
behave like objects here.

**Line 9: `1,2,b,a`.** Property order is defined: integer-like keys come first
in ascending numeric order, then string keys in insertion order. So `1` and
`2` move to the front regardless of where they were written.
*Common wrong prediction:* `b,2,a,1` (insertion order for everything), or
alphabetical order.

**Line 10: `true`.** `type Meters = number` and `type Feet = number` are
aliases of the same type, so assigning one to the other is not even a type
error, and at run time both are just `100`. TypeScript is structural: types
with the same shape are the same type. Distinguishing units needs a "brand"
(an intersection with a unique marker property) and still leaves nothing at
run time.

</details>

## Async ordering

Source: [`predict/06-async-ordering.ts`](predict/06-async-ordering.ts)

In what order do synchronous code, promise callbacks, `await` continuations,
and timers run?

<details><summary>Output and explanation</summary>

```text
A: sync start
E: async body, before await
G: sync end
K: executor body
M: really the end
C: queueMicrotask
D: promise.then
F: async body, after await
H: first then
J: after await 1
L: after executor
I: nested then
B: setTimeout 0
```

The rule: the event loop runs the current script to completion, then
**drains the entire microtask queue** (promise reactions, `await`
continuations, `queueMicrotask`) in FIFO order, including microtasks queued
while draining, and only then takes the next **macrotask** (a timer, I/O
callback). Separately, two things that look asynchronous run synchronously:
an `async` function's body up to its first `await`, and a `Promise`
executor.

**Synchronous phase: A, E, G, K, M.** `A` prints. `setTimeout` schedules `B` as
a macrotask. `queueMicrotask` enqueues `C`. `Promise.resolve().then` enqueues
`D` (the promise is already resolved). `run()` executes its body synchronously,
printing `E`, then `await null` suspends it and enqueues its continuation
(`F`). `G` prints. The next `then` enqueues `H`. The async IIFE runs until
`await 1` and enqueues its continuation (`J`). The `Promise` executor runs
immediately, printing `K`, and resolves, so the `then` enqueues `L`. `M`
prints.
*Common wrong prediction:* `E` after `G` (believing an async function's body
starts later) or `K` after `M` (believing the executor is deferred).

**Microtask phase: C, D, F, H, J, L, then I.** The queue at the end of the script
is `C, D, F, H, J, L`, in the order they were enqueued, and they run in that
order. When `H` runs, it enqueues `I` at the back of the queue, behind `J` and
`L`. The queue is drained completely, so `I` still runs before any macrotask.
*Common wrong predictions:* `I` immediately after `H` (thinking a nested
`then` runs right away), or `F` and `J` much later (from older engines where
`await` took three microtask ticks; since ES2019, awaiting a non-promise or a
native promise takes exactly one).

**Macrotask: B.** The timer was the first thing scheduled, but a `0` ms timeout
means "the next macrotask turn at the earliest", which comes only after the
microtask queue is empty.
*Common wrong prediction:* `B` right after the synchronous lines, reading
`setTimeout(fn, 0)` as "run immediately". A microtask that keeps enqueueing
microtasks can starve timers forever for the same reason.

</details>

## Where this leads

Every one of these rules shows up in production TypeScript: currency amounts
in floating point, numeric arrays sorted as strings, methods passed as
callbacks that lose `this`, API responses cast with `as` and never validated,
and ordering bugs between promises and timers. Compare them with the
[Go](../go/README.md), [Rust](../rust/README.md), and
[Python](../python/README.md) sets, using the pairings described in the
[predict introduction](../README.md#the-corpus).

## Key terms

- **IEEE 754 double**: the 64-bit floating-point format of every JavaScript `number`, exact for integers up to 2^53.
- **Hole**: an array index with no property at all, distinct from a property holding `undefined`.
- **Abstract Equality**: the coercing algorithm behind `==`, as opposed to the non-converting `===`.
- **Call-site `this`**: the rule that an ordinary function's `this` is set by how it is called, not where it is defined.
- **Type erasure**: the removal of all type annotations, interfaces, and assertions before the program runs.
- **Microtask queue**: the queue of promise reactions and `await` continuations, drained completely before the next macrotask.

## Summary

- Every `number` is a double; integers beyond 2^53 lose precision silently.
- Default `sort` compares strings in place, and ES5 iteration methods skip holes while spread and `for...of` do not.
- `==` coerces by a fixed algorithm, `null == undefined` and nothing else, and relational operators use a different numeric conversion.
- `this` comes from the call site unless fixed by `bind` or inherited by an arrow function.
- Types and assertions are erased; only JavaScript checks run at run time.
- Synchronous code runs first, then the whole microtask queue, then the next timer or I/O callback.

## Review questions

1. Why does `[10, 9, 1].sort()` return `[1, 10, 9]`?
   <details><summary>Answer</summary>With no comparator, `sort` converts each element to a string and compares UTF-16 code units, so `"10"` sorts before `"9"`. Pass `(a, b) => a - b` to sort numerically.</details>
2. Why does a method passed as a callback lose its object, and why does an arrow function not?
   <details><summary>Answer</summary>An ordinary function's `this` is decided at the call site; a bare call passes `undefined` in strict mode. An arrow function has no `this` of its own and reads it from the scope where it was defined, so the call site cannot change it.</details>
3. Why does a `setTimeout(fn, 0)` callback run after promise callbacks scheduled later in the same script?
   <details><summary>Answer</summary>A timer is a macrotask. After the script finishes, the event loop drains the entire microtask queue, including microtasks queued during draining, before taking the next macrotask.</details>
