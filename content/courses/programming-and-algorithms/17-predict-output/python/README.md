# Predicting Python output

Python is easy to read, and that is the trap: code that reads like pseudocode
invites you to run it in your head with pseudocode semantics. Real Python has
precise and sometimes surprising rules about when expressions are evaluated,
what a name is bound to, and where an attribute is looked up. A default list
that remembers earlier calls, lambdas that all return the same number, a copy
that is not independent, a generator that is empty the second time: each is
the language working exactly as specified. The six snippets in this section
make those rules visible.

Work through them as the [predict introduction](../README.md) describes: read
the source, write down the exact output line by line, then open the fold. Each
explanation states the rule from first principles (what objects exist, which
names refer to them, and when code actually runs), applies it to the snippet,
and names the common wrong prediction and its misconception. The goal is a
model of names, objects, and evaluation time that you can use to predict any
Python program, not a list of gotchas.

> **Learning objectives**
> - Explain why default arguments and class bodies are evaluated once, and use the `None` sentinel idiom.
> - Predict what closures capture, which constructs create a scope, and when `nonlocal` is required.
> - Distinguish identity (`is`) from equality (`==`), including CPython caching details and NaN in containers.
> - Predict how far a copy goes and when `+=` mutates in place.
> - Trace generator execution and exhaustion, and attribute lookup through the MRO and `super()`.

## Before you begin

The outputs were produced with CPython 3.13. Two lines in the
[identity snippet](#identity-versus-equality) depend on CPython implementation
details rather than the language definition, and they are marked as such. To
reproduce an output, run `python3 predict/NN-name.py` from this directory. The
aliasing snippet is the Python face of a model that also appears in
[Go's slice snippet](../go/README.md#slice-aliasing-and-append) and
[Rust's move snippet](../rust/README.md#moves-and-reference-counting).

## Mutable default arguments

Source: [`predict/01-mutable-default-arg.py`](../../../../../practice/predict/python/predict/01-mutable-default-arg.py)

When is a default argument value created, and who shares it?

<details><summary>Output and explanation</summary>

```text
1. [1]
2. [1, 2]
3. [3]
4. [1, 2, 4]
5. [1] [2]
6. 0 0 0
7. 1
8. ([1, 2, 4],)
9. 1 2
10. ['x', 'y']
```

The rule: `def` is an executable statement. When it runs, it evaluates each
default expression **once** and stores the resulting objects on the function
object (in `__defaults__`). Every call that omits the argument binds the
parameter to that same stored object. Nothing is re-evaluated per call.

**Line 1: `[1]`.** The first call appends 1 to the single default list and
returns it.

**Line 2: `[1, 2]`.** The second call gets the *same* list, which already holds
1.
*Common wrong prediction:* `[2]`, from the model (true in C++, JavaScript, and
Kotlin) that a default expression is evaluated fresh on each call.

**Line 3: `[3]`.** An explicit `[]` is a new list created at the call site, so
the shared default is not involved.

**Line 4: `[1, 2, 4]`.** Back to the shared default, which line 3 did not touch.
*Common wrong prediction:* `[3, 4]` or `[4]`, believing that passing a list in
line 3 somehow replaced or reset the default.

**Line 5: `[1] [2]`.** `add_safe` uses the idiom: default to the immutable
sentinel `None`, and create a fresh list inside the body, which *does* run per
call.

**Line 6: `0 0 0`.** The default `n=next(counter)` was evaluated once, when
`def stamped` ran, and produced 0. Calls do not call `next` again.
*Common wrong prediction:* `0 1 2`, the same per-call-evaluation model as line
2, now with a side effect that makes it visible.

**Line 7: `1`.** `counter` has been advanced exactly once (at `def` time), so
this `next` returns 1.

**Line 8: `([1, 2, 4],)`.** `__defaults__` is a tuple holding the one default
object, and it shows the accumulated mutations. The default is ordinary,
inspectable, mutable state.

**Line 9: `1 2`.** The same rule applies to a class body: it runs once, so
`seen = []` creates one list stored on the class. `self.seen` finds no
instance attribute and falls back to the class attribute, and `.append`
mutates that shared list. `a.note` makes its length 1, `b.note` makes it 2.
*Common wrong prediction:* `1 1`, from reading the class body as a per-instance
initialiser. Per-instance state belongs in `__init__`.

**Line 10: `['x', 'y']`.** Both instances appended to the one class-level list.

</details>

## Late-binding closures

Source: [`predict/02-late-binding-closures.py`](../../../../../practice/predict/python/predict/02-late-binding-closures.py)

What does a closure capture, and which scopes does a loop variable live in?

<details><summary>Output and explanation</summary>

```text
1. [2, 2, 2]
2. [0, 1, 2]
3. outer
4. 2
5. 1 3 6
6. 16 10
7. UnboundLocalError
8. [16]
```

The rule: a closure captures **variables** (through a cell object), not
values. A free name is looked up when the function body *runs*. Python has
function scope, not block scope: a `for` loop does not create a scope, but a
comprehension does (it is compiled as a hidden function). Whether a name is
local to a function is decided at compile time: any assignment to it anywhere
in the body makes it local throughout, unless declared `nonlocal` or `global`.

**Line 1: `[2, 2, 2]`.** All three lambdas close over the one `i` of the
comprehension's scope. They are called after the comprehension finished, when
that `i` holds its last value, 2.
*Common wrong prediction:* `[0, 1, 2]`, from the model that a lambda snapshots
the value it sees at creation. It snapshots nothing; it remembers where to look.

**Line 2: `[0, 1, 2]`.** `lambda i=i: i` evaluates the default `i` when each
lambda is created (the rule from
[mutable default arguments](#mutable-default-arguments)), copying the current value into
the lambda's own parameter. This is the standard fix (another is
`functools.partial`).

**Line 3: `outer`.** In Python 3 the comprehension's `i` lives in the
comprehension's own scope, so the module-level `i` is untouched.
*Common wrong prediction:* `2`, which was the Python 2 behaviour for list
comprehensions.

**Line 4: `2`.** A plain `for` loop binds `j` in the enclosing (module) scope
and does not delete it afterwards, so it holds the last value.
*Common wrong prediction:* a `NameError`, from block-scoped languages where the
loop variable disappears with the loop.

**Line 5: `1 3 6`.** `nonlocal total` makes `bump` write to `make_counter`'s
`total` cell rather than create a local, so the running total accumulates: 1,
then 3, then 6.

**Line 6: `16 10`.** `c` continues from 6, so `c(10)` is 16. `d` came from a
second call to `make_counter`, which created a new `total` cell starting at 0,
so `d(10)` is 10.
*Common wrong prediction:* `16 16`, believing closures from the same factory
share state. Each call of the outer function creates fresh cells.

**Line 7: `UnboundLocalError`.** `count += 1` is an assignment to `count`, so the
compiler makes `count` local to the inner `bump` for the whole body. `+=` must
first read that local, which has not been assigned yet, so it raises
`UnboundLocalError` (a subclass of `NameError`). The outer `count = 0` is never
consulted.
*Common wrong prediction:* `1`, assuming the read falls back to the enclosing
scope and the write then updates it. Scope is decided per name per function at
compile time, not per access at run time.

**Line 8: `[16]`.** `c.__closure__` is the tuple of cells the inner function
closes over: just `total`, now 16. The cell is real, inspectable state, which
makes line 1 concrete: the lambdas there all hold the same cell.

</details>

## Identity versus equality

Source: [`predict/03-identity-vs-equality.py`](../../../../../practice/predict/python/predict/03-identity-vs-equality.py)

What does `is` actually compare, and when does `==` surprise you?

<details><summary>Output and explanation</summary>

```text
1. True
2. True False
3. True False
4. True True False
5. False True True
6. False True
7. True True {1: 'bool'}
8. True False
9. True True
```

The rule: `a is b` asks whether two names refer to **the same object** (the
same address). `a == b` calls `__eq__` and asks whether the objects consider
themselves equal. Whether two equal immutable values are the same object
depends on caching and constant sharing, which are CPython implementation
details, not guarantees.

**Line 1: `True`.** *(CPython detail.)* CPython preallocates the integers from
-5 to 256 and every operation producing one of them returns the cached object.
(Two equal literals in the same code object would also share one constant.)
*Common wrong prediction:* `False`, but the more dangerous error is the
opposite: concluding from this line that `is` works for comparing integers.

**Line 2: `True False`.** *(CPython detail.)* `int("257")` builds the value at
run time, outside the small-int cache, so each call allocates a new object.
They are equal but not identical.
*Common wrong prediction:* `True True`, by generalising line 1. This is the
"right up until the value comes from input" bug: code that tested fine with
literals breaks on parsed data.

**Line 3: `True False`.** Each list display creates a new mutable list. Equal
contents, two objects.

**Line 4: `True True False`.** The two `"hello"` literals are the same constant
(and identifier-like string literals are interned), so `s1 is s2`. `"".join`
builds a new string at run time: equal, not identical.
*Common wrong prediction:* `True True True`, from thinking all equal strings
are interned.

**Line 5: `False True True`.** IEEE 754 says NaN is not equal to anything,
including itself, so `nan == nan` is False. But container comparison and `in`
check **identity first**: `x is y or x == y`. Both lists hold the same NaN
object, so the identity check succeeds before `==` is consulted.
*Common wrong prediction:* `False False False`, from assuming containers just
call `==` on elements.

**Line 6: `False True`.** 0.1, 0.2, and 0.3 have no exact binary
representation. The nearest doubles to 0.1 and 0.2 sum to
`0.30000000000000004`, which is a different double from the nearest one to 0.3.
Compare floats with a tolerance (or `math.isclose`).

**Line 7: `True True {1: 'bool'}`.** `bool` is a subclass of `int`, with `True`
equal to 1 and `hash(True) == hash(1)`. A dict uses hash then equality to find
keys, so `True` finds the existing key `1`. The dict keeps the **original key
object** (`1`) and replaces only the value.
*Common wrong prediction:* `{1: 'int', True: 'bool'}` (two keys) or
`{True: 'bool'}` (the key replaced too).

**Line 8: `True False`.** *(CPython detail.)* The empty tuple is a singleton,
because an immutable empty container can be shared safely. Empty lists are
mutable, so each `[]` must be a distinct object.

**Line 9: `True True`.** `None` is a guaranteed singleton, so `is None` is
correct and fast. `== None` also gives True here, but it calls `__eq__`, which a
class can override to return anything; that is why `is None` is the idiom and
the only identity check that belongs in ordinary code.

</details>

## Aliasing and copies

Source: [`predict/04-aliasing-and-copies.py`](../../../../../practice/predict/python/predict/04-aliasing-and-copies.py)

When do two names share one object, and how deep does a copy go?

<details><summary>Output and explanation</summary>

```text
1. [[9, 0, 0], [9, 0, 0], [9, 0, 0]]
2. [[9, 0, 0], [0, 0, 0], [0, 0, 0]]
3. [[1, 2, 99], [3, 4]]
4. [[1, 2], [3, 4]]
5. {'k': [1, 2]}
6. abcd abc
7. [1, 2, 3] [1, 2]
8. [1, 2, 3] [1, 2, 3]
9. [[1, 0], [2]] [[1, 0], [2], [3]]
10. ['a', 'bb', 'dd', 'ccc'] ['bb', 'a', 'ccc', 'dd']
```

The rule: Python variables are names bound to objects, and containers hold
references. Assignment binds a name; it never copies. Any copy (slicing,
`copy.copy`, `dict(d)`, `list(xs)`) is **shallow**: a new outer container
holding the same element references. Only `copy.deepcopy` recurses.

**Line 1: `[[9, 0, 0], [9, 0, 0], [9, 0, 0]]`.** `[[0] * 3] * 3` evaluates the
inner list once and then builds an outer list with three references to it.
There is one row object, seen three times, so one write shows up "everywhere".
*Common wrong prediction:* `[[9, 0, 0], [0, 0, 0], [0, 0, 0]]`, the model that
`*` copies the element. It repeats the reference. (For `[0] * 3` that is
harmless because ints are immutable.)

**Line 2: `[[9, 0, 0], [0, 0, 0], [0, 0, 0]]`.** The comprehension evaluates
`[0] * 3` once per iteration, creating three distinct rows.

**Line 3: `[[1, 2, 99], [3, 4]]`.** `copy.copy` made a new outer list containing
the same two inner lists, so appending to `nested[0]` is visible through
`shallow[0]`.
*Common wrong prediction:* `[[1, 2], [3, 4]]`, from thinking "copy" means
independent.

**Line 4: `[[1, 2], [3, 4]]`.** `deepcopy` copied the inner lists too.

**Line 5: `{'k': [1, 2]}`.** `dict(d)` is a shallow copy: a new dict whose value
for `"k"` is the same list object.

**Line 6: `abcd abc`.** Strings are immutable, so `s += "d"` creates a new
string and rebinds `s`. `t` still names the old one.

**Line 7: `[1, 2, 3] [1, 2]`.** `xs + [3]` creates a new list, then `xs` is
rebound to it. `ys` still names the original.

**Line 8: `[1, 2, 3] [1, 2, 3]`.** For a list, `+=` calls `__iadd__`, which
extends the list **in place** and returns the same object. `ps` and `qs` are
still one list.
*Common wrong prediction:* `[1, 2, 3] [1, 2]`, by assuming `x += y` always means
`x = x + y`. It only falls back to that when the type has no `__iadd__`, as
with immutable `str` in line 6.

**Line 9: `[[1, 0], [2]] [[1, 0], [2], [3]]`.** `rows[:]` is a shallow copy: a
new outer list sharing the inner lists. Appending to `rows[0]` mutates a shared
inner list (both see `[1, 0]`). Appending to `sliced` changes only its own
outer list.
*Common wrong prediction:* either fully shared (`[3]` appearing in `rows` too)
or fully independent (`[1]` in `sliced`). Shallow is exactly one level.

**Line 10: `['a', 'bb', 'dd', 'ccc'] ['bb', 'a', 'ccc', 'dd']`.** `sorted`
returns a new list and leaves `words` alone. Sorting by `len` gives lengths 1,
2, 2, 3; `'bb'` and `'dd'` tie, and Python's sort is **stable**, so they keep
their original relative order (`'bb'` came first).
*Common wrong prediction:* `'dd'` before `'bb'` (expecting an unstable sort or
a secondary alphabetical order), or `words` sorted too (confusing `sorted`
with `list.sort`).

</details>

## Generators and exhaustion

Source: [`predict/05-generators-and-exhaustion.py`](../../../../../practice/predict/python/predict/05-generators-and-exhaustion.py)

When does a generator's body run, and what happens when you iterate it twice?

<details><summary>Output and explanation</summary>

```text
1. created, nothing has run
  body starts
2. 1
  between yields
  body ends
3. [2]
4. []
5. 14 0
6. [(1, 'a'), (2, 'b')]
7. [4]
8. True [1, 2]
9. a
10. done
11. [0, 1, 2, 3]
```

The rule: calling a generator function runs **none** of its body; it returns a
generator object, a paused frame. Each `next` resumes the frame until the next
`yield` and suspends again. When the body finishes, `StopIteration` is raised,
and from then on the generator is permanently exhausted: iterating it again
silently yields nothing. Iterators in general (including `iter(list)`) are
one-shot cursors.

**Line 1.** `g = gen()` printed nothing: the body has not started.
*Common wrong prediction:* `  body starts` appearing before line 1, from
treating `gen()` as an ordinary call.

**Line 2 block: `  body starts`, then `2. 1`.** `next(g)` is evaluated before
`print` can output anything, and it runs the body up to the first `yield`,
printing `  body starts` on the way. Then `print` shows `2. 1`.

**Line 3 block: `  between yields`, `  body ends`, `3. [2]`.** `list(g)` resumes
from the first `yield`, prints, yields 2, resumes, prints `  body ends`, and
finishes. The 1 was already consumed by `next`, so the list is `[2]`; it is
printed after both side effects because `list(g)` completes first.
*Common wrong prediction:* `3. [1, 2]`, forgetting that the cursor already
moved.

**Line 4: `[]`.** The generator is exhausted. No error, just empty.
*Common wrong prediction:* `[1, 2]` (treating a generator like a list you can
reread) or an exception.

**Line 5: `14 0`.** The first `sum` consumes `0 + 1 + 4 + 9 = 14`. The second
gets an exhausted generator expression and sums nothing.
*Common wrong prediction:* `14 14`. This bug ships when a function takes "an
iterable", iterates it twice, and is passed a generator.

**Line 6: `[(1, 'a'), (2, 'b')]`.** `zip` stops at the shortest input.

**Line 7: `[4]`.** `zip` pulls from its arguments left to right. To discover
that it is done, it took 3 from `it`, then asked `"ab"` for a third item, got
`StopIteration`, and stopped. The 3 was already consumed and is thrown away.
*Common wrong prediction:* `[3, 4]`, assuming `zip` stops before touching the
longer iterator. Put the shortest iterable first if the leftovers matter.

**Line 8: `True [1, 2]`.** `any` short-circuits on the first truthy value. The
generator expression is lazy, so `note(3)` is never evaluated: `seen` records
only 1 and 2.
*Common wrong prediction:* `True [1, 2, 3]`, believing the generator
expression computes every element first.

**Line 9: `a`.** The first `next` runs to the `yield`.

**Line 10: `done`.** The second `next` runs `return "done"`, which ends the
generator by raising `StopIteration` with `.value == "done"`. A returned value
is not yielded; it is available only on the exception (and as the result of
`yield from`).
*Common wrong prediction:* `next(w)` returning `"done"` as a normal value.

**Line 11: `[0, 1, 2, 3]`.** `naturals()` is infinite, which is fine because it
is lazy. `zip` asks `range(4)` first each round; when `range` is exhausted,
`zip` stops before pulling another value from `naturals()`.

</details>

## Class attributes and the MRO

Source: [`predict/06-class-attrs-and-mro.py`](../../../../../practice/predict/python/predict/06-class-attrs-and-mro.py)

Where does attribute lookup look, where does assignment write, and what does
`super()` call?

<details><summary>Output and explanation</summary>

```text
1. ['a', 'b'] True
2. ['shadowed'] ['a', 'b']
3. 1 2 1 0
4. BCA
5. ['D', 'B', 'C', 'A', 'object']
6. BA CA
7. AttributeError False
```

The rule: **reading** `obj.attr` looks in the instance's `__dict__`, then in
each class along `type(obj).__mro__` (the method resolution order). **Writing**
`obj.attr = value` never walks: it always stores into the instance's own
`__dict__`. `super()` inside a method of class `X` means "the next class after
`X` in the MRO of the actual instance", not "my base class".

**Line 1: `['a', 'b'] True`.** `entries = []` is a class attribute: one list.
`self.entries.append(name)` *reads* `self.entries` (not found on the instance,
found on the class) and mutates that shared list.

**Line 2: `['shadowed'] ['a', 'b']`.** `a.entries = [...]` is a *write*, so it
creates an instance attribute on `a` that shadows the class attribute for `a`
only. `b` still reads through to the class list.

**Line 3: `1 2 1 0`.** `self.count += 1` is a read followed by a write. The read
finds the class's `0`, adds 1, and the write stores `1` on the **instance**.
From then on `c1` reads its own attribute: the second bump gives 2. `c2` has no
instance attribute yet, reads the class's `0`, and stores its own 1. The class
attribute was never written and is still 0.
*Common wrong prediction:* `1 2 3 3`, believing a class attribute is a shared
counter. It is a shared default that each instance silently replaces on its
first write. (Contrast line 1: mutating a shared list through a read is
shared; rebinding is not.)

**Line 4: `BCA`.** `D().who()` finds `B.who` first. Inside it, `super()` is the
class after `B` in **`D`'s** MRO, which is `C`, not `A`. `C.who` then calls
`super()`, the class after `C`, which is `A`. Result: `"B" + "C" + "A"`.
*Common wrong prediction:* `BA`, reading `super()` as "my parent class". That
model breaks under multiple inheritance; cooperative `super()` is what makes
each class in a diamond run exactly once.

**Line 5: `['D', 'B', 'C', 'A', 'object']`.** C3 linearisation: a class comes
before its bases, bases keep their declared left-to-right order (`B` before
`C`), and a shared base (`A`) comes after every class that inherits it.
*Common wrong prediction:* `D, B, A, C, A, object` or `D, B, A, C, object`, the
depth-first order, which would run `A` before `C`.

**Line 6: `BA CA`.** For a plain `B()` instance the MRO is `B, A, object`, so
`super()` in `B.who` is `A`. Same code as line 4, different instance, different
next class: `super()` is decided by the instance's MRO at run time.

**Line 7: `AttributeError False`.** `__slots__ = ("x",)` replaces the per-instance
`__dict__` with fixed storage for `x` only. Assigning `y` has nowhere to go, so
it raises `AttributeError`, and the instance has no `__dict__`.
*Common wrong prediction:* `assigned True`, from assuming every Python object
accepts arbitrary attributes. That is true only for objects with a `__dict__`.

</details>

## Where this leads

These rules are behind a large share of real Python bugs: shared mutable
defaults in long-running services, callbacks registered in a loop that all see
the last value, grids built with `[[0] * n] * n`, functions that silently
process nothing because a generator was consumed earlier. Compare them with
the [Go](../go/README.md), [Rust](../rust/README.md), and
[TypeScript](../ts/README.md) sets, using the pairings described in the
[predict introduction](../README.md#the-corpus).

## Key terms

- **Default value evaluation**: default expressions are evaluated once, when `def` runs, and stored in `__defaults__`.
- **Cell**: the object through which a closure refers to a variable of an enclosing scope.
- **Identity**: whether two names refer to the same object, tested with `is`.
- **Shallow copy**: a new outer container holding the same element references.
- **Generator**: a paused function frame that runs to the next `yield` on each `next` and is exhausted once its body finishes.
- **MRO**: the method resolution order, the C3-linearised list of classes that attribute reads and `super()` follow.

## Summary

- `def` and class bodies run once; per-call or per-instance state must be created in the body or `__init__`.
- Closures capture variables, looked up when the function runs; comprehensions create a scope, `for` loops do not.
- `is` compares objects, `==` calls `__eq__`; small-int and string caching are CPython details, and containers check identity before equality.
- Assignment never copies, every copy except `deepcopy` is shallow, and list `+=` mutates in place.
- Generators run nothing until iterated and are one-shot; `zip` may consume an element it then discards.
- Attribute reads walk the instance and the MRO, writes go to the instance, and `super()` follows the instance's MRO.

## Review questions

1. Why does `def f(x=[])` accumulate state across calls, and what is the fix?
   <details><summary>Answer</summary>The default list is created once, when `def` runs, and every call that omits `x` binds it to that same object. Default to `None` and create a new list inside the body, which runs on every call.</details>
2. Why does `[lambda: i for i in range(3)]` produce lambdas that all return 2?
   <details><summary>Answer</summary>All three lambdas close over the same `i` variable of the comprehension's scope and read it when called, after the comprehension has left it at 2. Binding it as a default (`lambda i=i: i`) copies the current value at creation.</details>
3. Why does `self.count += 1` on a class attribute `count = 0` not produce a shared counter?
   <details><summary>Answer</summary>`+=` reads `self.count` (falling through to the class's 0) and then writes `self.count`, and writes always go to the instance's own `__dict__`. Each instance ends up with its own attribute, and the class attribute stays 0.</details>
