# Go: predict the output

Six Go snippets, each with its exact output and an explanation of every
numbered line. Read the source, write down what you think it prints, then open
the fold. See the [predict introduction](../README.md) for how to use these
pages.

The module declares `go 1.26` in [`go.mod`](go.mod). That matters for snippet
01: loop-variable semantics are chosen by the module's Go version, not by the
compiler you happen to have installed.

To reproduce an output, run `go run ./predict/NN-name` from this directory.

## 01-loop-var-capture

Source: [`predict/01-loop-var-capture/main.go`](predict/01-loop-var-capture/main.go)

What does a closure created inside a loop close over, and which loops get a
fresh variable on each iteration?

<details><summary>Output and explanation</summary>

```text
1. 0 1 2
2. 3 3 3
3. 0a 1b 2c
4. 0 1 2
```

The rule underneath all four lines: a Go closure captures **variables**, not
values. It holds a reference to the variable's storage and reads it when it is
called. So the question is never "what was `i` when the closure was made", it
is "how many distinct `i` variables exist". Since Go 1.22 (for modules that
declare `go 1.22` or later), every variable declared by a `for` statement's
header with `:=` is a **new variable on each iteration**. A variable declared
outside the loop is still one variable.

**Line 1: `0 1 2`.** `i` is declared in the three-clause header (`i := 0`), so
each iteration gets its own `i`. Before the post statement `i++` runs, Go
creates the next iteration's `i` initialised from the current one, so the
increment never touches the variable an earlier closure captured. The three
closures hold three variables containing 0, 1, 2.
*Common wrong prediction:* `3 3 3`. That was the correct answer before Go 1.22,
and a great deal of Go folklore (and the `i := i` idiom) comes from it. The
misconception is that the language behaviour is fixed; it is versioned by the
`go` line in `go.mod`.

**Line 2: `3 3 3`.** `j` is declared before the loop, so the header only
assigns to it. There is exactly one `j`, all three closures share it, and they
run after the loop has left it at 3 (the value that made `j < 3` false).
*Common wrong prediction:* `0 1 2`, from overlearning the 1.22 change as
"closures in loops are safe now". The change is about variables *declared by
the loop*, not about closures.

**Line 3: `0a 1b 2c`.** `for i, x := range xs` declares both `i` and `x`, so
both are per-iteration. Each closure sees its own pair.
*Common wrong prediction:* `2c 2c 2c`, the pre-1.22 answer, where `i` and `x`
were single variables overwritten each iteration.

**Line 4: `0 1 2`.** `range 3` (range over an integer, also new in 1.22)
declares a fresh `i` per iteration, so `&i` is a different address each time
and each pointer keeps its own value. The variable escapes to the heap because
its address outlives the iteration; Go's escape analysis handles that
silently.
*Common wrong prediction:* `2 2 2` (or `3 3 3`), from the model where the loop
reuses one slot and every pointer points at it. That was true for range loops
before 1.22.

</details>

## 02-nil-interface

Source: [`predict/02-nil-interface/main.go`](predict/02-nil-interface/main.go)

When is an `error` nil, and what does a nil pointer still let you do?

<details><summary>Output and explanation</summary>

```text
1. false
2. true
3. *main.opErr <nil>
4. true
5. false true
6. 0 0 true 1
7. recovered: true
```

The rule: an interface value is a pair **(dynamic type, dynamic value)**. It is
`nil` only when both halves are nil. Storing a nil pointer of a concrete type
in an interface fills in the type half, so the interface is no longer nil.

**Line 1: `false`.** `mayFail(false)` returns `e`, a `*opErr` that is nil. The
`return` converts it to `error`, producing the pair `(*opErr, nil)`. Comparing
that to `nil` asks "are both halves nil", and the type half is not.
*Common wrong prediction:* `true`, because "`e` is nil and I returned `e`". The
misconception is that conversion to an interface is a no-op; it is the moment
the type gets recorded. This is the classic Go bug: a function declared to
return `error` that returns a typed nil pointer makes every `if err != nil`
check fire on success.

**Line 2: `true`.** `mayFailFixed` returns the untyped `nil` literal directly as
an `error`, which is the pair `(nil, nil)`. The fix is always the same: return
`nil` explicitly on the success path, never a typed variable that happens to
be nil.

**Line 3: `*main.opErr <nil>`.** `%T` prints the dynamic type, which is
`*main.opErr`, confirming the pair from line 1. `%v` on an error would call
`Error()`, which dereferences `e.code` on a nil pointer and panics. The `fmt`
package catches panics from formatting methods, and when the receiver is a nil
pointer it prints `<nil>` instead of a panic message.
*Common wrong prediction:* `<nil> <nil>` (believing the interface is nil), or a
crash (not knowing `fmt` guards nil receivers).

**Line 4: `true`.** A declared but unassigned `any` is the zero interface value
`(nil, nil)`.

**Line 5: `false true`.** Assigning the nil `*opErr` to `anything` records the
type, so `anything == nil` is false, while `p == nil` compares a plain pointer
and is true. Same bits in the value half, different answers, which is the
whole lesson in one line.

**Line 6: `0 0 true 1`.** A nil map is a valid, empty, read-only map: indexing a
missing key returns the zero value `0`, and `len` is 0. (Writing to it would
panic.) A nil slice is a valid empty slice: `s == nil` is true, and `append`
allocates a backing array, so the result has length 1.
*Common wrong prediction:* a panic on `m["missing"]`, from languages where a
null collection cannot be read. In Go, reading nil maps and appending to nil
slices are deliberately safe.

**Line 7: `recovered: true` (and no `7a.` line).** Calling a method on a nil
pointer is legal: the method receives a nil receiver, and could check for it.
`Error()` does not; it reads `e.code`, which is a nil pointer dereference and a
runtime panic. Go evaluates all arguments before calling `fmt.Println`, so the
panic happens while computing `e2.Error()`, before anything is printed: `7a.`
never appears. The panic unwinds to `main`'s deferred function, where
`recover()` returns the runtime error (non-nil), and it prints `true`.
*Common wrong predictions:* `7a. op failed: 0` (assuming a nil pointer behaves
like a zero struct), or a panic at the call itself (assuming method calls on
nil are illegal), or `7a.` printed before the panic (forgetting arguments are
evaluated first).

</details>

## 03-slice-aliasing

Source: [`predict/03-slice-aliasing/main.go`](predict/03-slice-aliasing/main.go)

When does `append` write into memory someone else can see, and when does it
quietly move to a new array?

<details><summary>Output and explanation</summary>

```text
1. 2 4
2. [1 2 3 99 5]
3. [2 3 99]
4. [1 2 3 99 5]
5. [2 3 100]
6. [1 2 3] [-1]
7. 2 [7 8]
8. [42 2 3]
```

The rule: a slice is a three-word header **(pointer, length, capacity)** that
views a backing array. `append(s, x)` checks capacity: if `len < cap`, it
writes `x` into the existing array at index `len` and returns a header with
`len+1`; if not, it allocates a new, larger array, copies, and returns a header
pointing there. Whether a write is shared depends only on that arithmetic.

**Line 1: `2 4`.** `b := a[1:3]` views elements 1 and 2, so `len = 3 - 1 = 2`.
Capacity runs from the start of the view to the end of the backing array:
`cap = 5 - 1 = 4`.
*Common wrong prediction:* `2 2`, from thinking a sub-slice is a copy of just
those elements. It is a window into the same array, and the window can extend.

**Line 2: `[1 2 3 99 5]`.** `b` has spare capacity (len 2, cap 4), so `append`
writes 99 into the backing array at `b`'s index 2, which is `a[3]`. The 4 in
`a` is overwritten.
*Common wrong prediction:* `[1 2 3 4 5]`, believing `append` never touches
memory outside the slice it was given.

**Line 3: `[2 3 99]`.** The new header for `b` has length 3 over the same array.

**Line 4: `[1 2 3 99 5]`.** The full slice expression `a[1:3:3]` sets capacity to
`3 - 1 = 2`, equal to the length. `append` has no room, so it allocates a fresh
array; `a` is untouched (still showing the 99 from line 2).
*Common wrong prediction:* `[1 2 3 100 5]`, by reusing line 2's reasoning
without noticing the third index. Capping capacity is how you hand out a slice
that cannot clobber the caller.

**Line 5: `[2 3 100]`.** `c` now owns a new array containing the copied 2, 3
and the appended 100.

**Line 6: `[1 2 3] [-1]`.** `d` has length 2 and capacity 2. `e := d[:1]`
shares `d`'s array. `append(d, 3)` exceeds capacity, so `d` moves to a new
array; `e` still points at the old one. Writing `e[0] = -1` changes only the
old array, which `d` no longer sees.
*Common wrong prediction:* `[-1 2 3] [-1]`, assuming `d` and `e` are still
aliases. The alias broke silently at the reallocation, which is why code that
keeps sub-slices across `append` is fragile.

**Line 7: `2 [7 8]`.** `copy` copies `min(len(dst), len(src))` elements and
returns that count. It never grows `dst`.
*Common wrong prediction:* `3 [7 8 9]`, treating `copy` like an assignment.

**Line 8: `[42 2 3]`.** Slices are passed by value, but the value is the header,
so the callee's header points at the same array. `mutate` writes `xs[0] = 42`
into that shared array, which the caller sees. `grow` appends to a slice with
`len 3, cap 3`: no room, so `append` allocates a new array and `xs` (the local
header) now points there. `xs[0] = 1000` writes to the new array. Even without
the reallocation, the caller's header would still say length 3, so the
appended 4 could never show up in `f`.
*Common wrong prediction:* `[1000 2 3 4]` or `[1000 2 3]`. The misconception is
"slices are references". They are values that contain a pointer: element
writes are shared, but length changes and reallocations are not.

</details>

## 04-defer-order

Source: [`predict/04-defer-order/main.go`](predict/04-defer-order/main.go)

In what order do deferred calls run, when are their arguments computed, and
can they change what a function returns?

<details><summary>Output and explanation</summary>

```text
1. 10
2. 5
3.
  body sees i = 100
  deferred saw i = 1
6. main body done
5. after
4. 2 1 0
```

Three rules decide everything here:

- Deferred calls run when the surrounding function returns, **last in, first
  out**.
- A deferred call's **function value and arguments are evaluated at the
  `defer` statement**; only the call itself waits.
- `return expr` first assigns `expr` to the result parameters, then runs the
  deferred calls, then returns. A deferred closure can change the result only if
  it can name the result parameter.

**Line 1: `10`.** `namedReturn` has a named result `n`. `return n` assigns 5 to
`n` (a no-op here), then the deferred closure runs `n *= 2` on that same
variable, and the function returns 10.
*Common wrong prediction:* `5`, assuming `return` freezes the value before
deferred code runs. With a named result, the deferred closure is editing the
return slot itself; this is how `defer` is used to wrap or set errors.

**Line 2: `5`.** `plainReturn` has an unnamed result. `return n` copies 5 into
the hidden result slot. The deferred closure doubles the *local* `n` to 10, but
nothing reads it again.
*Common wrong prediction:* `10`, by applying line 1's reasoning. The difference
is purely whether the closure can reach the result variable.

**Line 3 block.** `3.` prints, then `argsEvaluatedWhen` runs. `defer
fmt.Println("  deferred saw i =", i)` evaluates `i` right there, capturing 1.
The body sets `i = 100` and prints `body sees i = 100`. On return, the deferred
call prints the 1 it already captured.
*Common wrong prediction:* `deferred saw i = 100`, from thinking `defer` delays
the whole expression. Only the call is delayed; the arguments are not.

**Line 6: `6. main body done`.** Everything after `argsEvaluatedWhen()` in
`main` is deferred except this `Println`, so it is the next thing printed, and
it comes before 5 and 4 even though it is numbered after them.

**Line 5: `5. after`.** The last `defer` registered in `main` is a closure, so
it runs first. A closure's body is evaluated when it runs, and by then `msg`
has been reassigned to `"after"`.
*Common wrong prediction:* `5. before`, by applying the argument rule to a
closure. The closure takes no arguments; it reads the variable `msg` at call
time.

**Line 4: `4. 2 1 0`.** The registration order was `Println()`, then
`Print(" ", 0)`, `Print(" ", 1)`, `Print(" ", 2)`, then `Print("4.")`, then the
closure. Unwinding in reverse after the closure gives `4.`, ` 2`, ` 1`, ` 0`,
then the newline from the first-registered `Println()`. The values 0, 1, 2 were
fixed as arguments at each `defer`.
*Common wrong prediction:* `4. 0 1 2`, from reading the loop top to bottom and
forgetting LIFO; or the `4.` appearing at the end of the line, from not
noticing it was deferred after the loop.

</details>

## 05-strings-and-runes

Source: [`predict/05-strings-and-runes/main.go`](predict/05-strings-and-runes/main.go)

What are `len`, indexing, ranging, and slicing actually counting in a Go
string?

<details><summary>Output and explanation</summary>

```text
1. 6 5
2. 195 104
3. 0:h 1:é 3:l 4:l 5:o
4. 5 6
5. "h\xc3" "hé"
6. HÉLLO 6
7. [a b c]
8. 0 false 3
9. 0 2
10. héllo héllo!
```

The rule: a Go `string` is an immutable sequence of **bytes**, conventionally
UTF-8. `len` and indexing work on bytes; `for range` decodes UTF-8 and yields
**runes** (Unicode code points) with their byte offsets. `é` (U+00E9) encodes
as two bytes, `0xC3 0xA9`, so `"héllo"` is 6 bytes and 5 runes.

**Line 1: `6 5`.** `len(s)` counts bytes (6); `utf8.RuneCountInString` counts
code points (5).
*Common wrong prediction:* `5 5`, from treating `len` as "number of
characters".

**Line 2: `195 104`.** `s[1]` is the byte `0xC3` = 195, the first half of `é`;
`s[0]` is `'h'` = 104. Indexing a string yields a `byte` (`uint8`), and
`Println` prints it as a number.
*Common wrong prediction:* `é h` (expecting characters), or `233` (the code
point of `é`, expecting a rune).

**Line 3: `0:h 1:é 3:l 4:l 5:o`.** `range` over a string yields the **byte
offset** of each rune and the rune itself. `é` starts at byte 1 and occupies
bytes 1 and 2, so the next rune is at offset 3: offset 2 never appears.
*Common wrong prediction:* `0:h 1:é 2:l 3:l 4:o`, assuming the index counts
runes.

**Line 4: `5 6`.** Converting to `[]rune` decodes into 5 code points; converting
to `[]byte` copies the 6 bytes.

**Line 5: `"h\xc3" "hé"`.** Slicing is by byte offset and never checks rune
boundaries. `s[0:2]` takes `h` and half of `é`, which is invalid UTF-8; `%q`
escapes the lone byte as `\xc3`. `s[0:3]` ends on a boundary and is `"hé"`.
*Common wrong prediction:* `"hé" "hél"`, slicing by characters. Byte slicing
that cuts a rune is where mojibake comes from.

**Line 6: `HÉLLO 6`.** `strings.ToUpper` maps rune by rune. `É` (U+00C9) is also
two bytes in UTF-8, so the length stays 6. That is a coincidence of this
input: case mapping can change the byte length in general (for example, some
characters map to longer sequences), so never assume it preserves offsets.

**Line 7: `[a b c]`.** Map iteration order is deliberately randomised, so the
keys are collected and sorted before printing. Without `sort.Strings` the
order would vary between runs.

**Line 8: `0 false 3`.** Reading a missing key returns the value type's zero
(0); the comma-ok form reports `false`. Reading does not insert, so the length
stays 3.
*Common wrong prediction:* a panic or `len` of 4, from languages where a
missing key throws or a lookup creates the entry.

**Line 9: `0 2`.** After `delete`, reading `"a"` again gives the zero value,
and one entry is gone.

**Line 10: `héllo héllo!`.** Strings are immutable. `t := s` copies the string
header (pointer and length); `t += "!"` allocates a new string and points `t`
at it. `s` still refers to the original bytes.
*Common wrong prediction:* `héllo! héllo!`, treating `t` as an alias that
mutates shared storage.

</details>

## 06-channels-and-close

Source: [`predict/06-channels-and-close/main.go`](predict/06-channels-and-close/main.go)

What happens when you receive from a closed channel, an empty one, or a nil
one?

<details><summary>Output and explanation</summary>

```text
1. 2 3
2. 1 true
3. 2 true
4. 0 false
5. a b
6. nothing ready
7. nil channel is never ready
8. 5
```

The rules: closing a channel forbids further sends but keeps buffered values,
which receivers still get in FIFO order. Once the buffer is empty, a receive on
a closed channel **never blocks**: it returns the element type's zero value and
`ok == false`. A receive on a nil channel blocks forever. A `select` with a
`default` case never blocks: if no other case is ready, `default` runs.

**Line 1: `2 3`.** `len` of a buffered channel is the number of queued
elements (2); `cap` is the buffer size (3).

**Lines 2 and 3: `1 true`, `2 true`.** After `close`, the two buffered values
are still delivered, in order, with `ok == true`.
*Common wrong prediction:* `0 false` straight away, believing `close` discards
the buffer. It does not; it only marks the channel so that, after draining,
receivers see the end.

**Line 4: `0 false`.** The buffer is empty and the channel is closed, so the
receive returns immediately with the zero value of `int` and `ok == false`.
*Common wrong prediction:* a deadlock (thinking an empty channel always blocks)
or a panic (thinking receiving from a closed channel is an error). Only
*sending* on a closed channel panics. Never blocking on receive is exactly what
makes `close` a broadcast "we are done" signal to any number of receivers.

**Line 5: `a b`.** `for range ch2` receives until the channel is closed and
drained, then exits the loop.
*Common wrong prediction:* a deadlock after `b`. That would happen if `ch2`
were not closed: `range` would wait forever for a third value.

**Line 6: `nothing ready`.** `empty` is unbuffered with no sender, so the
receive case cannot proceed, and `select` takes `default`.

**Line 7: `nil channel is never ready`.** Operations on a nil channel block
forever, so its `select` case is never ready and `default` runs. This is a
feature: setting a channel variable to `nil` is how you disable one case of a
`select` loop.
*Common wrong prediction:* a panic, by analogy with nil maps or pointers.

**Line 8: `5`.** The worker sends `0*0`, `1*1`, `2*2` into a channel of capacity
3, so it never blocks and finishes; `wg.Wait()` returns, the channel is closed,
and the `range` sums `0 + 1 + 4 = 5`.
*Common wrong prediction:* a deadlock. It would deadlock if `results` had
capacity less than 3, because the worker would block on a send that nobody
receives until after `wg.Wait()`. The ordering is safe here only because the
buffer is large enough.

</details>
