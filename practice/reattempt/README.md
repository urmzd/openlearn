# Reattempt

Code you already wrote, handed back with the solution removed and the tests left
in place. This is the least comfortable of the three exercise kinds and the most
informative, because it separates two things that feel identical from the inside:
having learned something, and having finished it.

```bash
practice/bin/ol list  reattempt case-studies
practice/bin/ol start reattempt case-studies 02   # stripped, into .scratchpad/
practice/bin/ol check reattempt case-studies 02   # the original tests, verbatim
practice/bin/ol diff  reattempt case-studies 02   # against what you wrote before
```

## Nothing is duplicated

There is exactly one copy of every solution and it is the one the learning path
already links to. [`manifest.tsv`](manifest.tsv) points at files in place;
`ol start` reads the original, strips the region between its solution markers,
and writes the result into `.scratchpad/`. This directory holds no code at all.

That is why the sources carry a `SOLUTION-BEGIN` comment: it is the boundary
between the part you are asked to rebuild and the part that judges whether you
did. It changes nothing about how the file runs, and CI runs the originals
unchanged.

## What qualifies as a source

**It has to assert something.** That is the whole bar, and most code fails it.

A file that defines a function and never calls it exits zero whether the
implementation is correct, wrong, or missing entirely. Strip the solution out of
one and `ol check` reports PASS on an empty stub, which is worse than having no
exercise: it teaches you that you still know something you may well have
forgotten.

| Source | Status | Why |
|--------|--------|-----|
| [`case-studies/`](../../case-studies/) | 4 exercises, ready | Already self-testing with real assertions, and CI already depended on it |
| [`algorithms/`](../../algorithms/) | Not yet listed | LeetCode-shaped: a function definition and no assertions, so there is nothing to check against |

## Promoting an algorithms solution

The `algorithms/` tree is 60-odd committed solutions with no tests. Each one can
become a reattempt exercise, but the assertions have to come first, and that is
worth doing on its own terms: it moves a file from "parses" to "verified", which
is the more useful of the two states by a wide margin.

For one file:

1. Add a `if __name__ == "__main__":` block (or the language's equivalent) with
   assertions covering the normal case, the empty input, and whichever edge case
   the problem is actually about. It must exit nonzero when the answer is wrong.
2. Confirm it fails when it should: break the implementation deliberately and
   check that running it now exits nonzero. An assertion you have never seen fail
   is not evidence of anything.
3. Wrap the solution body in the markers, leaving the signature and the tests
   outside them.
4. Add a row to [`manifest.tsv`](manifest.tsv).
5. `ol verify reattempt algorithms` must report `OK` for it.

The runners `ol` already knows are `.py`, `.js`, `.ts`, and `.c`, which covers
most of the tree. Scheme, R, Prolog, and Perl files would each need a line adding
to `run_reattempt_file` in [`../bin/ol`](../bin/ol).

## Why this beats redoing it from scratch

Rewriting a solution in a fresh file lets you drift toward whatever you remember
best, and you grade yourself. Reattempting against the original's own tests
removes both escapes: the contract is fixed, the verdict is an exit code, and
`ol diff` shows you exactly where this attempt and your previous one disagree.
Where they disagree is usually the interesting part, and it is not always the new
one that is worse.
