# Algorithms

This book is for programmers who can already write working code and now want to choose an algorithm on purpose: to recognize the pattern under a problem and explain why the fast approach is fast. Each chapter opens with one pattern family's core insight, then works classic interview problems with runnable solutions beside the text. The last chapter is different in kind: it builds, in C, Rust, and Go, the data structures a production LLM server depends on.

## What you will be able to do

- Replace a nested loop with a hash-map lookup, a two-pointer sweep, or a sliding window.
- Phrase a search as a binary search over a monotone predicate, including a search over possible answers.
- Traverse, order, and find shortest paths in graphs with BFS, DFS, topological sort, and Dijkstra's algorithm.
- Set up a dynamic program by naming its state and transition, and prove a greedy choice safe with an exchange argument.
- Build and size the hash maps, heaps, radix trees, and probabilistic filters that real systems run on.

## Ways through this book

Chapters 1 to 12 take about seventeen days at two to three hours a day; Chapter 13 is read alongside another book. Give each problem a genuine twenty-minute attempt before you open its solution, and re-solve one earlier problem each day.

- **Preparing for coding interviews** (about three weeks): Chapters 1 to 11 in order, then [Chapter 12: Probabilistic data structures](12-probabilistic-structures/README.md), then two or three company guides from [*Interview Preparation*](../interview-preparation/README.md). Finish with one timed problem from each of Chapters 1 to 11. Keep the order: later patterns reuse earlier ones.
- **Getting comfortable with dynamic programming** (about a week): [Chapter 11: Recursion and divide-and-conquer](11-recursion-divide-conquer/README.md), then [Chapter 7: Dynamic programming](07-dynamic-programming/README.md), then [Chapter 8: Greedy algorithms](08-greedy/README.md), which makes more sense after a DP that greedy would get wrong.
- **Building the LLM system** (as you go): read [Chapter 1](01-arrays-hashing/README.md) and [Chapter 12](12-probabilistic-structures/README.md) first, then each section of [Chapter 13: Systems data structures](13-systems-data-structures/README.md), from [13.1 Growable array](13-systems-data-structures/01-growable-array.md) to [13.9 Consistent hash ring](13-systems-data-structures/09-consistent-hash-ring.md), just before the part of [*Build Your Own LLM System*](../build-an-llm/README.md) that first uses it.

## Before you start

You should be comfortable writing loops, functions, and recursion in one language and able to read JavaScript and Python; a few solutions are in C, TypeScript, and Java. The JavaScript files need a small driver to run in Node. Several chapters lean on discrete mathematics and probability from [*Mathematics for Computing*](../mathematics-for-computing/README.md), and Chapter 13 expects the [C](../programming-languages/02-language-and-tool-primers/03-c.md), [Rust](../programming-languages/02-language-and-tool-primers/04-rust.md), and [Go](../programming-languages/02-language-and-tool-primers/06-go.md) primers from *Programming Languages*. To go further, [*Competitive Programming*](../competitive-programming/README.md) covers contest techniques, and the repository's [practice directory](../../../practice/) has standalone exercises. New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.

## Further reading

Chapter openings cite the relevant CLRS chapters where they apply.

- Cormen, Leiserson, Rivest, and Stein, *Introduction to Algorithms* (CLRS), 4th ed., MIT Press. The comprehensive reference, with formal proofs.
- Antti Laaksonen, *Competitive Programmer's Handbook* ([free PDF](https://cses.fi/book/book.pdf)). Concise implementations and advanced topics such as segment trees.
- Steven Skiena, *The Algorithm Design Manual*. Practical advice on choosing an algorithm.
- *Codeless Data Structures and Algorithms*. Visual intuition without code.
