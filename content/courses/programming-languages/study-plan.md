# Study plan

This plan is a suggested pace for reading the book over two to three weeks. It follows the order that makes the ideas compound: the mathematics of types first, then the same concept seen through five type systems, then optional depth for readers who want the frontier. The [Language and Tool Primers](02-language-and-tool-primers/README.md) chapter is not scheduled here because it is best read just in time, one primer immediately before you first need that language or tool.

Throughout, keep a terminal open and run the programs in [`01-type-systems/code/`](01-type-systems/code/) as you read about them. Seeing a type checker accept or reject a term is worth more than any amount of reading about it.

## Week one: the mathematics of types

Begin with what a type system is in [Type Systems and Polymorphism](01-type-systems/README.md): typing judgments of the form $\Gamma \vdash e : \tau$, soundness as progress plus preservation, and the Curry-Howard correspondence between types and propositions. Chapters 1 to 9 of Pierce's *Types and Programming Languages* (TAPL), or the early volumes of *Software Foundations*, are good companions for this first day.

Next, spend a day on the taxonomy of polymorphism that goes back to Strachey and to Cardelli and Wegner: parametric, ad-hoc, subtype, and bounded. The Cardelli-Wegner paper, "On Understanding Types, Data Abstraction, and Polymorphism," is the classic source and still readable. Then climb the lambda-calculus ladder from the simply typed lambda calculus to System F, where parametric polymorphism and parametricity live; TAPL chapters 22 and 23 cover this ground.

Give the last two days of the week to Hindley-Milner and Algorithm W: unification, the occurs check, `let`-generalization, and principal types. Read [`hindley_milner.py`](01-type-systems/code/hindley_milner.py) alongside the text, run it, and then read the Damas-Milner paper with the code in front of you.

## Week two: one concept, five mechanisms

The second week takes the running example through five languages, one a day. Start with TypeScript in [`polymorphism.ts`](01-type-systems/code/polymorphism.ts): structural typing, variance, and type-level programming, with the TypeScript Handbook as reference. Move to Go generics in [`generics.go`](01-type-systems/code/generics.go), where constraints are type sets and unions and the compiler uses GC-shape stenciling; the official Go generics tutorial is enough background.

Then read Rust traits in [`traits.rs`](01-type-systems/code/traits.rs), paying attention to the split between static, monomorphized dispatch and dynamic `dyn` dispatch, and to associated types; chapters 10 and 17 of *The Rust Programming Language* cover the same ground. Follow it with Haskell typeclasses as dictionary passing and the higher-kinded `Functor` in [`typeclasses.hs`](01-type-systems/code/typeclasses.hs), with the Wadler-Blott paper on typeclasses as the source.

Finish the week with OCaml in [`inference.ml`](01-type-systems/code/inference.ml): Hindley-Milner inference from the user's side, plus modules and functors. Compare the principal types OCaml reports with the ones you derived by hand in week one; *Real World OCaml* is the companion text.

## Week three (optional): depth

If you want to go further, spend two days on subtyping and variance in depth, including the difference between declaration-site and use-site variance and the array-covariance soundness hole that Java and C# accept; TAPL chapters 15 and 16 are the reference. Spend a day on how generics compile, comparing monomorphization, erasure, and dictionary passing as a trade-off between code size, run time, and compile time.

Close with the frontier: higher-kinded types, associated types, GADTs, and dependent types at the top of the lambda cube. *Programming Language Foundations in Agda* (PLFA) is a free and gentle way in.
