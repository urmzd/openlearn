# Type Systems and Polymorphism

Every working programmer has fought a type checker, and most have wondered whether the fight was worth it. This chapter answers that question by treating a type system for what it is: a lightweight formal method, a decidable and purely syntactic proof system that rejects a whole class of programs *before they run*. Its correctness guarantee has a precise name, **soundness**, and a precise statement: soundness is **progress plus preservation**, or in the slogan, "well-typed programs don't get stuck." Nearly everything else in the area, from generics to variance annotations, is engineering built around that one theorem.

The second thread of the chapter is polymorphism, the machinery that buys back expressiveness without giving up the guarantee. "Polymorphism" turns out to be one word for four different mechanisms, and much of the confusion in "generics versus interfaces" arguments dissolves once you can name which one you mean. You will follow the classic lambda-calculus ladder from the simply typed calculus to System F, then see why full polymorphic inference is undecidable and how Hindley–Milner carves out the decidable slice that OCaml, Haskell, Elm, and the local inference in Rust, Swift, and TypeScript all descend from. From there the chapter turns practical: how ad-hoc polymorphism dispatches in Haskell, Rust, and Go; why `List<Dog>` is not automatically a `List<Animal>`; and why the same generic function has three different runtime stories depending on how a language compiles it.

The chapter is anchored by runnable code. [`code/hindley_milner.py`](code/hindley_milner.py) is the whole of Algorithm W in about 250 lines, and five companion programs in OCaml, TypeScript, Go, Rust, and Haskell walk the same running example (a `Shape` with `area`, a generic `Stack<T>`, and a `largest`/`Max` over ordered things) through type systems that make different choices. Running them side by side is the clearest way to see that the *concept* stays fixed while the *mechanism* changes.

> **Learning objectives**
> - Read typing judgments and inference rules, and state soundness as progress plus preservation.
> - Classify a piece of polymorphic code as parametric, ad-hoc, subtype, coercion, or bounded, and say which mechanism a given language feature uses.
> - Explain how Hindley–Milner inference works (unification, the occurs check, instantiation, and `let`-generalization) and trace it on small terms.
> - Derive the variance of a type constructor, including the contravariant rule for function arguments, and explain the soundness holes that real languages accept.
> - Predict the runtime cost of generics in a language from whether it monomorphizes, erases, or passes dictionaries.

## Before you begin

The chapter assumes the proof techniques from [Discrete Math 1](../../mathematics/11-discrete-math-1/) in *Mathematics*: proofs, induction, and relations (the subtype relation `<:` is a partial order). Comfort reading inference rules helps, but the notation is taught here. You will get the most out of the code if you have already written some Python, C, Rust, and Go, so the five-language comparison lands on familiar ground.

## In this chapter

1. [Soundness and polymorphism](01-soundness-and-polymorphism.md): Relate progress and preservation to kinds of polymorphism.
2. [System F and type inference](02-system-f-and-type-inference.md): Trace generalization, instantiation, and the occurs check.
3. [Dispatch, variance, and type expressiveness](03-dispatch-variance-and-type-expressiveness.md): Distinguish bounded dispatch from subtyping and derive variance.
4. [Generics at runtime and language comparison](04-generics-runtime-and-language-comparison.md): Connect monomorphization, erasure, and dictionary passing to runtime behavior.

## Companion code

- [`code/hindley_milner.py`](code/hindley_milner.py): Algorithm W end to end. It infers principal types for `\x.x`, `\f.\g.\x. f (g x)`, and `let id = \x.x in id id` with no annotations, then shows the occurs check rejecting `\x. x x`.
- [`code/inference.ml`](code/inference.ml): OCaml inferring the exact same principal types from the user's side.
- [`code/polymorphism.ts`](code/polymorphism.ts): structural, gradual typing with type-level computation (`DeepReadonly`) and explicit variance.
- [`code/generics.go`](code/generics.go): Go's type-set constraint model and GC-shape stenciling.
- [`code/traits.rs`](code/traits.rs): nominal traits and the split between static (`<T: Trait>`, monomorphized) and dynamic (`dyn Trait`, vtable) dispatch.
- [`code/typeclasses.hs`](code/typeclasses.hs): typeclasses as dictionary passing, plus the higher-kinded `Functor`.

A good way to work is to run the keystone first: read `unify`, `instantiate`, and `generalize` in [`hindley_milner.py`](code/hindley_milner.py), because that triple *is* ML type inference, and then open [`inference.ml`](code/inference.ml) to watch a production compiler reach the same answers. After that, run the four language files side by side. All of them print the same shape of output (`area`s, a `Stack`, a `largest`).
