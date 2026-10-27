# Dispatch, variance, and type expressiveness

This part of [Type Systems and Polymorphism](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Distinguish bounded dispatch from subtyping and derive variance.

## Ad-hoc polymorphism: typeclasses, traits, and interfaces

Parametric code never needs to know which type it is running on. Ad-hoc code does: when you call `area(shape)`, something has to find the *right* body for that shape's type. Languages answer this dispatch problem in different ways, and the answer defines much of a language's flavor.

**Typeclasses** in Haskell use Wadler and Blott's *dictionary passing*. A declaration `class Shape a where area :: a -> Double` desugars into an extra hidden argument: a record of method implementations, the *dictionary*, which the compiler infers and threads through. A constraint `Shape a => ...` says "I need this dictionary." Resolution happens entirely at compile time, by type.

**Traits** in Rust are typeclasses with a performance contract. `impl Shape for Circle` is nominal and explicit. A bound `<T: Shape>` is monomorphized, so there is static dispatch and no dictionary at runtime, while `dyn Shape` uses a vtable for dynamic dispatch. [`code/traits.rs`](code/traits.rs) shows both: compare `total_area<T: Shape>` (one copy per type) with `describe_all(&[Box<dyn Shape>])` (one copy, vtable dispatch). It is the same trait with two runtime models.

**Interfaces** in Go are *structural*: a type satisfies `Shape` simply by having an `Area() float64` method, with no declaration. Go generics add **type sets**, where a constraint interface can enumerate a *union of concrete types* (`~int | ~float64`), so `Sum[T Numeric]` can use `+` without method dispatch at all. That type-set design is unique to Go; see [`code/generics.go`](code/generics.go).

Behind all of these sits the question of **coherence**. Typeclasses enforce *one* instance per (class, type) pair globally, so `Set`-like code can never see two different `Ord` instances for the same type; Rust's orphan rule addresses the same concern. Go and Java interfaces make no such guarantee, which is more flexible and less safe.

## Subtyping, variance, and structural versus nominal typing

Subtyping raises a question that parametric polymorphism alone never does: when does `Container<Sub>` fit where a `Container<Super>` is expected? The rules that decide it are the ones programmers most often get wrong.

**Subtyping** $S <: T$ means "an $S$ is usable wherever a $T$ is expected." Records and objects subtype by **width** (having more fields makes a subtype) and by **depth** (fields subtype pointwise). Function types have their own rule, and it is the famous one.

**Variance** answers this: given `Dog <: Animal`, how are `F<Dog>` and `F<Animal>` related? There are three possibilities. A **covariant** constructor preserves the direction: a `Producer<Dog>` *is* a `Producer<Animal>`, because it only outputs values. Immutable containers and return types are covariant. A **contravariant** constructor reverses the direction: a `Consumer<Animal>` *is* a `Consumer<Dog>`, because it only takes inputs, and anything that can handle any animal can handle a dog. Function parameters are contravariant. An **invariant** constructor relates in neither direction: a mutable `Array<Dog>` is *not* an `Array<Animal>`, because through the `Animal` view you could store a `Cat`. Mutability forces invariance.

Putting the pieces together gives the function rule:

$$(A \to B) <: (A' \to B') \iff A' <: A \ \text{and}\ B <: B'$$

The argument is contravariant and the result is covariant. This one counterintuitive law explains Java's broken array covariance, C#'s `in`/`out` annotations, and TypeScript's `strictFunctionTypes`. [`code/polymorphism.ts`](code/polymorphism.ts) demonstrates and defends it. To feel the rule, uncomment the `const bad:` line in that file and watch the compiler reject assigning a `Consumer<Dog>` where a `Consumer<Animal>` is wanted. That single error is the contravariance rule defending you.

Languages also differ in *where* you state variance. **Declaration-site** variance (C# `in`/`out`, Scala `+T`/`-T`, Kotlin) annotates the type definition once. **Use-site** variance (Java's `? extends` and `? super`) annotates each use.

> **Warning:** Some languages deliberately accept soundness holes here. Java and C# **arrays are covariant but mutable**, so `Object[] a = new String[1]; a[0] = 1;` compiles and then throws `ArrayStoreException` *at runtime*. TypeScript's `any` and its bivariant method parameters are similar pragmatic compromises.

The last axis is how types are matched at all. **Structural** systems (TypeScript, Go) match types by **shape**; **nominal** systems (Rust, Haskell, Java, Scala) match by **declared name**. This is a design axis, not a quality axis. Structural typing enables gradual adoption and "just have the right fields." Nominal typing makes intent explicit and prevents *accidental* compatibility: two unrelated types `Meters` and `Feet` with the same shape stay distinct.

## The frontier: higher-kinded, associated, and dependent types

Type systems keep climbing past what most mainstream languages offer, and each rung buys more expressiveness at some cost to inference.

**Higher-kinded types (HKTs)** abstract over type *constructors*, not just types. `Functor f` works for `f = Maybe`, `f = []`, or `f = Tree`: anything of kind `* → *`. Haskell and Scala have them natively ([`code/typeclasses.hs`](code/typeclasses.hs) defines `Functor Tree`), Rust approximates them with generic associated types (GATs), and Go and Java cannot express them.

**Associated types** (Rust's `type Item;`, Haskell's type families) let a trait fix *one* related type per implementation, instead of letting the caller choose a parameter. `Iterator::Item` is the canonical example; see [`code/traits.rs`](code/traits.rs).

**GADTs** (generalized algebraic data types) let each constructor refine the result type. That lets you encode invariants such as a *well-typed* AST that cannot even represent `if 3 then …`.

**Dependent types** (Agda, Idris, Lean) let types mention *values*, as in `Vec n a`, a list whose length is part of its type. They collapse the Curry–Howard correspondence into a single language in which you write programs and prove theorems with the same syntax. This is the top of the ladder, and the price is that inference needs help and type checking can require proof search.

## Summary

Distinguish bounded dispatch from subtyping and derive variance. State the assumptions before applying the result to a different workload.

## Review questions

1. Why is a mutable container usually invariant?

   <details><summary>Answer</summary>

   Reading and writing place its element type in both positive and negative positions, so unrestricted covariance permits invalid writes.

   </details>

[Chapter contents](README.md) · [Next: Generics at runtime and language comparison](04-generics-runtime-and-language-comparison.md)
