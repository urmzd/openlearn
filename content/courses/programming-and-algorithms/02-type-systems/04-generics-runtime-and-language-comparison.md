# Generics at runtime and language comparison

This part of [Type Systems and Polymorphism](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Connect monomorphization, erasure, and dictionary passing to runtime behavior.

## How generics compile

Everything so far has been about what a type system accepts. The final question is what a generic function costs at runtime, and the answer depends on *how* the language compiles it. The same feature has three different runtime stories.

**Monomorphization** (Rust, C++ templates, and Go for non-pointer types) emits a specialized copy of the function for each concrete type. It is *fast*: calls are direct, inlinable, and unboxed. The cost is *code bloat* and slower compiles. C++ templates are monomorphization with duck typing, which is also why their error messages are notoriously long.

**Type erasure** (Java, and Scala on the JVM) keeps one copy. Type parameters vanish at runtime, so `List<String>` and `List<Integer>` are the same class, and values are **boxed**. The result is *small* and binary-compatible, but you lose types at runtime (`new T[]` is illegal and reflection cannot see type arguments) and you pay for boxing.

**Dictionary passing and GC-shape stenciling** (Haskell, and Go for pointer-shaped types) keep one copy parameterized by a runtime record of operations, the dictionary, or share one copy across all types with the same memory layout. This is the middle path: modest code size with some indirection.

The lesson for engineers is that "does this language have generics?" is the wrong question. "*How* are they implemented?" predicts your binary size, startup time, inlining behavior, and whether you can introspect types at runtime. [`code/generics.go`](code/generics.go) and [`code/traits.rs`](code/traits.rs) make the static-versus-dynamic choice explicit.

## Comparing mechanisms and languages

Two tables pull the chapter together. The first lines up each polymorphism mechanism against its theory and its spelling in four languages:

| Mechanism | One body or many? | Dispatch | Theory | TS | Go | Rust | Haskell/OCaml |
|-----------|-------------------|----------|--------|----|----|------|---------------|
| **Parametric** | one | none (uniform) | System F / HM | `<T>` generics | `[T any]` | `<T>` | `'a` / type vars |
| **Ad-hoc (overload)** | many | by static type | — | overloads | — (no overloading) | — | class methods |
| **Ad-hoc (class/trait)** | many | compile-time, by type | qualified types | structural iface | method-set iface + type sets | `trait` (static) | typeclass (dict) |
| **Subtype** | one (+ overrides) | runtime (vtable) | F-sub / `S<:T` | structural `<:` | `interface` value | `dyn Trait` | subtyping (limited) / variants |
| **Bounded** | one | compile-time | System F-sub | `<T extends C>` | `[T Constraint]` | `<T: Trait>` | `(C a) =>` / module sig |
| **Coercion** | n/a | implicit | — | limited | explicit only | explicit (`as`/`From`) | explicit |

The second summarizes the design choices each major language makes along the axes this chapter has covered:

| Language | Nominal / Structural | Inference | Generics compile to | Variance | Notable |
|----------|----------------------|-----------|---------------------|----------|---------|
| **TypeScript** | structural | local + `infer` (HM-ish) | erased to JS (none at runtime) | covariant by default, `strictFunctionTypes` for params | type-level computation, `any` escape hatch (unsound) |
| **Go** | structural interfaces | local only | GC-shape stenciling + dictionaries | invariant | type sets / unions, no HKT, no overloading |
| **Rust** | nominal traits | local (HM-style) | monomorphization (+ `dyn` vtables) | declared via lifetimes/variance | ownership in the types, associated types, GATs |
| **Haskell** | nominal | full HM + extensions | dictionary passing | role-based | typeclasses, HKT, GADTs, laziness |
| **OCaml** | nominal + structural objects | full HM | mostly uniform (boxed) + flambda | declared (`+`/`-`) | modules & functors, polymorphic variants |
| **Java/C#** | nominal | local (`var`/`<>`) | erasure (Java) / reified (C#) | use-site (Java) / declaration-site (C#) | array covariance hole, bounded wildcards |
| **Scala** | nominal | local + implicits | erasure (JVM) | declaration-site `+T`/`-T` | HKT, implicits/givens, path-dependent types |

## Where this leads

Type-system thinking reaches well beyond this chapter. The proof machinery that soundness is stated in (inference rules, derivations, induction, and relations such as the partial order `<:`) comes from [Discrete Math 1](../../mathematics/11-discrete-math-1/) in *Mathematics*. In [*Software Engineering*](../../software-engineering/), "make illegal states unrepresentable," design by contract, and orthogonality are type-system ideas applied to everyday code. [Information Theory](../../mathematics/17-information-theory/), also in *Mathematics*, offers a looser connection: the Curry–Howard–Lambek correspondence ties types to category theory, and a type can be read as a *compression* of a program's possible behaviors. In [LLM Systems & Inference](../../ai-engineering/20-inference-engines/) in *AI Engineering*, and throughout [*AI Engineering*](../../ai-engineering/), typed schemas (protobuf, Pydantic, TypeScript) are the contracts at every RPC and tool boundary, and structured-output decoding amounts to "make the model emit a well-typed value." A future chapter on compilers and parsing in this book will pick up where type checking begins: it is the phase after parsing, and GADTs can type the very ASTs a parser produces.

The ideas also show up directly in industry work. At OCaml and Haskell shops such as Jane Street and Citadel, "make illegal states unrepresentable" through variants and modules is a core interview theme, and HM inference, functors, and the value restriction are fair game. In Rust-heavy infrastructure (AWS, Cloudflare, Discord, Meta), trait bounds, `dyn` versus generics, variance, and the `Send`/`Sync` marker traits are daily design decisions. In large frontend and full-stack codebases (Stripe, Airbnb, Vercel), advanced TypeScript (conditional and mapped types, variance, the soundness gaps in `strict` mode) is how teams type an API so misuse will not compile. On compiler, language, and developer-tools teams, implementing inference, designing generics, and reasoning about soundness are the literal job. And in any large polyglot codebase, knowing *why* Go has no HKTs, *why* Java generics erase, and *why* `List<Dog>` is not a `List<Animal>` prevents whole categories of design mistakes and "fighting the type checker" churn.

### Further reading

- [*Types and Programming Languages*](https://www.cis.upenn.edu/~bcpierce/tapl/) by Benjamin Pierce: the definitive textbook, following the lambda calculus to STLC to System F to subtyping progression this chapter uses. The book is paid; Pierce's course slides and the companion *Software Foundations* are free.
- [*Programming Language Foundations in Agda*](https://plfa.github.io/) by Wadler, Kokke, and Siek (free, full text): STLC, type soundness, and inference, mechanized so the proofs run.
- [*Software Foundations* Vol. 2: *Programming Language Foundations*](https://softwarefoundations.cis.upenn.edu/plf-current/) (free): progress, preservation, and subtyping in Coq.
- Cardelli and Wegner, [*On Understanding Types, Data Abstraction, and Polymorphism*](https://dl.acm.org/doi/10.1145/6041.6042) (free PDF widely mirrored): the paper that names the polymorphism taxonomy.
- Damas and Milner, [*Principal type-schemes for functional programs*](https://dl.acm.org/doi/10.1145/582153.582176) (free): Algorithm W.
- Wadler, [*Theorems for Free!*](https://homepages.inf.ed.ac.uk/wadler/papers/free/free.pdf) (free): parametricity.
- Wadler and Blott, [*How to make ad-hoc polymorphism less ad hoc*](https://people.csail.mit.edu/dnj/teaching/6898/papers/wadler88.pdf) (free): typeclasses.
- The [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html), the [Go generics tutorial](https://go.dev/doc/tutorial/generics), and [Chapter 10 of the Rust Book](https://doc.rust-lang.org/book/ch10-00-generics.html) (all free).

## Key terms

- **Typing judgment**: the statement $\Gamma \vdash e : \tau$, that expression $e$ has type $\tau$ in context $\Gamma$.
- **Soundness**: the guarantee that well-typed programs do not get stuck, proved as progress plus preservation.
- **Parametric polymorphism**: one implementation that works uniformly at every type because it cannot inspect the type.
- **Ad-hoc polymorphism**: a different implementation per type, chosen by the types involved (overloading, typeclasses, traits, interfaces).
- **Bounded quantification**: parametric polymorphism restricted to types below a bound, as in `T: Ord`.
- **Principal type**: the most general type of a term, of which all its other types are instances.
- **Unification**: solving equations between types by finding the most general unifier.
- **Occurs check**: the test that a type variable does not appear inside the type it is being bound to, which keeps types finite.
- **Value restriction**: ML's rule that only syntactic values are generalized, keeping inference sound with mutable references.
- **Variance**: how subtyping on a type argument lifts to the constructed type: covariant, contravariant, or invariant.
- **Monomorphization**: compiling generics by emitting a specialized copy per concrete type.
- **Type erasure**: compiling generics to one copy whose type parameters vanish at runtime.

## Summary

- A type system is a decidable, conservative proof system; its central theorem is soundness, stated as progress plus preservation, and Curry–Howard identifies its types with propositions.
- "Polymorphism" names parametric, ad-hoc, subtype, and coercion mechanisms, with bounded quantification combining parametric and subtype polymorphism.
- System F gives full parametric polymorphism but undecidable inference; Hindley–Milner restricts polymorphism to `let` and rank-1 types so Algorithm W can infer principal types with no annotations.
- Unification with the occurs check is the engine of inference, and type errors are unification failures.
- Functions are contravariant in arguments and covariant in results; mutability forces invariance, and covariant mutable arrays are a deliberate soundness hole.
- How generics compile (monomorphization, erasure, or dictionaries and stenciling) determines their runtime cost more than whether a language "has generics."

## Review questions

1. What do progress and preservation each guarantee, and why do you need both for soundness?
   <details><summary>Answer</summary>Progress guarantees that a well-typed term is either a value or can take a step, so it is never stuck. Preservation guarantees that taking a step does not change the term's type. Progress alone says the current term is not stuck; preservation ensures the next term is still well typed, so progress applies again at every step. Together they give "well-typed programs don't go wrong."</details>

2. Why does `let id = \x.x in (id 1, id "a")` typecheck under Hindley–Milner while `(\id. (id 1, id "a")) (\x.x)` does not?
   <details><summary>Answer</summary>A `let`-bound value is generalized into a scheme $\forall a.\ a \to a$, and each use is instantiated with fresh variables, so `id` can be used at `int` and at `string`. A lambda parameter stays monomorphic: `id` gets a single type variable, and unifying it with both `int → ...` and `string → ...` fails.</details>

3. What does the occurs check prevent, and which term in the companion code triggers it?
   <details><summary>Answer</summary>It prevents binding a type variable to a type that contains it, which would build an infinite type. In `\x. x x`, inference must unify $\alpha = \alpha \to \beta$; the occurs check rejects this. `code/hindley_milner.py` demonstrates it on exactly that term.</details>

4. Given `Dog <: Animal`, is `(Animal → Dog) <: (Dog → Animal)`? Explain using the function rule.
   <details><summary>Answer</summary>Yes. The rule requires the target's argument to be a subtype of the source's argument (`Dog <: Animal`, contravariance) and the source's result to be a subtype of the target's result (`Dog <: Animal`, covariance). Both hold, so a function that accepts any animal and returns a dog can be used wherever a function from dogs to animals is expected.</details>

5. Why must a mutable `Array<Dog>` be invariant, and how does Java's array design violate this?
   <details><summary>Answer</summary>If `Array<Dog>` were a subtype of `Array<Animal>`, you could write a `Cat` through the `Animal` view and later read it as a `Dog`. Java makes arrays covariant anyway, so `Object[] a = new String[1]; a[0] = 1;` compiles, and the error is caught only at runtime with an `ArrayStoreException`.</details>

6. Compare the runtime cost of a Rust `<T: Shape>` function, a Java generic method, and a Haskell `Shape a =>` function.
   <details><summary>Answer</summary>Rust monomorphizes: one specialized copy per type, with direct, inlinable calls but larger binaries and slower compiles. Java erases: one copy with boxed values and no type arguments at runtime, so it is small but pays for boxing and loses runtime type information. Haskell passes a dictionary: one copy that receives a record of method implementations, a middle path with modest code size and some indirection.</details>

7. What is the difference between structural and nominal typing, and what does each protect against or enable?
   <details><summary>Answer</summary>Structural typing (TypeScript, Go interfaces) matches types by shape, which enables gradual adoption and lets any value with the right fields or methods fit. Nominal typing (Rust, Haskell, Java, Scala) matches by declared name, which makes intent explicit and prevents accidental compatibility, such as unrelated `Meters` and `Feet` types with the same shape being interchangeable.</details>

[Chapter contents](README.md)
