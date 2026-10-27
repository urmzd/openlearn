# System F and type inference

This part of [Type Systems and Polymorphism](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Trace generalization, instantiation, and the occurs check.

## From the simply typed lambda calculus to System F

Parametric polymorphism has a precise origin, and tracing it explains why inference is hard.

The **simply typed lambda calculus (STLC)** adds types to the lambda calculus and immediately loses Turing-completeness: every well-typed term halts, a property called *strong normalization*. STLC is also **monomorphic**. An identity function must be written separately at each type, and that limitation is the entire motivation for what follows.

**System F**, the polymorphic or second-order lambda calculus discovered independently by Girard (1972) and Reynolds (1974), adds **type abstraction** $\Lambda\alpha.\ e$ and **type application** $e\ [\tau]$. Now $\mathit{id} = \Lambda\alpha.\ \lambda(x{:}\alpha).\ x : \forall\alpha.\ \alpha \to \alpha$ is a *single* term usable at every type. This is exactly parametric polymorphism, and `Stack<T>` is its data-type form.

The catch is that System F type *inference* is undecidable. You can *check* an annotated System F program, but you cannot in general *recover* the $\Lambda$s and $[\tau]$s from unannotated code. Languages that expose full System F, such as Haskell with `RankNTypes` and Scala, require annotations at the hard spots.

**System Fω** goes one step further and adds **type-level functions**, operators that take types to types. That gives **higher-kinded types**: abstraction over type *constructors*, as in `Functor f`, which the frontier section returns to.

## Hindley–Milner: the decidable sweet spot

If full System F inference is undecidable, how do OCaml and Haskell infer types with no annotations at all? The answer is Hindley–Milner (Hindley 1969, Milner 1978, Damas and Milner 1982), and it is the keystone of this chapter.

HM is the restriction of System F in which polymorphism is introduced *only* at `let`, and quantifiers stay on the outside of a type (**prenex**, or rank-1, form). That restriction is exactly enough to make inference decidable *and* to give every term a unique best type.

That best type is the **principal type**: every typeable HM term has a single most-general type of which all its other types are instances. For example, `\f.\g.\x. f (g x)` has principal type `(a → b) → (c → a) → c → b`, and every other type you could give it is a specialization.

**Algorithm W** computes principal types with a recursive walk over the term that threads a **substitution** through, generating fresh type variables at each node and **unifying** them as constraints appear. It has four pieces: `unify` solves an equation $\tau_1 = \tau_2$; `instantiate` gives each *use* of a type scheme fresh variables; `generalize` closes a monotype into a $\forall$-scheme at a `let`; and the inference walk ties them together.

`let`-generalization is the whole trick. A lambda parameter stays monomorphic, because inside the body you do not yet know how it will be used, but a `let`-bound value is **generalized** so it can be reused at many types. That is why `let id = \x.x in (id 1, id "a")` typechecks while `(\id. (id 1, id "a")) (\x.x)` does not.

> **Warning:** Generalization interacts badly with mutation. With mutable references, naive generalization is *unsound*: you could store an `int` in a polymorphic reference and read it back as a `string`. ML's fix is the **value restriction**, which generalizes only syntactic values. It is a small, famous patch that keeps HM plus mutation sound.

[`code/hindley_milner.py`](code/hindley_milner.py) is Algorithm W end to end, and [`code/inference.ml`](code/inference.ml) shows OCaml inferring the same principal types in real code. Read them together: one *is* the inference engine, the other *uses* it.

## Unification and the occurs check

Inference reduces to solving equations between types, so the engine inside every ML-family compiler is a solver for those equations. **Unification** (Robinson, 1965) finds the **most general unifier**, the least-committal substitution that makes two types equal.

The algorithm is short. To unify $\tau_1$ and $\tau_2$: if either is a type variable $\alpha$, bind $\alpha$ to the other; if both are arrows, unify the arguments and then, under that substitution, the results; if both are the same constructor, succeed; otherwise fail with a type error.

There is one guard that matters. Before binding $\alpha := \tau$, the **occurs check** verifies that $\alpha$ does not occur *inside* $\tau$. Without it, `\x. x x` would unify $\alpha = \alpha \to \beta$ and build the **infinite type** $\alpha = \alpha \to \alpha \to \cdots$. The check is what keeps types finite, and it is the source of the classic error "occurs check: cannot construct the infinite type." [`code/hindley_milner.py`](code/hindley_milner.py) triggers it on `\x. x x`.

This view also explains what a type error *is*: a unification failure. ML messages say "expected `int`, got `string`" because they are reporting the two ends of a constraint that could not be made equal. The *location* blamed, however, is an artifact of traversal order, which is why HM error messages are famously sometimes misleading.

## Summary

Trace generalization, instantiation, and the occurs check. State the assumptions before applying the result to a different workload.

## Review questions

1. Why reject unifying a type variable with a type containing itself?

   <details><summary>Answer</summary>

   Without recursive-type rules this demands an infinite type, outside the inference system.

   </details>

[Chapter contents](README.md) · [Next: Dispatch, variance, and type expressiveness](03-dispatch-variance-and-type-expressiveness.md)
