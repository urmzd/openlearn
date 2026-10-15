# Programming Languages (2-3 weeks)

~8-10 hours per week. The math first (lambda-calculus ladder → Hindley-Milner inference), then the *same* concept across five type systems. Run the `code/` files as you read.

### Week 1: The Math of Types

| Day | Focus | Reference |
|-----|-------|-----------|
| 1 | What a type system is: judgments `Γ ⊢ e : τ`, soundness = progress + preservation, Curry-Howard | TAPL Ch 1-9 / Software Foundations |
| 2 | The polymorphism taxonomy (Strachey, Cardelli-Wegner): parametric, ad-hoc, subtype, bounded | Cardelli-Wegner paper |
| 3 | The lambda-calculus ladder: STLC → System F (parametric polymorphism, parametricity) | TAPL Ch 22-23 |
| 4-5 | Hindley-Milner + Algorithm W: unification, the occurs check, `let`-generalization, principal types. Read & run `code/hindley_milner.py` | Damas-Milner paper |

### Week 2: The Same Concept, Five Mechanisms

| Day | Focus | Reference |
|-----|-------|-----------|
| 1 | Structural typing + variance + type-level programming — `code/polymorphism.ts` | TS Handbook |
| 2 | Go generics: type sets/unions, GC-shape stenciling — `code/generics.go` | Go generics tutorial |
| 3 | Rust traits: static (monomorphized) vs dynamic (`dyn`) dispatch, associated types — `code/traits.rs` | Rust Book Ch 10, 17 |
| 4 | Haskell typeclasses (dictionary passing) + higher-kinded `Functor` — `code/typeclasses.hs` | Wadler-Blott paper |
| 5 | OCaml HM inference + modules/functors — `code/inference.ml`; compare its principal types to Week 1 | Real World OCaml |

### Week 3 (optional): Depth

| Day | Focus | Reference |
|-----|-------|-----------|
| 1-2 | Subtyping & variance in depth: declaration- vs use-site, the array-covariance soundness hole | TAPL Ch 15-16 |
| 3 | How generics compile: monomorphization vs erasure vs dictionaries — the runtime-cost trade-off | — |
| 4-5 | The frontier: HKTs, associated types, GADTs, dependent types (the lambda cube top) | PLFA |

---
