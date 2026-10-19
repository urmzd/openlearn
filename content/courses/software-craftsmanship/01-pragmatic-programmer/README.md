# The Pragmatic Programmer

## Overview

- **Primary reference**: *The Pragmatic Programmer* by Hunt & Thomas (20th Anniversary Edition) -- the foundational text on individual professional software craft
- **Supplementary**: *Clean Code* by Robert Martin, *A Philosophy of Software Design* by John Ousterhout, Martin Fowler's [Refactoring Catalog](https://refactoring.com/catalog/) (free)
- **Prerequisites**: Some professional programming experience (this is a "how to think" topic, not a "how to code" topic)
- **Estimated time**: 1.5-2 weeks at 6-8 hrs/week

## Key Takeaways

- Software development is a **craft**: take responsibility, think critically, adapt on feedback.
- The big levers are **DRY**, **orthogonality**, and **reversibility** -- they decide whether a system can change without breaking.
- **Crash early** and **assert the impossible**: corrupted state that keeps running is worse than a clean failure.
- Most pragmatic principles are about **managing coupling and complexity over time**, not clever code.

## How to Study

- Read one chapter per day; after each, identify one thing to apply to your current project.
- Best read cover-to-cover -- the chapters build on each other.
- For each principle, find a place in your own codebase where violating it cost you.

---

# Concepts & Techniques

## Core Insight

Software development is a craft. Pragmatic programmers take responsibility for their work, think critically about what they're doing, and continuously adapt their approach based on feedback.

## 1. A Pragmatic Philosophy

**Key principles**:
- **It's your life**: take charge of your career and your code. Don't say "I can't" -- say "I can, but here's what it costs"
- **The cat ate my source code**: take responsibility. Provide options, not excuses
- **Software entropy**: broken windows theory -- don't leave bad code unfixed, it gives permission for more
- **Good enough software**: know when to stop. Perfect is the enemy of shipped
- **Your knowledge portfolio**: invest regularly, diversify, review and rebalance

**How this appears at Staff+ level**: you set the quality bar for your team. If you tolerate broken windows, everyone does.

## 2. A Pragmatic Approach

**Key principles**:
- **DRY (Don't Repeat Yourself)**: every piece of knowledge has a single, unambiguous representation. Not just code -- also schemas, documentation, build scripts
- **Orthogonality**: components should be independent. Change one without affecting others. Test: if I change X, does Y break?
- **Reversibility**: keep decisions soft. Use configuration, dependency injection, feature flags. Assume requirements will change
- **Tracer bullets**: build thin vertical slices that work end-to-end. Not prototypes -- real code, minimal but complete
- **Prototypes**: throwaway code to explore ideas. Unlike tracer bullets, prototypes get discarded
- **Estimation**: learn to give estimates in ranges. Track your estimates vs actuals to calibrate

**Worked example**:
> Tracer bullet vs prototype: building a new API gateway. A *tracer bullet* would be one real endpoint, with real auth, real logging, real deployment -- but only one route. A *prototype* would be a script that tests latency between services, then gets thrown away.

## 3. The Basic Tools

**Key ideas**:
- **Power of plain text**: store knowledge in plain text when possible -- it survives everything
- **Shell games**: master the command line. Compose small tools. Automate everything you do twice
- **Version control**: everything goes in version control. Infrastructure, configs, docs, not just code
- **Debugging**: don't panic. Reproduce it first. Binary search the problem space. Read the error message

## 4. Pragmatic Paranoia

**Key principles**:
- **Design by Contract**: preconditions, postconditions, invariants. Define what a function promises
- **Dead programs tell no tales**: crash early. Don't rescue and continue with corrupted state
- **Assertive programming**: use assertions for things that "can't happen" -- they can and will
- **Resource management**: who allocates, deallocates. RAII, try-with-resources, context managers

## 5. Bend or Break

**Key principles**:
- **Decoupling**: law of Demeter -- "don't talk to strangers". Minimize what each module knows about others
- **Transforming programming**: think of programs as data pipelines, not as objects calling each other
- **Events and reactivity**: reduce coupling via events, pub/sub, reactive streams

## 6. Concurrency & While You Are Coding

**Key ideas**:
- **Temporal coupling**: identify what must be sequential vs what can be parallel
- **Refactoring**: refactor early, refactor often. Martin Fowler's catalog. Don't refactor and add features simultaneously
- **Testing**: test early, test often, test automatically. Tests are the first users of your code (deepened in [The Testing Mentality](../03-testing-mentality/))
- **Property-based testing**: generate random inputs, check invariants -- finds bugs that example-based tests miss
- **Naming things**: names reveal intent. If you can't name it, you don't understand it

## Technique Catalog

| Principle | When to Apply |
|-----------|---------------|
| DRY | When you find yourself copying *knowledge* (not just code) |
| Orthogonality | Drawing module boundaries; minimizing blast radius of change |
| Tracer bullets | Starting a new project or major feature |
| Reversibility | Making architectural decisions (keep them reversible) |
| Design by Contract | Defining a function/API's promises and invariants |
| Crash early | Anywhere continuing with corrupt state is dangerous |
| Broken windows | Deciding whether to "fix it later" (don't) |

## Connections to Other Tracks

| Concept | Connected Track | How |
|---------|-----------------|-----|
| DRY, orthogonality | [Software Architecture](../../systems/02-software-architecture/) | Architecture patterns formalize these principles |
| Refactoring | [Algorithms](../../algorithms/) | Refactoring algorithm code for clarity without changing behavior |
| Design-by-contract, invariants | [The Testing Mentality](../03-testing-mentality/) | Contracts become the properties you test |
| Artifacts as source | [Diagramming & Documentation](../) | The same "treat it like code" discipline for diagrams and docs |

## Company Relevance

| Company | How This Appears | Focus |
|---------|-----------------|-------|
| Anthropic | Production-quality, safety-critical engineering; thoughtful design | Pragmatic approach |
| Stripe | API design excellence, backwards compatibility | DRY, contracts |
| Amazon | Operational excellence, ownership, working backwards | Responsibility, estimation |
| Netflix | Freedom and responsibility, chaos engineering | Pragmatic paranoia |
| All Staff+ roles | Every company expects Staff+ engineers to drive engineering excellence | All principles |
