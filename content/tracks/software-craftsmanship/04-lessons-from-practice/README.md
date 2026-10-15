# Lessons from Practice

## Overview

- **Primary reference**: postmortems of shipped and shelved projects (2019-2021) -- university assignments, interview take-homes, internship applications
- **Supplementary**: [The Pragmatic Programmer](../01-pragmatic-programmer/) (the principles these violate), [The Testing Mentality](../03-testing-mentality/) (the discipline half of them are missing)
- **Prerequisites**: None. This topic is the cheapest way to acquire the mistakes without making them.
- **Estimated time**: 2-3 days

## Key Takeaways

- Almost every mistake below cost **under 30 minutes to prevent** and hours to live with. The gap between "works once" and "works reliably" is small and knowable in advance.
- **Judgment is what gets evaluated**, not feature count. A working app with `alert()` for errors and no tests reads worse than a smaller app with neither problem.
- **Infrastructure surface area compounds.** Each service is defensible alone; together they consume the timeline that was supposed to go to the actual work.
- **Uncommitted work is work that did not happen**, and undocumented work is work nobody can see.

## How to Study

- Read the lesson, then grep your own current project for the pattern. Every lesson here is mechanically detectable.
- For each one you find, decide: fix now, or write down why not. Both are valid; silence is not.
- Revisit when starting a time-boxed project. Themes 4 and 5 are the ones that decide whether a one-week build ships.

---

# Concepts & Techniques

## Core Insight

These lessons come from real projects, but the projects are not the point. Each one is a general failure mode that recurs across languages, stacks, and decades: unnamed constants, unhandled I/O, unbounded scope, uncommitted work. The provenance is kept as one line for context; the lesson is written to transfer.

The recurring shape: **the expensive mistakes were cheap to avoid, and every one of them was visible at the time.**

## Theme 1: The code is the documentation

*When the code is all a reader has, unreadable code is undocumented code. On constrained platforms this is literal; everywhere else it is true in practice, because the README goes stale and the code does not.*

### 1. Name every constant

*From an embedded robotics assignment (2019): sensor thresholds `3000`, `2200`, `125`, and a turn duration of `7`, none named or explained.*

A bare number encodes a decision that someone made once, under conditions nobody wrote down. Recalibrating means reverse-engineering the author's intent from raw values.

```js
// BAD -- what is 3000? Measured on what surface, under what lighting?
if (sensor.front > 3000) reverse();

// GOOD -- the name carries the decision, the comment carries the context.
const OBSTACLE_PROXIMITY = 3000; // calibrated on matte white, indoor lighting
if (sensor.front > OBSTACLE_PROXIMITY) reverse();
```

Cost to prevent: one line. Cost to skip: every future recalibration is an archaeology exercise.

### 2. Names are written for readers, not authors

*From the same project: a variable named `FAWAD` ("forward" in Arabic), and `SAVED` for a remembered color.*

A name that makes sense to you today serves exactly one person for about two weeks. Clever, personal, or abbreviated names are a tax on everyone else, including future-you.

Rule of thumb: if explaining the name takes a sentence, the name should have been that sentence, compressed.

### 3. Booleans model two states, not two meanings

*From a checkers game (2020): teams were `true`/`false`, square colors were booleans, piece presence was `false` or an object.*

`team === true` tells you nothing about which team. Booleans are correct only when the domain genuinely has two nameless states; the moment the states have names, use them.

```ts
// BAD -- the type permits the code, but the code says nothing.
if (piece.team === true) { ... }

// GOOD -- self-documenting, and the compiler rejects a third value.
type Team = "red" | "black";
if (piece.team === "red") { ... }
```

This is also the cheapest bug filter available: a union type makes the invalid state unrepresentable, where a boolean makes it a coin flip.

## Theme 2: Reliability in small code

*Throwaway scripts fail in exactly the same ways production systems do, just with nobody on call. The difference between "worked on my machine once" and "works" is usually ten lines.*

### 4. Every I/O call needs a failure path

*From a scraping script (2020): 2,084 sequential HTTP requests, no try/catch, no retry. One timeout killed the entire run.*

At any per-request failure rate above roughly 0.05%, a 2,000-request run without error handling is more likely to fail than succeed. Networks are not an edge case at that volume; they are the main case.

```js
// GOOD -- bounded retry with backoff. This is the whole tax.
async function fetchWithRetry(url, attempts = 3) {
  for (let i = 0; i < attempts; i++) {
    try {
      return await fetch(url);
    } catch (err) {
      if (i === attempts - 1) throw err;
      await sleep(2 ** i * 500); // 500ms, 1s, 2s
    }
  }
}
```

### 5. Bound your concurrency, and rate-limit on purpose

The same script fired requests as fast as the network allowed, and fetched them one at a time. Both are wrong, in opposite directions:

- **Sequential** wastes wall-clock time proportional to request count.
- **Unbounded parallel** is rude to the server and reliably earns a rate-limit or an IP ban.

The answer is a fixed-size batch with a deliberate delay between batches. Pick the concurrency number consciously; the default of "one" and the default of "all" are both accidents.

### 6. Validate at the boundary

*The same script wrote whatever it received straight to JSON. A CAPTCHA page or a redesigned layout produced structurally valid, semantically empty output.*

Scraped HTML, API responses, and user input are all untrusted shapes. Without a check at the boundary, the failure does not surface at the boundary -- it surfaces three steps downstream, as a mystery.

```js
// GOOD -- assert the record is meaningful before it enters the dataset.
if (!record.brand || !record.model || record.prices.length === 0) {
  log.warn({ url }, "selector matched nothing meaningful, skipping");
  continue;
}
```

Silent garbage in a dataset is worse than a crash, because it gets used.

## Theme 3: Judgment signals in evaluated work

*Interview take-homes, portfolio projects, and pull requests are all read for judgment, not feature count. The small choices are legible, and they are what gets read.*

### 7. Configuration lives in the environment

*From an interview CRUD app (2020): the API endpoint was a string literal in the HTTP client config.*

A hard-coded endpoint means the project runs in exactly one environment, and moving it is a code change. Every framework has a built-in answer to this, and using it takes under a minute.

```ts
// BAD
const client = axios.create({ baseURL: "http://prod-api.example.com" });

// GOOD
const client = axios.create({ baseURL: process.env.API_URL });
```

### 8. Every async action needs a visible state

The same app used `alert()` for errors and rendered nothing during requests. Both are UX failures with a shared root: the interface does not represent what the system is doing.

A user who clicks and sees nothing assumes it is broken. Loading state is one boolean and a conditional render; an inline error message is fewer lines than the `alert()` it replaces, and does not block the thread.

### 9. If you install it, use it

*From the checkers game: TypeScript in `package.json`, every file `.js`. A Redux turn slice that was never read. Buttons rendered with no handlers.*

*From the CRUD app: a TypeScript project with `any` on the fields that mattered.*

Unused infrastructure is a claim the code does not back. `any` on the interesting field defeats the point of the type system precisely where it would have paid off. Either commit to the tool or remove it -- the half-state is the only genuinely bad option, because it misleads the reader about what the code guarantees.

## Theme 4: Scope and infrastructure discipline

*Two projects failed the same way: the interesting part was never reached, because the platform around it consumed the time.*

### 10. Size the infrastructure to the timeline

*From a one-week internship application (2021): managed backend + GraphQL API + NoSQL store + object storage + serverless functions + API gateway + a deployment framework + a vision API + a static site generator + a monorepo with shared types. Roughly half the week went to CI/CD, environment variables, and build configuration.*

Every one of those choices is defensible in isolation. Together they created a surface area larger than the project. The signal to watch for is commit archaeology: **20+ commits in a single day, none of them touching the core feature.**

For a one-week build, the boring stack that supports the same demo is the correct stack. Infrastructure impresses on paper and disappears in practice; a working demo does the opposite.

### 11. Tracer-bullet the core mechanic first

*From the checkers game: the board renders beautifully -- highlighted moves, color-coded teams, king indicators -- and no rule is enforced. No turns, no captures, no promotion.*

Building the visible shell before the mechanic feels productive because progress is visible. But the shell is the part you already know how to build; the mechanic is where the unknowns are. Front-load the unknowns.

The [tracer bullet](../01-pragmatic-programmer/) is the discipline here: one thin slice through the *whole* system, core logic included, before broadening any layer.

## Theme 5: Evidence and closure

*Work that was done but not recorded, and projects abandoned mid-thought, are indistinguishable from work that never happened.*

### 12. Evidence beats description

*From the internship project: the game was finished. The final state was never committed -- the repo shows the core screen as an empty stub, making a completed project look abandoned. There was also no demo and no screenshot, only setup instructions for infrastructure that required cloud credentials to run.*

Two halves of one failure:

- **Commit early and often.** Small frequent commits preserve the arc of the work. `git commit` is not only version control; it is the only proof that the work happened.
- **Show it working.** Thirty seconds of screen capture outsells a README that requires credentials to verify. A reviewer who cannot run it will not run it.

### 13. Clean up before you shelve it

*From the checkers game, set aside deliberately: dead Redux slices, unwired buttons, an unused language toolchain, and two empty test files left in place.*

Setting a project aside is a legitimate decision. Leaving the debris is a separate one. When you know you are stopping, strip what was planned but not built, so the repository reflects what exists rather than what was intended.

The code you leave behind is the code that represents you.

## Technique Catalog

| Technique | When to apply |
|-----------|---------------|
| Extract named constants | Any literal that encodes a decision or a calibration |
| Union types over booleans | Any two-state field whose states have names |
| Bounded retry with backoff | Every network call, including in throwaway scripts |
| Fixed-size concurrency batches | Any loop over more than ~50 remote resources |
| Boundary validation | Every untrusted input: scraped, fetched, or user-supplied |
| Config from environment | Every endpoint, credential, or environment-varying value |
| Explicit loading and error states | Every user-triggered async action |
| Remove unused tooling | Before every review, and before shelving anything |
| Stack-vs-timeline check | At the start of any time-boxed build |
| Tracer bullet through core logic | Before broadening any single layer |
| Commit the core loop first | The moment it works, not when it is polished |
| Record a demo | Anything that will be evaluated by someone else |

## Connections to Other Tracks

| Concept | Connected Track | How |
|---------|-----------------|-----|
| Naming, DRY, tracer bullets, reversibility | [The Pragmatic Programmer](../01-pragmatic-programmer/) | The principles these lessons violate, stated positively |
| Missing tests, empty test files | [The Testing Mentality](../03-testing-mentality/) | Pure functions like move calculation are the easiest tests to write and the ones most often skipped |
| Code review as the judgment filter | [Software Engineering at Google](../02-swe-at-google/) | The signals in Theme 3 are exactly what reviewers read for |
| Union types, unrepresentable invalid states | [Type Systems](../07-type-systems/) | The formal version of lesson 3 |
| Retry, backoff, rate limiting, idempotency | [Distributed Workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/) | The production-scale form of Theme 2 |
| Infrastructure surface area vs delivery | [System Design](../../systems/01-system-design/) | Choosing the smallest architecture that meets the requirement |

## Company Relevance

| Company | How This Appears | Focus |
|---------|-----------------|-------|
| Any take-home | Tests, error handling, config hygiene read as judgment | Small choices, loud signals |
| Any code review | Magic numbers, dead code, `any` are the standard comments | Reviewability |
| Amazon | Written narratives and operational readiness | Evidence over assertion |
| Google | Readability review as a formal gate | Naming and clarity as policy |
| Any Staff+ role | Scoping infrastructure to the actual requirement | Judgment under a deadline |
