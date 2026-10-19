# Lessons from Practice

The principles in the first chapters of this book are easiest to appreciate when you see what happens without them. This chapter is a set of postmortems: thirteen lessons drawn from real projects, shipped and shelved, written between 2019 and 2021. They include university assignments, interview take-homes, and internship applications. None of the projects matters in itself. Each one illustrates a general failure mode that recurs across languages, stacks, and decades: unnamed constants, unhandled I/O, unbounded scope, uncommitted work.

The recurring shape is worth stating up front, because it is the reason the chapter exists: **the expensive mistakes were cheap to avoid, and every one of them was visible at the time.** Almost every mistake below cost under thirty minutes to prevent and hours to live with. The gap between "works once" and "works reliably" is small and knowable in advance. The provenance of each lesson is kept as one line for context; the lesson itself is written to transfer to whatever you are building now.

The lessons are grouped into five themes. The first two are about code: readability when the code is all a reader has, and reliability in code too small to seem worth the effort. The third is about how evaluated work is read, which is for judgment rather than feature count. The last two are about the project around the code: sizing infrastructure to the timeline, and leaving evidence that the work happened.

> **Learning objectives**
> - Recognize the thirteen failure modes in this chapter in your own code, and find each one mechanically (every lesson here can be found with a search).
> - Replace magic numbers, personal names, and meaningful booleans with named constants, reader-oriented names, and union types.
> - Give every I/O call a failure path, a deliberate concurrency bound, and validation at the boundary.
> - Size a project's infrastructure to its timeline and build a tracer bullet through the core mechanic first.
> - Leave a repository that shows what exists: committed, demonstrated, and cleaned up before it is set aside.

## Before you begin

There are no prerequisites. This chapter is the cheapest way to acquire these mistakes without making them. It pairs naturally with [The Pragmatic Programmer](../01-pragmatic-programmer/), which states positively the principles these lessons violate, and with [The Testing Mentality](../03-testing-mentality/), which supplies the discipline that half of them are missing.

The most useful way to read it is with your own current project open. After each lesson, search your code for the pattern. For each instance you find, decide whether to fix it now or to write down why not. Both are valid answers; silence is not. Come back to the fourth and fifth themes whenever you start a time-boxed project, because they are the ones that decide whether a one-week build ships.

## The code is the documentation

When the code is all a reader has, unreadable code is undocumented code. On constrained platforms this is literally true; everywhere else it is true in practice, because the README goes stale and the code does not. The first three lessons are about making the code carry its own meaning.

### Name every constant

The first lesson comes from an embedded robotics assignment in 2019, whose sensor thresholds were `3000`, `2200`, and `125`, with a turn duration of `7`, none of them named or explained.

A bare number encodes a decision that someone made once, under conditions nobody wrote down. Recalibrating means reverse-engineering the author's intent from raw values. The fix is a name for the decision and a comment for its context:

```js
// BAD -- what is 3000? Measured on what surface, under what lighting?
if (sensor.front > 3000) reverse();

// GOOD -- the name carries the decision, the comment carries the context.
const OBSTACLE_PROXIMITY = 3000; // calibrated on matte white, indoor lighting
if (sensor.front > OBSTACLE_PROXIMITY) reverse();
```

The cost to prevent is one line. The cost to skip it is that every future recalibration becomes an archaeology exercise.

### Write names for readers, not authors

The same project had a variable named `FAWAD` ("forward" in Arabic) and another named `SAVED` for a remembered color. A name that makes sense to you today serves exactly one person for about two weeks. Clever, personal, or abbreviated names are a tax on everyone else, including the future you.

A good rule of thumb: if explaining the name takes a sentence, the name should have been that sentence, compressed.

### Use booleans for two states, not two meanings

A checkers game from 2020 made teams `true` and `false`, made square colors booleans, and represented piece presence as either `false` or an object. The trouble is that `team === true` tells you nothing about which team. Booleans are correct only when the domain genuinely has two nameless states; the moment the states have names, use the names.

```ts
// BAD -- the type permits the code, but the code says nothing.
if (piece.team === true) { ... }

// GOOD -- self-documenting, and the compiler rejects a third value.
type Team = "red" | "black";
if (piece.team === "red") { ... }
```

This is also the cheapest bug filter available. A union type makes the invalid state unrepresentable, where a boolean makes it a coin flip.

## Reliability in small code

Readable code still has to work. Throwaway scripts fail in exactly the same ways production systems do, just with nobody on call. The difference between "worked on my machine once" and "works" is usually ten lines, and the next three lessons are those ten lines.

### Give every I/O call a failure path

A scraping script from 2020 made 2,084 sequential HTTP requests with no try/catch and no retry. One timeout killed the entire run.

The arithmetic is unforgiving. At any per-request failure rate above roughly 0.05%, a 2,000-request run without error handling is more likely to fail than to succeed. At that volume, network failures are not an edge case; they are the main case. A bounded retry with backoff is the whole tax:

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

### Bound your concurrency, and rate-limit on purpose

The same script got concurrency wrong in two opposite directions. It fetched one page at a time, and it fired those requests as fast as the network allowed. Sequential fetching wastes wall-clock time in proportion to the request count. Unbounded parallel fetching, the usual overcorrection, is rude to the server and reliably earns a rate limit or an IP ban.

The answer is a fixed-size batch with a deliberate delay between batches. Pick the concurrency number consciously: the default of "one" and the default of "all" are both accidents.

### Validate at the boundary

The same script wrote whatever it received straight to JSON. A CAPTCHA page or a redesigned layout produced output that was structurally valid and semantically empty.

Scraped HTML, API responses, and user input all have untrusted shapes. Without a check at the boundary, the failure does not surface at the boundary; it surfaces three steps downstream, as a mystery. Assert that each record is meaningful before it enters the dataset:

```js
// GOOD -- assert the record is meaningful before it enters the dataset.
if (!record.brand || !record.model || record.prices.length === 0) {
  log.warn({ url }, "selector matched nothing meaningful, skipping");
  continue;
}
```

Silent garbage in a dataset is worse than a crash, because it gets used.

## Judgment signals in evaluated work

Interview take-homes, portfolio projects, and pull requests are all read for judgment, not feature count. A working app that uses `alert()` for errors and has no tests reads worse than a smaller app with neither problem. The small choices are legible, and they are what gets read.

### Keep configuration in the environment

An interview CRUD app from 2020 put its API endpoint in a string literal in the HTTP client configuration. A hard-coded endpoint means the project runs in exactly one environment, and moving it is a code change. Every framework has a built-in answer to this, and using it takes under a minute:

```ts
// BAD
const client = axios.create({ baseURL: "http://prod-api.example.com" });

// GOOD
const client = axios.create({ baseURL: process.env.API_URL });
```

### Give every async action a visible state

The same app used `alert()` for errors and rendered nothing while a request was in flight. Both are UX failures with a shared root: the interface does not represent what the system is doing. A user who clicks and sees nothing assumes the app is broken. A loading state is one boolean and a conditional render, and an inline error message is fewer lines than the `alert()` it replaces and does not block the thread.

### If you install it, use it

The checkers game listed TypeScript in `package.json`, yet every file was `.js`. It had a Redux turn slice that was never read and buttons rendered with no handlers. The CRUD app, a TypeScript project, used `any` on exactly the fields that mattered.

Unused infrastructure is a claim the code does not back. And `any` on the interesting field defeats the point of the type system precisely where it would have paid off. Either commit to the tool or remove it. The half-state is the only genuinely bad option, because it misleads the reader about what the code guarantees.

## Scope and infrastructure discipline

The fourth theme moves from code to the project around it. Two projects failed in the same way: the interesting part was never reached, because the platform around it consumed the time.

### Size the infrastructure to the timeline

A one-week internship application from 2021 used a managed backend, a GraphQL API, a NoSQL store, object storage, serverless functions, an API gateway, a deployment framework, a vision API, a static site generator, and a monorepo with shared types. Roughly half the week went to CI/CD, environment variables, and build configuration.

Every one of those choices is defensible in isolation. Together they created a surface area larger than the project. Infrastructure surface area compounds: each service is reasonable alone, and together they consume the timeline that was supposed to go to the actual work. The signal to watch for is in the commit history: **20 or more commits in a single day, none of them touching the core feature.**

For a one-week build, the boring stack that supports the same demo is the correct stack. Infrastructure impresses on paper and disappears in practice; a working demo does the opposite.

### Tracer-bullet the core mechanic first

The checkers game's board rendered beautifully, with highlighted moves, color-coded teams, and king indicators, and enforced no rule at all: no turns, no captures, no promotion.

Building the visible shell before the mechanic feels productive, because the progress is visible. But the shell is the part you already know how to build; the mechanic is where the unknowns are. Front-load the unknowns. The [tracer bullet](../01-pragmatic-programmer/) from *The Pragmatic Programmer* is the discipline here: one thin slice through the whole system, core logic included, before broadening any layer.

## Evidence and closure

The last theme is about what remains after the work. Work that was done but not recorded, and projects abandoned mid-thought, are indistinguishable from work that never happened.

### Evidence beats description

The internship project's game was finished. The final state was never committed, so the repository shows the core screen as an empty stub, which makes a completed project look abandoned. There was also no demo and no screenshot, only setup instructions for infrastructure that needed cloud credentials to run.

That is two halves of one failure. First, commit early and often. Small, frequent commits preserve the arc of the work, and `git commit` is not only version control: it is the only proof that the work happened. Uncommitted work is work that did not happen. Second, show it working. Thirty seconds of screen capture outsells a README that requires credentials to verify, because a reviewer who cannot run it will not run it. Undocumented work is work nobody can see.

### Clean up before you shelve it

The checkers game was set aside deliberately, but it was left with dead Redux slices, unwired buttons, an unused language toolchain, and two empty test files.

Setting a project aside is a legitimate decision. Leaving the debris is a separate one. When you know you are stopping, strip what was planned but not built, so the repository reflects what exists rather than what was intended. The code you leave behind is the code that represents you.

## A catalog of the techniques

Each lesson reduces to a technique with a clear trigger. The table below collects them so you can use it as a checklist at the start of a project and again before a review.

| Technique | When to apply |
|-----------|---------------|
| Extract named constants | Any literal that encodes a decision or a calibration |
| Union types over booleans | Any two-state field whose states have names |
| Bounded retry with backoff | Every network call, including in throwaway scripts |
| Fixed-size concurrency batches | Any loop over more than about 50 remote resources |
| Boundary validation | Every untrusted input: scraped, fetched, or user-supplied |
| Config from environment | Every endpoint, credential, or environment-varying value |
| Explicit loading and error states | Every user-triggered async action |
| Remove unused tooling | Before every review, and before shelving anything |
| Stack-versus-timeline check | At the start of any time-boxed build |
| Tracer bullet through core logic | Before broadening any single layer |
| Commit the core loop first | The moment it works, not when it is polished |
| Record a demo | Anything that will be evaluated by someone else |

## Where this leads

These lessons are the negative image of principles stated elsewhere. Naming, DRY, tracer bullets, and reversibility are stated positively in [The Pragmatic Programmer](../01-pragmatic-programmer/). The missing tests and empty test files point to [The Testing Mentality](../03-testing-mentality/): pure functions like move calculation are the easiest tests to write and the ones most often skipped. The judgment signals of the third theme are exactly what reviewers read for, which is the subject of [Software Engineering at Google](../02-swe-at-google/) and of [Code Review and CI](../05-code-review-and-ci/) later in this book.

Several lessons have a formal or production-scale counterpart in other books. Union types and unrepresentable invalid states are the informal version of what [Type Systems](../../programming-languages/01-type-systems/) in *Programming Languages* treats formally. Retry, backoff, rate limiting, and idempotency reappear at production scale in [Durable Orchestration and Workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/) in *AI Platform Engineering*. The tension between infrastructure surface area and delivery is the core of [System Design](../../systems/01-system-design/) in *Systems*: choosing the smallest architecture that meets the requirement.

The same signals show up wherever work is evaluated. In any take-home, tests, error handling, and configuration hygiene are read as judgment: small choices, loud signals. In any code review, magic numbers, dead code, and `any` are the standard comments. Amazon's written narratives and operational-readiness reviews favor evidence over assertion. Google makes readability review a formal gate, turning naming and clarity into policy. And in any Staff-level role, scoping infrastructure to the actual requirement is judgment under a deadline.

## Key terms

- **Magic number**: a bare literal that encodes an unrecorded decision, such as a calibrated threshold.
- **Unrepresentable invalid state**: a type design, such as a union of named values, under which the compiler rejects values the domain does not allow.
- **Bounded retry with backoff**: retrying a failed call a fixed number of times with growing delays, then surfacing the error.
- **Boundary validation**: checking that untrusted input is meaningful at the point it enters the system, before it propagates.
- **Tracer bullet**: a thin, end-to-end slice through the whole system, core logic included, built before any single layer is broadened.
- **Infrastructure surface area**: the total configuration, integration, and deployment work that a stack's components demand together.

## Summary

- The expensive mistakes were cheap to avoid and visible at the time; most cost under thirty minutes to prevent and hours to live with.
- Code is the documentation that does not go stale: name constants, write names for readers, and use named states instead of meaningful booleans.
- Small code needs the same reliability as large code: failure paths for I/O, deliberate concurrency, and validation at the boundary.
- Evaluated work is read for judgment, not feature count; configuration, visible async state, and honest tooling are the signals.
- Size infrastructure to the timeline and tracer-bullet the core mechanic first.
- Uncommitted work did not happen, and undocumented work cannot be seen: commit, demonstrate, and clean up before shelving.

## Review questions

1. Why is a bare constant like `3000` more expensive than the one line it would take to name it?
   <details><summary>Answer</summary>The number encodes a decision made under conditions nobody recorded (surface, lighting, hardware). Without a name and a context comment, every recalibration requires reverse-engineering the author's intent from raw values. A named constant with a comment carries both the decision and its context for one line of cost.</details>
2. A 2,000-request script has no error handling, and each request fails independently with probability 0.05%. Why is the run likely to fail?
   <details><summary>Answer</summary>The probability that every request succeeds is $(1 - 0.0005)^{2000} \approx e^{-1} \approx 0.37$, so the run fails about 63% of the time. Above roughly 0.05% per request, failure is more likely than success, which is why bounded retry with backoff is mandatory at that volume.</details>
3. When is a boolean the right type, and what should you use otherwise?
   <details><summary>Answer</summary>A boolean is right only when the domain has two nameless states. When the states have names (red and black teams, say), use a union type such as `type Team = "red" | "black"`: it documents itself and the compiler rejects a third value, making the invalid state unrepresentable.</details>
4. Why is unvalidated input from a scraper worse than a crash?
   <details><summary>Answer</summary>A crash is visible at the boundary where it happens. Unvalidated input (a CAPTCHA page, a redesigned layout) produces structurally valid but empty records that surface as a mystery several steps downstream, and silent garbage in a dataset gets used.</details>
5. What signal in a project's history suggests the infrastructure is too large for the timeline, and what is the remedy?
   <details><summary>Answer</summary>Many commits in a single day (20 or more) with none touching the core feature. The remedy is the boring stack that supports the same demo, plus a tracer bullet through the core mechanic before any layer is broadened.</details>
6. Why is a half-adopted tool (TypeScript with `.js` files, or `any` on the important fields) worse than either adopting it fully or removing it?
   <details><summary>Answer</summary>It is a claim the code does not back. It misleads the reader about what the code guarantees, and `any` on the interesting field disables the type system exactly where it would have paid off.</details>
