# Software Engineering at Google

The previous chapter, [The Pragmatic Programmer](../01-pragmatic-programmer/), was about individual craft: how one engineer takes responsibility for code, keeps it changeable, and fails cleanly. This chapter widens the lens to organizational craft. Its guide is *Software Engineering at Google* by Titus Winters, Tom Manshreck, and Hyrum Wright, which is [free to read online](https://abseil.io/resources/swe-book), and its central claim is that software engineering is *programming integrated over time*. Writing code that works today is programming. Writing code that must survive for years, be maintained by thousands of engineers, and serve billions of users is engineering, and it demands a different set of habits.

Those habits fall into four areas, and the chapter takes them in order. It starts with culture, because at scale the way people work together is a load-bearing part of the system. It moves to process, the small set of practices (small changes, mandatory review, documentation treated as code, a budget for deprecation) that keep a large codebase changeable. It then looks at testing at scale, where Hyrum's Law reshapes what "correct" means, and ends with the tools and infrastructure that turn all of this from good intentions into the default path.

You do not need to read the Google book cover to cover. It works best as a reference: start with the chapters nearest your current pain, and as you read about each process (code review, CI, deprecation), map it onto how your own team does it today. Make sure you have internalized the test pyramid and Hyrum's Law before you design any public interface; the rest of this book leans on both.

> **Learning objectives**
> - Explain why time and scale, not code, are what separate software engineering from programming.
> - Describe the cultural practices (humility, respect, trust, knowledge sharing, readability) that make large teams work, and why they are engineering concerns.
> - Justify small changes, mandatory code review, docs-as-code, and an explicit deprecation budget as forms of leverage.
> - State Hyrum's Law and apply it when designing an API, planning a migration, or choosing between fakes and mocks.
> - Identify which tools (trunk-based version control, hermetic builds, code search, static analysis, CI, large-scale change tooling) multiply a team's effectiveness, and estimate when investing in them pays off.

## Before you begin

This chapter assumes the individual habits from [The Pragmatic Programmer](../01-pragmatic-programmer/) and some experience working in a team codebase. Two free companions are worth keeping open as you read: Google's [Engineering Practices](https://google.github.io/eng-practices/) guide, which documents how Google does code review, and the [Testing on the Toilet](https://testing.googleblog.com/) series of one-page testing lessons.

## In this chapter

This chapter is a single, self-contained reading. Its ideas are put into practice in later chapters of this book: [The Testing Mentality](../03-testing/) develops the test pyramid and test doubles, [Code Review and CI](../06-code-review-and-ci/) builds a review process and a CI gate, [Releases and Deprecation](../07-releases-and-deprecation/) turns Hyrum's Law into a release and deprecation policy, and [Maintenance](../08-maintenance/) works through migrations and dependency upgrades.

## Programming integrated over time

Start with the definition, because everything else follows from it. Programming is the act of producing code that solves a problem. Software engineering adds three dimensions that programming can ignore. The first is *time*: how long must this code keep working, and how many times will it have to change? A script you run once can be written any way you like; a library that will live for a decade has to absorb upgrades to its language, its dependencies, and its requirements. The second is *scale*: how many people work on the code and how many depend on it. Practices that are harmless for one person become expensive for a hundred and impossible for ten thousand. The third is *trade-offs*: engineering decisions are made under constraints, and a good engineer can say what each choice costs, not just whether it works.

Seen this way, most of Google's lessons are answers to a single question: what does it take for code to stay changeable as time and scale grow? The answers are cultural, procedural, and technical, in roughly that order of importance.

## Culture

It is tempting to treat culture as soft and separate from the real engineering. At scale it is not. A codebase that thousands of people touch is only as healthy as the way those people share knowledge, give feedback, and trust each other.

The book's model for working well on a team is **humility, respect, and trust**, abbreviated HRT. In practice it means hiding nothing, sharing work early rather than polishing it alone, and accepting feedback as information about the code rather than judgment of the person. An engineer who disappears for three months and emerges with a perfect design has usually built the wrong thing; one who shares a rough draft in week one gets corrected cheaply.

**Knowledge sharing** is how a large organization avoids depending on a few people who know how things work. Google invests in several mechanisms: readability reviews, in which certified reviewers check that code follows the language's idioms; go/links, short memorable URLs that make internal resources easy to find; documentation treated as code; and internal tech talks. Each one lowers the cost of learning something someone else already knows.

**Engineering for equity** is the recognition that software built by a narrow group tends to serve a narrow group. Building for everyone means testing with diverse inputs and users and actively avoiding encoding bias into products.

If you are a staff-level engineer or above, culture is part of your job description. If knowledge on your team is siloed, that is your problem to fix, not an unfortunate fact about the people involved.

## Processes

Culture sets expectations; process turns them into routine. The processes Google relies on are not elaborate. What makes them powerful is that they are applied to every change, so their benefits compound.

**Style guides and rules** come first. Consistency beats personal preference, because code is read far more often than it is written. Google's style guides exist to make code *readable by others*, not *writable by you*; a rule that slightly annoys the author but saves every future reader a moment of confusion is a good rule.

**Code review** is mandatory: every change is reviewed before it lands. Reviews are about readability and correctness, not gatekeeping, and they work best when changes are small. A reviewer can genuinely understand a 200-line change list (a CL, Google's term for a change); a 2,000-line one gets skimmed and approved. Review serves three purposes at once: it catches correctness problems, it transfers knowledge between author and reviewer, and it establishes and reinforces the team's norms. The social function matters as much as the technical one. You will build a review process of your own in [Code Review and CI](../06-code-review-and-ci/).

**Documentation** should be treated like code: versioned alongside it, reviewed, tested where possible, and deprecated when it goes stale. The hands-on version of this idea is [Documentation Writing](../12-documentation/) in *Software Engineering*.

**Testing** is organized by size. Small tests are unit tests: one process, fast, deterministic. Medium tests are integration tests that may touch a local database or several processes. Large tests are end-to-end tests that exercise the whole system. The healthy shape is a pyramid, with many small tests, fewer medium ones, and few large ones, because each step up costs more to run and is more likely to flake. [The Testing Mentality](../03-testing/) develops this in depth.

**Deprecation** completes the picture. Removing old systems is as important as building new ones, because every system you keep running costs maintenance, attention, and security exposure. Google's advice is to budget for deprecation explicitly in roadmaps, rather than hoping someone gets to it, and to make it incremental so that each step is safe. [Releases and Deprecation](../07-releases-and-deprecation/) turns this into a concrete policy.

## Testing at scale

The size-based pyramid tells you how many tests of each kind to write. The next question is what makes an individual test good, and what changes when the system under test has many users.

**Unit tests** should be fast, isolated, and deterministic, and they should test behavior, not implementation. A test that asserts on the result of a public method survives a refactoring; a test that asserts on which private helper was called breaks every time the internals change, even when nothing observable did. Google's Testing on the Toilet one-pagers are a good catalogue of these lessons.

**Test doubles** stand in for real dependencies, and they come in three main kinds. A *stub* returns canned answers. A *mock* records calls and lets the test assert on them. A *fake* is a lightweight working implementation, such as an in-memory database. Google prefers fakes. Mock-heavy tests are brittle, because they encode exactly how the code under test talks to its collaborators, and they do not test real behavior, because a mock does whatever you told it to. A fake behaves like the real thing, so a test against it checks something true.

**Larger tests** (integration, end-to-end, and load tests) are slower and more expensive, but they catch the bugs that live at interfaces, which unit tests by construction cannot see. You need them; you just need fewer of them, which is why the pyramid keeps its shape.

All of this is shaped by **Hyrum's Law**, named for one of the book's authors: *with a sufficient number of users of an API, all observable behaviors of your system will be depended on by somebody*. It does not matter what your documentation promises. If a behavior can be observed, some caller will come to rely on it. Suppose your API returns JSON objects with their keys in sorted order, even though the specification says nothing about ordering. Someone, somewhere, will write code that depends on that ordering. When you stop sorting, perhaps because a faster serializer does not, their code breaks, and from their point of view you broke it.

The practical lesson for testing is to test for what you promise, not for what happens to work. The broader lesson is for design: every observable behavior is a potential contract, so expose as little as you can, and expect any change to observable behavior to need a migration plan.

## Tools and infrastructure

Culture and process describe what good engineering looks like. Tools are what make it the path of least resistance. Google's infrastructure is built so that the right thing is also the easy thing.

**Version control and branch management.** Google keeps nearly all of its code in a single monorepo and practices trunk-based development: everyone commits to one main line, and unfinished features are hidden behind feature flags rather than kept on long-lived feature branches. This avoids the painful merges that long branches accumulate and means everyone is always building against the current state of the code.

**Build systems.** Google's build system, released as Bazel, emphasizes reproducible, hermetic builds. A hermetic build depends only on its declared inputs, not on whatever happens to be installed on the machine, so the same inputs always produce the same outputs. That property is what makes aggressive caching safe: build once, cache forever.

**Code search** is critical infrastructure. Every engineer at Google can search all of Google's code instantly, which makes it possible to find every caller of an API before changing it, and to learn from how others solved the same problem.

**Static analysis** covers compiler warnings, linters, and type checkers. Make it automatic and run it on every change. Do not require humans to catch what machines can; reviewer attention is too valuable to spend on formatting and null checks.

**Dependency management** is, in the book's view, the hardest problem in software engineering at scale. The classic example is the diamond dependency problem: your project depends on libraries A and B, both of which depend on library C, but they need incompatible versions of it. Strategies such as minimum version selection, in which the build picks the lowest version that satisfies every requirement, make the outcome predictable rather than accidental.

**Continuous integration** runs in stages. Presubmit checks run before a change lands, post-submit testing runs on the trunk after it lands, and release qualification decides whether a build is fit to ship.

**Large-scale changes (LSCs)** are changes that touch thousands of files, or many repositories, at once: renaming a widely used function, migrating off a deprecated API, or applying a new lint fix everywhere. They need dedicated tools to generate, test, review, and land the change atomically. Without such tools, codebase-wide improvements simply do not happen.

These investments are force multipliers, and the arithmetic is worth doing. Spending one week to make CI 30% faster pays for itself within about a month across 100 engineers, because every one of them waits on CI many times a day. A staff-level engineer should be the one doing that arithmetic and making the case.

## Putting the principles to work

The ideas above are easier to apply if you know when each one is the relevant tool. The table below pairs each principle with the situation that calls for it.

| Principle | When to apply it |
|-----------|------------------|
| Hyrum's Law | Designing APIs, planning migrations, deprecating features |
| Small CLs | Every code review (keep changes under about 200 lines) |
| Test pyramid | Designing the test strategy for a service |
| Fakes over mocks | Writing test doubles, when you want them to exhibit real behavior |
| Deprecation budget | Planning roadmaps, with explicit time set aside for cleanup |
| Hermetic builds | Making CI reproducible, so you can build once and cache forever |
| Large-scale changes | Repository-wide refactors that must land atomically |

## Where this leads

These lessons reach well beyond Google. Within this book, the chapters that follow put them into practice, starting with [The Testing Mentality](../03-testing/). Elsewhere, *Systems Engineering* picks up several threads. System design interviews in [Systems Engineering](../../systems/) are, among other things, a test of whether you would build systems that stay maintainable. [Cloud Native](../../systems/09-cloud-native/) is the practical implementation of CI/CD and build processes, and [Observability](../../systems/11-observability/) supplies what you need to deprecate a system safely: you cannot turn something off with confidence until you can see who still uses it. The docs-as-code idea has its hands-on counterpart in [Documentation Writing](../12-documentation/) in *Software Engineering*. Finally, the software development lifecycle described here is a useful contrast with the LLM development lifecycle covered in [Foundation Models & Architectures](../../ai-engineering/05-foundation-models/) in *AI Engineering*, which explains why building an LLM (growing weights from data) is a different discipline from building conventional software.

You will also see these ideas wherever large engineering organizations operate. At Google they are simply the culture: the book describes it directly, down to readability reviews and Hyrum's Law. Meta's motto of moving fast with stable infrastructure rests on a strong code review culture, testing, and CI. Stripe's reputation for API design depends on backwards compatibility and documentation, which is Hyrum's Law taken seriously. Amazon's emphasis on operational excellence and ownership is the process side of the same argument. And in nearly every company, staff-level and above roles come with the expectation that you will drive this kind of engineering excellence.

### Further reading

- Titus Winters, Tom Manshreck, and Hyrum Wright, *Software Engineering at Google*, [free online](https://abseil.io/resources/swe-book).
- Google, [Engineering Practices](https://google.github.io/eng-practices/), including the code reviewer's and change author's guides.
- Google Testing Blog, [Testing on the Toilet](https://testing.googleblog.com/).

## Key terms

- **Software engineering**: programming integrated over time; writing code that must be maintained across years, many engineers, and many users.
- **HRT**: humility, respect, and trust, the book's model for how engineers should work together.
- **Readability review**: a review by a certified reviewer that checks code follows a language's idioms and style.
- **CL (change list)**: a single proposed change submitted for review; Google's term for what others call a pull request or patch.
- **Test pyramid**: the target shape of a test suite, with many small tests, fewer medium tests, and few large tests.
- **Fake**: a lightweight working implementation of a dependency, preferred over mocks because it exhibits real behavior.
- **Hyrum's Law**: with enough users, every observable behavior of a system will be depended on by somebody.
- **Trunk-based development**: everyone commits to a single main line, with feature flags instead of long-lived branches.
- **Hermetic build**: a build that depends only on its declared inputs, so it is reproducible and safely cacheable.
- **Diamond dependency**: two dependencies of a project require incompatible versions of a shared third dependency.
- **Large-scale change (LSC)**: a change applied across thousands of files or many repositories, landed with dedicated tooling.

## Summary

- Software engineering is programming integrated over time: time, scale, and trade-offs are what distinguish it from programming.
- Culture is an engineering concern. Humility, respect, trust, knowledge sharing, and readability are load-bearing at scale.
- Process is leverage. Small CLs, mandatory review, docs-as-code, and an explicit deprecation budget keep a large codebase changeable.
- Good tests check behavior, not implementation; the suite should keep a pyramid shape, and fakes beat mocks.
- Hyrum's Law governs every API and migration: all observable behavior will eventually be depended on, so test for what you promise and expose as little as you can.
- Tooling (trunk-based development, hermetic builds, code search, static analysis, CI, LSC tooling) makes good practice the default, and investments in it multiply across every engineer.

## Review questions

1. What distinguishes software engineering from programming, according to *Software Engineering at Google*?
   <details><summary>Answer</summary>Time, scale, and trade-offs. Programming produces code that solves a problem; engineering produces code that must keep working and keep changing over years, be maintained by many engineers, and serve many users, with decisions made by weighing what each option costs. Hence "programming integrated over time."</details>

2. Code review at Google serves three purposes. What are they, and why does the book insist reviews be small?
   <details><summary>Answer</summary>Correctness, knowledge transfer, and establishing norms; the social function matters as much as the technical one. Small changes (roughly under 200 lines) are what make genuine review possible: a reviewer can understand a small CL, while a large one tends to be skimmed and approved, which defeats all three purposes.</details>

3. Your API returns JSON keys in sorted order, though the specification does not require it. You switch to a faster serializer that does not sort. What happens, and what principle explains it?
   <details><summary>Answer</summary>Some caller who depended on the ordering breaks. Hyrum's Law: with enough users, every observable behavior will be depended on, regardless of what the documentation promises. The lessons are to test for what you promise, minimize observable behavior, and treat changes to observable behavior as migrations.</details>

4. Why does Google prefer fakes over mocks?
   <details><summary>Answer</summary>A fake is a working lightweight implementation, so tests against it exercise real behavior. Mock-heavy tests encode how the code under test calls its collaborators, so they break on harmless refactors, and they only check that the code does what the mock was told to expect, not that the system actually works.</details>

5. What makes a build hermetic, and why does that matter for CI?
   <details><summary>Answer</summary>A hermetic build depends only on its declared inputs, not on the state of the machine it runs on, so the same inputs always produce the same outputs. That reproducibility is what makes caching safe ("build once, cache forever") and keeps CI results trustworthy.</details>

6. A colleague argues that spending a week speeding up CI is a distraction from feature work. How would you respond for a team of 100 engineers?
   <details><summary>Answer</summary>Do the arithmetic. A 30% speedup in CI saves time for every engineer on every change, many times a day; across 100 engineers, the one-week investment pays for itself within about a month and keeps paying afterward. Build and CI infrastructure are force multipliers.</details>

7. Why does the book say deprecation should be budgeted explicitly?
   <details><summary>Answer</summary>Removing old systems is as important as building new ones, since every system kept alive costs maintenance and attention, but cleanup never wins against new work unless time is set aside for it. Budgeting it in the roadmap, and doing it incrementally, makes it actually happen and keeps each step safe.</details>
