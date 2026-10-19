# Architecture decision records

The first end-to-end version of the system, the tracer, already crosses language boundaries through files and processes. Python writes a safetensors checkpoint and tokenizer data. The Rust engine reads them and streams tokens over HTTP and SSE, as built in [Your first endpoint](../../build-an-llm/11-serving/01-your-first-endpoint.md) in *Build Your Own LLM System*. The Go gateway checks an API key and proxies that stream, as built in [Tracer gateway](../../ai-platform-engineering/12-gateway/01-streaming-proxy.md) in *AI Platform Engineering*. Each of those choices was made for a reason that is in someone's head today and in nobody's head in six months.

Later in the build you will add gRPC between components, and someone, an author or a reviewer, will ask why the gateway still talks to the engine over HTTP. Without a record, that question gets answered by re-arguing it from scratch, or by changing the design and rediscovering the cost the hard way. So the record is written now, while the reasons are fresh. It is ADR-0001, and this section teaches you to write it.

You will first learn what a decision record is for and the rules it must satisfy, then build a complete one by hand (ADR-0002, the byte tokenizer), and finally walk through the model answer, [`docs/adr/0001-languages-meet-at-process-and-file-boundaries.md`](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md), section by section: why each sentence is there and where a reviewer would tighten it. The record describes what was built in [Vectors, matrices, row-major layout, and matmul](../../mathematics-for-computing/04-linear-algebra/01-vectors-matrices-and-matmul.md) (in *Mathematics for Computing*), the engine endpoint, and the gateway, and it is committed with the habits of [The repository and its CI gate](../../software-craftsmanship/05-code-review-and-ci/01-your-repo-and-ci-gate.md) from *Software Craftsmanship*. Before you read the solution, write down the forces and the alternatives you would list yourself.

> **Learning objectives**
> - Decide whether a choice deserves an ADR by asking what it would cost to reverse.
> - Write a record with all five sections (Status, Context, Decision, Consequences, Alternatives considered), with costs stated as plainly as benefits.
> - Apply the append-only rule: supersede an accepted record instead of editing it, and keep numbering and status lifecycle intact.
> - Check a record against the template's rules and explain what each rule protects.
> - Explain ADR-0001, the decision the whole system rests on, and how a reviewer would tighten it.

## What a decision record is

An architecture decision is a choice that is expensive to reverse and shapes later work: a language boundary, a wire format, a storage engine, a consistency model. Michael Nygard proposed recording each one in a short text file kept in the repository next to the code it governs. The file answers five questions, always in the same order, as the table shows.

| Section | The question it answers | What a good one contains |
|---|---|---|
| Status | Does this still bind us, and since when? | one of Proposed, Accepted, Deprecated, Superseded by ADR-NNNN, then the date `(YYYY-MM-DD)` |
| Context | What forces made a decision necessary? | the problem, the constraints, the facts true at the time, written so a newcomer understands them |
| Decision | What did we choose? | one decision, in active voice: "We will ..." |
| Consequences | What follows from it, good and bad? | each effect as its own item, the costs as plainly as the benefits |
| Alternatives considered | What else did we weigh, and why not? | every serious option you rejected, with the reason |

All five parts must be there. A decision with no rejected alternative and no listed cost is not a decision record; it is a description.

### When to write one

Write an ADR when a decision is hard to reverse, affects more than one component, or was argued about. Do not write one for a choice you could undo in an afternoon, such as a variable name or a loop order. The test is cost: if reversing it means changing a contract, migrating data, or retraining a model, record it. Amazon's version of the same idea is the distinction between one-way and two-way doors: record the one-way doors, and reverse the two-way doors freely.

### Append-only, numbered, and dated

An ADR is a log entry, not a wiki page. Once accepted, its text stays as it was, because its value is showing what you knew when you decided. When you change your mind, you write a new ADR that explains the new forces, and you change only the old ADR's status line, to `Superseded by ADR-NNNN (date)`. Reading the log in order then tells the whole story, including the dead ends.

Files are named `docs/adr/NNNN-<slug>.md`: a four-digit number that is never reused or skipped, then a short kebab-case slug. The number is the identity ("see ADR-0003"); the slug is for humans scanning a directory listing. The first line repeats the number: `# ADR-0003: <title>`.

A record moves through a status lifecycle. It starts as **Proposed** while it is under review in a pull request, becomes **Accepted** when merged, may become **Deprecated** when the thing it governs is removed, or becomes **Superseded by** a later record when a new decision replaces it. The date in the status line is the date of the last transition, so the status and its date together tell a reader whether the decision still binds and what its authors could have known.

### Consequences have both signs

Every real decision buys something and pays for it. Listing only benefits hides exactly the information a future reader needs: the costs you agreed to carry. When a cost later becomes intolerable, the record shows it was known and accepted, which is the signal to write the superseding ADR rather than to patch around it.

### Where ADRs sit among documents

In the Diátaxis map of documentation (tutorials, how-to guides, reference, explanation), an ADR is **explanation**: it says why, not how. Runbooks, like the one written in [First drill: the engine crashloops](../../systems/05-incident-response-and-chaos/01-first-drill.md), are how-to guides, and contracts are reference. Keep them apart: an ADR that turns into setup instructions stops being read.

## Working one by hand

The rules are easier to see in a finished record. Here is a complete one for a different decision the tracer already made, the byte tokenizer. Read it once, then follow how each section was produced.

```markdown
# ADR-0002: The tracer tokenizer is raw UTF-8 bytes

## Status

Accepted (2026-10-08)

## Context

The tracer must run end to end in Pass 1, before any tokenizer module exists.
The engine is std-only Rust with no tokenizer library: it parses JSON, but
it has no BPE code to apply a tokenizer.json. Python, Rust, and the conformance suite must agree on token
ids exactly, and usage.prompt_tokens must be computable by anyone.

## Decision

We will use the 256 byte values of UTF-8 as the vocabulary: token id i is
byte i, with no merges, no special tokens, and no end-of-sequence token.
config.json declares tl_tokenizer = "bytes".

## Consequences

- Good: encoding and decoding are one line in every language, so ids agree
  by construction.
- Good: prompt_tokens is the byte length of the prompt.
- Bad: sequences are 3 to 4 times longer than with BPE, so the bigram sees
  less context per token.
- Bad: one character can span several tokens, so streaming must hold back
  an incomplete UTF-8 sequence.

## Alternatives considered

| Option | Why not |
|---|---|
| Characters (Unicode code points) | needs a vocabulary file and an unknown-character id |
| BPE with zero merges | the same ids as bytes but needs the BPE machinery first |
```

The title came first, and it names the decision rather than the topic. "Tokenizer" is a topic; "The tracer tokenizer is raw UTF-8 bytes" is a claim someone could disagree with. The status followed directly: the record is merged and the system uses it, so it is `Accepted`, with the date it was merged.

The Context lists the forces, not the answer. Four facts drove the decision: the tokenizer must exist for the first end-to-end run, before any tokenizer module is written; the engine has no tokenizer library; three implementations must agree on ids exactly; and usage must be countable by anyone. Each fact is checkable, and none of them mentions bytes yet. A good context makes a reader expect a decision without announcing it.

The Decision is one sentence starting "We will," followed by the precise details a reader needs to recognize the decision in code: 256 ids, no merges, no special tokens, and the config key.

The Consequences count both columns: two goods and two bads. The second bad, an incomplete UTF-8 sequence while streaming, is the one most likely to bite later, and it is why `formats/tokenizer.md` has an incremental decoding rule. Finally, the Alternatives give each rejected option one row with one reason.

Now check the record against the rules in the next section. The file is `0002-the-tracer-tokenizer-is-raw-utf-8-bytes.md` (number and kebab slug); the title `# ADR-0002: ...` matches it; the five sections appear in order; the status is known and dated; the Context has 52 words (at least 25 are needed); the Consequences list 4 items; the Alternatives have 2 rows (the header row does not count); and there are no placeholders. Saved next to an ADR-0001, it satisfies every rule. On its own, it breaks the four rules that need an ADR-0001 to exist: the file, numbering from 0001, the Accepted status of 0001, and that record's topic. ADR-0001 has the same shape, about a different decision.

## The template and its rules

ADR-0001 lives at `docs/adr/0001-<slug>.md`, here `docs/adr/0001-languages-meet-at-process-and-file-boundaries.md`, committed with a conventional commit (`docs(adr): ...`). It starts from the template in [`course/contracts/templates/ADR.md`](../../../../course/contracts/templates/ADR.md), with every `<...>` replaced:

```markdown
# ADR-0001: <the decision, as a short claim>

## Status

Accepted (<YYYY-MM-DD>)

## Context

<The forces: what the system must do, the constraints, and the facts true
today. At least a paragraph. Do not state the decision here.>

## Decision

We will <the decision, in active voice, with the details a reader needs to
recognize it in the code>.

## Consequences

- Good: <one effect per bullet>
- Bad: <the costs you accept, as plainly as the benefits>

## Alternatives considered

| Option | Why we did not choose it |
|---|---|
| <option> | <reason> |
```

ADR-0001 must decide how the four languages of the system meet: which boundaries are files, which are processes, and whether any language is linked into another's process. It names the alternatives for each boundary and why they were not chosen.

A correct record set satisfies the rules below. The middle column says whether the rule checks one file's structure (unit) or a limit that is easy to fall just outside (boundary).

| Rule | Kind of check | Why it matters |
|---|---|---|
| `docs/adr/0001-<kebab-slug>.md` exists; every file in `docs/adr/` is named `NNNN-<slug>.md` | unit | later ADRs and readers find records by number |
| numbers run 0001, 0002, ... with no duplicate and no gap | boundary | "ADR-0003" names one decision forever |
| the first line is `# ADR-NNNN: <title>` with the file's own number | unit | the title and the file name never drift apart |
| exactly one each of Status, Context, Decision, Consequences, Alternatives considered, in that order | unit | every record reads the same way |
| the status line starts with a known status and carries a real `(YYYY-MM-DD)` date | boundary | a reader can tell if the decision still binds |
| `Superseded by ADR-NNNN` names an ADR that exists and has a larger number | boundary | the trail of reasons stays connected |
| ADR-0001 is Accepted (or Superseded by a later ADR) | unit | the system already runs on it |
| no `<text with spaces>` placeholder and no TODO, TBD, FIXME, or XXX outside code | boundary | a copied template is not a record |
| Context has at least 25 words, Decision 8, Consequences 10, Alternatives 6 | boundary | a one-line context explains nothing later |
| Consequences lists at least two items (bullets or table rows) | unit | costs are recorded next to benefits |
| Alternatives considered lists at least one option | unit | nothing rejected means nothing decided |
| ADR-0001 names the boundaries themselves: the file formats and the wire protocols | unit | ADR-0001 is the record of the boundaries every later part of the system builds on |

The hand-worked ADR-0002 satisfies all of these when saved next to an ADR-0001. Each rule is narrow on purpose. A renamed file, a gap in numbers, a drifted title, swapped sections, an undated status, a dangling supersede, a Proposed ADR-0001, a leftover placeholder, a one-line context, a single consequence, no alternatives, or a different topic each breaks exactly one row.

## Building ADR-0001 step by step

### Reasoning from first principles

Start with the question of what ADR-0001 is about. It is the decision every other part of the system depends on and the one most expensive to reverse: how Python (training), Rust (serving), Go (gateway and durable service), and C (optional kernels) exchange data. Reversing it would change every contract.

Next, the forces. There are four runtimes, each with its own memory model and failure modes. They need byte-exact reproducibility between them: the same weights and the same token ids. And each service must build, test, and restart independently. These forces are what the Context must state.

The forces imply the decision. If runtimes must not share memory or failure, they cannot be linked into one process; they must meet where neither owns the other's memory. That is either a file (safetensors, `tokenizer.json`, fixtures) or a network protocol (HTTP/JSON, SSE, gRPC). That is the Decision.

Then the costs. Every boundary now needs a versioned schema and parity fixtures, and every call is a serialization plus a network hop. Those are the Consequences, stated plainly.

Last, the alternatives. In-process bindings (PyO3, cgo, a C ABI) and making the C kernels the engine's backend were both plausible. Each has a reason for rejection that follows from the forces. Those are the Alternatives.

### The record, annotated

```markdown
# ADR-0001: Languages meet at process and file boundaries
```

The title is a claim someone could dispute ("we should use PyO3"), not a topic ("Language interop"). The slug in the file name is the same claim in kebab case.

```markdown
## Status

Accepted (2026-10-09)
```

The record is Accepted because the tracer already runs on it, and it is dated, so a reader knows it predates the gRPC components and the v2 API added later.

```markdown
## Context

Python trains models, Rust serves them, Go fronts the engine, and optional C exercises explore kernels. The programs need reproducible data exchange without coupling their memory layouts or runtimes.
```

Two sentences, both forces: who does what, and the constraint (reproducible exchange, with no coupled memory or runtimes). It does not say "files" or "HTTP" yet. A reader should be able to predict the decision from the context, not read it there. A reviewer could ask for one more fact here, the independent-restart requirement, which the Decision mentions but the Context does not motivate.

```markdown
## Decision

Python writes weights as safetensors and tokenizers as `tokenizer.json`. Shared fixture files establish parity between Python, Rust, Go, and standalone C exercises. Go and Rust communicate through HTTP/JSON, Server-Sent Events, and internal gRPC. Each service can be built, tested, and restarted independently.

The Rust engine uses `candle-core` and `candle-nn` for tensors. It implements its own model layers. Optional C kernels run in their own test binaries and do not become an engine backend.
```

The decision is stated boundary by boundary, which is what a reader needs to recognize it in the code: the two file formats, the fixture-based parity rule, and the three protocols between Go and Rust. The second paragraph explicitly closes the two in-process doors. Rust's tensors come from a Rust library (no C ABI), and C stays in its own binaries. The template asks for "We will ..."; this record states the decision in the present tense instead, which reads as a description of the current system. Rewriting the first sentence as "We will exchange data between languages only through files and network protocols:" followed by the details would make it unmistakably one decision.

```markdown
## Consequences

- File formats need versioned schemas and parity fixtures.
- HTTP and gRPC make service boundaries observable and independently testable.
- Optional C modules teach layout and numerical behavior without changing core pass gates.
- Each language owns its memory and errors within its process.
```

Four effects, one per bullet. The first is the cost: every format must be versioned and every language checked against fixtures, which is why `course/contracts/formats/` exists. The other three are benefits. A reviewer would ask for a second cost to be stated too: each cross-service call pays for serialization and a network hop. That is acceptable for token streaming, but it is the reason there is no in-process kernel call from Python. Labeling the bullets "Good:" and "Bad:", as the template suggests, would make the balance visible at a glance.

```markdown
## Alternatives considered

| Option | Why we did not choose it |
|---|---|
| In-process language bindings | They would couple ownership and failure behavior across runtimes. |
| A C backend for the Rust engine | It would make optional C exercises a core serving dependency. |
```

Each rejection traces to a force. Bindings break "no coupled runtimes": a panic in one language crashes the other's process, and memory ownership crosses the boundary. A C backend breaks "optional C": a core service would depend on an optional module.

### Why this design

The decision itself trades per-call speed for isolation. Process and file boundaries let each service be tested against fixtures alone, restarted alone, and observed at its boundary. For a system whose hot loop (matmul, attention) lives inside one language (Rust), the boundaries are crossed per request or per checkpoint, not per operation, so the cost is small.

The solution writes one record for the whole boundary rule rather than one per pair of languages. The forces are the same for every pair, so splitting would repeat the context and make superseding one pair ambiguous. This is not the pitfall of cramming several decisions into one record, because there is one rule, applied everywhere.

The sections are short on purpose. The record is read when someone proposes bindings or a shared library, and two sentences of context plus four consequences are enough to answer "why not?" in a minute.

## What can go wrong

Six mistakes account for most broken decision logs, and each maps onto one of the rules above.

The most damaging is **editing an accepted ADR when you change your mind**. The record then no longer says what was known when you decided, and reviews turn into arguments about history. ADR-0001 is dated and Accepted; a change of rule would be a new ADR that supersedes it.

The second is a **one-line context with a single, positive consequence**. A reader cannot tell whether the decision still holds when the forces change. ADR-0001 has two forces in the Context and four consequences, including the schema-and-fixtures cost.

The third is **renumbering records, or reusing a number after deleting a draft**, so that a link such as "see ADR-0003" now points at a different decision. In the solution, `docs/adr/` holds numbers 0001 to 0003, none reused.

The fourth is a **status without a date, or a status word outside the lifecycle** ("Done," "Final"), so nobody can order decisions or tell which one is current. The solution's status is `Accepted (2026-10-09)`.

The fifth is **leaving template text in place**. The file looks finished in a directory listing and says nothing. Every placeholder must be replaced.

The sixth is **one ADR for several decisions**, such as an "Architecture overview" record. Superseding one part then forces rewriting the rest. ADR-0001 avoids this by recording one rule (process and file boundaries) applied to every pair of languages.

> **Warning:** A Proposed ADR-0001 fails the rules even if every section is perfect. The system already runs on the decision, so its record must be Accepted (or Superseded by a later one).

## Where this leads

Every later decision record in the system follows this format. The [KV format v1 to v2 migration](../../software-craftsmanship/07-maintenance/01-kv-format-v1-to-v2.md) supersedes a format decision with a new ADR; the [API v1 to v2 migration](../../software-craftsmanship/07-maintenance/02-api-v1-to-v2.md) records its deprecation policy as an ADR; and the [KV v2 migration drill](../../systems/05-incident-response-and-chaos/05-kv-v2-migration.md) is not done until its ADR is written. The runbook from [First drill: the engine crashloops](../../systems/05-incident-response-and-chaos/01-first-drill.md) is a how-to guide, the document type an ADR is not. The [C4 diagrams in D2](../01-diagramming-c4/01-c4-diagrams-in-d2.md) show the boundaries ADR-0001 explains, and the record is part of [what the system does after the tracer milestone](../../build-an-llm/paths/course-p01-tracer/milestone.md).

Further reading, with the production practice each piece of this section corresponds to:

- **MADR** (Markdown Any Decision Records), the production version of `docs/adr/NNNN-*.md`, adds decision drivers, per-option pros and cons, and a template ecosystem: [adr.github.io/madr](https://adr.github.io/madr/) (free).
- **Rust RFCs, Python PEPs, Kubernetes KEPs** extend the five-section template with public review periods, champions, and staged acceptance: see `rust-lang/rfcs`, `python/peps`, and `kubernetes/enhancements`.
- **adr-tools and Log4brains** automate the rules of this section with a CLI to create and supersede records and a rendered, searchable decision log: see `npryce/adr-tools` and `thomvaill/log4brains`.
- **Design docs at Google** capture the design thinking before the decision, while the ADR keeps the outcome: *Software Engineering at Google*, ch. 10, [Documentation](https://abseil.io/resources/swe-book/html/ch10.html) (free).
- Michael Nygard, [*Documenting Architecture Decisions*](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) (2011, free), the original proposal; and [adr.github.io](https://adr.github.io/) (free).

The practice is widespread in industry. Amazon writes six-page narratives and distinguishes one-way from two-way door decisions. Google reviews design docs before code, so decisions are argued in writing and then kept. Spotify and ThoughtWorks keep ADRs in every service repository, so new engineers learn why from the repository itself.

## Key terms

- **Architecture decision record (ADR)**: a short, numbered, append-only file recording one hard-to-reverse decision with its context, consequences, and rejected alternatives.
- **Force**: a fact or constraint, true at the time of the decision, that made a decision necessary; forces belong in the Context.
- **Supersede**: replace an accepted decision with a new ADR, changing only the old record's status line to point forward.
- **Status lifecycle**: Proposed, Accepted, Deprecated, or Superseded by ADR-NNNN, each with the date of the last transition.
- **One-way door**: a decision that is expensive to reverse, and therefore worth recording.

## Summary

- An ADR records one decision: the forces at the time, what was chosen, what it costs, and what was rejected. All five sections must be present.
- ADRs are append-only. You never rewrite an accepted record; a new ADR supersedes it, and the old one points forward.
- The status and its date say whether the decision still binds and what its authors could have known.
- A decision with no rejected alternative and no listed cost is a description, not a decision record.
- ADR-0001 records the decision the whole system rests on: languages meet through process and file boundaries.

## Review questions

1. Why does the Context not mention safetensors or HTTP?
   <details><summary>Answer</summary>The Context states the forces; the Decision states the choice. If the Context already names the answer, a reader cannot tell whether the forces justify it or were written to fit it.</details>
2. A later stage of the build adds gRPC between gateway and engine. Does that need a superseding ADR?
   <details><summary>Answer</summary>No: gRPC is a process boundary, so it follows ADR-0001, and the record already lists internal gRPC. A new ADR is needed only if the rule changes, for example linking Rust into the Go gateway.</details>
3. Which force rejects in-process bindings, and what concrete failure would they cause?
   <details><summary>Answer</summary>"Without coupling memory layouts or runtimes." With bindings, a panic or segfault in one language takes down the other's process, and memory ownership crosses the boundary, so a bug in one runtime corrupts the other.</details>
4. What would you change to make the Consequences match the template's intent?
   <details><summary>Answer</summary>Mark each bullet Good or Bad and add the unstated cost: serialization and a network hop per cross-service call, and no fast in-process path from Python to Rust or C.</details>
5. Saved alone in an empty `docs/adr/`, the ADR-0002 example breaks four rules. Which ones, and why?
   <details><summary>Answer</summary>The rules that need ADR-0001 to exist: the file `0001-<slug>.md` must exist, numbering must start at 0001 with no gap, ADR-0001 must be Accepted, and ADR-0001 must name the language boundaries. ADR-0002 itself is well formed.</details>
