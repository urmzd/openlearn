# Architecture decision records

## Overview

| | |
|---|---|
| **Module** | `craft.02` · practice · docs · Pass 1 · 1 h reading |
| **Solution** | [`docs/adr/0001-languages-meet-at-process-and-file-boundaries.md`](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md): ADR-0001, the record of where the four languages meet and why |
| **Contract** | the ADR template in section 4 (file name, title line, five sections), also at [`course/contracts/templates/ADR.md`](../../../../course/contracts/templates/ADR.md) |
| **Needs** | nothing to call; the record is about what was built in [`M03.1`](../../mathematics-for-computing/03-linear-algebra/01-vectors-matrices-and-matmul.md) · [`L10.0`](../../../courses/build-an-llm/p10-serving/00-your-first-endpoint.md) · [`gw.00`](../../ai-platform-engineering/12-gateway/00-streaming-proxy.md), committed with the habits of [`craft.01`](../../software-craftsmanship/05-code-review-and-ci/01-your-repo-and-ci-gate.md) |
| **Used by** | no call site: a practice artifact. Every later ADR (craft.13, craft.14, ops.04) follows this format, and ADR-0001 is part of [what the system does after Pass 1](../../../courses/build-an-llm/paths/course-p01-tracer/milestone.md) |
| **Optional depth** | Michael Nygard, [*Documenting Architecture Decisions*](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) (2011, free); [adr.github.io](https://adr.github.io/) (free); *Software Engineering at Google*, ch. 10 [Documentation](https://abseil.io/resources/swe-book/html/ch10.html) (free) |

## Key Takeaways

- An ADR records **one** decision: the forces at the time, what was chosen, what it costs, and what was rejected. All five parts must be there.
- ADRs are **append-only**. You never rewrite an accepted record; a new ADR supersedes it, and the old one points forward.
- The status and its date say whether the decision still binds and what its authors could have known.
- A decision with no rejected alternative and no listed cost is not a decision record. It is a description.
- ADR-0001 records the decision the whole system rests on: languages meet through process and file boundaries.

## How to read this chapter

Sections 1 to 3 explain what a decision record is for and build one (ADR-0002, the byte tokenizer) step by step. Section 4 fixes the template and the rules a record must satisfy. Section 5 walks through the solution, ADR-0001, section by section: why each sentence is there, and where a reviewer would tighten it. Before reading it, write down the forces and the alternatives you would list yourself.

---

## 1. Why now

The tracer crosses language boundaries through files and processes. Python writes a safetensors checkpoint and tokenizer data; the Rust engine reads them and streams tokens over HTTP and SSE (`L10.0`). The Go gateway checks a key and proxies that stream (`gw.00`). Each choice was made for a reason that is in someone's head today and in nobody's head in six months. In Pass 7 you will add gRPC between components, and someone (an author, or a reviewer) will ask why the gateway still talks to the engine over HTTP. Without a record, that question gets answered by re-arguing it from scratch, or by changing it and rediscovering the cost. The record is written now, while the reasons are fresh, as ADR-0001.

## 2. Principles

**What a decision record is.** An architecture decision is a choice that is expensive to reverse and shapes later work: a language boundary, a wire format, a storage engine, a consistency model. Michael Nygard proposed recording each one in a short text file kept in the repository next to the code it governs. The file answers five questions, always in the same order:

| Section | The question it answers | What a good one contains |
|---|---|---|
| Status | Does this still bind us, and since when? | one of Proposed, Accepted, Deprecated, Superseded by ADR-NNNN, then the date `(YYYY-MM-DD)` |
| Context | What forces made a decision necessary? | the problem, the constraints, the facts true at the time, written so a newcomer understands them |
| Decision | What did we choose? | one decision, in active voice: "We will ..." |
| Consequences | What follows from it, good and bad? | each effect as its own item, the costs as plainly as the benefits |
| Alternatives considered | What else did we weigh, and why not? | every serious option you rejected, with the reason |

**When to write one.** Write an ADR when a decision is hard to reverse, affects more than one component, or was argued about. Do not write one for a choice you could undo in an afternoon (a variable name, a loop order). The test is cost: if reversing it means changing a contract, migrating data, or retraining a model, record it.

**Append-only.** An ADR is a log entry, not a wiki page. Once accepted, its text stays as it was, because its value is showing what you knew when you decided. When you change your mind, you write a new ADR that explains the new forces, and change only the old ADR's status line to `Superseded by ADR-NNNN (date)`. Reading the log in order then tells the whole story, including the dead ends.

**Numbering.** Files are `docs/adr/NNNN-<slug>.md`: a four-digit number that is never reused or skipped, then a short kebab-case slug. The number is the identity ("see ADR-0003"); the slug is for humans scanning a directory listing. The first line repeats the number: `# ADR-0003: <title>`.

**Status lifecycle.** A record starts as **Proposed** while it is under review (in a pull request), becomes **Accepted** when merged, may become **Deprecated** when the thing it governs is removed, or **Superseded by** a later record when a new decision replaces it. The date in the status line is the date of the last transition.

**Consequences are both signs.** Every real decision buys something and pays for it. Listing only benefits hides exactly the information a future reader needs: the costs you agreed to carry. When a cost later becomes intolerable, the record shows it was known and accepted, which is the signal to write the superseding ADR rather than to patch around it.

**Where ADRs sit among documents.** In the Diataxis map of documentation (tutorials, how-to guides, reference, explanation), an ADR is **explanation**: it says why, not how. Runbooks (`ops.00`) are how-to guides; contracts are reference. Keep them apart: an ADR that turns into setup instructions stops being read.

## 3. Worked example by hand

Here is a complete record for a different decision the tracer already made: the byte tokenizer. Read it once, then follow how each section was produced.

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

How it was written, step by step:

1. **Name the decision in the title.** Not "Tokenizer" (a topic) but "The tracer tokenizer is raw UTF-8 bytes" (a claim someone could disagree with).
2. **Status.** It is merged and the system uses it, so `Accepted`, with the date it was merged.
3. **Context: list the forces, not the answer.** Four facts drove it: it must exist in Pass 1, the engine has no tokenizer library, three implementations must agree exactly, and usage must be countable. Each fact is checkable. None of them mentions bytes yet: the context should make a reader expect a decision, not announce it.
4. **Decision: one sentence starting "We will".** Then the precise details a reader needs to recognize the decision in code: 256 ids, no merges, no special tokens, and the config key.
5. **Consequences: count both columns.** Two goods and two bads. The second bad (incomplete UTF-8 while streaming) is the one most likely to bite later; it is why `formats/tokenizer.md` has an incremental decoding rule.
6. **Alternatives: each with one reason.** Two rejected options, each in one row.

Check it against section 4: file `0002-the-tracer-tokenizer-is-raw-utf-8-bytes.md` (number, kebab slug); title `# ADR-0002: ...` matches; five sections in order; status known and dated; Context has 52 words (at least 25 needed); Consequences lists 4 items; Alternatives has 2 rows (the header row does not count); no placeholders. Saved next to an ADR-0001, it satisfies every rule. Alone, it breaks the four that need an ADR-0001 to exist (the file, the numbering from 0001, the Accepted status of 0001, and its topic). ADR-0001 is the same shape, about a different decision.

## 4. The artifact

**Path.** `docs/adr/0001-<slug>.md`, here `docs/adr/0001-languages-meet-at-process-and-file-boundaries.md`, committed with a conventional commit (`docs(adr): ...`, craft.01).

**Template.** From [`course/contracts/templates/ADR.md`](../../../../course/contracts/templates/ADR.md), with every `<...>` replaced:

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

**What ADR-0001 must decide.** How the four languages of the system meet: which boundaries are files, which are processes, and whether any language is linked into another's process. It names the alternatives for each boundary and why they were not chosen.

### What a correct solution must handle

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| `docs/adr/0001-<kebab-slug>.md` exists; every file in `docs/adr/` is named `NNNN-<slug>.md` | unit | later ADRs and readers find records by number |
| numbers run 0001, 0002, ... with no duplicate and no gap | boundary | "ADR-0003" names one decision forever (pitfall 3) |
| the first line is `# ADR-NNNN: <title>` with the file's own number | unit | the title and the file name never drift apart |
| exactly one each of Status, Context, Decision, Consequences, Alternatives considered, in that order | unit | every record reads the same way |
| the status line starts with a known status and carries a real `(YYYY-MM-DD)` date | boundary | a reader can tell if the decision still binds (pitfall 4) |
| `Superseded by ADR-NNNN` names an ADR that exists and has a larger number | boundary | the trail of reasons stays connected (pitfall 1) |
| ADR-0001 is Accepted (or Superseded by a later ADR) | unit | the system already runs on it |
| no `<text with spaces>` placeholder and no TODO, TBD, FIXME, or XXX outside code | boundary | a copied template is not a record (pitfall 5) |
| Context has at least 25 words, Decision 8, Consequences 10, Alternatives 6 | boundary | a one-line context explains nothing later (pitfall 2) |
| Consequences lists at least two items (bullets or table rows) | unit | costs are recorded next to benefits (pitfall 2) |
| Alternatives considered lists at least one option | unit | nothing rejected means nothing decided |
| ADR-0001 names the boundaries themselves: the file formats and the wire protocols | unit | ADR-0001 is the record of the boundaries every later module builds on |

The section 3 example satisfies all of these when saved as ADR-0002 next to an ADR-0001. Each rule is narrow on purpose: a renamed file, a gap in numbers, a drifted title, swapped sections, an undated status, a dangling supersede, a Proposed ADR-0001, a leftover placeholder, a one-line context, a single consequence, no alternatives, or a different topic each breaks exactly one row.

## 5. The solution, step by step

### 5.1 From first principles

1. **What decision is ADR-0001 about?** The one every module depends on and that is most expensive to reverse: how Python (training), Rust (serving), Go (gateway, durable), and C (optional kernels) exchange data. Reversing it would change every contract.
2. **What are the forces?** Four runtimes with their own memory models and failure modes; the need for byte-exact reproducibility between them (the same weights, the same token ids); independent build, test, and restart of each service. These forces are what the Context must state.
3. **What do the forces imply?** If runtimes must not share memory or failure, they cannot be linked into one process; they must meet where neither owns the other's memory: a file (safetensors, `tokenizer.json`, fixtures) or a network protocol (HTTP/JSON, SSE, gRPC). That is the Decision.
4. **What does it cost?** Every boundary now needs a versioned schema and parity fixtures, and a call is a serialization plus a network hop. Those are the Consequences to state plainly.
5. **What else was plausible?** In-process bindings (PyO3, cgo, a C ABI) and making the C kernels the engine's backend. Each has a reason to reject it that follows from step 2. Those are the Alternatives.

### 5.2 The record, annotated

```markdown
# ADR-0001: Languages meet at process and file boundaries
```

The title is a claim someone could dispute ("we should use PyO3"), not a topic ("Language interop"). The slug in the file name is the same claim in kebab case.

```markdown
## Status

Accepted (2026-10-09)
```

Accepted, because the Pass 1 system already runs on it; dated, so a reader knows it predates gRPC (Pass 7) and the v2 API (Pass 11).

```markdown
## Context

Python trains models, Rust serves them, Go fronts the engine, and optional C exercises explore kernels. The programs need reproducible data exchange without coupling their memory layouts or runtimes.
```

Two sentences, both forces: who does what, and the constraint (reproducible exchange, no coupled memory or runtimes). It does not say "files" or "HTTP" yet: a reader should be able to predict the decision from the context, not read it there. A reviewer could ask for one more fact here, the independent restart requirement, which the Decision mentions but the Context does not motivate.

```markdown
## Decision

Python writes weights as safetensors and tokenizers as `tokenizer.json`. Shared fixture files establish parity between Python, Rust, Go, and standalone C exercises. Go and Rust communicate through HTTP/JSON, Server-Sent Events, and internal gRPC. Each service can be built, tested, and restarted independently.

The Rust engine uses `candle-core` and `candle-nn` for tensors. It implements its own model layers. Optional C kernels run in their own test binaries and do not become an engine backend.
```

The decision is stated by boundary, which is what a reader needs to recognize it in the code: the two file formats, the fixture-based parity rule, the three protocols between Go and Rust. The second paragraph closes the two in-process doors explicitly: Rust's tensors come from a Rust library (no C ABI), and C stays in its own binaries. The template asks for "We will ..."; this record states the decision in the present tense instead, which reads as a description of the current system. Rewriting the first sentence as "We will exchange data between languages only through files and network protocols:" followed by the details would make it unmistakably one decision.

```markdown
## Consequences

- File formats need versioned schemas and parity fixtures.
- HTTP and gRPC make service boundaries observable and independently testable.
- Optional C modules teach layout and numerical behavior without changing core pass gates.
- Each language owns its memory and errors within its process.
```

Four effects, one per bullet. The first is the cost: every format must be versioned and every language checked against fixtures, which is why `course/contracts/formats/` exists. The other three are benefits. A reviewer would ask for the second cost to be stated too: each cross-service call pays serialization and a network hop, which is acceptable for token streaming but is the reason an in-process kernel call is not possible from Python. Labeling bullets "Good:" and "Bad:" as the template suggests makes the balance visible at a glance.

```markdown
## Alternatives considered

| Option | Why we did not choose it |
|---|---|
| In-process language bindings | They would couple ownership and failure behavior across runtimes. |
| A C backend for the Rust engine | It would make optional C exercises a core serving dependency. |
```

Each rejection traces to a force: bindings break "no coupled runtimes" (a panic in one language crashes the other's process, and ownership crosses the boundary); a C backend breaks "optional C" (a core service would depend on an optional module).

### 5.3 Why this solution

- **The decision itself.** Process and file boundaries trade per-call speed for isolation: each service can be tested against fixtures alone, restarted alone, and observed at its boundary. For a system whose hot loop (matmul, attention) lives inside one language (Rust), the boundaries are crossed per request or per checkpoint, not per operation, so the cost is small.
- **One record for the whole boundary rule, not one per pair.** The forces are the same for every pair of languages; splitting would repeat the context and make superseding one pair ambiguous. Pitfall 6 (several decisions in one record) does not apply: there is one rule, applied everywhere.
- **Short sections.** The record is read when someone proposes bindings or a shared library; two sentences of context and four consequences are enough to answer "why not?" in a minute.

### 5.4 Explain it back

1. Why does the Context not mention safetensors or HTTP?
   <details><summary>Answer</summary>The Context states the forces; the Decision states the choice. If the Context already names the answer, a reader cannot tell whether the forces justify it or were written to fit it.</details>
2. Pass 7 adds gRPC between gateway and engine. Does that need a superseding ADR?
   <details><summary>Answer</summary>No: gRPC is a process boundary, so it follows ADR-0001, and the record already lists internal gRPC. A new ADR is needed only if the rule changes, for example linking Rust into the Go gateway.</details>
3. Which force rejects in-process bindings, and what concrete failure would they cause?
   <details><summary>Answer</summary>"Without coupling memory layouts or runtimes." With bindings, a panic or segfault in one language takes down the other's process, and memory ownership crosses the boundary, so a bug in one runtime corrupts the other.</details>
4. What would you change to make the Consequences match the template's intent?
   <details><summary>Answer</summary>Mark each bullet Good or Bad and add the unstated cost: serialization and a network hop per cross-service call, and no fast in-process path from Python to Rust or C.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Editing an accepted ADR when you change your mind | the record no longer says what was known when you decided; reviews argue about history | the record is dated and Accepted; a change of rule would be a new ADR that supersedes it |
| 2 | Context of one line, consequences of one benefit | a reader cannot tell whether the decision still holds when the forces change | two forces in the Context, four consequences including the schema-and-fixtures cost |
| 3 | Renumbering records, or reusing a number after deleting a draft | links such as "see ADR-0003" now point at a different decision | numbers 0001 to 0003 in `docs/adr/`, none reused |
| 4 | Status without a date, or a status word outside the lifecycle ("Done", "Final") | nobody can order decisions or tell which one is current | `Accepted (2026-10-09)` |
| 5 | Leaving template text in place | the file looks finished in a listing and says nothing | every placeholder replaced |
| 6 | One ADR for several decisions ("Architecture overview") | superseding one part forces rewriting the rest | one rule (process and file boundaries) applied to every pair of languages |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.1` | Python matrix operations used in the tracer's model |
| Back | `L10.0` | the engine's HTTP and SSE surface, the between-process boundary |
| Back | `gw.00` | the gateway, the other side of that boundary |
| Back | `craft.01` | commit the record with a conventional commit; CI checks the docs with everything else |
| Forward | `ops.00` | its runbook is a how-to guide, the document type an ADR is not |
| Forward | `craft.13` | the KV format v1 to v2 migration supersedes a format decision with a new ADR |
| Forward | `craft.14` | the API v1 to v2 migration records its deprecation policy as an ADR |
| Forward | `ops.04` | the KV v2 rollout drill requires an ADR before it is done |
| Forward | `craft.09` | the C4 diagrams show the boundaries ADR-0001 explains |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `docs/adr/NNNN-*.md` | MADR (Markdown Any Decision Records) | decision drivers, per-option pros and cons, a template ecosystem | [adr.github.io/madr](https://adr.github.io/madr/) (free) |
| the five-section template | Rust RFCs, Python PEPs, Kubernetes KEPs | public review periods, champions, staged acceptance | `rust-lang/rfcs`, `python/peps`, `kubernetes/enhancements` |
| the rules of section 4 | adr-tools, Log4brains | CLI to create and supersede records; a rendered, searchable decision log | `npryce/adr-tools`, `thomvaill/log4brains` |
| ADR-0001 | design docs at Google | the design thinking before the decision; the ADR keeps the outcome | *Software Engineering at Google*, ch. 10 (free) |

## Company Relevance

| Company | Practice | Why it matters |
|---|---|---|
| Amazon | six-page narratives, one-way vs two-way door decisions | record the one-way doors; reverse two-way doors freely |
| Google | design docs reviewed before code | decisions are argued in writing, then kept |
| Spotify, ThoughtWorks | ADRs in every service repo | new engineers learn why from the repo itself |
