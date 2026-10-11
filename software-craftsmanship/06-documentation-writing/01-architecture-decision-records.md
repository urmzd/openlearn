<!-- ol:module craft.02 -->
# Architecture decision records

## Overview

| | |
|---|---|
| **Module** | `craft.02` · practice · docs · Pass 1 · 1 to 2 h |
| **You build** | `docs/adr/0001-<slug>.md`: ADR-0001, the record of where your four languages meet and why |
| **Contract** | the ADR template in section 4 (file name, title line, five sections) |
| **Tests** | `course/tests/craft.02/` (what they check: section 4) |
| **Needs** | nothing to call; you write about what you built in [`M03.1`](../../math/03-linear-algebra/01-vectors-matrices-and-matmul.md) · [`L10.0`](../../ml/08-tinyllm/p10-serving/00-your-first-endpoint.md) · [`gw.00`](../../ai-platform-engineering/12-gateway/00-streaming-proxy.md), committed with the habits of [`craft.01`](../08-code-review-and-ci/01-your-repo-and-ci-gate.md) |
| **Used by** | no call site: a practice artifact. Every later ADR (craft.13, craft.14, ops.04) follows this format |
| **Milestone** | [MS-P1](../../paths/course-p01-tracer/milestone.md) |
| **Optional depth** | Michael Nygard, [*Documenting Architecture Decisions*](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) (2011, free); [adr.github.io](https://adr.github.io/) (free); *Software Engineering at Google*, ch. 10 [Documentation](https://abseil.io/resources/swe-book/html/ch10.html) (free) |

## Key Takeaways

- An ADR records **one** decision: the forces at the time, what was chosen, what it costs, and what was rejected. The tests check all five parts are there.
- ADRs are **append-only**. You never rewrite an accepted record; a new ADR supersedes it, and the old one points forward.
- The status and its date say whether the decision still binds and what its authors could have known.
- A decision with no rejected alternative and no listed cost is not a decision record. It is a description.
- ADR-0001 records the decision your whole system rests on: languages meet through process and file boundaries.

## How to work this chapter

```bash
ol start craft.02           # nothing to stub: this module's artifact is a document
ol tests craft.02           # read the twelve checks first
$EDITOR docs/adr/0001-<slug>.md
ol check craft.02           # exit code is the verdict
git add docs/adr && git commit -m "docs(adr): record where the languages meet"
```

---

## 1. Why now

Your tracer crosses language boundaries through files and processes. Python writes a safetensors checkpoint and tokenizer data; the Rust engine reads them and streams tokens over HTTP and SSE (`L10.0`). Your Go gateway checks a key and proxies that stream (`gw.00`). Each choice was made for a reason that is in your head today and in nobody's head in six months. In Pass 7 you will add gRPC between components, and someone (you, or a reviewer) will ask why the gateway still talks to the engine over HTTP. Without a record, that question gets answered by re-arguing it from scratch, or by changing it and rediscovering the cost. This module writes the record now, while the reasons are fresh, as ADR-0001.

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

Here is a complete record for a different decision your tracer already made: the byte tokenizer. Read it once, then follow how each section was produced.

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

Check it against section 4: file `0002-the-tracer-tokenizer-is-raw-utf-8-bytes.md` (number, kebab slug); title `# ADR-0002: ...` matches; five sections in order; status known and dated; Context has 52 words (at least 25 needed); Consequences lists 4 items; Alternatives has 2 rows (the header row does not count); no placeholders. Saved next to an ADR-0001, it passes all twelve checks. Alone, it fails the four that need an ADR-0001 to exist (the file, the numbering from 0001, the Accepted status of 0001, and its topic). Your ADR-0001 is the same shape, about a different decision.

## 4. The artifact and its check

**Path.** `docs/adr/0001-<slug>.md` in your repo, for example `docs/adr/0001-languages-meet-at-process-and-file-boundaries.md`. Commit it with a conventional commit (`docs(adr): ...`, craft.01).

**Template.** Copy this (your repo also has it at `contracts/templates/ADR.md`), then replace every `<...>`:

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

**What ADR-0001 must decide.** How the four languages of your system meet. Write down what you built and why: in process (Python and Rust calling your C kernels), and between processes (the gateway and the engine). Name the alternatives you could have used for each boundary and why you did not.

**Run the check.**

```bash
ol check craft.02
```

prints one line per check and exits 0 only when all pass. A failure prints what to change and the check's WHY.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_adr_0001_exists_with_a_slug` | unit | `docs/adr/0001-<kebab-slug>.md` exists; every file in `docs/adr/` is named `NNNN-<slug>.md` | `ol milestone MS-P1` and later ADRs find records by number |
| `test_numbers_are_unique_and_have_no_gaps` | boundary | numbers run 0001, 0002, ... with no duplicate and no gap | "ADR-0003" names one decision forever (pitfall 3) |
| `test_title_line_matches_the_file_number` | unit | the first line is `# ADR-NNNN: <title>` with the file's own number | the title and the file name never drift apart |
| `test_sections_appear_in_template_order` | unit | exactly one each of Status, Context, Decision, Consequences, Alternatives considered, in that order | every record reads the same way |
| `test_status_is_known_and_dated` | boundary | the status line starts with a known status and carries a real `(YYYY-MM-DD)` date | a reader can tell if the decision still binds (pitfall 4) |
| `test_superseded_points_at_a_later_adr` | boundary | `Superseded by ADR-NNNN` names an ADR that exists and has a larger number | the trail of reasons stays connected (pitfall 1) |
| `test_adr_0001_is_accepted` | unit | ADR-0001 is Accepted (or Superseded by a later ADR) | your system already runs on it |
| `test_no_template_placeholders_left` | boundary | no `<text with spaces>` placeholder and no TODO, TBD, FIXME, or XXX outside code | a copied template is not a record (pitfall 5) |
| `test_every_section_has_substance` | boundary | Context has at least 25 words, Decision 8, Consequences 10, Alternatives 6 | a one-line context explains nothing later (pitfall 2) |
| `test_consequences_name_more_than_one_effect` | unit | Consequences lists at least two items (bullets or table rows) | costs are recorded next to benefits (pitfall 2) |
| `test_alternatives_name_a_rejected_option` | unit | Alternatives considered lists at least one option | nothing rejected means nothing decided |
| `test_adr_0001_records_the_language_boundaries` | unit | ADR-0001 mentions the ABI and HTTP | ADR-0001 is the record of the boundaries every later module builds on |

The section 3 example is the first case: saved as ADR-0002 next to an ADR-0001, it passes all twelve. Each check is also proven against a planted fault in a reference ADR (renamed file, a gap in numbers, a drifted title, swapped sections, an undated status, a dangling supersede, a Proposed ADR-0001, a leftover placeholder, a one-line context, a single consequence, no alternatives, a different topic): each fault fails exactly the check named for it.

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | Editing an accepted ADR when you change your mind | the record no longer says what was known when you decided; reviews argue about history | `test_superseded_points_at_a_later_adr` (write a new ADR and supersede) |
| 2 | Context of one line, consequences of one benefit | a reader cannot tell whether the decision still holds when the forces change | `test_every_section_has_substance`, `test_consequences_name_more_than_one_effect` |
| 3 | Renumbering records, or reusing a number after deleting a draft | links such as "see ADR-0003" now point at a different decision | `test_numbers_are_unique_and_have_no_gaps` |
| 4 | Status without a date, or a status word outside the lifecycle ("Done", "Final") | nobody can order decisions or tell which one is current | `test_status_is_known_and_dated` |
| 5 | Leaving template text in place | the file looks finished in a listing and says nothing | `test_no_template_placeholders_left` |
| 6 | One ADR for several decisions ("Architecture overview") | superseding one part forces rewriting the rest | review: one "We will" per record |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.1` | Python matrix operations used in the tracer's model |
| Back | `L10.0` | the engine's HTTP and SSE surface, the between-process boundary |
| Back | `gw.00` | the gateway, the other side of that boundary |
| Back | `craft.01` | commit the record with a conventional commit; CI keeps `ol check --all --ci` green |
| Forward | `ops.00` | its runbook is a how-to guide, the document type an ADR is not |
| Forward | `craft.13` | the KV format v1 to v2 migration supersedes a format decision with a new ADR |
| Forward | `craft.14` | the API v1 to v2 migration records its deprecation policy as an ADR |
| Forward | `ops.04` | the KV v2 rollout drill requires an ADR before it passes |
| Forward | `craft.09` | the C4 diagrams show the boundaries ADR-0001 explains |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `docs/adr/NNNN-*.md` | MADR (Markdown Any Decision Records) | decision drivers, per-option pros and cons, a template ecosystem | [adr.github.io/madr](https://adr.github.io/madr/) (free) |
| the five-section template | Rust RFCs, Python PEPs, Kubernetes KEPs | public review periods, champions, staged acceptance | `rust-lang/rfcs`, `python/peps`, `kubernetes/enhancements` |
| `ol check craft.02` | adr-tools, Log4brains | CLI to create and supersede records; a rendered, searchable decision log | `npryce/adr-tools`, `thomvaill/log4brains` |
| ADR-0001 | design docs at Google | the design thinking before the decision; the ADR keeps the outcome | *Software Engineering at Google*, ch. 10 (free) |

## Company Relevance

| Company | Practice | Why it matters |
|---|---|---|
| Amazon | six-page narratives, one-way vs two-way door decisions | record the one-way doors; reverse two-way doors freely |
| Google | design docs reviewed before code | decisions are argued in writing, then kept |
| Spotify, ThoughtWorks | ADRs in every service repo | new engineers learn why from the repo itself |
