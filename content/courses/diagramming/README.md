# Diagramming and Documentation

This book is about the writing that surrounds a system: the diagrams that show how it fits together, and the documents that record why it is built the way it is and how to run it. Code tells the next engineer what the system does. It cannot tell them which processes exist and how they talk, why the gateway speaks HTTP to the engine instead of linking it in, or what to type at 3 a.m. when the first-token latency budget is burning. Those answers live in diagrams, decision records, and runbooks, or they live nowhere.

The book is for engineers who already build software and want their designs to be understood by the next person who reads them, including themselves six months later. It is short, two chapters, and practical: every idea is applied to a real system, and every artifact it asks you to produce has a model answer you can compare against.

## What you will be able to do

By the end of the book you will be able to:

- Draw a system at the right C4 level of zoom (context, container, component) for the question and the audience, and leave code-level detail to generated output.
- Keep diagrams as text (D2 and Mermaid) that diff in review, compile in CI, and stay true as the code changes.
- Classify any document by its Diátaxis type and write it to that type, instead of mixing tutorial, how-to, reference, and explanation on one page.
- Write an append-only architecture decision record with its forces, costs, and rejected alternatives.
- Write a runbook that a paged engineer can follow from alert to confirmed recovery.

## How this book is organized

[Diagramming with the C4 Model](01-diagramming-c4/README.md) introduces C4's four levels of zoom, compares the text-to-diagram tools, shows how to pick a diagram type (sequence, state machine, entity-relationship) by the question it answers, and explains how to keep diagrams embedded and living so they do not rot. Its running example is Streamflow, an event-driven order platform. The chapter's section, [C4 diagrams in D2](01-diagramming-c4/01-c4-diagrams-in-d2.md), then draws the context, container, and engine-component views of a real LLM platform and explains why every box and arrow sits at the level it does.

[Documentation Writing](02-documentation-writing/README.md) treats documentation as code: versioned, reviewed, tested, and deprecated on purpose. It introduces the Diátaxis framework, the mechanics of clear sentences, and the documents a senior engineer owns. Its two sections cover the two documents a team reaches for most. [Architecture decision records](02-documentation-writing/01-architecture-decision-records.md) captures a decision, its alternatives, and its consequences in an immutable, numbered file, working one record by hand and annotating the platform's ADR-0001. [Runbooks and Diataxis docs](02-documentation-writing/02-runbooks-and-diataxis.md) builds a goal-sorted documentation index and one runbook per paging alert.

If you want to pace your reading, the [reading plan](study-plan.md) suggests a two-week schedule.

## Prerequisites and related books

You need no special background beyond having built software and having a system worth drawing. The examples come from the platform built in *[Build Your Own LLM System](../build-an-llm/)*, and the book is most rewarding if you have built at least its first end-to-end version, but every example is explained on its own terms.

Several books pair naturally with this one. *[Software Craftsmanship](../software-craftsmanship/)* supplies the principle behind treating diagrams and documents like source, and its migration chapters write the later decision records. *[Infrastructure](../infrastructure/)* is where the containers you draw are deployed, including the C4 deployment view. *[Systems](../systems/)* uses these artifacts most heavily: [System Design](../systems/01-system-design/) is live diagramming, [Observability](../systems/04-observability/) defines the alerts the runbooks answer, and every drill in [Incident Response and Chaos](../systems/05-incident-response-and-chaos/) ends in a runbook or postmortem.

## Conventions

Diagrams appear in two forms. Canonical architecture diagrams are written in D2, with the `.d2` source and the rendered `.svg` committed side by side; inline diagrams use Mermaid fenced blocks (```` ```mermaid ````), which render natively on GitHub. Code and document listings use fenced blocks with a language tag (`d2`, `markdown`, `bash`). Where a listing is abbreviated, a comment or `...` marks the omission.

Each section opens with its learning objectives, works an example by hand before presenting the model answer, and closes with key terms, a summary, and review questions. Answers are hidden in collapsible blocks; try each question before you open it. Call-outs marked **Warning** flag mistakes that reviewers see often, and **Tip** call-outs give a shortcut worth remembering.

The model answers (the C4 sources, ADRs, the documentation index, and the runbooks) live in [`course/solutions/docs/`](../../../course/solutions/docs/), and the templates they follow are in [`course/contracts/templates/`](../../../course/contracts/templates/).
