# Diagramming and Documentation

This book is about the writing that surrounds a system: the diagrams that show how it fits together, and the documents that record why it is built the way it is and how to run it. Code cannot tell the next engineer how its processes talk, why a decision went the way it did, or what to type at 3 a.m. when an alert fires. It is for engineers who already build software and want their designs understood by the next reader, including themselves six months later.

## What you will be able to do

- Draw a system at the right C4 level of zoom (context, container, component) for the question and the audience.
- Keep diagrams as text (D2 and Mermaid) that diff in review, render in CI, and stay true as the code changes.
- Classify any document by its Diátaxis type and write it to that type.
- Write an append-only architecture decision record with its forces, costs, and rejected alternatives.
- Write a runbook that a paged engineer can follow from alert to confirmed recovery.

## Ways through this book

Reading both chapters takes about two weeks at six to eight hours a week, one week per chapter.

- **The whole book, in order** (two weeks): [Chapter 1: Diagramming with the C4 Model](01-diagramming-c4/README.md), [1.1 C4 diagrams in D2](01-diagramming-c4/01-c4-diagrams-in-d2.md), [Chapter 2: Documentation Writing](02-documentation-writing/README.md), [2.1 Architecture decision records](02-documentation-writing/01-architecture-decision-records.md), then [2.2 Runbooks and Diataxis docs](02-documentation-writing/02-runbooks-and-diataxis.md). Along the way, redraw a system you know at C4 levels 1 and 2, then write one ADR and one runbook for it.
- **Drawing a system for a design review** (two to three days): [Chapter 1](01-diagramming-c4/README.md), then [1.1 C4 diagrams in D2](01-diagramming-c4/01-c4-diagrams-in-d2.md). Sketch each view yourself before you read the model answer.
- **Recording a decision** (one to two days): the Diátaxis and decision-record parts of [Chapter 2](02-documentation-writing/README.md), then [2.1 Architecture decision records](02-documentation-writing/01-architecture-decision-records.md).
- **Getting ready for on-call** (one to two days): the Diátaxis part of [Chapter 2](02-documentation-writing/README.md), then [2.2 Runbooks and Diataxis docs](02-documentation-writing/02-runbooks-and-diataxis.md), with [Observability](../systems/04-observability/) in *Systems and Architecture* open for the alerts your runbooks answer.

## Before you start

You need only to have built software and to have a system worth drawing. The examples come from the platform in *[Build Your Own LLM System](../build-an-llm/)*, but each is explained on its own terms. Related books: *[Software Craftsmanship](../software-craftsmanship/)* supplies the principle of treating diagrams and documents like source; *[Infrastructure](../infrastructure/)* deploys the containers you draw; and *[Systems and Architecture](../systems/)* uses these artifacts most, from [System Design](../systems/01-system-design/) to the drills in [Incident Response and Chaos](../systems/05-incident-response-and-chaos/). Every exercise has a model answer in [`course/solutions/docs/`](../../../course/solutions/docs/), and the templates those answers follow are in [`course/contracts/templates/`](../../../course/contracts/templates/).

New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.
