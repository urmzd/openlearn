# Diagramming and Documentation

This book is about the writing that surrounds a system: the diagrams that show how it fits together and the documents that record why it is built the way it is and how to run it. It is for engineers who already build software and want their designs to be understood by the next person who reads them, including themselves six months later.

## How this book is organized

[Diagramming with C4](01-diagramming-c4/README.md) introduces the C4 model's four levels of zoom (context, containers, components, code) and shows how to keep diagrams as text, so they live next to the code and change with it. Its section [C4 diagrams in D2](01-diagramming-c4/01-c4-diagrams-in-d2.md) draws the context and container views of a real LLM platform.

[Documentation writing](02-documentation-writing/README.md) covers the two documents a team reaches for most: the [architecture decision record](02-documentation-writing/01-architecture-decision-records.md), which captures a decision, its alternatives, and its consequences, and the [runbook](02-documentation-writing/02-runbooks-and-diataxis.md), organized with the Diátaxis split between tutorials, how-to guides, reference, and explanation.

## Related books

The examples come from the platform built in *Build Your Own LLM System*; its model answers (ADRs, runbooks, C4 sources) are in [`course/solutions/docs/`](../../../course/solutions/docs/). The incident chapters of *Systems* show runbooks in use.
