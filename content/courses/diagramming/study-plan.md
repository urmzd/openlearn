# Reading Plan

This book is short enough to read in two weeks at six to eight hours a week, with time left over to apply each idea to a system you work on. The plan below gives one week to each chapter. The rhythm that works best is to read a chapter's opening early in the week, practice on your own system in the middle of the week, and finish with the chapter's sections, which apply the ideas to the LLM platform from *[Build Your Own LLM System](../build-an-llm/)*. The book pairs well with *[Software Craftsmanship](../software-craftsmanship/)*, which supplies the principle of treating artifacts like source, and *[Infrastructure](../infrastructure/)*, where the systems you draw are deployed. Streamflow, the event-driven order platform, is the running example throughout.

## Week 1: Diagramming and the C4 model

Begin with [Diagramming with the C4 Model](01-diagramming-c4/README.md). On the first day, read about C4's four levels (context, container, component, code) and the map analogy behind them, with [c4model.com](https://c4model.com/) open beside you. On the second day, install `d2`, render the chapter's Streamflow diagrams from their sources, then edit one `.d2` file and look at the change with `git diff`; the [D2 documentation](https://d2lang.com/) covers anything the chapter does not. On the third day, work through the Mermaid diagrams (sequence, state, entity-relationship) and practice choosing a diagram by the question it answers, using the [Mermaid documentation](https://mermaid.js.org/).

Spend the last two days of the week on your own system. Redraw a system you know at levels 1 and 2, then set up rendering in CI so the diagrams stay alive. Close the week with [C4 diagrams in D2](01-diagramming-c4/01-c4-diagrams-in-d2.md), sketching each view yourself before you read the solution's.

## Week 2: Documentation and technical writing

Move on to [Documentation Writing](02-documentation-writing/README.md). On the first day, learn the four Diátaxis types (tutorial, how-to, reference, explanation) and why they must not be mixed, with [diataxis.fr](https://diataxis.fr/) as the reference. On the second day, take Google's [Technical Writing One](https://developers.google.com/tech-writing/one) and apply its rules (active voice, bottom line up front, tables for comparisons) to a document you already own. On the third day, read about docs-as-code: versioning, review, tested samples in CI, and deprecation, which chapter 10 of *Software Engineering at Google*, [Documentation](https://abseil.io/resources/swe-book/html/ch10.html), covers in depth.

Use the last two days to write one ADR and one runbook for a real system, guided by [Architecture decision records](02-documentation-writing/01-architecture-decision-records.md) and [Runbooks and Diataxis docs](02-documentation-writing/02-runbooks-and-diataxis.md). The [ADR site](https://adr.github.io/) and the [Write the Docs guide](https://www.writethedocs.org/guide/) are good companions while you write.
