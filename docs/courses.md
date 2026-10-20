# Course sources and publication

Each publication is a book, field guide, or reading path. The library has six primary books: Mathematics, Programming and Algorithms, Software Engineering, Data Engineering, AI Engineering, and Systems Engineering. It owns its chapters: every chapter lives in exactly one course directory, and other books cross-reference it rather than include it. Every manifest lives at `content/courses/<id>/openlearn.json`. Its paths are relative to the repository root. The root `openlearn.json` lists the available manifests.

## Teaching principle

Code generation is becoming easier; understanding, reasoning, testing, and validation remain the reader's responsibility. A book should make those skills observable. Explain the first principles and show the worked solution, with pauses where readers can commit to a prediction, defend a choice, or challenge an assumption before seeing the explanation.

When writing or revising a section:

1. State the problem, assumptions, constraints, and invariant. Derive the mechanism with a small example before introducing implementation detail.
2. Show the relevant code or derivation and explain how each important step follows from those principles.
3. Include a reasoning check: predict an output, explain a tradeoff, find a counterexample, or describe what breaks when an assumption changes. Supply a worked answer that explains why.
4. Connect the claim to evidence: a test, boundary case, calculation, trace, or controlled failure. Explain what the check establishes and what it leaves open. A conceptual question can ask readers to design the check without executing code.
5. Close the loop with validation: does the observed behavior meet the stated requirement? Ask readers to apply the idea to a changed case so recalling the answer alone is insufficient.

Do not equate reading completion, reproduced code, or passing a narrow test suite with mastery. Describe reference solutions in terms of their constraints and tradeoffs rather than calling them universally optimal. The reader-facing method lives in [How to read these books](../content/references/how-to-read.md).

## Key terms and concept previews

Each section ends with a `## Key terms` list, one `- **Term**: definition` per line. A chapter opening that teaches ideas of its own, such as a survey of the subject, has one too. These lists are the library's glossary. On the website, when a section uses a term that another section defines, its first use becomes a dotted link. Tapping it, or resting the mouse on it, previews the definition and links to the section that teaches it. The section also opens with a collapsed **Concepts from other sections** list, and the PDF prints the same list under each section's opening paragraph. A section can therefore be read on its own: it defines briefly what it borrows and points to the full treatment. Each book also gets a generated A to Z glossary page, linked from its menu and front page. GitHub shows the Markdown as written.

Write Key terms so the previews work:

- Define a term where it is taught, once. Another section that uses it does not repeat the definition; the website links it.
- Make each definition stand alone in one or two sentences: what it is, then what it is for or why it matters. A preview appears in a different section, so avoid "this", "above", and "the next section".
- Name a concept the way readers write it. `Eigenvector / eigenvalue` and `Server-sent events (SSE)` each give two names that link.
- Put the term in the title of the section that teaches it when you can. When several sections define the same term, the website prefers the same chapter, then the section whose title names the term, then the same book, then the more foundational book. Field guides are never linked from books.
- Multi-word names, acronyms, and hyphenated or numbered names link automatically. Single words that are also plain English ("batch", "identity", "kernel") do not, because they mean different things in different books. The allowlist and the exceptions live in `apps/web/scripts/concepts.mjs`.

Links are never added inside code, math, headings, existing links, bold or italic text, or the Key terms list itself.

## Ownership

Put a book's front page (`README.md`) and its chapters in `content/courses/<id>/`, one directory per chapter, one file per section:

```
<book>/
├── README.md              the Start here page
├── 01-<chapter>/
│   ├── README.md          the chapter opening
│   ├── 01-<section>.md    section 1.1
│   ├── 02-<section>/      section 1.2, a folder when it has pages of its own
│   │   ├── README.md
│   │   └── 01-<page>.md   page 1.2.1
│   └── 03-problem-set.md  problem sets are the last sections
└── 02-<chapter>/
```

Number folders and files 01, 02, ... with no gaps. Folders named `code`, `diagrams`, `assets`, or `images` hold supporting files, not sections. A chapter that another book needs stays where it is and is linked, not copied or listed in the other manifest. Reading guides (`computer-science`, `research`) own no chapters; they recommend an order across books. Source attribution lives in `content/references/sources.md`, the one appendix every book shares. The `docs/` directory describes the platform, not its learning content.

`course/` holds the reference solutions (`course/solutions/`) and interface contracts (`course/contracts/`) that the AI Engineering model-building chapters quote and explain. `practice/` holds executable prediction puzzles and data-structure implementations. Their explanations and worked answers belong to Programming and Algorithms. These are supporting directories, not competing course outlines. A course declares the solution files to ship in its portable pack with `runtime`.

## Manifest

The `openlearn.json` manifest defines publication for the web app, PDF, and source pack:

| Field | Meaning |
|---|---|
| `id`, `title` | Stable course identity and display name |
| `kind` | `book`, `field-guide`, or `reading-path`, used to group the library |
| `start` | Curriculum and study-plan documents, in order |
| `tagline` | Optional short course description for the web landing page |
| `tracks` | The chapters, in order. Each entry is one chapter directory (or file). An optional `part` groups consecutive chapters in the reader and PDF. Mark appendices with `"appendix": true` and a closing note with `"unnumbered": true` |
| `paths`, `featured` | Optional staged route; its stage order takes precedence over track order in the PDF |
| `reference` | Source register, contracts, or other supporting trees |
| `runtime` | Repository files/trees (such as solution code) included in portable packs |

Every course is a reading course with checks of reasoning and understanding. Chapters show and explain the solution; readers predict, reason, test, and validate. Nothing is graded.

## The Start here page

A book's `README.md` is its landing page. Keep it short and practical: one paragraph on what the book is and who it is for, **What you will be able to do** (three to five outcomes), **Ways through this book** (one pace sentence, then two to four goal-based routes that name chapters and sections by number), and **Before you start** (prerequisites and related books). Do not list the chapters: the website inserts numbered chapters and section links after the opening paragraph, with Start or Continue reading and a PDF download link when a verified published asset exists. There are no separate study plans; pacing lives in the routes. How sections, review questions, and solutions work is explained once, in [How to read these books](../content/references/how-to-read.md).

## Numbering

Chapter numbers come from the order of `tracks`: chapters are 1, 2, 3, ... and appendices A, B, ... Section numbers come from the order of files inside the chapter: its `README.md` is the chapter itself, the next file or folder is section 1.1, and the pages inside a section folder are 1.1.1, 1.1.2, ... Headings in the Markdown carry no numbers. The website, the PDF, and the course landing page add them ("Chapter 5: Durable Orchestration and Workers", "5.3 The task queue", "Appendix A: Capstones"), so reordering a book is a rename, not an edit to every heading.

The website groups the library into Books, Field guides, and Reading paths, alphabetically within each group. Inside each course, chapters follow the manifest's publication order; auxiliary references are grouped separately. It does not expose source tracks as a competing lesson catalog. A chapter used by only one course has explicit previous/next links. A shared chapter links back to the courses that teach it, so pagination cannot silently cross into another course.

## One publishing pipeline

`scripts/course_catalog.py` resolves every course, validates its paths, and produces its ordered publication inventory. A lesson appears once per PDF even when several stages reuse it. Across the website, shared lessons have one page and courses link to that page.

```bash
python3 scripts/course_catalog.py validate
python3 scripts/assemble_book.py --course mathematics --out /tmp/math.md
./scripts/build-book.sh --course mathematics
python3 scripts/course_catalog.py export --course mathematics --zip
```

The export writes `outputs/packs/mathematics/openlearn.json` and the selected source files beneath it. Relative paths remain intact. Links to material outside the pack become repository source links. `--reading-only` omits the `runtime` trees. Export refuses to overwrite an existing destination; use a fresh output directory for another build.

Source packs contain readable Markdown and any declared solution code. The web reader links to published PDFs for offline reading.

```bash
cd apps/web
OPENLEARN_COURSE=mathematics pnpm build
```

This produces a site containing only that course's published lessons and references. Without the variable, the site shows the full catalogue. The web generator creates navigation and download links; it does not maintain separate lesson prose. Existing subject URLs remain stable where source files were relocated.

CI builds one PDF and pack per course and attaches them to releases. Course pages link to those release artifacts and their source files. PDFs and generated website pages are outputs, never editing targets. `--path` remains available for an optional PDF of one LLM course pass.

## Add a course

1. Write its curriculum or study plan only where it adds information beyond existing lessons.
2. Create `content/courses/<id>/openlearn.json`, referencing existing lessons and any new chapters.
3. Register the manifest in the root `openlearn.json` and add its PDF/zip names to `sr.yaml`.
4. Run catalogue validation, assemble the PDF, and inspect an exported source pack. The web reader discovers the new course from the catalogue. Refresh `apps/web/release-assets.json` after publishing its PDF; see the [reader README](../apps/web/README.md).

## Consolidated subject ownership

AI Engineering combines machine learning, model and LLM construction, AI platform applications, evaluation, and responsible AI. General protocols, streaming, caching, durable execution, and deployment belong to Systems Engineering. General design patterns and authorization belong to Software Engineering.

Programming and Algorithms combines language primers, type systems, algorithms, competitive programming, and practice explanations. Software Engineering combines craftsmanship, testing, maintenance, security, diagrams, and documentation. Mathematics combines introductory reasoning with the computing mathematics sequence. Worked projects live alongside their subject; the [project index](../content/references/worked-projects.md) links to all five.

The root `course_aliases` maps retired publication IDs to their new book for CLI selection. The generator uses `source-migrations.json` through `legacy-source-paths.mjs` to retain chapter URLs, image URLs, and device reading-progress keys. Retired book home pages and `/learn/<old-id>/` redirect to the consolidated home. Competitive Programming retains its former home URL as the competitive-programming chapter. Each canonical chapter appears once, and new exports use the consolidated ID.

The verified PDF snapshot omits pre-merge editions of combined books. Refresh `apps/web/release-assets.json` after publishing matching PDFs to restore their download links. Source packs and independent website builds already use the new manifests.
