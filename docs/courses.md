# Course sources and publication

Each course is a book. It owns its chapters: every chapter lives in exactly one course directory, and other books cross-reference it rather than include it. Every manifest lives at `content/courses/<id>/openlearn.json`. Its paths are relative to the repository root. The root `openlearn.json` lists the available manifests.

## Ownership

Put a book's front matter (`README.md`, `study-plan.md`) and its chapters in `content/courses/<id>/`, one directory per chapter, one file per section:

```
<book>/
├── README.md              front matter
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

`course/` holds the reference solutions (`course/solutions/`) and interface contracts (`course/contracts/`) that the Build Your Own LLM System chapters quote and explain. `practice/` holds predict-the-output puzzles and data-structure implementations with worked answers. These are supporting directories, not competing course outlines. A course declares the solution files to ship in its portable pack with `runtime`.

## Manifest

The `openlearn.json` manifest defines publication for the web app, PDF, and source pack:

| Field | Meaning |
|---|---|
| `id`, `title` | Stable course identity and display name |
| `start` | Curriculum and study-plan documents, in order |
| `tagline` | Optional short course description for the web landing page |
| `tracks` | The chapters, in order. Each entry is one chapter directory (or file). Mark appendices with `"appendix": true` and a closing note with `"unnumbered": true` |
| `paths`, `featured` | Optional staged route; its stage order takes precedence over track order in the PDF |
| `reference` | Source register, contracts, or other supporting trees |
| `runtime` | Repository files/trees (such as solution code) included in portable packs |

Every course is a reading course. Chapters show the solution and explain it; nothing is graded.

## Numbering

Chapter numbers come from the order of `tracks`: chapters are 1, 2, 3, ... and appendices A, B, ... Section numbers come from the order of files inside the chapter: its `README.md` is the chapter itself, the next file or folder is section 1.1, and the pages inside a section folder are 1.1.1, 1.1.2, ... Headings in the Markdown carry no numbers. The website, the PDF, and the course landing page add them ("Chapter 5: Durable Orchestration and Workers", "5.3 The task queue", "Appendix A: Capstones"), so reordering a book is a rename, not an edit to every heading.

The website presents one alphabetical course catalog. Inside each course, chapters follow the manifest's publication order; auxiliary references are grouped separately. It does not expose source tracks as a competing lesson catalog. A chapter used by only one course has explicit previous/next links. A shared chapter links back to the courses that teach it, so pagination cannot silently cross into another course.

## One publishing pipeline

`scripts/course_catalog.py` resolves every course, validates its paths, and produces its ordered publication inventory. A lesson appears once per PDF even when several stages reuse it. Across the website, shared lessons have one page and courses link to that page.

```bash
python3 scripts/course_catalog.py validate
python3 scripts/assemble_book.py --course mathematics --out /tmp/math.md
./scripts/build-book.sh --course mathematics
python3 scripts/course_catalog.py export --course mathematics --zip
```

The export writes `outputs/packs/mathematics/openlearn.json` and the selected source files beneath it. Relative paths remain intact. Links to material outside the pack become repository source links. `--reading-only` omits the `runtime` trees. Export refuses to overwrite an existing destination; use a fresh output directory for another build.

Source packs contain readable Markdown and any declared solution code. In the PWA, use Save offline to download published lessons, diagrams, and the app resources needed to read them.

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
4. Run catalogue validation, assemble the PDF, and inspect an exported source pack. The PWA discovers the new course from the catalogue.
