# Web book reader

The web reader presents the same course manifests and Markdown sources as the PDF publisher. It is a static Astro and Starlight site with a mobile reading layout, chapter navigation, full-text search, and links to verified published PDFs.

## Learning principle

As code generation gets easier, understanding, reasoning, testing, and validation become more important. The books derive mechanisms from first principles, show the worked solution, and give readers opportunities to predict a result, explain a choice, or challenge an assumption before comparing their reasoning with the answer.

The interface supports that method: code, equations, and definitions remain visible alongside the explanation. Review answers keep their authored disclosure controls. Readers can pause and reason without animations hiding content or definitions requiring extra taps. A saved reading position helps them return; it does not certify understanding. See [How to read these books](../content/references/how-to-read.md) for the complete reading method and [course authoring](courses.md#teaching-principle) for writing guidance.

## Find a book

The library lists books alphabetically. **Find a book** filters titles and descriptions on the device. The header's **Search** searches the published text and links directly to matching sections. Full-text search requires a production build; use the preview server to test it.

Open a book for its introduction, Start or Continue reading link, and numbered chapter and section contents. The menu shows that book's chapters and supporting references. The Library and book contents links provide a quick way back.

## Read and check your understanding

A chapter reads as one page. When you near the end of a section, the next section of the same chapter loads below it. The address bar, title, and menu follow the section on screen. A chapter ends with a link to the next chapter. A thin line at the top shows reading progress; reaching a section's end marks it read.

Before revealing a review answer, commit to your prediction and explain why it follows. Use the worked solution to check your reasoning, then identify a test, counterexample, or changed assumption that could challenge it. Compare the evidence with the requirement, including what remains untested.

Without JavaScript, each section remains available as its own page with previous and next links. Wide code listings, tables, and display equations scroll within the reading column on small screens.

## PDFs and device storage

A PDF link appears on the shelf and book opening when `apps/web/release-assets.json` contains a matching published asset. Links point to a specific release rather than assuming a file exists in the latest release. Refresh the snapshot after publishing PDFs using the command in the [reader README](../apps/web/README.md). Download a PDF to read offline; source packs remain available from [releases](https://github.com/urmzd/openlearn/releases/latest).

Reading position and completion markers stay in browser storage, with no account or synchronization server. Clearing site data removes them. The current reader has no offline-course download, note editor, or progress import/export controls. Existing stored records are retained when progress is updated.

## Development

Use Node.js 24, pnpm 11.6, and Python 3.11 or later.

```bash
cd apps/web
pnpm install --frozen-lockfile
pnpm test
pnpm build
node --test tests/build.test.mjs
pnpm preview --host 127.0.0.1
```

Select one course with `OPENLEARN_COURSE=mathematics pnpm build`, or omit the variable for the full library. Set `OPENLEARN_SITE_URL` to the deployment URL for the sitemap. Deploy at the root of the hostname; generated publication URLs use `/`.

## Implementation

| File | Responsibility |
|---|---|
| `scripts/sync-content.mjs` | Generates lesson pages, numbered book contents, navigation, and the course inventory from canonical sources |
| `release-assets.json` | Records verified PDF filenames and release URLs used to generate download links |
| `src/components/Shelf.astro` | Renders the library, local book filter, reading links, and PDF links |
| `src/scripts/app.mjs` | Filters books, restores reading position, and connects the continuous reader to local progress |
| `src/scripts/reader.mjs` | Loads subsequent sections, renders diagrams, follows the active section, and records reading progress |
| `src/lib/learning.mjs` | Validates and merges stored records and calculates reading progress |
| `src/routeData.ts` | Limits the menu to the book being read |
| `src/styles/custom.css` | Defines typography, mobile layout, code and table containment, and print styling |

The build still generates the web manifest, icons, and offline-worker assets through `scripts/build-pwa.mjs`. The current client does not register that worker or expose its course-download controls. The worker and its tests remain as compatibility infrastructure; they are separate from the active reading interface.
