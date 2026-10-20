# Learning PWA

The web app is the installable learning platform. It uses the same course manifests and Markdown sources as the PDF publisher. It runs as a static site on GitHub Pages or any HTTPS host, without a native shell or application server.

## Install and read offline

Install the site from the browser's own menu (Install app, or Add to Home Screen in Safari). Installation is optional for reading and offline downloads.

The home page is the shelf: every book, with the ones you have started first, a **Continue reading** card, and your notes. Choose **Save offline** on a book. The app downloads every published section and its local diagrams. A book is marked saved only after all files have downloaded and their content hashes match. Keep the app open until the download finishes. Cancel a download with the same button, or remove a saved copy with **Remove offline copy**. Removing a download keeps notes and reading progress.

Visited pages are also cached, up to 60 pages. Use Save offline for a complete course. Links to external textbooks, videos, source repositories, and release PDFs require an internet connection; they are not copied into the offline course. Search works offline for the published catalogue, but opening a result still requires a saved or previously visited page.

## Reading

A chapter reads as one page. When you near the end of a section, the next section of the same chapter loads below it, and the address bar, title, and menu follow the section on screen. A chapter ends with a card that links to the next chapter. A thin line at the top shows how far through the chapter you are. Reaching the end of a section marks it read; there is no button.

Review answers start hidden behind "Think it through, then reveal the answer", and key terms show the term first with the definition on tap (or **Show all definitions**). Headings and figures ease in as they arrive unless the reader prefers reduced motion. Without JavaScript every section is still its own page with previous and next links.

## Progress and notes

Each section has an **Add a note** link; notes appear on the shelf. Shared lessons share their reading state across courses.

Progress is stored locally in the browser, with no account or synchronization server. Under **Your progress on this device** on the shelf, use **Export progress** to keep a JSON backup and **Import progress** to restore it or move it to another device. Import merges newer lesson records and preserves newer notes already on the device. Clearing site data or browser storage removes local progress and offline downloads. Browser storage limits can also evict downloads; keep backups of notes you need to retain.

## App updates

A content-derived revision identifies each release. Pages always load from the network when online, so readers see the latest text without doing anything. A new service worker installs in the background and takes over at once; there is no update prompt. A course download running in the old worker stops with "The app updated during this download" and can be retried. A saved book from an older release shows **Refresh offline copy**, and the older copy and its app resources stay available until the refresh succeeds or the copy is removed.

Downloading an updated course never replaces the previous saved copy with a partial download. A failed network request, hash mismatch, cancellation, or storage limit leaves the previous edition available. Remove old downloads when you no longer need them.

## Solutions

The PWA handles the reading and study workflow. Solutions are shown and explained in the chapters, so nothing needs to run in the browser. Use `scripts/course_catalog.py` for course exports. AI-generated timed desktop sessions and native MCP integration are no longer part of the app.

## Development

Use Node.js 24, pnpm 11.6, and Python 3.11 or later.

```bash
cd apps/web
pnpm install --frozen-lockfile
pnpm test
pnpm build
pnpm preview --host 127.0.0.1
```

Open the preview URL to test installation and service workers. The development server deliberately does not register a service worker, so cached builds do not interfere with editing. Service workers require HTTPS or a localhost origin.

```bash
OPENLEARN_COURSE=mathematics pnpm build
node --test tests/build.test.mjs
```

A selected-course build contains only its readings, supporting references, and the PWA study controls. Set `OPENLEARN_SITE_URL` to the deployment URL for its sitemap. Deploy at the root of the hostname: the manifest, service-worker scope, and publication URLs use `/`.

## Implementation

- `scripts/sync-content.mjs` creates the shared lesson pages and course inventory.
- `scripts/build-pwa.mjs` hashes built resources, creates install icons and the web manifest, and compiles the service worker's edition and app-shell inventory.
- `src/service-worker.js` manages complete course downloads, cancellation, offline navigation, bounded recent-page caching, and updates. Downloaded courses refer to one retained app-shell cache per edition.
- `src/lib/learning.mjs` validates progress backups, merges lesson records, and calculates reading progress.
- `src/scripts/app.mjs` connects the shelf, notes, backups, and offline controls without requiring a client-side application framework.
- `src/scripts/reader.mjs` runs continuous reading: loading the next section, tracking the section on screen, marking sections read, the progress line, reveal cards, and the ease-in.
- `src/routeData.ts` limits the menu to the book being read.

The offline worker accepts only same-origin requests and course IDs from the generated inventory. Course downloads verify SHA-256 hashes before committing metadata. Personal notes are rendered as text and never sent to a server.
