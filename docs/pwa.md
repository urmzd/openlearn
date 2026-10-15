# Learning PWA

The web app is the installable learning platform. It uses the same course manifests and Markdown sources as the PDF publisher. It runs as a static site on GitHub Pages or any HTTPS host, without a native shell or application server.

## Install and read offline

Open the course library and choose **Install app**. Browsers that support an install prompt display it; Safari provides Add to Home Screen or Add to Dock through its own menus. Installation is optional for reading and offline downloads.

Choose **Save offline** on a course. The app downloads every published reading and its local diagrams. A course is marked saved only after all files have downloaded and their content hashes match. Keep the app open until the download finishes. Cancel a download with the same button, or remove a saved copy from the library or My learning. Removing a download keeps notes and reading progress.

Visited pages are also cached, up to 60 pages. Use Save offline for a complete course. Links to external textbooks, videos, source repositories, and release PDFs require an internet connection; they are not copied into the offline course. Search works offline for the published catalogue, but opening a result still requires a saved or previously visited page.

## Reading progress and notes

Mark a lesson as read, bookmark it, or save a note. My learning shows started courses, the next reading, saved courses, and notes. Shared lessons share their reading state across courses. Reading completion is self-reported; it does not create a passing code-check verdict.

Progress is stored locally in the browser, with no account or synchronization server. Use **Export progress** to keep a JSON backup and **Import progress** to restore it or move it to another device. Import merges newer lesson records and preserves newer notes already on the device. Clearing site data or browser storage removes local progress and offline downloads. Browser storage limits can also evict downloads; keep backups of notes you need to retain.

## App updates

A content-derived revision identifies each published app edition. A new service worker waits while the current app is open and offers **Update app**. Finish or cancel downloads before updating. Older saved courses and their required app resources remain available until an updated course download succeeds or the saved copy is removed.

Downloading an updated course never replaces the previous saved copy with a partial download. A failed network request, hash mismatch, cancellation, or storage limit leaves the previous edition available. Remove old downloads when you no longer need them.

## Code exercises

Browsers cannot execute this project's native Python, Rust, Go, and C toolchains. Use `practice/bin/ol` for exercises, verified module checks, milestones, and course exports. The PWA handles the reading and study workflow. AI-generated timed desktop sessions and native MCP integration are no longer part of the app.

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
- `src/scripts/app.mjs` connects the library, lesson controls, notes, backups, and offline controls without requiring a client-side application framework.

The offline worker accepts only same-origin requests and course IDs from the generated inventory. Course downloads verify SHA-256 hashes before committing metadata. Personal notes are rendered as text and never sent to a server.
