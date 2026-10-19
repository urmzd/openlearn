# Learning PWA

An installable, static learning app built with Astro and Starlight. Course manifests and Markdown lessons are shared with the PDF publisher. The app adds complete offline course downloads, reading progress, bookmarks, notes, backups, and update controls.

## Develop

Use Node.js 24, pnpm 11.6, and Python 3.11 or later.

```bash
pnpm install --frozen-lockfile
pnpm dev
```

## Build and preview

```bash
pnpm test
pnpm build
node --test tests/build.test.mjs
pnpm preview --host 127.0.0.1
```

The production preview enables service workers and installability on localhost. Development mode leaves service workers disabled. Select one course with `OPENLEARN_COURSE=mathematics pnpm build`, or omit the variable to publish the catalogue. Set `OPENLEARN_SITE_URL` to the deployment's public URL.

See the [PWA guide](../../docs/pwa.md) for offline behavior, browser storage, installation, and testing. See [course publication](../../docs/courses.md) for source packs and PDF generation.
