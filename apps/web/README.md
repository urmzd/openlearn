# Book reader

A mobile book reader built with Astro and Starlight. Course manifests and Markdown lessons are shared with the PDF publisher. The reader provides chapter contents, continuous reading, inline code and mathematics, full-text search, and published PDF downloads. Reading position stays on the device.

## Reading principle

The reader supports books that derive ideas from first principles, show worked solutions, and ask readers to predict, reason, test, and validate. Generating code is becoming easier; understanding its assumptions and gathering evidence for its behavior remain essential. Keep code, equations, and definitions visible. Review answers retain their disclosure controls so readers can commit to their reasoning before comparing it with the explanation. Reading progress records position, not mastery. See [How to read these books](../../content/references/how-to-read.md).

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

Use the production preview to test the full-text search index. Select one course with `OPENLEARN_COURSE=mathematics pnpm build`, or omit the variable to publish the catalogue. Set `OPENLEARN_SITE_URL` to the deployment's public URL.

PDF links come from the verified release asset snapshot in `release-assets.json`. Refresh it after publishing a release:

```bash
gh api repos/urmzd/openlearn/releases/latest \
  --jq '[.assets[] | select(.name | endswith(".pdf")) | {name,url:.browser_download_url}]' \
  > release-assets.json
pnpm build
```

See [course publication](../../docs/guides/courses.md) for source packs and PDF generation.
