# Course website

The website and PDFs use the same course manifests and Markdown sources. No lesson or landing page is authored in this app: `scripts/sync-content.mjs` generates pages, navigation, and download links from the catalogue.

```bash
pnpm install --frozen-lockfile
pnpm dev
pnpm build
OPENLEARN_COURSE=mathematics pnpm build
```

The selected-course build contains only that course's lessons and references. Set `OPENLEARN_SITE_URL` to the independent deployment's public URL for its sitemap. Without `OPENLEARN_COURSE`, the app publishes the whole catalogue with shared lessons rendered once.

See [course publication](../../docs/courses.md). Generated files live in `src/content/docs/`, `src/generated/`, `public/repo/`, and `dist/`.
