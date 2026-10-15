# Supersource

Courses built from reusable Markdown lessons. Read them on the web, download a PDF, or open a standalone course pack in the desktop app. Each course's manifest defines its reading order, references, and available practice. The website and PDF use those same sources.

Start with the [course catalogue](content/courses/README.md), the [Computer Science curriculum](content/courses/computer-science/README.md), or [Build Your Own LLM System](content/courses/build-an-llm/README.md).

## Repository

| Directory | Owns |
|---|---|
| `content/courses/` | Course manifests, curricula, study plans, and course-specific chapters |
| `content/tracks/` | Shared subject lessons referenced by courses |
| `content/references/` | Shared source register and attribution |
| `content/case-studies/`, `content/interviews/` | Reusable applied material |
| `course/` | LLM grading infrastructure: contracts, tests, fixtures, reference implementations |
| `practice/` | Shared exercise engine and exercise sources; courses opt in through their manifests |
| `apps/desktop/`, `apps/web/` | The platform that opens and publishes courses |
| `docs/`, `scripts/`, `deploy/` | Platform documentation, publishing tools, and deployment |
| `packs/` | Other self-contained packs, including Languide |
| `outputs/` | Generated PDFs and portable packs, never authored content |

## Read and practise

```bash
practice/bin/ol learn course
practice/bin/ol course init --name forge
practice/bin/ol start lang.01
practice/bin/ol check lang.01
```

Each course states its assessment boundaries. Reading routes and suggested projects do not imply implemented automated grading.

## Publish a course

```bash
python3 scripts/course_catalog.py list
python3 scripts/course_catalog.py validate
./scripts/build-book.sh --course build-an-llm
python3 scripts/course_catalog.py export --course build-an-llm --zip
```

Outputs are `outputs/openlearn-build-an-llm.pdf` and `outputs/packs/openlearn-build-an-llm.zip`. Extract a pack and open its folder in the desktop app. The LLM pack includes its grading tools; reading-only exports omit execution tools.

```bash
cd apps/web
pnpm install --frozen-lockfile
OPENLEARN_COURSE=build-an-llm pnpm dev  # omit the variable for the whole catalogue
```

```bash
cd apps/desktop
make init
make dev
```

See [course authoring and publication](docs/courses.md), [desktop documentation](apps/desktop/README.md), and the [source register](content/references/sources.md).

## License

[Apache License 2.0](LICENSE). Individual resources retain their stated licenses and access terms.
