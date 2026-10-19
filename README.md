<p align="center">
  <h1 align="center">Supersource</h1>
  <p align="center">
    Free, self-paced learning across technology, mathematics, and engineering, for people from all backgrounds. A manual of worked solutions, explained from first principles. Read online or take a course offline.
    <br /><br />
    <a href="https://github.com/urmzd/openlearn/releases/latest">Download</a>
    &middot;
    <a href="https://github.com/urmzd/openlearn/issues">Report Bug</a>
    &middot;
    <a href="https://openlearn.urmzd.com">Browse Courses</a>
  </p>
</p>

<p align="center">
  <a href="https://github.com/urmzd/openlearn/actions/workflows/ci.yml"><img src="https://github.com/urmzd/openlearn/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/github/license/urmzd/openlearn" alt="License"></a>
</p>

## Screenshots

![Course catalog with alphabetically ordered courses and a search field](showcase/course-catalog.png)

<details>
<summary>See the course outline and reading experience</summary>

The course outline keeps chapters in order and supporting references separate.

![Mathematics Foundations course outline with ordered chapters and supporting references](showcase/mathematics-course.png)

Chapters introduce one idea at a time, with worked examples and rendered equations.

![Mathematics chapter showing a cost model, a worked example, and chapter navigation](showcase/mathematics-chapter.png)

</details>

Captured with [teasr](https://github.com/urmzd/teasr). Rebuild the website, then run `teasr run --config teasr.toml` to refresh these screenshots.

## Features

- **A manual, not a homework set:** every chapter builds an idea from first principles, works it by hand, then walks through the solution step by step and explains why each choice is the right one.
- **Course catalogue:** 18 learning routes across mathematics, algorithms, ML, systems, data engineering, infrastructure, and related subjects.
- **Build Your Own LLM System:** follow a complete LLM platform (tokenizers, attention, KV caches, kernels, inference, serving, and operations) through annotated reference solutions in [`course/solutions/`](course/solutions/).
- **Explain it back:** each chapter ends its walkthrough with questions about *why* the solution works, with answers, so you can check that you could defend the design to someone else.
- **Worked problem sets:** mathematics problem sets show every answer with its derivation and the common wrong answers.
- **Read anywhere:** browse shared lessons on the web, download course PDFs, or install the PWA and save courses for offline reading.
- **Publish from one source:** use a course manifest to select the same Markdown lessons for its website, PDF, and downloadable pack.

## Why a manual

Code is increasingly written by tools rather than by hand. What still matters is knowing what the optimal solution is and being able to explain why: which invariant it protects, which alternative it rejects, and what breaks if a line changes. The chapters are written for that. They show the solution instead of asking you to retype it, and spend their effort on the reasoning that produced it.

## Installation

Read the [website](https://openlearn.urmzd.com) without installing anything. Download PDFs and source packs from [releases](https://github.com/urmzd/openlearn/releases/latest). Install the web app from your browser. Choose Save offline on a course to download its lessons and diagrams.

To read the sources locally or publish a course:

```bash
git clone https://github.com/urmzd/openlearn.git
cd openlearn
```

| Requirement | Used for |
|---|---|
| Git | Clone the repository |
| Python 3.11+ | List, validate, and export courses |
| Pandoc and XeLaTeX | Build course PDFs (optional) |
| Node.js 24 and pnpm | Serve the website locally (optional) |

## Quick Start

| Start here | What to expect |
|---|---|
| [Technology Foundations](content/courses/computer-science/README.md) | Explore programming, mathematics, systems, data, AI, and other areas of technology at your own pace |
| [Build Your Own LLM System](content/courses/build-an-llm/README.md) | One LLM platform, end to end, explained module by module through its reference solution |
| [Course catalogue](content/courses/README.md) | Subject-specific study plans and other learning routes |
| [Practice](practice/README.md) | Predict-the-output puzzles and data-structure implementations, each with a worked answer |

Every chapter follows the same shape: why the idea matters now, the principles, a worked example by hand, the interface, **the solution step by step**, pitfalls, and where the idea is used next. Try to predict each step of a solution before you read it.

## Usage

### List and export a course

```bash
python3 scripts/course_catalog.py list
python3 scripts/course_catalog.py validate
python3 scripts/course_catalog.py export --course mathematics --zip
```

Read the Markdown sources in `outputs/packs/mathematics/` or share `outputs/packs/openlearn-mathematics.zip`. For offline reading in the PWA, use Save offline in the course library. Export refuses to overwrite an existing destination; use `--output` with a fresh directory for another export.

### Build a PDF

Install Pandoc, XeLaTeX, and Bash 4.3 or newer, then run:

```bash
./scripts/build-book.sh --course mathematics
```

The result is `outputs/openlearn-mathematics.pdf`. See [course publication](docs/courses.md) for the publishing workflow and [the build script](scripts/build-book.sh) for rendering options and recommended diagram tools.

### Serve a course

Install Node.js 24 and pnpm 11.6, then run:

```bash
cd apps/web
pnpm install --frozen-lockfile
OPENLEARN_COURSE=mathematics pnpm dev
```

Omit `OPENLEARN_COURSE` to serve the full catalogue.

## Configuration

| Setting | Purpose |
|---|---|
| Course `openlearn.json` | Reading order, shared lessons, and references |
| `OPENLEARN_COURSE` | Select one course for a website build or development server |
| `OPENLEARN_SITE_URL` | Set the public URL for an independent course deployment |

See [course authoring and publication](docs/courses.md) for manifest fields.

## Architecture

A course references canonical lessons through its manifest. The web app, PDF builder, and pack exporter use the same publication inventory. The installable PWA saves courses, reading progress, bookmarks, and notes on your device. Reference solutions live in [`course/solutions/`](course/solutions/) (LLM system) and [`practice/build/`](practice/build/) (data structures); chapters quote and explain them.

Read the [course publication guide](docs/courses.md) and [PWA guide](docs/pwa.md) for implementation details. Consult the [source register](content/references/sources.md) for attribution, resource access, and validation evidence.

## Agent Skill

This repo's conventions are available as portable agent skills in [`skills/`](skills/).

## License

[Apache License 2.0](LICENSE). Linked resources and individual assets retain their stated licenses and access terms.
