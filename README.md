<p align="center">
  <h1 align="center">Supersource</h1>
  <p align="center">
    Free, self-paced learning across technology, mathematics, and engineering, for people from all backgrounds. Read online, practise, or take a course offline.
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

## Features

- **Course catalogue:** 18 learning routes across mathematics, algorithms, ML, systems, data engineering, infrastructure, and related subjects.
- **Build Your Own LLM System:** implement tokenizers, attention, KV caches, inference, serving, and operations against readable interfaces and tests.
- **Practice:** predict program output, build implementations, and reattempt solved problems. Keep your work in a private local scratchpad.
- **Read anywhere:** browse shared lessons on the web, download course PDFs, or open portable packs in the OpenLearn desktop app.
- **Publish from one source:** use a course manifest to select the same Markdown lessons for its website, PDF, and downloadable pack.

Courses state their assessment boundaries. Suggested reading and projects do not imply automated grading.

## Installation

Read the [website](https://openlearn.urmzd.com) without installing anything. Download PDFs, portable course packs, and the desktop app from [releases](https://github.com/urmzd/openlearn/releases/latest). Extract a course pack and open its folder in the desktop app, or import its zip.

### Prerequisites

For local study and publishing:

| Requirement | Used for |
|---|---|
| Git | Clone the repository and track learner work |
| Python 3.11+ | List, validate, and export courses |
| uv | Run the LLM course grader and Python exercises |
| Course-specific language toolchains | Compile and run the exercises you choose |

```bash
git clone https://github.com/urmzd/openlearn.git
cd openlearn
```

Run `practice/bin/ol doctor` before starting the LLM course. See the [desktop build instructions](apps/desktop/README.md#build) for development prerequisites.

## Quick Start

From the checkout, list the available courses:

```bash
python3 scripts/course_catalog.py list
```

| Start here | What to expect |
|---|---|
| [Technology Foundations](content/courses/computer-science/README.md) | Explore programming, mathematics, systems, data, AI, and other areas of technology at your own pace |
| [Build Your Own LLM System](content/courses/build-an-llm/README.md) | A staged implementation course with module checks and runnable milestones |
| [Course catalogue](content/courses/README.md) | Subject-specific study plans and other learning routes |

For the LLM course, inspect the learning path and your toolchain:

```bash
practice/bin/ol learn course
practice/bin/ol doctor
```

Follow the [course overview](content/courses/build-an-llm/README.md) to create your learner repository and begin implementing modules.

## Usage

### Practise

```bash
practice/bin/ol list
practice/bin/ol score
```

Choose an exercise from the list, then follow the [practice guide](practice/README.md) to start and check an attempt. Attempts live in the gitignored `.scratchpad/` directory.

### Export a course

```bash
python3 scripts/course_catalog.py validate
python3 scripts/course_catalog.py export --course mathematics --zip
```

Open `outputs/packs/mathematics/` in the desktop app or import `outputs/packs/openlearn-mathematics.zip`. Export refuses to overwrite an existing destination; use `--output` with a fresh directory for another export.

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
| Course `openlearn.json` | Reading order, shared lessons, references, and available practice |
| `OPENLEARN_COURSE` | Select one course for a website build or development server |
| `OPENLEARN_SITE_URL` | Set the public URL for an independent course deployment |
| `OL_COURSE_HOME` | Choose where the LLM course stores your learner repository |

See [course authoring and publication](docs/courses.md) for manifest fields and the [practice guide](practice/README.md) for CLI configuration.

## Architecture

A course references canonical lessons through its manifest. The web app, PDF builder, and pack exporter use the same publication inventory. The desktop app opens those packs; courses opt into the exercise engine or LLM grading backend when applicable.

Read the [course publication guide](docs/courses.md) and [desktop architecture guide](apps/desktop/docs/architecture.md) for implementation details. Consult the [source register](content/references/sources.md) for attribution, resource access, and validation evidence.

## Agent Skill

This repo's conventions are available as portable agent skills in [`skills/`](skills/).

## License

[Apache License 2.0](LICENSE). Linked resources and individual assets retain their stated licenses and access terms.
