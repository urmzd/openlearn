# AGENTS.md

## Project Overview

**Languide** generates communication-focused language guides for casual and professional use. Each language lives under `languages/<slug>/chapters/` as numbered markdown files, which get combined and rendered to PDF via `scripts/build-guide.sh`.

The focus is **practical phrases for everyday interactions** — not tourism tips. Content is organized by scenario (restaurants, hotels, shopping, transport, emergencies, work) using tables and pattern templates.

Guides: Arabic, French, Japanese, Mandarin, Russian, Spanish, and Tajik and Persian.

## Repository Structure

```
languages/<slug>/chapters/   — numbered markdown chapters (00-cover.md … 11-work-and-business.md)
languages/<slug>/spec.md     — detailed content specification for that language
agents/system-prompt.md      — prompt for generating new language guides
agents/checklist.md          — quality checklist for guide content
paths/<slug>/                — staged learning path: path.tsv and README.md
scripts/build-guide.sh       — shell script for assembling chapters into PDFs
scripts/rtl.lua              — Pandoc filter that sets Arabic-script text right to left
scripts/tables.lua           — Pandoc filter that sizes table columns from their content
templates/tourism-guide.tex  — LaTeX template for PDF styling
outputs/                     — built PDFs (gitignored)
skills/                      — Claude Code skills for automating guide creation
```

## Key Conventions

- **Chapter numbering**: `00-cover.md` through `11-work-and-business.md` (see `agents/system-prompt.md` for the required structure)
- **Phrase format**: Always use 4-column tables: `English | Target Language | Transliteration | Notes`. The Tajik guide adds a `Persian script` column, because it teaches two scripts
- **Practice**: every chapter after the cover ends with a ten-row `Prompt | Answer` table
- **Pattern phrases**: Use `[brackets]` for customizable slots (e.g., `[thing] please`)
- **Politeness tiers**: Casual / Polite / Very Polite for each phrase category
- **No tourism tips**: Focus on how to communicate, not what to see
- **50+ phrases** per scenario chapter minimum

## Adding a New Language

Use the `/create-language` skill to generate a complete guide:

```
/create-language french
```

This reads the system prompt and checklist, references existing guides for format consistency, generates all 12 chapters, and validates against the checklist.

To manually add a language:

1. Create `languages/<slug>/chapters/` with all 12 chapter files following `agents/system-prompt.md`
2. Optionally create `languages/<slug>/spec.md` with detailed content spec
3. Validate against `agents/checklist.md`
4. Add a track to `openlearn.json` and a path under `paths/<slug>/`
5. Build with `./scripts/build-guide.sh <slug>`

## Building

```bash
./scripts/build-guide.sh                  # build all languages
./scripts/build-guide.sh <slug>           # build one language
./scripts/build-guide.sh --discover       # list available languages
```

Fonts by script: CJK guides (japanese, mandarin) need a CJK font, Arabic-script guides (arabic, tajik) need an Arabic font, and Cyrillic guides (russian, tajik) need DejaVu or Liberation. Use `--skip-font-check` in CI. The script needs Bash 4 or later.

## Skills

### `/create-language <language>`

Generates a complete communication-focused language guide. Creates all 12 chapter files under `languages/<slug>/chapters/` following the project's system prompt, validates against the checklist, and reports phrase counts.

## CI/CD

On push to `main`, GitHub Actions builds all guides and creates a semantic release with PDF artifacts attached.
