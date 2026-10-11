# AGENTS.md

## Project Overview

**Languide** generates communication-focused language guides for travelers. Each language lives under `languages/<slug>/chapters/` as numbered markdown files, which get combined and rendered to PDF via `scripts/build-guide.sh`.

The focus is **practical phrases for everyday interactions** — not tourism tips. Content is organized by scenario (restaurants, hotels, shopping, transport, emergencies) using tables and pattern templates.

## Repository Structure

```
languages/<slug>/chapters/   — numbered markdown chapters (00-cover.md … 10-cultural-guide.md)
languages/<slug>/spec.md     — detailed content specification for that language
agents/system-prompt.md      — prompt for generating new language guides
agents/checklist.md          — quality checklist for guide content
scripts/build-guide.sh       — shell script for assembling chapters into PDFs
templates/tourism-guide.tex  — LaTeX template for PDF styling
outputs/                     — built PDFs (gitignored)
skills/                      — Claude Code skills for automating guide creation
```

## Key Conventions

- **Chapter numbering**: `00-cover.md` through `10-cultural-guide.md` (see `agents/system-prompt.md` for the required structure)
- **Phrase format**: Always use 4-column tables: `English | Target Language | Transliteration | Notes`
- **Pattern phrases**: Use `[brackets]` for customizable slots (e.g., `[thing] please`)
- **Politeness tiers**: Casual / Polite / Very Polite for each phrase category
- **No tourism tips**: Focus on how to communicate, not what to see
- **50+ phrases** per scenario chapter minimum

## Adding a New Language

Use the `/create-language` skill to generate a complete guide:

```
/create-language french
```

This reads the system prompt and checklist, references existing guides for format consistency, generates all 11 chapters, and validates against the checklist.

To manually add a language:

1. Create `languages/<slug>/chapters/` with all 11 chapter files following `agents/system-prompt.md`
2. Optionally create `languages/<slug>/spec.md` with detailed content spec
3. Validate against `agents/checklist.md`
4. Build with `./scripts/build-guide.sh <slug>`

## Building

```bash
./scripts/build-guide.sh                  # build all languages
./scripts/build-guide.sh <slug>           # build one language
./scripts/build-guide.sh --discover       # list available languages
```

CJK languages (japanese, chinese, korean) require CJK fonts — use `--skip-font-check` in CI.

## Skills

### `/create-language <language>`

Generates a complete communication-focused language guide. Creates all 11 chapter files under `languages/<slug>/chapters/` following the project's system prompt, validates against the checklist, and reports phrase counts.

## CI/CD

On push to `main`, GitHub Actions builds all guides and creates a semantic release with PDF artifacts attached.
