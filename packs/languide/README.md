<p align="center">
  <h1 align="center">Languide</h1>
  <p align="center">
    Communication-focused language guides built from markdown
    <br /><br />
    <a href="https://github.com/urmzd/languide/releases">Download</a>
    &middot;
    <a href="https://github.com/urmzd/languide/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/urmzd/languide/releases">Releases</a>
  </p>
</p>

<p align="center">
  <a href="https://github.com/urmzd/languide/actions/workflows/cd.yml"><img src="https://github.com/urmzd/languide/actions/workflows/cd.yml/badge.svg" alt="CI"></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/github/license/urmzd/languide" alt="License"></a>
</p>

<table align="center">
  <tr>
    <td align="center">
      <img src="showcase/japanese-phrases.png" alt="Japanese pattern phrases" width="420" />
      <br />
      <sub><b>Japanese</b> — pattern phrases with romaji</sub>
    </td>
    <td align="center">
      <img src="showcase/spanish-phrases.png" alt="Spanish pattern templates" width="420" />
      <br />
      <sub><b>Spanish</b> — reusable phrase templates</sub>
    </td>
  </tr>
</table>

## Features

- **Scenario-based phrases** — greetings, restaurants, hotels, shopping, transport, emergencies
- **Pattern templates** with `[bracket]` slots for customizable phrases
- **Politeness tiers** — casual, polite, and very polite variants
- **Tables** for all phrase collections (no inline lists)
- **PDF generation** via a single shell script with Pandoc + XeLaTeX
- **Multi-language** workspace under `languages/<slug>/`

## Prerequisites

- [Pandoc](https://pandoc.org/) and a LaTeX engine (`xelatex` recommended)
- CJK font for Japanese/Chinese/Korean (e.g., Noto Sans CJK)

## Quick Start

```bash
./scripts/build-guide.sh                     # build all guides
./scripts/build-guide.sh japanese            # build a single guide
./scripts/build-guide.sh --discover          # list available languages
```

## Usage

### Adding a New Language

The easiest way is with [Claude Code](https://github.com/anthropics/claude-code):

```bash
claude "/create-language french"
```

This generates all 11 chapters following the system prompt, validates against the quality checklist, and reports phrase counts.

To add one manually:

1. Create `languages/<slug>/chapters/` with numbered chapters following the structure below
2. Use `agents/system-prompt.md` as the generation prompt
3. Verify against `agents/checklist.md`
4. Run `./scripts/build-guide.sh <slug>` to produce `outputs/<slug>-guide.pdf`

### CLI Reference

```
./scripts/build-guide.sh [options] [LANGUAGE...]

Options:
  --languages-dir DIR   Directory containing language folders (default: languages)
  --output-dir DIR      Output directory for PDFs (default: outputs)
  --skip-font-check     Skip CJK font detection
  --continue-on-error   Keep going if a language fails
  --manifest FILE       Write build manifest JSON to FILE
  --discover            List available languages and exit
```

### Fonts

CJK languages (Japanese, Chinese, Korean) require CJK fonts. If glyphs are missing:

```bash
# macOS
brew install --cask font-noto-sans-cjk font-noto-serif-cjk

# Debian/Ubuntu
sudo apt-get install fonts-noto-cjk
```

## Repository Layout

```
languages/<slug>/chapters/   Numbered markdown chapters (00-cover.md … 10-cultural-guide.md)
languages/<slug>/spec.md     Content specification for that language
agents/                      System prompt and quality checklist
skills/                      Claude Code skills for automating guide creation
scripts/build-guide.sh       Shell script for assembling chapters into PDFs
templates/                   LaTeX template for PDF styling
outputs/                     Built PDFs (gitignored)
```

### Chapter Structure

Each guide contains 11 chapters:

| Chapter | File | Content |
|---------|------|---------|
| Cover | `00-cover.md` | Introduction, usage notes, guide overview |
| Core Phrases | `01-core-phrases.md` | Pattern templates, greetings, emergencies, daily interactions |
| Language Foundations | `02-language-foundations.md` | Grammar, particles, sentence patterns |
| Pronunciation | `03-pronunciation.md` | Vowels, consonants, special sounds |
| Writing Systems | `04-writing-systems.md` | Scripts and characters, survival reading |
| Directions & Navigation | `05-directions-navigation.md` | Asking directions, landmarks |
| Transportation | `06-transportation.md` | Trains, buses, taxis, airports |
| Restaurants | `07-restaurants.md` | Ordering, dietary needs, payment |
| Shopping | `08-shopping.md` | Sizes, trying on, payment, returns |
| Hotels | `09-hotels.md` | Check-in/out, requests, problems |
| Cultural Guide | `10-cultural-guide.md` | Regional differences, etiquette |

### Conventions

- **Tables** for all phrase collections — 4-column format: `English | Target Language | Transliteration | Notes`
- **Pattern phrases** with `[brackets]` for customizable slots
- **Politeness tiers**: Casual / Polite / Very Polite
- **50+ phrases** per scenario chapter, **30+** for directions and emergencies
- **No tourism tips** — focus on how to communicate, not what to see

## Agent Skill

This repo's conventions are available as portable agent skills in [`skills/`](skills/).

| Resource | Purpose |
|----------|---------|
| `agents/system-prompt.md` | Prompt for generating new language guides |
| `agents/checklist.md` | Quality checklist for guide content |
| `skills/create-language/SKILL.md` | Claude Code skill that automates guide creation |
| `AGENTS.md` | Agent conventions and project context |

## License

[Apache-2.0](LICENSE)
