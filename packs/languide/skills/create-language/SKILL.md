---
name: create-language
description: "Generate a complete communication-focused language guide for travelers. Creates all 11 chapter files following the project's system prompt and checklist."
---

# Create Language Guide

You are generating a **communication-focused language guide** for travelers. Follow the system prompt at `agents/system-prompt.md` exactly, and validate against `agents/checklist.md` before finishing.

## Input

The user provides: `$ARGUMENTS` — the language name (e.g., "french", "korean", "portuguese").

Parse the language slug from the arguments (lowercase, no spaces). If no language is provided, ask the user.

## Steps

1. **Read the system prompt and checklist:**
   - Read `agents/system-prompt.md` for the full guide generation spec
   - Read `agents/checklist.md` for quality requirements

2. **Check if the language already exists:**
   - Check `languages/<slug>/chapters/` — if it exists, ask if the user wants to overwrite or skip

3. **Reference an existing guide for format:**
   - Read a few chapters from `languages/japanese/chapters/` (e.g., `00-cover.md`, `01-core-phrases.md`, `07-restaurants.md`) to match the exact format, table style, and depth

4. **Generate all 11 chapters** following the required structure:
   - `00-cover.md` — About, usage notes, guide structure overview
   - `01-core-phrases.md` — Pattern templates, greetings, small talk, asking for help, emergencies, daily interactions
   - `02-language-foundations.md` — Essential grammar, particles, pronouns, sentence patterns
   - `03-pronunciation.md` — Vowels, consonants, special sounds
   - `04-writing-systems.md` — Scripts/characters (if applicable), survival reading
   - `05-directions-navigation.md` — Asking directions, landmarks
   - `06-transportation.md` — Trains, buses, taxis, airports
   - `07-restaurants.md` — Ordering, dietary needs, payment
   - `08-shopping.md` — Sizes, trying on, payment, returns
   - `09-hotels.md` — Check-in/out, requests, problems
   - `10-cultural-guide.md` — Regional differences, etiquette affecting communication

   Write each file to `languages/<slug>/chapters/`.

5. **Quality requirements per chapter:**
   - **Tables for ALL phrases** — 4-column format: `English | Target Language | Transliteration | Notes`
   - **Pattern phrases** with `[brackets]` for customizable slots — 10+ patterns minimum
   - **Politeness tiers** (Casual / Polite / Very Polite) where relevant
   - **50+ phrases per scenario chapter** (chapters 05-09)
   - **30+ phrases** for directions and emergencies sections
   - **Regional variations** where they affect communication
   - **No tourism tips** — focus on how to communicate, not what to see

6. **Validate against checklist:**
   - Go through every item in `agents/checklist.md` and confirm compliance
   - Report any gaps to the user

7. **Optionally generate a spec:**
   - Ask the user if they'd like a `languages/<slug>/spec.md` created (detailed content specification like the Japanese one)

## Output

After generating all chapters, report:
- Number of chapters created
- Approximate phrase count per chapter
- Any checklist items that need attention
- Remind the user they can build the PDF with: `./scripts/build-guide.sh <slug>`
