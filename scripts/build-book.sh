#!/usr/bin/env bash
set -euo pipefail

# build-book.sh — Build a single PDF of the entire supersource curriculum.
#
# Concatenates every track/topic README (in learning order) into one book and
# renders it with pandoc + xelatex. Architecture SVGs are embedded via librsvg;
# Mermaid fenced blocks are rendered with mermaid-filter when available, and
# fall back to code blocks if rendering fails (the build never breaks).
#
# Usage:
#   ./scripts/build-book.sh [options]
#
# Options:
#   --output-dir DIR   Output directory for the PDF (default: outputs)
#   --skip-mermaid     Leave Mermaid blocks as code (no headless browser needed)
#   --no-toc           Omit the table of contents
#   --manifest FILE    Write a build manifest JSON to FILE
#   --path NAME        Build one path (content/courses/ai-engineering/paths/NAME/path.tsv) instead of the
#                      whole curriculum; writes supersource-NAME.pdf
#   --course ID       Build a course from its manifest
#   --help             Show this help
#
# Requirements: pandoc, a LaTeX engine (xelatex), python3.
# Recommended:  librsvg (rsvg-convert) for SVG, mermaid-filter for Mermaid.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$ROOT_DIR/outputs"
TITLE="Openlearn: The Complete Curriculum"
SUBTITLE="Free, self-paced study from foundations to Staff+ depth"
SKIP_MERMAID=false
WITH_TOC=true
MANIFEST=""
LEARN_PATH=""
COURSE_ID=""

# Font candidates (Unicode-capable serif/sans cover the math glyphs the
# curriculum uses: → ≠ ∝ σ Σ · etc).
# shellcheck disable=SC2034
SERIF_CANDIDATES=("DejaVu Serif" "Liberation Serif" "STIX Two Text" "Palatino")
# shellcheck disable=SC2034
SANS_CANDIDATES=("DejaVu Sans" "Liberation Sans" "Helvetica" "Arial")
# shellcheck disable=SC2034
MONO_CANDIDATES=("DejaVu Sans Mono" "Liberation Mono" "Menlo" "Courier New")

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[info] $*"; }
# POSIX BRE interval (\{0,1\}) instead of GNU \? so this works in BSD sed too.
usage() { sed -n '4,/^$/s/^# \{0,1\}//p' "$0"; exit 0; }

find_font() {
  local -n candidates=$1
  command -v fc-list >/dev/null 2>&1 || return 1
  local available
  available="$(fc-list --format '%{family}\n' 2>/dev/null)" || return 1
  for candidate in "${candidates[@]}"; do
    # here-string (not a pipe): `grep -q` short-circuiting under `pipefail`
    # would otherwise SIGPIPE the upstream and make a match look like a miss.
    if grep -qiF "$candidate" <<<"$available"; then
      echo "$candidate"; return 0
    fi
  done
  return 1
}

# --- argument parsing -------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --skip-mermaid) SKIP_MERMAID=true; shift ;;
    --no-toc) WITH_TOC=false; shift ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --path) LEARN_PATH="$2"; shift 2 ;;
    --course) COURSE_ID="$2"; shift 2 ;;
    --help|-h) usage ;;
    -*) die "Unknown option: $1" ;;
    *) die "Unexpected argument: $1" ;;
  esac
done

[[ -z "$LEARN_PATH" || -z "$COURSE_ID" ]] || die "choose --course or --path"

# --- preflight --------------------------------------------------------------

command -v pandoc >/dev/null 2>&1 || die "pandoc is required (e.g. brew install pandoc)."
command -v python3 >/dev/null 2>&1 || die "python3 is required."
command -v rsvg-convert >/dev/null 2>&1 || info "rsvg-convert not found — SVG diagrams may not embed (install librsvg)."

# --- assemble ---------------------------------------------------------------

WORK_DIR="$(mktemp -d)"
PUPPETEER_CFG="$ROOT_DIR/.puppeteer.json"   # mermaid-filter reads this from cwd
CREATED_PUPPETEER=false
cleanup() {
  rm -rf "$WORK_DIR"
  rm -f "$ROOT_DIR/mermaid-filter.err"
  $CREATED_PUPPETEER && rm -f "$PUPPETEER_CFG"
}
trap cleanup EXIT
COMBINED_MD="$WORK_DIR/supersource-book.md"

info "Assembling curriculum into one document..."
assemble_args=(--root "$ROOT_DIR" --out "$COMBINED_MD")
PDF_NAME="supersource-curriculum.pdf"
if [[ -n "$COURSE_ID" ]]; then
  COURSE_ID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["manifest"]["id"])' < <(python3 "$SCRIPT_DIR/course_catalog.py" index --course "$COURSE_ID"))"
  assemble_args+=(--course "$COURSE_ID")
  PDF_NAME="openlearn-$COURSE_ID.pdf"
  TITLE="$(python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["manifest"]["title"])' < <(python3 "$SCRIPT_DIR/course_catalog.py" index --course "$COURSE_ID"))"
  SUBTITLE="An OpenLearn course"
elif [[ -n "$LEARN_PATH" ]]; then
  assemble_args+=(--path "$LEARN_PATH")
  PDF_NAME="supersource-$LEARN_PATH.pdf"
  # The path's README H1 is its title, e.g. "Forward Deployed Engineer, ...".
  TITLE="$(sed -n 's/^# //p' "$ROOT_DIR/content/courses/ai-engineering/paths/$LEARN_PATH/README.md" | head -1)"
  SUBTITLE="An Openlearn course path"
fi
python3 "$SCRIPT_DIR/assemble_book.py" "${assemble_args[@]}"

# --- render -----------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"
PDF_PATH="$OUTPUT_DIR/$PDF_NAME"
BUILD_DATE="$(date -u +%Y-%m-%d)"

base_cmd=(
  pandoc "$COMBINED_MD" -o "$PDF_PATH"
  --from=gfm+raw_attribute
  --pdf-engine=xelatex
  --top-level-division=chapter
  --number-sections
  -V secnumdepth=0
  --resource-path="$ROOT_DIR"
  --lua-filter="$SCRIPT_DIR/table-widths.lua"
  -V documentclass=report
  -V geometry:margin=1in
  -V colorlinks=true -V linkcolor=RoyalBlue -V urlcolor=RoyalBlue -V toccolor=black
  -V "title=$TITLE"
  -V "subtitle=$SUBTITLE"
  -V "author=github.com/urmzd/openlearn"
  -V "date=$BUILD_DATE"
)
$WITH_TOC && base_cmd+=(--toc --toc-depth=2)

# Fonts (best-effort; pandoc uses LaTeX defaults if none found).
mf="$(find_font SERIF_CANDIDATES)" && base_cmd+=(-V "mainfont=$mf")
sf="$(find_font SANS_CANDIDATES)" && base_cmd+=(-V "sansfont=$sf")
cf="$(find_font MONO_CANDIDATES)" && base_cmd+=(-V "monofont=$cf")

# Mermaid: render via mermaid-filter if present, else leave as code blocks.
mermaid_used=false
cmd=("${base_cmd[@]}")
if ! $SKIP_MERMAID && command -v mermaid-filter >/dev/null 2>&1; then
  # mermaid-filter reads .puppeteer.json from the cwd (ROOT_DIR). Create one
  # (so headless Chromium can run as root in CI) unless the repo ships its own.
  if [[ ! -f "$PUPPETEER_CFG" ]]; then
    echo '{"args":["--no-sandbox","--disable-setuid-sandbox"]}' > "$PUPPETEER_CFG"
    CREATED_PUPPETEER=true
  fi
  export MERMAID_FILTER_FORMAT="${MERMAID_FILTER_FORMAT:-png}"
  cmd+=(--filter mermaid-filter)
  mermaid_used=true
  info "Rendering with Mermaid diagrams..."
else
  info "Mermaid rendering disabled — diagrams will appear as code blocks."
fi

status="success"
if ! ( cd "$ROOT_DIR" && "${cmd[@]}" ); then
  if $mermaid_used; then
    info "[warn] Mermaid rendering failed; retrying with Mermaid as code blocks..."
    ( cd "$ROOT_DIR" && "${base_cmd[@]}" ) || die "pandoc failed."
    status="success-no-mermaid"
  else
    die "pandoc failed."
  fi
fi

# mermaid-filter leaves an error log in cwd on partial failures; tidy it up.
rm -f "$ROOT_DIR/mermaid-filter.err"

SIZE="$(wc -c < "$PDF_PATH" | tr -d ' ')"
info "Done: $PDF_PATH (${SIZE} bytes, status=$status)"

# --- manifest ---------------------------------------------------------------

if [[ -n "$MANIFEST" || -d "$OUTPUT_DIR" ]]; then
  manifest_file="${MANIFEST:-$OUTPUT_DIR/build-manifest.json}"
  cat > "$manifest_file" <<EOF
{"build_date":"$BUILD_DATE","output_file":"$PDF_PATH","bytes":$SIZE,"status":"$status","mermaid":$mermaid_used}
EOF
  info "Wrote manifest: $manifest_file"
fi
