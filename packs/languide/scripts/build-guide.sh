#!/usr/bin/env bash
set -euo pipefail

# build-guide.sh — Build PDF language guides from markdown chapters.
#
# Usage:
#   ./scripts/build-guide.sh [options] [LANGUAGE...]
#
# Options:
#   --languages-dir DIR   Directory containing language folders (default: languages)
#   --output-dir DIR      Output directory for PDFs (default: outputs)
#   --skip-font-check     Skip CJK font detection
#   --continue-on-error   Keep going if a language fails
#   --manifest FILE       Write build manifest JSON to FILE
#   --discover            List available languages and exit
#   --help                Show this help
#
# Examples:
#   ./scripts/build-guide.sh                     # build all languages
#   ./scripts/build-guide.sh japanese            # build one language
#   ./scripts/build-guide.sh --discover          # list available languages

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LANGUAGES_DIR="$ROOT_DIR/languages"
OUTPUT_DIR="$ROOT_DIR/outputs"
TEMPLATE="$ROOT_DIR/templates/tourism-guide.tex"
SKIP_FONT_CHECK=false
CONTINUE_ON_ERROR=false
MANIFEST=""
DISCOVER=false
LANGUAGES=()

# CJK languages that need special font handling
CJK_LANGUAGES="japanese chinese korean"

# Font candidates (tried in order) — used via nameref in find_font()
# shellcheck disable=SC2034
CJK_FONT_CANDIDATES=(
  "Noto Sans CJK JP"
  "Noto Serif CJK JP"
  "Hiragino Sans W3"
  "Hiragino Mincho ProN"
  "Source Han Sans"
  "IPAGothic"
  "IPAMincho"
)

# shellcheck disable=SC2034
LATIN_SERIF_CANDIDATES=("DejaVu Serif" "Liberation Serif" "Palatino Linotype" "Palatino")
# shellcheck disable=SC2034
LATIN_SANS_CANDIDATES=("DejaVu Sans" "Liberation Sans" "Lato")

# --- helpers ----------------------------------------------------------------

die() { echo "[error] $*" >&2; exit 1; }
info() { echo "[info] $*"; }

usage() {
  sed -n '3,/^$/s/^# \?//p' "$0"
  exit 0
}

# Find the first available font from a candidate list via fc-list.
find_font() {
  local -n candidates=$1
  command -v fc-list >/dev/null 2>&1 || return 1
  local available
  available="$(fc-list --format '%{family}\n' 2>/dev/null)" || return 1
  for candidate in "${candidates[@]}"; do
    if [[ "$(echo "$available" | grep -ciF "$candidate")" -gt 0 ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

is_cjk() {
  local lang="$1"
  for cjk in $CJK_LANGUAGES; do
    [[ "$lang" == "$cjk" ]] && return 0
  done
  return 1
}

check_cjk_fonts() {
  if [[ "$(uname)" == "Darwin" ]]; then
    system_profiler SPFontsDataType 2>/dev/null | grep -qi 'hiragino\|noto sans cjk\|noto serif cjk\|source han\|ipa'
  elif command -v fc-list >/dev/null 2>&1; then
    fc-list :lang=ja 2>/dev/null | grep -q .
  else
    return 1
  fi
}

# Discover languages: directories under $LANGUAGES_DIR that have chapters/*.md
discover_languages() {
  for dir in "$LANGUAGES_DIR"/*/; do
    [[ -d "$dir" ]] || continue
    local slug
    slug="$(basename "$dir")"
    [[ "$slug" == .* ]] && continue
    local chapters_dir="$dir/chapters"
    [[ -d "$chapters_dir" ]] || continue
    local count
    count="$(find "$chapters_dir" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l | tr -d ' ')"
    [[ "$count" -gt 0 ]] || continue
    echo "$slug"
  done
}

# Build a single language guide. Prints the output PDF path on success.
build_language() {
  local slug="$1"
  local lang_dir="$LANGUAGES_DIR/$slug"
  local chapters_dir="$lang_dir/chapters"

  [[ -d "$chapters_dir" ]] || die "No chapters/ directory in $lang_dir"

  # CJK font check
  if ! $SKIP_FONT_CHECK && is_cjk "$slug"; then
    info "Checking CJK fonts for $slug..."
    check_cjk_fonts || die "No CJK fonts found. Install fonts-noto-cjk or use --skip-font-check."
  fi

  # Collect and sort chapter files
  local -a chapters
  mapfile -t chapters < <(find "$chapters_dir" -maxdepth 1 -name '*.md' -print | sort)
  [[ ${#chapters[@]} -gt 0 ]] || die "No markdown files in $chapters_dir"

  info "Combining ${#chapters[@]} chapters for $slug..."

  # Combine chapters
  local combined_dir="$lang_dir/outputs"
  mkdir -p "$combined_dir"
  local combined_md="$combined_dir/${slug}-guide.md"
  cat "${chapters[@]}" > "$combined_md"

  # Build pandoc command
  local pdf_path="$combined_dir/${slug}-guide.pdf"
  local -a cmd=(
    pandoc "$combined_md" -o "$pdf_path"
    --from=markdown
    --pdf-engine xelatex
    --toc --toc-depth=3 --number-sections
    -V "title=${slug^} Tourist Guide"
    -V documentclass=article
    -V geometry:margin=0.75in
    -V colorlinks=true
    -V linkcolor=NavyBlue
    -V urlcolor=NavyBlue
  )

  # Add template if it exists
  [[ -f "$TEMPLATE" ]] && cmd+=(--template "$TEMPLATE")

  # CJK support
  if is_cjk "$slug"; then
    cmd+=(-V usecjk=true)
    local cjk_font=""
    for _f in "${CJK_FONT_CANDIDATES[@]}"; do
      if [[ "$(fc-list --format '%{family}\n' 2>/dev/null | grep -ciF "$_f")" -gt 0 ]]; then
        cjk_font="$_f"; break
      fi
    done
    [[ -n "$cjk_font" ]] && cmd+=(-V "CJKmainfont=$cjk_font")
  fi

  # Latin fonts
  local main_font sans_font
  main_font="$(find_font LATIN_SERIF_CANDIDATES)" && cmd+=(-V "mainfont=$main_font")
  sans_font="$(find_font LATIN_SANS_CANDIDATES)" && cmd+=(-V "sansfont=$sans_font")

  info "Rendering PDF with pandoc..."
  "${cmd[@]}" || die "pandoc failed for $slug"

  # Move to output dir
  mkdir -p "$OUTPUT_DIR"
  local final_pdf="$OUTPUT_DIR/${slug}-guide.pdf"
  mv "$pdf_path" "$final_pdf"

  # Clean up combined markdown
  rm -f "$combined_md"

  echo "$final_pdf"
}

# --- argument parsing -------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --languages-dir) LANGUAGES_DIR="$2"; shift 2 ;;
    --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
    --skip-font-check) SKIP_FONT_CHECK=true; shift ;;
    --continue-on-error) CONTINUE_ON_ERROR=true; shift ;;
    --manifest)      MANIFEST="$2"; shift 2 ;;
    --discover)      DISCOVER=true; shift ;;
    --help|-h)       usage ;;
    -*)              die "Unknown option: $1" ;;
    *)               LANGUAGES+=("$1"); shift ;;
  esac
done

# --- main -------------------------------------------------------------------

command -v pandoc >/dev/null 2>&1 || die "pandoc is required. Install it (e.g. brew install pandoc)."

if $DISCOVER; then
  discover_languages
  exit 0
fi

# If no languages specified, build all
if [[ ${#LANGUAGES[@]} -eq 0 ]]; then
  mapfile -t LANGUAGES < <(discover_languages)
fi

[[ ${#LANGUAGES[@]} -gt 0 ]] || die "No languages found in $LANGUAGES_DIR"

info "Building ${#LANGUAGES[@]} language(s)..."

success=0
failure=0
manifest_builds=""

for slug in "${LANGUAGES[@]}"; do
  info "[$slug] Building..."
  if pdf_path="$(build_language "$slug" 2>&1)"; then
    # Last line of output is the PDF path
    final_line="$(echo "$pdf_path" | tail -1)"
    info "[$slug] Done: $final_line"
    manifest_builds="${manifest_builds}{\"slug\":\"$slug\",\"status\":\"success\",\"output_file\":\"$final_line\"},"
    ((++success))
  else
    echo "$pdf_path" >&2
    info "[$slug] Failed" >&2
    manifest_builds="${manifest_builds}{\"slug\":\"$slug\",\"status\":\"failed\",\"error\":\"build failed\"},"
    ((++failure))
    $CONTINUE_ON_ERROR || exit 1
  fi
done

# Write manifest
manifest_builds="${manifest_builds%,}"  # trim trailing comma
manifest_json="{\"build_time\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"languages_dir\":\"$LANGUAGES_DIR\",\"output_dir\":\"$OUTPUT_DIR\",\"builds\":[${manifest_builds}]}"

manifest_file="${MANIFEST:-$OUTPUT_DIR/build-manifest.json}"
mkdir -p "$(dirname "$manifest_file")"
echo "$manifest_json" > "$manifest_file"

info "Build complete: $success succeeded, $failure failed"
[[ $failure -eq 0 ]] || exit 1
