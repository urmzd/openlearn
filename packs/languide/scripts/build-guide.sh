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
#   --skip-font-check     Skip font detection
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

RTL_FILTER="$ROOT_DIR/scripts/rtl.lua"

# Languages by script. Each group needs its own font handling.
CJK_LANGUAGES="japanese chinese mandarin korean"
CHINESE_LANGUAGES="chinese mandarin"
RTL_LANGUAGES="arabic persian farsi tajik"
CYRILLIC_LANGUAGES="russian tajik"

# Font candidates (tried in order), used via nameref in find_font()
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

# Simplified Chinese needs its own glyph shapes, not the Japanese ones.
# shellcheck disable=SC2034
CHINESE_FONT_CANDIDATES=(
  "Noto Sans CJK SC"
  "Noto Serif CJK SC"
  "Source Han Sans SC"
  "PingFang SC"
  "Songti SC"
  "Heiti SC"
  "WenQuanYi Micro Hei"
)

# Arabic-script fonts that also carry the Persian letters.
# shellcheck disable=SC2034
RTL_FONT_CANDIDATES=(
  "Noto Naskh Arabic"
  "Amiri"
  "Scheherazade New"
  "Geeza Pro"
  "Al Bayan"
  "DejaVu Sans"
)

# DejaVu covers Cyrillic (with the Tajik letters), pinyin tone marks and
# the dotted letters used in Arabic transliteration, so it comes first.
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

# in_group SLUG "space separated list"
in_group() {
  local lang="$1" item
  for item in $2; do
    [[ "$lang" == "$item" ]] && return 0
  done
  return 1
}

is_cjk() { in_group "$1" "$CJK_LANGUAGES"; }

# Title on the PDF cover and in the page header.
guide_title() {
  case "$1" in
    tajik) echo "Tajik and Persian Communication Guide" ;;
    mandarin) echo "Mandarin Chinese Communication Guide" ;;
    *) echo "${1^} Communication Guide" ;;
  esac
}

# Succeeds when a font for the language's CJK script is installed.
check_cjk_fonts() {
  if in_group "$1" "$CHINESE_LANGUAGES"; then
    find_font CHINESE_FONT_CANDIDATES >/dev/null
  else
    find_font CJK_FONT_CANDIDATES >/dev/null
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
    check_cjk_fonts "$slug" || die "No CJK fonts found. Install fonts-noto-cjk or use --skip-font-check."
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
    -V "title=$(guide_title "$slug")"
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
    if in_group "$slug" "$CHINESE_LANGUAGES"; then
      cjk_font="$(find_font CHINESE_FONT_CANDIDATES)" || cjk_font=""
    else
      cjk_font="$(find_font CJK_FONT_CANDIDATES)" || cjk_font=""
    fi
    [[ -n "$cjk_font" ]] && cmd+=(-V "CJKmainfont=$cjk_font")
  fi

  # Right-to-left script support (Arabic, Persian)
  if in_group "$slug" "$RTL_LANGUAGES"; then
    local rtl_font=""
    rtl_font="$(find_font RTL_FONT_CANDIDATES)" || rtl_font=""
    if [[ -z "$rtl_font" ]]; then
      $SKIP_FONT_CHECK || die "No Arabic-script font found. Install fonts-noto-core (Noto Naskh Arabic) or use --skip-font-check."
      rtl_font="Noto Naskh Arabic"
    fi
    cmd+=(-V "rtlfont=$rtl_font" --lua-filter "$RTL_FILTER")
  fi

  # Cyrillic needs a text font that covers it
  if in_group "$slug" "$CYRILLIC_LANGUAGES" && ! $SKIP_FONT_CHECK; then
    local cyrillic_fonts
    cyrillic_fonts="$(fc-list :lang=ru --format '%{family}\n' 2>/dev/null || true)"
    grep -qiE 'dejavu|liberation|palatino' <<<"$cyrillic_fonts" \
      || die "No Cyrillic text font found. Install fonts-dejavu or use --skip-font-check."
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
