#!/usr/bin/env bash
# Convert workshop handout markdown files to PDF (requires pandoc + xelatex).
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found. Install: brew install pandoc basictex"
  exit 1
fi

ENGINE="${PDF_ENGINE:-xelatex}"
MARGIN="${PDF_MARGIN:-0.75in}"

for f in practitioner_7109_2page practitioner_7225_2page practitioner_7149_2page technical_reference_1page; do
  if [[ ! -f "${f}.md" ]]; then
    echo "Skip (missing): ${f}.md"
    continue
  fi
  echo "→ ${f}.pdf"
  pandoc "${f}.md" -o "${f}.pdf" \
    --pdf-engine="$ENGINE" \
    -V "geometry:margin=${MARGIN}"
done

echo "Done. PDFs in $(pwd)"
