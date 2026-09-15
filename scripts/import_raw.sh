#!/usr/bin/env bash
# Copy the frozen source data out of the original project into data/raw/ and
# write a checksum manifest (data/raw_manifest.sha256, tracked in git).
#
# Only true sources are copied; everything derived is rebuilt by `make`.
# Small hand-curated inputs live in data/manual/ and are tracked in git.
#
#   scripts/import_raw.sh ~/Dropbox/1848_unis
set -euo pipefail

SRC="${1:?usage: scripts/import_raw.sh /path/to/1848_unis}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/raw"
mkdir -p "$DEST/universities" "$DEST/tgn_rel_0126" "$DEST/ghgis"

echo "register extractions (CSV only; the JSON twins are not used)"
rsync -a --prune-empty-dirs --include='*/' --include='*.csv' --exclude='*' \
    "$SRC/data/universities/out_final/" "$DEST/universities/out_final/"

echo "Freiburg and Giessen parses"
rsync -a "$SRC/data/universities/raw/freiburg/freiburg.csv" \
         "$SRC/data/universities/raw/giessen/giessen.csv" "$DEST/universities/"

echo "TGN relational release (the tables unis.tgn.load reads)"
for t in TERM SUBJECT SUBJECT_RELS PTYPE_ROLE PTYPE_ROLE_RELS COORDINATES \
         LANGUAGE_RELS SOURCE_RELS_SUBJECT SCOPE_NOTES SUBJECT_MERGE; do
    rsync -a "$SRC/data/tgn_rel_0126/$t.out" "$DEST/tgn_rel_0126/"
done

echo "GHGIS boundary layers"
for l in 1820CORE 1826CORE 1830CORE 1834CORE 1848CORE \
         1820PROVINCES 1826PROVINCES 1830PROVINCES 1834PROVINCES 1848PROVINCES \
         1820DISTRICTS 1848GERMANCONFED; do
    rsync -a --exclude='.DS_Store' "$SRC/data/ghgis/GHGIS$l" "$DEST/ghgis/"
done

echo "checksums -> data/raw_manifest.sha256"
(cd "$DEST" && find . -type f ! -name '.DS_Store' | LC_ALL=C sort | xargs shasum -a 256) \
    > "$ROOT/data/raw_manifest.sha256"
echo "done: $(wc -l < "$ROOT/data/raw_manifest.sha256" | tr -d ' ') files"
