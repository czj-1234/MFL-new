#!/usr/bin/env bash
set -euo pipefail

# Collect Core72 postprocess results while excluding only known large/binary formats.
# NO file-size limit is applied.
# Run from the repository root.

SRC="results/acm_revision/core72_modality_first_v2_postprocess_r150"
OUT="core72_postprocess_small"
ARCHIVE="${OUT}.tar.gz"

if [[ ! -d "$SRC" ]]; then
  echo "[ERROR] Source directory not found:"
  echo "  $SRC"
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"

MANIFEST="$OUT/_MANIFEST.txt"
SKIPPED="$OUT/_SKIPPED.txt"
: > "$MANIFEST"
: > "$SKIPPED"

copied=0

echo "Collecting Core72 postprocess result files..."
echo "Source: $SRC"
echo "No file-size limit."
echo

while IFS= read -r -d '' f; do
  # Exclude only known raw/binary/checkpoint formats.
  case "$f" in
    *.npz|*.npy|*.pt|*.pth|*.ckpt|*.bin|*.pkl|*.pickle|*.safetensors|*.h5|*.hdf5)
      echo "[EXCLUDED_BINARY] $f" >> "$SKIPPED"
      continue
      ;;
  esac

  # Keep analysis-friendly result/metadata files regardless of size.
  case "$f" in
    *.csv|*.json|*.tsv|*.txt|*.yaml|*.yml|*.md)
      ;;
    *)
      echo "[EXCLUDED_OTHER] $f" >> "$SKIPPED"
      continue
      ;;
  esac

  dest="$OUT/$f"
  mkdir -p "$(dirname "$dest")"
  cp -p "$f" "$dest"

  printf '%s\t%s\n' "$(du -h "$f" | cut -f1)" "$f" >> "$MANIFEST"
  copied=$((copied + 1))
done < <(find "$SRC" -type f -print0)

{
  echo
  echo "===== SUMMARY ====="
  echo "Copied files: $copied"
  echo "Source: $SRC"
  echo "No file-size limit was used."
  echo "Excluded formats: npz, npy, pt, pth, ckpt, bin, pkl, pickle, safetensors, h5, hdf5"
} >> "$MANIFEST"

rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" "$OUT"

echo
echo "=============================================="
echo "[DONE]"
echo "Archive: $ARCHIVE"
echo "Size   : $(du -h "$ARCHIVE" | cut -f1)"
echo "Files  : $copied"
echo
echo "No size limit was applied."
echo "Only binary/raw formats such as NPZ/NPY/PT were excluded."
echo "Download:"
echo "  $ARCHIVE"
echo "=============================================="
