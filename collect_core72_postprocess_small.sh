#!/usr/bin/env bash
set -euo pipefail

# Collect ONLY lightweight files from Core72 postprocess results.
# Run from the repository root.
#
# Source:
#   results/acm_revision/core72_modality_first_v2_postprocess_r150
#
# Excluded:
#   *.npz *.npy *.pt *.pth *.ckpt *.bin *.pkl *.pickle *.safetensors
#   any file larger than MAX_MB (default: 10 MB)

SRC="results/acm_revision/core72_modality_first_v2_postprocess_r150"
OUT="core72_postprocess_small"
ARCHIVE="${OUT}.tar.gz"
MAX_MB="${MAX_MB:-10}"
MAX_BYTES=$((MAX_MB * 1024 * 1024))

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

echo "Collecting Core72 postprocess results..."
echo "Source : $SRC"
echo "Max file size: ${MAX_MB} MB"
echo

while IFS= read -r -d '' f; do
  # Skip raw/binary arrays and checkpoints.
  case "$f" in
    *.npz|*.npy|*.pt|*.pth|*.ckpt|*.bin|*.pkl|*.pickle|*.safetensors|*.h5|*.hdf5)
      echo "[BINARY] $f" >> "$SKIPPED"
      continue
      ;;
  esac

  # Keep analysis-friendly result files only.
  case "$f" in
    *.csv|*.json|*.tsv|*.txt|*.yaml|*.yml|*.md)
      ;;
    *)
      continue
      ;;
  esac

  size_bytes=$(stat -c '%s' "$f")
  if (( size_bytes > MAX_BYTES )); then
    echo "[TOO_LARGE] $f ($(du -h "$f" | cut -f1))" >> "$SKIPPED"
    continue
  fi

  dest="$OUT/$f"
  mkdir -p "$(dirname "$dest")"
  cp -p "$f" "$dest"

  echo "$f" >> "$MANIFEST"
  copied=$((copied + 1))
done < <(find "$SRC" -type f -print0)

{
  echo
  echo "===== SUMMARY ====="
  echo "Copied files: $copied"
  echo "Source: $SRC"
  echo "Excluded binary/raw formats: npz, npy, pt, pth, ckpt, bin, pkl, pickle, safetensors, h5, hdf5"
  echo "Max included file size: ${MAX_MB} MB"
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
echo "NPZ/NPY/PT/checkpoint files were NOT included."
echo "Download this file:"
echo "  $ARCHIVE"
echo "=============================================="
