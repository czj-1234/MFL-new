#!/usr/bin/env bash
set -euo pipefail

# Collect ONLY lightweight result files from:
#   1) Core72 FL runs
#   2) Defense70 FL runs
#
# Explicitly NOT included:
#   - Core72 postprocess/attack directory
#   - Defense preparation directory
#   - NPZ/NPY/PT/PTH/checkpoints and other large binary files

MAX_MB="${MAX_MB:-20}"
OUT="${OUT:-acm_core72_defense70_small}"
ARCHIVE="${OUT}.tar.gz"

ROOTS=(
  "results/acm_revision/core72_modality_first_v2_r150"
  "results/acm_revision/defense70_r150"
)

rm -rf "$OUT"
mkdir -p "$OUT"

MANIFEST="$OUT/_MANIFEST.txt"
SKIPPED="$OUT/_SKIPPED.txt"
: > "$MANIFEST"
: > "$SKIPPED"

copied=0
max_bytes=$((MAX_MB * 1024 * 1024))

echo "Collecting lightweight files from Core72 + Defense70 only..."
echo

for root in "${ROOTS[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "[WARN] Missing directory: $root" | tee -a "$SKIPPED"
    continue
  fi

  echo "[SCAN] $root"

  while IFS= read -r -d '' f; do
    case "$f" in
      *.npz|*.npy|*.pt|*.pth|*.ckpt|*.bin|*.pkl|*.pickle|*.safetensors)
        echo "[BINARY] $f" >> "$SKIPPED"
        continue
        ;;
    esac

    case "$f" in
      *.json|*.csv|*.tsv|*.yaml|*.yml|*.txt|*.md)
        ;;
      *)
        continue
        ;;
    esac

    size_bytes=$(stat -c '%s' "$f")
    if (( size_bytes > max_bytes )); then
      echo "[TOO_LARGE] $f ($(du -h "$f" | cut -f1))" >> "$SKIPPED"
      continue
    fi

    dest="$OUT/$f"
    mkdir -p "$(dirname "$dest")"
    cp -p "$f" "$dest"

    echo "$f" >> "$MANIFEST"
    copied=$((copied + 1))
  done < <(find "$root" -type f -print0)
done

{
  echo
  echo "===== SUMMARY ====="
  echo "Copied files: $copied"
  echo "Sources:"
  printf '  %s\n' "${ROOTS[@]}"
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
echo "Included only:"
echo "  Core72   -> results/acm_revision/core72_modality_first_v2_r150"
echo "  Defense70-> results/acm_revision/defense70_r150"
echo
echo "NPZ/NPY/PT/PTH/checkpoints were excluded."
echo "=============================================="
