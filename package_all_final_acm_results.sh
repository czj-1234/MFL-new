#!/usr/bin/env bash
set -euo pipefail

# Final lightweight package for ACM revision analysis.
#
# INCLUDED source folders:
#   1) Core72 FL results
#   2) Core72 postprocess / attack results
#   3) Defense70 FL results
#   4) Defense70 defense-aware postprocess results
#   5) Defense70 preparation/calibration summaries
#
# NOT included:
#   - raw update arrays / checkpoints (*.npz, *.npy, *.pt, ...)
#   - update_metadata.csv
#   - logs
#   - smoke/preflight/diagnostics folders
#
# No size limit is applied to allowed result files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_DIR="$(pwd)"

find_repo_root() {
  local d
  for d in \
    "$SCRIPT_DIR" "$START_DIR" \
    "$SCRIPT_DIR/.." "$SCRIPT_DIR/../.." "$SCRIPT_DIR/../../.." \
    "$START_DIR/.." "$START_DIR/../.." "$START_DIR/../../.."
  do
    if [[ -d "$d/src/acm_revision" && -d "$d/results/acm_revision" ]]; then
      (cd "$d" && pwd)
      return 0
    fi
  done
  return 1
}

REPO_ROOT="$(find_repo_root || true)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "[ERROR] Could not locate the MFL-new repository root."
  exit 1
fi
cd "$REPO_ROOT"

OUT="acm_final_results_small"
ARCHIVE="${OUT}.tar.gz"

ROOTS=(
  "results/acm_revision/core72_modality_first_v2_r150"
  "results/acm_revision/core72_modality_first_v2_postprocess_r150"
  "results/acm_revision/defense70_r150"
  "results/acm_revision/defense70_postprocess_r150"
  "results/acm_revision/defense70_prep"
)

rm -rf "$OUT"
mkdir -p "$OUT"

MANIFEST="$OUT/_MANIFEST.txt"
SKIPPED="$OUT/_SKIPPED.txt"
: > "$MANIFEST"
: > "$SKIPPED"

copied=0

echo "[INFO] Repository root: $REPO_ROOT"
echo "[INFO] Collecting final ACM result files..."
echo

for root in "${ROOTS[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "[WARN] Missing folder: $root" | tee -a "$SKIPPED"
    continue
  fi

  echo "[SCAN] $root"

  while IFS= read -r -d '' f; do
    base="$(basename "$f")"

    # Exclude raw/binary/update/checkpoint payloads.
    case "$f" in
      *.npz|*.npy|*.pt|*.pth|*.ckpt|*.bin|*.pkl|*.pickle|*.safetensors|*.h5|*.hdf5)
        echo "[BINARY] $f" >> "$SKIPPED"
        continue
        ;;
    esac

    # update_metadata.csv is large and no longer needed because postprocess is complete.
    if [[ "$base" == "update_metadata.csv" ]]; then
      echo "[RAW_METADATA] $f" >> "$SKIPPED"
      continue
    fi

    keep=0
    case "$f" in
      *.csv|*.json|*.tsv|*.yaml|*.yml|*.txt|*.md)
        keep=1
        ;;
    esac

    case "$base" in
      CORE72_VALIDATION_PASS|VALIDATION_PASS|DEFENSE_POSTPROCESS_PASS|PREPARATION_READY|READY)
        keep=1
        ;;
    esac

    if [[ "$keep" -ne 1 ]]; then
      echo "[OTHER] $f" >> "$SKIPPED"
      continue
    fi

    dest="$OUT/$f"
    mkdir -p "$(dirname "$dest")"
    cp -p "$f" "$dest"

    printf '%s\t%s\n' "$(du -h "$f" | cut -f1)" "$f" >> "$MANIFEST"
    copied=$((copied + 1))
  done < <(find "$root" -type f -print0)
done

{
  echo
  echo "===== SUMMARY ====="
  echo "Copied files: $copied"
  echo "Repository root: $REPO_ROOT"
  echo
  echo "Included source folders:"
  printf '  %s\n' "${ROOTS[@]}"
  echo
  echo "Excluded:"
  echo "  raw NPZ/NPY/PT/PTH/checkpoints and other binary arrays"
  echo "  update_metadata.csv"
  echo "  unrelated smoke/preflight/diagnostics directories"
  echo
  echo "No size limit was applied to allowed result files."
} >> "$MANIFEST"

rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" "$OUT"

echo
echo "=================================================="
echo "[DONE]"
echo "Archive : $REPO_ROOT/$ARCHIVE"
echo "Size    : $(du -h "$ARCHIVE" | cut -f1)"
echo "Files   : $copied"
echo
echo "Download this file:"
echo "  $ARCHIVE"
echo "=================================================="
