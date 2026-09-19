#!/usr/bin/env bash
set -euo pipefail

# Auto-detect MFL-new repository root whether this script is placed in:
#   - MFL-new/
#   - MFL-new/results/acm_revision/
#   - another subdirectory inside MFL-new/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_DIR="$(pwd)"

find_repo_root() {
  local d
  for d in \
    "$SCRIPT_DIR" \
    "$START_DIR" \
    "$SCRIPT_DIR/.." \
    "$SCRIPT_DIR/../.." \
    "$SCRIPT_DIR/../../.." \
    "$START_DIR/.." \
    "$START_DIR/../.." \
    "$START_DIR/../../.."
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
  echo "Script directory: $SCRIPT_DIR"
  echo "Current directory: $START_DIR"
  echo "Expected a directory containing both:"
  echo "  src/acm_revision"
  echo "  results/acm_revision"
  exit 1
fi

cd "$REPO_ROOT"
echo "[INFO] Repository root detected: $REPO_ROOT"

ROOT="results/acm_revision/core72_modality_first_v2_r150"
OUT_ROOT="results/acm_revision/core72_modality_first_v2_postprocess_r150"
CELL="${OUT_ROOT}/image_only__c0p9__r150"
LOG_DIR="logs/acm_revision/postprocess"
LOG="${LOG_DIR}/image_only_c0p9_complete.log"

if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] Core72 raw result directory not found: $REPO_ROOT/$ROOT"
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_ROOT"

# Remove only the PARTIAL derived postprocess output for image_only c=0.9.
# Raw Core72 FL results are never touched.
if [[ -d "$CELL" ]]; then
  echo "[INFO] Removing partial derived output: $CELL"
  rm -rf "$CELL"
fi

echo "[START] image_only c=0.9 Core72 postprocess"

python -u -m src.acm_revision.core72_postprocess \
  --root "$ROOT" \
  --setting image_only \
  --concentration 0.9 \
  --rounds 150 \
  --output-root "$OUT_ROOT" \
  2>&1 | tee "$LOG"

echo "[VALIDATE]"
python -u -m src.acm_revision.core72_validate \
  --output-dir "$CELL" \
  2>&1 | tee -a "$LOG"

if [[ ! -f "$CELL/CORE72_VALIDATION_PASS" ]]; then
  echo "[ERROR] Validation marker was not created."
  exit 2
fi

echo
echo "=================================================="
echo "[DONE] image_only c=0.9 postprocess complete"
echo "Output: $REPO_ROOT/$CELL"
echo "Log:    $REPO_ROOT/$LOG"
echo "=================================================="
