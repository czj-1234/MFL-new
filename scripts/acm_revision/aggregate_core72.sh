#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 <postprocess-root-A-or-merged-root> [postprocess-root-B]" >&2
  exit 2
fi

ROOT_A="$1"
ROOT_B="${2:-}"
OUTPUT_DIR="${CORE72_FINAL_ROOT:-results/acm_revision/core72_final}"

if [[ ! -d "${ROOT_A}" ]]; then
  echo "Missing postprocess root: ${ROOT_A}" >&2
  exit 2
fi

if [[ -n "${ROOT_B}" ]]; then
  if [[ ! -d "${ROOT_B}" ]]; then
    echo "Missing postprocess root: ${ROOT_B}" >&2
    exit 2
  fi
  python -m src.acm_revision.core72_aggregate \
    --post-roots "${ROOT_A}" "${ROOT_B}" \
    --output-dir "${OUTPUT_DIR}"
else
  python -m src.acm_revision.core72_aggregate \
    --post-roots "${ROOT_A}" \
    --output-dir "${OUTPUT_DIR}"
fi

[[ -f "${OUTPUT_DIR}/CORE72_AGGREGATE_PASS" ]] || {
  echo "Core72 aggregate did not produce PASS marker." >&2
  exit 1
}
echo "[CORE72 AGGREGATE PASS] ${OUTPUT_DIR}"
