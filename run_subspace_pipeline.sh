#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Counterfactual Subspace Pipeline
# 1) Learn split A/B/C rank-5 subspaces
# 2) Evaluate subspace stability
# 3) Combine all shadow raw updates
# 4) Learn final rank-5 subspace
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If this script is copied into the project root, use that directory.
# Otherwise, run it from the current working directory.
if [[ -d "${ROOT_DIR}/src" && -d "${ROOT_DIR}/results" ]]; then
  PROJECT_DIR="${ROOT_DIR}"
else
  PROJECT_DIR="$(pwd)"
fi

cd "${PROJECT_DIR}"

BASE="results/defense_seed42_assoc07"
SHADOW="${BASE}/shadow"
SUBSPACE="${BASE}/subspace"
COMBINED="${SUBSPACE}/combined"
LOG_DIR="logs"

mkdir -p "${SUBSPACE}" "${COMBINED}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/subspace_pipeline.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "Counterfactual Subspace Pipeline"
echo "Project directory: ${PROJECT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Started at: $(date)"
echo "============================================================"

required_files=()

for S in A B C; do
  required_files+=(
    "${SHADOW}/split_${S}/modality_exclusive/assoc_07/classifier_updates_raw.npz"
    "${SHADOW}/split_${S}/modality_exclusive/counterfactual/classifier_updates_raw.npz"
  )
done

for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "[ERROR] Missing required file:"
    echo "        ${file}"
    exit 1
  fi
done

echo
echo "[1/4] Learning split-specific rank-5 subspaces..."

for S in A B C; do
  echo
  echo "---- Split ${S} ----"

  python -u -m src.analysis.learn_counterfactual_subspace \
    --associated "${SHADOW}/split_${S}/modality_exclusive/assoc_07/classifier_updates_raw.npz" \
    --counterfactual "${SHADOW}/split_${S}/modality_exclusive/counterfactual/classifier_updates_raw.npz" \
    --rank 5 \
    --output "${SUBSPACE}/split_${S}_r5.npz"

  if [[ ! -f "${SUBSPACE}/split_${S}_r5.npz" ]]; then
    echo "[ERROR] Failed to create split_${S}_r5.npz"
    exit 1
  fi
done

echo
echo "[2/4] Evaluating A/B/C subspace stability..."

python -u -m src.analysis.subspace_stability \
  --basis-a "${SUBSPACE}/split_A_r5.npz" \
  --basis-b "${SUBSPACE}/split_B_r5.npz" \
  --basis-c "${SUBSPACE}/split_C_r5.npz" \
  --output "${SUBSPACE}/stability.json"

if [[ ! -f "${SUBSPACE}/stability.json" ]]; then
  echo "[ERROR] Failed to create stability.json"
  exit 1
fi

echo
echo "[3/4] Combining A/B/C raw shadow updates..."

python -u -m src.analysis.combine_shadow_updates \
  --associated \
    "${SHADOW}/split_A/modality_exclusive/assoc_07/classifier_updates_raw.npz" \
    "${SHADOW}/split_B/modality_exclusive/assoc_07/classifier_updates_raw.npz" \
    "${SHADOW}/split_C/modality_exclusive/assoc_07/classifier_updates_raw.npz" \
  --counterfactual \
    "${SHADOW}/split_A/modality_exclusive/counterfactual/classifier_updates_raw.npz" \
    "${SHADOW}/split_B/modality_exclusive/counterfactual/classifier_updates_raw.npz" \
    "${SHADOW}/split_C/modality_exclusive/counterfactual/classifier_updates_raw.npz" \
  --output-dir "${COMBINED}"

if [[ ! -f "${COMBINED}/associated_all.npz" ]]; then
  echo "[ERROR] Missing combined associated_all.npz"
  exit 1
fi

if [[ ! -f "${COMBINED}/counterfactual_all.npz" ]]; then
  echo "[ERROR] Missing combined counterfactual_all.npz"
  exit 1
fi

echo
echo "[4/4] Learning final rank-5 counterfactual subspace..."

python -u -m src.analysis.learn_counterfactual_subspace \
  --associated "${COMBINED}/associated_all.npz" \
  --counterfactual "${COMBINED}/counterfactual_all.npz" \
  --rank 5 \
  --output "${SUBSPACE}/final_r5.npz"

if [[ ! -f "${SUBSPACE}/final_r5.npz" ]]; then
  echo "[ERROR] Failed to create final_r5.npz"
  exit 1
fi

echo
echo "============================================================"
echo "ALL SUBSPACE STEPS COMPLETED"
echo "Finished at: $(date)"
echo
echo "Generated files:"
echo "  ${SUBSPACE}/split_A_r5.npz"
echo "  ${SUBSPACE}/split_B_r5.npz"
echo "  ${SUBSPACE}/split_C_r5.npz"
echo "  ${SUBSPACE}/stability.json"
echo "  ${COMBINED}/associated_all.npz"
echo "  ${COMBINED}/counterfactual_all.npz"
echo "  ${SUBSPACE}/final_r5.npz"
echo "============================================================"
