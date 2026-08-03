#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROUNDS=2
CONCENTRATION=0.7
ROOT="results/acm_revision/core72_smoke_r${ROUNDS}"
POST_ROOT="results/acm_revision/core72_smoke_postprocess_r${ROUNDS}"
PASS_MARKER="${POST_ROOT}/ALL_SETTINGS_SMOKE_PASS"
JOB_SCRIPT="scripts/acm_revision/run_core72_job.sh"

if [[ -f "${PASS_MARKER}" ]]; then
  echo "[CORE72 SMOKE SKIP] ${PASS_MARKER}"
  exit 0
fi

export CORE72_RESULT_ROOT="${ROOT}"
export CORE72_CONFIG_ROOT="configs/acm_revision/generated/core72_smoke_r${ROUNDS}"
export CORE72_LOG_ROOT="logs/acm_revision/core72_smoke_r${ROUNDS}"
export CORE72_NUM_CLIENTS=4
export CORE72_SAMPLES_PER_CLIENT=8
export CORE72_MAX_LOCAL_STEPS=1
export CORE72_SKETCH_DIM=256
export GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-12000}"

for setting in image_only text_only modality_exclusive; do
  bash "${JOB_SCRIPT}" "${GPU}" "${setting}" "${CONCENTRATION}" shadow_train 991 "${ROUNDS}"
  bash "${JOB_SCRIPT}" "${GPU}" "${setting}" "${CONCENTRATION}" shadow_val 992 "${ROUNDS}"
  bash "${JOB_SCRIPT}" "${GPU}" "${setting}" "${CONCENTRATION}" target 993 "${ROUNDS}"

  python -m src.acm_revision.core72_postprocess \
    --root "${ROOT}" \
    --setting "${setting}" \
    --concentration "${CONCENTRATION}" \
    --rounds "${ROUNDS}" \
    --output-root "${POST_ROOT}" \
    --shadow-train-seeds 991 \
    --shadow-val-seeds 992 \
    --target-seeds 993 \
    --smoke

  ctag="${CONCENTRATION//./p}"
  output_dir="${POST_ROOT}/${setting}__c${ctag}__r${ROUNDS}"
  python -m src.acm_revision.core72_validate \
    --output-dir "${output_dir}" \
    --target-seeds 993 \
    --smoke
done

mkdir -p "$(dirname "${PASS_MARKER}")"
printf 'PASS\n' > "${PASS_MARKER}"
echo "[CORE72 SMOKE PASS] all three settings passed on GPU ${GPU}"
