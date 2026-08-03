#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
ROUNDS=150
SEEDS=(42 43 44 45 46)
CONCENTRATIONS=(0.9 1.0)
POPULATIONS=(shadow_train shadow_val target)
JOB_SCRIPT="scripts/acm_revision/run_privacy_capture_job.sh"
SMOKE_SCRIPT="scripts/acm_revision/smoke_test_privacy_pipeline.sh"
RESULT_ROOT="results/acm_revision/privacy_capture_r${ROUNDS}"
POST_ROOT="results/acm_revision/privacy_postprocess_r${ROUNDS}"

for file in "${JOB_SCRIPT}" "${SMOKE_SCRIPT}"; do
  [[ -f "${file}" ]] || { echo "Missing ${file}. Run from repository root." >&2; exit 2; }
done

if [[ "${RUN_PRIVACY_SMOKE:-1}" == "1" ]]; then
  bash "${SMOKE_SCRIPT}" "${GPU}"
fi

echo "============================================================"
echo "Server B COMPLETE privacy queue"
echo "GPU ${GPU}: concentrations ${CONCENTRATIONS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Populations per seed/concentration: ${POPULATIONS[*]}"
echo "Rounds/job: ${ROUNDS}"
echo "Total on Server B: 30 FL jobs"
echo "============================================================"

for concentration in "${CONCENTRATIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for population in "${POPULATIONS[@]}"; do
      echo "[SERVER B] GPU=${GPU} c=${concentration} seed=${seed} population=${population}"
      bash "${JOB_SCRIPT}" "${GPU}" "${concentration}" "${ROUNDS}" "${seed}" "${population}"
    done
  done
done

if [[ "${RUN_PRIVACY_POSTPROCESS:-1}" == "1" ]]; then
  for concentration in "${CONCENTRATIONS[@]}"; do
    echo "[POSTPROCESS] Server B concentration=${concentration}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -m src.acm_revision.privacy_postprocess \
      --root "${RESULT_ROOT}" \
      --concentration "${concentration}" \
      --rounds "${ROUNDS}" \
      --seeds "42,43,44,45,46" \
      --output-root "${POST_ROOT}"
    ctag="${concentration//./p}"
    python -m src.acm_revision.privacy_validate \
      --output-dir "${POST_ROOT}/c${ctag}_r${ROUNDS}"
  done
fi

echo "[DONE] Server B privacy capture and validated post-processing completed."
