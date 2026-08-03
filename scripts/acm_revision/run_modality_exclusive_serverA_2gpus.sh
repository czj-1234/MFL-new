#!/usr/bin/env bash
set -euo pipefail

GPU0="${1:-0}"
GPU1="${2:-1}"
ROUNDS=150
SEEDS=(42 43 44 45 46)
POPULATIONS=(shadow_train shadow_val target)
JOB_SCRIPT="scripts/acm_revision/run_privacy_capture_job.sh"
SMOKE_SCRIPT="scripts/acm_revision/smoke_test_privacy_pipeline.sh"
RESULT_ROOT="results/acm_revision/privacy_capture_r${ROUNDS}"
POST_ROOT="results/acm_revision/privacy_postprocess_r${ROUNDS}"

if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU IDs must be different." >&2
  exit 2
fi
for file in "${JOB_SCRIPT}" "${SMOKE_SCRIPT}"; do
  [[ -f "${file}" ]] || { echo "Missing ${file}. Run from repository root." >&2; exit 2; }
done

if [[ "${RUN_PRIVACY_SMOKE:-1}" == "1" ]]; then
  bash "${SMOKE_SCRIPT}" "${GPU0}"
fi

echo "============================================================"
echo "Server A COMPLETE privacy queue"
echo "GPU ${GPU0}: concentrations 0.5 and 0.8"
echo "GPU ${GPU1}: concentrations 0.6 and 0.9"
echo "Seeds: ${SEEDS[*]}"
echo "Populations per seed/concentration: ${POPULATIONS[*]}"
echo "Rounds/job: ${ROUNDS}"
echo "Total on Server A: 60 FL jobs"
echo "============================================================"

run_queue() {
  local gpu="$1"
  shift
  local concentrations=("$@")
  local concentration seed population
  for concentration in "${concentrations[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for population in "${POPULATIONS[@]}"; do
        echo "[SERVER A] GPU=${gpu} c=${concentration} seed=${seed} population=${population}"
        bash "${JOB_SCRIPT}" "${gpu}" "${concentration}" "${ROUNDS}" "${seed}" "${population}"
      done
    done
  done
}

run_queue "${GPU0}" 0.5 0.8 &
PID0=$!
run_queue "${GPU1}" 0.6 0.9 &
PID1=$!

cleanup() {
  kill -TERM "${PID0}" "${PID1}" 2>/dev/null || true
}
trap cleanup INT TERM

STATUS0=0
STATUS1=0
wait "${PID0}" || STATUS0=$?
wait "${PID1}" || STATUS1=$?
trap - INT TERM

if (( STATUS0 != 0 || STATUS1 != 0 )); then
  echo "[FAIL] Server A training queues exited with ${STATUS0}/${STATUS1}." >&2
  exit 1
fi

if [[ "${RUN_PRIVACY_POSTPROCESS:-1}" == "1" ]]; then
  for concentration in 0.5 0.6 0.8 0.9; do
    echo "[POSTPROCESS] Server A concentration=${concentration}"
    CUDA_VISIBLE_DEVICES="${GPU0}" python -m src.acm_revision.privacy_postprocess \
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

echo "[DONE] Server A privacy capture and validated post-processing completed."
