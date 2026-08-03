#!/usr/bin/env bash
set -euo pipefail

SERVER="${1:?server id A or B required}"
GPU0="${2:-0}"
GPU1="${3:-1}"
ROUNDS=150
JOB_SCRIPT="scripts/acm_revision/run_core72_job.sh"
SMOKE_SCRIPT="scripts/acm_revision/smoke_test_core72.sh"
RESULT_ROOT="results/acm_revision/core72_r${ROUNDS}"
POST_ROOT="results/acm_revision/core72_postprocess_r${ROUNDS}"
SERVER="${SERVER^^}"
PREFLIGHT="results/acm_revision/core72_preflight_server${SERVER}.json"

if [[ "${SERVER}" != "A" && "${SERVER}" != "B" ]]; then
  echo "SERVER must be A or B" >&2
  exit 2
fi
if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU ids must be different" >&2
  exit 2
fi
for path in "${JOB_SCRIPT}" "${SMOKE_SCRIPT}"; do
  [[ -f "${path}" ]] || { echo "Missing ${path}; run from repository root." >&2; exit 2; }
done

python -m src.acm_revision.core72_plan validate \
  --check-data \
  --output "${PREFLIGHT}"

if [[ "${RUN_CORE72_SMOKE:-1}" == "1" ]]; then
  bash "${SMOKE_SCRIPT}" "${GPU0}"
fi

mapfile -t QUEUE0 < <(python -m src.acm_revision.core72_plan jobs --server "${SERVER}" --queue 0)
mapfile -t QUEUE1 < <(python -m src.acm_revision.core72_plan jobs --server "${SERVER}" --queue 1)

echo "============================================================"
echo "CORE72 SERVER ${SERVER}"
echo "GPU ${GPU0}: ${#QUEUE0[@]} jobs"
echo "GPU ${GPU1}: ${#QUEUE1[@]} jobs"
echo "Every job: 150 rounds, strict capture, exact head updates every round"
echo "============================================================"

run_queue() {
  local gpu="$1"
  shift
  local line setting concentration population seed rounds num_clients samples job_id
  for line in "$@"; do
    IFS=$'\t' read -r setting concentration population seed rounds num_clients samples job_id <<< "${line}"
    echo "[SERVER ${SERVER}] GPU=${gpu} ${job_id}"
    CORE72_NUM_CLIENTS="${num_clients}" CORE72_SAMPLES_PER_CLIENT="${samples}" \
      bash "${JOB_SCRIPT}" "${gpu}" "${setting}" "${concentration}" "${population}" "${seed}" "${rounds}"
  done
}

run_queue "${GPU0}" "${QUEUE0[@]}" &
PID0=$!
run_queue "${GPU1}" "${QUEUE1[@]}" &
PID1=$!
cleanup() { kill -TERM "${PID0}" "${PID1}" 2>/dev/null || true; }
trap cleanup INT TERM
STATUS0=0
STATUS1=0
wait "${PID0}" || STATUS0=$?
wait "${PID1}" || STATUS1=$?
trap - INT TERM
if (( STATUS0 != 0 || STATUS1 != 0 )); then
  echo "[FAIL] Server ${SERVER} queues exited with ${STATUS0}/${STATUS1}" >&2
  exit 1
fi

if [[ "${RUN_CORE72_POSTPROCESS:-1}" == "1" ]]; then
  if [[ "${SERVER}" == "A" ]]; then
    CELLS=("image_only 0.5" "image_only 0.7" "image_only 0.9" "text_only 0.5" "text_only 0.7")
  else
    CELLS=("text_only 0.9" "modality_exclusive 0.5" "modality_exclusive 0.7" "modality_exclusive 0.9")
  fi
  for cell in "${CELLS[@]}"; do
    read -r setting concentration <<< "${cell}"
    python -m src.acm_revision.core72_postprocess \
      --root "${RESULT_ROOT}" \
      --setting "${setting}" \
      --concentration "${concentration}" \
      --rounds "${ROUNDS}" \
      --output-root "${POST_ROOT}"
    ctag="${concentration//./p}"
    output_dir="${POST_ROOT}/${setting}__c${ctag}__r${ROUNDS}"
    python -m src.acm_revision.core72_validate --output-dir "${output_dir}"
  done
fi

echo "[DONE] CORE72 Server ${SERVER}: training, post-processing and validation passed."
